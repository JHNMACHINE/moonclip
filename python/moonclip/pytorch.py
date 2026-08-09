"""
moonclip.pytorch — PyTorch-aware checkpoint manager.

Usage:
    from moonclip import CheckpointManager

    mgr = CheckpointManager("./checkpoints")
    mgr.save(step=1000, model=model, optimizer=optimizer, metadata={"loss": "0.634"})
    mgr.load_latest(model=model, optimizer=optimizer)
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional, Tuple

import torch

from moonclip._env import _detect_distributed_env


def _tensor_to_bytes(t: "torch.Tensor") -> bytes:
    """
    Convert a tensor to raw bytes efficiently.
    Handles all dtypes including bfloat16 (which numpy doesn't support).
    """
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        # numpy doesn't support bf16 — reinterpret as int16 (same size)
        return t.view(torch.int16).numpy().tobytes()
    else:
        try:
            return t.numpy().tobytes()
        except TypeError:
            # Fallback for any other unsupported dtype
            return t.view(torch.uint8).numpy().tobytes()


# Dtypes the Rust extension can ingest directly from a tensor's data_ptr.
_DIRECT_DTYPES = {
    torch.float32,
    torch.float64,
    torch.float16,
    torch.bfloat16,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
}
if hasattr(torch, "uint16"):
    _DIRECT_DTYPES.add(torch.uint16)


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}
if hasattr(torch, "uint16"):
    _DTYPE_MAP["uint16"] = torch.uint16


def _tensor_from_raw(shape, dtype_str: str, raw) -> "torch.Tensor":
    """Rebuild a tensor from raw bytes + shape + dtype string."""
    dtype = _DTYPE_MAP.get(dtype_str, torch.float32)
    if len(raw) == 0:
        return torch.empty(shape, dtype=dtype)
    buf = raw if isinstance(raw, bytearray) else bytearray(raw)
    return torch.frombuffer(buf, dtype=dtype).reshape(shape).clone()


def _reconstruct_from_tensors(meta: Any, tensors_in: dict) -> Any:
    """
    Recursively walks meta (the template structure).
    Replaces any tensor placeholder dicts with the reconstructed PyTorch tensor.
    """
    if isinstance(meta, dict) and "__tensor__" in meta:
        key = meta["__tensor__"]
        if key not in tensors_in:
            return None
        return _tensor_from_raw(meta["shape"], meta["dtype"], tensors_in[key])
    elif isinstance(meta, dict):
        return {k: _reconstruct_from_tensors(v, tensors_in) for k, v in meta.items()}
    elif isinstance(meta, list):
        return [_reconstruct_from_tensors(v, tensors_in) for v in meta]
    elif isinstance(meta, tuple):
        return tuple(_reconstruct_from_tensors(v, tensors_in) for v in meta)
    else:
        return meta


def _flatten_state(value: Any, key_prefix: str, counter: list, out_tensors: dict) -> Any:
    """
    Recursively replace tensors in a state-dict structure with placeholder
    dicts, adding each tensor to `out_tensors` under a unique key.

    The returned template mirrors the original structure; on load,
    _reconstruct_from_tensors rebuilds it. Tensors are handed to the Rust
    extension individually, so hashing/compression/delta tracking run in
    parallel per tensor and nothing large goes through pickle.
    """
    if torch.is_tensor(value) and value.dtype in _DIRECT_DTYPES:
        t = value.detach().cpu().contiguous()
        key = f"{key_prefix}/t{counter[0]}"
        counter[0] += 1
        out_tensors[key] = t
        return {
            "__tensor__": key,
            "shape": list(t.shape),
            "dtype": str(t.dtype).replace("torch.", ""),
        }
    if isinstance(value, dict):
        return {
            k: _flatten_state(v, key_prefix, counter, out_tensors)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_flatten_state(v, key_prefix, counter, out_tensors) for v in value]
    if isinstance(value, tuple):
        return tuple(_flatten_state(v, key_prefix, counter, out_tensors) for v in value)
    return value


def _flatten_and_extract_tensors(val: Any, prefix: str, tensors_out: dict) -> Any:
    """
    Recursively walks val (which can be dict, list, tuple, tensor, etc.).
    Extracts all Tensors into tensors_out as (shape, dtype, bytes) tuples,
    keyed by their path under `prefix`.
    Returns a copy of val where Tensors are replaced by placeholder dicts.
    """
    if torch.is_tensor(val):
        t = val.detach().cpu().contiguous()
        shape = list(t.shape)
        dtype = str(t.dtype).replace("torch.", "")
        tensors_out[prefix] = (shape, dtype, _tensor_to_bytes(t))
        return {
            "__tensor__": prefix,
            "shape": shape,
            "dtype": dtype,
        }
    elif isinstance(val, dict):
        return {
            k: _flatten_and_extract_tensors(v, f"{prefix}/{k}", tensors_out)
            for k, v in val.items()
        }
    elif isinstance(val, list):
        return [
            _flatten_and_extract_tensors(v, f"{prefix}/{idx}", tensors_out)
            for idx, v in enumerate(val)
        ]
    elif isinstance(val, tuple):
        return tuple(
            _flatten_and_extract_tensors(v, f"{prefix}/{idx}", tensors_out)
            for idx, v in enumerate(val)
        )
    else:
        return val


def flatten_state_dict(
    state_dict: dict,
    prefix: str = "",
) -> Tuple[dict, Any]:
    """
    Flatten a PyTorch state_dict into individual tensor bytes.

    Extracts all tensors and returns them in the format expected by
    MoonclipManager.save_tensors() and CheckpointManager.save_raw().

    Useful for the background executor pattern: serialize tensors in the
    main thread, then submit the dict to a background thread for saving.

    Args:
        state_dict: A PyTorch state_dict (from model or optimizer).
        prefix: Key prefix for tensor names (e.g. "model", "optimizer").

    Returns:
        Tuple of (tensors_dict, metadata_structure) where:
        - tensors_dict: {name: (shape, dtype, bytes)}, including a
          "<prefix>._metadata" entry (pickled template) so checkpoints
          saved via save_raw can be applied back to objects on load
        - metadata_structure: the state_dict with tensors replaced by
          placeholder dicts that _reconstruct_from_tensors understands

    Example:
        tensors, meta = flatten_state_dict(model.state_dict(), "model")
        # tensors = {"model/weight": ([512, 512], "float32", b"..."), ...}
        # Submit to background:
        executor.submit(mgr.save_raw, step=1000, tensors=tensors)
    """
    tensors_out: dict = {}
    metadata = _flatten_and_extract_tensors(state_dict, prefix, tensors_out)
    tensors_out[f"{prefix}._metadata"] = ([], "uint8", pickle.dumps(metadata))
    return tensors_out, metadata


class CheckpointManager:
    """
    PyTorch-aware wrapper around MoonclipManager.

    Handles per-tensor serialization: each tensor from the state_dict
    is stored individually, enabling per-tensor delta tracking (DECK-style).
    """

    def __init__(
        self,
        storage_root: str = "./checkpoints",
        compression_level: int = 3,
        max_full_snapshots: int = 5,
        max_deltas_per_full: int = 10,
        full_every_steps: int = 5000,
        delta_max_ratio: float = 0.95,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        merge_stride: int = 0,
        merge_max_chain: int = 10,
        save_dtype: str = "none",
        max_total_snapshots: Optional[int] = None,
        async_save: bool = True,
        **kwargs,
    ):
        from moonclip import MoonclipManager

        if "delta_threshold" in kwargs:
            raise TypeError(
                "delta_threshold was removed in favour of delta_max_ratio, which "
                "has different semantics: it caps the size of a compressed delta "
                "relative to the compressed full tensor, instead of thresholding "
                "the fraction of differing bytes. The old criterion rejected "
                "essentially every delta on dense training. Drop the argument to "
                "take the default (0.95), or set delta_max_ratio explicitly."
            )

        # Auto-detect from torchrun / torch.distributed environment
        if world_size is None or rank is None:
            detected_world_size, detected_rank = _detect_distributed_env()
            if world_size is None:
                world_size = detected_world_size
            if rank is None:
                rank = detected_rank

        self.world_size = world_size
        self.rank = rank
        self.save_dtype = save_dtype
        self._best_metric: Optional[float] = None

        self._mgr = MoonclipManager(
            storage_root=storage_root,
            compression_level=compression_level,
            max_full_snapshots=max_full_snapshots,
            max_deltas_per_full=max_deltas_per_full,
            full_every_steps=full_every_steps,
            delta_max_ratio=delta_max_ratio,
            world_size=world_size,
            rank=rank,
            merge_stride=merge_stride,
            merge_max_chain=merge_max_chain,
            save_dtype=save_dtype,
            max_total_snapshots=max_total_snapshots,
            async_save=async_save,
            **kwargs,
        )

    def save(
        self,
        step: int,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Save a training checkpoint.

        Model weights use per-tensor delta tracking: unchanged tensors are
        skipped entirely (zero I/O) and changed tensors use XOR delta + zstd.

        Optimizer, scheduler, and scaler state are saved as single compressed
        blobs (their internal buffers change entirely every step, making
        per-tensor delta tracking counterproductive).
        """
        all_tensors = {}

        # Modello: tensori passati direttamente a Rust (zero pickle), con un
        # piccolo template "model._metadata" che conserva shape/dtype per la
        # ricostruzione al load.
        if model is not None:
            template = {}
            for name, param in model.state_dict().items():
                if torch.is_tensor(param) and param.dtype in _DIRECT_DTYPES:
                    t = param.detach().cpu().contiguous()
                    key = f"model/{name}"
                    all_tensors[key] = t
                    template[name] = {
                        "__tensor__": key,
                        "shape": list(t.shape),
                        "dtype": str(t.dtype).replace("torch.", ""),
                    }
                else:
                    # Non-tensor or exotic dtype: keep it in the template
                    template[name] = param
            all_tensors["model._metadata"] = ([], "uint8", pickle.dumps(template))

        def _add_state(prefix: str, obj: Any):
            """Per-tensor serialization for optimizer/scheduler/aux state.

            Each tensor (exp_avg, exp_avg_sq, ...) goes to Rust directly:
            hashing and compression run in parallel per tensor and nothing
            large is pickled. Only the non-tensor skeleton is pickled into
            "<prefix>._metadata". Checkpoints saved by older versions as a
            single "<prefix>._blob" are still loadable.
            """
            sd = obj.state_dict() if hasattr(obj, "state_dict") else obj
            counter = [0]
            template = _flatten_state(sd, prefix, counter, all_tensors)
            all_tensors[f"{prefix}._metadata"] = ([], "uint8", pickle.dumps(template))

        if optimizer is not None:
            _add_state("optimizer", optimizer)
        if scheduler is not None:
            _add_state("scheduler", scheduler)
        if scaler is not None:
            _add_state("scaler", scaler)
        # extra items (se sono tensori) possono essere passati direttamente
        if extra:
            for name, obj in extra.items():
                _add_state(f"extra/{name}", obj)

        if self.world_size > 1:
            import torch.distributed as dist

            if not dist.is_initialized():
                raise RuntimeError(
                    "torch.distributed must be initialized for multi-rank save"
                )

            snap_id = ""
            if self.rank == 0:
                snap_id = self._mgr.create_snapshot(step=step, metadata=metadata or {})

            # Broadcast snapshot ID from rank 0 to all other ranks
            objects = [snap_id]
            dist.broadcast_object_list(objects, src=0)
            snap_id = objects[0]

            # Save this rank's tensors sequentially to avoid manifest.json race condition
            for r in range(self.world_size):
                if self.rank == r:
                    self._mgr.save_rank(snap_id, all_tensors)
                dist.barrier()

            # Rank 0 finalizes
            if self.rank == 0:
                self._mgr.finalize_snapshot(snap_id)

            return snap_id
        else:
            return self._mgr.save_tensors(
                step=step,
                tensors=all_tensors,
                metadata=metadata or {},
            )

    def load(
        self,
        snap_id: str,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Load a checkpoint and optionally apply to objects."""
        raw = self._mgr.load(snap_id)
        return self._apply_loaded(raw, model, optimizer, scheduler, scaler)

    def load_latest(
        self,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Load the most recent checkpoint."""
        if self.world_size > 1:
            import torch.distributed as dist

            if not dist.is_initialized():
                raise RuntimeError(
                    "torch.distributed must be initialized for multi-rank load"
                )

            snap_id = ""
            if self.rank == 0:
                snap_id, _ = self._mgr.load_latest()
            objects = [snap_id]
            dist.broadcast_object_list(objects, src=0)
            snap_id = objects[0]

            raw = self._mgr.load(snap_id)
            state_dicts = self._apply_loaded(raw, model, optimizer, scheduler, scaler)
            return snap_id, state_dicts
        else:
            snap_id, raw = self._mgr.load_latest()
            state_dicts = self._apply_loaded(raw, model, optimizer, scheduler, scaler)
            return snap_id, state_dicts

    def _apply_loaded(self, raw, model, optimizer, scheduler, scaler):
        """Group loaded tensors by prefix and apply to objects.

        Supports two formats:
        - Blob: prefix._blob → single pickled state_dict (new, for optimizer/scheduler)
        - Per-tensor: prefix._metadata + individual tensors (original, for model weights)
        """
        result = {}

        prefix_to_obj = {}
        if model is not None:
            prefix_to_obj["model"] = model
        if optimizer is not None:
            prefix_to_obj["optimizer"] = optimizer
        if scheduler is not None:
            prefix_to_obj["scheduler"] = scheduler
        if scaler is not None:
            prefix_to_obj["scaler"] = scaler

        # First pass: load blobs (optimizer/scheduler/scaler)
        for key, value in raw.items():
            if key.endswith("._blob"):
                prefix = key[:-6]  # strip "._blob"
                sd = pickle.loads(value)
                obj = prefix_to_obj.get(prefix)
                if obj is not None and hasattr(obj, "load_state_dict"):
                    obj.load_state_dict(sd)
                result[prefix] = sd

        # Second pass: load per-tensor (model weights, or old-format optimizer)
        for key, value in raw.items():
            if key.endswith("._metadata"):
                prefix = key[:-10]  # strip "._metadata"
                if prefix in result:
                    continue  # already loaded via blob
                meta = pickle.loads(value)
                sd = _reconstruct_from_tensors(meta, raw)
                obj = prefix_to_obj.get(prefix)
                if obj is not None and hasattr(obj, "load_state_dict"):
                    obj.load_state_dict(sd)
                result[prefix] = sd

        return result

    def list_snapshots(self):
        return self._mgr.list_snapshots()

    def resume(
        self,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
    ) -> int:
        """
        Auto-resume from the latest checkpoint if one exists.

        Returns the next training step to start from:
        - If checkpoints exist: loads the latest, returns last_step + 1
        - If no checkpoints: returns 0 (start from scratch)

        Also returns metadata from the checkpoint via self.last_resume_metadata.

        Usage:
            mgr = CheckpointManager("./ckpts")
            start_step = mgr.resume(model=model, optimizer=optimizer)
            for step in range(start_step, 100_000):
                ...
                mgr.save(step=step, model=model, optimizer=optimizer)
        """
        snaps = self.list_snapshots()
        if not snaps:
            self.last_resume_metadata = {}
            return 0

        snap_id, state_dicts = self.load_latest(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

        last_step = snaps[-1]["step"]
        self.last_resume_metadata = snaps[-1].get("metadata", {})

        return last_step + 1

    def merge_now(self):
        self._mgr.merge_now()

    def sync_now(self):
        """Force sync all local data to remote storage."""
        self._mgr.sync_now()

    def flush(self):
        """Block until any in-flight background save completes.

        Saves run on a background thread by default (async_save=True), so
        save() returns as soon as the tensor data has been copied. Call
        flush() when you need the checkpoint durably on disk (e.g. right
        before exiting). Loads and list_snapshots() flush automatically.
        """
        self._mgr.flush()

    def save_raw(
        self,
        step: int,
        tensors: dict,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Save pre-flattened tensors directly (for background executor pattern).

        Args:
            step: Training step.
            tensors: Dict mapping name → (shape, dtype, bytes), as returned
                     by flatten_state_dict().
            metadata: Optional string metadata.

        Returns:
            Snapshot UUID string.
        """
        return self._mgr.save_tensors(
            step=step, tensors=tensors, metadata=metadata or {}
        )

    def stats(self) -> Dict[str, Any]:
        """
        Return aggregate checkpoint statistics.

        Shows how much storage was saved by delta tracking, skipping,
        and compression across all snapshots.

        Returns dict with keys:
            total_snapshots, full_snapshots, delta_snapshots,
            total_tensors_saved, skipped_tensors, delta_tensors, full_tensors,
            total_raw_bytes, total_compressed_bytes, compression_ratio,
            estimated_naive_bytes, total_saved_bytes, savings_percent
        """
        snaps = self.list_snapshots()
        if not snaps:
            return {"total_snapshots": 0}

        total_full_snaps = sum(1 for s in snaps if not s["is_delta"])
        total_delta_snaps = sum(1 for s in snaps if s["is_delta"])

        total_skipped = sum(s["skipped_tensors"] for s in snaps)
        total_delta = sum(s["delta_tensors"] for s in snaps)
        total_full = sum(s["full_tensors"] for s in snaps)
        total_tensors = total_skipped + total_delta + total_full

        total_raw = sum(s["total_raw"] for s in snaps)
        total_compressed = sum(s["total_compressed"] for s in snaps)

        # Estimate naive cost: if every snapshot were a full save with no compression
        # Use the raw size of the first full snapshot as the baseline per-snapshot cost
        first_full = next((s for s in snaps if not s["is_delta"]), snaps[0])
        naive_per_snapshot = first_full["total_raw"]
        estimated_naive = naive_per_snapshot * len(snaps)

        saved = estimated_naive - total_compressed
        savings_pct = (saved / estimated_naive * 100) if estimated_naive > 0 else 0.0
        compression_ratio = (total_compressed / total_raw) if total_raw > 0 else 1.0

        return {
            "total_snapshots": len(snaps),
            "full_snapshots": total_full_snaps,
            "delta_snapshots": total_delta_snaps,
            "total_tensors_saved": total_tensors,
            "skipped_tensors": total_skipped,
            "delta_tensors": total_delta,
            "full_tensors": total_full,
            "total_raw_bytes": total_raw,
            "total_compressed_bytes": total_compressed,
            "compression_ratio": compression_ratio,
            "estimated_naive_bytes": estimated_naive,
            "total_saved_bytes": saved,
            "savings_percent": savings_pct,
        }

    def print_stats(self):
        """Print a human-readable summary of checkpoint statistics."""
        s = self.stats()
        if s["total_snapshots"] == 0:
            print("[Moonclip] No checkpoints yet")
            return

        def _fmt(b):
            if b >= 1 << 30:
                return f"{b / (1 << 30):.2f} GB"
            elif b >= 1 << 20:
                return f"{b / (1 << 20):.1f} MB"
            elif b >= 1 << 10:
                return f"{b / (1 << 10):.1f} KB"
            return f"{b} B"

        print(f"\n{'─' * 60}")
        print("  Moonclip Checkpoint Stats")
        print(f"{'─' * 60}")
        print(
            f"  Snapshots:  {s['total_snapshots']}  ({s['full_snapshots']} full, {s['delta_snapshots']} delta)"
        )
        print(
            f"  Tensors:    {s['total_tensors_saved']}  ({s['full_tensors']} full, {s['delta_tensors']} delta, {s['skipped_tensors']} skipped)"
        )
        print(f"  Raw size:   {_fmt(s['total_raw_bytes'])}")
        print(
            f"  On disk:    {_fmt(s['total_compressed_bytes'])}  (compression: {s['compression_ratio']:.1%})"
        )
        print(
            f"  Naive cost: {_fmt(s['estimated_naive_bytes'])}  (if every save were full, uncompressed)"
        )
        print(
            f"  Saved:      {_fmt(s['total_saved_bytes'])}  ({s['savings_percent']:.1f}% vs naive)"
        )
        print(f"{'─' * 60}\n")

    # ─── Convenience save methods ────────────────────────────────────

    def save_best(
        self,
        step: int,
        metric: float,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        metric_name: str = "val_loss",
        lower_is_better: bool = True,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Save a checkpoint only if the metric improves.

        Tracks the best metric value internally. Returns the snapshot ID
        if saved, None if the metric did not improve.

        Args:
            step: Training step.
            metric: Current metric value (e.g. validation loss).
            model: Model to save.
            optimizer: Optimizer to save.
            metric_name: Name of the metric for metadata.
            lower_is_better: If True, lower metric = better (loss).
                             If False, higher = better (accuracy).

        Usage:
            val_loss = evaluate(model)
            mgr.save_best(step, metric=val_loss, model=model, optimizer=optimizer)
        """
        is_better = False
        if not hasattr(self, "_best_metric") or self._best_metric is None:
            is_better = True
        elif lower_is_better and metric < self._best_metric:
            is_better = True
        elif not lower_is_better and metric > self._best_metric:
            is_better = True

        if not is_better:
            return None

        self._best_metric = metric

        meta = metadata.copy() if metadata else {}
        meta["best"] = "true"
        meta[metric_name] = f"{metric:.6f}"

        snap_id = self.save(
            step=step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            metadata=meta,
        )
        print(f"[Moonclip] New best {metric_name}={metric:.6f} at step {step}")
        return snap_id

    def save_final(
        self,
        step: int,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Save a final checkpoint and merge all deltas.

        Call at the end of training. Saves, merges pending deltas,
        and optionally syncs to remote storage.

        Usage:
            mgr.save_final(step=100_000, model=model, optimizer=optimizer)
        """
        meta = metadata.copy() if metadata else {}
        meta["final"] = "true"

        snap_id = self.save(
            step=step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            metadata=meta,
        )
        self.merge_now()
        self.sync_now()
        print(f"[Moonclip] Final checkpoint saved at step {step}")
        return snap_id

    def save_to_pt(
        self,
        path: str,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
    ) -> str:
        """
        Export the current state as a standard PyTorch .pt file.

        Creates a file compatible with torch.load(). Useful for
        sharing models without requiring Moonclip.

        Args:
            path: Output file path (e.g. "model.pt").
            model: Model to export.
            optimizer: Optimizer to export (optional).
            scheduler: Scheduler to export (optional).

        Usage:
            mgr.save_to_pt("harold_v0.9_final.pt", model=model, optimizer=optimizer)
        """
        state = {}
        if model is not None:
            state["model"] = model.state_dict()
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()
        if scaler is not None:
            state["scaler"] = scaler.state_dict()

        torch.save(state, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"[Moonclip] Exported to {path} ({size_mb:.1f} MB)")
        return path

    def save_to_safetensors(
        self,
        path: str,
        model: Optional[Any] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Export model weights as a safetensors file.

        Creates a file compatible with the safetensors format.
        Only model weights are saved (no optimizer state).

        Requires: pip install safetensors

        Args:
            path: Output file path (e.g. "model.safetensors").
            model: Model to export.
            metadata: Optional string metadata dict.

        Usage:
            mgr.save_to_safetensors("harold_v0.9.safetensors", model=model)
        """
        try:
            from safetensors.torch import save_file
        except ImportError:
            raise ImportError(
                "safetensors is required for save_to_safetensors. "
                "Install with: pip install safetensors"
            )

        if model is None:
            raise ValueError("model is required for save_to_safetensors")

        state_dict = model.state_dict()
        # safetensors only supports tensors, filter out non-tensors
        tensors = {k: v for k, v in state_dict.items() if torch.is_tensor(v)}

        save_file(tensors, path, metadata=metadata)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(
            f"[Moonclip] Exported to {path} ({size_mb:.1f} MB, {len(tensors)} tensors)"
        )
        return path
