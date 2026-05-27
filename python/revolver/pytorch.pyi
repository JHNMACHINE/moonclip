"""
revolver.pytorch — PyTorch-aware checkpoint manager.

Usage:
    from revolver import CheckpointManager

    mgr = CheckpointManager("./checkpoints")
    mgr.save(step=1000, model=model, optimizer=optimizer, metadata={"loss": "0.634"})
    mgr.load_latest(model=model, optimizer=optimizer)
"""
from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None  # type: ignore


def _detect_distributed_env() -> Tuple[int, int]:
    """
    Auto-detect world_size and rank from the distributed environment.

    Detection priority:
    1. torch.distributed (if already initialized)
    2. torchrun env vars: RANK, WORLD_SIZE
    3. DeepSpeed / other launchers: same env vars
    4. Defaults: world_size=1, rank=0

    torchrun sets: RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
    """
    # 1. torch.distributed already initialized
    if torch is not None:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                return dist.get_world_size(), dist.get_rank()
        except Exception:
            pass

    # 2. Environment variables (torchrun, deepspeed, etc.)
    env_world_size = os.environ.get("WORLD_SIZE")
    env_rank = os.environ.get("RANK")

    if env_world_size is not None and env_rank is not None:
        try:
            return int(env_world_size), int(env_rank)
        except ValueError:
            pass

    # 3. Defaults
    return 1, 0


def _flatten_and_extract_tensors(val: Any, prefix: str, tensors_out: dict) -> Any:
    """
    Recursively walks val (which can be dict, list, tuple, tensor, etc.).
    Extracts all Tensors into tensors_out (keyed by prefix).
    Returns a copy of val where Tensors are replaced by placeholder dicts.
    """
    if torch.is_tensor(val):
        t = val.detach().cpu().contiguous()
        shape = list(t.shape)
        dtype = str(t.dtype).replace("torch.", "")
        raw_bytes = t.numpy().tobytes()
        tensors_out[prefix] = (shape, dtype, raw_bytes)
        return {
            "__tensor__": prefix,
            "shape": shape,
            "dtype": dtype,
        }
    elif isinstance(val, dict):
        return {k: _flatten_and_extract_tensors(v, f"{prefix}/{k}", tensors_out) for k, v in val.items()}
    elif isinstance(val, list):
        return [_flatten_and_extract_tensors(v, f"{prefix}/{idx}", tensors_out) for idx, v in enumerate(val)]
    elif isinstance(val, tuple):
        return tuple(_flatten_and_extract_tensors(v, f"{prefix}/{idx}", tensors_out) for idx, v in enumerate(val))
    else:
        return val


def _reconstruct_from_tensors(meta: Any, tensors_in: dict) -> Any:
    """
    Recursively walks meta (the template structure).
    Replaces any tensor placeholder dicts with the reconstructed PyTorch tensor.
    """
    DTYPE_MAP = {
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

    if isinstance(meta, dict) and "__tensor__" in meta:
        prefix = meta["__tensor__"]
        shape = meta["shape"]
        dtype_str = meta["dtype"]
        if prefix not in tensors_in:
            return None
        raw = tensors_in[prefix]
        dtype = DTYPE_MAP.get(dtype_str, torch.float32)
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).clone()
    elif isinstance(meta, dict):
        return {k: _reconstruct_from_tensors(v, tensors_in) for k, v in meta.items()}
    elif isinstance(meta, list):
        return [_reconstruct_from_tensors(v, tensors_in) for v in meta]
    elif isinstance(meta, tuple):
        return tuple(_reconstruct_from_tensors(v, tensors_in) for v in meta)
    else:
        return meta


class CheckpointManager:
    """
    PyTorch-aware wrapper around RevolverManager.

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
        delta_threshold: float = 0.5,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        merge_stride: int = 0,
        merge_max_chain: int = 10,
        **kwargs,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for CheckpointManager")

        from revolver import RevolverManager

        # Auto-detect from torchrun / torch.distributed environment
        # Priority: explicit arg > torch.distributed > env vars > defaults
        if world_size is None or rank is None:
            detected_world_size, detected_rank = _detect_distributed_env()
            if world_size is None:
                world_size = detected_world_size
            if rank is None:
                rank = detected_rank

        self.world_size = world_size
        self.rank = rank

        self._mgr = RevolverManager(
            storage_root=storage_root,
            compression_level=compression_level,
            max_full_snapshots=max_full_snapshots,
            max_deltas_per_full=max_deltas_per_full,
            full_every_steps=full_every_steps,
            delta_threshold=delta_threshold,
            world_size=world_size,
            rank=rank,
            merge_stride=merge_stride,
            merge_max_chain=merge_max_chain,
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
        Save a training checkpoint with per-tensor delta tracking.

        Each tensor is stored individually, so unchanged tensors are
        skipped entirely (zero I/O) and changed tensors use XOR delta.
        """
        all_tensors = {}

        def _add(prefix: str, obj: Any):
            sd = obj.state_dict() if hasattr(obj, "state_dict") else obj
            meta = _flatten_and_extract_tensors(sd, prefix, all_tensors)
            meta_bytes = pickle.dumps(meta)
            all_tensors[f"{prefix}._metadata"] = ([], "uint8", meta_bytes)

        if model is not None:
            _add("model", model)
        if optimizer is not None:
            _add("optimizer", optimizer)
        if scheduler is not None:
            _add("scheduler", scheduler)
        if scaler is not None:
            _add("scaler", scaler)
        if extra:
            for name, obj in extra.items():
                _add(name, obj)

        if self.world_size > 1:
            import torch.distributed as dist
            if not dist.is_initialized():
                raise RuntimeError("torch.distributed must be initialized for multi-rank save")

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
                raise RuntimeError("torch.distributed must be initialized for multi-rank load")

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
        """Group loaded tensors by prefix and apply to objects."""
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

        for key, value in raw.items():
            if key.endswith("._metadata"):
                prefix = key[:-10]  # strip "._metadata"
                meta = pickle.loads(value)
                sd = _reconstruct_from_tensors(meta, raw)
                
                obj = prefix_to_obj.get(prefix)
                if obj is not None:
                    if hasattr(obj, "load_state_dict"):
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