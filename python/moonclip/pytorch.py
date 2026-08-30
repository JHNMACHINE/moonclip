"""
moonclip.pytorch — PyTorch-aware checkpoint manager.

Usage:
    from moonclip import CheckpointManager

    mgr = CheckpointManager("./checkpoints")
    mgr.save(step=1000, model=model, optimizer=optimizer, metadata={"loss": "0.634"})
    mgr.load_latest(model=model, optimizer=optimizer)
"""

from __future__ import annotations

import ctypes
import os
import pickle
import sys
from typing import Any, Dict, Optional, Tuple, Union

import torch

from moonclip._env import _detect_distributed_env


def _stream_can_encode(text: str) -> bool:
    """Whether `text` survives the trip to stdout as written."""
    encoding = getattr(sys.stdout, "encoding", None)
    if not encoding:
        return False
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _emit(text: str) -> None:
    """Print, without letting the console's encoding take the run down.

    Every message here is reporting, printed from inside a training loop. On a
    Windows console at cp1252 `print` itself raises `UnicodeEncodeError` for a
    character the codepage has no room for — so a progress line kills a run
    that was checkpointing perfectly well. It has two ways in: the box-drawing
    rule this module used to print unconditionally, and any path or tensor name
    outside the codepage, which no amount of care in this file can rule out.

    So the unrepresentable characters are replaced and the line still goes out.
    Losing a glyph is not a failure; losing the run to a status line is.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(encoding, "replace").decode(encoding, "replace"))


def _tensor_to_bytes(t: "torch.Tensor") -> bytes:
    """
    Convert a tensor to raw bytes.

    Reads the tensor's own buffer rather than going through NumPy. The NumPy
    route needed a special case for bfloat16 (which NumPy has no dtype for) and
    another for everything else it refuses, and it made NumPy a hard
    requirement of the byte path — while never appearing in this package's
    dependencies. A torch build without NumPy raises "Numpy is not available"
    from `.numpy()`, which is a strange way to fail to save a checkpoint.

    Reading `data_ptr()` has neither problem: raw bytes are raw bytes, so every
    dtype works the same way, including bfloat16 and complex64.
    """
    t = t.detach().cpu().contiguous()
    nbytes = t.numel() * t.element_size()
    if nbytes == 0:
        # `data_ptr()` may be null for an empty tensor, and string_at would
        # read from address zero.
        return b""
    # Safe: `t` is alive for the duration of the call, contiguous, and on the
    # host, so exactly `nbytes` readable bytes follow the pointer.
    return ctypes.string_at(t.data_ptr(), nbytes)


class _PinnedStaging:
    """Page-locked host buffers for the device-to-host copy, reused per save.

    A copy into ordinary (pageable) host memory cannot go straight over PCIe:
    CUDA stages it through an internal pinned buffer, and the transfer cannot
    be issued asynchronously. Measured on an RTX 5060 Ti (PCIe 4.0 x8), 3.8 GiB
    of weights:

        pageable    2735 ms    1.5 GB/s
        pinned       290 ms   14.1 GB/s     9.4x

    14.1 GB/s is about 90% of what that link can carry, so the pinned path is
    limited by the bus rather than by software.

    The buffers are kept between saves because pinning is a kernel operation
    and an expensive one: page-locking those same 3.8 GiB took 2521 ms, which
    is nine times the copy it makes possible. Allocated per save it would be a
    large net loss; allocated once it is paid at the first checkpoint and never
    again.

    **The returned tensors are the buffers**, not copies of them, so the next
    save overwrites what the previous one handed out. That is safe for the way
    Moonclip uses them — `save_raw` copies into Rust before it returns — and it
    is why this is not a public API. Do not hold on to what `to_host` returns.

    Page-locked memory cannot be swapped, so this reserves host RAM the machine
    cannot reclaim: one copy of whatever gets checkpointed. Pass
    ``pin_device_copies=False`` to a CheckpointManager to trade the speed back.
    """

    __slots__ = ("_buffers", "_used", "enabled")

    def __init__(self, enabled: bool = True):
        self._buffers: Dict[str, "torch.Tensor"] = {}
        self._used = False
        self.enabled = enabled

    def to_host(self, key: str, t: "torch.Tensor") -> "torch.Tensor":
        """Copy `t` to host memory, through a reused pinned buffer."""
        if not t.is_cuda or not self.enabled:
            return t.detach().cpu().contiguous()

        source = t.detach()
        if not source.is_contiguous():
            source = source.contiguous()

        buffer = self._buffers.get(key)
        if (
            buffer is None
            or buffer.shape != source.shape
            or buffer.dtype != source.dtype
        ):
            buffer = torch.empty(
                source.shape, dtype=source.dtype, pin_memory=True
            )
            self._buffers[key] = buffer

        buffer.copy_(source, non_blocking=True)
        self._used = True
        return buffer

    def finish(self) -> None:
        """Wait for the copies issued since the last call.

        `non_blocking` copies are queued, not done. Handing a buffer to the
        writer before the transfer lands would store whatever the buffer held
        previously — the last checkpoint's weights, which look entirely
        plausible and are wrong.
        """
        if self._used:
            torch.cuda.synchronize()
            self._used = False


_DEFAULT_STAGING: Optional["_PinnedStaging"] = None


def _default_staging() -> "_PinnedStaging":
    """Staging shared by callers that do not bring their own.

    `flatten_state_dict` is called directly by Ravex and by anyone following
    the background-executor recipe in its docstring, none of whom should have
    to know that pinned memory exists to get a 9x device copy. A process
    trains one model at a time, so one set of buffers keyed by tensor name is
    the right shape for this.

    Set MOONCLIP_NO_PINNED_STAGING=1 to disable it: page-locked memory cannot
    be swapped, and on a box where host RAM is the binding constraint that
    matters more than the copy.
    """
    global _DEFAULT_STAGING
    if _DEFAULT_STAGING is None:
        _DEFAULT_STAGING = _PinnedStaging(
            enabled=os.environ.get("MOONCLIP_NO_PINNED_STAGING", "") not in ("1", "true")
        )
    return _DEFAULT_STAGING


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
    # The byte path reads a tensor's own buffer, so it saves these without
    # noticing they are complex — and until 0.0.6 nothing here could rebuild
    # them, which turned a saved complex tensor into a load-time failure (or,
    # worse, into float32 through the old fallback). Rare in weights, ordinary
    # in an FFT-based model's buffers.
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}
if hasattr(torch, "uint16"):
    _DTYPE_MAP["uint16"] = torch.uint16


#: A snapshot id is a UUID in its canonical form, always this many characters.
_SNAP_ID_LEN = 36


def _broadcast_snapshot_id(dist, snap_id: str, rank: int) -> str:
    """Share rank 0's snapshot id with every rank.

    `dist.broadcast_object_list` would be the obvious call and is the wrong
    one: it pickles through ``tensor.numpy()``, so it needs NumPy — which
    neither this package nor torch declares as a requirement, and which a
    CPU-only torch install frequently does not have. On such an install every
    multi-rank save died inside PyTorch with "Numpy is not available", and the
    ranks that were only waiting on the collective reported a bare
    "Connection closed by peer" instead.

    A UUID is a fixed 36 characters, so a plain byte broadcast carries it with
    no serialisation at all — no NumPy, no pickle, and one collective instead
    of the two that broadcasting an object of unknown size needs.
    """
    # NCCL only moves CUDA tensors; gloo is happy with host memory.
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if dist.get_backend() == "nccl"
        else torch.device("cpu")
    )
    buffer = torch.zeros(_SNAP_ID_LEN, dtype=torch.uint8, device=device)
    if rank == 0:
        encoded = snap_id.encode("ascii")[:_SNAP_ID_LEN]
        buffer[: len(encoded)] = torch.frombuffer(
            bytearray(encoded), dtype=torch.uint8
        ).to(device)

    dist.broadcast(buffer, src=0)
    return bytes(buffer.tolist()).rstrip(b"\x00").decode("ascii")


def _tensor_from_raw(shape, dtype_str: str, raw) -> "torch.Tensor":
    """Rebuild a tensor from raw bytes + shape + dtype string.

    An unknown dtype raises. It used to fall back to float32, which is the
    worst possible answer: the bytes are reinterpreted under a dtype that is
    not theirs, and `frombuffer` is happy to do it whenever the element sizes
    divide. The tensor that comes back has the right shape, plausible-looking
    numbers, and no relationship to what was saved — a checkpoint that loads
    and is wrong is harder to notice than one that refuses to load.
    """
    dtype = _DTYPE_MAP.get(dtype_str)
    if dtype is None:
        raise ValueError(
            f"checkpoint holds a tensor of dtype '{dtype_str}', which this "
            f"version of Moonclip cannot rebuild. Known dtypes: "
            f"{', '.join(sorted(_DTYPE_MAP))}."
        )
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


def _flatten_state(
    value: Any,
    key_prefix: str,
    counter: list,
    out_tensors: dict,
    staging: Optional["_PinnedStaging"] = None,
) -> Any:
    """
    Recursively replace tensors in a state-dict structure with placeholder
    dicts, adding each tensor to `out_tensors` under a unique key.

    The returned template mirrors the original structure; on load,
    _reconstruct_from_tensors rebuilds it. Tensors are handed to the Rust
    extension individually, so hashing/compression/delta tracking run in
    parallel per tensor and nothing large goes through pickle.
    """
    if torch.is_tensor(value) and value.dtype in _DIRECT_DTYPES:
        key = f"{key_prefix}/t{counter[0]}"
        t = (staging or _default_staging()).to_host(key, value)
        counter[0] += 1
        out_tensors[key] = t
        return {
            "__tensor__": key,
            "shape": list(t.shape),
            "dtype": str(t.dtype).replace("torch.", ""),
        }
    if isinstance(value, dict):
        return {
            k: _flatten_state(v, key_prefix, counter, out_tensors, staging)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            _flatten_state(v, key_prefix, counter, out_tensors, staging) for v in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _flatten_state(v, key_prefix, counter, out_tensors, staging) for v in value
        )
    return value


def _flatten_and_extract_tensors(
    val: Any,
    prefix: str,
    tensors_out: dict,
    as_tensors: bool = False,
    staging: Optional["_PinnedStaging"] = None,
) -> Any:
    """
    Recursively walks val (which can be dict, list, tuple, tensor, etc.).
    Extracts all Tensors into tensors_out as (shape, dtype, bytes) tuples,
    keyed by their path under `prefix`.
    Returns a copy of val where Tensors are replaced by placeholder dicts.

    With `as_tensors`, tensors the Rust extension can read directly are put in
    `tensors_out` as tensors instead of bytes, so the byte copy happens in Rust
    (in parallel, with the GIL released) rather than here. Dtypes the extension
    cannot address stay on the byte path.
    """
    if torch.is_tensor(val):
        t = (staging or _default_staging()).to_host(prefix, val)
        shape = list(t.shape)
        dtype = str(t.dtype).replace("torch.", "")
        if as_tensors and t.dtype in _DIRECT_DTYPES:
            tensors_out[prefix] = t
        else:
            tensors_out[prefix] = (shape, dtype, _tensor_to_bytes(t))
        return {
            "__tensor__": prefix,
            "shape": shape,
            "dtype": dtype,
        }
    elif isinstance(val, dict):
        return {
            k: _flatten_and_extract_tensors(v, f"{prefix}/{k}", tensors_out, as_tensors, staging)
            for k, v in val.items()
        }
    elif isinstance(val, list):
        return [
            _flatten_and_extract_tensors(v, f"{prefix}/{idx}", tensors_out, as_tensors, staging)
            for idx, v in enumerate(val)
        ]
    elif isinstance(val, tuple):
        return tuple(
            _flatten_and_extract_tensors(v, f"{prefix}/{idx}", tensors_out, as_tensors, staging)
            for idx, v in enumerate(val)
        )
    else:
        return val


def flatten_state_dict(
    state_dict: dict,
    prefix: str = "",
    as_tensors: bool = False,
    staging: Optional["_PinnedStaging"] = None,
) -> Tuple[dict, Any]:
    """
    Flatten a PyTorch state_dict into individual tensors.

    Extracts all tensors and returns them in the format expected by
    MoonclipManager.save_tensors() and CheckpointManager.save_raw().

    Useful for the background executor pattern: serialize tensors in the
    main thread, then submit the dict to a background thread for saving.

    Args:
        state_dict: A PyTorch state_dict (from model or optimizer).
        prefix: Key prefix for tensor names (e.g. "model", "optimizer").
        as_tensors: Hand tensors to the extension directly instead of
            converting them to bytes here. The copy still happens — it has to,
            the writer must not read tensors the next training step is
            mutating — but it happens once, in Rust, across all cores with the
            GIL released, instead of twice (``.tobytes()`` here, then again on
            the way into Rust) on this thread. Measured on 3.8 GiB of fp32
            weights: 1393 ms of blocked training becomes 271 ms.

            **The copy is only complete when ``save_raw`` returns**, not when
            this function does: with ``as_tensors`` the returned dict aliases
            live parameter memory for a model already on CPU. Do not hand that
            dict to a background thread and let training continue — pass it
            straight to ``save_raw``/``save_tensors``, which block until the
            copy is taken.

    Returns:
        Tuple of (tensors_dict, metadata_structure) where:
        - tensors_dict: {name: (shape, dtype, bytes)}, or {name: Tensor} under
          `as_tensors`, including a "<prefix>._metadata" entry (pickled
          template) so checkpoints saved via save_raw can be applied back to
          objects on load
        - metadata_structure: the state_dict with tensors replaced by
          placeholder dicts that _reconstruct_from_tensors understands

    Example:
        tensors, meta = flatten_state_dict(model.state_dict(), "model")
        # tensors = {"model/weight": ([512, 512], "float32", b"..."), ...}
        # Submit to background:
        executor.submit(mgr.save_raw, step=1000, tensors=tensors)
    """
    staging = staging or _default_staging()
    tensors_out: dict = {}
    metadata = _flatten_and_extract_tensors(
        state_dict, prefix, tensors_out, as_tensors, staging
    )
    # The device copies were queued, not completed. Nothing may read these
    # buffers until they land.
    staging.finish()
    tensors_out[f"{prefix}._metadata"] = ([], "uint8", pickle.dumps(metadata))
    return tensors_out, metadata


def _resolve_topology(world_size, rank) -> Tuple[int, int]:
    """Settle what topology this manager is for, or refuse to guess.

    Until 0.0.9 an unstated topology was read out of the environment and
    adopted silently. That looked like a convenience and behaved like a trap,
    because adopting it can only ever take capability away: `Coordinator::save`
    refuses outright when `world_size != 1`, directing the caller to the
    explicit create_snapshot/save_rank/finalize flow. So the same code saved
    fine under `python train.py` and raised under `torchrun`, decided by
    nothing the code could see.

    Ravex is where that bill came due. It builds this manager per rank, caught
    the refusal in the `except` around backend construction, and fell back to
    `torch.save` with a single log line — so every distributed run quietly
    lost Moonclip checkpointing and kept training.

    Now the environment is consulted only to decide whether the question is
    ambiguous. One process and nothing to disagree with is still (1, 0), so
    single-process callers see no change at all. A launcher that says
    otherwise, with nothing stated here, is an error naming both numbers and
    the ways forward — including `"auto"`, which is the old behaviour, still
    available and now asked for.
    """
    if world_size == "auto" or rank == "auto":
        detected_world_size, detected_rank = _detect_distributed_env()
        if world_size == "auto":
            world_size = detected_world_size
        if rank == "auto":
            rank = detected_rank

    if world_size is not None and rank is not None:
        return int(world_size), int(rank)

    detected_world_size, detected_rank = _detect_distributed_env()
    if detected_world_size <= 1:
        # Nothing to disagree with: one process, one store.
        return (
            1 if world_size is None else int(world_size),
            0 if rank is None else int(rank),
        )

    raise ValueError(
        "\n".join(
            [
                "the launcher says this is rank %d of %d, and CheckpointManager "
                "was not told which topology to use. Until 0.0.9 it adopted "
                "those numbers silently, which turned save() into an error "
                "decided by how the process was started. Say which you want:"
                % (detected_rank, detected_world_size),
                "  world_size=1, rank=0            one store per process, which "
                "is what you want when each rank writes to its own directory",
                "  world_size=%d, rank=%d            these ranks share one "
                "store, through create_snapshot/save_rank/finalize"
                % (detected_world_size, detected_rank),
                '  world_size="auto", rank="auto"  whatever the launcher says, '
                "which is what happened before 0.0.9",
            ]
        )
    )


def unflatten_state_dict(raw: dict) -> Dict[str, Any]:
    """Rebuild the state dicts that :func:`flatten_state_dict` took apart.

    The inverse, and public because the forward direction is: a caller that
    flattens its own state and hands the result to ``save_tensors`` had no
    supported way back, and the only implementation lived inside
    ``CheckpointManager._apply_loaded`` welded to the step that calls
    ``load_state_dict`` on live objects. Anything holding the bytes but no
    objects — a converter, a checkpoint inspector, Ravex — had to either
    reimplement this or take the applying it did not want.

    Args:
        raw: ``{name: bytes}`` exactly as ``MoonclipManager.load`` and
            ``load_latest`` return it.

    Returns:
        ``{prefix: state_dict}``, one entry per prefix that was flattened —
        ``"model"``, ``"optimizer"``, and whatever else the caller named.
        Prefixes with no recoverable structure are left out rather than
        guessed at.

    Two shapes are read, because two were written. A ``<prefix>._blob`` entry
    is a whole pickled state dict, which is what optimizer and scheduler state
    became once storing their scalars as individual tensors turned out to cost
    more in manifest entries than it saved. A ``<prefix>._metadata`` entry is
    the older per-tensor form, still what model weights use, where the pickle
    holds the tree with placeholders and the tensors are separate entries —
    which is the form that makes per-tensor delta tracking possible at all.

    A prefix carrying both is read as a blob: that is the newer writer, and it
    is self-contained.
    """
    result: Dict[str, Any] = {}

    for key, value in raw.items():
        if key.endswith("._blob"):
            result[key[: -len("._blob")]] = pickle.loads(value)

    for key, value in raw.items():
        if not key.endswith("._metadata"):
            continue
        prefix = key[: -len("._metadata")]
        if prefix in result:
            continue
        result[prefix] = _reconstruct_from_tensors(pickle.loads(value), raw)

    return result


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
        world_size: Union[int, str, None] = None,
        rank: Union[int, str, None] = None,
        merge_stride: int = 0,
        merge_max_chain: int = 10,
        save_dtype: str = "none",
        max_total_snapshots: Optional[int] = None,
        async_save: bool = True,
        keep_base_in_memory: bool = True,
        pin_device_copies: bool = True,
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

        world_size, rank = _resolve_topology(world_size, rank)

        self.world_size = world_size
        self.rank = rank
        self.save_dtype = save_dtype
        self._best_metric: Optional[float] = None
        # Its own buffers rather than the module default: two managers
        # checkpointing different models would otherwise collide on tensor
        # names and reallocate on every save.
        self._staging = _PinnedStaging(enabled=pin_device_copies)

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
            keep_base_in_memory=keep_base_in_memory,
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

        # Model: tensors go straight to Rust (no pickle), alongside a small
        # "model._metadata" template that keeps shape and dtype for the
        # reconstruction on load.
        if model is not None:
            template = {}
            for name, param in model.state_dict().items():
                if torch.is_tensor(param) and param.dtype in _DIRECT_DTYPES:
                    key = f"model/{name}"
                    t = self._staging.to_host(key, param)
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
            template = _flatten_state(sd, prefix, counter, all_tensors, self._staging)
            all_tensors[f"{prefix}._metadata"] = ([], "uint8", pickle.dumps(template))

        if optimizer is not None:
            _add_state("optimizer", optimizer)
        if scheduler is not None:
            _add_state("scheduler", scheduler)
        if scaler is not None:
            _add_state("scaler", scaler)
        # extra items go through the same path: tensors reach Rust directly
        if extra:
            for name, obj in extra.items():
                _add_state(f"extra/{name}", obj)

        # Every device copy above was queued, not completed. Nothing may read
        # those buffers until they land.
        self._staging.finish()

        if self.world_size > 1:
            import torch.distributed as dist

            if not dist.is_initialized():
                raise RuntimeError(
                    "torch.distributed must be initialized for multi-rank save"
                )

            snap_id = ""
            if self.rank == 0:
                snap_id = self._mgr.create_snapshot(step=step, metadata=metadata or {})

            snap_id = _broadcast_snapshot_id(dist, snap_id, self.rank)

            # All at once. Until 0.0.6 this loop ran the ranks one at a time,
            # each waiting on a barrier for its turn, because every rank
            # rewrote the shared manifest.json and concurrent writers lost each
            # other's entries. A rank now writes only its own pack — the
            # snapshot is assembled from those in finalize — so the shards go
            # in parallel, which is what the README always claimed.
            self._mgr.save_rank(snap_id, all_tensors)

            # One barrier, and it is load-bearing: rank 0 assembles the
            # snapshot from the packs on disk, so every rank has to have
            # finished writing before it looks.
            dist.barrier()

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
            snap_id = _broadcast_snapshot_id(dist, snap_id, self.rank)

            raw = self._mgr.load(snap_id)
            state_dicts = self._apply_loaded(raw, model, optimizer, scheduler, scaler)
            return snap_id, state_dicts
        else:
            snap_id, raw = self._mgr.load_latest()
            state_dicts = self._apply_loaded(raw, model, optimizer, scheduler, scaler)
            return snap_id, state_dicts

    def _apply_loaded(self, raw, model, optimizer, scheduler, scaler):
        """Rebuild the state dicts in `raw`, then load them into the objects.

        The rebuilding is `unflatten_state_dict`; everything left here is the
        applying, which is the half a caller holding no objects does not want.
        """
        result = unflatten_state_dict(raw)

        prefix_to_obj = {}
        if model is not None:
            prefix_to_obj["model"] = model
        if optimizer is not None:
            prefix_to_obj["optimizer"] = optimizer
        if scheduler is not None:
            prefix_to_obj["scheduler"] = scheduler
        if scaler is not None:
            prefix_to_obj["scaler"] = scaler

        for prefix, state in result.items():
            obj = prefix_to_obj.get(prefix)
            if obj is not None and hasattr(obj, "load_state_dict"):
                obj.load_state_dict(state)

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

    def restore_from_remote(self) -> bool:
        """Pull this store back from the remote. Returns whether anything came.

        For a machine that came up without the store it had — replaced after a
        preemption, or simply handed a different rank than last time. The
        remote was push-only until 2026-08-19: measured on a six-node bench, a
        replaced node started from scratch while its data sat in the bucket,
        and the ranks that *did* have their stores started over with it,
        because a resume is agreed at the oldest step everyone holds.

        Call it before reading anything: it replaces the manifest this manager
        loaded when the store was still empty.
        """
        return self._mgr.restore_from_remote()

    def sync_now(self):
        """Push all local data to remote storage and wait for it.

        Blocks until the upload finishes, and raises if it did not: a run
        whose checkpoints are not reaching the remote should find that out
        while it can still react, not discover it after the instance is gone.
        """
        self._mgr.sync_now()

    def flush(self):
        """Block until any in-flight background save completes.

        Saves run on a background thread by default (async_save=True), so
        save() returns as soon as the tensor data has been copied. Call
        flush() when you need the checkpoint durably on disk — the data is
        fsynced before the file is renamed into place, so it survives a power
        cut, unless MOONCLIP_FSYNC=0 (e.g. right
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

    def last_queue_wait(self) -> float:
        """
        Seconds the last save waited for the previous one to drain.

        Moonclip allows one save in flight, so a writer that has not caught up
        stops the next save before it does anything. A caller timing its own
        save sees that wait folded into the same number as the shadow copy,
        and the two want different answers: the copy is memory bandwidth and
        scales with the model, the wait is backpressure and scales with the
        cadence and the storage. Reading this straight after a save separates
        them.

        Zero when nothing was in flight, and when async_save is off.
        """
        return self._mgr.last_queue_wait()

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
            _emit("[Moonclip] No checkpoints yet")
            return

        def _fmt(b):
            if b >= 1 << 30:
                return f"{b / (1 << 30):.2f} GB"
            elif b >= 1 << 20:
                return f"{b / (1 << 20):.1f} MB"
            elif b >= 1 << 10:
                return f"{b / (1 << 10):.1f} KB"
            return f"{b} B"

        # Drawn with whatever this console can actually render, rather than
        # assuming UTF-8: `_emit` would replace the rule character by character
        # and print sixty question marks, which is worse than a plain dash.
        rule = ("─" if _stream_can_encode("─") else "-") * 60

        _emit(f"\n{rule}")
        _emit("  Moonclip Checkpoint Stats")
        _emit(rule)
        _emit(
            f"  Snapshots:  {s['total_snapshots']}  ({s['full_snapshots']} full, {s['delta_snapshots']} delta)"
        )
        _emit(
            f"  Tensors:    {s['total_tensors_saved']}  ({s['full_tensors']} full, {s['delta_tensors']} delta, {s['skipped_tensors']} skipped)"
        )
        _emit(f"  Raw size:   {_fmt(s['total_raw_bytes'])}")
        _emit(
            f"  On disk:    {_fmt(s['total_compressed_bytes'])}  (compression: {s['compression_ratio']:.1%})"
        )
        _emit(
            f"  Naive cost: {_fmt(s['estimated_naive_bytes'])}  (if every save were full, uncompressed)"
        )
        _emit(
            f"  Saved:      {_fmt(s['total_saved_bytes'])}  ({s['savings_percent']:.1f}% vs naive)"
        )
        _emit(f"{rule}\n")

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
        _emit(f"[Moonclip] New best {metric_name}={metric:.6f} at step {step}")
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
        _emit(f"[Moonclip] Final checkpoint saved at step {step}")
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
        _emit(f"[Moonclip] Exported to {path} ({size_mb:.1f} MB)")
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
        _emit(
            f"[Moonclip] Exported to {path} ({size_mb:.1f} MB, {len(tensors)} tensors)"
        )
        return path
