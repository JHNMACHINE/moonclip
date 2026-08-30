from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

def flatten_state_dict(
    state_dict: dict,
    prefix: str = "",
    as_tensors: bool = False,
) -> Tuple[dict, Any]:
    """Flatten a PyTorch state_dict into individual tensors."""
    ...

def unflatten_state_dict(raw: dict) -> Dict[str, Any]:
    """Rebuild the state dicts that flatten_state_dict took apart.

    Takes the flat {name: bytes} map as MoonclipManager.load returns it, and
    gives back {prefix: state_dict}. Applies to nothing — for a caller that
    wants the objects loaded as well, that is CheckpointManager.load.
    """
    ...

class CheckpointManager:
    """PyTorch-aware checkpoint manager."""

    world_size: int
    rank: int
    save_dtype: Union[str, Mapping[str, str], None]
    last_resume_metadata: Dict[str, str]
    _mgr: Any
    _best_metric: Optional[float]

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
        save_dtype: Union[str, Mapping[str, str], None] = None,
        max_total_snapshots: Optional[int] = None,
        async_save: bool = True,
        keep_base_in_memory: bool = True,
        pin_device_copies: bool = True,
        **kwargs: Any,
    ) -> None: ...

    def save(
        self,
        step: int,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str: ...

    def save_raw(
        self,
        step: int,
        tensors: dict,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Save pre-flattened tensors directly (for background executor pattern)."""
        ...

    def last_queue_wait(self) -> float:
        """Seconds the last save waited for the previous one to drain."""
        ...

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
        """Save checkpoint only if the metric improves. Returns snap_id or None."""
        ...

    def save_final(
        self,
        step: int,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Save final checkpoint, merge deltas, sync remote."""
        ...

    def save_to_pt(
        self,
        path: str,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
    ) -> str:
        """Export current state as a standard PyTorch .pt file."""
        ...

    def save_to_safetensors(
        self,
        path: str,
        model: Optional[Any] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Export model weights as a safetensors file. Requires: pip install safetensors"""
        ...

    def load(
        self,
        snap_id: str,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
    ) -> Dict[str, Any]: ...

    def load_latest(
        self,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
    ) -> Tuple[str, Dict[str, Any]]: ...

    def resume(
        self,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
    ) -> int:
        """Auto-resume from latest checkpoint. Returns next step (0 if none)."""
        ...

    def list_snapshots(self) -> List[Dict[str, Any]]: ...
    def merge_now(self) -> None: ...
    def sync_now(self) -> None: ...

    def flush(self) -> None:
        """Block until every pending background save has completed."""
        ...

    def stats(self) -> Dict[str, Any]:
        """Return aggregate checkpoint statistics."""
        ...

    def print_stats(self) -> None:
        """Print human-readable checkpoint statistics."""
        ...
