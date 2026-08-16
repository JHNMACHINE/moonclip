from typing import Any, Dict, List, Optional, Tuple

def flatten_state_dict(
    state_dict: dict,
    prefix: str = "",
    as_tensors: bool = False,
) -> Tuple[dict, Any]:
    """Flatten a PyTorch state_dict into individual tensors."""
    ...

class CheckpointManager:
    """PyTorch-aware checkpoint manager with auto-detection of torchrun environment."""

    world_size: int
    rank: int
    save_dtype: str
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
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        merge_stride: int = 0,
        merge_max_chain: int = 10,
        save_dtype: str = "none",
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
