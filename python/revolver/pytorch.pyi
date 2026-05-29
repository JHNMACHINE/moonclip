from typing import Any, Dict, List, Optional, Tuple

def flatten_state_dict(
    state_dict: dict,
    prefix: str = "",
) -> Tuple[dict, Any]:
    """
    Flatten a PyTorch state_dict into individual tensor bytes.

    Returns:
        Tuple of (tensors_dict, metadata_structure) where:
        - tensors_dict: {name: (shape, dtype, bytes)}
        - metadata_structure: state_dict with tensors replaced by placeholders
    """
    ...

class CheckpointManager:
    """
    PyTorch-aware checkpoint manager with auto-detection of torchrun environment.
    """

    world_size: int
    rank: int
    save_dtype: str
    last_resume_metadata: Dict[str, str]
    _mgr: Any  # RevolverManager (Rust)

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

    def stats(self) -> Dict[str, Any]:
        """Return aggregate checkpoint statistics."""
        ...

    def print_stats(self) -> None:
        """Print human-readable checkpoint statistics."""
        ...
