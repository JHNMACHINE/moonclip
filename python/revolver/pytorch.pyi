from typing import Any, Dict, List, Optional, Tuple

class CheckpointManager:
    """
    PyTorch-aware checkpoint manager with auto-detection of torchrun environment.

    If world_size/rank are not specified, auto-detects from:
    1. torch.distributed (if initialized)
    2. torchrun env vars (RANK, WORLD_SIZE)
    3. Defaults to single-rank (world_size=1, rank=0)
    """

    world_size: int
    rank: int

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

    def list_snapshots(self) -> List[Dict[str, Any]]: ...
    def merge_now(self) -> None: ...

    last_resume_metadata: Dict[str, str]

    def resume(
        self,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
    ) -> int:
        """
        Auto-resume from the latest checkpoint if one exists.

        Returns:
            Next training step (0 if no checkpoints, last_step + 1 otherwise).
        """
        ...
