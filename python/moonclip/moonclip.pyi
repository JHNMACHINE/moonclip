from typing import Dict, List, Optional, Tuple, Any

__version__: str

class MoonclipManager:
    """
    High-performance checkpoint manager for ML training.

    DECK-inspired architecture with per-tensor delta tracking,
    rank-aware saves, hierarchical merging, and lineage graph.
    """

    def __init__(
        self,
        storage_root: str = "./checkpoints",
        compression_level: int = 3,
        max_full_snapshots: int = 5,
        max_deltas_per_full: int = 10,
        full_every_steps: int = 5000,
        delta_max_ratio: float = 0.95,
        world_size: int = 1,
        rank: int = 0,
        merge_stride: int = 0,
        merge_max_chain: int = 10,
        rollback_interval_steps: int = 10000,
        max_rollback_snapshots: int = 3,
        max_total_snapshots: Optional[int] = None,
        s3_bucket: Optional[str] = None,
        s3_region: str = "us-east-1",
        s3_prefix: str = "",
        s3_endpoint: Optional[str] = None,
        s3_access_key: Optional[str] = None,
        s3_secret_key: Optional[str] = None,
        s3_path_style: bool = False,
        sync_every_n_saves: int = 100,
        save_dtype: str = "none",
        async_save: bool = True,
        keep_base_in_memory: bool = True,
    ) -> None:
        """
        Initialize the MoonclipManager.

        Args:
            storage_root: Local directory to store snapshots.
            compression_level: zstd compression level (1-9, 0 = None).
            max_full_snapshots: Max number of full snapshots to retain.
            max_deltas_per_full: Max delta snapshots to allow between full saves.
            full_every_steps: Force a full snapshot every N steps.
            delta_max_ratio: Store a tensor as a XOR delta only when the compressed
                delta is smaller than this fraction of the compressed full tensor.
                Above it, save the tensor in full.
            world_size: Total number of ranks in multi-rank training.
            rank: Rank of this manager instance.
            merge_stride: After this many consecutive delta snapshots, fold them
                into the newest one and delete the rest (0 = disable). Every
                delta is written against the last full snapshot, so the newest
                already describes the whole state and the fold reads no bytes —
                but the folded steps stop being restorable. Read it as how
                coarse the checkpoint history may become.
            merge_max_chain: How many deltas may accumulate before they are
                merged back into a new full snapshot.
            rollback_interval_steps: Steps between rollback snapshots. A full
                snapshot landing on a multiple of this is exempt from retention
                (0 = no rollback snapshots).
            max_rollback_snapshots: How many of those exemptions to keep, newest
                first. Older rollback snapshots become ordinary snapshots again
                and are pruned by retention like any other; without this cap the
                exempt set would grow for the length of the run. 0 disables
                rollback protection entirely.
            max_total_snapshots: Hard cap on accessible snapshots, full and
                delta together. Defaults to
                max_full_snapshots * (1 + max_deltas_per_full).
            s3_bucket: Optional S3 bucket for batched remote sync.
            s3_region: S3 region.
            s3_prefix: Optional S3 prefix directory.
            s3_endpoint: Optional custom S3 endpoint URL.
            s3_access_key: S3 access key.
            s3_secret_key: S3 secret key.
            s3_path_style: Use path-style S3 URLs instead of virtual hosting.
                Automatically forced to True when s3_endpoint is set.
            sync_every_n_saves: Sync local data to remote S3 every N saves.
            save_dtype: Target dtype for saving float tensors ("none", "bf16", "fp16").
                If set, float tensors are cast in Rust before compression, and
                auto-uncast back to original dtype on load.
            async_save: Run single-rank saves on a background thread.
                save_tensors() returns as soon as the tensor data has been
                copied; hashing, compression and the disk write overlap with
                training. Errors surface on the next save/load/flush call.
                Loads and listing always wait for pending saves first.
            keep_base_in_memory: Keep the last full snapshot's raw bytes, so the
                next delta does not read and decompress a base this process
                just wrote. Costs one retained copy of the saved state in host
                memory — turn it off where memory is the binding constraint.
        """
        ...

    def save_tensors(
        self,
        step: int,
        tensors: Dict[str, Tuple[List[int], str, bytes]],
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Save a checkpoint (single-rank mode).

        Tensors may also be passed as `torch.Tensor` objects instead of
        `(shape, dtype, bytes)` tuples, which skips a copy.

        **The caller must not mutate those tensors' storage while this call is
        running.** Moonclip reads them through `data_ptr()` with the GIL
        released, so it is reading the live buffer, not a copy — the copy is
        what this call is taking. Assigning into a tensor from another Python
        thread races the read and stores a mixture of both states;
        `resize_()`, `set_()` or letting the tensor be freed reallocates the
        storage and leaves the pointer dangling, which is undefined behaviour
        rather than a bad checkpoint.

        The thread that called this is blocked inside it, so an ordinary
        single-threaded training loop cannot violate this and nothing needs to
        change. It is worth knowing about if a second thread touches the model
        — an EMA updater, a pruning callback, an async evaluation loop. Pass
        `(shape, dtype, bytes)` tuples there, or copy first.

        Args:
            step: Training step.
            tensors: Dict mapping tensor_name -> (shape, dtype, bytes), or
                tensor_name -> torch.Tensor (CPU, contiguous).
            metadata: Optional dict of string metadata.

        Returns:
            Snapshot UUID string.
        """
        ...

    def create_snapshot(
        self,
        step: int,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Create a new snapshot entry (rank 0 only in multi-rank).

        Args:
            step: Training step.
            metadata: Optional dict of string metadata.

        Returns:
            Snapshot UUID string.
        """
        ...

    def save_rank(
        self,
        snap_id: str,
        tensors: Dict[str, Tuple[List[int], str, bytes]],
    ) -> None:
        """
        Save this rank's tensors into an existing snapshot (multi-rank).

        Args:
            snap_id: Snapshot UUID string.
            tensors: Dict mapping tensor_name -> (shape, dtype, bytes).
        """
        ...

    def finalize_snapshot(self, snap_id: str) -> None:
        """
        Finalize a snapshot (rank 0 only in multi-rank).
        Marks it as complete and triggers retention/merging.

        Args:
            snap_id: Snapshot UUID string.
        """
        ...

    def flush(self) -> None:
        """
        Block until any in-flight background save completes.
        Raises if the background save failed.
        """
        ...

    def load(self, snap_id: str) -> Dict[str, bytearray]:
        """
        Load a snapshot.

        Args:
            snap_id: Snapshot UUID string.

        Returns:
            Dict mapping tensor_name -> raw bytes (as bytearray).
        """
        ...

    def load_latest(self) -> Tuple[str, Dict[str, bytearray]]:
        """
        Load the latest finalized snapshot.

        Returns:
            Tuple of (snapshot_id_str, dict of tensor_name -> raw bytes as bytearray).
        """
        ...

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """
        List all finalized snapshots.

        Returns:
            List of snapshot info dicts containing details such as step, id,
            is_delta, ranks, total_raw, total_compressed, and metadata.
        """
        ...

    def merge_now(self) -> None:
        """
        Force merge all pending deltas into a full checkpoint.
        """
        ...

    def sync_now(self) -> None:
        """
        Force sync all local data to remote S3 storage immediately.
        """
        ...
