from typing import Dict, List, Mapping, Optional, Tuple, Union, Any

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
        save_dtype: Union[str, Mapping[str, str], None] = None,
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
            save_dtype: Target dtype for saving float tensors — one of "none",
                "bf16", "fp16", "fp32", "fp64", "fp8" (= "fp8_e4m3") or
                "fp8_e5m2". If set, float tensors are cast in Rust before
                compression, and auto-uncast back to original dtype on load.

                It may also be a dict of glob pattern to dtype, which is how
                the components of a checkpoint get different precision:

                    save_dtype={"optimizer/*": "bf16"}

                `*` matches any run of characters; everything else is literal;
                **the first matching rule wins**, in the order written; a
                tensor matching no rule is stored as it arrived. So an
                exception is written by putting it first:

                    save_dtype={"model/*": "none", "*": "bf16"}

                This is worth doing rather than merely possible. Measured on
                a 1.5B model under FSDP2, the optimizer moments are ~85% of
                the bytes written and barely delta at all — two consecutive
                Adam moments differ across nearly every mantissa bit — while
                the weights are the third that deltas well (−70%). The
                moments are also the part that tolerates the least precision:
                `exp_avg_sq` enters Adam through `sqrt(v)`, which halves the
                relative error. Casting only them to bf16 halves 85% of the
                volume and leaves the model untouched.

                A pattern matching none of a rank's tensors is reported on
                stderr rather than accepted quietly: {"weight": "bf16"} is
                the plausible mistake, since names arrive with their prefix
                ("model/weight"), and it would otherwise cast nothing and
                say nothing.

                Integer, bool and complex tensors are stored unchanged
                whatever this says — they are transported, never cast. "fp64"
                is a valid target but only ever widens; it recovers no
                precision the source did not have.

                The float8 targets quantize against a per-tensor scale kept in
                the manifest, so they are a quarter the size of fp32 but keep
                only four significant bits: expect a few percent of relative
                error on every element. That is fine for an archived copy or
                for analysis, and it is not fine for a checkpoint a run will
                resume from — optimizer moments in particular do not survive
                it. Tensors that are *already* float8 are stored bit-exact
                whatever this is set to, and need no setting at all.
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

    def last_queue_wait(self) -> float:
        """
        Seconds the last save_tensors() waited for the previous save to drain.

        One save is allowed in flight. When the writer has not finished, the
        next save_tensors() blocks before any of its own work starts, and from
        the outside that is indistinguishable from the shadow copy having been
        slow: one call, one duration, two unrelated causes. The copy is memory
        bandwidth and grows with the model; this is backpressure and grows with
        the checkpoint cadence and the speed of the storage. Told apart, each
        points at a different fix; added together they point at neither.

        Call it right after save_tensors(), on the same thread, and it
        describes that call. It is a single value overwritten by each save.

        Zero when there was nothing to wait for, and always zero when
        async_save is off — the write then happens on the calling thread, and
        that is the write itself rather than a queue in front of it.
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

    def load_tensors(self, snap_id: str, names: List[str]) -> Dict[str, bytearray]:
        """
        Load only the named tensors out of a snapshot.

        ``load`` reads the rank's whole pack in one request and decompresses
        every tensor in it, which is right when every tensor is wanted. This
        reads only the byte ranges the named tensors occupy — which over S3 or
        R2 is the difference between a few kilobytes and the checkpoint.

        Args:
            snap_id: Snapshot UUID string.
            names: Tensor names, as ``describe`` reports them.

        Returns:
            Dict mapping tensor_name -> raw bytes (as bytearray).

        Raises:
            RuntimeError: if a name is not in the snapshot. A missing name is
                an error rather than an absent key: answering a typo with
                nothing is how a caller ends up reasoning about a checkpoint it
                never read.
        """
        ...

    def describe(self, snap_id: str) -> Dict[str, Any]:
        """
        What a snapshot holds, without reading any of it.

        Reads the manifest already in memory and touches no storage at all.
        The dict carries ``id``, ``step``, ``created_at``, ``is_delta``,
        ``base_snapshot_id``, ``rank``, ``ranks``, ``metadata``, and
        ``tensors`` — one entry per tensor with ``name``, ``shape``, ``dtype``,
        ``stored_dtype``, ``storage`` (``full`` / ``delta`` / ``skipped`` /
        ``alias``), ``alias_of``, ``raw_size`` and ``compressed_size``.

        ``dtype`` is what a load hands back; ``stored_dtype`` is what is on
        disk. They differ exactly when ``save_dtype`` cast the tensor.

        For anything that needs a checkpoint's shapes but not its bytes — a
        reshard planning how N old shards map onto M new ones, an inspector,
        a size report — this is the question, and until now the only way to
        ask it was to load the snapshot and measure it.
        """
        ...

    def describe_latest(self) -> Dict[str, Any]:
        """
        The newest finalized snapshot, described rather than loaded.

        See :meth:`describe`.
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

    def sync_prefix(self, prefix: str) -> None:
        """
        Queue the upload of everything under ``prefix`` to the remote, and
        return without waiting. For files written into the store directory
        beside the checkpoints, which the periodic sync does not look at.

        A file already on the remote is skipped by name, so this is right only
        for files that are never rewritten, and cheapest when ``prefix`` names
        one file. A no-op without a remote.
        """
        ...

    def restore_from_remote(self) -> bool:
        """
        Fill an empty store from the remote. Returns whether anything came.

        For a machine that came up without the store it had. Call it before
        reading anything: it replaces the manifest this manager loaded when
        the store was still empty.
        """
        ...
