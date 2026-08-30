use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::cast::DTypePolicy;
use crate::error::{Result, MoonclipError};
use crate::inflight::InFlight;
use crate::manifest::*;
use crate::merger::{DeltaMerger, MergerConfig};
use crate::pack;
use crate::remote_sync::{PendingDeletes, RemoteSyncConfig, RemoteSyncer};
use crate::storage::StorageBackend;
use crate::tensor::{self, BaseCache, TensorData};

/// Configuration for the coordinator.
pub struct CoordinatorConfig {
    pub world_size: u32,
    pub rank: u32,
    pub compression: CompressionAlgo,
    pub retention: RetentionPolicy,
    pub lineage: LineageConfig,
    /// A tensor is stored as a XOR delta only when the delta compresses to
    /// less than this fraction of the compressed full tensor. See
    /// [`crate::delta::pays_off`].
    pub delta_max_ratio: f64,
    pub merger: Option<MergerConfig>,
    pub remote_storage: Option<Arc<dyn StorageBackend>>,
    pub remote_sync: Option<RemoteSyncConfig>,
    /// Target dtype for saving float tensors, chosen per tensor name.
    /// An empty policy keeps every tensor as it arrives.
    pub save_dtype: DTypePolicy,
    /// Run single-rank saves on a background thread (bound-1 queue).
    /// `save()` returns as soon as the tensor data is handed off; any
    /// error surfaces on the next save/load/flush call. Loads, listing
    /// and multi-rank operations always wait for pending saves first.
    pub async_save: bool,
    /// Keep the last full snapshot's raw bytes in memory, so the next delta
    /// does not read and decompress a base this process just wrote.
    ///
    /// That read-and-decompress was 36% of the write path's CPU, and it
    /// decompressed bytes this process had written moments earlier. With them
    /// retained it drops to nothing, and the delta path ends up costing
    /// slightly *less* than a full save while still writing half the bytes.
    ///
    /// The bytes are not copied to get here — they are the ones handed in for
    /// the full save, moved rather than dropped. Measured peak RSS cost is
    /// **+1.00x the saved state**, exactly one retained copy: for a 1B model
    /// checkpointed with its Adam state, about +11 GiB.
    ///
    /// **That is the reason to turn it off**, on a box whose memory is the
    /// binding constraint. Set it false to trade the CPU back.
    ///
    /// Applied per tensor: a tensor `save_dtype` casts is not retained,
    /// because a later delta is computed against the post-cast bytes on disk
    /// and those are not what arrives here. The rest of the snapshot still is.
    pub keep_base_in_memory: bool,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        CoordinatorConfig {
            world_size: 1,
            rank: 0,
            compression: CompressionAlgo::Zstd { level: 3 },
            retention: RetentionPolicy::default(),
            lineage: LineageConfig::default(),
            delta_max_ratio: 0.95,
            merger: None,
            remote_storage: None,
            remote_sync: None,
            save_dtype: DTypePolicy::none(),
            async_save: true,
            keep_base_in_memory: true,
        }
    }
}

/// The snapshot-level facts a pack needs to describe itself.
///
/// Passed down rather than read back from the manifest: at the moment the pack
/// is written the manifest does not know about this snapshot yet, which is
/// precisely the window the embedded descriptor closes.
struct SnapshotContext {
    step: u64,
    created_at: DateTime<Utc>,
    metadata: HashMap<String, String>,
}

/// Rebuild a snapshot from the descriptions its own packs carry.
///
/// Every rank's pack describes the snapshot as well as its own contribution,
/// so a snapshot can be reassembled without a surviving rank 0 and without any
/// separate file. Returns None when the packs do not agree, are not of this
/// format, or do not add up to a complete snapshot — all of which mean there
/// is nothing safe to readmit.
fn rebuild_from_packs(
    storage: &dyn StorageBackend,
    id: &str,
    files: &[String],
) -> Option<Snapshot> {
    let mut descriptors: Vec<pack::PackDescriptor> = Vec::new();

    for file in files.iter().filter(|f| f.ends_with(".pack")) {
        let header = storage.get_range(file, 0, pack::HEADER_LEN as usize).ok()?;
        // No header means a pack written before this format. It still loads
        // through the manifest; it just cannot be recovered without one.
        let (offset, length) = pack::decode_header(&header)?;
        let encoded = storage.get_range(file, offset, length as usize).ok()?;
        if encoded.len() != length as usize {
            return None; // truncated write
        }
        let descriptor: pack::PackDescriptor = serde_json::from_slice(&encoded).ok()?;
        if descriptor.snapshot_id.to_string() != id {
            return None; // a pack claiming to belong somewhere else
        }
        descriptors.push(descriptor);
    }

    let first = descriptors.first()?.clone();

    // Every rank has to be here. A snapshot missing one rank's shard is not a
    // smaller checkpoint, it is an unusable one.
    let ranks: HashMap<u32, RankEntry> = descriptors
        .iter()
        .map(|d| (d.rank.rank, d.rank.clone()))
        .collect();
    if ranks.len() as u32 != first.world_size {
        return None;
    }

    Some(Snapshot {
        id: first.snapshot_id,
        step: first.step,
        created_at: first.created_at,
        ranks,
        base_snapshot_id: first.base_snapshot_id,
        metadata: first.metadata,
        compression: first.compression,
        finalized: true,
    })
}

/// Whether every byte the snapshot claims is present and intact.
///
/// Presence is not enough. A pack of the right length can still be a truncated
/// write padded by the filesystem, or a partial flush — and a snapshot readmitted
/// on those terms would fail much later, during a resume, which is the worst
/// moment to discover it. So each tensor's stored hash is checked against the
/// bytes actually on disk.
///
/// Reading the pack to do that is affordable precisely because this only runs
/// for orphans, which exist only after a crash.
fn snapshot_data_is_intact(storage: &dyn StorageBackend, snapshot: &Snapshot) -> bool {
    for rank_entry in snapshot.ranks.values() {
        for tensor in &rank_entry.tensors {
            // Skipped tensors and aliases store no bytes of their own; they
            // resolve through the base or through a sibling.
            let Some(ref expected) = tensor.hash_compressed else {
                continue;
            };

            let read = match (&rank_entry.pack_file, &tensor.filename) {
                (Some(pack), _) => {
                    storage.get_range(pack, tensor.offset, tensor.compressed_size as usize)
                }
                (None, Some(file)) => storage.get(file),
                (None, None) => return false,
            };

            let Ok(bytes) = read else { return false };
            if bytes.len() != tensor.compressed_size as usize {
                return false;
            }
            if &crate::hash::hash_hex(&bytes) != expected {
                return false;
            }
        }
    }
    true
}

/// Recover snapshot data that no manifest entry refers to, or delete it.
///
/// A checkpoint lands in two stages: the bytes go to storage, then the manifest
/// that names them. A process killed between the two leaves data with nothing
/// pointing at it. Resume is right to ignore it while it stays that way — a
/// snapshot the manifest does not list must never be trusted — but leaving it
/// there forever is not right either: `apply_retention` walks
/// `manifest.snapshots`, so it cannot see those files and they outlive the run.
///
/// Seen in the field, not hypothesised: an FSDP run over eight GPUs killed with
/// SIGKILL mid-checkpoint left three snapshot directories against one manifest
/// entry, and resumed four steps further back than it needed to.
///
/// So each orphan is judged rather than assumed:
///
/// * it describes itself (`snapshot.json`), its data passes its own hashes, it
///   is finalised, and any base it deltas against is present → **put it back in
///   the manifest**. It was a complete checkpoint; only the bookkeeping was lost.
/// * anything else — no description, failed hashes, a missing base → **delete
///   it**. That is a write that never finished, and there is nothing to save.
///
/// **Startup is the only safe moment.** Here this process has submitted no save
/// and the background saver does not exist yet, so nothing it might touch is in
/// flight. What remains is a *different* process writing to the same storage
/// root concurrently — but two coordinators sharing a root already overwrite
/// each other's manifest, so that arrangement is broken well before this is.
///
/// Only directories named by a parsable UUID are considered. Anything else
/// under `snapshots/` was put there by something other than this code, and
/// guessing about it is how a cleanup routine deletes data it did not own.
///
/// Failures are logged and swallowed: this is housekeeping and must never stop
/// a training run from starting. Returns true when the manifest changed.
fn recover_or_reclaim_orphans(storage: &dyn StorageBackend, manifest: &mut Manifest) -> bool {
    let known: std::collections::HashSet<String> = manifest
        .snapshots
        .iter()
        .map(|s| s.id.to_string())
        .collect();

    let files = match storage.list("snapshots") {
        Ok(files) => files,
        Err(e) => {
            eprintln!("[Moonclip] Could not list snapshots: {e}");
            return false;
        }
    };

    let mut orphans: HashMap<String, Vec<String>> = HashMap::new();
    for file in files {
        let Some(id) = file
            .strip_prefix("snapshots/")
            .and_then(|rest| rest.split('/').next())
        else {
            continue;
        };
        if known.contains(id) || Uuid::parse_str(id).is_err() {
            continue;
        }
        orphans.entry(id.to_string()).or_default().push(file);
    }

    if orphans.is_empty() {
        return false;
    }

    // Read every candidate's description first, then admit them oldest-first:
    // a delta can only go back if its base is already there, and its base may
    // itself be one of these orphans.
    let mut candidates: Vec<Snapshot> = Vec::new();
    let mut rejects: Vec<String> = Vec::new();

    for (id, files) in &orphans {
        let described = rebuild_from_packs(storage, id, files);
        match described {
            Some(snapshot) if snapshot_data_is_intact(storage, &snapshot) => {
                candidates.push(snapshot)
            }
            _ => rejects.push(id.clone()),
        }
    }
    candidates.sort_by_key(|s| s.step);

    let mut recovered = 0usize;
    for snapshot in candidates {
        let base_present = match snapshot.base_snapshot_id {
            None => true,
            Some(base) => manifest.snapshots.iter().any(|s| s.id == base),
        };
        if !base_present {
            // A delta whose base is gone reconstructs nothing.
            rejects.push(snapshot.id.to_string());
            continue;
        }
        manifest.snapshots.push(snapshot);
        recovered += 1;
    }

    let mut reclaimed = 0usize;
    for id in &rejects {
        for file in orphans.get(id).into_iter().flatten() {
            if let Err(e) = storage.delete(file) {
                eprintln!("[Moonclip] Could not delete orphaned {file}: {e}");
            }
        }
        // Best effort: the bytes are already gone, and an empty directory left
        // behind costs an inode, not a checkpoint.
        let _ = storage.remove_dir(&format!("snapshots/{id}"));
        reclaimed += 1;
    }

    if recovered > 0 {
        manifest.snapshots.sort_by_key(|s| s.step);
        eprintln!(
            "[Moonclip] Recovered {recovered} complete checkpoint(s) a previous \
             run wrote but was killed before recording"
        );
    }
    if reclaimed > 0 {
        eprintln!(
            "[Moonclip] Discarded {reclaimed} incomplete snapshot(s) left by an \
             interrupted run"
        );
    }

    recovered > 0
}

/// The last full snapshot's raw tensor bytes, kept for the next delta.
struct RetainedBase {
    /// Which snapshot these bytes are. A base cache is only usable for the
    /// snapshot it was taken from; anything else and the delta would be
    /// computed against the wrong thing.
    snap_id: Uuid,
    tensors: Arc<HashMap<String, Vec<u8>>>,
}

/// Shared state + save/load logic. Owned via Arc by the Coordinator and
/// (when async saving is enabled) by the background save thread.
/// Read the manifest off a store, or start a fresh one shaped by the config.
///
/// Shared by construction and by [`Core::restore_from_remote`], which needs
/// exactly the same reading after it has pulled the file down: a restore that
/// left the in-memory manifest as it found it would put the snapshots on disk
/// and leave the manager still believing there are none.
fn load_manifest(storage: &dyn StorageBackend, config: &CoordinatorConfig) -> Result<Manifest> {
    match storage.get("manifest.json") {
        Ok(data) => {
            // Trailing zeros: a manifest is rewritten in place, so a shorter
            // one can leave the tail of a longer one behind it.
            let end = data
                .iter()
                .rposition(|&b| b != 0)
                .map(|i| i + 1)
                .unwrap_or(0);
            serde_json::from_slice(&data[..end])
                .map_err(|e| MoonclipError::Serialization(e.to_string()))
        }
        Err(MoonclipError::NotFound(_)) => Ok(Manifest {
            world_size: config.world_size,
            retention: config.retention.clone(),
            lineage: config.lineage.clone(),
            ..Default::default()
        }),
        Err(e) => Err(e),
    }
}

/// Whether every file a snapshot names is actually here.
///
/// Used after a restore from remote, where the manifest and the data it points
/// at travel separately and can arrive out of step. A snapshot that is missing
/// a pack is not a degraded snapshot, it is an unloadable one, and leaving it
/// in the manifest turns a resume into a failure at the first read.
fn snapshot_is_whole(storage: &dyn StorageBackend, snapshot: &Snapshot) -> bool {
    for rank in snapshot.ranks.values() {
        if let Some(ref pack) = rank.pack_file {
            if !storage.exists(pack).unwrap_or(false) {
                return false;
            }
        }
        for tensor in &rank.tensors {
            if let Some(ref file) = tensor.filename {
                if !storage.exists(file).unwrap_or(false) {
                    return false;
                }
            }
        }
    }
    true
}

pub(crate) struct Core {
    pub(crate) storage: Arc<dyn StorageBackend>,
    pub(crate) manifest: Arc<Mutex<Manifest>>,
    config: CoordinatorConfig,
    merger: Option<DeltaMerger>,
    syncer: Option<RemoteSyncer>,
    /// Snapshots being read right now, and the ones a merge has claimed.
    /// See [`crate::inflight`]: it is what keeps a merge from deleting the
    /// base a save is diffing against.
    in_flight: Arc<InFlight>,
    /// Files retention and the merger have deleted locally, waiting for the
    /// syncer to take them off the remote too.
    pending_deletes: Arc<PendingDeletes>,
    /// See [`CoordinatorConfig::keep_base_in_memory`].
    retained_base: Mutex<Option<RetainedBase>>,
    /// Whether the `save_dtype` patterns have already been checked against a
    /// real set of tensor names. Once per process: the answer cannot change
    /// between steps, and a run checkpointing every hundred steps would
    /// otherwise print the same complaint for a week.
    dtype_patterns_checked: AtomicBool,
}

/// The main checkpoint coordinator.
///
/// In single-rank mode (world_size=1), this handles everything.
/// In multi-rank mode, each rank has its own Coordinator instance
/// with the same storage backend and manifest. Rank 0 is responsible
/// for creating/finalizing snapshots.
pub struct Coordinator {
    // Declared before `core` so pending saves drain before teardown.
    saver: Option<AsyncSaver>,
    pub(crate) core: Arc<Core>,
}

impl Coordinator {
    pub fn new(storage: Arc<dyn StorageBackend>, config: CoordinatorConfig) -> Result<Self> {
        let manifest = load_manifest(storage.as_ref(), &config)?;

        // Rank 0 only, for the same reason only rank 0 finalises: every rank
        // shares one storage root, and ranks 1..N write into directories rank 0
        // created.
        let mut manifest = manifest;
        if config.rank == 0 && recover_or_reclaim_orphans(storage.as_ref(), &mut manifest) {
            // Persist immediately: a recovered checkpoint that only exists in
            // this process's memory would be lost again to the next crash, and
            // would be re-recovered on every startup until one happened not to.
            if let Ok(json) = serde_json::to_vec_pretty(&manifest) {
                if let Err(e) = storage.put("manifest.json", &json) {
                    eprintln!("[Moonclip] Could not persist recovered snapshots: {e}");
                }
            }
        }

        let manifest = Arc::new(Mutex::new(manifest));

        let in_flight = Arc::new(InFlight::default());
        // Only the syncer drains this, so with no remote there is nothing to
        // drain it and nothing on a remote to delete — collecting keys would
        // be a leak that grows for the length of the run.
        let has_remote = config.remote_storage.is_some() && config.remote_sync.is_some();
        let pending_deletes = Arc::new(if has_remote {
            PendingDeletes::default()
        } else {
            PendingDeletes::disabled()
        });

        // Spawn merger if configured
        let merger = config.merger.clone().map(|mc| {
            DeltaMerger::new(
                mc,
                Arc::clone(&storage),
                Arc::clone(&manifest),
                config.compression.clone(),
                Arc::clone(&in_flight),
                Arc::clone(&pending_deletes),
            )
        });

        // Spawn remote syncer if configured
        let syncer = match (&config.remote_storage, &config.remote_sync) {
            (Some(remote), Some(sync_config)) => Some(RemoteSyncer::new(
                Arc::clone(&storage),
                Arc::clone(remote),
                sync_config.clone(),
                Arc::clone(&pending_deletes),
            )),
            _ => None,
        };

        let use_async = config.async_save && config.world_size == 1;

        let core = Arc::new(Core {
            storage,
            manifest,
            config,
            merger,
            syncer,
            in_flight,
            pending_deletes,
            retained_base: Mutex::new(None),
            dtype_patterns_checked: AtomicBool::new(false),
        });

        let saver = if use_async {
            Some(AsyncSaver::new(Arc::clone(&core)))
        } else {
            None
        };

        Ok(Coordinator { saver, core })
    }

    /// Save a checkpoint for this rank.
    ///
    /// For single-rank (world_size=1): creates snapshot, saves tensors, finalizes.
    /// With async_save (default), the heavy work (hashing, delta detection,
    /// compression, disk write) runs on a background thread and this call
    /// returns as soon as the previous save has drained.
    ///
    /// For multi-rank: rank 0 must call create_snapshot first, then all ranks
    /// call save_rank, then rank 0 calls finalize.
    ///
    /// `tensors` is a list of TensorData (name + bytes + shape + dtype).
    pub fn save(
        &self,
        step: u64,
        tensors: Vec<TensorData>,
        metadata: HashMap<String, String>,
    ) -> Result<Uuid> {
        if self.core.config.world_size != 1 {
            return Err(MoonclipError::Config(
                "Multi-rank save requires explicit create_snapshot/save_rank/finalize flow. \
                 Use save_rank() instead."
                    .into(),
            ));
        }

        let snap_id = Uuid::new_v4();
        if let Some(ref saver) = self.saver {
            saver.submit(SaveJob {
                snap_id,
                step,
                tensors,
                metadata,
            })?;
        } else {
            self.core.save_sync(snap_id, step, tensors, metadata)?;
        }
        Ok(snap_id)
    }

    /// How long the last save spent waiting for the previous one to drain.
    ///
    /// One save is allowed in flight, so a writer that has not finished stops
    /// the next `save` before any of its work begins. From outside Moonclip
    /// that wait is indistinguishable from a slow shadow copy — the caller
    /// sees one call that took longer — and the two have nothing in common:
    /// the copy is memory bandwidth and scales with the model, this is
    /// backpressure and scales with the cadence and the storage. A caller that
    /// reports its own phase timings needs them apart to name the right one.
    ///
    /// Read it on the thread that just called `save`, where it describes that
    /// save. It is one value, overwritten by each submission, and Moonclip
    /// takes one save at a time from one thread; nothing here makes it a
    /// meaningful answer to a thread that did not ask.
    ///
    /// Zero when there is nothing to wait for: with `async_save` off the write
    /// happens on the calling thread, and that is the write, not a queue.
    pub fn last_queue_wait(&self) -> Duration {
        match self.saver {
            Some(ref saver) => Duration::from_nanos(saver.last_wait.load(Ordering::Relaxed)),
            None => Duration::ZERO,
        }
    }

    /// Block until any in-flight background save completes and surface
    /// its error, if any.
    pub fn flush(&self) -> Result<()> {
        match self.saver {
            Some(ref s) => s.flush(),
            None => Ok(()),
        }
    }

    fn wait_idle(&self) {
        if let Some(ref s) = self.saver {
            s.wait_idle();
        }
    }

    /// Create a new snapshot entry (rank 0 only in multi-rank).
    /// Returns the snapshot ID that all ranks should use.
    pub fn create_snapshot(&self, step: u64, metadata: HashMap<String, String>) -> Result<Uuid> {
        self.flush()?;
        self.core.create_snapshot(step, metadata)
    }

    /// Save this rank's tensors into an existing snapshot.
    /// Called by each rank independently.
    pub fn save_rank(&self, snap_id: Uuid, tensors: Vec<TensorData>) -> Result<()> {
        self.flush()?;
        self.core.save_rank(snap_id, tensors)
    }

    /// Finalize a snapshot (rank 0 only in multi-rank).
    /// Marks it as complete and triggers retention/merging.
    pub fn finalize_snapshot(&self, snap_id: Uuid) -> Result<()> {
        self.flush()?;
        self.core.finalize_snapshot(snap_id)
    }

    /// Load a snapshot for this rank. Returns tensor name → raw bytes.
    pub fn load(&self, snap_id: Uuid) -> Result<HashMap<String, Vec<u8>>> {
        self.flush()?;
        self.core.load(snap_id)
    }

    /// Load the latest finalized snapshot.
    pub fn load_latest(&self) -> Result<(Uuid, HashMap<String, Vec<u8>>)> {
        self.flush()?;
        self.core.load_latest()
    }

    /// List all snapshots.
    pub fn list_snapshots(&self) -> Vec<SnapshotInfo> {
        self.wait_idle();
        self.core.list_snapshots()
    }

    /// Force merge all pending deltas into a full checkpoint.
    /// Fold every pending delta into one full snapshot, and wait for it.
    ///
    /// Returns when the merge is done rather than when it is queued: the
    /// caller that matters is `save_final`, which merges and then uploads, and
    /// an upload that starts while the merge is still running leaves the run's
    /// last checkpoint — the one everything was folded into — on the machine.
    pub fn merge_now(&self) -> Result<()> {
        self.wait_idle();
        self.core.merge_now()
    }

    /// Force sync all local data to remote storage immediately.
    /// Push everything to remote storage and wait for it.
    ///
    /// Returns the outcome rather than swallowing it: a run that cannot learn
    /// its checkpoints never left the machine has no way to react.
    pub fn sync_now(&self) -> Result<()> {
        self.wait_idle();
        self.core.sync_now()
    }

    /// Pull this store back from the remote, for a machine that has none.
    ///
    /// Returns whether anything was restored. The caller decides when: only it
    /// knows whether this run means to resume, and a whole checkpoint coming
    /// down the wire is not something to discover by having built a manager.
    ///
    /// See [`crate::remote_sync::restore_from_remote`] for what the gap cost
    /// before this existed.
    pub fn restore_from_remote(&self) -> Result<bool> {
        self.wait_idle();
        self.core.restore_from_remote()
    }
}

impl Core {
    /// Full single-rank save pipeline (runs on the caller thread or the
    /// background save thread).
    ///
    /// The pipeline is a handful of parallel passes over several gigabytes, and
    /// they run in Moonclip's own rayon pool rather than the global one — see
    /// [`crate::pool`]. Installing it here rather than at the callers covers
    /// both entry points, this being the one place both go through.
    fn save_sync(
        &self,
        snap_id: Uuid,
        step: u64,
        tensors: Vec<TensorData>,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        crate::pool::install(move || self.save_sync_in_pool(snap_id, step, tensors, metadata))
    }

    fn save_sync_in_pool(
        &self,
        snap_id: Uuid,
        step: u64,
        tensors: Vec<TensorData>,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        let snap_dir = format!("snapshots/{}", snap_id);
        let save_started = Instant::now();
        let raw_bytes: u64 = tensors.iter().map(|t| t.data.len() as u64).sum();

        // The base, and a pin on it that lasts until this save is recorded.
        //
        // Both decisions are made under the manifest lock, together: picking a
        // base and then pinning it in two steps is the race itself. A base a
        // merge has already claimed is not picked up at all — this save writes
        // a full snapshot instead, which costs bytes for one step and cannot
        // end up as a delta against something that no longer exists.
        let manifest = self.manifest.lock().unwrap();
        let force_full = manifest.should_force_full(step);
        let base_snap = if force_full {
            None
        } else {
            manifest
                .last_full_snapshot()
                .filter(|b| !self.in_flight.is_doomed(b.id))
                .cloned()
        };
        let base_pin = base_snap
            .as_ref()
            .map(|b| self.in_flight.pin(std::slice::from_ref(&b.id)));
        drop(manifest);

        let created_at = Utc::now();
        let context = SnapshotContext {
            step,
            created_at,
            metadata: metadata.clone(),
        };
        let rank_entry =
            self.save_rank_tensors(snap_id, &snap_dir, &base_snap, tensors, &context)?;

        let snapshot = Snapshot {
            id: snap_id,
            step,
            created_at,
            ranks: {
                let mut m = HashMap::new();
                m.insert(self.config.rank, rank_entry);
                m
            },
            base_snapshot_id: base_snap.as_ref().map(|s| s.id),
            metadata,
            compression: self.config.compression.clone(),
            finalized: true,
        };

        let mut manifest = self.manifest.lock().unwrap();
        manifest.snapshots.push(snapshot);
        // apply_retention persists the manifest when done.
        let evicted = self.apply_retention(&mut manifest)?;
        drop(manifest);
        // Released here, and not before: the pin has to cover the push above,
        // or a merge is free to fold the base away in the window between the
        // last byte of this pack and the snapshot appearing in the manifest —
        // which is the window `crate::inflight` exists to close, and it leaves
        // a delta naming a base that is no longer on disk.
        //
        // It cannot be held past this point either. The unlink below waits for
        // readers, so a pin still held there is one this thread would wait on
        // itself, and it takes only the right arrangement of rollback
        // protection for retention to choose the very snapshot this save
        // pinned. Between the two, so neither can happen.
        drop(base_pin);
        // Unlinked after the lock is released, never under it: the unlink
        // waits for readers, and a reader reacquires the manifest lock while
        // pinned. See `Core::remove_snapshots`.
        crate::merger::unlink_snapshots(
            &self.storage,
            &self.pending_deletes,
            &self.in_flight,
            &evicted,
            crate::inflight::RETENTION_UNLINK_WAIT,
        );

        // Notify merger
        if let Some(ref merger) = self.merger {
            merger.notify();
        }

        // Notify remote syncer
        if let Some(ref syncer) = self.syncer {
            syncer.notify_save();
        }

        crate::profile::report(
            &format!("step {step}"),
            save_started.elapsed(),
            raw_bytes,
        );

        Ok(())
    }

    fn create_snapshot(&self, step: u64, metadata: HashMap<String, String>) -> Result<Uuid> {
        let snap_id = Uuid::new_v4();

        // A base a merge has already claimed is not picked up, exactly as in
        // the single-rank path: it will not exist by the time anyone loads the
        // delta written against it. Rank 0 chooses for every rank here, so
        // getting it wrong costs the whole snapshot rather than one shard.
        let manifest = self.manifest.lock().unwrap();
        let force_full = manifest.should_force_full(step);
        let base_id = if force_full {
            None
        } else {
            manifest
                .last_full_snapshot()
                .filter(|b| !self.in_flight.is_doomed(b.id))
                .map(|s| s.id)
        };
        drop(manifest);

        let snapshot = Snapshot {
            id: snap_id,
            step,
            created_at: Utc::now(),
            ranks: HashMap::new(),
            base_snapshot_id: base_id,
            metadata,
            compression: self.config.compression.clone(),
            finalized: false, // Not yet finalized
        };

        let mut manifest = self.manifest.lock().unwrap();
        manifest.snapshots.push(snapshot);
        self.persist_manifest(&manifest)?;
        drop(manifest);

        Ok(snap_id)
    }

    fn save_rank(&self, snap_id: Uuid, tensors: Vec<TensorData>) -> Result<()> {
        crate::pool::install(move || self.save_rank_in_pool(snap_id, tensors))
    }

    fn save_rank_in_pool(&self, snap_id: Uuid, tensors: Vec<TensorData>) -> Result<()> {
        // Re-read manifest from storage for multi-rank correctness
        // (another rank may have created the snapshot)
        self.reload_manifest()?;

        let manifest = self.manifest.lock().unwrap();
        let snap = manifest
            .find_snapshot(snap_id)
            .ok_or_else(|| MoonclipError::NotFound(format!("Snapshot {snap_id}")))?
            .clone();

        let base_snap = snap
            .base_snapshot_id
            .and_then(|id| manifest.find_snapshot(id).cloned());
        // Pinned in the same critical section that read it, and held until
        // this rank's pack is written. `save_rank_tensors` reads the base off
        // storage after the lock is gone — without this, rank 0's own merger
        // could unlink those packs mid-read, which is the same race the
        // single-rank path guards. Rank 0 already refused a doomed base in
        // `create_snapshot`; this covers a merge that starts afterwards.
        let _base_pin = base_snap
            .as_ref()
            .map(|b| self.in_flight.pin(std::slice::from_ref(&b.id)));
        drop(manifest);

        let snap_dir = format!("snapshots/{}", snap_id);
        let context = SnapshotContext {
            step: snap.step,
            created_at: snap.created_at,
            metadata: snap.metadata.clone(),
        };
        let rank_entry =
            self.save_rank_tensors(snap_id, &snap_dir, &base_snap, tensors, &context)?;

        // **This rank writes its pack and stops there.**
        //
        // It used to reload the manifest, insert its own entry and write the
        // whole file back. Every rank is a separate process sharing one
        // storage root, so that sequence is read-modify-write with no lock
        // around it: two ranks that reload before either persists each write
        // back a manifest containing only their own entry, and the loser's
        // shard is referenced by nothing. The Python layer worked around it by
        // putting a barrier between the ranks and saving them one at a time —
        // which made the README's "each rank saves its own shard
        // independently" false, and a multi-GPU save serial.
        //
        // Nothing needs writing here anyway: the pack already carries a
        // descriptor holding exactly this `RankEntry` (see `crate::pack`), and
        // `finalize_snapshot` reads them back to assemble the snapshot. One
        // writer, no lock, and the shards go in parallel.
        let _ = rank_entry;
        Ok(())
    }

    /// Every rank's entry for `snap_id`, read back from the packs themselves.
    ///
    /// A rank that has not written yet is simply absent, which is what the
    /// caller counts against `world_size`. A pack that is present but
    /// unreadable is an error rather than an absence: silently finalizing a
    /// snapshot one rank short produces a checkpoint that cannot be restored,
    /// and does it at the only moment anyone was watching.
    fn collect_rank_shards(&self, snap_id: Uuid) -> Result<HashMap<u32, RankEntry>> {
        let prefix = format!("snapshots/{snap_id}");
        let files = match self.storage.list(&prefix) {
            Ok(f) => f,
            Err(MoonclipError::NotFound(_)) => Vec::new(),
            Err(e) => return Err(e),
        };

        let mut shards = HashMap::new();
        for file in files.iter().filter(|f| f.ends_with(".pack")) {
            let header = self.storage.get_range(file, 0, pack::HEADER_LEN as usize)?;
            let (offset, length) = pack::decode_header(&header).ok_or_else(|| {
                MoonclipError::Storage(format!("{file} has no pack header"))
            })?;
            let encoded = self.storage.get_range(file, offset, length as usize)?;
            if encoded.len() != length as usize {
                return Err(MoonclipError::Storage(format!(
                    "{file} is truncated: the rank that wrote it did not finish"
                )));
            }
            let descriptor: pack::PackDescriptor = serde_json::from_slice(&encoded)
                .map_err(|e| MoonclipError::Serialization(format!("{file}: {e}")))?;
            if descriptor.snapshot_id != snap_id {
                return Err(MoonclipError::Storage(format!(
                    "{file} belongs to snapshot {}, not {snap_id}",
                    descriptor.snapshot_id
                )));
            }
            shards.insert(descriptor.rank.rank, descriptor.rank);
        }

        Ok(shards)
    }

    fn finalize_snapshot(&self, snap_id: Uuid) -> Result<()> {
        // Re-read from storage to see all ranks' contributions
        self.reload_manifest()?;

        // The ranks are collected from the packs they wrote, not from what
        // they managed to get into the manifest — see `save_rank_in_pool`.
        // Every rank's `RankEntry` is inside its own pack's descriptor, so
        // this is the same data by a route that has one writer.
        let shards = self.collect_rank_shards(snap_id)?;

        let mut manifest = self.manifest.lock().unwrap();
        if let Some(snap) = manifest.snapshots.iter_mut().find(|s| s.id == snap_id) {
            // Verify all ranks have reported
            let expected = self.config.world_size;
            let actual = shards.len() as u32;
            if actual < expected {
                return Err(MoonclipError::Config(format!(
                    "Cannot finalize: only {actual}/{expected} ranks have saved"
                )));
            }
            snap.ranks = shards;
            snap.finalized = true;
        } else {
            return Err(MoonclipError::NotFound(format!("Snapshot {snap_id}")));
        }

        let evicted = self.apply_retention(&mut manifest)?;
        drop(manifest);
        // Unlinked after the lock is released, never under it: the unlink
        // waits for readers, and a reader reacquires the manifest lock while
        // pinned. See `Core::remove_snapshots`.
        crate::merger::unlink_snapshots(
            &self.storage,
            &self.pending_deletes,
            &self.in_flight,
            &evicted,
            crate::inflight::RETENTION_UNLINK_WAIT,
        );

        if let Some(ref merger) = self.merger {
            merger.notify();
        }

        if let Some(ref syncer) = self.syncer {
            syncer.notify_save();
        }

        Ok(())
    }

    fn load(&self, snap_id: Uuid) -> Result<HashMap<String, Vec<u8>>> {
        crate::pool::install(move || self.load_in_pool(snap_id))
    }

    fn load_in_pool(&self, snap_id: Uuid) -> Result<HashMap<String, Vec<u8>>> {
        use rayon::prelude::*;

        // Pinned for the duration, base included: a merge running beside this
        // load would otherwise delete the packs it is reading. The pin is what
        // holds off the unlink — a merge that has already claimed these blocks
        // in `unlink_snapshots` until this guard drops. Checking `is_doomed`
        // here instead would not do: it can be overtaken between the manifest
        // read and the first byte read. See `crate::inflight`.
        //
        // Note the lock discipline this depends on: the base entries are read
        // under the manifest lock *below*, while this pin is held. That is
        // only safe because nothing waits on a pin while holding the manifest
        // lock.
        let manifest = self.manifest.lock().unwrap();
        let snap = manifest
            .find_snapshot(snap_id)
            .ok_or_else(|| MoonclipError::NotFound(format!("Snapshot {snap_id}")))?
            .clone();
        let mut pinned = vec![snap.id];
        pinned.extend(snap.base_snapshot_id);
        let _pin = self.in_flight.pin(&pinned);
        drop(manifest);

        let rank_entry = snap.ranks.get(&self.config.rank).ok_or_else(|| {
            MoonclipError::NotFound(format!(
                "Rank {} not found in snapshot {snap_id}",
                self.config.rank
            ))
        })?;

        // Pre-load pack file if present (single read for all tensors)
        let pack_data = if let Some(ref pack_file) = rank_entry.pack_file {
            Some(self.storage.get(pack_file)?)
        } else {
            None
        };

        // Prefetch the base snapshot's entries + pack once, shared across
        // all tensors (instead of re-reading per delta/skipped tensor).
        type BaseCtx = (
            Uuid,
            HashMap<String, TensorEntry>,
            CompressionAlgo,
            Option<Arc<Vec<u8>>>,
        );
        let needs_base = rank_entry
            .tensors
            .iter()
            .any(|t| t.storage != TensorStorage::Full);
        let base_ctx: Option<BaseCtx> = match (needs_base, snap.base_snapshot_id) {
            (true, Some(base_id)) => {
                let manifest = self.manifest.lock().unwrap();
                let base_snap = manifest.find_snapshot(base_id).cloned();
                drop(manifest);
                match base_snap {
                    Some(bs) => match bs.ranks.get(&self.config.rank) {
                        Some(re) => {
                            let pack = re
                                .pack_file
                                .as_ref()
                                .map(|p| self.storage.get(p))
                                .transpose()?
                                .map(Arc::new);
                            let entries = re
                                .tensors
                                .iter()
                                .map(|t| (t.name.clone(), t.clone()))
                                .collect();
                            Some((base_id, entries, bs.compression.clone(), pack))
                        }
                        None => None,
                    },
                    None => None,
                }
            }
            _ => None,
        };

        let resolver = |base_id: Uuid,
                        tensor_name: &str|
         -> Result<crate::tensor::BaseEntry> {
            if let Some((cached_id, entries, comp, pack)) = &base_ctx {
                if *cached_id == base_id {
                    let entry = entries.get(tensor_name).cloned().ok_or_else(|| {
                        MoonclipError::NotFound(format!(
                            "Tensor '{}' not in base snapshot {base_id}",
                            tensor_name
                        ))
                    })?;
                    return Ok((entry, comp.clone(), pack.clone()));
                }
            }
            self.find_base_tensor_entry_with_pack(base_id, tensor_name)
        };

        let pairs: Result<Vec<(String, Vec<u8>)>> = rank_entry
            .tensors
            .par_iter()
            .filter(|entry| entry.storage != TensorStorage::Alias)
            .map(|entry| {
                tensor::load_tensor(
                    entry,
                    snap.base_snapshot_id,
                    self.storage.as_ref(),
                    &snap.compression,
                    pack_data.as_deref(),
                    &resolver,
                )
                .map(|data| (entry.name.clone(), data))
            })
            .collect();

        let mut out: HashMap<String, Vec<u8>> = pairs?.into_iter().collect();

        // Second pass: aliases share bytes with a tensor already loaded above.
        for entry in &rank_entry.tensors {
            if entry.storage != TensorStorage::Alias {
                continue;
            }
            let target = entry.alias_of.as_deref().ok_or_else(|| {
                MoonclipError::NotFound(format!("Alias '{}' has no target", entry.name))
            })?;
            let data = out.get(target).cloned().ok_or_else(|| {
                MoonclipError::NotFound(format!(
                    "Alias '{}' points at '{}', which is not in this snapshot",
                    entry.name, target
                ))
            })?;
            out.insert(entry.name.clone(), data);
        }

        Ok(out)
    }

    fn load_latest(&self) -> Result<(Uuid, HashMap<String, Vec<u8>>)> {
        let manifest = self.manifest.lock().unwrap();
        let snap = manifest
            .snapshots
            .iter()
            .rev()
            .find(|s| s.finalized)
            .ok_or_else(|| MoonclipError::NotFound("No finalized snapshots".into()))?
            .clone();
        drop(manifest);

        let id = snap.id;
        let data = self.load(id)?;
        Ok((id, data))
    }

    fn list_snapshots(&self) -> Vec<SnapshotInfo> {
        let manifest = self.manifest.lock().unwrap();
        manifest
            .snapshots
            .iter()
            .filter(|s| s.finalized)
            .map(|s| {
                let mut total_compressed = 0u64;
                let mut total_raw = 0u64;
                let mut total_skipped = 0usize;
                let mut total_delta = 0usize;
                let mut total_full = 0usize;
                for re in s.ranks.values() {
                    total_compressed += re.total_compressed;
                    total_raw += re.total_raw;
                    total_skipped += re.skipped_count;
                    total_delta += re.delta_count;
                    total_full += re.full_count;
                }
                SnapshotInfo {
                    id: s.id,
                    step: s.step,
                    is_delta: s.base_snapshot_id.is_some(),
                    metadata: s.metadata.clone(),
                    total_compressed,
                    total_raw,
                    skipped_tensors: total_skipped,
                    delta_tensors: total_delta,
                    full_tensors: total_full,
                    ranks: s.ranks.len() as u32,
                }
            })
            .collect()
    }

    fn merge_now(&self) -> Result<()> {
        match self.merger {
            Some(ref merger) => merger.force_full_merge(),
            None => Ok(()),
        }
    }

    fn sync_now(&self) -> Result<()> {
        match self.syncer {
            Some(ref syncer) => syncer.sync_now(),
            None => Ok(()),
        }
    }

    fn restore_from_remote(&self) -> Result<bool> {
        let Some(remote) = self.config.remote_storage.as_ref() else {
            return Ok(false);
        };
        if !crate::remote_sync::restore_from_remote(&self.storage, remote)? {
            return Ok(false);
        }
        // The manifest in memory was read at construction, when the store was
        // empty. Everything downstream asks it, not the disk.
        let mut fresh = load_manifest(self.storage.as_ref(), &self.config)?;

        // And it is not to be trusted as it stands. `manifest.json` is
        // rewritten and re-uploaded on its own, while snapshot data goes up
        // separately, so a bucket can hold a manifest naming snapshots whose
        // packs never made it. Seen on a six-node bench on 2026-08-19: the
        // restore succeeded and the first `load` then failed on a pack that
        // was in no bucket at all.
        //
        // Dropping the entries whose files did not arrive turns that into a
        // resume from an older step, which is the difference between losing
        // some progress and losing the run.
        let before = fresh.snapshots.len();
        let storage = Arc::clone(&self.storage);
        fresh
            .snapshots
            .retain(|snapshot| snapshot_is_whole(storage.as_ref(), snapshot));
        let dropped = before - fresh.snapshots.len();
        if dropped > 0 {
            eprintln!(
                "[Moonclip] {dropped} snapshot(s) named by the remote manifest                  did not arrive with it and were left out; resuming from what did."
            );
            if let Ok(json) = serde_json::to_vec_pretty(&fresh) {
                let _ = self.storage.put("manifest.json", &json);
            }
        }

        let usable = !fresh.snapshots.is_empty();
        *self.manifest.lock().unwrap() = fresh;
        Ok(usable)
    }

    // ── Internal ─────────────────────────────────────────────────────

    fn save_rank_tensors(
        &self,
        snap_id: Uuid,
        snap_dir: &str,
        base_snap: &Option<Snapshot>,
        tensors: Vec<TensorData>,
        context: &SnapshotContext,
    ) -> Result<RankEntry> {
        // ── 1. Build lazy base cache (no upfront disk read) ──────────
        let base_cache = self.build_base_cache(base_snap);

        self.check_dtype_patterns(&tensors);

        // ── 2. Deduplicate, then process the survivors in parallel ───
        // Tied embeddings and repeated buffers hold identical bytes under
        // several names; store them once and point the rest at the copy.
        let dedup = tensor::dedup_plan(&tensors);
        let unique: Vec<&TensorData> = tensors
            .iter()
            .zip(&dedup)
            .filter(|(_, alias)| alias.is_none())
            .map(|(t, _)| t)
            .collect();

        let mut processed = tensor::process_tensors_parallel(
            &unique,
            base_cache.as_ref(),
            &self.config.compression,
            self.config.delta_max_ratio,
            &self.config.save_dtype,
        )?;
        drop(unique);

        // Alias entries inherit dtype/cast/hash from the stored tensor, so
        // they must be built after processing. They carry no data, so they
        // do not affect the pack offsets assigned below.
        let stored: HashMap<&str, &TensorEntry> = processed
            .iter()
            .map(|pt| (pt.entry.name.as_str(), &pt.entry))
            .collect();
        let alias_entries: Vec<TensorEntry> = tensors
            .iter()
            .zip(&dedup)
            .filter_map(|(t, alias)| {
                let target = stored.get(alias.as_deref()?)?;
                Some(tensor::make_alias_entry(t, target))
            })
            .collect();
        drop(stored);
        processed.extend(alias_entries.into_iter().map(|entry| tensor::ProcessedTensor {
            entry,
            write_data: None,
        }));
        // A full snapshot's bytes are exactly what the next delta needs as a
        // base. Retaining them here moves buffers that were about to be
        // freed; on a delta snapshot they are not the base, so they go.
        if base_snap.is_none() {
            self.retain_base(snap_id, tensors, &processed);
        } else {
            drop(tensors); // raw tensor data no longer needed
        }

        // ── 3. Assign pack offsets and write all parts in one file ──
        // Offsets start after the header, not at zero: the pack carries its own
        // description (see `crate::pack`). Every reader takes these offsets
        // from the manifest rather than assuming the first tensor sits at the
        // start of the file, so the shift is invisible to them.
        let mut offset = pack::HEADER_LEN;
        for pt in &mut processed {
            if let Some(ref data) = pt.write_data {
                pt.entry.offset = offset;
                offset += data.len() as u64;
            }
        }
        let blobs_end = offset;

        // ── 4. Build the rank entry, which the descriptor carries ───
        //
        // Always a pack, even when this rank has no bytes to write.
        //
        // A step where nothing changed is the ordinary case in fine-tuning:
        // whole stretches of the model do not move, every tensor comes out
        // `Skipped`, and there is not a single blob to store. Skipping the
        // pack then saved one small file and cost the checkpoint its
        // description — `recover_or_reclaim_orphans` rebuilds a snapshot the
        // manifest lost from the descriptors in its packs, so a snapshot with
        // no pack is not merely unrecovered, there is nothing for recovery to
        // read. A process killed between the save and the manifest write left
        // it unrecoverable, and `crate::pack` claimed that state was
        // impossible.
        //
        // What it costs: a header plus a JSON descriptor per rank per empty
        // snapshot, a few hundred bytes, against a checkpoint that can be
        // brought back.
        let pack_file = format!("{}/rank_{}.pack", snap_dir, self.config.rank);

        let mut entries = Vec::new();
        let mut total_compressed = 0u64;
        let mut total_raw = 0u64;
        let mut skipped = 0usize;
        let mut delta = 0usize;
        let mut full = 0usize;

        // By reference, and cloning only the entries: the descriptor has to be
        // built before the pack is written, and `processed` still owns the
        // compressed blobs. Consuming it here would mean copying those blobs
        // to write them — gigabytes, on the path this whole session was spent
        // making cheaper.
        for pt in &processed {
            total_compressed += pt.entry.compressed_size;
            total_raw += pt.entry.raw_size;
            match pt.entry.storage {
                // Aliases write no bytes, same as a skipped tensor.
                TensorStorage::Skipped | TensorStorage::Alias => skipped += 1,
                TensorStorage::DeltaXor => delta += 1,
                TensorStorage::Full => full += 1,
            }
            entries.push(pt.entry.clone());
        }

        let rank_entry = RankEntry {
            rank: self.config.rank,
            tensors: entries,
            pack_file: Some(pack_file),
            total_compressed,
            total_raw,
            skipped_count: skipped,
            delta_count: delta,
            full_count: full,
        };

        // ── 5. Write header + blobs + descriptor, in one object ─────
        let pack_filename = rank_entry.pack_file.as_ref().expect("set just above");
        let descriptor = pack::PackDescriptor {
            snapshot_id: snap_id,
            step: context.step,
            created_at: context.created_at,
            base_snapshot_id: base_snap.as_ref().map(|s| s.id),
            metadata: context.metadata.clone(),
            compression: self.config.compression.clone(),
            world_size: self.config.world_size,
            rank: rank_entry.clone(),
        };
        let encoded = serde_json::to_vec(&descriptor)
            .map_err(|e| MoonclipError::Serialization(e.to_string()))?;
        let header = pack::encode_header(blobs_end, encoded.len() as u64);

        let mut parts: Vec<&[u8]> = Vec::with_capacity(processed.len() + 2);
        parts.push(&header);
        parts.extend(processed.iter().filter_map(|pt| pt.write_data.as_deref()));
        parts.push(&encoded);

        crate::profile::time(crate::profile::Phase::PackWrite, || {
            self.storage.put_parts(pack_filename, &parts)
        })?;

        Ok(rank_entry)
    }

    /// Build a lazy BaseCache referring to the base snapshot.
    /// Tensor data is read on demand during processing.
    fn build_base_cache(&self, base_snap: &Option<Snapshot>) -> Option<BaseCache> {
        let base = base_snap.as_ref()?;
        let rank_entry = base.ranks.get(&self.config.rank)?;

        let entries: HashMap<String, TensorEntry> = rank_entry
            .tensors
            .iter()
            .map(|t| (t.name.clone(), t.clone()))
            .collect();

        // Only for the snapshot the bytes were taken from. After a restart
        // there is nothing retained and this is None, which is correct rather
        // than merely safe: the disk copy is the same data.
        let raw = self
            .retained_base
            .lock()
            .unwrap()
            .as_ref()
            .filter(|r| r.snap_id == base.id)
            .map(|r| Arc::clone(&r.tensors));

        Some(BaseCache {
            storage: Arc::clone(&self.storage),
            pack_file: rank_entry.pack_file.clone(),
            entries,
            compression: base.compression.clone(),
            raw,
        })
    }

    /// Keep this snapshot's bytes as the base for the next delta.
    ///
    /// Called only for full snapshots, and it *moves* the incoming data —
    /// these are the buffers that would otherwise be dropped at the end of
    /// the save, so retaining them allocates nothing.
    ///
    /// Report `save_dtype` patterns that match none of this rank's tensors.
    ///
    /// A pattern that fires on nothing is this feature's quiet failure:
    /// `{"weight": "bf16"}` reads as if it did something, matches no tensor —
    /// the names arrive with their prefix, `model/weight` — and writes a full-precision
    /// checkpoint without a word. It is the same shape as the `save_dtype`
    /// typo that [`crate::cast::DType::parse`] refuses, and it gets the same
    /// treatment rather than a different one because it is harder to catch.
    ///
    /// A complaint, not a refusal: on a sharded save a pattern that matches
    /// nothing *on this rank* is legitimate, and a rank that stops the run
    /// over it would be worse than the checkpoint it was trying to protect.
    fn check_dtype_patterns(&self, tensors: &[TensorData]) {
        if self.dtype_patterns_checked.swap(true, Ordering::Relaxed) {
            return;
        }
        let names: Vec<&str> = tensors.iter().map(|t| t.name.as_str()).collect();
        let dead = self.config.save_dtype.dead_patterns(&names);
        if dead.is_empty() {
            return;
        }
        eprintln!(
            "[Moonclip] save_dtype pattern(s) {} matched none of this rank's \
             {} tensors, so they cast nothing. Patterns are globs over the \
             full tensor name (e.g. '{}'); '*' matches any run of characters.",
            dead.join(", "),
            names.len(),
            names.first().copied().unwrap_or("model/weight"),
        );
    }

    /// Decided per tensor, because `save_dtype` is. What a later delta is
    /// computed against is the bytes on disk; for a tensor that was cast,
    /// those are not the bytes that arrived here, and retaining them would
    /// XOR the next delta against the wrong base. So the test is
    /// `original_dtype.is_none()`, which is set by
    /// [`tensor::process_tensor`] exactly when the stored bytes and the
    /// incoming bytes are the same bytes.
    ///
    /// The damage is not silent corruption, and it is worth being accurate
    /// about which it is. A wrong base is caught three times over: a cast
    /// that changes the width fails the length check in
    /// [`crate::delta::compute_delta`], a cast that keeps it produces a
    /// high-entropy XOR that [`crate::delta::pays_off`] rejects, and
    /// anything that survives both fails the hash the load path verifies
    /// after applying the delta. What it costs is the optimization it was
    /// meant to provide — a full save where a delta belonged — or, in the
    /// worst case, a checkpoint that refuses to load at resume. Two of those
    /// three guards are heuristics about size, which is not what a
    /// correctness argument should rest on; this is.
    ///
    /// Under the old all-or-nothing rule a single cast turned the whole
    /// optimization off. Per tensor, the ordinary configuration — moments to
    /// bf16, weights untouched — keeps the weights in memory, which is the
    /// third of the state where the delta actually pays.
    ///
    /// A partial map needs no special handling downstream: `BaseCache` looks
    /// each tensor up by name and falls back to storage on a miss, so a
    /// tensor left out here simply takes the path it took before.
    fn retain_base(
        &self,
        snap_id: Uuid,
        tensors: Vec<TensorData>,
        processed: &[tensor::ProcessedTensor],
    ) {
        if !self.config.keep_base_in_memory {
            return;
        }

        // Only tensors stored in full, and uncast, can be delta'd against
        // later from memory.
        let full: std::collections::HashSet<&str> = processed
            .iter()
            .filter(|pt| {
                pt.entry.storage == TensorStorage::Full && pt.entry.original_dtype.is_none()
            })
            .map(|pt| pt.entry.name.as_str())
            .collect();

        let retained: HashMap<String, Vec<u8>> = tensors
            .into_iter()
            .filter(|t| full.contains(t.name.as_str()))
            .map(|t| (t.name, t.data))
            .collect();

        *self.retained_base.lock().unwrap() = Some(RetainedBase {
            snap_id,
            tensors: Arc::new(retained),
        });
    }

    /// Find base tensor entry + pack data for loading (fallback path for
    /// base snapshots not covered by the per-load prefetch).
    /// Returns (TensorEntry, CompressionAlgo, Option<pack_data>).
    fn find_base_tensor_entry_with_pack(
        &self,
        base_id: Uuid,
        tensor_name: &str,
    ) -> Result<crate::tensor::BaseEntry> {
        let manifest = self.manifest.lock().unwrap();
        let base_snap = manifest
            .find_snapshot(base_id)
            .ok_or_else(|| MoonclipError::NotFound(format!("Base snapshot {base_id}")))?;

        let rank_entry = base_snap.ranks.get(&self.config.rank).ok_or_else(|| {
            MoonclipError::NotFound(format!(
                "Rank {} not in base snapshot {base_id}",
                self.config.rank
            ))
        })?;

        let entry = rank_entry
            .tensors
            .iter()
            .find(|t| t.name == tensor_name)
            .ok_or_else(|| {
                MoonclipError::NotFound(format!(
                    "Tensor '{}' not in base snapshot {base_id}",
                    tensor_name
                ))
            })?
            .clone();

        let compression = base_snap.compression.clone();
        let pack_file = rank_entry.pack_file.clone();
        drop(manifest);

        let pack_data = match pack_file {
            Some(ref pack) => Some(Arc::new(self.storage.get(pack)?)),
            None => None,
        };

        Ok((entry, compression, pack_data))
    }

    /// Re-read manifest from storage (for multi-rank sync).
    fn reload_manifest(&self) -> Result<()> {
        match self.storage.get("manifest.json") {
            Ok(data) => {
                // Strip trailing null bytes from page-aligned storage
                let end = data
                    .iter()
                    .rposition(|&b| b != 0)
                    .map(|i| i + 1)
                    .unwrap_or(0);
                let trimmed = &data[..end];
                let new_manifest: Manifest = serde_json::from_slice(trimmed)
                    .map_err(|e| MoonclipError::Serialization(e.to_string()))?;
                let mut m = self.manifest.lock().unwrap();
                *m = new_manifest;
                Ok(())
            }
            Err(MoonclipError::NotFound(_)) => Ok(()),
            Err(e) => Err(e),
        }
    }

    fn persist_manifest(&self, manifest: &Manifest) -> Result<()> {
        let json = serde_json::to_vec_pretty(manifest)
            .map_err(|e| MoonclipError::Serialization(e.to_string()))?;
        self.storage.put("manifest.json", &json)
    }

    /// Apply retention policy: cap full snapshots, then cap total snapshots.
    ///
    /// Returns the snapshots it evicted. Their files are still on disk: the
    /// caller unlinks them after dropping the manifest lock, because the
    /// unlink waits for readers and that wait deadlocks under this lock. See
    /// [`Core::remove_snapshots`].
    fn apply_retention(&self, manifest: &mut Manifest) -> Result<Vec<Snapshot>> {
        let max_full = manifest.retention.max_full_snapshots;
        let max_total = manifest.retention.effective_total_cap();
        let rollback_ids = manifest.rollback_snapshot_ids();
        let mut evicted: Vec<Snapshot> = Vec::new();

        // ── Pass 1: Cap full snapshot count ──────────────────────────
        // Loop until we're within the full-snapshot limit.
        loop {
            let full_count = manifest
                .snapshots
                .iter()
                .filter(|s| s.base_snapshot_id.is_none() && s.finalized)
                .count();

            if full_count <= max_full {
                break;
            }

            // Find oldest removable full snapshot (not a rollback)
            let oldest_removable = manifest
                .snapshots
                .iter()
                .find(|s| {
                    s.base_snapshot_id.is_none()
                        && s.finalized
                        && !rollback_ids.contains(&s.id)
                })
                .map(|s| s.id);

            match oldest_removable {
                Some(oldest_id) => {
                    evicted.extend(self.remove_snapshot_group(manifest, oldest_id)?)
                }
                None => break, // All remaining are rollback-protected
            }
        }

        // ── Pass 2: Cap total snapshot count ─────────────────────────
        //
        // Oldest *delta* first, one at a time. Removing whole groups here is
        // what the cap used to do, and it is catastrophic in the configuration
        // that matters: every delta is computed against the last full, so an
        // ordinary run is a single group thousands of steps long. Dropping
        // "the oldest group" to get under the cap then deletes the base and
        // every delta hanging off it — the entire history, not its oldest
        // slice. Measured: three snapshots with `max_total_snapshots: 2` left
        // **zero**, and `keep_last` is exactly this knob.
        //
        // A delta is a leaf: nothing is stored against it, so dropping the
        // oldest costs one restore point and nothing else. The full stays for
        // as long as a delta still needs it as a base — which means the oldest
        // surviving step may be a full older than the cap would suggest, and
        // that is the honest answer rather than an unreadable checkpoint.
        loop {
            let total = manifest.snapshots.iter().filter(|s| s.finalized).count();
            if total <= max_total {
                break;
            }

            let oldest_delta = manifest
                .snapshots
                .iter()
                .find(|s| {
                    s.base_snapshot_id.is_some()
                        && s.finalized
                        && !rollback_ids.contains(&s.id)
                })
                .map(|s| s.id);

            if let Some(delta_id) = oldest_delta {
                evicted.extend(self.remove_snapshots(manifest, &[delta_id])?);
                continue;
            }

            // Nothing but fulls left, so a group is now a single snapshot and
            // removing one loses only itself. Never the last one though: a cap
            // of one means one checkpoint, not none.
            let removable_fulls: Vec<Uuid> = manifest
                .snapshots
                .iter()
                .filter(|s| {
                    s.base_snapshot_id.is_none()
                        && s.finalized
                        && !rollback_ids.contains(&s.id)
                })
                .map(|s| s.id)
                .collect();

            if max_total >= 1 && removable_fulls.len() <= 1 {
                break;
            }
            match removable_fulls.first() {
                Some(&oldest_id) => {
                    evicted.extend(self.remove_snapshot_group(manifest, oldest_id)?)
                }
                None => break,
            }
        }

        self.persist_manifest(manifest)?;
        Ok(evicted)
    }

    /// Remove a full snapshot and all its dependent deltas.
    fn remove_snapshot_group(
        &self,
        manifest: &mut Manifest,
        full_id: Uuid,
    ) -> Result<Vec<Snapshot>> {
        let to_remove: Vec<Uuid> = manifest
            .snapshots
            .iter()
            .filter(|s| s.id == full_id || s.base_snapshot_id == Some(full_id))
            .map(|s| s.id)
            .collect();
        self.remove_snapshots(manifest, &to_remove)
    }

    /// Drop these snapshots from the manifest and hand their descriptions back
    /// for the caller to unlink.
    ///
    /// The caller owns the question of what is safe to remove: a delta can go
    /// on its own, a full only with everything computed against it.
    ///
    /// **It does not delete.** This runs with the manifest lock held, and the
    /// unlink has to wait for readers — a wait that cannot happen under that
    /// lock, because `load_in_pool` reacquires it while pinned and the two
    /// would deadlock. So the files go back to the caller, which unlinks them
    /// through `merger::unlink_snapshots` once the lock is gone. Same rule as
    /// the merger, for the same reason: retention evicting a snapshot out from
    /// under a load that is reading it is the same failure as a merge doing
    /// it.
    fn remove_snapshots(
        &self,
        manifest: &mut Manifest,
        to_remove: &[Uuid],
    ) -> Result<Vec<Snapshot>> {
        let evicted: Vec<Snapshot> = manifest
            .snapshots
            .iter()
            .filter(|s| to_remove.contains(&s.id))
            .cloned()
            .collect();

        manifest.snapshots.retain(|s| !to_remove.contains(&s.id));
        Ok(evicted)
    }
}

// ── Background save worker ───────────────────────────────────────────

struct SaveJob {
    snap_id: Uuid,
    step: u64,
    tensors: Vec<TensorData>,
    metadata: HashMap<String, String>,
}

struct SaverShared {
    busy: bool,
    error: Option<String>,
}

/// Dedicated thread that runs the save pipeline. At most one save is
/// in flight (bound-1 queue): submitting waits for the previous job.
struct AsyncSaver {
    tx: Option<mpsc::Sender<SaveJob>>,
    shared: Arc<(Mutex<SaverShared>, Condvar)>,
    handle: Option<thread::JoinHandle<()>>,
    /// How long the most recent `submit` waited for the previous save to
    /// drain. Written by the caller's thread in `submit`, read by that same
    /// thread once `save` returns — see [`Coordinator::last_queue_wait`].
    last_wait: AtomicU64,
}

impl AsyncSaver {
    fn new(core: Arc<Core>) -> Self {
        let (tx, rx) = mpsc::channel::<SaveJob>();
        let shared = Arc::new((
            Mutex::new(SaverShared {
                busy: false,
                error: None,
            }),
            Condvar::new(),
        ));
        let shared2 = Arc::clone(&shared);

        let handle = thread::Builder::new()
            .name("moonclip-async-saver".into())
            .spawn(move || {
                for job in rx {
                    // A panic here used to kill this thread with `busy` still
                    // set and nobody left to clear it, so the next `submit`
                    // waited on the condvar forever: the training loop hung
                    // silently and permanently, which on rented hardware is an
                    // idle GPU billing until a human notices. Catching it turns
                    // a panic into the same thing a storage failure already is
                    // — an error the next save, flush or load reports.
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        core.save_sync(job.snap_id, job.step, job.tensors, job.metadata)
                    }));

                    let mut state = shared2.0.lock().unwrap();
                    state.busy = false;
                    match result {
                        Ok(Ok(())) => {}
                        Ok(Err(e)) => state.error = Some(e.to_string()),
                        Err(panic) => {
                            let what = panic
                                .downcast_ref::<&str>()
                                .map(|s| (*s).to_string())
                                .or_else(|| panic.downcast_ref::<String>().cloned())
                                .unwrap_or_else(|| "unknown payload".into());
                            state.error = Some(format!("save thread panicked: {what}"));
                        }
                    }
                    drop(state);
                    shared2.1.notify_all();
                }
            })
            .expect("Failed to spawn moonclip-async-saver thread");

        AsyncSaver {
            tx: Some(tx),
            shared,
            handle: Some(handle),
            last_wait: AtomicU64::new(0),
        }
    }

    /// Wait until no save is in flight. Does not consume errors.
    fn wait_idle(&self) {
        let (lock, cvar) = &*self.shared;
        let mut state = lock.lock().unwrap();
        while state.busy {
            state = cvar.wait(state).unwrap();
        }
    }

    /// Wait until idle and surface any pending background error (once).
    fn flush(&self) -> Result<()> {
        let (lock, cvar) = &*self.shared;
        let mut state = lock.lock().unwrap();
        while state.busy {
            state = cvar.wait(state).unwrap();
        }
        match state.error.take() {
            Some(e) => Err(MoonclipError::Storage(format!(
                "Background save failed: {e}"
            ))),
            None => Ok(()),
        }
    }

    /// Submit a job, waiting for the previous one to drain first.
    ///
    /// The wait and the claim of the in-flight slot happen under a single
    /// lock acquisition. Doing them separately — wait for idle, release,
    /// re-acquire, set busy — lets two threads both observe an idle saver and
    /// both queue a job. That breaks the bound-1 invariant, and because
    /// `busy` is one flag for what is then two outstanding jobs, the worker
    /// clears it after the first completes: `flush()` returns while a save is
    /// still queued, and the caller sees a manifest missing that snapshot.
    fn submit(&self, job: SaveJob) -> Result<()> {
        // This function blocks. Doing that on a Moonclip pool thread is the
        // deadlock described in `crate::python::save_tensors`: the wait can
        // only end when a save pipeline finishes, that pipeline runs on this
        // same pool, and a blocked worker may be sitting on a piece of it.
        // Callers must submit from outside the pool.
        debug_assert!(
            rayon::current_thread_index().is_none(),
            "AsyncSaver::submit blocked on a Moonclip pool thread"
        );
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| MoonclipError::Storage("Background saver is shut down".into()))?;

        let (lock, cvar) = &*self.shared;
        let queued = Instant::now();
        let mut state = lock.lock().unwrap();
        while state.busy {
            state = cvar.wait(state).unwrap();
        }
        // How long that wait was is the difference between "the copy is slow"
        // and "the writer never catches up", and the caller cannot tell them
        // apart from the outside. Kept for the caller as well as logged: a
        // number only `MOONCLIP_PROFILE=1` will show is a number nobody has
        // when the question comes up, which is on a rented machine mid-run.
        let waited = queued.elapsed();
        self.last_wait
            .store(waited.as_nanos() as u64, Ordering::Relaxed);
        // Surface a failure from the previous save before starting another,
        // matching what the old `self.flush()?` on entry did.
        if let Some(e) = state.error.take() {
            return Err(MoonclipError::Storage(format!("Background save failed: {e}")));
        }
        state.busy = true;
        drop(state);
        crate::profile::note_queue_wait(waited);

        if tx.send(job).is_err() {
            lock.lock().unwrap().busy = false;
            cvar.notify_all();
            return Err(MoonclipError::Storage(
                "Background saver channel closed".into(),
            ));
        }
        Ok(())
    }

    fn shutdown(&mut self) {
        self.wait_idle();
        self.tx.take(); // close channel → worker exits
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for AsyncSaver {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Public snapshot info returned by list_snapshots.
#[derive(Debug, Clone)]
pub struct SnapshotInfo {
    pub id: Uuid,
    pub step: u64,
    pub is_delta: bool,
    pub metadata: HashMap<String, String>,
    pub total_compressed: u64,
    pub total_raw: u64,
    pub skipped_tensors: usize,
    pub delta_tensors: usize,
    pub full_tensors: usize,
    pub ranks: u32,
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::LocalStorage;

    fn make_coordinator(dir: &std::path::Path) -> Coordinator {
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir).unwrap());
        let config = CoordinatorConfig {
            world_size: 1,
            rank: 0,
            compression: CompressionAlgo::Zstd { level: 1 },
            retention: RetentionPolicy {
                max_full_snapshots: 5,
                max_deltas_per_full: 10,
                full_snapshot_every_steps: 10000,
                max_total_snapshots: None,
            },
            delta_max_ratio: 0.95,
            ..Default::default()
        };
        Coordinator::new(storage, config).unwrap()
    }

    fn sample_tensors(seed: u8) -> Vec<TensorData> {
        vec![TensorData {
            name: "w".into(),
            shape: vec![8192],
            dtype: "uint8".into(),
            data: vec![seed; 8192],
        }]
    }

    #[test]
    fn single_rank_save_load() {
        let dir = tempfile::tempdir().unwrap();
        let coord = make_coordinator(dir.path());

        let snap_id = coord.save(100, sample_tensors(42), HashMap::new()).unwrap();
        let loaded = coord.load(snap_id).unwrap();
        assert_eq!(loaded["w"], vec![42u8; 8192]);
    }

    #[test]
    fn sync_save_mode_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let config = CoordinatorConfig {
            compression: CompressionAlgo::Zstd { level: 1 },
            async_save: false,
            ..Default::default()
        };
        let coord = Coordinator::new(storage, config).unwrap();

        let snap_id = coord.save(100, sample_tensors(7), HashMap::new()).unwrap();
        let loaded = coord.load(snap_id).unwrap();
        assert_eq!(loaded["w"], vec![7u8; 8192]);
    }

    #[test]
    fn delta_chain_multiple_steps() {
        let dir = tempfile::tempdir().unwrap();
        let coord = make_coordinator(dir.path());

        let mut data = vec![0u8; 10_000];
        coord.save(100, vec![TensorData {
            name: "w".into(), shape: vec![10_000], dtype: "uint8".into(), data: data.clone(),
        }], HashMap::new()).unwrap();

        for step in (200..=500).step_by(100) {
            let idx = step as usize % data.len();
            data[idx] = (step / 100) as u8;
            coord.save(step, vec![TensorData {
                name: "w".into(), shape: vec![10_000], dtype: "uint8".into(), data: data.clone(),
            }], HashMap::new()).unwrap();
        }

        let (_, loaded) = coord.load_latest().unwrap();
        assert_eq!(loaded["w"], data);
    }

    #[test]
    fn retention_caps_total_snapshots() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let config = CoordinatorConfig {
            world_size: 1,
            rank: 0,
            compression: CompressionAlgo::Zstd { level: 1 },
            retention: RetentionPolicy {
                max_full_snapshots: 2,
                max_deltas_per_full: 5,
                full_snapshot_every_steps: 5, // full every 5 steps
                max_total_snapshots: Some(4), // hard cap at 4
            },
            delta_max_ratio: 0.95,
            ..Default::default()
        };
        let coord = Coordinator::new(storage, config).unwrap();

        // Save 10 steps → should never exceed 5 total
        for i in 0..10 {
            coord.save(i, vec![TensorData {
                name: "w".into(), shape: vec![100], dtype: "uint8".into(), data: vec![i as u8; 100],
            }], HashMap::new()).unwrap();
        }

        let snaps = coord.list_snapshots();
        assert!(
            snaps.len() <= 5,
            "Expected <=5 total snapshots, got {}",
            snaps.len()
        );
    }

    #[test]
    fn retention_policy_removes_old_snapshots() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let config = CoordinatorConfig {
            world_size: 1,
            rank: 0,
            compression: CompressionAlgo::Zstd { level: 1 },
            retention: RetentionPolicy {
                max_full_snapshots: 2,
                max_deltas_per_full: 5,
                full_snapshot_every_steps: 1, // force full every step
                max_total_snapshots: None,
            },
            delta_max_ratio: 0.95,
            ..Default::default()
        };
        let coord = Coordinator::new(storage, config).unwrap();

        for i in 0..6 {
            coord.save(i, vec![TensorData {
                name: "w".into(), shape: vec![100], dtype: "uint8".into(), data: vec![i as u8; 100],
            }], HashMap::new()).unwrap();
        }

        let snaps = coord.list_snapshots();
        let full_count = snaps.iter().filter(|s| !s.is_delta).count();
        assert!(
            full_count <= 2,
            "Expected <=2 full snapshots, got {full_count}"
        );
    }

    #[test]
    fn empty_tensor_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let coord = make_coordinator(dir.path());

        let tensors = vec![TensorData {
            name: "empty".into(), shape: vec![0], dtype: "float32".into(), data: vec![],
        }];
        let snap_id = coord.save(1, tensors, HashMap::new()).unwrap();
        let loaded = coord.load(snap_id).unwrap();
        assert_eq!(loaded["empty"].len(), 0);
    }

    #[test]
    fn many_tensors_save_load() {
        let dir = tempfile::tempdir().unwrap();
        let coord = make_coordinator(dir.path());

        let tensors: Vec<TensorData> = (0..50)
            .map(|i| TensorData {
                name: format!("layer.{}.weight", i),
                shape: vec![64, 64],
                dtype: "float32".into(),
                data: vec![i as u8; 64 * 64 * 4],
            })
            .collect();

        let snap_id = coord.save(100, tensors.clone(), HashMap::new()).unwrap();
        let loaded = coord.load(snap_id).unwrap();

        assert_eq!(loaded.len(), 50);
        for t in &tensors {
            assert_eq!(loaded[&t.name], t.data);
        }
    }

    #[test]
    fn special_characters_in_tensor_names() {
        let dir = tempfile::tempdir().unwrap();
        let coord = make_coordinator(dir.path());

        let tensors = vec![
            TensorData {
                name: "module.layers.0.self_attn.q_proj.weight".into(),
                shape: vec![64], dtype: "float32".into(), data: vec![1u8; 256],
            },
            TensorData {
                name: "model/encoder/block_0/layer_0".into(),
                shape: vec![32], dtype: "float32".into(), data: vec![2u8; 128],
            },
        ];

        let snap_id = coord.save(1, tensors.clone(), HashMap::new()).unwrap();
        let loaded = coord.load(snap_id).unwrap();

        assert_eq!(loaded["module.layers.0.self_attn.q_proj.weight"], tensors[0].data);
        assert_eq!(loaded["model/encoder/block_0/layer_0"], tensors[1].data);
    }

    #[test]
    fn force_full_after_n_steps() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let config = CoordinatorConfig {
            world_size: 1,
            rank: 0,
            compression: CompressionAlgo::Zstd { level: 1 },
            retention: RetentionPolicy {
                max_full_snapshots: 10,
                max_deltas_per_full: 100,
                full_snapshot_every_steps: 3,
                max_total_snapshots: None,
            },
            delta_max_ratio: 0.95,
            ..Default::default()
        };
        let coord = Coordinator::new(storage, config).unwrap();

        let base = vec![0u8; 8192];
        for step in 0..9 {
            let mut d = base.clone();
            d[0] = step as u8;
            coord.save(step, vec![TensorData {
                name: "w".into(), shape: vec![8192], dtype: "uint8".into(), data: d,
            }], HashMap::new()).unwrap();
        }

        let snaps = coord.list_snapshots();
        let full_count = snaps.iter().filter(|s| !s.is_delta).count();
        assert!(full_count >= 3, "Expected >=3 full snapshots, got {full_count}");
    }

    #[test]
    fn load_nonexistent_snapshot_errors() {
        let dir = tempfile::tempdir().unwrap();
        let coord = make_coordinator(dir.path());
        let fake_id = uuid::Uuid::new_v4();
        assert!(coord.load(fake_id).is_err());
    }

    #[test]
    fn metadata_preserved() {
        let dir = tempfile::tempdir().unwrap();
        let coord = make_coordinator(dir.path());

        let mut meta = HashMap::new();
        meta.insert("loss".into(), "0.123".into());
        meta.insert("lr".into(), "3e-4".into());
        meta.insert("epoch".into(), "5".into());

        coord.save(100, sample_tensors(0), meta).unwrap();

        let snaps = coord.list_snapshots();
        assert_eq!(snaps[0].metadata["loss"], "0.123");
        assert_eq!(snaps[0].metadata["lr"], "3e-4");
        assert_eq!(snaps[0].metadata["epoch"], "5");
    }

    /// Every concurrent `save()` must be in the manifest once `flush()`
    /// returns.
    ///
    /// Regression: `submit` waited for the saver to go idle, released the
    /// lock, then re-acquired it to set `busy`. Two threads could both clear
    /// the wait and both queue a job against what is a single flag, so the
    /// worker cleared `busy` after the first of them finished — `flush()`
    /// returned with a save still queued and the manifest came up short.
    ///
    /// Repeated, because one pass through a lost-update window is not
    /// guaranteed to lose anything: the pre-fix code failed a few runs in ten.
    #[test]
    fn concurrent_saves_all_land() {
        const THREADS: u64 = 10;
        const TRIALS: usize = 20;

        for trial in 0..TRIALS {
            let dir = tempfile::tempdir().unwrap();
            let storage: Arc<dyn StorageBackend> =
                Arc::new(LocalStorage::new(dir.path()).unwrap());
            let config = CoordinatorConfig {
                world_size: 1,
                rank: 0,
                compression: CompressionAlgo::Zstd { level: 1 },
                retention: RetentionPolicy {
                    // Deliberately generous: a short manifest must mean a lost
                    // save, never retention pruning.
                    max_full_snapshots: 1000,
                    max_deltas_per_full: 1000,
                    full_snapshot_every_steps: 100_000,
                    max_total_snapshots: None,
                },
                delta_max_ratio: 0.95,
                ..Default::default()
            };
            let coord = Arc::new(Coordinator::new(storage, config).unwrap());

            let handles: Vec<_> = (1..=THREADS)
                .map(|step| {
                    let c = Arc::clone(&coord);
                    thread::spawn(move || c.save(step, sample_tensors(step as u8), HashMap::new()))
                })
                .collect();
            for h in handles {
                h.join().expect("worker panicked").expect("save failed");
            }
            coord.flush().unwrap();

            assert_eq!(
                coord.list_snapshots().len(),
                THREADS as usize,
                "trial {trial}: a concurrent save is missing from the manifest"
            );
        }
    }

    /// The queue wait is available to the caller, not only to a log line.
    ///
    /// GPU-61: a caller timing its own save could not tell the shadow copy
    /// from the wait for the previous writer, so a phase dominated by a memcpy
    /// was read as a writer in deficit — and an issue was opened against the
    /// writer on the strength of it.
    #[test]
    fn the_queue_wait_is_reported_to_the_caller() {
        let dir = tempfile::tempdir().unwrap();
        let coord = make_coordinator(dir.path());

        // Nothing has ever been in flight, so there was nothing to wait for.
        coord.save(1, sample_tensors(1), HashMap::new()).unwrap();
        let first = coord.last_queue_wait();

        // No flush in between. `submit` sets `busy` before it returns, so the
        // second save cannot get past the condvar until the first has drained
        // — the wait is forced by the ordering, not hoped for from timing.
        let big = vec![TensorData {
            name: "w".into(),
            shape: vec![16 << 20],
            dtype: "uint8".into(),
            data: (0..(16u32 << 20)).map(|i| (i ^ (i >> 7)) as u8).collect(),
        }];
        coord.save(2, big.clone(), HashMap::new()).unwrap();
        coord.save(3, big, HashMap::new()).unwrap();
        let queued = coord.last_queue_wait();
        coord.flush().unwrap();

        assert!(
            queued > first,
            "a save submitted behind an in-flight one reported no wait \
             (first {first:?}, queued {queued:?})"
        );
        assert!(
            queued > Duration::ZERO,
            "the wait for the previous writer came back as zero"
        );
    }

    /// With no background saver there is no queue to wait in. The write then
    /// happens on the calling thread, and calling that backpressure would name
    /// the write itself as the thing standing in front of the write.
    #[test]
    fn a_synchronous_save_waits_for_nobody() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let config = CoordinatorConfig {
            world_size: 1,
            rank: 0,
            async_save: false,
            compression: CompressionAlgo::Zstd { level: 1 },
            ..Default::default()
        };
        let coord = Coordinator::new(storage, config).unwrap();

        coord.save(1, sample_tensors(1), HashMap::new()).unwrap();
        coord.save(2, sample_tensors(2), HashMap::new()).unwrap();

        assert_eq!(coord.last_queue_wait(), Duration::ZERO);
    }

    #[test]
    fn pack_file_created() {
        let dir = tempfile::tempdir().unwrap();
        let coord = make_coordinator(dir.path());

        let tensors = vec![
            TensorData { name: "a".into(), shape: vec![100], dtype: "uint8".into(), data: vec![1u8; 100] },
            TensorData { name: "b".into(), shape: vec![200], dtype: "uint8".into(), data: vec![2u8; 200] },
        ];

        coord.save(1, tensors, HashMap::new()).unwrap();
        coord.flush().unwrap(); // drain the background save

        // Check that a .pack file exists (not individual .bin files)
        let manifest = coord.core.manifest.lock().unwrap();
        let snap = &manifest.snapshots[0];
        let re = snap.ranks.get(&0).unwrap();
        assert!(re.pack_file.is_some(), "Expected pack_file to be set");
        assert!(
            re.pack_file.as_ref().unwrap().ends_with(".pack"),
            "Expected .pack extension"
        );
    }

    // ── Reclaiming orphaned snapshots ───────────────────────────────

    /// Files under a snapshot directory, whatever their depth.
    fn snapshot_files(storage: &dyn StorageBackend, id: &str) -> Vec<String> {
        storage
            .list("snapshots")
            .unwrap_or_default()
            .into_iter()
            .filter(|f| f.starts_with(&format!("snapshots/{id}/")))
            .collect()
    }

    /// A snapshot directory holding a pack with no descriptor — either a
    /// write that died before the header landed, or a pack from before the
    /// format carried one.
    fn orphan_on_disk(storage: &dyn StorageBackend, id: Uuid) {
        storage
            .put(&format!("snapshots/{id}/rank_0.pack"), &[7u8; 4096])
            .unwrap();
    }

    /// Rewrite the manifest without `drop`, leaving its data and sidecar on
    /// disk — exactly the state a process killed between the two writes leaves.
    fn forget_snapshot(storage: &dyn StorageBackend, drop: Uuid) {
        let raw = storage.get("manifest.json").unwrap();
        let end = raw.iter().rposition(|&b| b != 0).map(|i| i + 1).unwrap_or(0);
        let mut manifest: Manifest = serde_json::from_slice(&raw[..end]).unwrap();
        manifest.snapshots.retain(|s| s.id != drop);
        storage
            .put("manifest.json", &serde_json::to_vec_pretty(&manifest).unwrap())
            .unwrap();
    }

    fn open(storage: &Arc<dyn StorageBackend>) -> Coordinator {
        Coordinator::new(
            Arc::clone(storage),
            CoordinatorConfig {
                compression: CompressionAlgo::Zstd { level: 1 },
                retention: RetentionPolicy {
                    full_snapshot_every_steps: 1,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap()
    }

    /// The case the whole sidecar exists for: a checkpoint that finished
    /// writing, whose manifest update never happened, comes back.
    #[test]
    fn a_complete_checkpoint_missing_from_the_manifest_is_recovered() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());

        let coord = open(&storage);
        coord.save(0, evolving_state(0), HashMap::new()).unwrap();
        coord.save(1, evolving_state(1), HashMap::new()).unwrap();
        coord.flush().unwrap();
        let lost = coord.list_snapshots()[1].id;
        drop(coord);

        forget_snapshot(storage.as_ref(), lost);

        let restarted = open(&storage);
        let steps: Vec<u64> = restarted.list_snapshots().iter().map(|s| s.step).collect();
        assert_eq!(steps, vec![0, 1], "the finished checkpoint was not recovered");

        // Recovered is only worth something if it reads back.
        let loaded = restarted.load(lost).expect("recovered snapshot must load");
        assert_eq!(loaded.get("w").unwrap()[0], 1u8);
    }

    #[test]
    fn a_recovered_checkpoint_is_written_back_to_the_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());

        let coord = open(&storage);
        coord.save(0, evolving_state(0), HashMap::new()).unwrap();
        coord.flush().unwrap();
        let lost = coord.list_snapshots()[0].id;
        drop(coord);

        forget_snapshot(storage.as_ref(), lost);
        drop(open(&storage)); // recovers, and should persist

        // A second restart must find it already in the manifest rather than
        // rediscovering it — otherwise the recovery only ever lived in memory
        // and the next crash loses it again.
        let raw = storage.get("manifest.json").unwrap();
        let end = raw.iter().rposition(|&b| b != 0).map(|i| i + 1).unwrap_or(0);
        let manifest: Manifest = serde_json::from_slice(&raw[..end]).unwrap();
        assert_eq!(manifest.snapshots.len(), 1);
        assert_eq!(manifest.snapshots[0].id, lost);
    }

    /// Presence is not integrity. A pack of the right length whose bytes are
    /// wrong must not be readmitted: it would pass startup and fail at resume,
    /// which is the worst moment to find out.
    #[test]
    fn a_checkpoint_with_corrupted_data_is_not_recovered() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());

        let coord = open(&storage);
        coord.save(0, evolving_state(0), HashMap::new()).unwrap();
        coord.flush().unwrap();
        let lost = coord.list_snapshots()[0].id;
        drop(coord);

        forget_snapshot(storage.as_ref(), lost);

        // Flip a byte inside the compressed data, keeping the pack's length.
        //
        // Where matters: LocalStorage pads to a page boundary, so most of this
        // file is zero padding that nothing ever reads. Corrupting there proves
        // nothing — the check is driven by each tensor's recorded offset and
        // size, which the sidecar carries.
        let pack_path = format!("snapshots/{lost}/rank_0.pack");
        let header = storage
            .get_range(&pack_path, 0, pack::HEADER_LEN as usize)
            .unwrap();
        let (offset, length) = pack::decode_header(&header).expect("our own header");
        let encoded = storage.get_range(&pack_path, offset, length as usize).unwrap();
        let described: pack::PackDescriptor = serde_json::from_slice(&encoded).unwrap();
        let tensor = &described.rank.tensors[0];
        assert!(tensor.compressed_size > 8, "nothing to corrupt");

        let mut bytes = storage.get(&pack_path).unwrap();
        bytes[tensor.offset as usize + tensor.compressed_size as usize / 2] ^= 0xff;
        storage.put(&pack_path, &bytes).unwrap();

        let restarted = open(&storage);
        assert!(
            restarted.list_snapshots().is_empty(),
            "corrupted data was readmitted to the manifest"
        );
        assert!(
            !storage.exists(&pack_path).unwrap_or(false),
            "the corrupted snapshot should have been discarded"
        );
    }

    /// A delta's pack describes itself exactly as a full's does — same code
    /// path, `base_snapshot_id` set instead of null — so it recovers too, and
    /// still reconstructs against its base afterwards.
    #[test]
    fn a_delta_is_recovered_and_still_applies_to_its_base() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let config = || CoordinatorConfig {
            compression: CompressionAlgo::Zstd { level: 1 },
            // Keep the second save a delta against the first.
            retention: RetentionPolicy {
                full_snapshot_every_steps: 1000,
                ..Default::default()
            },
            ..Default::default()
        };

        let coord = Coordinator::new(Arc::clone(&storage), config()).unwrap();
        coord.save(0, evolving_state(0), HashMap::new()).unwrap();
        coord.save(1, evolving_state(1), HashMap::new()).unwrap();
        coord.flush().unwrap();
        let snaps = coord.list_snapshots();
        assert!(snaps[1].is_delta, "second save should be a delta");
        let delta = snaps[1].id;
        drop(coord);

        // Lose only the delta's manifest entry; its base stays recorded.
        forget_snapshot(storage.as_ref(), delta);

        let restarted = Coordinator::new(Arc::clone(&storage), config()).unwrap();
        let steps: Vec<u64> = restarted.list_snapshots().iter().map(|s| s.step).collect();
        assert_eq!(steps, vec![0, 1], "the delta was not recovered");

        // Recovering a delta is only meaningful if the XOR still resolves
        // against the base it names.
        let loaded = restarted.load(delta).expect("recovered delta must load");
        assert_eq!(loaded.get("w").unwrap()[0], 1u8);
    }

    /// A merge collapses the base and every delta after it into one snapshot,
    /// then deletes the originals. From that moment it is the only copy of the
    /// run's history — so it is the snapshot that must survive a manifest
    /// update that never landed, and the one whose loss costs everything.
    #[test]
    fn a_merged_snapshot_is_recovered_like_any_other() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let config = || CoordinatorConfig {
            compression: CompressionAlgo::Zstd { level: 1 },
            // Keep the second save a delta, so the merge has a chain to fold.
            retention: RetentionPolicy {
                full_snapshot_every_steps: 1000,
                ..Default::default()
            },
            ..Default::default()
        };

        let coord = Coordinator::new(Arc::clone(&storage), config()).unwrap();
        coord.save(0, evolving_state(0), HashMap::new()).unwrap();
        coord.save(1, evolving_state(1), HashMap::new()).unwrap();
        coord.flush().unwrap();
        assert!(coord.list_snapshots()[1].is_delta, "second save should be a delta");
        drop(coord);

        // Merged directly rather than through `merge_now`: the merger runs on
        // its own thread with no completion to wait on, and this test is about
        // what the merge writes, not when.
        let raw = storage.get("manifest.json").unwrap();
        let end = raw.iter().rposition(|&b| b != 0).map(|i| i + 1).unwrap_or(0);
        let manifest: Manifest = serde_json::from_slice(&raw[..end]).unwrap();
        let manifest = Arc::new(Mutex::new(manifest));
        crate::merger::do_full_merge(
            &storage,
            &manifest,
            &CompressionAlgo::Zstd { level: 1 },
            &Arc::new(InFlight::default()),
            &Arc::new(crate::remote_sync::PendingDeletes::default()),
        )
        .unwrap();
        let merged = {
            let m = manifest.lock().unwrap();
            assert_eq!(m.snapshots.len(), 1, "base and delta collapse into one");
            m.snapshots[0].id
        };

        forget_snapshot(storage.as_ref(), merged);

        let restarted = Coordinator::new(Arc::clone(&storage), config()).unwrap();
        let snaps = restarted.list_snapshots();
        assert_eq!(snaps.len(), 1, "the merged snapshot was not recovered");
        assert_eq!(snaps[0].id, merged);
        assert!(!snaps[0].is_delta, "a merge yields a full");

        // Recovered is only worth something if it reads back as the state the
        // merge computed — the last step's weights, not the base's.
        let loaded = restarted.load(merged).expect("recovered merge must load");
        assert_eq!(loaded.get("w").unwrap()[0], 1u8);
    }

    /// A delta reconstructs nothing without its base.
    #[test]
    fn a_delta_whose_base_is_gone_is_not_recovered() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());

        let coord = Coordinator::new(
            Arc::clone(&storage),
            CoordinatorConfig {
                compression: CompressionAlgo::Zstd { level: 1 },
                // Keep the second save a delta against the first.
                retention: RetentionPolicy {
                    full_snapshot_every_steps: 1000,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        coord.save(0, evolving_state(0), HashMap::new()).unwrap();
        coord.save(1, evolving_state(1), HashMap::new()).unwrap();
        coord.flush().unwrap();
        let snaps = coord.list_snapshots();
        let (base, delta) = (snaps[0].id, snaps[1].id);
        assert!(snaps[1].is_delta, "second save should be a delta");
        drop(coord);

        // Lose both entries, and the base's data with them.
        forget_snapshot(storage.as_ref(), base);
        forget_snapshot(storage.as_ref(), delta);
        for file in snapshot_files(storage.as_ref(), &base.to_string()) {
            storage.delete(&file).unwrap();
        }

        let restarted = open(&storage);
        assert!(
            restarted.list_snapshots().is_empty(),
            "a delta was recovered with no base to apply it to"
        );
    }

    #[test]
    fn a_snapshot_without_a_description_is_reclaimed_at_startup() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());

        // A real snapshot, recorded in the manifest by a normal save.
        let coord = Coordinator::new(
            Arc::clone(&storage),
            CoordinatorConfig {
                compression: CompressionAlgo::Zstd { level: 1 },
                ..Default::default()
            },
        )
        .unwrap();
        coord.save(0, evolving_state(0), HashMap::new()).unwrap();
        coord.flush().unwrap();
        let kept = coord.list_snapshots()[0].id;
        drop(coord);

        let orphan = Uuid::new_v4();
        orphan_on_disk(storage.as_ref(), orphan);
        assert!(!snapshot_files(storage.as_ref(), &orphan.to_string()).is_empty());

        // Starting a coordinator is what reclaims: the run that left the
        // orphan is gone, and this is the moment nothing can be in flight.
        let restarted = Coordinator::new(
            Arc::clone(&storage),
            CoordinatorConfig {
                compression: CompressionAlgo::Zstd { level: 1 },
                ..Default::default()
            },
        )
        .unwrap();

        assert!(
            snapshot_files(storage.as_ref(), &orphan.to_string()).is_empty(),
            "the orphaned snapshot's data is still on disk"
        );
        assert!(
            !storage
                .exists(&format!("snapshots/{orphan}"))
                .unwrap_or(false),
            "the empty directory was left behind"
        );
        assert!(
            !snapshot_files(storage.as_ref(), &kept.to_string()).is_empty(),
            "a snapshot the manifest lists must survive"
        );
        assert_eq!(restarted.list_snapshots().len(), 1);
        assert!(restarted.load(kept).is_ok(), "the kept snapshot must still load");
    }

    /// The dangerous mistake would be reclaiming by directory listing alone and
    /// deleting a snapshot that is perfectly good.
    #[test]
    fn a_listed_snapshot_is_never_reclaimed() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let config = || CoordinatorConfig {
            compression: CompressionAlgo::Zstd { level: 1 },
            retention: RetentionPolicy {
                full_snapshot_every_steps: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let coord = Coordinator::new(Arc::clone(&storage), config()).unwrap();
        for step in 0..3u64 {
            coord.save(step, evolving_state(step), HashMap::new()).unwrap();
        }
        coord.flush().unwrap();
        let before: Vec<Uuid> = coord.list_snapshots().iter().map(|s| s.id).collect();
        drop(coord);

        let restarted = Coordinator::new(Arc::clone(&storage), config()).unwrap();
        let after: Vec<Uuid> = restarted.list_snapshots().iter().map(|s| s.id).collect();
        assert_eq!(before, after, "restarting must not remove any snapshot");

        for id in &after {
            let loaded = restarted.load(*id).unwrap_or_else(|e| {
                panic!("snapshot {id} was listed after a restart but cannot be read: {e}")
            });
            assert!(loaded.contains_key("w"));
        }
    }

    /// The mirror case: the manifest names a snapshot whose files are gone.
    /// Reclamation must not read that as a reason to do anything drastic, and
    /// above all must not panic — this runs inside the constructor, so a panic
    /// here means no coordinator at all.
    #[test]
    fn a_manifest_entry_with_no_files_does_not_panic() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let config = || CoordinatorConfig {
            compression: CompressionAlgo::Zstd { level: 1 },
            ..Default::default()
        };

        let coord = Coordinator::new(Arc::clone(&storage), config()).unwrap();
        coord.save(0, evolving_state(0), HashMap::new()).unwrap();
        coord.flush().unwrap();
        let id = coord.list_snapshots()[0].id;
        drop(coord);

        // Delete the data but leave the manifest entry, as a half-finished
        // cleanup or a truncated volume would.
        for file in snapshot_files(storage.as_ref(), &id.to_string()) {
            storage.delete(&file).unwrap();
        }

        let restarted = Coordinator::new(Arc::clone(&storage), config()).unwrap();
        assert_eq!(
            restarted.list_snapshots().len(),
            1,
            "the entry is still listed; reclamation does not edit the manifest"
        );
        // Reading it fails, which is honest — but it fails as an error.
        assert!(restarted.load(id).is_err());
    }

    /// Anything under `snapshots/` that is not a snapshot was put there by
    /// something else, and a cleanup routine that guesses is how unrelated data
    /// gets deleted.
    #[test]
    fn files_that_are_not_snapshots_are_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());

        storage.put("snapshots/notes.txt", b"someone put this here").unwrap();
        storage.put("snapshots/scratch/data.bin", b"and this").unwrap();

        let _coord = Coordinator::new(
            Arc::clone(&storage),
            CoordinatorConfig {
                compression: CompressionAlgo::Zstd { level: 1 },
                ..Default::default()
            },
        )
        .unwrap();

        assert!(storage.exists("snapshots/notes.txt").unwrap());
        assert!(storage.exists("snapshots/scratch/data.bin").unwrap());
    }

    /// Ranks 1..N write into directories rank 0 created; if they each reclaimed
    /// on the way up, one rank's startup would delete another's snapshot.
    #[test]
    fn only_rank_zero_reclaims() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());

        let orphan = Uuid::new_v4();
        orphan_on_disk(storage.as_ref(), orphan);

        let _rank_one = Coordinator::new(
            Arc::clone(&storage),
            CoordinatorConfig {
                world_size: 2,
                rank: 1,
                compression: CompressionAlgo::Zstd { level: 1 },
                ..Default::default()
            },
        )
        .unwrap();

        assert!(
            !snapshot_files(storage.as_ref(), &orphan.to_string()).is_empty(),
            "a non-zero rank reclaimed, and could have deleted a snapshot \
             another rank was still writing into"
        );
    }

    // ── Retention ───────────────────────────────────────────────────
    //
    // Retention deletes files. The tests above count what survives, which is
    // the easy half: a count still passes when the survivors have been ruined.
    // A delta whose base was pruned is a snapshot the manifest still lists and
    // nothing can read, and a training run finds that out at resume.

    /// Weights that change a little each step, the way training leaves them,
    /// and large enough for the delta engine to engage at all — the 100-byte
    /// tensors the older retention tests use are under `DELTA_MIN_SIZE`, so
    /// they never build the delta chains that make pruning dangerous.
    fn evolving_state(step: u64) -> Vec<TensorData> {
        let mut data = vec![7u8; 8192];
        data[(step as usize * 37) % 8192] = step as u8;
        data[0] = step as u8; // marker: which step these bytes are
        vec![TensorData {
            name: "w".into(),
            shape: vec![8192],
            dtype: "uint8".into(),
            data,
        }]
    }

    /// `keep_last: N` must keep the last N, not zero.
    ///
    /// The older retention tests all use a short `full_snapshot_every_steps`,
    /// so they build many small groups and pruning one whole group leaves the
    /// others. The configuration that ships is the opposite — one full, then
    /// deltas for thousands of steps, all in a single group — and there the
    /// total cap used to remove that group and erase the entire history.
    ///
    /// Found on a real run: 8 GPUs, `keep_last: 2`, three checkpoints handed
    /// off, and an empty store at the end.
    #[test]
    fn the_total_cap_trims_the_history_instead_of_erasing_it() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let coord = Coordinator::new(
            Arc::clone(&storage),
            CoordinatorConfig {
                compression: CompressionAlgo::Zstd { level: 1 },
                retention: RetentionPolicy {
                    max_full_snapshots: 5,
                    max_deltas_per_full: 10,
                    // One full at the start, everything after it a delta
                    // against it: what a training run actually looks like.
                    full_snapshot_every_steps: 5000,
                    max_total_snapshots: Some(2),
                },
                ..Default::default()
            },
        )
        .unwrap();

        for step in 0..3u64 {
            coord.save(step, evolving_state(step), HashMap::new()).unwrap();
        }
        coord.flush().unwrap();

        let snaps = coord.list_snapshots();
        assert!(
            !snaps.is_empty(),
            "the total cap erased every checkpoint the run had written"
        );
        assert_eq!(snaps.len(), 2, "the cap is two, so two survive");

        // The newest must be there — it is the one a resume would take — and
        // the base it deltas against has to have survived with it.
        assert_eq!(snaps.last().unwrap().step, 2, "the newest step was pruned");
        for info in &snaps {
            let loaded = coord.load(info.id).unwrap_or_else(|e| {
                panic!("step {} survived the cap but cannot be read: {e}", info.step)
            });
            assert_eq!(loaded.get("w").unwrap()[0], info.step as u8);
        }
    }

    /// The cap must still bite once the history is all fulls, and still not
    /// take the last one.
    ///
    /// Starts at step 1: step 0 is a multiple of `rollback_interval_steps` and
    /// is therefore rollback-protected, which is a different rule and has its
    /// own test.
    #[test]
    fn the_total_cap_still_prunes_a_history_of_fulls() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let coord = Coordinator::new(
            Arc::clone(&storage),
            CoordinatorConfig {
                compression: CompressionAlgo::Zstd { level: 1 },
                retention: RetentionPolicy {
                    max_full_snapshots: 10,
                    max_deltas_per_full: 10,
                    full_snapshot_every_steps: 1, // every save is a full
                    max_total_snapshots: Some(2),
                },
                ..Default::default()
            },
        )
        .unwrap();

        for step in 1..6u64 {
            coord.save(step, evolving_state(step), HashMap::new()).unwrap();
        }
        coord.flush().unwrap();

        let steps: Vec<u64> = coord.list_snapshots().iter().map(|s| s.step).collect();
        assert_eq!(steps, vec![4, 5], "the cap should keep the newest two");
    }

    #[test]
    fn every_snapshot_left_after_retention_still_loads() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let config = CoordinatorConfig {
            world_size: 1,
            rank: 0,
            compression: CompressionAlgo::Zstd { level: 1 },
            retention: RetentionPolicy {
                max_full_snapshots: 2,
                max_deltas_per_full: 5,
                full_snapshot_every_steps: 3,
                max_total_snapshots: Some(6),
            },
            delta_max_ratio: 0.95,
            ..Default::default()
        };
        let coord = Coordinator::new(storage, config).unwrap();

        // Long enough that retention prunes several times over.
        for step in 0..14u64 {
            coord.save(step, evolving_state(step), HashMap::new()).unwrap();
        }
        coord.flush().unwrap();

        let snaps = coord.list_snapshots();
        assert!(!snaps.is_empty(), "retention removed everything");
        assert!(snaps.len() <= 6, "the total cap was not applied");

        for info in &snaps {
            let loaded = coord
                .load(info.id)
                .unwrap_or_else(|e| panic!("step {} survived retention but cannot be read: {e}", info.step));
            let w = loaded.get("w").expect("tensor missing from a listed snapshot");
            assert_eq!(
                w[0], info.step as u8,
                "step {} came back as the state of another step",
                info.step
            );
            assert_eq!(w.len(), 8192);
        }
    }

    /// A backend that panics rather than returning an error, standing in for
    /// any bug that unwinds inside the save pipeline.
    struct PanickingStorage;

    impl StorageBackend for PanickingStorage {
        fn put(&self, _rel_path: &str, _data: &[u8]) -> Result<()> {
            panic!("disk on fire")
        }
        fn get(&self, rel_path: &str) -> Result<Vec<u8>> {
            Err(MoonclipError::NotFound(rel_path.into()))
        }
        fn exists(&self, _rel_path: &str) -> Result<bool> {
            Ok(false)
        }
        fn delete(&self, _rel_path: &str) -> Result<()> {
            Ok(())
        }
        fn list(&self, _prefix: &str) -> Result<Vec<String>> {
            Ok(Vec::new())
        }
    }

    /// A panic on the background saver must surface as an error, never as a
    /// hang.
    ///
    /// Regression: the worker died with `busy` still set and no one left to
    /// clear it, so the next `submit` waited on the condvar forever. The
    /// training loop stopped without a message, which on rented hardware is an
    /// idle GPU billing until a human notices. This is exactly how the bug was
    /// found — the test that provoked it ran for ten minutes before being
    /// killed.
    ///
    /// Run on a side thread with a deadline, so a regression fails this test
    /// instead of hanging the whole suite the way it did the first time.
    #[test]
    fn a_panic_in_the_save_thread_is_reported_not_hung() {
        let (done_tx, done_rx) = mpsc::channel();

        thread::spawn(move || {
            let storage: Arc<dyn StorageBackend> = Arc::new(PanickingStorage);
            let coord = Coordinator::new(
                storage,
                CoordinatorConfig {
                    compression: CompressionAlgo::Zstd { level: 1 },
                    ..Default::default()
                },
            )
            .unwrap();

            let _ = coord.save(0, evolving_state(0), HashMap::new());
            // Either this save or the flush has to report the failure; what
            // matters is that one of them returns at all.
            let second = coord.save(1, evolving_state(1), HashMap::new());
            let flushed = coord.flush();
            let _ = done_tx.send(second.is_err() || flushed.is_err());
        });

        match done_rx.recv_timeout(std::time::Duration::from_secs(30)) {
            Ok(reported) => assert!(
                reported,
                "the panic was swallowed: neither the next save nor flush reported it"
            ),
            Err(_) => panic!(
                "the save path hung after a panic on the background thread \
                 instead of reporting it"
            ),
        }
    }

    /// `rollback_interval_steps` reaches this from the Python constructor, and
    /// zero is the value someone will use to mean "no rollback snapshots".
    /// Retention runs inside every save, so a panic here is a lost run.
    #[test]
    fn a_rollback_interval_of_zero_does_not_bring_down_the_save() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let config = CoordinatorConfig {
            world_size: 1,
            rank: 0,
            compression: CompressionAlgo::Zstd { level: 1 },
            lineage: LineageConfig {
                rollback_interval_steps: 0,
                max_rollback_snapshots: 3,
            },
            ..Default::default()
        };
        let coord = Coordinator::new(storage, config).unwrap();

        for step in 0..4u64 {
            coord.save(step, evolving_state(step), HashMap::new()).unwrap();
        }
        coord.flush().unwrap();
        assert!(!coord.list_snapshots().is_empty());
    }

    /// Snapshots on the rollback interval are meant to outlive the ordinary
    /// retention cap — that is the whole point of keeping them.
    #[test]
    fn rollback_snapshots_survive_the_retention_cap() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let config = CoordinatorConfig {
            world_size: 1,
            rank: 0,
            compression: CompressionAlgo::Zstd { level: 1 },
            retention: RetentionPolicy {
                max_full_snapshots: 1,
                max_deltas_per_full: 2,
                full_snapshot_every_steps: 2,
                max_total_snapshots: Some(3),
            },
            lineage: LineageConfig {
                rollback_interval_steps: 4,
                max_rollback_snapshots: 3,
            },
            delta_max_ratio: 0.95,
            ..Default::default()
        };
        let coord = Coordinator::new(storage, config).unwrap();

        for step in 0..12u64 {
            coord.save(step, evolving_state(step), HashMap::new()).unwrap();
        }
        coord.flush().unwrap();

        let snaps = coord.list_snapshots();
        let protected: Vec<u64> = snaps
            .iter()
            .filter(|s| !s.is_delta && s.step % 4 == 0)
            .map(|s| s.step)
            .collect();
        assert!(
            !protected.is_empty(),
            "every rollback-interval snapshot was pruned; steps left: {:?}",
            snaps.iter().map(|s| s.step).collect::<Vec<_>>()
        );

        // And they have to be readable, not merely listed.
        for info in snaps.iter().filter(|s| !s.is_delta && s.step % 4 == 0) {
            let loaded = coord.load(info.id).unwrap_or_else(|e| {
                panic!("rollback snapshot at step {} cannot be read: {e}", info.step)
            });
            assert_eq!(loaded.get("w").unwrap()[0], info.step as u8);
        }
    }

    /// The other side of the previous test, and the point of the cap: without
    /// it, every snapshot that ever fell on the interval keeps its exemption
    /// and a long run ends with a store retention is not allowed to touch.
    /// Same shape as the test above, `max_rollback_snapshots` lowered to one.
    #[test]
    fn only_the_newest_rollback_snapshots_stay_protected() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let config = CoordinatorConfig {
            world_size: 1,
            rank: 0,
            compression: CompressionAlgo::Zstd { level: 1 },
            retention: RetentionPolicy {
                max_full_snapshots: 1,
                max_deltas_per_full: 2,
                full_snapshot_every_steps: 2,
                max_total_snapshots: Some(3),
            },
            lineage: LineageConfig {
                rollback_interval_steps: 4,
                max_rollback_snapshots: 1,
            },
            delta_max_ratio: 0.95,
            ..Default::default()
        };
        let coord = Coordinator::new(storage, config).unwrap();

        for step in 0..12u64 {
            coord.save(step, evolving_state(step), HashMap::new()).unwrap();
        }
        coord.flush().unwrap();

        let steps: Vec<u64> = coord.list_snapshots().iter().map(|s| s.step).collect();
        let on_interval: Vec<u64> = steps.iter().copied().filter(|s| s % 4 == 0).collect();

        assert!(
            on_interval.len() <= 1,
            "a cap of one left {} protected snapshots: {steps:?}",
            on_interval.len()
        );
        assert!(
            !steps.contains(&0),
            "step 0 outlived a cap of one; steps left: {steps:?}"
        );
    }

    /// Tied weights, and the order they arrive in changes between saves.
    ///
    /// Deduplication keeps the first occurrence and records the second as an
    /// `Alias` of it, so the pair `{emb, head}` sharing one buffer stores
    /// whichever came first and points the other at it. Swap the order on the
    /// next save — which costs nothing more than wrapping the model
    /// differently — and the tensor that used to be the alias is now the one
    /// carrying the bytes. It is unchanged since the base, so it is written
    /// `Skipped`, and a skipped tensor resolves through the base by name:
    /// straight onto the base's `Alias` entry, which holds no bytes.
    #[test]
    fn tied_weights_survive_a_change_in_tensor_order() {
        let dir = tempfile::tempdir().unwrap();
        let coord = make_coordinator(dir.path());

        let shared = vec![7u8; 8192];
        let tied = |first: &str, second: &str| {
            vec![
                TensorData {
                    name: first.into(),
                    shape: vec![8192],
                    dtype: "uint8".into(),
                    data: shared.clone(),
                },
                TensorData {
                    name: second.into(),
                    shape: vec![8192],
                    dtype: "uint8".into(),
                    data: shared.clone(),
                },
            ]
        };

        coord.save(100, tied("emb", "head"), HashMap::new()).unwrap();
        let second = coord.save(200, tied("head", "emb"), HashMap::new()).unwrap();
        coord.flush().unwrap();

        let loaded = coord.load(second).unwrap_or_else(|e| {
            panic!("the tied pair became unreadable when the order changed: {e}")
        });
        assert_eq!(loaded["emb"], shared);
        assert_eq!(loaded["head"], shared);
    }

    /// A checkpoint where nothing changed is still a checkpoint.
    ///
    /// With every tensor skipped there are no bytes to write, so the save path
    /// writes no pack — and the pack is what carries the descriptor that
    /// startup recovery rebuilds a lost snapshot from. Kill the process
    /// between the pack and the manifest and this one cannot come back: it is
    /// not merely unrecovered, `recover_or_reclaim_orphans` has nothing to
    /// judge it by. Fine-tuning, where whole stretches of the model do not
    /// move, is where this is the common case rather than the odd one.
    #[test]
    fn a_checkpoint_that_changed_nothing_can_still_be_recovered() {
        let dir = tempfile::tempdir().unwrap();
        let unchanged = sample_tensors(3);

        let coord = make_coordinator(dir.path());
        coord.save(100, unchanged.clone(), HashMap::new()).unwrap();
        let quiet = coord.save(200, unchanged.clone(), HashMap::new()).unwrap();
        coord.flush().unwrap();
        drop(coord);

        // The crash: the pack landed, the manifest naming it did not.
        let manifest_path = dir.path().join("manifest.json");
        let raw = std::fs::read(&manifest_path).unwrap();
        // Same trailing-zero trim the loader does in `Coordinator::new`.
        let end = raw.iter().rposition(|&b| b != 0).map(|i| i + 1).unwrap_or(0);
        let mut manifest: Manifest = serde_json::from_slice(&raw[..end]).unwrap();
        assert_eq!(manifest.snapshots.len(), 2, "both saves were recorded");
        manifest.snapshots.retain(|s| s.id != quiet);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let reopened = make_coordinator(dir.path());
        let steps: Vec<u64> = reopened.list_snapshots().iter().map(|s| s.step).collect();
        assert!(
            steps.contains(&200),
            "the checkpoint that changed nothing was not recovered; steps: {steps:?}"
        );

        let loaded = reopened.load(quiet).unwrap();
        assert_eq!(loaded["w"], vec![3u8; 8192]);
    }

    /// The merger deletes the base a save is reading.
    ///
    /// `save_sync_in_pool` takes its base from the manifest and then *drops
    /// the lock* before `save_rank_tensors` reads that base off storage. The
    /// merger holds the same lock only while it decides what to fold, so the
    /// two windows overlap: the merge can publish and delete the base
    /// snapshot's packs while a save is in the middle of diffing against
    /// them. It shows up two ways — the loud one is the save failing with
    /// `Checkpoint not found: snapshots/…/rank_0.pack`, the quiet one is a
    /// delta reaching the manifest with a `base_snapshot_id` that no longer
    /// names anything, which breaks `load_latest` now and is discarded as an
    /// orphan at the next startup.
    ///
    /// The interleaving is forced rather than raced for. A sleep here proves
    /// nothing: on 8 KB tensors both sides finish before either is preempted,
    /// and the test would pass on a version that still has the bug. So the
    /// storage backend itself runs the merge at the exact instant the save
    /// reaches for the base — the one ordering the lock discipline has to
    /// survive, and the one a real run hits when the base is gigabytes.
    #[test]
    fn a_merge_does_not_delete_the_base_a_save_is_reading() {
        /// Runs `armed` once, on the first pack read that follows arming it.
        #[derive(Default)]
        struct Interleave {
            armed: Mutex<Option<Box<dyn Fn() + Send>>>,
            fired: std::sync::atomic::AtomicBool,
        }

        impl Interleave {
            fn maybe_fire(&self) {
                if self.fired.load(std::sync::atomic::Ordering::SeqCst) {
                    return;
                }
                let hook = self.armed.lock().unwrap();
                if let Some(run) = hook.as_ref() {
                    self.fired
                        .store(true, std::sync::atomic::Ordering::SeqCst);
                    run();
                }
            }
        }

        struct MergeAtBaseRead {
            inner: Arc<LocalStorage>,
            interleave: Arc<Interleave>,
        }

        impl StorageBackend for MergeAtBaseRead {
            fn put(&self, p: &str, data: &[u8]) -> Result<()> {
                self.inner.put(p, data)
            }
            fn put_parts(&self, p: &str, parts: &[&[u8]]) -> Result<()> {
                self.inner.put_parts(p, parts)
            }
            fn get(&self, p: &str) -> Result<Vec<u8>> {
                if p.ends_with(".pack") {
                    self.interleave.maybe_fire();
                }
                self.inner.get(p)
            }
            fn get_range(&self, p: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
                if p.ends_with(".pack") {
                    self.interleave.maybe_fire();
                }
                self.inner.get_range(p, offset, len)
            }
            fn exists(&self, p: &str) -> Result<bool> {
                self.inner.exists(p)
            }
            fn delete(&self, p: &str) -> Result<()> {
                self.inner.delete(p)
            }
            fn list(&self, prefix: &str) -> Result<Vec<String>> {
                self.inner.list(prefix)
            }
            fn remove_dir(&self, p: &str) -> Result<()> {
                self.inner.remove_dir(p)
            }
        }

        let dir = tempfile::tempdir().unwrap();
        let plain = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let interleave = Arc::new(Interleave::default());
        let storage: Arc<dyn StorageBackend> = Arc::new(MergeAtBaseRead {
            inner: Arc::clone(&plain),
            interleave: Arc::clone(&interleave),
        });

        let compression = CompressionAlgo::Zstd { level: 1 };
        let config = CoordinatorConfig {
            world_size: 1,
            rank: 0,
            compression: compression.clone(),
            retention: RetentionPolicy {
                max_full_snapshots: 5,
                max_deltas_per_full: 10,
                full_snapshot_every_steps: 10000,
                max_total_snapshots: None,
            },
            delta_max_ratio: 0.95,
            // Synchronous, and with the base off storage rather than out of
            // memory: the race is between a save reading the base and a merge
            // deleting it, and neither happens when the bytes never leave the
            // process.
            async_save: false,
            keep_base_in_memory: false,
            ..Default::default()
        };
        let coord = Coordinator::new(storage, config).unwrap();

        // A base and two deltas, with nothing interleaved yet.
        for step in 0..3u64 {
            coord.save(step, evolving_state(step), HashMap::new()).unwrap();
        }

        // From here, the next save's reach for the base runs the merge first.
        let merge_storage: Arc<dyn StorageBackend> = Arc::clone(&plain) as Arc<dyn StorageBackend>;
        let merge_manifest = Arc::clone(&coord.core.manifest);
        let merge_compression = compression.clone();
        // The coordinator's own registry, not a fresh one: what is being
        // tested is that the save's pin is visible to the merge.
        let merge_in_flight = Arc::clone(&coord.core.in_flight);
        *interleave.armed.lock().unwrap() = Some(Box::new(move || {
            let _ = crate::merger::do_full_merge(
                &merge_storage,
                &merge_manifest,
                &merge_compression,
                &merge_in_flight,
                &Arc::new(crate::remote_sync::PendingDeletes::default()),
            );
        }));

        coord
            .save(3, evolving_state(3), HashMap::new())
            .unwrap_or_else(|e| panic!("the save lost its base to a merge running beside it: {e}"));
        coord.flush().unwrap();

        assert!(
            interleave.fired.load(std::sync::atomic::Ordering::SeqCst),
            "the merge never ran: this probe proved nothing"
        );

        // The quiet variant: a delta whose base the merge already deleted.
        let manifest = coord.core.manifest.lock().unwrap();
        let ids: Vec<Uuid> = manifest.snapshots.iter().map(|s| s.id).collect();
        let dangling: Vec<(u64, Uuid)> = manifest
            .snapshots
            .iter()
            .filter_map(|s| s.base_snapshot_id.map(|b| (s.step, b)))
            .filter(|(_, base)| !ids.contains(base))
            .collect();
        assert!(
            dangling.is_empty(),
            "deltas left pointing at a base the merge removed: {dangling:?}"
        );
        drop(manifest);

        let (_, loaded) = coord
            .load_latest()
            .unwrap_or_else(|e| panic!("the newest checkpoint is unreadable after merging: {e}"));
        assert_eq!(loaded["w"][0], 3, "the newest checkpoint is not step 3");
    }

    /// Deterministic float32 bytes that actually differ between steps.
    ///
    /// Constant runs compress to nothing and delta against anything, so a
    /// test built on them proves nothing about the delta path.
    fn fp32_noise(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        (0..n)
            .flat_map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                // Keep the exponent sane so the bf16 cast is a real cast and
                // not a parade of NaNs.
                let v = ((s >> 40) as f32 / 1.0e5) - 5.0;
                v.to_le_bytes()
            })
            .collect()
    }

    fn cast_coordinator(dir: &std::path::Path, policy: DTypePolicy) -> Coordinator {
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir).unwrap());
        let config = CoordinatorConfig {
            compression: CompressionAlgo::Zstd { level: 1 },
            async_save: false,
            keep_base_in_memory: true,
            save_dtype: policy,
            ..Default::default()
        };
        Coordinator::new(storage, config).unwrap()
    }

    fn two_component_state(seed: u64) -> Vec<TensorData> {
        vec![
            TensorData {
                name: "ravex/models/layer.weight".into(),
                shape: vec![4096],
                dtype: "float32".into(),
                data: fp32_noise(4096, seed),
            },
            TensorData {
                name: "ravex/optimizers/exp_avg".into(),
                shape: vec![4096],
                dtype: "float32".into(),
                data: fp32_noise(4096, seed + 977),
            },
        ]
    }

    /// The failure this feature could have introduced, and the reason
    /// `retain_base` is decided per tensor.
    ///
    /// With one component cast and the other not, the in-memory base is
    /// partial: it holds the weights, whose stored bytes are the incoming
    /// bytes, and not the moments, whose stored bytes are bf16. If the
    /// retention were still all-or-nothing in either direction, the next
    /// delta would be XOR-ed against the wrong base for one of the two — and
    /// every tensor would still pass its own hash check on the way back,
    /// because the hash is of what was written, not of what should have
    /// been. So the assertion has to be on the reconstructed values.
    #[test]
    fn a_per_component_cast_keeps_the_next_delta_correct() {
        let dir = tempfile::tempdir().unwrap();
        let coord = cast_coordinator(
            dir.path(),
            DTypePolicy::from_rules([("ravex/optimizers/*", "bf16")]).unwrap(),
        );

        coord.save(1, two_component_state(1), HashMap::new()).unwrap();

        // A second step, close to the first: this is what produces a delta
        // rather than a full, which is the case under test.
        let mut next = two_component_state(1);
        next[0].data[..64].copy_from_slice(&fp32_noise(16, 31));
        next[1].data[..64].copy_from_slice(&fp32_noise(16, 32));
        let snap = coord.save(2, next.clone(), HashMap::new()).unwrap();

        let loaded = coord.load(snap).unwrap();

        // The weights were not cast, so they come back byte for byte.
        assert_eq!(
            loaded["ravex/models/layer.weight"], next[0].data,
            "uncast weights must survive the delta exactly"
        );

        // The moments were cast to bf16 and back, so they come back rounded,
        // not equal. What must hold is that they are the *right* numbers:
        // bf16 keeps 8 significant bits, so a relative error over 1% means
        // the delta was applied against the wrong base, not that it rounded.
        let got = &loaded["ravex/optimizers/exp_avg"];
        assert_eq!(got.len(), next[1].data.len());
        for (i, (g, w)) in got
            .chunks_exact(4)
            .zip(next[1].data.chunks_exact(4))
            .enumerate()
        {
            let g = f32::from_le_bytes(g.try_into().unwrap());
            let w = f32::from_le_bytes(w.try_into().unwrap());
            let tol = w.abs() * 0.01 + 1e-6;
            assert!(
                (g - w).abs() <= tol,
                "element {i}: got {g}, wanted {w} within {tol} — a bf16 round \
                 trip cannot be this far off, so the base was wrong"
            );
        }
    }

    /// Under the old rule any cast turned the in-memory base off for the
    /// whole snapshot. The uncast third of the state has no reason to lose it.
    #[test]
    fn an_uncast_tensor_is_still_retained_when_another_is_cast() {
        let dir = tempfile::tempdir().unwrap();
        let coord = cast_coordinator(
            dir.path(),
            DTypePolicy::from_rules([("ravex/optimizers/*", "bf16")]).unwrap(),
        );
        coord.save(1, two_component_state(5), HashMap::new()).unwrap();

        let retained = coord.core.retained_base.lock().unwrap();
        let retained = retained.as_ref().expect("a full save must retain a base");
        assert!(
            retained.tensors.contains_key("ravex/models/layer.weight"),
            "the uncast component must stay in memory"
        );
        assert!(
            !retained.tensors.contains_key("ravex/optimizers/exp_avg"),
            "the cast component must not: what is on disk is bf16, and these \
             bytes are fp32"
        );
    }

    /// A uniform cast still has to reconstruct, and this is the case that
    /// used to be covered by switching retention off entirely.
    #[test]
    fn a_uniform_cast_still_round_trips_across_a_delta() {
        let dir = tempfile::tempdir().unwrap();
        let coord = cast_coordinator(dir.path(), DTypePolicy::parse("bf16").unwrap());

        coord.save(1, two_component_state(9), HashMap::new()).unwrap();
        let mut next = two_component_state(9);
        next[0].data[..64].copy_from_slice(&fp32_noise(16, 77));
        let snap = coord.save(2, next.clone(), HashMap::new()).unwrap();

        let loaded = coord.load(snap).unwrap();
        for t in &next {
            let got = &loaded[&t.name];
            assert_eq!(got.len(), t.data.len(), "{}", t.name);
            for (g, w) in got.chunks_exact(4).zip(t.data.chunks_exact(4)) {
                let g = f32::from_le_bytes(g.try_into().unwrap());
                let w = f32::from_le_bytes(w.try_into().unwrap());
                assert!((g - w).abs() <= w.abs() * 0.01 + 1e-6, "{}", t.name);
            }
        }
    }

    /// Integer and bool state travels through a float cast untouched — this
    /// is the half of the dtype question that is about transport, not
    /// conversion. `save_dtype` names a target for floats; a step counter and
    /// a causal mask are not floats and have no business being reinterpreted
    /// because one was set.
    #[test]
    fn non_float_state_is_untouched_by_any_save_dtype() {
        let dir = tempfile::tempdir().unwrap();
        let coord = cast_coordinator(dir.path(), DTypePolicy::parse("bf16").unwrap());

        let step: Vec<u8> = (0..8192u32).flat_map(|i| (i as i64).to_le_bytes()).collect();
        let mask: Vec<u8> = (0..8192).map(|i| (i % 2) as u8).collect();
        let tensors = vec![
            TensorData {
                name: "ravex/optimizers/step".into(),
                shape: vec![8192],
                dtype: "int64".into(),
                data: step.clone(),
            },
            TensorData {
                name: "ravex/models/causal_mask".into(),
                shape: vec![8192],
                dtype: "bool".into(),
                data: mask.clone(),
            },
        ];
        let snap = coord.save(1, tensors, HashMap::new()).unwrap();
        let loaded = coord.load(snap).unwrap();
        assert_eq!(loaded["ravex/optimizers/step"], step);
        assert_eq!(loaded["ravex/models/causal_mask"], mask);
    }

    /// bfloat16 bytes: the top half of a float32, which is what the cast
    /// does. Values stay well inside float16 range on purpose — see the test
    /// that uses them.
    fn bf16_noise(n: usize, seed: u64) -> Vec<u8> {
        fp32_noise(n, seed)
            .chunks_exact(4)
            .flat_map(|c| [c[2], c[3]])
            .collect()
    }

    /// How every tensor of a snapshot was stored, by name.
    fn storage_of(coord: &Coordinator, snap: Uuid) -> HashMap<String, TensorStorage> {
        let manifest = coord.core.manifest.lock().unwrap();
        manifest
            .find_snapshot(snap)
            .expect("snapshot must be in the manifest")
            .ranks[&0]
            .tensors
            .iter()
            .map(|t| (t.name.clone(), t.storage.clone()))
            .collect()
    }

    /// The corruption a per-tensor `retain_base` has to avoid, in the one
    /// shape where nothing else would catch it: a cast that does not change
    /// the width.
    ///
    /// A tensor arriving as bfloat16 and stored as float16 is the same number
    /// of bytes before and after, so the length check in
    /// [`crate::delta::compute_delta`] — which quietly saves the fp32→bf16
    /// and fp32→fp8 cases by falling back to a full save — does not fire.
    /// Retaining the incoming bytes would XOR the next delta against bfloat16
    /// while the base on disk is float16, and the two are different bytes for
    /// the same value.
    ///
    /// bf16 → fp16 → bf16 is bit-exact for these values: float16 has ten
    /// mantissa bits to bfloat16's seven and the exponents are in range, so
    /// the round trip is lossless and the assertion can be on equality rather
    /// than on a tolerance. That matters — a tolerance is what lets a wrong
    /// base slip through as rounding.
    #[test]
    fn an_equal_width_cast_does_not_poison_the_retained_base() {
        let dir = tempfile::tempdir().unwrap();
        let coord = cast_coordinator(
            dir.path(),
            DTypePolicy::from_rules([("ravex/optimizers/*", "fp16")]).unwrap(),
        );

        let state = |seed: u64| {
            vec![
                TensorData {
                    name: "ravex/models/layer.weight".into(),
                    shape: vec![8192],
                    dtype: "float32".into(),
                    data: fp32_noise(8192, seed),
                },
                TensorData {
                    name: "ravex/optimizers/exp_avg".into(),
                    shape: vec![8192],
                    dtype: "bfloat16".into(),
                    data: bf16_noise(8192, seed + 977),
                },
            ]
        };

        coord.save(1, state(3), HashMap::new()).unwrap();

        let mut next = state(3);
        next[0].data[..64].copy_from_slice(&fp32_noise(16, 41));
        next[1].data[..64].copy_from_slice(&bf16_noise(32, 42));
        let snap = coord.save(2, next.clone(), HashMap::new()).unwrap();

        // Without this the test is vacuous: a full save reconstructs
        // correctly no matter what the retained base held.
        let stored = storage_of(&coord, snap);
        assert_eq!(
            stored["ravex/optimizers/exp_avg"],
            TensorStorage::DeltaXor,
            "the cast component must be stored as a delta for this test to \
             be about deltas at all"
        );

        let loaded = coord.load(snap).unwrap();
        assert_eq!(loaded["ravex/models/layer.weight"], next[0].data);
        assert_eq!(
            loaded["ravex/optimizers/exp_avg"], next[1].data,
            "bf16 → fp16 → bf16 is lossless here, so anything but equality \
             means the delta was applied to the wrong base"
        );
    }
}
