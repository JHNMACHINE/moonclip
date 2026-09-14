use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
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


mod recovery;
mod saver;

use recovery::{load_manifest, recover_or_reclaim_orphans, snapshot_is_whole};
use saver::{AsyncSaver, SaveJob};

#[cfg(test)]
mod tests;

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
    /// Set when [`Core::lock_manifest`] found the lock poisoned, meaning a
    /// panic unwound out of a critical section and the in-memory manifest may
    /// be half-updated. Cleared by the next successful re-read from storage.
    /// See [`Core::heal_manifest`].
    manifest_stale: AtomicBool,
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
            manifest_stale: AtomicBool::new(false),
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
    /// its error, if any — and report a background thread that has died.
    ///
    /// The saver's error is taken once and consumed; a dead merger or syncer
    /// is reported **every time**, because it is not an event that happened
    /// but a state that will not improve. Nothing restarts those threads, so a
    /// one-shot report would be one message a caller could miss on its way to
    /// discovering, at resume time, that half a run's deltas were never folded
    /// or never left the machine.
    ///
    /// The saved data is still on local disk in both cases, and still
    /// readable. What has stopped is the consolidating and the uploading.
    pub fn flush(&self) -> Result<()> {
        if let Some(ref s) = self.saver {
            s.flush()?;
        }
        self.core.check_background_threads()
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

    /// What a snapshot holds, without reading a byte of it.
    ///
    /// Name, shape, dtype and how each tensor is stored, for this rank, read
    /// off the manifest already in memory. No storage access at all — see
    /// [`Core::describe`] for why that is worth its own call.
    pub fn describe(&self, snap_id: Uuid) -> Result<SnapshotDescription> {
        self.wait_idle();
        self.core.describe(snap_id)
    }

    /// The newest finalized snapshot's description.
    pub fn describe_latest(&self) -> Result<SnapshotDescription> {
        self.wait_idle();
        self.core.describe_latest()
    }

    /// Load the named tensors and nothing else.
    ///
    /// [`Self::load`] reads the rank's whole pack in one go and decompresses
    /// every tensor in it, which is right when every tensor is wanted. This
    /// reads only the spans the named tensors occupy. See
    /// [`Core::load_tensors`].
    pub fn load_tensors(
        &self,
        snap_id: Uuid,
        names: &[String],
    ) -> Result<HashMap<String, Vec<u8>>> {
        self.flush()?;
        self.core.load_tensors(snap_id, names)
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
    /// Report a background thread that is gone while it was still meant to be
    /// running. See [`Coordinator::flush`].
    ///
    /// Both are named in one error when both are gone, so a caller reading a
    /// log does not fix one and rediscover the other.
    fn check_background_threads(&self) -> Result<()> {
        let mut gone: Vec<&str> = Vec::new();
        if self.merger.as_ref().is_some_and(|m| m.is_dead()) {
            gone.push("merger (delta chains are no longer being consolidated)");
        }
        if self.syncer.as_ref().is_some_and(|s| s.is_dead()) {
            gone.push("remote syncer (nothing is reaching the remote store)");
        }
        if gone.is_empty() {
            return Ok(());
        }
        Err(MoonclipError::Storage(format!(
            "Background thread gone: {}. Checkpoints are still being written \
             locally and are still readable.",
            gone.join("; ")
        )))
    }

    /// Take the manifest lock, surviving a panic that poisoned it.
    ///
    /// Every access to `self.manifest` goes through here rather than
    /// `.lock().unwrap()`. See [`crate::manifest::lock_manifest`] for why
    /// recovering beats walling the coordinator off; what this adds on top is
    /// remembering that it happened, so the in-memory copy can be replaced
    /// with the one on disk before anybody writes it back out.
    fn lock_manifest(&self) -> std::sync::MutexGuard<'_, Manifest> {
        if self.manifest.is_poisoned() {
            self.manifest_stale.store(true, Ordering::Relaxed);
        }
        crate::manifest::lock_manifest(&self.manifest)
    }

    /// Replace a possibly half-updated in-memory manifest with the one on
    /// storage. A no-op unless a panic actually poisoned the lock.
    ///
    /// Called at the top of the two write paths that do **not** already
    /// re-read — `save_sync_in_pool` and `create_snapshot`. The multi-rank
    /// paths (`save_rank_in_pool`, `finalize_snapshot`) open with a
    /// `reload_manifest` of their own for a different reason and get this for
    /// free, which is why `reload_manifest` is where the flag is cleared.
    ///
    /// The read paths deliberately skip it. A stale manifest costs a reader a
    /// `NotFound` or a snapshot it cannot see yet, both of which are ordinary
    /// errors it already handles; a writer persisting one costs the entries
    /// that went missing. Only the second is worth an extra round trip to
    /// storage on a path that has none.
    ///
    /// On a failed re-read the flag goes back up, so the next writer retries
    /// rather than quietly proceeding on the copy this was called to distrust.
    fn heal_manifest(&self) -> Result<()> {
        // Both questions, and the first is the one that catches the common
        // case: nothing has taken the lock since the panic, so the poison flag
        // is still standing and `lock_manifest` has had no chance to notice.
        // Asking only `manifest_stale` here re-read one save too late — the
        // save that should have been healed took the lock, raised the flag,
        // and persisted the copy it had.
        //
        // The flag still earns its place for the other order: a `load` or a
        // `list_snapshots` between the panic and the next write clears the
        // poison as it recovers, and by the time a writer arrives
        // `is_poisoned` is false while the copy in memory is no less suspect.
        if self.manifest.is_poisoned() {
            self.manifest_stale.store(true, Ordering::Relaxed);
        }
        if !self.manifest_stale.load(Ordering::Relaxed) {
            return Ok(());
        }
        self.reload_manifest().inspect_err(|_| {
            self.manifest_stale.store(true, Ordering::Relaxed);
        })
    }

    /// Full single-rank save pipeline (runs on the caller thread or the
    /// background save thread).
    ///
    /// The pipeline is a handful of parallel passes over several gigabytes, and
    /// they run in Moonclip's own rayon pool rather than the global one — see
    /// [`crate::pool`]. Installing it here rather than at the callers covers
    /// both entry points, this being the one place both go through.
    pub(super) fn save_sync(
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
        // Before anything reads the manifest: if a previous panic poisoned the
        // lock, this save would otherwise diff against a half-updated copy and
        // then persist it. See `heal_manifest`.
        self.heal_manifest()?;

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
        let manifest = self.lock_manifest();
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

        let mut manifest = self.lock_manifest();
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
        // See `save_sync_in_pool`: the other write path that does not already
        // open with a re-read.
        self.heal_manifest()?;

        let snap_id = Uuid::new_v4();

        // A base a merge has already claimed is not picked up, exactly as in
        // the single-rank path: it will not exist by the time anyone loads the
        // delta written against it. Rank 0 chooses for every rank here, so
        // getting it wrong costs the whole snapshot rather than one shard.
        let manifest = self.lock_manifest();
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

        let mut manifest = self.lock_manifest();
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

        let manifest = self.lock_manifest();
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

        let mut manifest = self.lock_manifest();
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
        let manifest = self.lock_manifest();
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
            tensor::OwnedPack,
        );
        let needs_base = rank_entry
            .tensors
            .iter()
            .any(|t| t.storage != TensorStorage::Full);
        let base_ctx: Option<BaseCtx> = match (needs_base, snap.base_snapshot_id) {
            (true, Some(base_id)) => {
                let manifest = self.lock_manifest();
                let base_snap = manifest.find_snapshot(base_id).cloned();
                drop(manifest);
                match base_snap {
                    Some(bs) => match bs.ranks.get(&self.config.rank) {
                        Some(re) => {
                            let pack = match re.pack_file {
                                Some(ref p) => {
                                    tensor::OwnedPack::Whole(Arc::new(self.storage.get(p)?))
                                }
                                None => tensor::OwnedPack::PerFile,
                            };
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
            self.find_base_tensor_entry_with_pack(base_id, tensor_name, true)
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
                    match pack_data {
                        Some(ref bytes) => tensor::PackSource::Whole(bytes),
                        None => tensor::PackSource::PerFile,
                    },
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
        let manifest = self.lock_manifest();
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

    /// What a snapshot holds, with none of its bytes.
    ///
    /// Everything here is already in the manifest — shape, dtype, how the
    /// tensor is stored, how many bytes it takes — so this touches no storage
    /// and costs a clone. It exists because there was no way to ask: `load`
    /// materialized the whole snapshot, and a caller that only wanted to know
    /// *how long* a shard was had to read the shard to find out.
    ///
    /// Resharding is the caller that pays for that. Planning how N old shards
    /// map onto M new ones needs every old length before it can cut the first
    /// slice, so a reshard read every old checkpoint in full, measured it, and
    /// threw it away — N complete reads of data it did not want.
    ///
    /// This rank's tensors, the same set [`Core::load`] returns. A snapshot
    /// this rank did not write is a `NotFound` rather than an empty list: the
    /// two mean different things and only one of them is a checkpoint.
    fn describe(&self, snap_id: Uuid) -> Result<SnapshotDescription> {
        let manifest = self.lock_manifest();
        let snap = manifest
            .find_snapshot(snap_id)
            .ok_or_else(|| MoonclipError::NotFound(format!("Snapshot {snap_id}")))?;
        Self::describe_snapshot(snap, self.config.rank)
    }

    fn describe_latest(&self) -> Result<SnapshotDescription> {
        let manifest = self.lock_manifest();
        let snap = manifest
            .snapshots
            .iter()
            .rev()
            .find(|s| s.finalized)
            .ok_or_else(|| MoonclipError::NotFound("No finalized snapshots".into()))?;
        Self::describe_snapshot(snap, self.config.rank)
    }

    fn describe_snapshot(snap: &Snapshot, rank: u32) -> Result<SnapshotDescription> {
        let rank_entry = snap.ranks.get(&rank).ok_or_else(|| {
            MoonclipError::NotFound(format!("Rank {rank} not found in snapshot {}", snap.id))
        })?;

        let tensors = rank_entry
            .tensors
            .iter()
            .map(|t| TensorDescription {
                name: t.name.clone(),
                shape: t.shape.clone(),
                // What a load hands back, which is the question a caller
                // sizing a buffer is asking. `stored_dtype` is the other one —
                // what is on disk — and the two differ exactly when
                // `save_dtype` cast the tensor on the way down.
                dtype: t.original_dtype.clone().unwrap_or_else(|| t.dtype.clone()),
                stored_dtype: t.dtype.clone(),
                storage: t.storage.clone(),
                alias_of: t.alias_of.clone(),
                raw_size: t.raw_size,
                compressed_size: t.compressed_size,
                hash_raw: t.hash_raw.clone(),
            })
            .collect();

        Ok(SnapshotDescription {
            id: snap.id,
            step: snap.step,
            created_at: snap.created_at,
            is_delta: snap.base_snapshot_id.is_some(),
            base_snapshot_id: snap.base_snapshot_id,
            metadata: snap.metadata.clone(),
            rank,
            ranks: snap.ranks.len() as u32,
            tensors,
        })
    }

    /// Load the named tensors and nothing else.
    ///
    /// [`Core::load`] reads the rank's pack with a single `get` and slices
    /// every tensor out of it, which is the right shape when every tensor is
    /// wanted. Asking for a handful that way costs the whole checkpoint — over
    /// a network, the whole checkpoint over the wire — so this reads each
    /// wanted tensor's span on its own instead, and resolves deltas and skips
    /// against a base it also refuses to read whole.
    ///
    /// Sequential rather than parallel, deliberately: the callers are asking
    /// for a few small tensors, and a rayon pass over three of them buys
    /// nothing worth the coordination.
    ///
    /// A name that is not in the snapshot is an error naming it, not a gap in
    /// the returned map. Answering a typo with silence is how a caller ends up
    /// reasoning about a checkpoint it never read.
    fn load_tensors(&self, snap_id: Uuid, names: &[String]) -> Result<HashMap<String, Vec<u8>>> {
        let manifest = self.lock_manifest();
        let snap = manifest
            .find_snapshot(snap_id)
            .ok_or_else(|| MoonclipError::NotFound(format!("Snapshot {snap_id}")))?
            .clone();
        // Pinned for the same reason a full load pins: a merge running beside
        // this would otherwise unlink the packs being read. The base too,
        // since a skipped tensor is read from it. See `crate::inflight`.
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

        let pack = match rank_entry.pack_file {
            Some(ref path) => tensor::PackSource::Ranged(path),
            None => tensor::PackSource::PerFile,
        };
        // `false`: one tensor's worth of base, not the base checkpoint. That
        // is the whole point of this call, and it is the path a skipped tensor
        // takes — which for something small and unchanging, a state template
        // among them, is nearly every step.
        let resolver = |base_id: Uuid, tensor_name: &str| -> Result<crate::tensor::BaseEntry> {
            self.find_base_tensor_entry_with_pack(base_id, tensor_name, false)
        };

        let mut out = HashMap::with_capacity(names.len());
        for name in names {
            let entry = rank_entry
                .tensors
                .iter()
                .find(|t| &t.name == name)
                .ok_or_else(|| {
                    MoonclipError::NotFound(format!("Tensor '{name}' not in snapshot {snap_id}"))
                })?;

            // An alias holds no bytes of its own; they are under another name
            // in this same rank entry. A full load resolves these in a second
            // pass over everything it read, which is not available here — the
            // target may well not be among the names asked for.
            let entry = if entry.storage == TensorStorage::Alias {
                let target = entry.alias_of.as_deref().ok_or_else(|| {
                    MoonclipError::NotFound(format!("Alias '{name}' has no target"))
                })?;
                rank_entry
                    .tensors
                    .iter()
                    .find(|t| t.name == target)
                    .ok_or_else(|| {
                        MoonclipError::NotFound(format!(
                            "Alias '{name}' points at '{target}', which is not in this snapshot"
                        ))
                    })?
            } else {
                entry
            };

            let data = tensor::load_tensor(
                entry,
                snap.base_snapshot_id,
                self.storage.as_ref(),
                &snap.compression,
                pack,
                &resolver,
            )?;
            out.insert(name.clone(), data);
        }

        Ok(out)
    }

    fn list_snapshots(&self) -> Vec<SnapshotInfo> {
        let manifest = self.lock_manifest();
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
        *self.lock_manifest() = fresh;
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

    /// Find base tensor entry + where its bytes are (fallback path for base
    /// snapshots not covered by the per-load prefetch).
    ///
    /// `whole_pack` says whether to read the base pack now or leave each
    /// tensor to a ranged read. Reading it whole is right when the caller is
    /// resolving many tensors against this base, and wrong when it wants one:
    /// `load_tensors` asks for a handful and would otherwise pull the entire
    /// base checkpoint to reconstruct a few kilobytes of it.
    fn find_base_tensor_entry_with_pack(
        &self,
        base_id: Uuid,
        tensor_name: &str,
        whole_pack: bool,
    ) -> Result<crate::tensor::BaseEntry> {
        let manifest = self.lock_manifest();
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

        let pack = match pack_file {
            Some(ref path) if whole_pack => {
                tensor::OwnedPack::Whole(Arc::new(self.storage.get(path)?))
            }
            Some(path) => tensor::OwnedPack::Ranged(path.into()),
            None => tensor::OwnedPack::PerFile,
        };

        Ok((entry, compression, pack))
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
                let mut m = self.lock_manifest();
                *m = new_manifest;
                // Whatever a panic may have left half-written in memory is
                // gone: this is the file, read whole. See `heal_manifest`.
                self.manifest_stale.store(false, Ordering::Relaxed);
                Ok(())
            }
            // Nothing on storage to re-read, so the copy in memory is all
            // there is and distrusting it further buys nothing.
            Err(MoonclipError::NotFound(_)) => {
                self.manifest_stale.store(false, Ordering::Relaxed);
                Ok(())
            }
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

/// One tensor as the manifest describes it. See [`SnapshotDescription`].
#[derive(Debug, Clone)]
pub struct TensorDescription {
    pub name: String,
    pub shape: Vec<usize>,
    /// The dtype a load hands back: the tensor's original one when
    /// `save_dtype` cast it on the way down, otherwise what is on disk.
    pub dtype: String,
    /// The dtype actually stored. Differs from `dtype` only under a cast.
    pub stored_dtype: String,
    pub storage: TensorStorage,
    /// For an `Alias`, the tensor in this same snapshot holding the bytes.
    pub alias_of: Option<String>,
    pub raw_size: u64,
    /// Bytes on disk. Zero for `Skipped` and `Alias`, which store none — the
    /// honest answer to "what does this snapshot cost", and not the same
    /// question as "how big is this tensor", which is `raw_size`.
    pub compressed_size: u64,
    /// xxHash3-128 of the tensor's raw bytes, as the manifest records it: the
    /// value skip detection compares. The same whether the tensor was stored
    /// whole, as a delta, skipped or aliased, because it hashes what the
    /// tensor holds rather than how it was written. Not cryptographic.
    ///
    /// Exposed so a caller can identify a snapshot's content without reading
    /// `manifest.json` itself, which on Windows is not a safe thing to do while
    /// the writer may be renaming a new manifest over it.
    pub hash_raw: String,
}

/// What a snapshot holds, without any of its bytes. See [`Coordinator::describe`].
#[derive(Debug, Clone)]
pub struct SnapshotDescription {
    pub id: Uuid,
    pub step: u64,
    pub created_at: DateTime<Utc>,
    pub is_delta: bool,
    pub base_snapshot_id: Option<Uuid>,
    pub metadata: HashMap<String, String>,
    /// The rank these tensors belong to — this coordinator's own.
    pub rank: u32,
    /// How many ranks wrote into this snapshot.
    pub ranks: u32,
    pub tensors: Vec<TensorDescription>,
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

