use std::collections::HashMap;
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::cast::DType;
use crate::error::{Result, MoonclipError};
use crate::manifest::*;
use crate::merger::{DeltaMerger, MergerConfig};
use crate::pack;
use crate::remote_sync::{RemoteSyncConfig, RemoteSyncer};
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
    /// Target dtype for saving float tensors. DType::None = keep original.
    pub save_dtype: DType,
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
    /// Ignored when `save_dtype` casts: a later delta is computed against the
    /// post-cast bytes on disk, which are not what arrives here.
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
            save_dtype: DType::None,
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
pub(crate) struct Core {
    pub(crate) storage: Arc<dyn StorageBackend>,
    pub(crate) manifest: Arc<Mutex<Manifest>>,
    config: CoordinatorConfig,
    merger: Option<DeltaMerger>,
    syncer: Option<RemoteSyncer>,
    /// See [`CoordinatorConfig::keep_base_in_memory`].
    retained_base: Mutex<Option<RetainedBase>>,
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
        // Load or create manifest
        let manifest = match storage.get("manifest.json") {
            Ok(data) => {
                let end = data
                    .iter()
                    .rposition(|&b| b != 0)
                    .map(|i| i + 1)
                    .unwrap_or(0);
                serde_json::from_slice(&data[..end])
                    .map_err(|e| MoonclipError::Serialization(e.to_string()))?
            }
            Err(MoonclipError::NotFound(_)) => Manifest {
                world_size: config.world_size,
                retention: config.retention.clone(),
                lineage: config.lineage.clone(),
                ..Default::default()
            },
            Err(e) => return Err(e),
        };

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

        // Spawn merger if configured
        let merger = config.merger.clone().map(|mc| {
            DeltaMerger::new(
                mc,
                Arc::clone(&storage),
                Arc::clone(&manifest),
                config.compression.clone(),
            )
        });

        // Spawn remote syncer if configured
        let syncer = match (&config.remote_storage, &config.remote_sync) {
            (Some(remote), Some(sync_config)) => Some(RemoteSyncer::new(
                Arc::clone(&storage),
                Arc::clone(remote),
                sync_config.clone(),
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
            retained_base: Mutex::new(None),
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
    pub fn merge_now(&self) {
        self.wait_idle();
        self.core.merge_now();
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
}

impl Core {
    /// Full single-rank save pipeline (runs on the caller thread or the
    /// background save thread).
    fn save_sync(
        &self,
        snap_id: Uuid,
        step: u64,
        tensors: Vec<TensorData>,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        let snap_dir = format!("snapshots/{}", snap_id);
        let save_started = std::time::Instant::now();
        let raw_bytes: u64 = tensors.iter().map(|t| t.data.len() as u64).sum();

        let manifest = self.manifest.lock().unwrap();
        let force_full = manifest.should_force_full(step);
        let base_snap = if force_full {
            None
        } else {
            manifest.last_full_snapshot().cloned()
        };
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
        self.apply_retention(&mut manifest)?;
        drop(manifest);

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

        let manifest = self.manifest.lock().unwrap();
        let force_full = manifest.should_force_full(step);
        let base_id = if force_full {
            None
        } else {
            manifest.last_full_snapshot().map(|s| s.id)
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
        drop(manifest);

        let snap_dir = format!("snapshots/{}", snap_id);
        let context = SnapshotContext {
            step: snap.step,
            created_at: snap.created_at,
            metadata: snap.metadata.clone(),
        };
        let rank_entry =
            self.save_rank_tensors(snap_id, &snap_dir, &base_snap, tensors, &context)?;

        // Re-read manifest again (another rank may have saved concurrently)
        self.reload_manifest()?;

        // Update the snapshot with this rank's data
        let mut manifest = self.manifest.lock().unwrap();
        if let Some(snap) = manifest.snapshots.iter_mut().find(|s| s.id == snap_id) {
            snap.ranks.insert(self.config.rank, rank_entry);
        }
        self.persist_manifest(&manifest)?;

        Ok(())
    }

    fn finalize_snapshot(&self, snap_id: Uuid) -> Result<()> {
        // Re-read from storage to see all ranks' contributions
        self.reload_manifest()?;

        let mut manifest = self.manifest.lock().unwrap();
        if let Some(snap) = manifest.snapshots.iter_mut().find(|s| s.id == snap_id) {
            // Verify all ranks have reported
            let expected = self.config.world_size;
            let actual = snap.ranks.len() as u32;
            if actual < expected {
                return Err(MoonclipError::Config(format!(
                    "Cannot finalize: only {actual}/{expected} ranks have saved"
                )));
            }
            snap.finalized = true;
        } else {
            return Err(MoonclipError::NotFound(format!("Snapshot {snap_id}")));
        }

        self.apply_retention(&mut manifest)?;
        drop(manifest);

        if let Some(ref merger) = self.merger {
            merger.notify();
        }

        if let Some(ref syncer) = self.syncer {
            syncer.notify_save();
        }

        Ok(())
    }

    fn load(&self, snap_id: Uuid) -> Result<HashMap<String, Vec<u8>>> {
        use rayon::prelude::*;

        let manifest = self.manifest.lock().unwrap();
        let snap = manifest
            .find_snapshot(snap_id)
            .ok_or_else(|| MoonclipError::NotFound(format!("Snapshot {snap_id}")))?
            .clone();
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
         -> Result<(TensorEntry, CompressionAlgo, Option<Arc<Vec<u8>>>)> {
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

    fn merge_now(&self) {
        if let Some(ref merger) = self.merger {
            merger.force_full_merge();
        }
    }

    fn sync_now(&self) -> Result<()> {
        match self.syncer {
            Some(ref syncer) => syncer.sync_now(),
            None => Ok(()),
        }
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
        let has_data = blobs_end > pack::HEADER_LEN;
        let pack_file =
            has_data.then(|| format!("{}/rank_{}.pack", snap_dir, self.config.rank));

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
            pack_file,
            total_compressed,
            total_raw,
            skipped_count: skipped,
            delta_count: delta,
            full_count: full,
        };

        // ── 5. Write header + blobs + descriptor, in one object ─────
        if let Some(ref pack_filename) = rank_entry.pack_file {
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
        }

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
    /// Skipped when a cast is configured: what a later delta is computed
    /// against is the post-cast bytes stored on disk, and those are not what
    /// arrives here. Getting that wrong would corrupt every delta, so the
    /// cast path keeps reading the base from storage.
    fn retain_base(
        &self,
        snap_id: Uuid,
        tensors: Vec<TensorData>,
        processed: &[tensor::ProcessedTensor],
    ) {
        if !self.config.keep_base_in_memory || self.config.save_dtype != DType::None {
            return;
        }

        // Only tensors actually stored in full can be delta'd against later.
        let full: std::collections::HashSet<&str> = processed
            .iter()
            .filter(|pt| pt.entry.storage == TensorStorage::Full)
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
    ) -> Result<(TensorEntry, CompressionAlgo, Option<Arc<Vec<u8>>>)> {
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
    fn apply_retention(&self, manifest: &mut Manifest) -> Result<()> {
        let max_full = manifest.retention.max_full_snapshots;
        let max_total = manifest.retention.effective_total_cap();
        let rollback_ids = manifest.rollback_snapshot_ids();

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
                Some(oldest_id) => self.remove_snapshot_group(manifest, oldest_id)?,
                None => break, // All remaining are rollback-protected
            }
        }

        // ── Pass 2: Cap total snapshot count ─────────────────────────
        // Remove oldest snapshots (full groups) until total ≤ max_total.
        loop {
            let total = manifest.snapshots.iter().filter(|s| s.finalized).count();
            if total <= max_total {
                break;
            }

            // Find oldest removable full snapshot
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
                Some(oldest_id) => self.remove_snapshot_group(manifest, oldest_id)?,
                None => break,
            }
        }

        self.persist_manifest(manifest)?;
        Ok(())
    }

    /// Remove a full snapshot and all its dependent deltas.
    fn remove_snapshot_group(&self, manifest: &mut Manifest, full_id: Uuid) -> Result<()> {
        let to_remove: Vec<Uuid> = manifest
            .snapshots
            .iter()
            .filter(|s| s.id == full_id || s.base_snapshot_id == Some(full_id))
            .map(|s| s.id)
            .collect();

        // Delete files from storage
        for snap in manifest
            .snapshots
            .iter()
            .filter(|s| to_remove.contains(&s.id))
        {
            for rank_entry in snap.ranks.values() {
                // Delete pack file
                if let Some(ref pack_file) = rank_entry.pack_file {
                    let _ = self.storage.delete(pack_file);
                }
                // Delete legacy individual files
                for tensor in &rank_entry.tensors {
                    if let Some(ref filename) = tensor.filename {
                        let _ = self.storage.delete(filename);
                    }
                }
            }
            // No separate description to clean up: it lives inside the pack
            // that was just deleted.
            let _ = self.storage.remove_dir(&format!("snapshots/{}", snap.id));
        }

        manifest.snapshots.retain(|s| !to_remove.contains(&s.id));
        Ok(())
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
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| MoonclipError::Storage("Background saver is shut down".into()))?;

        let (lock, cvar) = &*self.shared;
        let mut state = lock.lock().unwrap();
        while state.busy {
            state = cvar.wait(state).unwrap();
        }
        // Surface a failure from the previous save before starting another,
        // matching what the old `self.flush()?` on entry did.
        if let Some(e) = state.error.take() {
            return Err(MoonclipError::Storage(format!("Background save failed: {e}")));
        }
        state.busy = true;
        drop(state);

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
}
