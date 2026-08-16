use std::collections::HashMap;
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread;

use chrono::Utc;
use uuid::Uuid;

use crate::cast::DType;
use crate::error::{Result, MoonclipError};
use crate::manifest::*;
use crate::merger::{DeltaMerger, MergerConfig};
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
    pub fn sync_now(&self) {
        self.wait_idle();
        self.core.sync_now();
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

        let rank_entry = self.save_rank_tensors(snap_id, &snap_dir, &base_snap, tensors)?;

        let snapshot = Snapshot {
            id: snap_id,
            step,
            created_at: Utc::now(),
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
        let rank_entry = self.save_rank_tensors(snap_id, &snap_dir, &base_snap, tensors)?;

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

    fn sync_now(&self) {
        if let Some(ref syncer) = self.syncer {
            syncer.sync_now();
        }
    }

    // ── Internal ─────────────────────────────────────────────────────

    fn save_rank_tensors(
        &self,
        snap_id: Uuid,
        snap_dir: &str,
        base_snap: &Option<Snapshot>,
        tensors: Vec<TensorData>,
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
        let mut offset = 0u64;
        for pt in &mut processed {
            if let Some(ref data) = pt.write_data {
                pt.entry.offset = offset;
                offset += data.len() as u64;
            }
        }

        let pack_file = if offset > 0 {
            let parts: Vec<&[u8]> = processed
                .iter()
                .filter_map(|pt| pt.write_data.as_deref())
                .collect();
            let pack_filename = format!("{}/rank_{}.pack", snap_dir, self.config.rank);
            crate::profile::time(crate::profile::Phase::PackWrite, || {
                self.storage.put_parts(&pack_filename, &parts)
            })?;
            Some(pack_filename)
        } else {
            None
        };

        // ── 4. Build rank entry ─────────────────────────────────────
        let mut entries = Vec::new();
        let mut total_compressed = 0u64;
        let mut total_raw = 0u64;
        let mut skipped = 0usize;
        let mut delta = 0usize;
        let mut full = 0usize;

        for pt in processed {
            total_compressed += pt.entry.compressed_size;
            total_raw += pt.entry.raw_size;
            match pt.entry.storage {
                // Aliases write no bytes, same as a skipped tensor.
                TensorStorage::Skipped | TensorStorage::Alias => skipped += 1,
                TensorStorage::DeltaXor => delta += 1,
                TensorStorage::Full => full += 1,
            }
            entries.push(pt.entry);
        }

        Ok(RankEntry {
            rank: self.config.rank,
            tensors: entries,
            pack_file,
            total_compressed,
            total_raw,
            skipped_count: skipped,
            delta_count: delta,
            full_count: full,
        })
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
                    let result = core.save_sync(job.snap_id, job.step, job.tensors, job.metadata);
                    let mut state = shared2.0.lock().unwrap();
                    state.busy = false;
                    if let Err(e) = result {
                        state.error = Some(e.to_string());
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
}
