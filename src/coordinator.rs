use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use chrono::Utc;
use uuid::Uuid;

use crate::cast::DType;
use crate::error::{Result, RevolverError};
use crate::hash::hash_hex;
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
    pub delta_threshold: f64,
    pub merger: Option<MergerConfig>,
    pub remote_storage: Option<Arc<dyn StorageBackend>>,
    pub remote_sync: Option<RemoteSyncConfig>,
    /// Target dtype for saving float tensors. DType::None = keep original.
    pub save_dtype: DType,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        CoordinatorConfig {
            world_size: 1,
            rank: 0,
            compression: CompressionAlgo::Zstd { level: 3 },
            retention: RetentionPolicy::default(),
            lineage: LineageConfig::default(),
            delta_threshold: 0.5,
            merger: None,
            remote_storage: None,
            remote_sync: None,
            save_dtype: DType::None,
        }
    }
}

/// The main checkpoint coordinator.
///
/// In single-rank mode (world_size=1), this handles everything.
/// In multi-rank mode, each rank has its own Coordinator instance
/// with the same storage backend and manifest. Rank 0 is responsible
/// for creating/finalizing snapshots.
pub struct Coordinator {
    pub(crate) storage: Arc<dyn StorageBackend>,
    pub(crate) manifest: Arc<Mutex<Manifest>>,
    config: CoordinatorConfig,
    merger: Option<DeltaMerger>,
    syncer: Option<RemoteSyncer>,
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
                    .map_err(|e| RevolverError::Serialization(e.to_string()))?
            }
            Err(RevolverError::NotFound(_)) => Manifest {
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

        Ok(Coordinator {
            storage,
            manifest,
            config,
            merger,
            syncer,
        })
    }

    /// Save a checkpoint for this rank.
    ///
    /// For single-rank (world_size=1): creates snapshot, saves tensors, finalizes.
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
        if self.config.world_size == 1 {
            // Single-rank fast path
            let snap_id = Uuid::new_v4();
            let snap_dir = format!("snapshots/{}", snap_id);

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
            self.persist_manifest(&manifest)?;
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

            Ok(snap_id)
        } else {
            // Multi-rank: save this rank's tensors into a pre-created snapshot.
            // The caller is responsible for coordination.
            Err(RevolverError::Config(
                "Multi-rank save requires explicit create_snapshot/save_rank/finalize flow. \
                 Use save_rank() instead."
                    .into(),
            ))
        }
    }

    /// Create a new snapshot entry (rank 0 only in multi-rank).
    /// Returns the snapshot ID that all ranks should use.
    pub fn create_snapshot(&self, step: u64, metadata: HashMap<String, String>) -> Result<Uuid> {
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

    /// Save this rank's tensors into an existing snapshot.
    /// Called by each rank independently.
    pub fn save_rank(&self, snap_id: Uuid, tensors: Vec<TensorData>) -> Result<()> {
        // Re-read manifest from storage for multi-rank correctness
        // (another rank may have created the snapshot)
        self.reload_manifest()?;

        let manifest = self.manifest.lock().unwrap();
        let snap = manifest
            .find_snapshot(snap_id)
            .ok_or_else(|| RevolverError::NotFound(format!("Snapshot {snap_id}")))?
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

    /// Finalize a snapshot (rank 0 only in multi-rank).
    /// Marks it as complete and triggers retention/merging.
    pub fn finalize_snapshot(&self, snap_id: Uuid) -> Result<()> {
        // Re-read from storage to see all ranks' contributions
        self.reload_manifest()?;

        let mut manifest = self.manifest.lock().unwrap();
        if let Some(snap) = manifest.snapshots.iter_mut().find(|s| s.id == snap_id) {
            // Verify all ranks have reported
            let expected = self.config.world_size;
            let actual = snap.ranks.len() as u32;
            if actual < expected {
                return Err(RevolverError::Config(format!(
                    "Cannot finalize: only {actual}/{expected} ranks have saved"
                )));
            }
            snap.finalized = true;
        } else {
            return Err(RevolverError::NotFound(format!("Snapshot {snap_id}")));
        }

        self.persist_manifest(&manifest)?;
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

    /// Load a snapshot for this rank. Returns tensor name → raw bytes.
    pub fn load(&self, snap_id: Uuid) -> Result<HashMap<String, Vec<u8>>> {
        let manifest = self.manifest.lock().unwrap();
        let snap = manifest
            .find_snapshot(snap_id)
            .ok_or_else(|| RevolverError::NotFound(format!("Snapshot {snap_id}")))?
            .clone();
        drop(manifest);

        let rank_entry = snap.ranks.get(&self.config.rank).ok_or_else(|| {
            RevolverError::NotFound(format!(
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

        let mut result = HashMap::new();

        for entry in &rank_entry.tensors {
            let data = tensor::load_tensor(
                entry,
                snap.base_snapshot_id,
                self.storage.as_ref(),
                &snap.compression,
                pack_data.as_deref(),
                &|base_id, tensor_name| self.find_base_tensor_entry_with_pack(base_id, tensor_name),
            )?;
            result.insert(entry.name.clone(), data);
        }

        Ok(result)
    }

    /// Load the latest finalized snapshot.
    pub fn load_latest(&self) -> Result<(Uuid, HashMap<String, Vec<u8>>)> {
        let manifest = self.manifest.lock().unwrap();
        let snap = manifest
            .snapshots
            .iter()
            .rev()
            .find(|s| s.finalized)
            .ok_or_else(|| RevolverError::NotFound("No finalized snapshots".into()))?
            .clone();
        drop(manifest);

        let id = snap.id;
        let data = self.load(id)?;
        Ok((id, data))
    }

    /// List all snapshots.
    pub fn list_snapshots(&self) -> Vec<SnapshotInfo> {
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

    /// Force merge all pending deltas into a full checkpoint.
    pub fn merge_now(&self) {
        if let Some(ref merger) = self.merger {
            merger.force_full_merge();
        }
    }

    /// Force sync all local data to remote storage immediately.
    pub fn sync_now(&self) {
        if let Some(ref syncer) = self.syncer {
            syncer.sync_now();
        }
    }

    // ── Internal ─────────────────────────────────────────────────────

    fn save_rank_tensors(
        &self,
        _snap_id: Uuid,
        snap_dir: &str,
        base_snap: &Option<Snapshot>,
        tensors: Vec<TensorData>,
    ) -> Result<RankEntry> {
        // ── 1. Build base cache (single disk read) ──────────────────
        let base_cache = self.build_base_cache(base_snap)?;

        // ── 2. Process tensors in parallel (no disk I/O) ────────────
        let mut processed = tensor::process_tensors_parallel(
            &tensors,
            base_cache.as_ref(),
            &self.config.compression,
            self.config.delta_threshold,
            &self.config.save_dtype,
        )?;

        // ── 3. Pack all write data into single buffer ───────────────
        let mut pack_buf: Vec<u8> = Vec::new();
        for pt in &mut processed {
            if let Some(ref data) = pt.write_data {
                pt.entry.offset = pack_buf.len() as u64;
                pack_buf.extend_from_slice(data);
            }
        }

        // ── 4. Single disk write ────────────────────────────────────
        let pack_file = if !pack_buf.is_empty() {
            // Hash the entire pack for integrity
            let pack_hash = hash_hex(&pack_buf);
            let pack_filename = format!("{}/rank_{}.pack", snap_dir, self.config.rank);
            self.storage.put(&pack_filename, &pack_buf)?;

            // Set compressed hash per tensor (from their slice in the pack)
            for pt in &mut processed {
                if pt.write_data.is_some() {
                    let start = pt.entry.offset as usize;
                    let end = start + pt.entry.compressed_size as usize;
                    pt.entry.sha256_compressed = Some(hash_hex(&pack_buf[start..end]));
                }
            }

            drop(pack_buf); // Free the buffer

            // Store the overall pack hash in metadata (optional, for verification)
            let _ = pack_hash;
            Some(pack_filename)
        } else {
            None
        };

        // ── 5. Build rank entry ─────────────────────────────────────
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
                TensorStorage::Skipped => skipped += 1,
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

    /// Build a BaseCache from the base snapshot.
    /// Reads the pack file (or legacy individual files) in a single pass.
    fn build_base_cache(&self, base_snap: &Option<Snapshot>) -> Result<Option<BaseCache>> {
        let base = match base_snap {
            Some(s) => s,
            None => return Ok(None),
        };

        let rank_entry = match base.ranks.get(&self.config.rank) {
            Some(re) => re,
            None => return Ok(None),
        };

        let entries: HashMap<String, TensorEntry> = rank_entry
            .tensors
            .iter()
            .map(|t| (t.name.clone(), t.clone()))
            .collect();

        // Try pack file first (single read)
        if let Some(ref pack_file) = rank_entry.pack_file {
            let pack_data = self.storage.get(pack_file)?;
            return Ok(Some(BaseCache {
                pack_data,
                entries,
                compression: base.compression.clone(),
                legacy_files: HashMap::new(),
            }));
        }

        // Legacy mode: pre-load all individual tensor files
        let mut legacy_files = HashMap::new();
        for entry in &rank_entry.tensors {
            if entry.storage == TensorStorage::Full {
                if let Some(ref filename) = entry.filename {
                    match self.storage.get(filename) {
                        Ok(data) => { legacy_files.insert(filename.clone(), data); }
                        Err(_) => {} // Skip missing files
                    }
                }
            }
        }

        Ok(Some(BaseCache {
            pack_data: Vec::new(),
            entries,
            compression: base.compression.clone(),
            legacy_files,
        }))
    }

    /// Find base tensor entry + pack data for loading.
    /// Returns (TensorEntry, CompressionAlgo, Option<pack_data>).
    fn find_base_tensor_entry_with_pack(
        &self,
        base_id: Uuid,
        tensor_name: &str,
    ) -> Result<(TensorEntry, CompressionAlgo, Option<Vec<u8>>)> {
        let manifest = self.manifest.lock().unwrap();
        let base_snap = manifest
            .find_snapshot(base_id)
            .ok_or_else(|| RevolverError::NotFound(format!("Base snapshot {base_id}")))?;

        let rank_entry = base_snap.ranks.get(&self.config.rank).ok_or_else(|| {
            RevolverError::NotFound(format!(
                "Rank {} not in base snapshot {base_id}",
                self.config.rank
            ))
        })?;

        let entry = rank_entry
            .tensors
            .iter()
            .find(|t| t.name == tensor_name)
            .ok_or_else(|| {
                RevolverError::NotFound(format!(
                    "Tensor '{}' not in base snapshot {base_id}",
                    tensor_name
                ))
            })?
            .clone();

        let compression = base_snap.compression.clone();

        // Load pack file if present
        let pack_data = if let Some(ref pack_file) = rank_entry.pack_file {
            // TODO: cache this across calls (currently re-reads per tensor)
            Some(self.storage.get(pack_file)?)
        } else {
            None
        };

        drop(manifest);
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
                    .map_err(|e| RevolverError::Serialization(e.to_string()))?;
                let mut m = self.manifest.lock().unwrap();
                *m = new_manifest;
                Ok(())
            }
            Err(RevolverError::NotFound(_)) => Ok(()),
            Err(e) => Err(e),
        }
    }

    fn persist_manifest(&self, manifest: &Manifest) -> Result<()> {
        let json = serde_json::to_vec_pretty(manifest)
            .map_err(|e| RevolverError::Serialization(e.to_string()))?;
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
            delta_threshold: 0.5,
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
            delta_threshold: 0.5,
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
            delta_threshold: 0.5,
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
            delta_threshold: 0.5,
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

    #[test]
    fn pack_file_created() {
        let dir = tempfile::tempdir().unwrap();
        let coord = make_coordinator(dir.path());

        let tensors = vec![
            TensorData { name: "a".into(), shape: vec![100], dtype: "uint8".into(), data: vec![1u8; 100] },
            TensorData { name: "b".into(), shape: vec![200], dtype: "uint8".into(), data: vec![2u8; 200] },
        ];

        coord.save(1, tensors, HashMap::new()).unwrap();

        // Check that a .pack file exists (not individual .bin files)
        let manifest = coord.manifest.lock().unwrap();
        let snap = &manifest.snapshots[0];
        let re = snap.ranks.get(&0).unwrap();
        assert!(re.pack_file.is_some(), "Expected pack_file to be set");
        assert!(
            re.pack_file.as_ref().unwrap().ends_with(".pack"),
            "Expected .pack extension"
        );
    }
}
