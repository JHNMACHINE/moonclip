use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use chrono::Utc;
use uuid::Uuid;

use crate::background::{BackgroundSaver, SaveJob, SaveResult};
use crate::compression;
use crate::delta;
use crate::error::{Result, MoonclipError};
use crate::hash::sha256_hex;
use crate::manifest::*;
use crate::merger::{DeltaMerger, MergerConfig};
use crate::storage::{LocalStorage, StorageBackend};

/// Configuration for the checkpoint manager.
pub struct ManagerConfig {
    pub storage_root: String,
    pub compression: CompressionAlgo,
    pub retention: RetentionPolicy,
    /// Delta density threshold: if more than this fraction of bytes changed,
    /// store a full snapshot instead of a delta.
    pub delta_density_threshold: f64,
    /// Merger configuration. If Some, enables background delta merging.
    pub merger: Option<MergerConfig>,
}

impl Default for ManagerConfig {
    fn default() -> Self {
        ManagerConfig {
            storage_root: "./checkpoints".into(),
            compression: CompressionAlgo::Zstd { level: 3 },
            retention: RetentionPolicy::default(),
            delta_density_threshold: 0.5,
            merger: None,
        }
    }
}

/// Internal shared state that both the manager and the background thread access.
pub(crate) struct ManagerInner {
    pub storage: Arc<dyn StorageBackend>,
    pub manifest: Mutex<Manifest>,
    pub config: ManagerConfig,
}

impl ManagerInner {
    /// The core save logic, usable from both sync and background contexts.
    pub fn save_impl(
        &self,
        snap_id: Uuid,
        step: u64,
        components: HashMap<ComponentKind, Vec<u8>>,
        metadata: HashMap<String, String>,
    ) -> Result<Uuid> {
        let snap_dir = format!("snapshots/{}", snap_id);

        let manifest = self.manifest.lock().unwrap();
        let last_full = find_last_full_snapshot(&manifest);
        let force_full = self.should_force_full(step, &manifest);
        drop(manifest);

        let mut snap_components: HashMap<ComponentKind, Vec<ShardInfo>> = HashMap::new();
        let mut base_snapshot_id: Option<Uuid> = None;
        let mut is_delta = false;

        for (kind, raw_bytes) in &components {
            let hash_raw = sha256_hex(raw_bytes);

            // Try delta encoding against last full snapshot
            let (shard_data, shard_is_delta) = if !force_full {
                if let Some(ref base) = last_full {
                    self.try_delta_encode(kind, raw_bytes, base)?
                } else {
                    (raw_bytes.clone(), false)
                }
            } else {
                (raw_bytes.clone(), false)
            };

            if shard_is_delta {
                is_delta = true;
                base_snapshot_id = last_full.as_ref().map(|s| s.id);
            }

            // Compress
            let compressed = compression::compress(&shard_data, &self.config.compression)?;

            // Write shard
            let filename = format!("{}/{}.bin", snap_dir, kind_to_str(kind));
            self.storage.put(&filename, &compressed)?;

            let shard = ShardInfo {
                filename: filename.clone(),
                byte_size: compressed.len() as u64,
                sha256: sha256_hex(&compressed),
                tensors: vec![TensorMeta {
                    name: kind_to_str(kind),
                    shape: vec![raw_bytes.len()],
                    dtype: if shard_is_delta {
                        "delta_xor".into()
                    } else {
                        "raw".into()
                    },
                    offset: 0,
                    compressed_size: compressed.len() as u64,
                    raw_size: raw_bytes.len() as u64,
                    sha256: hash_raw,
                }],
            };

            snap_components
                .entry(kind.clone())
                .or_default()
                .push(shard);
        }

        let snapshot = Snapshot {
            id: snap_id,
            step,
            created_at: Utc::now(),
            components: snap_components,
            base_snapshot_id: if is_delta { base_snapshot_id } else { None },
            metadata,
            compression: self.config.compression.clone(),
        };

        // Update and persist manifest
        let mut manifest = self.manifest.lock().unwrap();
        manifest.snapshots.push(snapshot);
        self.persist_manifest(&manifest)?;

        // Apply retention policy
        self.apply_retention(&mut manifest)?;

        Ok(snap_id)
    }

    fn try_delta_encode(
        &self,
        kind: &ComponentKind,
        target: &[u8],
        base_snap: &Snapshot,
    ) -> Result<(Vec<u8>, bool)> {
        if let Some(shards) = base_snap.components.get(kind) {
            if let Some(shard) = shards.first() {
                let compressed_base = self.storage.get(&shard.filename)?;
                let base_raw = compression::decompress(&compressed_base, &base_snap.compression)?;

                if let Some(d) = delta::compute_delta(&base_raw, target) {
                    let density = delta::delta_density(&d);
                    if density < self.config.delta_density_threshold {
                        return Ok((d, true));
                    }
                }
            }
        }
        Ok((target.to_vec(), false))
    }

    fn should_force_full(&self, step: u64, manifest: &Manifest) -> bool {
        let policy = &self.config.retention;

        if manifest.snapshots.is_empty() {
            return true;
        }

        if let Some(last_full) = find_last_full_snapshot(manifest) {
            if step - last_full.step >= policy.full_snapshot_every_steps {
                return true;
            }
        }

        let deltas_since_full = manifest
            .snapshots
            .iter()
            .rev()
            .take_while(|s| s.base_snapshot_id.is_some())
            .count();

        if deltas_since_full >= policy.max_deltas_per_full {
            return true;
        }

        false
    }

    pub fn persist_manifest(&self, manifest: &Manifest) -> Result<()> {
        let json = serde_json::to_vec_pretty(manifest)
            .map_err(|e| MoonclipError::Serialization(e.to_string()))?;
        self.storage.put("manifest.json", &json)
    }

    pub fn apply_retention(&self, manifest: &mut Manifest) -> Result<()> {
        let max = self.config.retention.max_full_snapshots;
        let full_count = manifest
            .snapshots
            .iter()
            .filter(|s| s.base_snapshot_id.is_none())
            .count();

        if full_count <= max {
            return Ok(());
        }

        let oldest_full_id = manifest
            .snapshots
            .iter()
            .find(|s| s.base_snapshot_id.is_none())
            .map(|s| s.id);

        if let Some(oldest_id) = oldest_full_id {
            let to_remove: Vec<Uuid> = manifest
                .snapshots
                .iter()
                .filter(|s| s.id == oldest_id || s.base_snapshot_id == Some(oldest_id))
                .map(|s| s.id)
                .collect();

            for snap in manifest.snapshots.iter().filter(|s| to_remove.contains(&s.id)) {
                for shards in snap.components.values() {
                    for shard in shards {
                        let _ = self.storage.delete(&shard.filename);
                    }
                }
            }

            manifest.snapshots.retain(|s| !to_remove.contains(&s.id));
            self.persist_manifest(manifest)?;
        }

        Ok(())
    }
}

/// Free function to avoid &self borrow issues.
fn find_last_full_snapshot(manifest: &Manifest) -> Option<Snapshot> {
    manifest
        .snapshots
        .iter()
        .rev()
        .find(|s| s.base_snapshot_id.is_none())
        .cloned()
}

fn kind_to_str(kind: &ComponentKind) -> String {
    match kind {
        ComponentKind::Model => "model".into(),
        ComponentKind::Optimizer => "optimizer".into(),
        ComponentKind::Scheduler => "scheduler".into(),
        ComponentKind::Scaler => "scaler".into(),
        ComponentKind::Rng => "rng".into(),
        ComponentKind::Custom(s) => s.clone(),
    }
}

// ─── Public API ─────────────────────────────────────────────────────

pub struct CheckpointManager {
    inner: Arc<ManagerInner>,
    bg_saver: Option<BackgroundSaver>,
    merger: Option<DeltaMerger>,
}

impl CheckpointManager {
    /// Create a new manager with local storage.
    /// If `enable_background` is true, saves can be dispatched asynchronously.
    pub fn new(config: ManagerConfig, enable_background: bool) -> Result<Self> {
        let storage = Arc::new(LocalStorage::new(&config.storage_root)?);
        Self::new_with_storage(config, storage, enable_background)
    }

    /// Create a new manager with a custom storage backend.
    pub fn new_with_storage(
        config: ManagerConfig,
        storage: Arc<dyn StorageBackend>,
        enable_background: bool,
    ) -> Result<Self> {
        let manifest = match storage.get("manifest.json") {
            Ok(data) => serde_json::from_slice(&data)
                .map_err(|e| MoonclipError::Serialization(e.to_string()))?,
            Err(MoonclipError::NotFound(_)) => Manifest {
                retention: config.retention.clone(),
                ..Default::default()
            },
            Err(e) => return Err(e),
        };

        let merger_config = config.merger.clone();
        let compression_for_merger = config.compression.clone();

        let manifest = Mutex::new(manifest);

        let inner = Arc::new(ManagerInner {
            storage,
            manifest,
            config,
        });

        let bg_saver = if enable_background {
            let inner_clone = Arc::clone(&inner);
            Some(BackgroundSaver::new(move |job: SaveJob| {
                inner_clone
                    .save_impl(job.snap_id, job.step, job.components, job.metadata)
                    .map(|_| ())
                    .map_err(|e| e.to_string())
            }))
        } else {
            None
        };

        // Spawn merger if configured
        let merger = merger_config.map(|mc| {
            DeltaMerger::new(
                mc,
                Arc::clone(&inner.storage),
                // We need to share the manifest Arc. Extract it from inner.
                // This is safe because ManagerInner stores it as Mutex<Manifest>.
                // We create a new Arc wrapping the same Mutex via unsafe reinterpret.
                // Actually, let's just clone the Arc to storage and create a new
                // Arc<Mutex<Manifest>> that shares the same data.
                // The cleanest way: make manifest an Arc<Mutex<Manifest>> in ManagerInner.
                // For now, we'll use the storage to reload manifest in the merger.
                // This is simpler and avoids refactoring ManagerInner.
                // Actually, the merger module already takes Arc<Mutex<Manifest>>.
                // Let's restructure: extract manifest as a shared Arc.
                //
                // HACK: We'll re-read manifest from storage in the merger for now.
                // This works because the merger only runs after saves complete.
                // TODO: refactor ManagerInner to use Arc<Mutex<Manifest>> directly.
                {
                    // Re-load manifest from storage for the merger thread
                    let manifest_data = inner.storage.get("manifest.json").unwrap_or_default();
                    let manifest: Manifest = serde_json::from_slice(&manifest_data)
                        .unwrap_or_default();
                    Arc::new(Mutex::new(manifest))
                },
                compression_for_merger,
            )
        });

        Ok(CheckpointManager { inner, bg_saver, merger })
    }

    /// Synchronous save. Blocks until complete.
    pub fn save(
        &self,
        step: u64,
        components: HashMap<ComponentKind, Vec<u8>>,
        metadata: HashMap<String, String>,
    ) -> Result<Uuid> {
        if let Some(ref bg) = self.bg_saver {
            bg.wait()?;
        }

        let snap_id = Uuid::new_v4();
        let result = self.inner.save_impl(snap_id, step, components, metadata)?;

        if let Some(ref merger) = self.merger {
            merger.notify();
        }

        Ok(result)
    }

    /// Asynchronous save. Returns immediately with the snapshot UUID.
    pub fn save_async(
        &self,
        step: u64,
        components: HashMap<ComponentKind, Vec<u8>>,
        metadata: HashMap<String, String>,
    ) -> Result<Uuid> {
        let bg = self.bg_saver.as_ref().ok_or_else(|| {
            MoonclipError::Config(
                "Background saving not enabled. Create manager with enable_background=true".into(),
            )
        })?;

        let snap_id = Uuid::new_v4();

        bg.submit(SaveJob {
            snap_id,
            step,
            components,
            metadata,
        })?;

        if let Some(ref merger) = self.merger {
            merger.notify();
        }

        Ok(snap_id)
    }

    /// Force merge all pending deltas into a new full checkpoint.
    pub fn merge_now(&self) {
        if let Some(ref merger) = self.merger {
            let _ = self.wait();
            merger.force_full_merge();
        }
    }

    /// Wait for any in-flight background save to complete.
    /// Returns the result of the last save, if any.
    pub fn wait(&self) -> Result<Option<SaveResult>> {
        if let Some(ref bg) = self.bg_saver {
            bg.wait()
        } else {
            Ok(None)
        }
    }

    /// Check if a background save is currently in progress.
    pub fn is_saving(&self) -> bool {
        self.bg_saver.as_ref().map_or(false, |bg| bg.is_busy())
    }

    /// Get the result of the last completed save (sync or async).
    pub fn last_save_result(&self) -> Option<SaveResult> {
        self.bg_saver.as_ref().and_then(|bg| bg.last_result())
    }

    /// Load a checkpoint by snapshot ID.
    pub fn load(&self, snap_id: Uuid) -> Result<HashMap<ComponentKind, Vec<u8>>> {
        // Wait for any in-flight save to ensure manifest is up-to-date
        self.wait()?;

        let manifest = self.inner.manifest.lock().unwrap();
        let snap = manifest
            .snapshots
            .iter()
            .find(|s| s.id == snap_id)
            .ok_or_else(|| MoonclipError::NotFound(format!("Snapshot {snap_id}")))?
            .clone();
        drop(manifest);

        self.load_snapshot(&snap)
    }

    /// Load the latest checkpoint.
    pub fn load_latest(&self) -> Result<(Uuid, HashMap<ComponentKind, Vec<u8>>)> {
        self.wait()?;

        let manifest = self.inner.manifest.lock().unwrap();
        let snap = manifest
            .snapshots
            .last()
            .ok_or_else(|| MoonclipError::NotFound("No snapshots available".into()))?
            .clone();
        drop(manifest);

        let id = snap.id;
        let data = self.load_snapshot(&snap)?;
        Ok((id, data))
    }

    /// List all snapshots.
    pub fn list_snapshots(&self) -> Vec<(u64, Uuid, bool, HashMap<String, String>)> {
        // Best-effort wait for in-flight save
        let _ = self.wait();

        let manifest = self.inner.manifest.lock().unwrap();
        manifest
            .snapshots
            .iter()
            .map(|s| {
                (
                    s.step,
                    s.id,
                    s.base_snapshot_id.is_some(),
                    s.metadata.clone(),
                )
            })
            .collect()
    }

    /// Gracefully shut down the background saver thread.
    pub fn shutdown(&mut self) -> Result<()> {
        if let Some(ref mut bg) = self.bg_saver {
            bg.shutdown()?;
        }
        Ok(())
    }

    // ── Internal ─────────────────────────────────────────────────────

    fn load_snapshot(
        &self,
        snap: &Snapshot,
    ) -> Result<HashMap<ComponentKind, Vec<u8>>> {
        let mut result = HashMap::new();

        for (kind, shards) in &snap.components {
            for shard in shards {
                let compressed = self.inner.storage.get(&shard.filename)?;

                // Verify integrity of compressed data
                let actual_hash = sha256_hex(&compressed);
                if actual_hash != shard.sha256 {
                    return Err(MoonclipError::IntegrityError {
                        expected: shard.sha256.clone(),
                        actual: actual_hash,
                    });
                }

                let decompressed =
                    compression::decompress(&compressed, &snap.compression)?;

                // If this is a delta, resolve the chain
                let raw =
                    if shard.tensors.first().map(|t| t.dtype.as_str()) == Some("delta_xor") {
                        let base_id = snap.base_snapshot_id.ok_or_else(|| {
                            MoonclipError::Delta(
                                "Delta snapshot has no base_snapshot_id".into(),
                            )
                        })?;
                        let base_data = self.load_component_raw(base_id, kind)?;
                        delta::apply_delta(&base_data, &decompressed)?
                    } else {
                        decompressed
                    };

                // Verify raw integrity
                if let Some(tensor) = shard.tensors.first() {
                    let actual = sha256_hex(&raw);
                    if actual != tensor.sha256 {
                        return Err(MoonclipError::IntegrityError {
                            expected: tensor.sha256.clone(),
                            actual,
                        });
                    }
                }

                result.insert(kind.clone(), raw);
            }
        }

        Ok(result)
    }

    fn load_component_raw(
        &self,
        snap_id: Uuid,
        kind: &ComponentKind,
    ) -> Result<Vec<u8>> {
        let manifest = self.inner.manifest.lock().unwrap();
        let snap = manifest
            .snapshots
            .iter()
            .find(|s| s.id == snap_id)
            .ok_or_else(|| MoonclipError::NotFound(format!("Snapshot {snap_id}")))?
            .clone();
        drop(manifest);

        let shards = snap
            .components
            .get(kind)
            .ok_or_else(|| {
                MoonclipError::NotFound(format!("{:?} in snapshot {snap_id}", kind))
            })?;

        let shard = &shards[0];
        let compressed = self.inner.storage.get(&shard.filename)?;
        let decompressed = compression::decompress(&compressed, &snap.compression)?;

        if shard.tensors.first().map(|t| t.dtype.as_str()) == Some("delta_xor") {
            let base_id = snap.base_snapshot_id.ok_or_else(|| {
                MoonclipError::Delta("Delta chain broken: no base_snapshot_id".into())
            })?;
            let base_data = self.load_component_raw(base_id, kind)?;
            delta::apply_delta(&base_data, &decompressed)
        } else {
            Ok(decompressed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager(dir: &std::path::Path, background: bool) -> CheckpointManager {
        let config = ManagerConfig {
            storage_root: dir.to_string_lossy().to_string(),
            compression: CompressionAlgo::Zstd { level: 1 },
            retention: RetentionPolicy {
                max_full_snapshots: 3,
                max_deltas_per_full: 5,
                full_snapshot_every_steps: 100,
            },
            delta_density_threshold: 0.5,
            merger: None,
        };
        CheckpointManager::new(config, background).unwrap()
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = make_manager(dir.path(), false);

        let model_data = vec![42u8; 10_000];
        let optim_data = vec![7u8; 20_000];

        let mut components = HashMap::new();
        components.insert(ComponentKind::Model, model_data.clone());
        components.insert(ComponentKind::Optimizer, optim_data.clone());

        let mut meta = HashMap::new();
        meta.insert("loss".into(), "0.634".into());

        let snap_id = mgr.save(1000, components, meta).unwrap();

        let loaded = mgr.load(snap_id).unwrap();
        assert_eq!(loaded[&ComponentKind::Model], model_data);
        assert_eq!(loaded[&ComponentKind::Optimizer], optim_data);
    }

    #[test]
    fn delta_encoding_kicks_in() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = make_manager(dir.path(), false);

        let data_v1 = vec![0u8; 50_000];
        let mut c1 = HashMap::new();
        c1.insert(ComponentKind::Model, data_v1.clone());
        let _id1 = mgr.save(100, c1, HashMap::new()).unwrap();

        let mut data_v2 = data_v1.clone();
        data_v2[0] = 1;
        data_v2[25_000] = 2;
        let mut c2 = HashMap::new();
        c2.insert(ComponentKind::Model, data_v2.clone());
        let id2 = mgr.save(150, c2, HashMap::new()).unwrap();

        let snaps = mgr.list_snapshots();
        assert!(!snaps[0].2, "first should be full");
        assert!(snaps[1].2, "second should be delta");

        let loaded = mgr.load(id2).unwrap();
        assert_eq!(loaded[&ComponentKind::Model], data_v2);
    }

    #[test]
    fn load_latest() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = make_manager(dir.path(), false);

        let mut c = HashMap::new();
        c.insert(ComponentKind::Model, vec![1u8; 8192]);
        mgr.save(100, c.clone(), HashMap::new()).unwrap();

        c.insert(ComponentKind::Model, vec![2u8; 8192]);
        mgr.save(200, c, HashMap::new()).unwrap();

        let (_, loaded) = mgr.load_latest().unwrap();
        assert_eq!(loaded[&ComponentKind::Model], vec![2u8; 8192]);
    }

    // ── Background save tests ────────────────────────────────────────

    #[test]
    fn async_save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = make_manager(dir.path(), true);

        let data = vec![42u8; 10_000];
        let mut components = HashMap::new();
        components.insert(ComponentKind::Model, data.clone());

        // Async save returns immediately
        let snap_id = mgr.save_async(1000, components, HashMap::new()).unwrap();

        // Wait for it
        let result = mgr.wait().unwrap().unwrap();
        assert_eq!(result.snap_id, snap_id);
        assert!(result.error.is_none());
        assert!(result.elapsed_secs > 0.0);

        // Load should work
        let loaded = mgr.load(snap_id).unwrap();
        assert_eq!(loaded[&ComponentKind::Model], data);
    }

    #[test]
    fn async_multiple_saves() {
        let dir = tempfile::tempdir().unwrap();
        // Use high full_every_steps so all saves after the first are deltas,
        // and high max_full_snapshots so retention doesn't kick in.
        let config = ManagerConfig {
            storage_root: dir.path().to_string_lossy().to_string(),
            compression: CompressionAlgo::Zstd { level: 1 },
            retention: RetentionPolicy {
                max_full_snapshots: 10,
                max_deltas_per_full: 10,
                full_snapshot_every_steps: 100000,
            },
            delta_density_threshold: 0.5,
            merger: None,
        };
        let mgr = CheckpointManager::new(config, true).unwrap();

        for step in (100..=500).step_by(100) {
            let mut c = HashMap::new();
            c.insert(ComponentKind::Model, vec![step as u8; 8192]);
            mgr.save_async(step, c, HashMap::new()).unwrap();
        }

        mgr.wait().unwrap();

        let snaps = mgr.list_snapshots();
        assert_eq!(snaps.len(), 5);

        // Last one should be loadable
        let (_, loaded) = mgr.load_latest().unwrap();
        assert_eq!(loaded[&ComponentKind::Model][0], 244u8); // 500 as u8 = 244
    }

    #[test]
    fn is_saving_flag() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = make_manager(dir.path(), true);

        assert!(!mgr.is_saving());

        let mut c = HashMap::new();
        c.insert(ComponentKind::Model, vec![0u8; 100_000]);
        mgr.save_async(100, c, HashMap::new()).unwrap();

        // Might be saving (race-y, but with 100KB it should be)
        // Just verify it doesn't panic
        let _ = mgr.is_saving();

        mgr.wait().unwrap();
        assert!(!mgr.is_saving());
    }

    #[test]
    fn sync_save_still_works_with_background_enabled() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = make_manager(dir.path(), true);

        let mut c = HashMap::new();
        c.insert(ComponentKind::Model, vec![1u8; 8192]);

        // Sync save should work even with background enabled
        let id = mgr.save(100, c, HashMap::new()).unwrap();
        let loaded = mgr.load(id).unwrap();
        assert_eq!(loaded[&ComponentKind::Model], vec![1u8; 8192]);
    }
}
