use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::collections::HashMap;
use std::sync::Arc;

use crate::coordinator::{Coordinator, CoordinatorConfig};
use crate::manifest::{CompressionAlgo, LineageConfig, RetentionPolicy};
use crate::merger::MergerConfig;
use crate::remote_sync::RemoteSyncConfig;
use crate::s3::{S3Config, S3Storage};
use crate::storage::LocalStorage;
use crate::tensor::TensorData;

fn extract_metadata(metadata: Option<&PyDict>) -> PyResult<HashMap<String, String>> {
    match metadata {
        Some(d) => {
            let mut m = HashMap::new();
            for (k, v) in d.iter() {
                m.insert(k.extract::<String>()?, v.extract::<String>()?);
            }
            Ok(m)
        }
        None => Ok(HashMap::new()),
    }
}

/// Extract tensors from a Python dict: {"name": (shape_list, dtype_str, bytes)}
fn extract_tensors(tensors: &PyDict) -> PyResult<Vec<TensorData>> {
    let mut result = Vec::new();
    for (key, value) in tensors.iter() {
        let name: String = key.extract()?;
        let tuple = value.downcast::<pyo3::types::PyTuple>()?;
        let shape: Vec<usize> = tuple.get_item(0)?.extract()?;
        let dtype: String = tuple.get_item(1)?.extract()?;
        let data: &[u8] = tuple.get_item(2)?.extract()?;
        result.push(TensorData {
            name,
            shape,
            dtype,
            data: data.to_vec(),
        });
    }
    Ok(result)
}

/// High-performance checkpoint manager for ML training.
///
/// DECK-inspired architecture with per-tensor delta tracking,
/// rank-aware saves, hierarchical merging, and lineage graph.
///
/// Example (single rank):
///     import revolver
///     mgr = revolver.RevolverManager("./checkpoints")
///     snap_id = mgr.save_tensors(step=1000, tensors={...})
///     snap_id, tensors = mgr.load_latest()
///
/// Example (multi-rank FSDP):
///     mgr = revolver.RevolverManager("./checkpoints", world_size=8, rank=dist.get_rank())
///     snap_id = mgr.create_snapshot(step=1000)
///     mgr.save_rank(snap_id, tensors={...})
///     # barrier
///     if rank == 0: mgr.finalize_snapshot(snap_id)
#[pyclass]
pub struct RevolverManager {
    inner: Coordinator,
}

#[pymethods]
impl RevolverManager {
    #[new]
    #[pyo3(signature = (
        storage_root = "./checkpoints",
        compression_level = 3,
        max_full_snapshots = 5,
        max_deltas_per_full = 10,
        full_every_steps = 5000,
        delta_threshold = 0.5,
        world_size = 1,
        rank = 0,
        merge_stride = 0,
        merge_max_chain = 10,
        rollback_interval_steps = 10000,
        max_rollback_snapshots = 3,
        s3_bucket = None,
        s3_region = "us-east-1",
        s3_prefix = "",
        s3_endpoint = None,
        s3_access_key = None,
        s3_secret_key = None,
        s3_path_style = false,
        sync_every_n_saves = 100,
    ))]
    fn new(
        storage_root: &str,
        compression_level: i32,
        max_full_snapshots: usize,
        max_deltas_per_full: usize,
        full_every_steps: u64,
        delta_threshold: f64,
        world_size: u32,
        rank: u32,
        merge_stride: usize,
        merge_max_chain: usize,
        rollback_interval_steps: u64,
        max_rollback_snapshots: usize,
        s3_bucket: Option<&str>,
        s3_region: &str,
        s3_prefix: &str,
        s3_endpoint: Option<&str>,
        s3_access_key: Option<&str>,
        s3_secret_key: Option<&str>,
        s3_path_style: bool,
        sync_every_n_saves: u64,
    ) -> PyResult<Self> {
        let compression = if compression_level == 0 {
            CompressionAlgo::None
        } else {
            CompressionAlgo::Zstd { level: compression_level }
        };

        let config = CoordinatorConfig {
            world_size,
            rank,
            compression,
            retention: RetentionPolicy {
                max_full_snapshots,
                max_deltas_per_full,
                full_snapshot_every_steps: full_every_steps,
            },
            lineage: LineageConfig {
                rollback_interval_steps,
                max_rollback_snapshots,
            },
            delta_threshold,
            merger: if merge_stride > 0 {
                Some(MergerConfig { stride: merge_stride, max_chain_depth: merge_max_chain })
            } else {
                None
            },
            remote_storage: None, // Set below if S3 is configured
            remote_sync: None,    // Set below if S3 is configured
        };

        // Storage setup:
        // - If S3 is configured: local is primary (page-aligned), S3 is remote (batched sync)
        // - If no S3: local only
        let (storage, mut config) = if let Some(bucket) = s3_bucket {
            let ak = s3_access_key.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("s3_access_key required")
            })?;
            let sk = s3_secret_key.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("s3_secret_key required")
            })?;

            let remote: Arc<dyn crate::storage::StorageBackend> =
                Arc::new(S3Storage::new(S3Config {
                    bucket: bucket.into(),
                    prefix: s3_prefix.into(),
                    region: s3_region.into(),
                    endpoint: s3_endpoint.map(|s| s.into()),
                    access_key: ak.into(),
                    secret_key: sk.into(),
                    path_style: s3_path_style,
                    timeout_secs: 30,
                })?);

            // Local is primary, S3 is synced in batches
            let local: Arc<dyn crate::storage::StorageBackend> =
                Arc::new(LocalStorage::new(storage_root)?);

            let mut cfg = config;
            cfg.remote_storage = Some(Arc::clone(&remote));
            cfg.remote_sync = Some(RemoteSyncConfig {
                sync_every_n_saves: sync_every_n_saves,
            });
            (local, cfg)
        } else {
            let local: Arc<dyn crate::storage::StorageBackend> =
                Arc::new(LocalStorage::new(storage_root)?);
            (local, config)
        };

        let inner = Coordinator::new(storage, config)?;
        Ok(RevolverManager { inner })
    }

    /// Save a checkpoint (single-rank mode).
    ///
    /// Args:
    ///     step: Training step.
    ///     tensors: Dict mapping tensor_name → (shape, dtype, bytes).
    ///     metadata: Optional dict of string metadata.
    ///
    /// Returns:
    ///     Snapshot UUID string.
    #[pyo3(signature = (step, tensors, metadata = None))]
    fn save_tensors(
        &self,
        step: u64,
        tensors: &PyDict,
        metadata: Option<&PyDict>,
    ) -> PyResult<String> {
        let tensor_data = extract_tensors(tensors)?;
        let meta = extract_metadata(metadata)?;
        let id = self.inner.save(step, tensor_data, meta)?;
        Ok(id.to_string())
    }

    /// Create a new snapshot (multi-rank: rank 0 only).
    #[pyo3(signature = (step, metadata = None))]
    fn create_snapshot(
        &self,
        step: u64,
        metadata: Option<&PyDict>,
    ) -> PyResult<String> {
        let meta = extract_metadata(metadata)?;
        let id = self.inner.create_snapshot(step, meta)?;
        Ok(id.to_string())
    }

    /// Save this rank's tensors into an existing snapshot (multi-rank).
    fn save_rank(&self, snap_id: &str, tensors: &PyDict) -> PyResult<()> {
        let uuid = uuid::Uuid::parse_str(snap_id)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let tensor_data = extract_tensors(tensors)?;
        self.inner.save_rank(uuid, tensor_data)?;
        Ok(())
    }

    /// Finalize a snapshot (multi-rank: rank 0 only).
    fn finalize_snapshot(&self, snap_id: &str) -> PyResult<()> {
        let uuid = uuid::Uuid::parse_str(snap_id)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.inner.finalize_snapshot(uuid)?;
        Ok(())
    }

    /// Load a snapshot. Returns dict of tensor_name → bytes.
    fn load<'py>(&self, py: Python<'py>, snap_id: &str) -> PyResult<&'py PyDict> {
        let uuid = uuid::Uuid::parse_str(snap_id)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let tensors = self.inner.load(uuid)?;
        let dict = PyDict::new(py);
        for (name, data) in tensors {
            dict.set_item(name, PyBytes::new(py, &data))?;
        }
        Ok(dict)
    }

    /// Load the latest snapshot.
    fn load_latest<'py>(&self, py: Python<'py>) -> PyResult<(&'py pyo3::types::PyString, &'py PyDict)> {
        let (id, tensors) = self.inner.load_latest()?;
        let dict = PyDict::new(py);
        for (name, data) in tensors {
            dict.set_item(name, PyBytes::new(py, &data))?;
        }
        Ok((pyo3::types::PyString::new(py, &id.to_string()), dict))
    }

    /// List all finalized snapshots.
    fn list_snapshots<'py>(&self, py: Python<'py>) -> PyResult<Vec<&'py PyDict>> {
        let snaps = self.inner.list_snapshots();
        let mut result = Vec::new();
        for s in snaps {
            let d = PyDict::new(py);
            d.set_item("step", s.step)?;
            d.set_item("id", s.id.to_string())?;
            d.set_item("is_delta", s.is_delta)?;
            d.set_item("total_compressed", s.total_compressed)?;
            d.set_item("total_raw", s.total_raw)?;
            d.set_item("skipped_tensors", s.skipped_tensors)?;
            d.set_item("delta_tensors", s.delta_tensors)?;
            d.set_item("full_tensors", s.full_tensors)?;
            d.set_item("ranks", s.ranks)?;
            let meta = PyDict::new(py);
            for (k, v) in &s.metadata {
                meta.set_item(k, v)?;
            }
            d.set_item("metadata", meta)?;
            result.push(d);
        }
        Ok(result)
    }

    /// Force merge all pending deltas into a full checkpoint.
    fn merge_now(&self) {
        self.inner.merge_now();
    }

    /// Force sync all local data to remote storage immediately.
    fn sync_now(&self) {
        self.inner.sync_now();
    }
}

#[pymodule]
fn revolver(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RevolverManager>()?;
    m.add("__version__", "1.0.0")?;
    Ok(())
}
