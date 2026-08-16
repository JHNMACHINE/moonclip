use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyString, PyTuple};
use std::collections::HashMap;
use std::sync::Arc;

use crate::cast::DType;
use crate::coordinator::{Coordinator, CoordinatorConfig};
use crate::manifest::{CompressionAlgo, LineageConfig, RetentionPolicy};
use crate::merger::MergerConfig;
use crate::remote_sync::RemoteSyncConfig;
use crate::s3::{S3Config, S3Storage};
use crate::storage::LocalStorage;
use crate::tensor::TensorData;

fn extract_metadata(metadata: Option<Bound<'_, PyDict>>) -> PyResult<HashMap<String, String>> {
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

fn get_element_size(dtype: &str) -> PyResult<usize> {
    match dtype {
        "torch.float32" | "torch.int32" => Ok(4),
        "torch.float64" | "torch.int64" => Ok(8),
        "torch.float16" | "torch.bfloat16" => Ok(2),
        "torch.int16" | "torch.uint16" => Ok(2),
        "torch.int8" | "torch.uint8" => Ok(1),
        "torch.bool" => Ok(1),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported dtype {}",
            dtype
        ))),
    }
}

/// Tensor data captured under the GIL. Anything large is kept as a raw
/// pointer + length so the actual byte copy happens in parallel with the
/// GIL released, instead of serially inside the collect loop.
enum PendingBytes {
    Owned(Vec<u8>),
    Borrowed { ptr: usize, len: usize },
}

struct PendingTensor {
    name: String,
    shape: Vec<usize>,
    dtype: String,
    bytes: PendingBytes,
}

/// Borrow a Python buffer without copying it, if that is safe to do.
///
/// Only `bytes` qualifies: it is immutable, so nothing can resize or free it
/// while the GIL is released, and the caller's dict holds it alive for the
/// whole call. `bytearray` and everything else fall back to a copy taken here
/// under the GIL — a mutable buffer could be reallocated out from under the
/// borrowed pointer.
fn borrow_or_copy(value: &Bound<'_, PyAny>) -> PyResult<PendingBytes> {
    if let Ok(b) = value.cast::<PyBytes>() {
        let slice = b.as_bytes();
        return Ok(PendingBytes::Borrowed {
            ptr: slice.as_ptr() as usize,
            len: slice.len(),
        });
    }
    Ok(PendingBytes::Owned(value.extract::<Vec<u8>>()?))
}

fn collect_tensors(tensors: &Bound<'_, PyDict>) -> PyResult<Vec<PendingTensor>> {
    let mut result = Vec::new();
    for (key, value) in tensors.iter() {
        let name: String = key.extract()?;
        // Vecchio formato: tupla (shape, dtype, bytes)
        if let Ok(tuple) = value.cast::<PyTuple>() {
            let shape: Vec<usize> = tuple.get_item(0)?.extract()?;
            let dtype: String = tuple.get_item(1)?.extract()?;
            let bytes = borrow_or_copy(&tuple.get_item(2)?)?;
            result.push(PendingTensor {
                name,
                shape,
                dtype,
                bytes,
            });
        }
        // Nuovo formato: tensore PyTorch direttamente. `value` is already a
        // Bound<PyAny>, so the presence of `data_ptr` is the only test that
        // distinguishes a tensor here.
        else if value.hasattr("data_ptr")? {
            let shape: Vec<usize> = value.getattr("shape")?.extract()?;
            let dtype_full = value.getattr("dtype")?.str()?.to_string();
            let dtype = dtype_full
                .strip_prefix("torch.")
                .unwrap_or(&dtype_full)
                .to_string();

            // Both checks guard the raw read below, and neither is paranoia:
            // `data_ptr()` on a CUDA tensor is a device address, and reading it
            // as host memory is a segfault at best. A non-contiguous tensor's
            // elements are not the `numel * element_size` bytes that follow the
            // pointer, so the read would silently store the wrong data — and
            // could run past the end of the storage. Refuse both; the caller
            // gets a message naming the fix.
            let device = value.getattr("device")?.getattr("type")?.extract::<String>()?;
            if device != "cpu" {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "tensor '{}' is on device '{}'; move it to CPU first \
                     (t.detach().cpu()) — Moonclip reads host memory directly",
                    name, device
                )));
            }
            if !value.call_method0("is_contiguous")?.extract::<bool>()? {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "tensor '{}' is not contiguous; call .contiguous() first — \
                     Moonclip reads the bytes that follow data_ptr()",
                    name
                )));
            }

            let numel: usize = value.call_method0("numel")?.extract()?;
            let data_ptr: usize = value.call_method0("data_ptr")?.extract()?;
            let element_size = get_element_size(&format!("torch.{}", dtype))?;
            let nbytes = numel * element_size;
            result.push(PendingTensor {
                name,
                shape,
                dtype,
                bytes: PendingBytes::Borrowed {
                    ptr: data_ptr,
                    len: nbytes,
                },
            });
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Value for '{}' is not a supported type (must be tuple or torch.Tensor)",
                name
            )));
        }
    }
    Ok(result)
}

/// Copy pending tensor bytes into owned buffers, in parallel.
///
/// This is the shadow copy, and it is the only full copy of the state that
/// the save path takes: it exists so the training loop can mutate its tensors
/// again the moment `save` returns, while the writer works from these buffers.
///
/// SAFETY: borrowed pointers come either from CPU-contiguous torch tensors
/// (checked in `collect_tensors`) or from immutable `bytes` objects. Both are
/// held alive by the caller's dict for the whole duration of the call, and
/// this runs — with the GIL released, so the training thread is still blocked
/// inside the call — strictly within that window, reading and never writing.
fn materialize_tensors(pending: Vec<PendingTensor>) -> Vec<TensorData> {
    use rayon::prelude::*;

    pending
        .into_par_iter()
        .map(|p| {
            let data = match p.bytes {
                PendingBytes::Owned(v) => v,
                // An empty tensor's `data_ptr()` may be null, and
                // `from_raw_parts` requires a non-null pointer even for a
                // zero length.
                PendingBytes::Borrowed { len: 0, .. } => Vec::new(),
                PendingBytes::Borrowed { ptr, len } => {
                    unsafe { std::slice::from_raw_parts(ptr as *const u8, len) }.to_vec()
                }
            };
            TensorData {
                name: p.name,
                shape: p.shape,
                dtype: p.dtype,
                data,
            }
        })
        .collect()
}

/// High-performance checkpoint manager for ML training.
#[pyclass]
pub struct MoonclipManager {
    inner: Coordinator,
}

#[pymethods]
impl MoonclipManager {
    #[new]
    #[pyo3(signature = (
        storage_root = "./checkpoints",
        compression_level = 3,
        max_full_snapshots = 5,
        max_deltas_per_full = 10,
        full_every_steps = 5000,
        delta_max_ratio = 0.95,
        world_size = 1,
        rank = 0,
        merge_stride = 0,
        merge_max_chain = 10,
        rollback_interval_steps = 10000,
        max_rollback_snapshots = 3,
        max_total_snapshots = None,
        s3_bucket = None,
        s3_region = "us-east-1",
        s3_prefix = "",
        s3_endpoint = None,
        s3_access_key = None,
        s3_secret_key = None,
        s3_path_style = false,
        sync_every_n_saves = 100,
        save_dtype = "none",
        async_save = true,
        keep_base_in_memory = true,
    ))]
    fn new(
        storage_root: &str,
        compression_level: i32,
        max_full_snapshots: usize,
        max_deltas_per_full: usize,
        full_every_steps: u64,
        delta_max_ratio: f64,
        world_size: u32,
        rank: u32,
        merge_stride: usize,
        merge_max_chain: usize,
        rollback_interval_steps: u64,
        max_rollback_snapshots: usize,
        max_total_snapshots: Option<usize>,
        s3_bucket: Option<&str>,
        s3_region: &str,
        s3_prefix: &str,
        s3_endpoint: Option<&str>,
        s3_access_key: Option<&str>,
        s3_secret_key: Option<&str>,
        s3_path_style: bool,
        sync_every_n_saves: u64,
        save_dtype: &str,
        async_save: bool,
        keep_base_in_memory: bool,
    ) -> PyResult<Self> {
        let compression = if compression_level == 0 {
            CompressionAlgo::None
        } else {
            CompressionAlgo::Zstd {
                level: compression_level,
            }
        };

        let mut config = CoordinatorConfig {
            world_size,
            rank,
            compression,
            retention: RetentionPolicy {
                max_full_snapshots,
                max_deltas_per_full,
                full_snapshot_every_steps: full_every_steps,
                max_total_snapshots,
            },
            lineage: LineageConfig {
                rollback_interval_steps,
                max_rollback_snapshots,
            },
            delta_max_ratio,
            merger: if merge_stride > 0 {
                Some(MergerConfig {
                    stride: merge_stride,
                    max_chain_depth: merge_max_chain,
                })
            } else {
                None
            },
            remote_storage: None,
            remote_sync: None,
            save_dtype: DType::from_str(save_dtype),
            async_save,
            keep_base_in_memory,
        };

        let storage: Arc<dyn crate::storage::StorageBackend> = if let Some(bucket) = s3_bucket {
            let ak = s3_access_key.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "s3_access_key required when s3_bucket is set",
                )
            })?;
            let sk = s3_secret_key.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "s3_secret_key required when s3_bucket is set",
                )
            })?;

            let remote: Arc<dyn crate::storage::StorageBackend> = Arc::new(S3Storage::new(
                S3Config {
                    bucket: bucket.into(),
                    prefix: s3_prefix.into(),
                    region: s3_region.into(),
                    endpoint: s3_endpoint.map(|s| s.into()),
                    access_key: ak.into(),
                    secret_key: sk.into(),
                    path_style: s3_path_style,
                    timeout_secs: 30,
                }
                .with_auto_path_style(),
            )?);

            let local: Arc<dyn crate::storage::StorageBackend> =
                Arc::new(LocalStorage::new(storage_root)?);

            config.remote_storage = Some(Arc::clone(&remote));
            config.remote_sync = Some(RemoteSyncConfig { sync_every_n_saves });
            local
        } else {
            Arc::new(LocalStorage::new(storage_root)?)
        };

        let inner = Coordinator::new(storage, config)?;
        Ok(MoonclipManager { inner })
    }

    /// Save a checkpoint (single-rank mode).
    #[pyo3(signature = (step, tensors, metadata = None))]
    fn save_tensors(
        &self,
        py: Python<'_>,
        step: u64,
        tensors: Bound<'_, PyDict>,
        metadata: Option<Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        let pending = collect_tensors(&tensors)?;
        let meta = extract_metadata(metadata)?;
        let id = py.detach(|| {
            let tensor_data = materialize_tensors(pending);
            self.inner.save(step, tensor_data, meta)
        })?;
        Ok(id.to_string())
    }

    /// Create a new snapshot (multi-rank: rank 0 only).
    #[pyo3(signature = (step, metadata = None))]
    fn create_snapshot(&self, step: u64, metadata: Option<Bound<'_, PyDict>>) -> PyResult<String> {
        let meta = extract_metadata(metadata)?;
        let id = self.inner.create_snapshot(step, meta)?;
        Ok(id.to_string())
    }

    /// Save this rank's tensors into an existing snapshot (multi-rank).
    fn save_rank(&self, py: Python<'_>, snap_id: &str, tensors: Bound<'_, PyDict>) -> PyResult<()> {
        let uuid = uuid::Uuid::parse_str(snap_id)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let pending = collect_tensors(&tensors)?;
        py.detach(|| {
            let tensor_data = materialize_tensors(pending);
            self.inner.save_rank(uuid, tensor_data)
        })?;
        Ok(())
    }

    /// Finalize a snapshot (multi-rank: rank 0 only).
    fn finalize_snapshot(&self, snap_id: &str) -> PyResult<()> {
        let uuid = uuid::Uuid::parse_str(snap_id)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.inner.finalize_snapshot(uuid)?;
        Ok(())
    }

    /// Block until any in-flight background save completes.
    /// Raises if the background save failed.
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.flush())?;
        Ok(())
    }

    /// Load a snapshot. Returns dict of tensor_name → bytearray.
    fn load<'py>(&self, py: Python<'py>, snap_id: &str) -> PyResult<Bound<'py, PyDict>> {
        let uuid = uuid::Uuid::parse_str(snap_id)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let tensors = py.detach(|| self.inner.load(uuid))?;
        let dict = PyDict::new(py);
        for (name, data) in tensors {
            dict.set_item(name, PyByteArray::new(py, &data))?;
        }
        Ok(dict)
    }

    /// Load the latest snapshot.
    fn load_latest<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyString>, Bound<'py, PyDict>)> {
        let (id, tensors) = py.detach(|| self.inner.load_latest())?;
        let dict = PyDict::new(py);
        for (name, data) in tensors {
            dict.set_item(name, PyByteArray::new(py, &data))?;
        }
        let id_str = PyString::new(py, &id.to_string());
        Ok((id_str, dict))
    }

    /// List all finalized snapshots.
    fn list_snapshots<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let snaps = py.detach(|| self.inner.list_snapshots());
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

    /// Force merge all pending deltas.
    fn merge_now(&self, py: Python<'_>) {
        py.detach(|| self.inner.merge_now());
    }

    /// Force sync to remote storage.
    fn sync_now(&self, py: Python<'_>) {
        py.detach(|| self.inner.sync_now());
    }
}

#[pymodule]
fn moonclip(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<MoonclipManager>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
