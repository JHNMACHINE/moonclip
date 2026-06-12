use std::collections::HashMap;

use crate::cast::{self, DType, is_castable_float};
use crate::compression;
use crate::delta;
use crate::error::{Result, RevolverError};
use crate::hash::hash_hex;
use crate::manifest::{CompressionAlgo, TensorEntry, TensorStorage};
use crate::storage::StorageBackend;

/// A tensor ready to be saved: name + raw bytes + metadata.
/// This is the Rust-side representation of one tensor from a PyTorch state_dict.
#[derive(Debug, Clone)]
pub struct TensorData {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub data: Vec<u8>,
}

/// Result of processing a single tensor for saving.
pub struct ProcessedTensor {
    pub entry: TensorEntry,
    /// The data to write to storage (None if skipped).
    pub write_data: Option<Vec<u8>>,
}

/// Pre-loaded base snapshot data for delta comparison.
///
/// Instead of each parallel worker hitting disk independently,
/// the coordinator reads the base pack file once and shares it
/// immutably across rayon threads.
pub struct BaseCache {
    /// Raw bytes of the base snapshot's pack file.
    /// Empty if no pack file (legacy per-file storage).
    pub pack_data: Vec<u8>,
    /// Tensor entries from the base snapshot, keyed by name.
    pub entries: HashMap<String, TensorEntry>,
    /// Compression algo of the base snapshot.
    pub compression: CompressionAlgo,
    /// For legacy mode: individual file data loaded ahead of time.
    /// Keyed by filename.
    pub legacy_files: HashMap<String, Vec<u8>>,
}

impl BaseCache {
    /// Extract compressed bytes for a tensor from the pack file or legacy files.
    fn get_compressed(&self, entry: &TensorEntry) -> Option<Vec<u8>> {
        if !self.pack_data.is_empty() {
            // Pack mode: slice from pack buffer
            let start = entry.offset as usize;
            let end = start + entry.compressed_size as usize;
            if end <= self.pack_data.len() {
                Some(self.pack_data[start..end].to_vec())
            } else {
                None
            }
        } else if let Some(ref filename) = entry.filename {
            // Legacy mode: from pre-loaded files
            self.legacy_files.get(filename).map(|data| {
                let expected = entry.compressed_size as usize;
                if expected > 0 && data.len() > expected {
                    data[..expected].to_vec()
                } else {
                    data.clone()
                }
            })
        } else {
            None
        }
    }

    /// Decompress a base tensor to raw bytes. Returns None for delta/skipped bases.
    fn decompress_tensor(&self, entry: &TensorEntry) -> Result<Option<Vec<u8>>> {
        if entry.storage != TensorStorage::Full {
            return Ok(None); // Can't delta-chain against non-full bases
        }
        match self.get_compressed(entry) {
            Some(compressed) => {
                let raw = compression::decompress(&compressed, &self.compression)?;
                Ok(Some(raw))
            }
            None => Ok(None),
        }
    }
}

/// Process a tensor for saving: optionally cast dtype, compare with base,
/// decide skip/delta/full, compress, and return the entry + data to write.
///
/// `base_cache`: pre-loaded base snapshot data (None for first save).
pub fn process_tensor(
    tensor: &TensorData,
    base_cache: Option<&BaseCache>,
    compression: &CompressionAlgo,
    delta_threshold: f64,
    save_dtype: &DType,
) -> Result<ProcessedTensor> {
    use std::borrow::Cow;

    // Cast if requested and applicable — otherwise borrow without cloning
    let (working_data, working_dtype): (Cow<[u8]>, Cow<str>) =
        if *save_dtype != DType::None && is_castable_float(&tensor.dtype) {
            let (d, t) = cast::cast_tensor(&tensor.data, &tensor.dtype, save_dtype)?;
            (Cow::Owned(d), Cow::Owned(t))
        } else {
            (Cow::Borrowed(&tensor.data), Cow::Borrowed(&tensor.dtype))
        };

    let raw_hash = hash_hex(&working_data);

    // Determine original_dtype field (set only if we actually cast)
    let orig_dtype = if *working_dtype != tensor.dtype {
        Some(tensor.dtype.clone())
    } else {
        None
    };

    // Check if we can skip (identical to base)
    if let Some(cache) = base_cache {
        if let Some(base_entry) = cache.entries.get(&tensor.name) {
            if base_entry.sha256_raw == raw_hash {
                return Ok(ProcessedTensor {
                    entry: TensorEntry {
                        name: tensor.name.clone(),
                        shape: tensor.shape.clone(),
                        dtype: working_dtype.to_string(),
                        original_dtype: orig_dtype.clone(),
                        storage: TensorStorage::Skipped,
                        filename: None,
                        offset: 0,
                        compressed_size: 0,
                        raw_size: working_data.len() as u64,
                        sha256_raw: raw_hash,
                        sha256_compressed: None,
                    },
                    write_data: None,
                });
            }

            // Try delta encoding (only against full bases with matching size)
            if base_entry.raw_size == working_data.len() as u64
                && base_entry.storage == TensorStorage::Full
            {
                // Decompress base from cache — no disk I/O here
                if let Some(base_raw) = cache.decompress_tensor(base_entry)? {
                    if let Some(xor_delta) = delta::compute_delta(&base_raw, &working_data) {
                        let density = delta::delta_density(&xor_delta);
                        if density < delta_threshold {
                            let compressed = compression::compress(&xor_delta, compression)?;

                            return Ok(ProcessedTensor {
                                entry: TensorEntry {
                                    name: tensor.name.clone(),
                                    shape: tensor.shape.clone(),
                                    dtype: working_dtype.to_string(),
                                    original_dtype: orig_dtype.clone(),
                                    storage: TensorStorage::DeltaXor,
                                    filename: None, // Set by coordinator after packing
                                    offset: 0,      // Set by coordinator after packing
                                    compressed_size: compressed.len() as u64,
                                    raw_size: working_data.len() as u64,
                                    sha256_raw: raw_hash,
                                    sha256_compressed: None, // Set after packing
                                },
                                write_data: Some(compressed),
                            });
                        }
                    }
                }
            }
        }
    }

    // Full save
    make_full_entry(tensor, &working_data, &working_dtype, &orig_dtype, &raw_hash, compression)
}

fn make_full_entry(
    tensor: &TensorData,
    working_data: &[u8],
    working_dtype: &str,
    original_dtype: &Option<String>,
    raw_hash: &str,
    compression: &CompressionAlgo,
) -> Result<ProcessedTensor> {
    let compressed = compression::compress(working_data, compression)?;

    Ok(ProcessedTensor {
        entry: TensorEntry {
            name: tensor.name.clone(),
            shape: tensor.shape.clone(),
            dtype: working_dtype.to_string(),
            original_dtype: original_dtype.clone(),
            storage: TensorStorage::Full,
            filename: None, // Set by coordinator after packing
            offset: 0,      // Set by coordinator after packing
            compressed_size: compressed.len() as u64,
            raw_size: working_data.len() as u64,
            sha256_raw: raw_hash.to_string(),
            sha256_compressed: None, // Set after packing
        },
        write_data: Some(compressed),
    })
}

/// Load a tensor's raw bytes, resolving delta chains if needed.
/// If the tensor was saved with a cast (e.g. fp32→bf16), it is
/// automatically uncast back to the original dtype.
///
/// `pack_data`: pre-loaded pack file bytes (if available). If None,
/// falls back to reading individual files from storage.
pub fn load_tensor(
    entry: &TensorEntry,
    snap_base_id: Option<uuid::Uuid>,
    storage: &dyn StorageBackend,
    compression: &CompressionAlgo,
    pack_data: Option<&[u8]>,
    find_base_entry: &dyn Fn(uuid::Uuid, &str) -> Result<(TensorEntry, CompressionAlgo, Option<Vec<u8>>)>,
) -> Result<Vec<u8>> {
    let raw = load_tensor_raw(entry, snap_base_id, storage, compression, pack_data, find_base_entry)?;

    // Uncast if needed (e.g. bf16 on disk → fp32 for training)
    if let Some(ref orig_dtype) = entry.original_dtype {
        if orig_dtype != &entry.dtype {
            return cast::uncast_tensor(&raw, &entry.dtype, orig_dtype);
        }
    }

    Ok(raw)
}

/// Extract compressed bytes for a tensor, using pack_data if available,
/// otherwise falling back to individual file reads.
fn extract_compressed(
    entry: &TensorEntry,
    storage: &dyn StorageBackend,
    pack_data: Option<&[u8]>,
) -> Result<Vec<u8>> {
    // Try pack file first
    if let Some(pack) = pack_data {
        let start = entry.offset as usize;
        let end = start + entry.compressed_size as usize;
        if end <= pack.len() {
            return Ok(pack[start..end].to_vec());
        }
    }

    // Fallback to individual file
    let filename = entry.filename.as_ref().ok_or_else(|| {
        RevolverError::NotFound(format!("No filename for tensor '{}'", entry.name))
    })?;
    let mut data = storage.get(filename)?;
    let expected_len = entry.compressed_size as usize;
    if expected_len > 0 && data.len() > expected_len {
        data.truncate(expected_len);
    }
    Ok(data)
}

/// Load raw bytes without uncasting.
fn load_tensor_raw(
    entry: &TensorEntry,
    snap_base_id: Option<uuid::Uuid>,
    storage: &dyn StorageBackend,
    compression: &CompressionAlgo,
    pack_data: Option<&[u8]>,
    find_base_entry: &dyn Fn(uuid::Uuid, &str) -> Result<(TensorEntry, CompressionAlgo, Option<Vec<u8>>)>,
) -> Result<Vec<u8>> {
    match entry.storage {
        TensorStorage::Skipped => {
            // Load from base snapshot
            let base_id = snap_base_id.ok_or_else(|| {
                RevolverError::Delta(format!(
                    "Tensor '{}' is skipped but snapshot has no base",
                    entry.name
                ))
            })?;
            let (base_entry, base_compression, base_pack) = find_base_entry(base_id, &entry.name)?;
            load_tensor_raw(&base_entry, None, storage, &base_compression, base_pack.as_deref(), find_base_entry)
        }
        TensorStorage::Full => {
            let compressed = extract_compressed(entry, storage, pack_data)?;

            // Verify compressed integrity (cheap with xxHash3)
            if let Some(ref expected) = entry.sha256_compressed {
                let actual = hash_hex(&compressed);
                if &actual != expected {
                    return Err(RevolverError::IntegrityError {
                        expected: expected.clone(),
                        actual,
                    });
                }
            }

            compression::decompress(&compressed, compression)
        }
        TensorStorage::DeltaXor => {
            let compressed = extract_compressed(entry, storage, pack_data)?;
            let delta_data = compression::decompress(&compressed, compression)?;

            // Load base tensor
            let base_id = snap_base_id.ok_or_else(|| {
                RevolverError::Delta(format!(
                    "Delta tensor '{}' has no base snapshot",
                    entry.name
                ))
            })?;
            let (base_entry, base_compression, base_pack) = find_base_entry(base_id, &entry.name)?;
            let base_raw =
                load_tensor_raw(&base_entry, None, storage, &base_compression, base_pack.as_deref(), find_base_entry)?;

            // Apply XOR delta
            let raw = delta::apply_delta(&base_raw, &delta_data)?;

            // Verify reconstructed tensor integrity
            let actual = hash_hex(&raw);
            if actual != entry.sha256_raw {
                return Err(RevolverError::IntegrityError {
                    expected: entry.sha256_raw.clone(),
                    actual,
                });
            }

            Ok(raw)
        }
    }
}

/// Process multiple tensors in parallel using rayon.
/// Returns entries and data to write, preserving order.
pub fn process_tensors_parallel(
    tensors: &[TensorData],
    base_cache: Option<&BaseCache>,
    compression: &CompressionAlgo,
    delta_threshold: f64,
    save_dtype: &DType,
) -> Result<Vec<ProcessedTensor>> {
    use rayon::prelude::*;

    tensors
        .par_iter()
        .map(|tensor| {
            process_tensor(
                tensor,
                base_cache,
                compression,
                delta_threshold,
                save_dtype,
            )
        })
        .collect()
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::cast::DType;
    use crate::storage::LocalStorage;

    fn make_base_cache(
        storage: &LocalStorage,
        entries: Vec<TensorEntry>,
        compression: &CompressionAlgo,
    ) -> BaseCache {
        let mut legacy_files = HashMap::new();
        for e in &entries {
            if let Some(ref filename) = e.filename {
                if let Ok(data) = storage.get(filename) {
                    legacy_files.insert(filename.clone(), data);
                }
            }
        }
        let entry_map = entries.into_iter().map(|e| (e.name.clone(), e)).collect();
        BaseCache {
            pack_data: Vec::new(),
            entries: entry_map,
            compression: compression.clone(),
            legacy_files,
        }
    }

    #[test]
    fn skip_identical_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let storage = LocalStorage::new(dir.path()).unwrap();
        let compression = CompressionAlgo::Zstd { level: 1 };

        // Save a "base" tensor
        let data = vec![42u8; 8192];
        let raw_hash = hash_hex(&data);
        let compressed = compression::compress(&data, &compression).unwrap();
        let base_filename = "snapshots/base/rank_0/test_tensor.bin".to_string();
        storage.put(&base_filename, &compressed).unwrap();

        let base_entry = TensorEntry {
            name: "test_tensor".into(),
            shape: vec![8192],
            dtype: "uint8".into(),
            storage: TensorStorage::Full,
            filename: Some(base_filename),
            offset: 0,
            compressed_size: compressed.len() as u64,
            raw_size: 8192,
            sha256_raw: raw_hash.clone(),
            sha256_compressed: None,
            original_dtype: None,
        };

        let cache = make_base_cache(&storage, vec![base_entry], &compression);

        // Process same tensor again → should be Skipped
        let tensor = TensorData {
            name: "test_tensor".into(),
            shape: vec![8192],
            dtype: "uint8".into(),
            data: data.clone(),
        };

        let result = process_tensor(
            &tensor,
            Some(&cache),
            &compression,
            0.5,
            &DType::None,
        )
        .unwrap();

        assert_eq!(result.entry.storage, TensorStorage::Skipped);
        assert!(result.write_data.is_none());
    }

    #[test]
    fn delta_on_small_change() {
        let dir = tempfile::tempdir().unwrap();
        let storage = LocalStorage::new(dir.path()).unwrap();
        let compression = CompressionAlgo::Zstd { level: 1 };

        let data_v1 = vec![0u8; 50_000];
        let compressed_v1 = compression::compress(&data_v1, &compression).unwrap();
        let base_filename = "snapshots/base/rank_0/big_tensor.bin".to_string();
        storage.put(&base_filename, &compressed_v1).unwrap();

        let base_entry = TensorEntry {
            name: "big_tensor".into(),
            shape: vec![50_000],
            dtype: "uint8".into(),
            storage: TensorStorage::Full,
            filename: Some(base_filename),
            offset: 0,
            compressed_size: compressed_v1.len() as u64,
            raw_size: 50_000,
            sha256_raw: hash_hex(&data_v1),
            sha256_compressed: None,
            original_dtype: None,
        };

        let cache = make_base_cache(&storage, vec![base_entry], &compression);

        // Change 2 bytes
        let mut data_v2 = data_v1.clone();
        data_v2[0] = 1;
        data_v2[25000] = 2;

        let tensor = TensorData {
            name: "big_tensor".into(),
            shape: vec![50_000],
            dtype: "uint8".into(),
            data: data_v2,
        };

        let result = process_tensor(
            &tensor,
            Some(&cache),
            &compression,
            0.5,
            &DType::None,
        )
        .unwrap();

        assert_eq!(result.entry.storage, TensorStorage::DeltaXor);
        assert!(result.write_data.is_some());
        assert!(result.entry.compressed_size < 1000);
    }

    #[test]
    fn full_save_no_base() {
        let compression = CompressionAlgo::Zstd { level: 1 };

        let tensor = TensorData {
            name: "new_tensor".into(),
            shape: vec![1024],
            dtype: "float32".into(),
            data: vec![7u8; 4096],
        };

        let result = process_tensor(
            &tensor,
            None,
            &compression,
            0.5,
            &DType::None,
        )
        .unwrap();

        assert_eq!(result.entry.storage, TensorStorage::Full);
        assert!(result.write_data.is_some());
    }
}
