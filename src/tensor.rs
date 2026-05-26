use std::collections::HashMap;

use crate::compression;
use crate::delta;
use crate::error::{Result, RevolverError};
use crate::hash::sha256_hex;
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

/// Process a tensor for saving: compare with base, decide skip/delta/full,
/// compress, and return the entry + data to write.
///
/// This is the core of DECK's "zero-cost delta tracking" adapted to dense models:
/// - If sha256 matches base → skip (zero I/O)
/// - If same size and delta density < threshold → XOR delta + compress
/// - Otherwise → full save + compress
pub fn process_tensor(
    tensor: &TensorData,
    base_entry: Option<&TensorEntry>,
    base_storage: &dyn StorageBackend,
    compression: &CompressionAlgo,
    delta_threshold: f64,
    snap_dir: &str,
    rank: u32,
) -> Result<ProcessedTensor> {
    let raw_hash = sha256_hex(&tensor.data);

    // Check if we can skip (identical to base)
    if let Some(base) = base_entry {
        if base.sha256_raw == raw_hash {
            return Ok(ProcessedTensor {
                entry: TensorEntry {
                    name: tensor.name.clone(),
                    shape: tensor.shape.clone(),
                    dtype: tensor.dtype.clone(),
                    storage: TensorStorage::Skipped,
                    filename: None,
                    compressed_size: 0,
                    raw_size: tensor.data.len() as u64,
                    sha256_raw: raw_hash,
                    sha256_compressed: None,
                },
                write_data: None,
            });
        }

        // Try delta encoding
        if base.raw_size == tensor.data.len() as u64 {
            if let Some(ref base_filename) = base.filename {
                if base.storage != TensorStorage::Skipped {
                    // Load base tensor data
                    let mut base_compressed = base_storage.get(base_filename)?;
                    // Truncate padding from page-aligned storage
                    let base_comp_size = base.compressed_size as usize;
                    if base_comp_size > 0 && base_compressed.len() > base_comp_size {
                        base_compressed.truncate(base_comp_size);
                    }
                    let base_algo = compression; // assume same compression
                    let base_raw = if base.storage == TensorStorage::DeltaXor {
                        // Can't delta against a delta — need the full data.
                        // This would require chain resolution. For simplicity,
                        // fall through to full save.
                        // In practice, merging should prevent deep chains.
                        return make_full_entry(tensor, &raw_hash, compression, snap_dir, rank);
                    } else {
                        compression::decompress(&base_compressed, base_algo)?
                    };

                    if let Some(xor_delta) = delta::compute_delta(&base_raw, &tensor.data) {
                        let density = delta::delta_density(&xor_delta);
                        if density < delta_threshold {
                            // Delta is worth it
                            let compressed = compression::compress(&xor_delta, compression)?;
                            let filename = tensor_filename(snap_dir, rank, &tensor.name);
                            let compressed_hash = sha256_hex(&compressed);

                            return Ok(ProcessedTensor {
                                entry: TensorEntry {
                                    name: tensor.name.clone(),
                                    shape: tensor.shape.clone(),
                                    dtype: tensor.dtype.clone(),
                                    storage: TensorStorage::DeltaXor,
                                    filename: Some(filename),
                                    compressed_size: compressed.len() as u64,
                                    raw_size: tensor.data.len() as u64,
                                    sha256_raw: raw_hash,
                                    sha256_compressed: Some(compressed_hash),
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
    make_full_entry(tensor, &raw_hash, compression, snap_dir, rank)
}

fn make_full_entry(
    tensor: &TensorData,
    raw_hash: &str,
    compression: &CompressionAlgo,
    snap_dir: &str,
    rank: u32,
) -> Result<ProcessedTensor> {
    let compressed = compression::compress(&tensor.data, compression)?;
    let filename = tensor_filename(snap_dir, rank, &tensor.name);
    let compressed_hash = sha256_hex(&compressed);

    Ok(ProcessedTensor {
        entry: TensorEntry {
            name: tensor.name.clone(),
            shape: tensor.shape.clone(),
            dtype: tensor.dtype.clone(),
            storage: TensorStorage::Full,
            filename: Some(filename),
            compressed_size: compressed.len() as u64,
            raw_size: tensor.data.len() as u64,
            sha256_raw: raw_hash.to_string(),
            sha256_compressed: Some(compressed_hash),
        },
        write_data: Some(compressed),
    })
}

/// Load a tensor's raw bytes, resolving delta chains if needed.
pub fn load_tensor(
    entry: &TensorEntry,
    snap_base_id: Option<uuid::Uuid>,
    storage: &dyn StorageBackend,
    compression: &CompressionAlgo,
    // Function to look up a base tensor entry by name and snapshot ID
    find_base_entry: &dyn Fn(uuid::Uuid, &str) -> Result<(TensorEntry, CompressionAlgo)>,
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
            let (base_entry, base_compression) = find_base_entry(base_id, &entry.name)?;
            load_tensor(&base_entry, None, storage, &base_compression, find_base_entry)
        }
        TensorStorage::Full => {
            let filename = entry.filename.as_ref().ok_or_else(|| {
                RevolverError::NotFound(format!("No filename for tensor '{}'", entry.name))
            })?;
            let mut compressed = storage.get(filename)?;

            // Truncate padding (storage may pad to page boundary for SSD alignment)
            let expected_len = entry.compressed_size as usize;
            if expected_len > 0 && compressed.len() > expected_len {
                compressed.truncate(expected_len);
            }

            // Verify compressed integrity
            if let Some(ref expected) = entry.sha256_compressed {
                let actual = sha256_hex(&compressed);
                if &actual != expected {
                    return Err(RevolverError::IntegrityError {
                        expected: expected.clone(),
                        actual,
                    });
                }
            }

            let raw = compression::decompress(&compressed, compression)?;

            // Verify raw integrity
            let actual = sha256_hex(&raw);
            if actual != entry.sha256_raw {
                return Err(RevolverError::IntegrityError {
                    expected: entry.sha256_raw.clone(),
                    actual,
                });
            }

            Ok(raw)
        }
        TensorStorage::DeltaXor => {
            let filename = entry.filename.as_ref().ok_or_else(|| {
                RevolverError::NotFound(format!("No filename for delta tensor '{}'", entry.name))
            })?;
            let mut compressed = storage.get(filename)?;

            // Truncate padding
            let expected_len = entry.compressed_size as usize;
            if expected_len > 0 && compressed.len() > expected_len {
                compressed.truncate(expected_len);
            }

            let delta_data = compression::decompress(&compressed, compression)?;

            // Load base tensor
            let base_id = snap_base_id.ok_or_else(|| {
                RevolverError::Delta(format!(
                    "Delta tensor '{}' has no base snapshot",
                    entry.name
                ))
            })?;
            let (base_entry, base_compression) = find_base_entry(base_id, &entry.name)?;
            let base_raw =
                load_tensor(&base_entry, None, storage, &base_compression, find_base_entry)?;

            // Apply XOR delta
            let raw = delta::apply_delta(&base_raw, &delta_data)?;

            // Verify
            let actual = sha256_hex(&raw);
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

/// Generate a filename for a tensor within a snapshot.
/// Sanitize the tensor name for filesystem safety.
fn tensor_filename(snap_dir: &str, rank: u32, tensor_name: &str) -> String {
    let safe_name = tensor_name
        .replace('/', "__")
        .replace('\\', "__")
        .replace(' ', "_")
        .replace(':', "_");
    format!("{}/rank_{}/{}.bin", snap_dir, rank, safe_name)
}

/// Process multiple tensors in parallel using rayon.
/// Returns entries and data to write, preserving order.
pub fn process_tensors_parallel(
    tensors: &[TensorData],
    base_entries: &HashMap<String, TensorEntry>,
    base_storage: &dyn StorageBackend,
    compression: &CompressionAlgo,
    delta_threshold: f64,
    snap_dir: &str,
    rank: u32,
) -> Result<Vec<ProcessedTensor>>
where
{
    // We can't easily parallelize due to StorageBackend not being Sync in all cases.
    // Process sequentially for now; rayon can be added per-tensor for compression.
    let mut results = Vec::with_capacity(tensors.len());

    for tensor in tensors {
        let base_entry = base_entries.get(&tensor.name);
        let processed = process_tensor(
            tensor,
            base_entry,
            base_storage,
            compression,
            delta_threshold,
            snap_dir,
            rank,
        )?;
        results.push(processed);
    }

    Ok(results)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::LocalStorage;

    #[test]
    fn skip_identical_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let storage = LocalStorage::new(dir.path()).unwrap();
        let compression = CompressionAlgo::Zstd { level: 1 };

        // Save a "base" tensor
        let data = vec![42u8; 8192];
        let raw_hash = sha256_hex(&data);
        let compressed = compression::compress(&data, &compression).unwrap();
        let base_filename = "snapshots/base/rank_0/test_tensor.bin".to_string();
        storage.put(&base_filename, &compressed).unwrap();

        let base_entry = TensorEntry {
            name: "test_tensor".into(),
            shape: vec![8192],
            dtype: "uint8".into(),
            storage: TensorStorage::Full,
            filename: Some(base_filename),
            compressed_size: compressed.len() as u64,
            raw_size: 8192,
            sha256_raw: raw_hash.clone(),
            sha256_compressed: Some(sha256_hex(&compressed)),
        };

        // Process same tensor again → should be Skipped
        let tensor = TensorData {
            name: "test_tensor".into(),
            shape: vec![8192],
            dtype: "uint8".into(),
            data: data.clone(),
        };

        let result = process_tensor(
            &tensor,
            Some(&base_entry),
            &storage,
            &compression,
            0.5,
            "snapshots/new",
            0,
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
            compressed_size: compressed_v1.len() as u64,
            raw_size: 50_000,
            sha256_raw: sha256_hex(&data_v1),
            sha256_compressed: Some(sha256_hex(&compressed_v1)),
        };

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
            Some(&base_entry),
            &storage,
            &compression,
            0.5,
            "snapshots/new",
            0,
        )
        .unwrap();

        assert_eq!(result.entry.storage, TensorStorage::DeltaXor);
        assert!(result.write_data.is_some());
        // Delta should be very small
        assert!(result.entry.compressed_size < 1000);
    }

    #[test]
    fn full_save_no_base() {
        let dir = tempfile::tempdir().unwrap();
        let storage = LocalStorage::new(dir.path()).unwrap();
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
            &storage,
            &compression,
            0.5,
            "snapshots/first",
            0,
        )
        .unwrap();

        assert_eq!(result.entry.storage, TensorStorage::Full);
        assert!(result.write_data.is_some());
    }
}