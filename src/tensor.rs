use std::collections::HashMap;
use std::sync::Arc;

use crate::cast::{self, DType, is_castable_float};
use crate::compression;
use crate::delta;
use crate::error::{Result, MoonclipError};
use crate::hash::hash_hex;
use crate::manifest::{CompressionAlgo, TensorEntry, TensorStorage};
use crate::profile;
use crate::shuffle;
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

/// Tensors smaller than this skip the delta machinery entirely.
/// Must match delta::DELTA_MIN_SIZE semantics.
const DELTA_MIN_SIZE: usize = 4096;

/// Raw bytes sampled from the start of a base tensor to judge the delta.
const SAMPLE_RAW: usize = 64 * 1024;
/// Compressed window read from storage to produce the sample.
const SAMPLE_WINDOW: usize = 256 * 1024;
/// Minimum sample size to trust the verdict.
const SAMPLE_MIN: usize = 8 * 1024;

/// Lazy handle to the base snapshot's data for delta comparison.
///
/// Tensor bytes are read from storage on demand (range reads from the
/// pack file), so unchanged tensors (hash match) and tensors whose
/// sampled window rules the delta out cost little or no base I/O.
pub struct BaseCache {
    /// Storage to read base data from.
    pub storage: Arc<dyn StorageBackend>,
    /// Pack file of the base snapshot (None for legacy per-file storage).
    pub pack_file: Option<String>,
    /// Tensor entries from the base snapshot, keyed by name.
    pub entries: HashMap<String, TensorEntry>,
    /// Compression algo of the base snapshot.
    pub compression: CompressionAlgo,
}

impl BaseCache {
    /// Read the compressed bytes for a tensor (full entry).
    fn read_compressed(&self, entry: &TensorEntry) -> Result<Option<Vec<u8>>> {
        if let Some(ref pack) = self.pack_file {
            let data =
                self.storage
                    .get_range(pack, entry.offset, entry.compressed_size as usize)?;
            if data.len() == entry.compressed_size as usize {
                return Ok(Some(data));
            }
            return Ok(None);
        }
        if let Some(ref filename) = entry.filename {
            match self.storage.get(filename) {
                Ok(mut data) => {
                    let expected = entry.compressed_size as usize;
                    if expected > 0 && data.len() > expected {
                        data.truncate(expected);
                    }
                    Ok(Some(data))
                }
                Err(_) => Ok(None), // missing base file → full save
            }
        } else {
            Ok(None)
        }
    }

    /// Best-effort decompression of the first ~`max_raw` bytes of a base
    /// tensor, reading only a small window of compressed data.
    /// Returns None when no usable sample could be produced.
    fn sample_prefix(&self, entry: &TensorEntry, max_raw: usize) -> Option<Vec<u8>> {
        if entry.storage != TensorStorage::Full {
            return None;
        }
        let window = (entry.compressed_size as usize).min(SAMPLE_WINDOW);
        let compressed = if let Some(ref pack) = self.pack_file {
            self.storage.get_range(pack, entry.offset, window).ok()?
        } else {
            // Legacy per-file storage: read the file (cheap, page-cached).
            let filename = entry.filename.as_ref()?;
            let mut data = self.storage.get(filename).ok()?;
            data.truncate(window);
            data
        };
        compression::decompress_prefix(&compressed, &self.compression, max_raw).ok()
    }

    /// Decompress a base tensor to raw bytes. Returns None for delta/skipped bases.
    fn decompress_tensor(&self, entry: &TensorEntry) -> Result<Option<Vec<u8>>> {
        if entry.storage != TensorStorage::Full {
            return Ok(None); // Can't delta-chain against non-full bases
        }
        match self.read_compressed(entry)? {
            Some(compressed) => {
                let raw = compression::decompress(&compressed, &self.compression)?;
                Ok(Some(raw))
            }
            None => Ok(None),
        }
    }
}

/// Plan intra-snapshot deduplication.
///
/// Returns, for each input tensor, `Some(name)` of the tensor that will
/// actually hold its bytes when it is a duplicate, or `None` when it is
/// the one to store.
///
/// Models routinely contain several names bound to identical bytes: tied
/// input/output embeddings share storage outright, and per-layer buffers
/// such as causal masks are built identically in every block. Storing them
/// once per name inflates every checkpoint — a tied 32k-vocab embedding in
/// fp32 is written twice. `torch.save` avoids this because pickle memoizes
/// shared storages; Moonclip has to do it explicitly.
///
/// Keyed on (hash, dtype): the cast applied later is deterministic, so
/// equal input bytes of the same dtype stay equal afterwards. Shapes may
/// differ — each alias entry keeps its own.
pub fn dedup_plan(tensors: &[TensorData]) -> Vec<Option<String>> {
    use rayon::prelude::*;

    let hashes: Vec<String> = profile::time(profile::Phase::DedupHash, || {
        tensors.par_iter().map(|t| hash_hex(&t.data)).collect()
    });

    let mut first_seen: HashMap<(&str, &str), &str> = HashMap::new();
    let mut plan = vec![None; tensors.len()];

    for (i, t) in tensors.iter().enumerate() {
        if t.data.len() < DELTA_MIN_SIZE {
            continue; // not worth an extra manifest entry
        }
        let key = (hashes[i].as_str(), t.dtype.as_str());
        match first_seen.get(&key) {
            Some(owner) => plan[i] = Some((*owner).to_string()),
            None => {
                first_seen.insert(key, t.name.as_str());
            }
        }
    }

    plan
}

/// Build the entry for a tensor whose bytes live under another name.
/// Metadata is inherited from the tensor that was actually stored, so the
/// two agree on dtype, cast and hash; only name and shape are its own.
pub fn make_alias_entry(tensor: &TensorData, target: &TensorEntry) -> TensorEntry {
    TensorEntry {
        name: tensor.name.clone(),
        shape: tensor.shape.clone(),
        dtype: target.dtype.clone(),
        original_dtype: target.original_dtype.clone(),
        storage: TensorStorage::Alias,
        alias_of: Some(target.name.clone()),
        filename: None,
        offset: 0,
        compressed_size: 0,
        raw_size: target.raw_size,
        sha256_raw: target.sha256_raw.clone(),
        sha256_compressed: None,
        shuffled: false,
    }
}

/// Process a tensor for saving: optionally cast dtype, compare with base,
/// decide skip/delta/full, compress, and return the entry + data to write.
///
/// `base_cache`: lazy handle to the base snapshot (None for first save).
/// `delta_max_ratio`: see [`crate::delta::pays_off`].
pub fn process_tensor(
    tensor: &TensorData,
    base_cache: Option<&BaseCache>,
    compression: &CompressionAlgo,
    delta_max_ratio: f64,
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

    let raw_hash = profile::time(profile::Phase::RawHash, || hash_hex(&working_data));

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
                        alias_of: None,
                        filename: None,
                        offset: 0,
                        compressed_size: 0,
                        raw_size: working_data.len() as u64,
                        sha256_raw: raw_hash,
                        sha256_compressed: None,
                        shuffled: false,
                    },
                    write_data: None,
                });
            }

            // Try delta encoding (only against full bases with matching size)
            if base_entry.raw_size == working_data.len() as u64
                && base_entry.storage == TensorStorage::Full
                && working_data.len() >= DELTA_MIN_SIZE
            {
                // Judge the delta on a decompressed prefix of the base:
                // reading a small window avoids decompressing the whole
                // base tensor just to throw the delta away.
                let sample = profile::time(profile::Phase::SamplePrefix, || {
                    cache.sample_prefix(base_entry, SAMPLE_RAW)
                });
                let sampled_verdict = match sample {
                    Some(prefix) if prefix.len() >= SAMPLE_MIN => {
                        Some(profile::time(profile::Phase::PaysOff, || {
                            delta::pays_off(
                                &prefix,
                                &working_data,
                                compression,
                                delta_max_ratio,
                                shuffle::element_size(&working_dtype),
                            )
                        }))
                    }
                    _ => None, // no usable window → decide after decompressing
                };

                if sampled_verdict != Some(false) {
                    let base = profile::time(profile::Phase::BaseDecompress, || {
                        cache.decompress_tensor(base_entry)
                    })?;
                    if let Some(base_raw) = base {
                        // Without an earlier window, take the verdict now
                        // from the decompressed base — no extra I/O here.
                        let worth_it = sampled_verdict.unwrap_or_else(|| {
                            profile::time(profile::Phase::PaysOff, || {
                                delta::pays_off(
                                    &base_raw,
                                    &working_data,
                                    compression,
                                    delta_max_ratio,
                                    shuffle::element_size(&working_dtype),
                                )
                            })
                        });

                        let xor = profile::time(profile::Phase::XorDelta, || {
                            worth_it
                                .then(|| delta::compute_delta(&base_raw, &working_data))
                                .flatten()
                        });
                        if let Some(xor_delta) = xor {
                            // Group the delta into byte planes before zstd
                            // sees it. The unchanged high bytes of every
                            // float then form long runs instead of being
                            // interleaved with the noisy low ones: on real
                            // AdamW deltas this compresses 2.45x faster and
                            // 19% smaller. See `crate::shuffle`.
                            let itemsize = shuffle::element_size(&working_dtype);
                            let shuffled = itemsize > 1;
                            let to_compress = if shuffled {
                                Cow::Owned(profile::time(profile::Phase::Shuffle, || {
                                    shuffle::shuffle(&xor_delta, itemsize)
                                }))
                            } else {
                                Cow::Borrowed(&xor_delta[..])
                            };
                            let compressed =
                                profile::time(profile::Phase::CompressDelta, || {
                                    compression::compress(&to_compress, compression)
                                })?;
                            let compressed_hash = hash_hex(&compressed);

                            return Ok(ProcessedTensor {
                                entry: TensorEntry {
                                    name: tensor.name.clone(),
                                    shape: tensor.shape.clone(),
                                    dtype: working_dtype.to_string(),
                                    original_dtype: orig_dtype.clone(),
                                    storage: TensorStorage::DeltaXor,
                                    alias_of: None,
                                    filename: None, // Set by coordinator after packing
                                    offset: 0,      // Set by coordinator after packing
                                    compressed_size: compressed.len() as u64,
                                    raw_size: working_data.len() as u64,
                                    sha256_raw: raw_hash,
                                    sha256_compressed: Some(compressed_hash),
                                    shuffled,
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
    let compressed = profile::time(profile::Phase::CompressFull, || {
        compression::compress(working_data, compression)
    })?;
    let compressed_hash = hash_hex(&compressed);

    Ok(ProcessedTensor {
        entry: TensorEntry {
            name: tensor.name.clone(),
            shape: tensor.shape.clone(),
            dtype: working_dtype.to_string(),
            original_dtype: original_dtype.clone(),
            storage: TensorStorage::Full,
            alias_of: None,
            filename: None, // Set by coordinator after packing
            offset: 0,      // Set by coordinator after packing
            compressed_size: compressed.len() as u64,
            raw_size: working_data.len() as u64,
            sha256_raw: raw_hash.to_string(),
            sha256_compressed: Some(compressed_hash),
            shuffled: false,
        },
        write_data: Some(compressed),
    })
}

/// Resolver for base-snapshot tensor entries during load.
/// Returns (entry, compression of the base snapshot, optional shared pack bytes).
pub type BaseEntryResolver<'a> = dyn Fn(uuid::Uuid, &str) -> Result<(TensorEntry, CompressionAlgo, Option<Arc<Vec<u8>>>)>
    + Sync
    + 'a;

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
    find_base_entry: &BaseEntryResolver<'_>,
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
        MoonclipError::NotFound(format!("No filename for tensor '{}'", entry.name))
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
    find_base_entry: &BaseEntryResolver<'_>,
) -> Result<Vec<u8>> {
    match entry.storage {
        // Aliases are resolved by the caller once every other tensor in
        // the snapshot has been loaded — their bytes live under another
        // name in the same rank entry.
        TensorStorage::Alias => Err(MoonclipError::NotFound(format!(
            "Tensor '{}' is an alias of '{}' and must be resolved after the \
             snapshot's other tensors",
            entry.name,
            entry.alias_of.as_deref().unwrap_or("?")
        ))),
        TensorStorage::Skipped => {
            // Load from base snapshot
            let base_id = snap_base_id.ok_or_else(|| {
                MoonclipError::Delta(format!(
                    "Tensor '{}' is skipped but snapshot has no base",
                    entry.name
                ))
            })?;
            let (base_entry, base_compression, base_pack) = find_base_entry(base_id, &entry.name)?;
            load_tensor_raw(
                &base_entry,
                None,
                storage,
                &base_compression,
                base_pack.as_ref().map(|p| p.as_slice()),
                find_base_entry,
            )
        }
        TensorStorage::Full => {
            let compressed = extract_compressed(entry, storage, pack_data)?;

            // Verify compressed integrity (cheap with xxHash3)
            if let Some(ref expected) = entry.sha256_compressed {
                let actual = hash_hex(&compressed);
                if &actual != expected {
                    return Err(MoonclipError::IntegrityError {
                        expected: expected.clone(),
                        actual,
                    });
                }
            }

            compression::decompress(&compressed, compression)
        }
        TensorStorage::DeltaXor => {
            let compressed = extract_compressed(entry, storage, pack_data)?;
            let mut delta_data = compression::decompress(&compressed, compression)?;
            if entry.shuffled {
                delta_data = shuffle::unshuffle(&delta_data, shuffle::element_size(&entry.dtype));
            }

            // Load base tensor
            let base_id = snap_base_id.ok_or_else(|| {
                MoonclipError::Delta(format!(
                    "Delta tensor '{}' has no base snapshot",
                    entry.name
                ))
            })?;
            let (base_entry, base_compression, base_pack) = find_base_entry(base_id, &entry.name)?;
            let base_raw = load_tensor_raw(
                &base_entry,
                None,
                storage,
                &base_compression,
                base_pack.as_ref().map(|p| p.as_slice()),
                find_base_entry,
            )?;

            // Apply XOR delta
            let raw = delta::apply_delta(&base_raw, &delta_data)?;

            // Verify reconstructed tensor integrity
            let actual = hash_hex(&raw);
            if actual != entry.sha256_raw {
                return Err(MoonclipError::IntegrityError {
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
///
/// Takes references so the caller can leave deduplicated tensors out
/// without copying the ones it keeps.
pub fn process_tensors_parallel(
    tensors: &[&TensorData],
    base_cache: Option<&BaseCache>,
    compression: &CompressionAlgo,
    delta_max_ratio: f64,
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
                delta_max_ratio,
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

    /// Deterministic pseudo-random bytes.
    ///
    /// Real weight tensors barely compress, so tests that exercise the
    /// delta/full decision must not use runs of constant bytes: those
    /// compress to almost nothing in full form, and a delta against them
    /// legitimately saves nothing.
    fn noise(len: usize, seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 24) as u8
            })
            .collect()
    }

    fn td(name: &str, data: Vec<u8>, dtype: &str) -> TensorData {
        TensorData {
            name: name.into(),
            shape: vec![data.len()],
            dtype: dtype.into(),
            data,
        }
    }

    /// Regression: tied input/output embeddings are the same bytes under
    /// two names. Storing both is what made Moonclip's checkpoints larger
    /// than `torch.save`'s, whose pickler memoizes shared storages.
    #[test]
    fn dedup_plan_aliases_tied_weights() {
        let shared = noise(64_000, 0x77);
        let tensors = vec![
            td("token_emb.weight", shared.clone(), "float32"),
            td("blocks.0.attn.qkv.weight", noise(64_000, 0x88), "float32"),
            td("head.weight", shared, "float32"),
        ];

        let plan = dedup_plan(&tensors);
        assert_eq!(plan[0], None, "first occurrence holds the bytes");
        assert_eq!(plan[1], None, "distinct tensor is stored on its own");
        assert_eq!(plan[2].as_deref(), Some("token_emb.weight"));
    }

    #[test]
    fn dedup_plan_separates_dtypes_and_small_tensors() {
        let bytes = noise(64_000, 0x99);
        let tensors = vec![
            td("a", bytes.clone(), "float32"),
            // Same bytes, different dtype: the cast applied later differs,
            // so these must not share storage.
            td("b", bytes, "int32"),
            // Below the size threshold: an extra manifest entry would cost
            // more than the bytes it saves.
            td("tiny_a", vec![1u8; 64], "float32"),
            td("tiny_b", vec![1u8; 64], "float32"),
        ];

        let plan = dedup_plan(&tensors);
        assert_eq!(plan[1], None, "different dtype must not alias");
        assert_eq!(plan[3], None, "tiny tensors are not aliased");
    }

    fn make_base_cache(
        storage: Arc<dyn StorageBackend>,
        entries: Vec<TensorEntry>,
        compression: &CompressionAlgo,
    ) -> BaseCache {
        let entry_map = entries.into_iter().map(|e| (e.name.clone(), e)).collect();
        BaseCache {
            storage,
            pack_file: None,
            entries: entry_map,
            compression: compression.clone(),
        }
    }

    #[test]
    fn skip_identical_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
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
            alias_of: None,
            filename: Some(base_filename),
            offset: 0,
            compressed_size: compressed.len() as u64,
            raw_size: 8192,
            sha256_raw: raw_hash.clone(),
            sha256_compressed: None,
            shuffled: false,
            original_dtype: None,
        };

        let cache = make_base_cache(Arc::clone(&storage), vec![base_entry], &compression);

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
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let compression = CompressionAlgo::Zstd { level: 1 };

        let data_v1 = noise(50_000, 0xabcd);
        let compressed_v1 = compression::compress(&data_v1, &compression).unwrap();
        let base_filename = "snapshots/base/rank_0/big_tensor.bin".to_string();
        storage.put(&base_filename, &compressed_v1).unwrap();

        let base_entry = TensorEntry {
            name: "big_tensor".into(),
            shape: vec![50_000],
            dtype: "uint8".into(),
            storage: TensorStorage::Full,
            alias_of: None,
            filename: Some(base_filename),
            offset: 0,
            compressed_size: compressed_v1.len() as u64,
            raw_size: 50_000,
            sha256_raw: hash_hex(&data_v1),
            sha256_compressed: None,
            shuffled: false,
            original_dtype: None,
        };

        let cache = make_base_cache(Arc::clone(&storage), vec![base_entry], &compression);

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
            0.95,
            &DType::None,
        )
        .unwrap();

        assert_eq!(result.entry.storage, TensorStorage::DeltaXor);
        assert!(result.write_data.is_some());
        assert!(result.entry.compressed_size < 1000);
    }

    #[test]
    fn dense_change_falls_back_to_full() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let compression = CompressionAlgo::Zstd { level: 1 };

        let data_v1 = noise(200_000, 0x1111);
        let compressed_v1 = compression::compress(&data_v1, &compression).unwrap();
        let base_filename = "snapshots/base/rank_0/dense.bin".to_string();
        storage.put(&base_filename, &compressed_v1).unwrap();

        let base_entry = TensorEntry {
            name: "dense".into(),
            shape: vec![200_000],
            dtype: "uint8".into(),
            storage: TensorStorage::Full,
            alias_of: None,
            filename: Some(base_filename),
            offset: 0,
            compressed_size: compressed_v1.len() as u64,
            raw_size: 200_000,
            sha256_raw: hash_hex(&data_v1),
            sha256_compressed: None,
            shuffled: false,
            original_dtype: None,
        };

        let cache = make_base_cache(Arc::clone(&storage), vec![base_entry], &compression);

        // Unrelated data: the XOR is as incompressible as the tensor
        // itself, so the delta buys nothing and must be stored Full.
        let data_v2 = noise(200_000, 0x2222);
        let tensor = TensorData {
            name: "dense".into(),
            shape: vec![200_000],
            dtype: "uint8".into(),
            data: data_v2,
        };

        let result = process_tensor(&tensor, Some(&cache), &compression, 0.95, &DType::None).unwrap();
        assert_eq!(result.entry.storage, TensorStorage::Full);
        assert!(result.write_data.is_some());
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
        assert!(result.entry.sha256_compressed.is_some());
    }
}
