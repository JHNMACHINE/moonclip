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
    /// Raw bytes of the base snapshot, when the coordinator kept them after
    /// writing it. Present only for the snapshot those bytes came from.
    ///
    /// Reading and decompressing the base was 36% of the write path's CPU,
    /// and it decompresses the same bytes this process wrote moments earlier.
    /// Holding them costs one resident copy of the state and removes that
    /// work entirely. See `CoordinatorConfig::keep_base_in_memory`.
    pub raw: Option<Arc<HashMap<String, Vec<u8>>>>,
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

    /// Whether this tensor's base bytes are already in memory, making
    /// `base_bytes` free and the sampled pre-check pointless.
    fn has_raw(&self, name: &str) -> bool {
        self.raw.as_ref().is_some_and(|r| r.contains_key(name))
    }

    /// Raw bytes of a base tensor. Returns None for delta/skipped bases.
    ///
    /// Borrows from the in-memory base when it is there, so the common case
    /// costs nothing at all — not even a copy.
    fn base_bytes(&self, entry: &TensorEntry) -> Result<Option<std::borrow::Cow<'_, [u8]>>> {
        use std::borrow::Cow;

        if entry.storage != TensorStorage::Full {
            return Ok(None); // Can't delta-chain against non-full bases
        }
        if let Some(bytes) = self.raw.as_ref().and_then(|r| r.get(&entry.name)) {
            return Ok(Some(Cow::Borrowed(bytes.as_slice())));
        }
        match self.read_compressed(entry)? {
            Some(compressed) => Ok(Some(Cow::Owned(compression::decompress(
                &compressed,
                &self.compression,
            )?))),
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
        hash_raw: target.hash_raw.clone(),
        hash_compressed: None,
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
            if base_entry.hash_raw == raw_hash {
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
                        hash_raw: raw_hash,
                        hash_compressed: None,
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
                // The sampled window exists to avoid decompressing a base that
                // the delta is going to be thrown away against. With the base
                // already in memory there is nothing to avoid, and the probe
                // would be pure added work.
                let sample = if cache.has_raw(&tensor.name) {
                    None
                } else {
                    profile::time(profile::Phase::SamplePrefix, || {
                        cache.sample_prefix(base_entry, SAMPLE_RAW)
                    })
                };
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
                        cache.base_bytes(base_entry)
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
                                    hash_raw: raw_hash,
                                    hash_compressed: Some(compressed_hash),
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
            hash_raw: raw_hash.to_string(),
            hash_compressed: Some(compressed_hash),
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

/// Find a tensor in the base snapshot, following the alias if the base stored
/// one under that name.
///
/// Deduplication keeps the first of a set of identical tensors and records the
/// rest as `Alias` entries pointing at it — and which one comes first is
/// decided by the order the state dict was iterated in. That order is not
/// stable across saves: wrapping a model differently, or an optimizer that
/// rebuilds its groups, is enough to swap them. When it swaps, the tensor that
/// was the alias is now the one carrying bytes, it is unchanged since the base
/// so the save writes it `Skipped`, and resolving that skip lands on the
/// base's alias entry — which holds no bytes and cannot be read on its own.
///
/// Following the reference here is what the loader already does within a
/// single snapshot; the base is no different. Without it a perfectly good
/// checkpoint of a model with tied weights is simply unreadable.
fn resolve_base_entry(
    base_id: uuid::Uuid,
    name: &str,
    find_base_entry: &BaseEntryResolver<'_>,
) -> Result<(TensorEntry, CompressionAlgo, Option<Arc<Vec<u8>>>)> {
    let mut current = name.to_string();
    // One hop is the rule: an alias points at a tensor that holds bytes. The
    // bound is for a manifest that is corrupt or written by something else —
    // a cycle here would hang a load rather than fail it.
    for _ in 0..8 {
        let (entry, compression, pack) = find_base_entry(base_id, &current)?;
        if entry.storage != TensorStorage::Alias {
            return Ok((entry, compression, pack));
        }
        current = entry.alias_of.clone().ok_or_else(|| {
            MoonclipError::NotFound(format!("Alias '{current}' in the base has no target"))
        })?;
    }

    Err(MoonclipError::Delta(format!(
        "Alias chain starting at '{name}' in base snapshot {base_id} does not end"
    )))
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
            let (base_entry, base_compression, base_pack) =
                resolve_base_entry(base_id, &entry.name, find_base_entry)?;
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
            if let Some(ref expected) = entry.hash_compressed {
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
            let (base_entry, base_compression, base_pack) =
                resolve_base_entry(base_id, &entry.name, find_base_entry)?;
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
            if actual != entry.hash_raw {
                return Err(MoonclipError::IntegrityError {
                    expected: entry.hash_raw.clone(),
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
            raw: None,
        }
    }

    /// The same base, but with its raw bytes already in memory — what the
    /// coordinator hands over when it retained the last full snapshot.
    fn make_base_cache_in_memory(
        storage: Arc<dyn StorageBackend>,
        entries: Vec<TensorEntry>,
        compression: &CompressionAlgo,
        raw: Vec<(&str, Vec<u8>)>,
    ) -> BaseCache {
        let mut cache = make_base_cache(storage, entries, compression);
        cache.raw = Some(Arc::new(
            raw.into_iter().map(|(n, d)| (n.to_string(), d)).collect(),
        ));
        cache
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
            hash_raw: raw_hash.clone(),
            hash_compressed: None,
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
            hash_raw: hash_hex(&data_v1),
            hash_compressed: None,
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
            hash_raw: hash_hex(&data_v1),
            hash_compressed: None,
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
        assert!(result.entry.hash_compressed.is_some());
    }

    // ── Read path ───────────────────────────────────────────────────
    //
    // Everything above tests what a save decides. These test that the
    // decision can be undone, which is the only thing a checkpoint is for.
    // A save that cannot be read back is worse than no save at all: the run
    // paid for it and finds out at resume.

    /// Put processed bytes where the entry will look for them.
    fn persist(
        storage: &dyn StorageBackend,
        mut entry: TensorEntry,
        data: Option<Vec<u8>>,
        path: &str,
    ) -> TensorEntry {
        if let Some(bytes) = data {
            storage.put(path, &bytes).unwrap();
            entry.filename = Some(path.into());
        }
        entry
    }

    /// A resolver over a fixed set of base entries, standing in for what the
    /// coordinator reads out of the manifest at load time.
    fn base_resolver(
        entries: Vec<TensorEntry>,
        compression: CompressionAlgo,
    ) -> impl Fn(uuid::Uuid, &str) -> Result<(TensorEntry, CompressionAlgo, Option<Arc<Vec<u8>>>)> + Sync
    {
        let map: HashMap<String, TensorEntry> =
            entries.into_iter().map(|e| (e.name.clone(), e)).collect();
        move |_id, name| {
            map.get(name)
                .cloned()
                .map(|e| (e, compression.clone(), None))
                .ok_or_else(|| MoonclipError::NotFound(name.into()))
        }
    }

    /// Store `data` as a Full tensor and hand back its entry, the way a first
    /// save leaves the base snapshot.
    fn store_base(
        storage: &Arc<dyn StorageBackend>,
        name: &str,
        data: &[u8],
        dtype: &str,
        compression: &CompressionAlgo,
    ) -> TensorEntry {
        let tensor = td(name, data.to_vec(), dtype);
        let processed = process_tensor(&tensor, None, compression, 0.95, &DType::None).unwrap();
        assert_eq!(processed.entry.storage, TensorStorage::Full);
        persist(
            storage.as_ref(),
            processed.entry,
            processed.write_data,
            &format!("base/{name}.bin"),
        )
    }

    /// Keeping the base in memory is an optimisation, so the only thing that
    /// matters is that it changes nothing: the same delta, byte for byte, as
    /// the base read back from disk.
    #[test]
    fn an_in_memory_base_produces_the_same_delta_as_disk() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let compression = CompressionAlgo::Zstd { level: 1 };

        let v1 = noise(40_000, 0x1357);
        let base_entry = store_base(&storage, "w", &v1, "float32", &compression);

        let mut v2 = v1.clone();
        v2[7] ^= 0x3f;
        v2[30_000] ^= 0x81;
        let tensor = td("w", v2, "float32");

        let from_disk = process_tensor(
            &tensor,
            Some(&make_base_cache(
                Arc::clone(&storage),
                vec![base_entry.clone()],
                &compression,
            )),
            &compression,
            0.95,
            &DType::None,
        )
        .unwrap();

        let from_memory = process_tensor(
            &tensor,
            Some(&make_base_cache_in_memory(
                Arc::clone(&storage),
                vec![base_entry],
                &compression,
                vec![("w", v1)],
            )),
            &compression,
            0.95,
            &DType::None,
        )
        .unwrap();

        assert_eq!(from_disk.entry.storage, TensorStorage::DeltaXor);
        assert_eq!(from_memory.entry.storage, from_disk.entry.storage);
        assert_eq!(from_memory.entry.hash_raw, from_disk.entry.hash_raw);
        assert_eq!(from_memory.entry.shuffled, from_disk.entry.shuffled);
        assert_eq!(
            from_memory.write_data, from_disk.write_data,
            "the retained base must produce an identical delta"
        );
    }

    /// Proves the retained bytes are actually what gets used: there is no
    /// base on disk at all here, so a delta can only come from memory.
    #[test]
    fn an_in_memory_base_needs_no_disk_read() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let compression = CompressionAlgo::Zstd { level: 1 };

        let v1 = noise(40_000, 0x2468);
        // An entry that describes a file which was never written.
        let base_entry = TensorEntry {
            name: "w".into(),
            shape: vec![40_000],
            dtype: "float32".into(),
            original_dtype: None,
            storage: TensorStorage::Full,
            alias_of: None,
            filename: Some("nowhere/w.bin".into()),
            offset: 0,
            compressed_size: 1234,
            raw_size: 40_000,
            hash_raw: hash_hex(&v1),
            hash_compressed: None,
            shuffled: false,
        };

        let mut v2 = v1.clone();
        v2[900] ^= 0x11;
        let cache = make_base_cache_in_memory(
            Arc::clone(&storage),
            vec![base_entry],
            &compression,
            vec![("w", v1)],
        );

        let processed = process_tensor(
            &td("w", v2, "float32"),
            Some(&cache),
            &compression,
            0.95,
            &DType::None,
        )
        .unwrap();

        assert_eq!(
            processed.entry.storage,
            TensorStorage::DeltaXor,
            "with no base on disk, a delta can only have come from memory"
        );
    }

    #[test]
    fn a_full_tensor_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let compression = CompressionAlgo::Zstd { level: 1 };

        let data = noise(40_000, 0x11);
        let entry = store_base(&storage, "w", &data, "float32", &compression);

        let resolve = base_resolver(vec![], compression.clone());
        let got =
            load_tensor(&entry, None, storage.as_ref(), &compression, None, &resolve).unwrap();
        assert_eq!(got, data);
    }

    /// The guarantee the byte-shuffle format change rests on. If the filter
    /// were inverted wrongly this would not raise — it would hand back
    /// plausible noise — so the assertion has to be on the bytes.
    #[test]
    fn a_shuffled_delta_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let compression = CompressionAlgo::Zstd { level: 1 };
        let base_id = uuid::Uuid::new_v4();

        let v1 = noise(40_000, 0xabcd);
        let base_entry = store_base(&storage, "w", &v1, "float32", &compression);

        let mut v2 = v1.clone();
        v2[100] ^= 0xff;
        v2[20_000] ^= 0x0f;

        let cache = make_base_cache(
            Arc::clone(&storage),
            vec![base_entry.clone()],
            &compression,
        );
        let tensor = td("w", v2.clone(), "float32");
        let processed = process_tensor(&tensor, Some(&cache), &compression, 0.95, &DType::None)
            .unwrap();

        assert_eq!(processed.entry.storage, TensorStorage::DeltaXor);
        assert!(
            processed.entry.shuffled,
            "an fp32 delta must go through the byte shuffle"
        );

        let entry = persist(
            storage.as_ref(),
            processed.entry,
            processed.write_data,
            "delta/w.bin",
        );
        let resolve = base_resolver(vec![base_entry], compression.clone());
        let got = load_tensor(
            &entry,
            Some(base_id),
            storage.as_ref(),
            &compression,
            None,
            &resolve,
        )
        .unwrap();
        assert_eq!(got, v2, "the delta must reconstruct the updated tensor");
    }

    #[test]
    fn a_skipped_tensor_resolves_through_the_base() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let compression = CompressionAlgo::Zstd { level: 1 };
        let base_id = uuid::Uuid::new_v4();

        let data = noise(40_000, 0x55);
        let base_entry = store_base(&storage, "w", &data, "float32", &compression);

        let cache = make_base_cache(
            Arc::clone(&storage),
            vec![base_entry.clone()],
            &compression,
        );
        let tensor = td("w", data.clone(), "float32");
        let processed = process_tensor(&tensor, Some(&cache), &compression, 0.95, &DType::None)
            .unwrap();
        assert_eq!(processed.entry.storage, TensorStorage::Skipped);
        assert!(processed.write_data.is_none(), "a skip writes no bytes");

        // The snapshot stores nothing for this tensor, so the whole value has
        // to come from the base.
        let resolve = base_resolver(vec![base_entry], compression.clone());
        let got = load_tensor(
            &processed.entry,
            Some(base_id),
            storage.as_ref(),
            &compression,
            None,
            &resolve,
        )
        .unwrap();
        assert_eq!(got, data);
    }

    /// Corruption has to be reported, not returned. Weights that are subtly
    /// wrong restart a run that then trains on garbage.
    #[test]
    fn corrupted_bytes_are_caught_not_returned() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let compression = CompressionAlgo::Zstd { level: 1 };

        let data = noise(40_000, 0x77);
        let entry = store_base(&storage, "w", &data, "float32", &compression);

        let path = entry.filename.clone().unwrap();
        let mut stored = storage.get(&path).unwrap();
        stored[entry.compressed_size as usize / 2] ^= 0xff;
        storage.put(&path, &stored).unwrap();

        let resolve = base_resolver(vec![], compression.clone());
        let result = load_tensor(&entry, None, storage.as_ref(), &compression, None, &resolve);
        assert!(
            matches!(result, Err(MoonclipError::IntegrityError { .. })),
            "expected an integrity error, got {result:?}"
        );
    }

    #[test]
    fn an_alias_cannot_be_loaded_on_its_own() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let compression = CompressionAlgo::Zstd { level: 1 };

        let data = noise(40_000, 0x99);
        let target = store_base(&storage, "token_emb", &data, "float32", &compression);
        let alias = make_alias_entry(&td("head", data, "float32"), &target);

        let resolve = base_resolver(vec![], compression.clone());
        let err = load_tensor(&alias, None, storage.as_ref(), &compression, None, &resolve)
            .unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("token_emb"),
            "the error must name the tensor holding the bytes, got: {message}"
        );
    }

    /// Saving in bf16 to halve the checkpoint must still hand training back
    /// fp32 buffers, or `load_state_dict` fails on dtype.
    #[test]
    fn a_cast_tensor_comes_back_in_its_original_dtype() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let compression = CompressionAlgo::Zstd { level: 1 };

        let data = noise(40_000, 0xbb);
        let tensor = td("w", data.clone(), "float32");
        let processed =
            process_tensor(&tensor, None, &compression, 0.95, &DType::BFloat16).unwrap();

        assert_eq!(processed.entry.dtype, "bfloat16");
        assert_eq!(processed.entry.original_dtype.as_deref(), Some("float32"));
        assert_eq!(
            processed.entry.raw_size as usize,
            data.len() / 2,
            "bf16 is half of fp32 on disk"
        );

        let entry = persist(storage.as_ref(), processed.entry, processed.write_data, "w.bin");
        let resolve = base_resolver(vec![], compression.clone());
        let got =
            load_tensor(&entry, None, storage.as_ref(), &compression, None, &resolve).unwrap();
        assert_eq!(
            got.len(),
            data.len(),
            "load must uncast back to fp32 width, not hand back bf16"
        );
    }
}
