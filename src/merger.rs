use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

use uuid::Uuid;

use crate::compression;
use crate::delta;
use crate::error::{Result, MoonclipError};
use crate::hash::hash_hex;
use crate::manifest::*;
use crate::storage::StorageBackend;

/// Configuration for the delta merger.
#[derive(Debug, Clone)]
pub struct MergerConfig {
    /// Merge stride: after `stride` consecutive deltas, merge them into one.
    pub stride: usize,
    /// Max delta chain depth before forcing a full checkpoint merge.
    pub max_chain_depth: usize,
}

impl Default for MergerConfig {
    fn default() -> Self {
        MergerConfig {
            stride: 3,
            max_chain_depth: 10,
        }
    }
}

pub(crate) enum MergeCommand {
    CheckAndMerge,
    ForceFullMerge,
    Shutdown,
}

pub struct DeltaMerger {
    // Mutex-wrapped so DeltaMerger is Sync (mpsc::Sender is Send but not Sync);
    // the coordinator shares it with the background save thread.
    sender: Option<Mutex<mpsc::Sender<MergeCommand>>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl DeltaMerger {
    pub fn new(
        config: MergerConfig,
        storage: Arc<dyn StorageBackend>,
        manifest: Arc<Mutex<Manifest>>,
        compression: CompressionAlgo,
    ) -> Self {
        let (tx, rx) = mpsc::channel();

        let handle = thread::Builder::new()
            .name("moonclip-bg-merger".into())
            .spawn(move || {
                for cmd in rx {
                    match cmd {
                        MergeCommand::CheckAndMerge => {
                            if let Err(e) =
                                do_stride_merge(&config, &storage, &manifest, &compression)
                            {
                                eprintln!("[Moonclip merger] stride merge error: {e}");
                            }
                        }
                        MergeCommand::ForceFullMerge => {
                            if let Err(e) =
                                do_full_merge(&storage, &manifest, &compression)
                            {
                                eprintln!("[Moonclip merger] full merge error: {e}");
                            }
                        }
                        MergeCommand::Shutdown => break,
                    }
                }
            })
            .expect("Failed to spawn merger thread");

        DeltaMerger {
            sender: Some(Mutex::new(tx)),
            handle: Some(handle),
        }
    }

    pub fn notify(&self) {
        if let Some(ref tx) = self.sender {
            let _ = tx.lock().unwrap().send(MergeCommand::CheckAndMerge);
        }
    }

    pub fn force_full_merge(&self) {
        if let Some(ref tx) = self.sender {
            let _ = tx.lock().unwrap().send(MergeCommand::ForceFullMerge);
        }
    }

    pub fn shutdown(&mut self) {
        if let Some(ref tx) = self.sender {
            let _ = tx.lock().unwrap().send(MergeCommand::Shutdown);
        }
        self.sender.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for DeltaMerger {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Merge `stride` consecutive deltas into one merged delta.
fn do_stride_merge(
    config: &MergerConfig,
    storage: &Arc<dyn StorageBackend>,
    manifest_lock: &Arc<Mutex<Manifest>>,
    compression: &CompressionAlgo,
) -> Result<()> {
    let manifest = manifest_lock.lock().unwrap();
    let delta_count = manifest.pending_delta_count();

    if delta_count < config.stride {
        return Ok(());
    }

    if delta_count >= config.max_chain_depth {
        drop(manifest);
        return do_full_merge(storage, manifest_lock, compression);
    }

    Ok(())
}

/// Merge all deltas into the base, producing a new full snapshot.
fn do_full_merge(
    storage: &Arc<dyn StorageBackend>,
    manifest_lock: &Arc<Mutex<Manifest>>,
    compression: &CompressionAlgo,
) -> Result<()> {
    let manifest = manifest_lock.lock().unwrap();

    let base_snap = manifest
        .last_full_snapshot()
        .cloned()
        .ok_or_else(|| MoonclipError::NotFound("No full snapshot".into()))?;

    let base_id = base_snap.id;

    let deltas: Vec<Snapshot> = manifest
        .snapshots
        .iter()
        .filter(|s| s.base_snapshot_id == Some(base_id) && s.finalized)
        .cloned()
        .collect();

    if deltas.is_empty() {
        return Ok(());
    }

    drop(manifest);

    let merged_id = Uuid::new_v4();
    let merged_dir = format!("snapshots/{}", merged_id);
    let mut merged_ranks: HashMap<u32, RankEntry> = HashMap::new();

    let mut all_ranks: Vec<u32> = base_snap.ranks.keys().cloned().collect();
    all_ranks.sort();

    for &rank in &all_ranks {
        let base_rank = match base_snap.ranks.get(&rank) {
            Some(r) => r,
            None => continue,
        };

        let mut merged_tensors = Vec::new();
        let mut total_compressed = 0u64;
        let mut total_raw = 0u64;

        for base_tensor in &base_rank.tensors {
            // What the delta chain does to this tensor after the base.
            let chain: Vec<&TensorEntry> = deltas
                .iter()
                .filter_map(|d| d.ranks.get(&rank))
                .filter_map(|r| r.tensors.iter().find(|t| t.name == base_tensor.name))
                .collect();

            // An alias holds no bytes of its own. Unless the chain later
            // overwrites it, carry the reference through — the tensor it
            // points at is materialized in this same rank entry.
            let stays_alias = base_tensor.storage == TensorStorage::Alias
                && chain.iter().all(|t| {
                    matches!(t.storage, TensorStorage::Skipped | TensorStorage::Alias)
                });
            if stays_alias {
                let latest = chain
                    .iter()
                    .rev()
                    .find(|t| t.storage == TensorStorage::Alias)
                    .copied()
                    .unwrap_or(base_tensor);
                total_raw += latest.raw_size;
                merged_tensors.push(latest.clone());
                continue;
            }

            let mut current_data = if base_tensor.storage == TensorStorage::Alias {
                // A Full later in the chain replaces this wholesale; a
                // delta can never target an alias, since deltas are only
                // computed against Full bases.
                Vec::new()
            } else {
                load_tensor_data(base_tensor, &base_snap, storage, compression)?
            };

            for delta_snap in &deltas {
                if let Some(delta_rank) = delta_snap.ranks.get(&rank) {
                    if let Some(delta_tensor) = delta_rank
                        .tensors
                        .iter()
                        .find(|t| t.name == base_tensor.name)
                    {
                        match delta_tensor.storage {
                            // Unchanged, or a reference whose bytes are
                            // materialized under the target's own name.
                            TensorStorage::Skipped | TensorStorage::Alias => {}
                            TensorStorage::Full => {
                                current_data = load_tensor_data(
                                    delta_tensor,
                                    delta_snap,
                                    storage,
                                    compression,
                                )?;
                            }
                            TensorStorage::DeltaXor => {
                                let compressed = load_compressed_data(delta_tensor, delta_snap, storage)?;
                                let delta_bytes = compression::decompress(&compressed, compression)?;
                                current_data = delta::apply_delta(&current_data, &delta_bytes)?;
                            }
                        }
                    }
                }
            }

            // Store as full
            let raw_hash = hash_hex(&current_data);
            let compressed = compression::compress(&current_data, compression)?;
            let filename = format!(
                "{}/rank_{}/{}.bin",
                merged_dir,
                rank,
                base_tensor.name.replace('/', "__").replace(' ', "_")
            );
            storage.put(&filename, &compressed)?;

            let compressed_hash = hash_hex(&compressed);
            total_compressed += compressed.len() as u64;
            total_raw += current_data.len() as u64;

            merged_tensors.push(TensorEntry {
                name: base_tensor.name.clone(),
                shape: base_tensor.shape.clone(),
                dtype: base_tensor.dtype.clone(),
                storage: TensorStorage::Full,
                alias_of: None,
                filename: Some(filename),
                offset: 0,
                compressed_size: compressed.len() as u64,
                raw_size: current_data.len() as u64,
                sha256_raw: raw_hash,
                sha256_compressed: Some(compressed_hash),
                original_dtype: None,
            });
        }

        let alias_count = merged_tensors
            .iter()
            .filter(|t| t.storage == TensorStorage::Alias)
            .count();
        let full_count = merged_tensors.len() - alias_count;
        merged_ranks.insert(
            rank,
            RankEntry {
                rank,
                tensors: merged_tensors,
                pack_file: None, // Merger still uses individual files
                total_compressed,
                total_raw,
                skipped_count: alias_count,
                delta_count: 0,
                full_count,
            },
        );
    }

    let merged_snap = Snapshot {
        id: merged_id,
        step: deltas.last().unwrap().step,
        created_at: chrono::Utc::now(),
        ranks: merged_ranks,
        base_snapshot_id: None,
        metadata: {
            let mut m = deltas.last().unwrap().metadata.clone();
            m.insert("full_merge".into(), "true".into());
            m.insert("merged_deltas".into(), format!("{}", deltas.len()));
            m
        },
        compression: compression.clone(),
        finalized: true,
    };

    let mut manifest = manifest_lock.lock().unwrap();
    let delta_ids: Vec<Uuid> = deltas.iter().map(|s| s.id).collect();

    // Delete old files
    for snap in deltas.iter().chain(std::iter::once(&base_snap)) {
        for rank_entry in snap.ranks.values() {
            if let Some(ref pack_file) = rank_entry.pack_file {
                let _ = storage.delete(pack_file);
            }
            for tensor in &rank_entry.tensors {
                if let Some(ref filename) = tensor.filename {
                    let _ = storage.delete(filename);
                }
            }
        }
    }

    manifest
        .snapshots
        .retain(|s| !delta_ids.contains(&s.id) && s.id != base_id);
    manifest.snapshots.push(merged_snap);
    manifest.snapshots.sort_by_key(|s| s.step);

    let json = serde_json::to_vec_pretty(&*manifest)
        .map_err(|e| MoonclipError::Serialization(e.to_string()))?;
    storage.put("manifest.json", &json)?;

    Ok(())
}

/// Load tensor compressed data, supporting both pack files and legacy individual files.
fn load_compressed_data(
    entry: &TensorEntry,
    snap: &Snapshot,
    storage: &Arc<dyn StorageBackend>,
) -> Result<Vec<u8>> {
    // Try individual file first (merger legacy + old snapshots)
    if let Some(ref filename) = entry.filename {
        let mut data = storage.get(filename)?;
        let expected = entry.compressed_size as usize;
        if expected > 0 && data.len() > expected {
            data.truncate(expected);
        }
        return Ok(data);
    }

    // Try pack file
    let rank_entry = snap.ranks.values().find(|re| {
        re.tensors.iter().any(|t| t.name == entry.name)
    });

    if let Some(re) = rank_entry {
        if let Some(ref pack_file) = re.pack_file {
            let pack_data = storage.get(pack_file)?;
            let start = entry.offset as usize;
            let end = start + entry.compressed_size as usize;
            if end <= pack_data.len() {
                return Ok(pack_data[start..end].to_vec());
            }
        }
    }

    Err(MoonclipError::NotFound(format!(
        "No data found for tensor '{}'", entry.name
    )))
}

fn load_tensor_data(
    entry: &TensorEntry,
    snap: &Snapshot,
    storage: &Arc<dyn StorageBackend>,
    compression: &CompressionAlgo,
) -> Result<Vec<u8>> {
    match entry.storage {
        TensorStorage::Full => {
            let compressed = load_compressed_data(entry, snap, storage)?;
            compression::decompress(&compressed, compression)
        }
        _ => Err(MoonclipError::Delta(format!(
            "Merger can only process Full tensors, got {:?} for '{}'",
            entry.storage, entry.name
        ))),
    }
}
