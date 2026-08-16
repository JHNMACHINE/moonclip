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

/// Collapse the delta chain once it grows past `max_chain_depth`.
///
/// **The stride merge itself is not implemented.** The name and `stride` came
/// from a design where every `stride` deltas were combined into one merged
/// delta, keeping load-time chains short without rewriting the base. Nothing
/// does that: between `stride` and `max_chain_depth` this returns having done
/// nothing, and the only thing that ever runs is the full merge at the depth
/// limit. `stride` therefore acts as an on/off switch, not a stride.
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
                                let mut delta_bytes =
                                    compression::decompress(&compressed, compression)?;
                                if delta_tensor.shuffled {
                                    delta_bytes = crate::shuffle::unshuffle(
                                        &delta_bytes,
                                        crate::shuffle::element_size(&delta_tensor.dtype),
                                    );
                                }
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
                shuffled: false,
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

    // Publish the new manifest *before* deleting anything it replaces.
    //
    // The other order — delete, then record — has a window in which the
    // snapshots are gone from disk but still the only thing manifest.json
    // names. A crash there does not cost a merge, it costs the entire
    // training run: every checkpoint the manifest points at has been erased,
    // and the merged replacement is not yet referenced by anything.
    //
    // The old list is kept so a failed write leaves the in-memory manifest
    // describing what is actually on disk.
    let previous = std::mem::take(&mut manifest.snapshots);
    manifest.snapshots = previous
        .iter()
        .filter(|s| !delta_ids.contains(&s.id) && s.id != base_id)
        .cloned()
        .collect();
    manifest.snapshots.push(merged_snap);
    manifest.snapshots.sort_by_key(|s| s.step);

    let json = match serde_json::to_vec_pretty(&*manifest) {
        Ok(j) => j,
        Err(e) => {
            manifest.snapshots = previous;
            return Err(MoonclipError::Serialization(e.to_string()));
        }
    };
    if let Err(e) = storage.put("manifest.json", &json) {
        manifest.snapshots = previous;
        return Err(e);
    }
    drop(manifest);

    // Durable and unreferenced: deleting these can no longer lose data a
    // reader could reach.
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
            // By range, never the whole pack. This is called once per tensor
            // per snapshot in the merge loop, so reading the entire file here
            // makes a merge O(tensors x snapshots) passes over the whole
            // checkpoint — gigabytes of reads to rewrite megabytes. The save
            // path's BaseCache has always read the pack this way.
            let want = entry.compressed_size as usize;
            let data = storage.get_range(pack_file, entry.offset, want)?;
            if data.len() == want {
                return Ok(data);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::LocalStorage;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const ZSTD3: CompressionAlgo = CompressionAlgo::Zstd { level: 3 };

    /// Deterministic pseudo-random bytes. Real tensors barely compress, and a
    /// merger test on runs of constant bytes would hide any amount of
    /// redundant I/O behind a pack that fits in a single page.
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

    /// Wraps a real backend and counts reads, so a test can state how much
    /// I/O a merge is allowed to do and not only what it produces.
    struct CountingStorage {
        inner: LocalStorage,
        whole_file_reads: AtomicUsize,
        /// When set, `put` of this path fails — used to land a crash in the
        /// window between dropping the old snapshots and recording the new one.
        fail_put: Option<String>,
    }

    impl CountingStorage {
        fn new(inner: LocalStorage) -> Self {
            CountingStorage {
                inner,
                whole_file_reads: AtomicUsize::new(0),
                fail_put: None,
            }
        }
    }

    impl StorageBackend for CountingStorage {
        fn put(&self, rel_path: &str, data: &[u8]) -> Result<()> {
            if self.fail_put.as_deref() == Some(rel_path) {
                return Err(MoonclipError::Storage(format!(
                    "injected failure: {rel_path}"
                )));
            }
            self.inner.put(rel_path, data)
        }
        fn get(&self, rel_path: &str) -> Result<Vec<u8>> {
            self.whole_file_reads.fetch_add(1, Ordering::Relaxed);
            self.inner.get(rel_path)
        }
        fn get_range(&self, rel_path: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
            self.inner.get_range(rel_path, offset, len)
        }
        fn exists(&self, rel_path: &str) -> Result<bool> {
            self.inner.exists(rel_path)
        }
        fn delete(&self, rel_path: &str) -> Result<()> {
            self.inner.delete(rel_path)
        }
        fn list(&self, prefix: &str) -> Result<Vec<String>> {
            self.inner.list(prefix)
        }
    }

    /// Write `tensors` as one packed snapshot, the way the save path does.
    fn packed_snapshot(
        storage: &dyn StorageBackend,
        tensors: &[(String, Vec<u8>, TensorStorage)],
        base: Option<Uuid>,
    ) -> Snapshot {
        let id = Uuid::new_v4();
        let pack = format!("snapshots/{id}/rank_0.pack");

        let mut entries = Vec::new();
        let mut parts: Vec<Vec<u8>> = Vec::new();
        let mut offset = 0u64;
        for (name, raw, kind) in tensors {
            let compressed = compression::compress(raw, &ZSTD3).unwrap();
            entries.push(TensorEntry {
                name: name.clone(),
                shape: vec![raw.len()],
                dtype: "uint8".into(),
                original_dtype: None,
                storage: kind.clone(),
                alias_of: None,
                filename: None,
                offset,
                compressed_size: compressed.len() as u64,
                raw_size: raw.len() as u64,
                sha256_raw: hash_hex(raw),
                sha256_compressed: Some(hash_hex(&compressed)),
                shuffled: false,
            });
            offset += compressed.len() as u64;
            parts.push(compressed);
        }

        let refs: Vec<&[u8]> = parts.iter().map(|p| p.as_slice()).collect();
        storage.put_parts(&pack, &refs).unwrap();

        Snapshot {
            id,
            step: if base.is_some() { 1 } else { 0 },
            created_at: chrono::Utc::now(),
            ranks: HashMap::from([(
                0u32,
                RankEntry {
                    rank: 0,
                    tensors: entries,
                    pack_file: Some(pack),
                    total_compressed: offset,
                    total_raw: tensors.iter().map(|(_, r, _)| r.len() as u64).sum(),
                    skipped_count: 0,
                    delta_count: 0,
                    full_count: tensors.len(),
                },
            )]),
            base_snapshot_id: base,
            metadata: HashMap::new(),
            compression: ZSTD3,
            finalized: true,
        }
    }

    /// A base full snapshot plus one delta that rewrites every tensor.
    /// Returns the manifest and what each tensor must read back as.
    fn scenario(
        storage: &dyn StorageBackend,
        count: usize,
        size: usize,
    ) -> (Manifest, Vec<Vec<u8>>) {
        let base_tensors: Vec<(String, Vec<u8>, TensorStorage)> = (0..count)
            .map(|i| {
                (
                    format!("w{i}"),
                    noise(size, i as u64 + 1),
                    TensorStorage::Full,
                )
            })
            .collect();
        let base = packed_snapshot(storage, &base_tensors, None);

        let mut finals = Vec::new();
        let delta_tensors: Vec<(String, Vec<u8>, TensorStorage)> = base_tensors
            .iter()
            .enumerate()
            .map(|(i, (name, raw, _))| {
                let mut updated = raw.clone();
                updated[i % size] ^= 0xa5;
                updated[size / 2] ^= 0x0f;
                let xor = delta::compute_delta(raw, &updated).unwrap();
                finals.push(updated);
                (name.clone(), xor, TensorStorage::DeltaXor)
            })
            .collect();
        let delta_snap = packed_snapshot(storage, &delta_tensors, Some(base.id));

        (
            Manifest {
                snapshots: vec![base, delta_snap],
                ..Default::default()
            },
            finals,
        )
    }

    #[test]
    fn full_merge_reconstructs_the_delta_chain() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let (manifest, expected) = scenario(storage.as_ref(), 4, 40_000);
        let manifest = Arc::new(Mutex::new(manifest));

        do_full_merge(&storage, &manifest, &ZSTD3).unwrap();

        let m = manifest.lock().unwrap();
        assert_eq!(m.snapshots.len(), 1, "base and delta collapse into one");
        let merged = &m.snapshots[0];
        assert!(merged.base_snapshot_id.is_none(), "the result is a full");

        let rank = merged.ranks.get(&0).unwrap();
        for (i, want) in expected.iter().enumerate() {
            let entry = rank
                .tensors
                .iter()
                .find(|t| t.name == format!("w{i}"))
                .unwrap();
            assert_eq!(entry.storage, TensorStorage::Full);
            let got = load_tensor_data(entry, merged, &storage, &ZSTD3).unwrap();
            assert_eq!(&got, want, "tensor w{i} must survive the merge intact");
        }
    }

    #[test]
    fn full_merge_does_not_reread_the_pack_for_every_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let counting = Arc::new(CountingStorage::new(LocalStorage::new(dir.path()).unwrap()));
        let storage: Arc<dyn StorageBackend> = counting.clone();

        let tensors = 16;
        let (manifest, _) = scenario(storage.as_ref(), tensors, 40_000);
        counting.whole_file_reads.store(0, Ordering::Relaxed);

        let manifest = Arc::new(Mutex::new(manifest));
        do_full_merge(&storage, &manifest, &ZSTD3).unwrap();

        // Two packs hold everything the merge needs: the base and the delta.
        // Reading a whole pack per tensor makes a merge O(tensors x snapshots)
        // passes over the entire checkpoint — on a real model, hundreds of
        // gigabytes of reads to rewrite a few.
        let whole = counting.whole_file_reads.load(Ordering::Relaxed);
        assert!(
            whole <= 2,
            "merging {tensors} tensors read {whole} whole files; the pack \
             should be read by range, or once per snapshot"
        );
    }

    #[test]
    fn a_failed_manifest_write_does_not_destroy_the_snapshots() {
        let dir = tempfile::tempdir().unwrap();
        let mut counting = CountingStorage::new(LocalStorage::new(dir.path()).unwrap());
        counting.fail_put = Some("manifest.json".into());
        let storage: Arc<dyn StorageBackend> = Arc::new(counting);

        let (manifest, _) = scenario(storage.as_ref(), 3, 20_000);
        let packs: Vec<String> = manifest
            .snapshots
            .iter()
            .filter_map(|s| s.ranks.get(&0).and_then(|r| r.pack_file.clone()))
            .collect();
        assert_eq!(packs.len(), 2);

        let manifest = Arc::new(Mutex::new(manifest));
        let result = do_full_merge(&storage, &manifest, &ZSTD3);
        assert!(result.is_err(), "the injected manifest write must fail");

        // The merge did not complete, so the snapshots it was merging are
        // still the only record of the run. Deleting them before the
        // replacement is durable leaves a manifest pointing at nothing.
        for pack in &packs {
            assert!(
                storage.exists(pack).unwrap(),
                "{pack} was deleted before the new manifest was durable"
            );
        }
    }
}
