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
use crate::pack;
use crate::storage::StorageBackend;

/// Configuration for the delta merger.
///
/// # Known defect
///
/// [`do_full_merge`] rebuilds a merged snapshot by walking the **base**
/// snapshot's tensor list, so the merge only ever knows about the tensors that
/// existed at the base. A tensor that first appears in a later delta is not in
/// that list and is dropped; a tensor removed after the base is resurrected
/// from it. The merged snapshot loads without complaint either way — the
/// parameters are simply gone.
///
/// This is safe when the set of tensor names is fixed for the whole run, which
/// covers ordinary training. It is not safe for anything that grows or prunes
/// the model between checkpoints. Merging is opt-in for that reason: the
/// Python constructor defaults `merge_stride` to 0, which leaves this
/// unconstructed.
#[derive(Debug, Clone)]
pub struct MergerConfig {
    /// After `stride` consecutive deltas, fold them into the newest one.
    ///
    /// The fold costs nothing to perform and costs the intermediate steps as
    /// restore points: a run set to `3` keeps roughly every third checkpoint.
    /// Read it as how coarse the history may become. See [`do_stride_merge`].
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
                    // Per command, not around the loop: `install` blocks a pool
                    // worker for as long as the closure runs, and this thread
                    // spends nearly all its life waiting on `rx`.
                    match cmd {
                        MergeCommand::CheckAndMerge => {
                            if let Err(e) = crate::pool::install(|| {
                                do_stride_merge(&config, &storage, &manifest, &compression)
                            }) {
                                eprintln!("[Moonclip merger] stride merge error: {e}");
                            }
                        }
                        MergeCommand::ForceFullMerge => {
                            if let Err(e) = crate::pool::install(|| {
                                do_full_merge(&storage, &manifest, &compression)
                            }) {
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

/// Fold a run of `stride` consecutive deltas into the newest of them, and
/// collapse into the base once the run passes `max_chain_depth`.
///
/// **A delta chain is not a chain here.** Every delta is computed against the
/// last *full* snapshot, never against the delta before it — `save_sync` takes
/// its base from `last_full_snapshot()`, and a full is the only thing that can
/// be one. So each delta describes the entire state relative to that same base,
/// and a load reads exactly two snapshots no matter how many have accumulated.
///
/// That is what makes the fold nearly free. Merging D1..Dk means keeping Dk,
/// which already supersedes the others completely, and dropping the rest: no
/// bytes are read, rewritten or recompressed. What storage stops holding is k
/// diffs against one base where one of them answers every question the others
/// could.
///
/// **What it costs is history.** Those snapshots are restore points, and after
/// the fold they are gone: with `stride: 3` a run keeps roughly every third
/// checkpoint rather than every one. Merging N snapshots into one means losing
/// N-1 restore points in any format — it is why the merger is opt-in, and why
/// `stride` should be read as "how coarse may the history become".
///
/// Rollback snapshots are never at risk: `rollback_snapshot_ids` marks fulls
/// only, and this touches nothing but deltas.
fn do_stride_merge(
    config: &MergerConfig,
    storage: &Arc<dyn StorageBackend>,
    manifest_lock: &Arc<Mutex<Manifest>>,
    compression: &CompressionAlgo,
) -> Result<()> {
    let mut manifest = manifest_lock.lock().unwrap();
    let delta_count = manifest.pending_delta_count();

    // The depth limit is tested first so that it holds whatever `stride` is
    // set to. Tested second, a `stride` larger than `max_chain_depth` returned
    // early every time and the depth cap could never fire.
    if delta_count >= config.max_chain_depth {
        drop(manifest);
        return do_full_merge(storage, manifest_lock, compression);
    }

    if config.stride == 0 || delta_count < config.stride {
        return Ok(());
    }

    // The trailing run of deltas, oldest first.
    let tail_start = manifest.snapshots.len() - delta_count;
    let tail = &manifest.snapshots[tail_start..];

    // The survivor is the newest finalized delta. An unfinalized snapshot is
    // one other ranks are still writing into: not this thread's to judge, and
    // not something to fold a history into.
    let Some(survivor) = tail.iter().rev().find(|s| s.finalized) else {
        return Ok(());
    };

    // Only deltas against the same base are superseded by it. Deltas against
    // some other base describe a different state, and dropping one would be
    // discarding history rather than folding it.
    let superseded: Vec<Uuid> = tail
        .iter()
        .filter(|s| {
            s.finalized
                && s.id != survivor.id
                && s.base_snapshot_id == survivor.base_snapshot_id
        })
        .map(|s| s.id)
        .collect();

    if superseded.is_empty() {
        return Ok(());
    }

    let doomed: Vec<Snapshot> = manifest
        .snapshots
        .iter()
        .filter(|s| superseded.contains(&s.id))
        .cloned()
        .collect();

    // Publish first, delete after — for the reason spelled out in
    // `do_full_merge`: a crash between the two must never leave the manifest
    // naming snapshots whose bytes are already gone.
    let previous = std::mem::take(&mut manifest.snapshots);
    manifest.snapshots = previous
        .iter()
        .filter(|s| !superseded.contains(&s.id))
        .cloned()
        .collect();

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

    for snap in &doomed {
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
        let _ = storage.remove_dir(&format!("snapshots/{}", snap.id));
    }

    Ok(())
}

/// Merge all deltas into the base, producing a new full snapshot.
///
/// The result is written the way the save path writes one: a pack per rank,
/// carrying its own [`pack::PackDescriptor`]. It has to be. Startup recovery
/// (`recover_or_reclaim_orphans` in `crate::coordinator`) rebuilds a snapshot
/// the manifest has lost from the descriptors in its packs, and *deletes*
/// whatever it cannot rebuild. A merge that wrote loose files would produce
/// exactly the snapshot that recovery discards — and the merge is the moment
/// the run's entire history collapses into that one snapshot, so it is the
/// worst one to lose.
pub(crate) fn do_full_merge(
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

    // Fixed before the first pack is written, because every rank's descriptor
    // repeats them: a reader that finds two ranks describing different
    // snapshots has to reject the whole thing.
    let last_delta = deltas.last().unwrap();
    let merged_step = last_delta.step;
    let merged_created_at = chrono::Utc::now();
    let merged_metadata = {
        let mut m = last_delta.metadata.clone();
        m.insert("full_merge".into(), "true".into());
        m.insert("merged_deltas".into(), format!("{}", deltas.len()));
        m
    };
    let world_size = all_ranks.len() as u32;

    for &rank in &all_ranks {
        let base_rank = match base_snap.ranks.get(&rank) {
            Some(r) => r,
            None => continue,
        };

        let mut merged_tensors = Vec::new();
        // The compressed blobs, held until the whole rank is known: the pack's
        // descriptor records every offset, so it cannot be written before the
        // last tensor has one.
        let mut blobs: Vec<Vec<u8>> = Vec::new();
        let mut offset = pack::HEADER_LEN;
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
                load_tensor_data(base_tensor, &base_snap, rank, storage, compression)?
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
                                    rank,
                                    storage,
                                    compression,
                                )?;
                            }
                            TensorStorage::DeltaXor => {
                                let compressed =
                                    load_compressed_data(delta_tensor, delta_snap, rank, storage)?;
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

            // Store as full, at its place in the pack.
            let raw_hash = hash_hex(&current_data);
            let compressed = compression::compress(&current_data, compression)?;
            let compressed_hash = hash_hex(&compressed);
            total_compressed += compressed.len() as u64;
            total_raw += current_data.len() as u64;

            merged_tensors.push(TensorEntry {
                name: base_tensor.name.clone(),
                shape: base_tensor.shape.clone(),
                dtype: base_tensor.dtype.clone(),
                storage: TensorStorage::Full,
                alias_of: None,
                filename: None,
                offset,
                compressed_size: compressed.len() as u64,
                raw_size: current_data.len() as u64,
                hash_raw: raw_hash,
                hash_compressed: Some(compressed_hash),
                shuffled: false,
                original_dtype: None,
            });

            offset += compressed.len() as u64;
            blobs.push(compressed);
        }

        let blobs_end = offset;
        let alias_count = merged_tensors
            .iter()
            .filter(|t| t.storage == TensorStorage::Alias)
            .count();
        let full_count = merged_tensors.len() - alias_count;

        // A rank of nothing but aliases holds no bytes, so there is no pack to
        // write — the same call the save path makes when a snapshot changed
        // nothing.
        let pack_file = (!blobs.is_empty()).then(|| format!("{}/rank_{}.pack", merged_dir, rank));

        let rank_entry = RankEntry {
            rank,
            tensors: merged_tensors,
            pack_file,
            total_compressed,
            total_raw,
            skipped_count: alias_count,
            delta_count: 0,
            full_count,
        };

        if let Some(ref pack_filename) = rank_entry.pack_file {
            let descriptor = pack::PackDescriptor {
                snapshot_id: merged_id,
                step: merged_step,
                created_at: merged_created_at,
                // A merge produces a full: there is no base left to apply.
                base_snapshot_id: None,
                metadata: merged_metadata.clone(),
                compression: compression.clone(),
                world_size,
                rank: rank_entry.clone(),
            };
            let encoded = serde_json::to_vec(&descriptor)
                .map_err(|e| MoonclipError::Serialization(e.to_string()))?;
            let header = pack::encode_header(blobs_end, encoded.len() as u64);

            let mut parts: Vec<&[u8]> = Vec::with_capacity(blobs.len() + 2);
            parts.push(&header);
            parts.extend(blobs.iter().map(|b| b.as_slice()));
            parts.push(&encoded);
            storage.put_parts(pack_filename, &parts)?;
        }

        merged_ranks.insert(rank, rank_entry);
    }

    let merged_snap = Snapshot {
        id: merged_id,
        step: merged_step,
        created_at: merged_created_at,
        ranks: merged_ranks,
        base_snapshot_id: None,
        metadata: merged_metadata,
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
///
/// `rank` is passed rather than searched for. Looking the tensor up by name
/// across `snap.ranks` finds *a* rank holding that name, and under sharding
/// every rank holds the same names — so on a multi-rank snapshot it read
/// another rank's shard from another rank's pack, at an offset that means
/// nothing there.
fn load_compressed_data(
    entry: &TensorEntry,
    snap: &Snapshot,
    rank: u32,
    storage: &Arc<dyn StorageBackend>,
) -> Result<Vec<u8>> {
    // Try individual file first (snapshots from before the pack format)
    if let Some(ref filename) = entry.filename {
        let mut data = storage.get(filename)?;
        let expected = entry.compressed_size as usize;
        if expected > 0 && data.len() > expected {
            data.truncate(expected);
        }
        return Ok(data);
    }

    if let Some(re) = snap.ranks.get(&rank) {
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
    rank: u32,
    storage: &Arc<dyn StorageBackend>,
    compression: &CompressionAlgo,
) -> Result<Vec<u8>> {
    match entry.storage {
        TensorStorage::Full => {
            let compressed = load_compressed_data(entry, snap, rank, storage)?;
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
        step: u64,
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
                hash_raw: hash_hex(raw),
                hash_compressed: Some(hash_hex(&compressed)),
                shuffled: false,
            });
            offset += compressed.len() as u64;
            parts.push(compressed);
        }

        let refs: Vec<&[u8]> = parts.iter().map(|p| p.as_slice()).collect();
        storage.put_parts(&pack, &refs).unwrap();

        Snapshot {
            id,
            step,
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

    /// A base full snapshot plus `deltas` snapshots, each one a complete XOR
    /// against that same base — which is how the save path writes them, since
    /// `last_full_snapshot()` is the only thing that can be a base.
    ///
    /// Returns the manifest and what each tensor must read back as *at the
    /// newest delta*.
    fn delta_run(
        storage: &dyn StorageBackend,
        count: usize,
        size: usize,
        deltas: usize,
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
        let base = packed_snapshot(storage, &base_tensors, None, 0);

        let mut snapshots = vec![base];
        let base_id = snapshots[0].id;
        let mut finals = Vec::new();

        for d in 1..=deltas {
            finals.clear();
            let delta_tensors: Vec<(String, Vec<u8>, TensorStorage)> = base_tensors
                .iter()
                .enumerate()
                .map(|(i, (name, raw, _))| {
                    let mut updated = raw.clone();
                    updated[(i + d) % size] ^= 0xa5;
                    updated[size / 2] ^= d as u8;
                    let xor = delta::compute_delta(raw, &updated).unwrap();
                    finals.push(updated);
                    (name.clone(), xor, TensorStorage::DeltaXor)
                })
                .collect();
            snapshots.push(packed_snapshot(
                storage,
                &delta_tensors,
                Some(base_id),
                d as u64,
            ));
        }

        (
            Manifest {
                snapshots,
                ..Default::default()
            },
            finals,
        )
    }

    /// A base full snapshot plus one delta that rewrites every tensor.
    fn scenario(
        storage: &dyn StorageBackend,
        count: usize,
        size: usize,
    ) -> (Manifest, Vec<Vec<u8>>) {
        delta_run(storage, count, size, 1)
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
            let got = load_tensor_data(entry, merged, 0, &storage, &ZSTD3).unwrap();
            assert_eq!(&got, want, "tensor w{i} must survive the merge intact");
        }
    }

    /// A merged snapshot has to describe itself the way a saved one does.
    /// Without the descriptor it is data the startup recovery cannot rebuild —
    /// and therefore data it deletes — at the one moment the whole run's
    /// history has just been collapsed into a single snapshot.
    #[test]
    fn a_merged_snapshot_carries_its_own_descriptor() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let (manifest, expected) = scenario(storage.as_ref(), 3, 30_000);
        let manifest = Arc::new(Mutex::new(manifest));

        do_full_merge(&storage, &manifest, &ZSTD3).unwrap();

        let m = manifest.lock().unwrap();
        let merged = &m.snapshots[0];
        let rank = merged.ranks.get(&0).unwrap();
        let pack_file = rank
            .pack_file
            .clone()
            .expect("a merge must produce a pack, not loose files");

        let header = storage
            .get_range(&pack_file, 0, pack::HEADER_LEN as usize)
            .unwrap();
        let (offset, length) = pack::decode_header(&header).expect("our own header");
        let encoded = storage.get_range(&pack_file, offset, length as usize).unwrap();
        let described: pack::PackDescriptor = serde_json::from_slice(&encoded).unwrap();

        assert_eq!(described.snapshot_id, merged.id);
        assert_eq!(described.step, merged.step);
        assert_eq!(described.base_snapshot_id, None, "a merge yields a full");
        assert_eq!(described.world_size, 1);
        assert_eq!(described.metadata["full_merge"], "true");

        // The description has to be usable on its own: reading each tensor
        // through the descriptor's offsets, with the manifest ignored, must
        // return what the merge computed.
        for (i, want) in expected.iter().enumerate() {
            let entry = described
                .rank
                .tensors
                .iter()
                .find(|t| t.name == format!("w{i}"))
                .unwrap();
            let bytes = storage
                .get_range(&pack_file, entry.offset, entry.compressed_size as usize)
                .unwrap();
            assert_eq!(hash_hex(&bytes), entry.hash_compressed.clone().unwrap());
            assert_eq!(&compression::decompress(&bytes, &ZSTD3).unwrap(), want);
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

    // ── Stride merge ────────────────────────────────────────────────

    fn stride_config(stride: usize, max_chain_depth: usize) -> MergerConfig {
        MergerConfig {
            stride,
            max_chain_depth,
        }
    }

    /// The fold itself: `stride` deltas in, one out — the newest, which already
    /// describes the whole state against the same base.
    #[test]
    fn a_stride_merge_folds_the_run_into_its_newest_delta() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let (manifest, expected) = delta_run(storage.as_ref(), 3, 20_000, 3);

        let base_id = manifest.snapshots[0].id;
        let newest = manifest.snapshots[3].id;
        let dropped: Vec<String> = manifest.snapshots[1..3]
            .iter()
            .filter_map(|s| s.ranks.get(&0).and_then(|r| r.pack_file.clone()))
            .collect();
        assert_eq!(dropped.len(), 2);

        let manifest = Arc::new(Mutex::new(manifest));
        do_stride_merge(&stride_config(3, 10), &storage, &manifest, &ZSTD3).unwrap();

        {
            let m = manifest.lock().unwrap();
            let ids: Vec<Uuid> = m.snapshots.iter().map(|s| s.id).collect();
            assert_eq!(ids, vec![base_id, newest], "only the newest delta stays");
        }
        for pack in &dropped {
            assert!(
                !storage.exists(pack).unwrap(),
                "{pack} was superseded but its bytes are still on disk"
            );
        }

        // The survivor has to still reconstruct. It is a XOR against the base,
        // and nothing it needs was in the snapshots just dropped — that is the
        // whole claim the fold rests on.
        do_full_merge(&storage, &manifest, &ZSTD3).unwrap();
        let m = manifest.lock().unwrap();
        let merged = &m.snapshots[0];
        let rank = merged.ranks.get(&0).unwrap();
        for (i, want) in expected.iter().enumerate() {
            let entry = rank
                .tensors
                .iter()
                .find(|t| t.name == format!("w{i}"))
                .unwrap();
            let got = load_tensor_data(entry, merged, 0, &storage, &ZSTD3).unwrap();
            assert_eq!(&got, want, "w{i} must read back as the newest delta wrote it");
        }
    }

    #[test]
    fn a_run_shorter_than_the_stride_is_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let (manifest, _) = delta_run(storage.as_ref(), 2, 10_000, 2);
        let before: Vec<Uuid> = manifest.snapshots.iter().map(|s| s.id).collect();

        let manifest = Arc::new(Mutex::new(manifest));
        do_stride_merge(&stride_config(3, 10), &storage, &manifest, &ZSTD3).unwrap();

        let m = manifest.lock().unwrap();
        let after: Vec<Uuid> = m.snapshots.iter().map(|s| s.id).collect();
        assert_eq!(before, after, "two deltas is not a stride of three");
    }

    /// Regression: the depth limit was tested after the stride, so a `stride`
    /// larger than `max_chain_depth` returned early every time and the depth
    /// cap — the only thing bounding how much a load has to reconstruct —
    /// could never fire.
    #[test]
    fn the_depth_limit_fires_even_with_a_larger_stride() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let (manifest, _) = delta_run(storage.as_ref(), 2, 10_000, 3);

        let manifest = Arc::new(Mutex::new(manifest));
        do_stride_merge(&stride_config(20, 3), &storage, &manifest, &ZSTD3).unwrap();

        let m = manifest.lock().unwrap();
        assert_eq!(m.snapshots.len(), 1, "the depth limit should have merged");
        assert!(
            m.snapshots[0].base_snapshot_id.is_none(),
            "a full merge yields a full"
        );
    }

    /// A snapshot other ranks are still writing into is not this thread's to
    /// fold away, and must not become the survivor either.
    #[test]
    fn an_unfinalized_delta_is_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let (mut manifest, _) = delta_run(storage.as_ref(), 2, 10_000, 3);

        let in_flight = manifest.snapshots[3].id;
        manifest.snapshots[3].finalized = false;
        let survivor = manifest.snapshots[2].id;
        let base_id = manifest.snapshots[0].id;

        let manifest = Arc::new(Mutex::new(manifest));
        do_stride_merge(&stride_config(3, 10), &storage, &manifest, &ZSTD3).unwrap();

        let m = manifest.lock().unwrap();
        let ids: Vec<Uuid> = m.snapshots.iter().map(|s| s.id).collect();
        assert_eq!(
            ids,
            vec![base_id, survivor, in_flight],
            "the newest *finalized* delta survives, and the in-flight one is untouched"
        );
    }

    /// Same discipline as the full merge: nothing is deleted until the manifest
    /// that stops naming it is durable.
    #[test]
    fn a_failed_manifest_write_keeps_the_superseded_deltas() {
        let dir = tempfile::tempdir().unwrap();
        let mut counting = CountingStorage::new(LocalStorage::new(dir.path()).unwrap());
        counting.fail_put = Some("manifest.json".into());
        let storage: Arc<dyn StorageBackend> = Arc::new(counting);

        let (manifest, _) = delta_run(storage.as_ref(), 2, 10_000, 3);
        let before: Vec<Uuid> = manifest.snapshots.iter().map(|s| s.id).collect();
        let packs: Vec<String> = manifest
            .snapshots
            .iter()
            .filter_map(|s| s.ranks.get(&0).and_then(|r| r.pack_file.clone()))
            .collect();

        let manifest = Arc::new(Mutex::new(manifest));
        let result = do_stride_merge(&stride_config(3, 10), &storage, &manifest, &ZSTD3);
        assert!(result.is_err(), "the injected manifest write must fail");

        for pack in &packs {
            assert!(
                storage.exists(pack).unwrap(),
                "{pack} was deleted before the new manifest was durable"
            );
        }
        let m = manifest.lock().unwrap();
        let after: Vec<Uuid> = m.snapshots.iter().map(|s| s.id).collect();
        assert_eq!(
            before, after,
            "the in-memory manifest must still describe what is on disk"
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
