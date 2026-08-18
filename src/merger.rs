use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

use uuid::Uuid;

use crate::compression;
use crate::delta;
use crate::error::{Result, MoonclipError};
use crate::hash::hash_hex;
use crate::inflight::{InFlight, FORCED_CLAIM_WAIT, UNLINK_WAIT};
use crate::manifest::*;
use crate::pack;
use crate::remote_sync::PendingDeletes;
use crate::storage::StorageBackend;

/// What a merge actually did.
///
/// A merge that stood down is not a failure and not a success, and collapsing
/// it into `Ok(())` is how `save_final` came to report a fold that never
/// happened: it merges and then uploads, so "skipped because something was
/// being read" is an answer it has to be able to see. See
/// [`DeltaMerger::force_full_merge`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MergeOutcome {
    /// The snapshots were folded and published.
    Merged,
    /// There was nothing to fold — no deltas against the current base.
    NothingToMerge,
    /// A reader was inside the inputs and would not leave. Opportunistic
    /// merges treat this as "try again on the next notify"; a forced one has
    /// already waited [`FORCED_CLAIM_WAIT`] and must report it.
    Busy,
}

/// Configuration for the delta merger.
///
/// Folding is opt-in — the Python constructor defaults `merge_stride` to 0,
/// which leaves this unconstructed — because it trades restore points for
/// space: N snapshots become one, and the N-1 steps in between stop being
/// checkpoints anyone can go back to.
///
/// Until 0.0.6 it also traded away tensors. [`do_full_merge`] rebuilt the
/// merged snapshot from the *base* snapshot's tensor list, so a parameter the
/// model gained after the base was dropped and one it lost came back, in
/// silence. It walks the newest delta now, which is the state dict as it stood
/// at the step the merge stands in for.
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
    /// The channel, when present, is how the caller waits for the fold to
    /// finish. `save_final` needs that: it merges and then syncs, and a merge
    /// that is still running when the upload starts means the merged snapshot
    /// — the one holding the whole run — is not among the files that go up.
    ForceFullMerge(Option<mpsc::Sender<Result<()>>>),
    Shutdown,
}

pub struct DeltaMerger {
    // Mutex-wrapped so DeltaMerger is Sync (mpsc::Sender is Send but not Sync);
    // the coordinator shares it with the background save thread.
    sender: Option<Mutex<mpsc::Sender<MergeCommand>>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl DeltaMerger {
    /// Crate-internal since 0.0.6: a merger has to share the coordinator's
    /// in-flight registry, or it deletes snapshots out from under saves and
    /// loads that are reading them. There is no way to hand one in from
    /// outside the crate, and no way to make one safely without it.
    pub(crate) fn new(
        config: MergerConfig,
        storage: Arc<dyn StorageBackend>,
        manifest: Arc<Mutex<Manifest>>,
        compression: CompressionAlgo,
        in_flight: Arc<InFlight>,
        pending_deletes: Arc<PendingDeletes>,
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
                                do_stride_merge(&config, &storage, &manifest, &compression, &in_flight, &pending_deletes)
                            }) {
                                eprintln!("[Moonclip merger] stride merge error: {e}");
                            }
                        }
                        MergeCommand::ForceFullMerge(done) => {
                            // A forced merge does not stand down. `save_final`
                            // folds the run and uploads straight afterwards,
                            // so a merge skipped because something was being
                            // read used to be reported as success and the
                            // unmerged chain went to the bucket. Retry until
                            // the reader leaves, then give up loudly.
                            //
                            // The wait is here rather than inside the claim
                            // because this holds no manifest lock — the
                            // closure has returned by then. See
                            // `crate::inflight` on lock order.
                            let deadline = std::time::Instant::now() + FORCED_CLAIM_WAIT;
                            let outcome = loop {
                                let attempt = crate::pool::install(|| {
                                    do_full_merge(&storage, &manifest, &compression, &in_flight, &pending_deletes)
                                });
                                match attempt {
                                    Ok(MergeOutcome::Busy) => {
                                        let left = deadline
                                            .checked_duration_since(std::time::Instant::now());
                                        match left {
                                            Some(left) => in_flight.wait_for_any_unpin(
                                                left.min(std::time::Duration::from_secs(1)),
                                            ),
                                            None => {
                                                break Err(MoonclipError::Storage(
                                                    "Forced merge stood down: something has \
                                                     been reading the snapshots it would fold \
                                                     for longer than the merge waits"
                                                        .into(),
                                                ))
                                            }
                                        }
                                    }
                                    Ok(_) => break Ok(()),
                                    Err(e) => break Err(e),
                                }
                            };
                            match done {
                                // Somebody is waiting on this one, so the
                                // failure is theirs to see rather than the
                                // log's to keep.
                                Some(tx) => {
                                    let _ = tx.send(outcome);
                                }
                                None => {
                                    if let Err(e) = outcome {
                                        eprintln!("[Moonclip merger] full merge error: {e}");
                                    }
                                }
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

    /// Fold everything into one full snapshot and **wait for it**.
    ///
    /// Sending the command and returning is not enough for the one caller that
    /// matters. `save_final` merges and then syncs to remote storage, and the
    /// merge runs on another thread: without the wait, the upload lists the
    /// files while the merge is still writing, and the run's final checkpoint
    /// — every earlier one having just been folded into it — is the one thing
    /// that does not reach the bucket.
    pub(crate) fn force_full_merge(&self) -> Result<()> {
        let Some(ref sender) = self.sender else {
            return Ok(());
        };
        let (tx, rx) = mpsc::channel();
        if sender
            .lock()
            .unwrap()
            .send(MergeCommand::ForceFullMerge(Some(tx)))
            .is_err()
        {
            // The merger thread is gone, so there is nothing to wait for and
            // nothing was merged. Not an error: shutdown races with this.
            return Ok(());
        }
        match rx.recv() {
            Ok(outcome) => outcome,
            // Same case, one step later: the thread died holding the channel.
            Err(_) => Ok(()),
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
    in_flight: &Arc<InFlight>,
    pending_deletes: &Arc<PendingDeletes>,
) -> Result<()> {
    let mut manifest = manifest_lock.lock().unwrap();
    let delta_count = manifest.pending_delta_count();

    // The depth limit is tested first so that it holds whatever `stride` is
    // set to. Tested second, a `stride` larger than `max_chain_depth` returned
    // early every time and the depth cap could never fire.
    if delta_count >= config.max_chain_depth {
        drop(manifest);
        // Opportunistic like the rest of this function: a full merge that
        // stands down because something is being read runs again on the next
        // notify, and the depth cap fires then instead.
        return do_full_merge(storage, manifest_lock, compression, in_flight, pending_deletes)
            .map(|_| ());
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

    // Nothing is folded out from under a reader. Taken while the manifest lock
    // is still held, so the set decided above is the set claimed here — see
    // `crate::inflight`. A refusal is not an error: the next save notifies the
    // merger again, and by then the reader is gone.
    let Some(_claim) = in_flight.claim(&superseded) else {
        return Ok(());
    };

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

    // The manifest lock is gone before this waits, which is not optional: a
    // reader reacquires it *after* pinning, so waiting while holding it would
    // deadlock against the reader being waited for. See `crate::inflight`.
    unlink_snapshots(storage, pending_deletes, in_flight, &doomed, UNLINK_WAIT);

    Ok(())
}

/// Remove the files of snapshots the manifest no longer names.
///
/// **Waits for readers first.** Publishing has already happened, so nothing
/// new can reach these snapshots; what can still be inside them is a load that
/// pinned them before they were dropped, or — the case the claim cannot cover
/// — one that pinned them *after* the merge claimed and before it got here.
/// Unlinking under that reader is the failure this whole mechanism exists to
/// prevent, and the wait is the only place it can be stopped.
///
/// `wait` bounds that, and the two callers want different bounds: a merge runs
/// on the merger thread and can afford [`UNLINK_WAIT`], while retention runs on
/// the save path, where a long block is a stalled checkpoint. Timing out leaves
/// the files alone rather than pulling them out from under the reader. They are
/// already out of the manifest, so what they cost is disk until the next
/// startup, where `recover_or_reclaim_orphans` either folds them back in — they
/// are, after all, intact — or discards them. Both are better than a load that
/// fails halfway through.
///
/// The caller must not hold the manifest lock — see `crate::inflight`.
pub(crate) fn unlink_snapshots(
    storage: &Arc<dyn StorageBackend>,
    pending_deletes: &Arc<PendingDeletes>,
    in_flight: &Arc<InFlight>,
    doomed: &[Snapshot],
    wait: std::time::Duration,
) {
    if doomed.is_empty() {
        return;
    }
    let ids: Vec<Uuid> = doomed.iter().map(|s| s.id).collect();
    if !in_flight.wait_until_unpinned(&ids, wait) {
        eprintln!(
            "[Moonclip] {} dropped snapshot(s) are still being read after {}s; \
             leaving their files rather than unlinking under the reader",
            ids.len(),
            wait.as_secs()
        );
        return;
    }

    for snap in doomed {
        for rank_entry in snap.ranks.values() {
            if let Some(ref pack_file) = rank_entry.pack_file {
                let _ = storage.delete(pack_file);
                // The remote holds a copy of everything that was ever synced,
                // and a merge makes most of it unreachable at once.
                pending_deletes.push(pack_file.clone());
            }
            for tensor in &rank_entry.tensors {
                if let Some(ref filename) = tensor.filename {
                    let _ = storage.delete(filename);
                    pending_deletes.push(filename.clone());
                }
            }
        }
        let _ = storage.remove_dir(&format!("snapshots/{}", snap.id));
    }
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
///
/// Returns [`MergeOutcome::Busy`] rather than standing down silently when a
/// reader is inside the inputs. An opportunistic caller ignores that — it will
/// be notified again anyway — while a forced one retries; see
/// [`DeltaMerger::force_full_merge`].
pub(crate) fn do_full_merge(
    storage: &Arc<dyn StorageBackend>,
    manifest_lock: &Arc<Mutex<Manifest>>,
    compression: &CompressionAlgo,
    in_flight: &Arc<InFlight>,
    pending_deletes: &Arc<PendingDeletes>,
) -> Result<MergeOutcome> {
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
        return Ok(MergeOutcome::NothingToMerge);
    }

    // The base and every delta folded into it are about to stop existing, so
    // claim them before the manifest lock goes: a save that picks this base up
    // afterwards writes a full snapshot instead of a delta against something
    // that will be gone, and a save already inside one of them makes the claim
    // fail. The claim does not block — it runs under the manifest lock, and a
    // reader reacquires that lock after pinning, so blocking here would
    // deadlock against the reader being waited for. Retrying is the caller's
    // job, from outside the lock.
    let claimed: Vec<Uuid> = std::iter::once(base_id).chain(deltas.iter().map(|s| s.id)).collect();
    let Some(_claim) = in_flight.claim(&claimed) else {
        return Ok(MergeOutcome::Busy);
    };

    drop(manifest);

    let merged_id = Uuid::new_v4();
    let merged_dir = format!("snapshots/{}", merged_id);
    let mut merged_ranks: HashMap<u32, RankEntry> = HashMap::new();

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

    let mut all_ranks: Vec<u32> = base_snap
        .ranks
        .keys()
        .chain(last_delta.ranks.keys())
        .cloned()
        .collect();
    all_ranks.sort_unstable();
    all_ranks.dedup();
    let world_size = all_ranks.len() as u32;

    for &rank in &all_ranks {
        // **The newest delta decides which tensors exist, not the base.**
        //
        // Every delta is written against the last full snapshot and carries an
        // entry for each tensor in the state dict — `Skipped` for the ones that
        // match the base. So a delta is a complete description of the state at
        // its step, and it is exactly what `Core::load` walks when that
        // snapshot is loaded. A merge stands in for the newest delta, so it has
        // to walk the same list.
        //
        // Walking the *base's* list instead, which is what this did until
        // 0.0.6, gets both edges wrong and neither one loudly: a tensor the
        // model gained after the base is absent from that list and vanishes
        // from the merged snapshot, and a tensor the model dropped is still in
        // it and comes back from the dead. Adapters, growing heads and pruning
        // all produce one or the other, and the merged checkpoint loads
        // perfectly well afterwards — with the wrong set of parameters.
        //
        // The intermediate deltas are not replayed either. They cannot add
        // information: each describes the same base, so the newest supersedes
        // all of them. Replaying them in order also actively corrupted the
        // "changed and changed back" case, where the newest delta says
        // `Skipped` (identical to the base) but the sequence left the earlier
        // value in place.
        let source = match last_delta
            .ranks
            .get(&rank)
            .or_else(|| base_snap.ranks.get(&rank))
        {
            Some(r) => r,
            None => continue,
        };
        let source_snap = if last_delta.ranks.contains_key(&rank) {
            last_delta
        } else {
            &base_snap
        };

        let mut merged_tensors = Vec::new();
        // The compressed blobs, held until the whole rank is known: the pack's
        // descriptor records every offset, so it cannot be written before the
        // last tensor has one.
        let mut blobs: Vec<Vec<u8>> = Vec::new();
        let mut offset = pack::HEADER_LEN;
        let mut total_compressed = 0u64;
        let mut total_raw = 0u64;

        for entry in &source.tensors {
            // An alias holds no bytes of its own: it points at another tensor
            // in this same rank entry, which is materialized below under its
            // own name. Carry the reference through untouched.
            if entry.storage == TensorStorage::Alias {
                total_raw += entry.raw_size;
                merged_tensors.push(entry.clone());
                continue;
            }

            let current_data = match entry.storage {
                TensorStorage::Full => {
                    load_tensor_data(entry, source_snap, rank, storage, compression)?
                }
                TensorStorage::Skipped => {
                    // Unchanged since the base, so the bytes are the base's.
                    base_tensor_data(&base_snap, rank, &entry.name, storage, compression)?
                }
                TensorStorage::DeltaXor => {
                    let base_raw =
                        base_tensor_data(&base_snap, rank, &entry.name, storage, compression)?;
                    let compressed = load_compressed_data(entry, source_snap, rank, storage)?;
                    let mut delta_bytes = compression::decompress(&compressed, compression)?;
                    if entry.shuffled {
                        delta_bytes = crate::shuffle::unshuffle(
                            &delta_bytes,
                            crate::shuffle::element_size(&entry.dtype),
                        );
                    }
                    delta::apply_delta(&base_raw, &delta_bytes)?
                }
                TensorStorage::Alias => unreachable!("handled above"),
            };

            // Store as full, at its place in the pack.
            let raw_hash = hash_hex(&current_data);
            let compressed = compression::compress(&current_data, compression)?;
            let compressed_hash = hash_hex(&compressed);
            total_compressed += compressed.len() as u64;
            total_raw += current_data.len() as u64;

            merged_tensors.push(TensorEntry {
                name: entry.name.clone(),
                shape: entry.shape.clone(),
                dtype: entry.dtype.clone(),
                storage: TensorStorage::Full,
                alias_of: None,
                filename: None,
                offset,
                compressed_size: compressed.len() as u64,
                raw_size: current_data.len() as u64,
                hash_raw: raw_hash,
                hash_compressed: Some(compressed_hash),
                shuffled: false,
                original_dtype: entry.original_dtype.clone(),
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

        // A rank of nothing but aliases holds no bytes, and still gets a pack:
        // the descriptor in it is what startup recovery rebuilds a snapshot
        // from, and a merge is the worst moment to produce something recovery
        // cannot read — the run's whole history has just collapsed into this
        // one snapshot. Same rule as the save path.
        let pack_file = format!("{}/rank_{}.pack", merged_dir, rank);

        let rank_entry = RankEntry {
            rank,
            tensors: merged_tensors,
            pack_file: Some(pack_file),
            total_compressed,
            total_raw,
            skipped_count: alias_count,
            delta_count: 0,
            full_count,
        };

        let pack_filename = rank_entry.pack_file.as_ref().expect("set just above");
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

    // Durable and unreferenced: nothing new can reach these. What still can is
    // a load that pinned them before they were dropped, or after the claim —
    // so the unlink waits, outside the manifest lock. See `unlink_snapshots`.
    let spent: Vec<Snapshot> = deltas
        .into_iter()
        .chain(std::iter::once(base_snap))
        .collect();
    unlink_snapshots(storage, pending_deletes, in_flight, &spent, UNLINK_WAIT);

    Ok(MergeOutcome::Merged)
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

/// The base snapshot's bytes for `name`, following an alias if that is what
/// the base stored.
///
/// A tensor can be `Skipped` in the delta — unchanged — while the base holds
/// it as an `Alias` of a tied weight: deduplication keeps the first occurrence
/// and points the second at it, and which of the two comes first depends only
/// on the order the state dict was iterated in. So "look the name up in the
/// base" is not enough; the lookup has to resolve the reference the same way a
/// load does.
fn base_tensor_data(
    base_snap: &Snapshot,
    rank: u32,
    name: &str,
    storage: &Arc<dyn StorageBackend>,
    compression: &CompressionAlgo,
) -> Result<Vec<u8>> {
    let base_rank = base_snap.ranks.get(&rank).ok_or_else(|| {
        MoonclipError::NotFound(format!("Rank {rank} is not in base snapshot {}", base_snap.id))
    })?;

    let mut current = name;
    // Aliases point at a tensor that carries bytes, so one hop is the rule.
    // The bound is for a base written by some future version, or corrupted:
    // an alias cycle here would otherwise hang the merger thread.
    for _ in 0..8 {
        let entry = base_rank
            .tensors
            .iter()
            .find(|t| t.name == current)
            .ok_or_else(|| {
                MoonclipError::NotFound(format!(
                    "Tensor '{current}' is not in base snapshot {}",
                    base_snap.id
                ))
            })?;

        match entry.storage {
            TensorStorage::Alias => {
                current = entry.alias_of.as_deref().ok_or_else(|| {
                    MoonclipError::NotFound(format!("Alias '{current}' has no target"))
                })?;
            }
            _ => return load_tensor_data(entry, base_snap, rank, storage, compression),
        }
    }

    Err(MoonclipError::Delta(format!(
        "Alias chain starting at '{name}' in base snapshot {} does not end",
        base_snap.id
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

    /// No reader is inside anything: the merges below are the only thing
    /// touching these snapshots. See `crate::inflight`.
    fn unread() -> Arc<InFlight> {
        Arc::new(InFlight::default())
    }

    /// No remote configured, so the queue of keys to remove from it goes
    /// nowhere. See `crate::remote_sync::PendingDeletes`.
    fn no_remote() -> Arc<PendingDeletes> {
        Arc::new(PendingDeletes::default())
    }

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

        do_full_merge(&storage, &manifest, &ZSTD3, &unread(), &no_remote()).unwrap();

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

        do_full_merge(&storage, &manifest, &ZSTD3, &unread(), &no_remote()).unwrap();

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
        do_full_merge(&storage, &manifest, &ZSTD3, &unread(), &no_remote()).unwrap();

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
        do_stride_merge(&stride_config(3, 10), &storage, &manifest, &ZSTD3, &unread(), &no_remote()).unwrap();

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
        do_full_merge(&storage, &manifest, &ZSTD3, &unread(), &no_remote()).unwrap();
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
        do_stride_merge(&stride_config(3, 10), &storage, &manifest, &ZSTD3, &unread(), &no_remote()).unwrap();

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
        do_stride_merge(&stride_config(20, 3), &storage, &manifest, &ZSTD3, &unread(), &no_remote()).unwrap();

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
        do_stride_merge(&stride_config(3, 10), &storage, &manifest, &ZSTD3, &unread(), &no_remote()).unwrap();

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
        let result = do_stride_merge(&stride_config(3, 10), &storage, &manifest, &ZSTD3, &unread(), &no_remote());
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
        let result = do_full_merge(&storage, &manifest, &ZSTD3, &unread(), &no_remote());
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

    // ── The tensor set a merge is supposed to produce ───────────────
    //
    // A delta is complete with respect to its base: `save_sync` writes an
    // entry for *every* tensor in the state dict, `Skipped` for the ones that
    // match the base. So a delta's tensor list is the state dict as it stood
    // at that step — which is exactly what `Core::load` iterates, and
    // therefore what a merge standing in for that delta has to reproduce.
    //
    // Both directions of getting that wrong lose data silently: a tensor the
    // model gained after the base disappears, and one it dropped comes back.

    /// The state dict grew after the base: a parameter that only ever appears
    /// in the delta must survive the fold.
    #[test]
    fn a_merge_keeps_tensors_added_after_the_base() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());

        let w0 = noise(20_000, 1);
        let adapter = noise(20_000, 99);

        let base = packed_snapshot(
            storage.as_ref(),
            &[("w0".into(), w0.clone(), TensorStorage::Full)],
            None,
            0,
        );
        let base_id = base.id;

        // The delta carries w0 unchanged and the tensor the model just gained.
        let delta = packed_snapshot(
            storage.as_ref(),
            &[
                ("w0".into(), Vec::new(), TensorStorage::Skipped),
                ("adapter".into(), adapter.clone(), TensorStorage::Full),
            ],
            Some(base_id),
            1,
        );

        let manifest = Arc::new(Mutex::new(Manifest {
            snapshots: vec![base, delta],
            ..Default::default()
        }));

        do_full_merge(&storage, &manifest, &ZSTD3, &unread(), &no_remote()).unwrap();

        let m = manifest.lock().unwrap();
        let merged = m.snapshots.iter().find(|s| s.base_snapshot_id.is_none()).unwrap();
        let rank = merged.ranks.get(&0).unwrap();
        let names: Vec<&str> = rank.tensors.iter().map(|t| t.name.as_str()).collect();

        let entry = rank
            .tensors
            .iter()
            .find(|t| t.name == "adapter")
            .unwrap_or_else(|| panic!("'adapter' did not survive the merge; kept: {names:?}"));
        let got = load_tensor_data(entry, merged, 0, &storage, &ZSTD3).unwrap();
        assert_eq!(got, adapter, "'adapter' survived the merge with the wrong bytes");
    }

    /// The state dict shrank after the base: a parameter the model dropped
    /// must not be resurrected by the fold.
    #[test]
    fn a_merge_does_not_resurrect_tensors_removed_after_the_base() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());

        let base = packed_snapshot(
            storage.as_ref(),
            &[
                ("kept".into(), noise(20_000, 1), TensorStorage::Full),
                ("dropped".into(), noise(20_000, 2), TensorStorage::Full),
            ],
            None,
            0,
        );
        let base_id = base.id;

        // The step after: the model no longer has `dropped`, so the save path
        // writes no entry for it at all.
        let delta = packed_snapshot(
            storage.as_ref(),
            &[("kept".into(), Vec::new(), TensorStorage::Skipped)],
            Some(base_id),
            1,
        );

        let manifest = Arc::new(Mutex::new(Manifest {
            snapshots: vec![base, delta],
            ..Default::default()
        }));

        do_full_merge(&storage, &manifest, &ZSTD3, &unread(), &no_remote()).unwrap();

        let m = manifest.lock().unwrap();
        let merged = m.snapshots.iter().find(|s| s.base_snapshot_id.is_none()).unwrap();
        let names: Vec<&str> = merged
            .ranks
            .get(&0)
            .unwrap()
            .tensors
            .iter()
            .map(|t| t.name.as_str())
            .collect();

        assert!(
            !names.contains(&"dropped"),
            "a tensor removed from the model came back through the merge: {names:?}"
        );
        assert!(names.contains(&"kept"));
    }

    /// Changed, then changed back. The newest delta says `Skipped`, meaning
    /// "identical to the base" — so the merged snapshot has to hold the base's
    /// bytes, not the intermediate delta's.
    #[test]
    fn a_merge_takes_the_newest_delta_as_the_answer() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());

        let original = noise(20_000, 5);
        let mut touched = original.clone();
        touched[7] ^= 0xff;
        let xor = delta::compute_delta(&original, &touched).unwrap();

        let base = packed_snapshot(
            storage.as_ref(),
            &[("w".into(), original.clone(), TensorStorage::Full)],
            None,
            0,
        );
        let base_id = base.id;
        let d1 = packed_snapshot(
            storage.as_ref(),
            &[("w".into(), xor, TensorStorage::DeltaXor)],
            Some(base_id),
            1,
        );
        let d2 = packed_snapshot(
            storage.as_ref(),
            &[("w".into(), Vec::new(), TensorStorage::Skipped)],
            Some(base_id),
            2,
        );

        let manifest = Arc::new(Mutex::new(Manifest {
            snapshots: vec![base, d1, d2],
            ..Default::default()
        }));

        do_full_merge(&storage, &manifest, &ZSTD3, &unread(), &no_remote()).unwrap();

        let m = manifest.lock().unwrap();
        let merged = m.snapshots.iter().find(|s| s.base_snapshot_id.is_none()).unwrap();
        let entry = merged
            .ranks
            .get(&0)
            .unwrap()
            .tensors
            .iter()
            .find(|t| t.name == "w")
            .unwrap();
        let got = load_tensor_data(entry, merged, 0, &storage, &ZSTD3).unwrap();
        assert_eq!(
            got, original,
            "the merge replayed an intermediate delta the newest one had undone"
        );
    }

    /// Standing down has to be distinguishable from having merged.
    ///
    /// `do_full_merge` returned `Ok(())` for both, so `save_final` — which
    /// merges and then uploads — reported a fold that never happened and sent
    /// the unmerged chain to the bucket. A reader inside the inputs is the
    /// case that produces it.
    #[test]
    fn a_merge_refused_by_a_reader_says_so_instead_of_reporting_success() {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let (manifest, _) = scenario(storage.as_ref(), 2, 20_000);

        let base_id = manifest.snapshots[0].id;
        let manifest = Arc::new(Mutex::new(manifest));
        let in_flight = unread();

        // A load is inside the base, which is one of the snapshots the fold
        // would consume.
        let reader = in_flight.pin(&[base_id]);
        assert_eq!(
            do_full_merge(&storage, &manifest, &ZSTD3, &in_flight, &no_remote()).unwrap(),
            MergeOutcome::Busy,
            "a merge that stood down reported the same thing as one that folded"
        );
        assert!(
            manifest.lock().unwrap().snapshots.len() > 1,
            "the merge claimed to stand down but folded anyway"
        );

        drop(reader);
        assert_eq!(
            do_full_merge(&storage, &manifest, &ZSTD3, &in_flight, &no_remote()).unwrap(),
            MergeOutcome::Merged,
            "the merge stayed refused after the reader left"
        );
    }

    /// The unlink is the step a reader cannot survive, and the claim cannot
    /// cover it: a load that pins *after* the merge has claimed meets no
    /// refusal at all. Proven at the registry level in `crate::inflight`; here
    /// it is the merger actually parking on it rather than deleting.
    #[test]
    fn a_merge_does_not_unlink_under_a_reader_that_arrived_after_the_claim() {
        /// Pins the base on the merge's first pack read, which is the one
        /// moment that is reliably *after* the claim and before the unlink.
        ///
        /// Pinning from the test thread instead would race the claim and
        /// usually lose: the merge would be refused outright, which is the
        /// other half of the mechanism and already covered above.
        struct PinOnRead {
            inner: Arc<LocalStorage>,
            in_flight: Arc<InFlight>,
            base_id: Uuid,
            held: Mutex<Option<crate::inflight::PinGuard>>,
            armed: std::sync::atomic::AtomicBool,
        }

        impl PinOnRead {
            fn arrive(&self) {
                use std::sync::atomic::Ordering;
                if self.armed.swap(false, Ordering::SeqCst) {
                    *self.held.lock().unwrap() = Some(self.in_flight.pin(&[self.base_id]));
                }
            }
        }

        impl StorageBackend for PinOnRead {
            fn put(&self, p: &str, data: &[u8]) -> Result<()> {
                self.inner.put(p, data)
            }
            fn put_parts(&self, p: &str, parts: &[&[u8]]) -> Result<()> {
                self.inner.put_parts(p, parts)
            }
            fn get(&self, p: &str) -> Result<Vec<u8>> {
                if p.ends_with(".pack") {
                    self.arrive();
                }
                self.inner.get(p)
            }
            fn get_range(&self, p: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
                if p.ends_with(".pack") {
                    self.arrive();
                }
                self.inner.get_range(p, offset, len)
            }
            fn exists(&self, p: &str) -> Result<bool> {
                self.inner.exists(p)
            }
            fn delete(&self, p: &str) -> Result<()> {
                self.inner.delete(p)
            }
            fn list(&self, prefix: &str) -> Result<Vec<String>> {
                self.inner.list(prefix)
            }
            fn remove_dir(&self, p: &str) -> Result<()> {
                self.inner.remove_dir(p)
            }
        }

        let dir = tempfile::tempdir().unwrap();
        let plain = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let (manifest, _) = scenario(plain.as_ref(), 2, 20_000);

        let base_id = manifest.snapshots[0].id;
        let base_packs: Vec<String> = manifest.snapshots[0]
            .ranks
            .values()
            .filter_map(|r| r.pack_file.clone())
            .collect();
        assert!(!base_packs.is_empty(), "the base wrote no pack to protect");

        let manifest = Arc::new(Mutex::new(manifest));
        let in_flight = unread();
        let hooked = Arc::new(PinOnRead {
            inner: Arc::clone(&plain),
            in_flight: Arc::clone(&in_flight),
            base_id,
            held: Mutex::new(None),
            armed: std::sync::atomic::AtomicBool::new(true),
        });

        // Reported over a channel rather than a join handle: the assertion
        // below is that the merge has *not* finished, and `join` can only wait
        // for one that has.
        let (finished, merged) = mpsc::channel();
        let merging = {
            let storage: Arc<dyn StorageBackend> = Arc::clone(&hooked) as Arc<dyn StorageBackend>;
            let manifest = Arc::clone(&manifest);
            let in_flight = Arc::clone(&in_flight);
            std::thread::spawn(move || {
                let outcome = do_full_merge(&storage, &manifest, &ZSTD3, &in_flight, &no_remote());
                let _ = finished.send(outcome);
            })
        };

        // Publish happens before the unlink, so a manifest of one snapshot
        // means the merge has reached the step this is about.
        let started = std::time::Instant::now();
        while manifest.lock().unwrap().snapshots.len() > 1 {
            assert!(
                started.elapsed() < std::time::Duration::from_secs(30),
                "the merge never published"
            );
            std::thread::yield_now();
        }
        assert!(
            hooked.held.lock().unwrap().is_some(),
            "the reader never arrived: this probe proved nothing"
        );

        // The load-bearing assertion, and the reason it is phrased as a
        // timeout: having published, the merge is one step from unlinking, and
        // the only thing between it and those files is the pin. Checking that
        // the packs still exist would race the unlink and pass either way —
        // this cannot, because a merge that is not blocked finishes these 20 KB
        // in microseconds.
        assert!(
            merged
                .recv_timeout(std::time::Duration::from_secs(2))
                .is_err(),
            "the merge ran to completion with a reader still inside its inputs"
        );
        for pack in &base_packs {
            assert!(
                plain.exists(pack).unwrap(),
                "{pack} was unlinked while a reader was inside it"
            );
        }

        hooked.held.lock().unwrap().take();
        let outcome = merged
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("the merge never woke after the reader left");
        assert_eq!(outcome.unwrap(), MergeOutcome::Merged);
        merging.join().unwrap();
        for pack in &base_packs {
            assert!(
                !plain.exists(pack).unwrap(),
                "{pack} outlived the reader that was holding it back"
            );
        }
    }
}
