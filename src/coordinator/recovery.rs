//! Putting a store back together at open time.
//!
//! Everything here runs once, in `Core::new`, before anything has trained, and
//! every function takes the storage and the manifest rather than a `Core`:
//! recovery decides what the coordinator's starting state *is*, so it cannot
//! be a method on the thing it is deciding.
//!
//! Two failures are covered, and they are not the same one. A checkpoint whose
//! packs are on disk but whose manifest entry never landed is **complete and
//! unreferenced** — a process killed between the two writes — and is recovered
//! by reading the packs' own descriptors. A checkpoint whose data is missing or
//! corrupt is neither, and is reclaimed instead of resurrected.

use super::*;

fn rebuild_from_packs(
    storage: &dyn StorageBackend,
    id: &str,
    files: &[String],
) -> Option<Snapshot> {
    let mut descriptors: Vec<pack::PackDescriptor> = Vec::new();

    for file in files.iter().filter(|f| f.ends_with(".pack")) {
        let header = storage.get_range(file, 0, pack::HEADER_LEN as usize).ok()?;
        // No header means a pack written before this format. It still loads
        // through the manifest; it just cannot be recovered without one.
        let (offset, length) = pack::decode_header(&header)?;
        let encoded = storage.get_range(file, offset, length as usize).ok()?;
        if encoded.len() != length as usize {
            return None; // truncated write
        }
        let descriptor: pack::PackDescriptor = serde_json::from_slice(&encoded).ok()?;
        if descriptor.snapshot_id.to_string() != id {
            return None; // a pack claiming to belong somewhere else
        }
        descriptors.push(descriptor);
    }

    let first = descriptors.first()?.clone();

    // Every rank has to be here. A snapshot missing one rank's shard is not a
    // smaller checkpoint, it is an unusable one.
    let ranks: HashMap<u32, RankEntry> = descriptors
        .iter()
        .map(|d| (d.rank.rank, d.rank.clone()))
        .collect();
    if ranks.len() as u32 != first.world_size {
        return None;
    }

    Some(Snapshot {
        id: first.snapshot_id,
        step: first.step,
        created_at: first.created_at,
        ranks,
        base_snapshot_id: first.base_snapshot_id,
        metadata: first.metadata,
        compression: first.compression,
        finalized: true,
    })
}

/// Whether every byte the snapshot claims is present and intact.
///
/// Presence is not enough. A pack of the right length can still be a truncated
/// write padded by the filesystem, or a partial flush — and a snapshot readmitted
/// on those terms would fail much later, during a resume, which is the worst
/// moment to discover it. So each tensor's stored hash is checked against the
/// bytes actually on disk.
///
/// Reading the pack to do that is affordable precisely because this only runs
/// for orphans, which exist only after a crash.
fn snapshot_data_is_intact(storage: &dyn StorageBackend, snapshot: &Snapshot) -> bool {
    for rank_entry in snapshot.ranks.values() {
        for tensor in &rank_entry.tensors {
            // Skipped tensors and aliases store no bytes of their own; they
            // resolve through the base or through a sibling.
            let Some(ref expected) = tensor.hash_compressed else {
                continue;
            };

            let read = match (&rank_entry.pack_file, &tensor.filename) {
                (Some(pack), _) => {
                    storage.get_range(pack, tensor.offset, tensor.compressed_size as usize)
                }
                (None, Some(file)) => storage.get(file),
                (None, None) => return false,
            };

            let Ok(bytes) = read else { return false };
            if bytes.len() != tensor.compressed_size as usize {
                return false;
            }
            if &crate::hash::hash_hex(&bytes) != expected {
                return false;
            }
        }
    }
    true
}

/// Recover snapshot data that no manifest entry refers to, or delete it.
///
/// A checkpoint lands in two stages: the bytes go to storage, then the manifest
/// that names them. A process killed between the two leaves data with nothing
/// pointing at it. Resume is right to ignore it while it stays that way — a
/// snapshot the manifest does not list must never be trusted — but leaving it
/// there forever is not right either: `apply_retention` walks
/// `manifest.snapshots`, so it cannot see those files and they outlive the run.
///
/// Seen in the field, not hypothesised: an FSDP run over eight GPUs killed with
/// SIGKILL mid-checkpoint left three snapshot directories against one manifest
/// entry, and resumed four steps further back than it needed to.
///
/// So each orphan is judged rather than assumed:
///
/// * it describes itself (`snapshot.json`), its data passes its own hashes, it
///   is finalised, and any base it deltas against is present → **put it back in
///   the manifest**. It was a complete checkpoint; only the bookkeeping was lost.
/// * anything else — no description, failed hashes, a missing base → **delete
///   it**. That is a write that never finished, and there is nothing to save.
///
/// **Startup is the only safe moment.** Here this process has submitted no save
/// and the background saver does not exist yet, so nothing it might touch is in
/// flight. What remains is a *different* process writing to the same storage
/// root concurrently — but two coordinators sharing a root already overwrite
/// each other's manifest, so that arrangement is broken well before this is.
///
/// Only directories named by a parsable UUID are considered. Anything else
/// under `snapshots/` was put there by something other than this code, and
/// guessing about it is how a cleanup routine deletes data it did not own.
///
/// Failures are logged and swallowed: this is housekeeping and must never stop
/// a training run from starting. Returns true when the manifest changed.
pub(super) fn recover_or_reclaim_orphans(storage: &dyn StorageBackend, manifest: &mut Manifest) -> bool {
    let known: std::collections::HashSet<String> = manifest
        .snapshots
        .iter()
        .map(|s| s.id.to_string())
        .collect();

    let files = match storage.list("snapshots") {
        Ok(files) => files,
        Err(e) => {
            eprintln!("[Moonclip] Could not list snapshots: {e}");
            return false;
        }
    };

    let mut orphans: HashMap<String, Vec<String>> = HashMap::new();
    for file in files {
        let Some(id) = file
            .strip_prefix("snapshots/")
            .and_then(|rest| rest.split('/').next())
        else {
            continue;
        };
        if known.contains(id) || Uuid::parse_str(id).is_err() {
            continue;
        }
        orphans.entry(id.to_string()).or_default().push(file);
    }

    if orphans.is_empty() {
        return false;
    }

    // Read every candidate's description first, then admit them oldest-first:
    // a delta can only go back if its base is already there, and its base may
    // itself be one of these orphans.
    let mut candidates: Vec<Snapshot> = Vec::new();
    let mut rejects: Vec<String> = Vec::new();

    for (id, files) in &orphans {
        let described = rebuild_from_packs(storage, id, files);
        match described {
            Some(snapshot) if snapshot_data_is_intact(storage, &snapshot) => {
                candidates.push(snapshot)
            }
            _ => rejects.push(id.clone()),
        }
    }
    candidates.sort_by_key(|s| s.step);

    let mut recovered = 0usize;
    for snapshot in candidates {
        let base_present = match snapshot.base_snapshot_id {
            None => true,
            Some(base) => manifest.snapshots.iter().any(|s| s.id == base),
        };
        if !base_present {
            // A delta whose base is gone reconstructs nothing.
            rejects.push(snapshot.id.to_string());
            continue;
        }
        manifest.snapshots.push(snapshot);
        recovered += 1;
    }

    let mut reclaimed = 0usize;
    for id in &rejects {
        for file in orphans.get(id).into_iter().flatten() {
            if let Err(e) = storage.delete(file) {
                eprintln!("[Moonclip] Could not delete orphaned {file}: {e}");
            }
        }
        // Best effort: the bytes are already gone, and an empty directory left
        // behind costs an inode, not a checkpoint.
        let _ = storage.remove_dir(&format!("snapshots/{id}"));
        reclaimed += 1;
    }

    if recovered > 0 {
        manifest.snapshots.sort_by_key(|s| s.step);
        eprintln!(
            "[Moonclip] Recovered {recovered} complete checkpoint(s) a previous \
             run wrote but was killed before recording"
        );
    }
    if reclaimed > 0 {
        eprintln!(
            "[Moonclip] Discarded {reclaimed} incomplete snapshot(s) left by an \
             interrupted run"
        );
    }

    recovered > 0
}

pub(super) fn load_manifest(storage: &dyn StorageBackend, config: &CoordinatorConfig) -> Result<Manifest> {
    match storage.get("manifest.json") {
        Ok(data) => {
            // Trailing zeros: a manifest is rewritten in place, so a shorter
            // one can leave the tail of a longer one behind it.
            let end = data
                .iter()
                .rposition(|&b| b != 0)
                .map(|i| i + 1)
                .unwrap_or(0);
            serde_json::from_slice(&data[..end])
                .map_err(|e| MoonclipError::Serialization(e.to_string()))
        }
        Err(MoonclipError::NotFound(_)) => Ok(Manifest {
            world_size: config.world_size,
            retention: config.retention.clone(),
            lineage: config.lineage.clone(),
            ..Default::default()
        }),
        Err(e) => Err(e),
    }
}

/// Whether every file a snapshot names is actually here.
///
/// Used after a restore from remote, where the manifest and the data it points
/// at travel separately and can arrive out of step. A snapshot that is missing
/// a pack is not a degraded snapshot, it is an unloadable one, and leaving it
/// in the manifest turns a resume into a failure at the first read.
pub(super) fn snapshot_is_whole(storage: &dyn StorageBackend, snapshot: &Snapshot) -> bool {
    for rank in snapshot.ranks.values() {
        if let Some(ref pack) = rank.pack_file {
            if !storage.exists(pack).unwrap_or(false) {
                return false;
            }
        }
        for tensor in &rank.tensors {
            if let Some(ref file) = tensor.filename {
                if !storage.exists(file).unwrap_or(false) {
                    return false;
                }
            }
        }
    }
    true
}
