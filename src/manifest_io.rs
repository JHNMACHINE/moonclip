//! How a [`Manifest`] lies in storage: a small index and a file per snapshot.
//!
//! **Why not one file.** The manifest used to be written whole, pretty-printed,
//! at every save. Each snapshot in it carries an entry for every tensor -
//! name, shape, two hashes - so the file grew by a model's worth of entries
//! per save, and with a delta at every step it was rewritten once a step at a
//! size proportional to the window. Measured on 2026-10-22 (GPU-261): a model
//! of four parameters, no window to speak of, and 250 steps went from 0.8 s to
//! 20 s as the manifest reached 13 MB; a training loop never waits on the
//! writer until the writer is this far behind, and then it waits every step.
//!
//! **What goes where.** A snapshot's `ranks` - the tensor entries, almost all
//! of its bytes - never change once it is finalised. They go to
//! `snapshots/<id>/ranks.json`, written once, beside the packs they describe.
//! What does change after that - `pinned`, and `finalized` for a multi-rank
//! snapshot still collecting its shards - stays in `manifest.json`, which is
//! now an index: one short entry per snapshot, naming the file that holds the
//! rest. A save writes its own file and an index of the window's heads, and
//! the index of a 3000-snapshot window is smaller than one snapshot of a
//! large model used to be.
//!
//! **The order is the guarantee.** The file goes down before the index that
//! names it, as packs do before the manifest. A crash between the two leaves
//! a snapshot directory the index does not name, which startup recovery
//! already knows how to judge (`recover_or_reclaim_orphans`), and a remote
//! sync sends the index last for the same reason.
//!
//! **Older stores still open.** An entry with `ranks` inline is read as it
//! always was. A snapshot keeps its `ranks` inline until it is finalised, so
//! the pre-finalisation state a multi-rank save goes through is unchanged.
//! The format version is 3; a Moonclip that predates it cannot read a store
//! written by this one.

use crate::error::{MoonclipError, Result};
use crate::manifest::{
    CompressionAlgo, LineageConfig, Manifest, RankEntry, RetentionPolicy, Snapshot,
};
use crate::storage::StorageBackend;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// The index, at the root of the store. The name every reader looks for.
pub const MANIFEST: &str = "manifest.json";

/// Each snapshot's own file, inside its directory.
pub const RANKS_FILE: &str = "ranks.json";

/// The first version with snapshots out of line.
pub const FORMAT_VERSION: u32 = 3;

/// Where `id`'s ranks are kept once it is finalised.
pub fn ranks_path(id: Uuid) -> String {
    format!("snapshots/{id}/{RANKS_FILE}")
}

#[derive(Serialize)]
struct IndexOut<'a> {
    format_version: u32,
    world_size: u32,
    snapshots: Vec<HeadOut<'a>>,
    retention: &'a RetentionPolicy,
    lineage: &'a LineageConfig,
}

/// A snapshot as the index holds it. The fields are `Snapshot`'s, in its
/// order, so an index entry with `ranks` inline is exactly an old one.
#[derive(Serialize)]
struct HeadOut<'a> {
    id: &'a Uuid,
    step: u64,
    created_at: &'a DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ranks: Option<&'a HashMap<u32, RankEntry>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ranks_in: Option<String>,
    base_snapshot_id: &'a Option<Uuid>,
    metadata: &'a HashMap<String, String>,
    compression: &'a CompressionAlgo,
    finalized: bool,
    pinned: bool,
}

#[derive(Deserialize)]
struct IndexIn {
    #[serde(default)]
    format_version: u32,
    world_size: u32,
    snapshots: Vec<HeadIn>,
    retention: RetentionPolicy,
    lineage: LineageConfig,
}

#[derive(Deserialize)]
struct HeadIn {
    id: Uuid,
    step: u64,
    created_at: DateTime<Utc>,
    #[serde(default)]
    ranks: Option<HashMap<u32, RankEntry>>,
    #[serde(default)]
    ranks_in: Option<String>,
    base_snapshot_id: Option<Uuid>,
    metadata: HashMap<String, String>,
    compression: CompressionAlgo,
    finalized: bool,
    #[serde(default)]
    pinned: bool,
}

/// A snapshot's own file. Names the snapshot it belongs to, so one put in
/// the wrong directory is refused rather than read as another's tensors.
#[derive(Serialize)]
struct RanksOut<'a> {
    snapshot_id: &'a Uuid,
    ranks: &'a HashMap<u32, RankEntry>,
}

#[derive(Deserialize)]
struct RanksIn {
    snapshot_id: Uuid,
    ranks: HashMap<u32, RankEntry>,
}

fn serialization(e: serde_json::Error) -> MoonclipError {
    MoonclipError::Serialization(e.to_string())
}

/// Whether `snapshot` goes out of line. A finalised one with something to
/// say: an unfinalised multi-rank snapshot has no ranks yet, and its file
/// would be written before the only thing it exists to hold.
fn out_of_line(snapshot: &Snapshot) -> bool {
    snapshot.finalized && !snapshot.ranks.is_empty()
}

/// Write `manifest`: the file of every finalised snapshot that has none yet,
/// then the index.
///
/// A snapshot's file is written once in the life of the store - its ranks do
/// not change after finalisation - and which ones exist is remembered in
/// `manifest.ranks_written`, so a save pays for its own snapshot and the
/// index, and for nothing else in the window.
pub fn persist(storage: &dyn StorageBackend, manifest: &mut Manifest) -> Result<()> {
    let unwritten: Vec<Uuid> = manifest
        .snapshots
        .iter()
        .filter(|s| out_of_line(s) && !manifest.ranks_written.contains(&s.id))
        .map(|s| s.id)
        .collect();
    for snapshot in manifest
        .snapshots
        .iter()
        .filter(|s| unwritten.contains(&s.id))
    {
        let body = serde_json::to_vec(&RanksOut {
            snapshot_id: &snapshot.id,
            ranks: &snapshot.ranks,
        })
        .map_err(serialization)?;
        storage.put(&ranks_path(snapshot.id), &body)?;
    }
    manifest.ranks_written.extend(unwritten);
    // What retention and the merger removed is not on disk anymore either, and
    // a set that only grew would be a leak the length of the run.
    let present: HashSet<Uuid> = manifest.snapshots.iter().map(|s| s.id).collect();
    manifest.ranks_written.retain(|id| present.contains(id));

    let index = IndexOut {
        format_version: FORMAT_VERSION,
        world_size: manifest.world_size,
        snapshots: manifest
            .snapshots
            .iter()
            .map(|s| {
                let written = manifest.ranks_written.contains(&s.id);
                HeadOut {
                    id: &s.id,
                    step: s.step,
                    created_at: &s.created_at,
                    ranks: (!written).then_some(&s.ranks),
                    ranks_in: written.then(|| ranks_path(s.id)),
                    base_snapshot_id: &s.base_snapshot_id,
                    metadata: &s.metadata,
                    compression: &s.compression,
                    finalized: s.finalized,
                    pinned: s.pinned,
                }
            })
            .collect(),
        retention: &manifest.retention,
        lineage: &manifest.lineage,
    };
    let json = serde_json::to_vec(&index).map_err(serialization)?;
    storage.put(MANIFEST, &json)
}

/// Read the manifest in storage, or `None` when there is none.
///
/// `previous` is the manifest this process already holds, if any. A snapshot
/// in both is taken from it instead of from its file: its ranks cannot have
/// changed, and a multi-rank save re-reads the manifest at every step, which
/// would otherwise read every file in the window each time.
///
/// A snapshot whose file is missing or unreadable is left out, with a line
/// saying so, and so is any delta taken against it: an index naming data
/// nobody can read is the restore-from-a-partial-sync case, and resuming from
/// what is whole is the difference between losing some progress and losing
/// the run. An index that does not parse is an error, as it always was.
pub fn load(
    storage: &dyn StorageBackend,
    previous: Option<&mut Manifest>,
) -> Result<Option<Manifest>> {
    let data = match storage.get(MANIFEST) {
        Ok(data) => data,
        Err(MoonclipError::NotFound(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    // Trailing zeros: a manifest rewritten in place can leave the tail of a
    // longer one behind it.
    let end = data.iter().rposition(|&b| b != 0).map(|i| i + 1).unwrap_or(0);
    let index: IndexIn = serde_json::from_slice(&data[..end]).map_err(serialization)?;

    let mut held: HashMap<Uuid, HashMap<u32, RankEntry>> = HashMap::new();
    if let Some(previous) = previous {
        for snapshot in previous.snapshots.iter_mut() {
            if previous.ranks_written.contains(&snapshot.id) {
                held.insert(snapshot.id, std::mem::take(&mut snapshot.ranks));
            }
        }
    }

    let mut ranks_written = HashSet::new();
    let mut snapshots = Vec::with_capacity(index.snapshots.len());
    let mut unreadable: HashSet<Uuid> = HashSet::new();
    for head in index.snapshots {
        let ranks = match (head.ranks, head.ranks_in) {
            (Some(inline), _) => inline,
            (None, Some(path)) => {
                let ranks = match held.remove(&head.id) {
                    Some(ranks) => Some(ranks),
                    None => read_ranks(storage, &path, head.id),
                };
                match ranks {
                    Some(ranks) => {
                        ranks_written.insert(head.id);
                        ranks
                    }
                    None => {
                        unreadable.insert(head.id);
                        continue;
                    }
                }
            }
            (None, None) => HashMap::new(),
        };
        snapshots.push(Snapshot {
            id: head.id,
            step: head.step,
            created_at: head.created_at,
            ranks,
            base_snapshot_id: head.base_snapshot_id,
            metadata: head.metadata,
            compression: head.compression,
            finalized: head.finalized,
            pinned: head.pinned,
        });
    }
    if !unreadable.is_empty() {
        // Bases are fulls, never deltas, so one pass finds every delta whose
        // base went.
        let before = snapshots.len();
        snapshots.retain(|s| s.base_snapshot_id.is_none_or(|base| !unreadable.contains(&base)));
        eprintln!(
            "[Moonclip] {} snapshot(s) in the manifest have no readable {RANKS_FILE}, \
             and {} delta(s) were taken against them; left out, resuming from what is whole.",
            unreadable.len(),
            before - snapshots.len()
        );
    }

    Ok(Some(Manifest {
        format_version: index.format_version.max(FORMAT_VERSION),
        world_size: index.world_size,
        snapshots,
        retention: index.retention,
        lineage: index.lineage,
        ranks_written,
    }))
}

fn read_ranks(
    storage: &dyn StorageBackend,
    path: &str,
    id: Uuid,
) -> Option<HashMap<u32, RankEntry>> {
    let data = storage.get(path).ok()?;
    let end = data.iter().rposition(|&b| b != 0).map(|i| i + 1).unwrap_or(0);
    let read: RanksIn = serde_json::from_slice(&data[..end]).ok()?;
    (read.snapshot_id == id).then_some(read.ranks)
}
