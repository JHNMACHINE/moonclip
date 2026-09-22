use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};
use uuid::Uuid;

// ─── Tensor-level metadata ──────────────────────────────────────────

/// How a tensor is stored in a snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TensorStorage {
    /// Full tensor data (compressed).
    Full,
    /// XOR delta against the same tensor in the base snapshot.
    DeltaXor,
    /// Tensor unchanged from base — no data stored, just a reference.
    Skipped,
    /// Byte-for-byte identical to another tensor in the *same* snapshot
    /// (see `alias_of`) — no data stored. Tied embeddings and repeated
    /// buffers would otherwise be written once per name.
    Alias,
}

/// Metadata for a single tensor within a rank's checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    /// Fully qualified name, e.g. "model.layers.0.self_attn.q_proj.weight"
    pub name: String,
    /// Shape as a list of dims.
    pub shape: Vec<usize>,
    /// Element dtype string as stored on disk, e.g. "bfloat16", "float32".
    pub dtype: String,
    /// Original dtype before casting (e.g. "float32" if saved as bf16).
    /// If None, dtype == original dtype (no cast was applied).
    #[serde(default)]
    pub original_dtype: Option<String>,
    /// Dequantization multiplier for a tensor stored in a float8 `dtype`:
    /// `original ≈ stored × quant_scale`, elementwise.
    ///
    /// `None` for every other dtype, which needs no side channel to be read
    /// back. Optional and defaulted so manifests written before float8
    /// existed still deserialize — they carry no float8 entries, so `None` is
    /// not a missing value there but the correct one.
    ///
    /// A tensor that arrives *already* float8 stores no scale either: it is
    /// copied byte for byte and `original_dtype` stays `None`, so nothing on
    /// the load path tries to undo a cast that never happened.
    #[serde(default)]
    pub quant_scale: Option<f32>,
    /// How this tensor is stored.
    pub storage: TensorStorage,
    /// For `Alias`: the name of the tensor in this same snapshot that
    /// holds the bytes. None for every other storage kind.
    #[serde(default)]
    pub alias_of: Option<String>,
    /// Path to the compressed data file (None if Skipped).
    /// With packed files, this is None — use RankEntry.pack_file instead.
    pub filename: Option<String>,
    /// Byte offset within the pack file (0 for legacy per-file storage).
    #[serde(default)]
    pub offset: u64,
    /// Byte size of compressed data on disk.
    pub compressed_size: u64,
    /// Byte size of raw (uncompressed) tensor data.
    pub raw_size: u64,
    /// Hash of the raw tensor bytes (used for skip detection).
    ///
    /// xxHash3-128, and has been since v2 — the old `sha256_` name outlived
    /// the algorithm by long enough to be quoted as fact in review. The alias
    /// keeps manifests written under the old name readable.
    #[serde(alias = "sha256_raw")]
    pub hash_raw: String,
    /// Hash of the compressed data on disk (None if Skipped).
    #[serde(alias = "sha256_compressed")]
    pub hash_compressed: Option<String>,
    /// Whether the stored bytes were byte-shuffled before compression
    /// (see `crate::shuffle`). Only XOR deltas are shuffled.
    ///
    /// Defaults to false so snapshots written before the filter existed still
    /// deserialize, and read back as what they are: unshuffled.
    #[serde(default)]
    pub shuffled: bool,
}

// ─── Rank-level metadata ────────────────────────────────────────────

/// All tensor entries for a single rank in a snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankEntry {
    pub rank: u32,
    pub tensors: Vec<TensorEntry>,
    /// Single pack file containing all tensor data for this rank.
    /// If Some, tensor offsets are relative to this file.
    /// If None, each tensor uses its own `filename` (legacy mode).
    #[serde(default)]
    pub pack_file: Option<String>,
    /// Total compressed bytes on disk for this rank.
    pub total_compressed: u64,
    /// Total raw bytes for this rank.
    pub total_raw: u64,
    /// Number of tensors skipped (unchanged from base).
    pub skipped_count: usize,
    /// Number of tensors stored as delta.
    pub delta_count: usize,
    /// Number of tensors stored as full.
    pub full_count: usize,
}

// ─── Snapshot ───────────────────────────────────────────────────────

/// Compression algorithm used.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgo {
    None,
    Zstd { level: i32 },
}

impl Default for CompressionAlgo {
    fn default() -> Self {
        CompressionAlgo::Zstd { level: 3 }
    }
}

/// A single save event across all ranks at a given step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub id: Uuid,
    pub step: u64,
    pub created_at: DateTime<Utc>,
    /// Per-rank tensor entries. Key = rank id.
    pub ranks: HashMap<u32, RankEntry>,
    /// If this is a delta snapshot, points to the base.
    pub base_snapshot_id: Option<Uuid>,
    /// User metadata (loss, lr, etc.).
    pub metadata: HashMap<String, String>,
    /// Compression algorithm used.
    pub compression: CompressionAlgo,
    /// Whether all ranks have finished saving (for multi-rank coordination).
    pub finalized: bool,
    /// Kept whatever retention says, until somebody unpins it (GPU-154).
    ///
    /// Rollback protection is periodic - every Nth step deserves it - and that
    /// is a different question from "this one". A fork starts from the step
    /// somebody chose, which almost never lands on the interval, and the base
    /// disappearing under it is a run with no way back.
    ///
    /// `serde(default)` because stores written before this field exist and
    /// must keep opening.
    #[serde(default)]
    pub pinned: bool,
}

// ─── Lineage ────────────────────────────────────────────────────────

/// Lineage configuration for rollback support (DECK Section 4.3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageConfig {
    /// Keep a rollback-safe full snapshot every N steps.
    /// These survive retention cleanup for data corruption recovery.
    pub rollback_interval_steps: u64,
    /// Max number of rollback snapshots to keep.
    pub max_rollback_snapshots: usize,
}

impl Default for LineageConfig {
    fn default() -> Self {
        LineageConfig {
            rollback_interval_steps: 10000,
            max_rollback_snapshots: 3,
        }
    }
}

// ─── Retention ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Keep at most N full snapshots in the active lineage.
    pub max_full_snapshots: usize,
    /// Keep at most N delta snapshots between each pair of full snapshots.
    pub max_deltas_per_full: usize,
    /// Force a full snapshot every N steps.
    pub full_snapshot_every_steps: u64,
    /// Hard cap on total accessible snapshots (full + delta).
    /// If None, defaults to max_full_snapshots * (1 + max_deltas_per_full).
    #[serde(default)]
    pub max_total_snapshots: Option<usize>,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        RetentionPolicy {
            max_full_snapshots: 5,
            max_deltas_per_full: 10,
            full_snapshot_every_steps: 5000,
            max_total_snapshots: None,
        }
    }
}

impl RetentionPolicy {
    /// Effective total snapshot cap.
    pub fn effective_total_cap(&self) -> usize {
        self.max_total_snapshots
            .unwrap_or(self.max_full_snapshots * (1 + self.max_deltas_per_full))
    }
}

// ─── Manifest v2 ────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: u32,
    /// Total number of ranks (world_size).
    pub world_size: u32,
    /// All snapshots, ordered by step.
    pub snapshots: Vec<Snapshot>,
    /// Retention policy for active lineage.
    pub retention: RetentionPolicy,
    /// Lineage configuration for rollback.
    pub lineage: LineageConfig,
}

impl Default for Manifest {
    fn default() -> Self {
        Manifest {
            format_version: 2,
            world_size: 1,
            snapshots: Vec::new(),
            retention: RetentionPolicy::default(),
            lineage: LineageConfig::default(),
        }
    }
}

impl Manifest {
    /// Find the last full (non-delta) snapshot.
    pub fn last_full_snapshot(&self) -> Option<&Snapshot> {
        self.snapshots
            .iter()
            .rev()
            .find(|s| s.base_snapshot_id.is_none() && s.finalized)
    }

    /// Find a snapshot by ID.
    pub fn find_snapshot(&self, id: Uuid) -> Option<&Snapshot> {
        self.snapshots.iter().find(|s| s.id == id)
    }

    /// Count consecutive deltas at the tail.
    pub fn pending_delta_count(&self) -> usize {
        self.snapshots
            .iter()
            .rev()
            .take_while(|s| s.base_snapshot_id.is_some())
            .count()
    }

    /// Whether a full snapshot should be forced at the given step.
    pub fn should_force_full(&self, step: u64) -> bool {
        if self.snapshots.is_empty() {
            return true;
        }

        // A cap of one keeps one snapshot, so a delta would be written
        // against a base that retention is about to remove - and the
        // delta cannot outlive it. Only a full can be the one that stays.
        if self.retention.effective_total_cap() <= 1 {
            return true;
        }

        if let Some(last_full) = self.last_full_snapshot() {
            if step.saturating_sub(last_full.step) >= self.retention.full_snapshot_every_steps {
                return true;
            }
        } else {
            return true; // No full snapshot exists
        }

        if self.pending_delta_count() >= self.retention.max_deltas_per_full {
            return true;
        }

        false
    }

    /// Get IDs of snapshots marked as rollback-safe.
    ///
    /// At most `max_rollback_snapshots` of them, newest first: protection is a
    /// standing exemption from retention, so without a cap every snapshot that
    /// ever landed on the interval stays forever and the store grows without
    /// bound. Past the cap the oldest simply stop being exempt — they are not
    /// deleted here, they go back to being ordinary snapshots that retention
    /// prunes in its own order. Which is what the two knobs mean together:
    /// `rollback_interval_steps` says which snapshots deserve protection,
    /// `max_rollback_snapshots` says how far back it reaches.
    /// Every snapshot retention must leave alone: pinned, rollback-protected,
    /// and the base a pinned delta is computed against.
    ///
    /// The base is the part worth spelling out. Pinning step 1200 protects the
    /// delta that *is* step 1200, and a delta without its base is bytes
    /// nothing can read - so the pin has to reach one step further back than
    /// the thing that was pinned.
    pub fn protected_snapshot_ids(&self) -> Vec<Uuid> {
        let mut ids = self.rollback_snapshot_ids();
        for snapshot in self.snapshots.iter().filter(|s| s.pinned) {
            ids.push(snapshot.id);
            if let Some(base) = snapshot.base_snapshot_id {
                ids.push(base);
            }
        }
        ids.sort();
        ids.dedup();
        ids
    }

    pub fn rollback_snapshot_ids(&self) -> Vec<Uuid> {
        // Zero disables rollback protection, the same meaning zero carries
        // for `merge_stride`, `compression_level` and `sync_every_n_saves`.
        // Reaching the modulo with it panics, and this runs inside retention,
        // which runs inside every save.
        if self.lineage.rollback_interval_steps == 0 || self.lineage.max_rollback_snapshots == 0 {
            return Vec::new();
        }

        // `snapshots` is ordered by step, so walking it backwards takes the
        // newest — the ones a rollback would actually reach for.
        let mut ids: Vec<Uuid> = self
            .snapshots
            .iter()
            .rev()
            .filter(|s| {
                s.base_snapshot_id.is_none()
                    && s.finalized
                    && s.step % self.lineage.rollback_interval_steps == 0
            })
            .take(self.lineage.max_rollback_snapshots)
            .map(|s| s.id)
            .collect();
        ids.reverse();
        ids
    }
}

// ─── Taking the lock ────────────────────────────────────────────────

/// Take the manifest lock, recovering it when a panic has poisoned it.
///
/// `catch_unwind` in `coordinator::saver` restores the *thread* after a panic
/// in the save pipeline; it does not touch the poison flag a `Mutex` sets when
/// a guard is dropped during unwinding. Left alone, the flag turns one panic
/// under this lock into a `PanicException` on every subsequent `save`, `load`,
/// `flush` and `list_snapshots` — raised on the caller's thread, from a place
/// unconnected to whatever actually failed, for the rest of the process.
///
/// Recovering is the right trade here and not merely the convenient one. The
/// invariant a poisoned `Manifest` could break is "the in-memory copy might be
/// half-updated", and that copy is **re-readable from storage**: the on-disk
/// manifest is only ever replaced whole, so `Core::reload_manifest` gets back a
/// consistent one. Walling off the coordinator for the rest of the run buys
/// nothing a re-read does not, and this is a library whose contract is that a
/// failed checkpoint costs that checkpoint, not the run.
///
/// The poison flag is cleared as well as bypassed, so `Mutex::is_poisoned`
/// stays a signal about the *last* panic rather than a latch that is stuck on
/// forever; [`crate::coordinator`] reads it to decide whether to re-read from
/// storage before trusting what is in memory.
pub(crate) fn lock_manifest(lock: &Mutex<Manifest>) -> MutexGuard<'_, Manifest> {
    lock.lock().unwrap_or_else(|poisoned| {
        lock.clear_poison();
        poisoned.into_inner()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_full_snapshot(step: u64) -> Snapshot {
        Snapshot {
            pinned: false,
            id: Uuid::new_v4(), step, created_at: Utc::now(),
            ranks: HashMap::new(), base_snapshot_id: None,
            metadata: HashMap::new(), compression: CompressionAlgo::default(),
            finalized: true,
        }
    }

    fn make_delta_snapshot(step: u64, base_id: Uuid) -> Snapshot {
        Snapshot {
            pinned: false,
            id: Uuid::new_v4(), step, created_at: Utc::now(),
            ranks: HashMap::new(), base_snapshot_id: Some(base_id),
            metadata: HashMap::new(), compression: CompressionAlgo::default(),
            finalized: true,
        }
    }

    #[test]
    fn empty_manifest_forces_full() {
        let m = Manifest::default();
        assert!(m.should_force_full(0));
    }

    #[test]
    fn force_full_after_steps() {
        let mut m = Manifest { retention: RetentionPolicy { full_snapshot_every_steps: 10, ..Default::default() }, ..Default::default() };
        m.snapshots.push(make_full_snapshot(0));
        assert!(!m.should_force_full(5));
        assert!(m.should_force_full(10));
    }

    #[test]
    fn force_full_after_max_deltas() {
        let mut m = Manifest { retention: RetentionPolicy { max_deltas_per_full: 3, full_snapshot_every_steps: 100000, ..Default::default() }, ..Default::default() };
        let full = make_full_snapshot(0);
        let base_id = full.id;
        m.snapshots.push(full);
        for i in 1..=3 { m.snapshots.push(make_delta_snapshot(i, base_id)); }
        assert!(m.should_force_full(4));
    }

    #[test]
    fn pending_delta_count() {
        let mut m = Manifest::default();
        assert_eq!(m.pending_delta_count(), 0);
        let full = make_full_snapshot(0);
        let base_id = full.id;
        m.snapshots.push(full);
        m.snapshots.push(make_delta_snapshot(1, base_id));
        m.snapshots.push(make_delta_snapshot(2, base_id));
        assert_eq!(m.pending_delta_count(), 2);
        m.snapshots.push(make_full_snapshot(3));
        assert_eq!(m.pending_delta_count(), 0);
    }

    #[test]
    fn last_full_skips_unfinalized() {
        let mut m = Manifest::default();
        m.snapshots.push(make_full_snapshot(0));
        let mut unf = make_full_snapshot(100);
        unf.finalized = false;
        m.snapshots.push(unf);
        assert_eq!(m.last_full_snapshot().unwrap().step, 0);
    }

    #[test]
    fn rollback_ids() {
        let mut m = Manifest { lineage: LineageConfig { rollback_interval_steps: 100, max_rollback_snapshots: 5 }, ..Default::default() };
        m.snapshots.push(make_full_snapshot(0));
        m.snapshots.push(make_full_snapshot(50));
        m.snapshots.push(make_full_snapshot(100));
        assert_eq!(m.rollback_snapshot_ids().len(), 2); // 0 and 100
    }

    /// Protection is capped, newest first. Without the cap the exemption list
    /// grows with the run and retention has nothing left it is allowed to
    /// delete.
    #[test]
    fn rollback_ids_are_capped_at_the_newest_n() {
        let mut m = Manifest {
            lineage: LineageConfig { rollback_interval_steps: 100, max_rollback_snapshots: 2 },
            ..Default::default()
        };
        for step in [0u64, 100, 200, 300] {
            m.snapshots.push(make_full_snapshot(step));
        }

        let protected = m.rollback_snapshot_ids();
        assert_eq!(protected.len(), 2);

        let steps: Vec<u64> = m
            .snapshots
            .iter()
            .filter(|s| protected.contains(&s.id))
            .map(|s| s.step)
            .collect();
        assert_eq!(steps, vec![200, 300], "the newest two are the protected ones");
    }

    /// Zero means no protection, the same as it does for the interval — and it
    /// is the reading someone reaches for to turn the feature off.
    #[test]
    fn a_max_of_zero_protects_nothing() {
        let mut m = Manifest {
            lineage: LineageConfig { rollback_interval_steps: 100, max_rollback_snapshots: 0 },
            ..Default::default()
        };
        m.snapshots.push(make_full_snapshot(0));
        m.snapshots.push(make_full_snapshot(100));
        assert!(m.rollback_snapshot_ids().is_empty());
    }

    #[test]
    fn effective_total_cap_default() {
        let p = RetentionPolicy {
            max_full_snapshots: 3,
            max_deltas_per_full: 10,
            ..Default::default()
        };
        assert_eq!(p.effective_total_cap(), 33);
    }

    #[test]
    fn effective_total_cap_explicit() {
        let p = RetentionPolicy {
            max_full_snapshots: 3,
            max_deltas_per_full: 10,
            max_total_snapshots: Some(5),
            ..Default::default()
        };
        assert_eq!(p.effective_total_cap(), 5);
    }

    #[test]
    fn serialization_roundtrip() {
        let mut m = Manifest::default();
        m.snapshots.push(make_full_snapshot(100));
        let json = serde_json::to_vec(&m).unwrap();
        let d: Manifest = serde_json::from_slice(&json).unwrap();
        assert_eq!(d.snapshots.len(), 1);
        assert_eq!(d.snapshots[0].step, 100);
    }
}
