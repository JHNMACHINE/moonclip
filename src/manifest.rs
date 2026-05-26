use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
}

/// Metadata for a single tensor within a rank's checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    /// Fully qualified name, e.g. "model.layers.0.self_attn.q_proj.weight"
    pub name: String,
    /// Shape as a list of dims.
    pub shape: Vec<usize>,
    /// Element dtype string, e.g. "bfloat16", "float32".
    pub dtype: String,
    /// How this tensor is stored.
    pub storage: TensorStorage,
    /// Path to the compressed data file (None if Skipped).
    pub filename: Option<String>,
    /// Byte size of compressed data on disk.
    pub compressed_size: u64,
    /// Byte size of raw (uncompressed) tensor data.
    pub raw_size: u64,
    /// SHA-256 of the raw tensor bytes (always present, used for skip detection).
    pub sha256_raw: String,
    /// SHA-256 of the compressed data on disk (None if Skipped).
    pub sha256_compressed: Option<String>,
}

// ─── Rank-level metadata ────────────────────────────────────────────

/// All tensor entries for a single rank in a snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankEntry {
    pub rank: u32,
    pub tensors: Vec<TensorEntry>,
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
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        RetentionPolicy {
            max_full_snapshots: 5,
            max_deltas_per_full: 10,
            full_snapshot_every_steps: 5000,
        }
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
    pub fn rollback_snapshot_ids(&self) -> Vec<Uuid> {
        self.snapshots
            .iter()
            .filter(|s| {
                s.base_snapshot_id.is_none()
                    && s.finalized
                    && s.step % self.lineage.rollback_interval_steps == 0
            })
            .map(|s| s.id)
            .collect()
    }
}
