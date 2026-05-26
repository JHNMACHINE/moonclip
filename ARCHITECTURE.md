# Revolver v1.0 — Architecture Design

## Design Principles

1. **Rank-native**: Every operation is rank-aware from day one. Single-rank is just `world_size=1`.
2. **Per-tensor granularity**: Delta tracking at tensor level, not blob level.
3. **Scale-independent**: Same code path for 2.74B on 1 GPU and 70B on 64 GPUs.
4. **Zero-copy where possible**: Tensors go directly from PyTorch to Rust as byte slices.
5. **DECK-aligned**: Follow the three pillars — zero-cost tracking, multi-layer staging, hierarchical merging.

## Core Concepts

### Rank
A logical unit that owns a shard of the model. Maps 1:1 to a GPU in FSDP.
Each rank saves/loads independently. Rank 0 is the coordinator.

### TensorRecord
A single named tensor within a checkpoint. Stored individually, not as part of
a monolithic blob. Each record has: name, shape, dtype, raw bytes, sha256.

### Snapshot
One "save event" across ALL ranks at a given step. Contains:
- snapshot_id (UUID)
- step
- per-rank list of TensorRecords (or delta references)
- base_snapshot_id (if delta)
- metadata

### Delta Strategy
For each tensor in a new checkpoint:
1. Compare sha256 with same tensor in base snapshot
2. If identical → skip entirely (DECK "zero-cost": most tensors don't change much between saves)
3. If different but same shape → XOR delta + zstd
4. If different shape → full save (rare: only on architecture change)

This is the DECK insight: in a 2.74B model with 100+ tensors, many haven't changed
significantly between step 1000 and step 2000. Skip them entirely.

### Coordinator Pattern
```
Rank 0                    Rank 1..N
  │                          │
  ├─ create_snapshot()       │
  │   → snap_id, step        │
  │                          │
  ├─ broadcast snap_id ──────┤
  │                          │
  ├─ save_rank(0, tensors)   ├─ save_rank(i, tensors)
  │   (parallel, async)      │   (parallel, async)
  │                          │
  ├─ barrier ────────────────┤
  │                          │
  ├─ finalize_snapshot()     │
  │   → update manifest      │
  │   → trigger merger       │
```

For single-rank (Harold today): this collapses to just save_rank(0, tensors) + finalize.

### Storage Layout
```
checkpoints/
├── manifest.json                     # Global manifest (written by rank 0)
├── snapshots/
│   ├── {snap_id}/
│   │   ├── rank_0/
│   │   │   ├── model.layers.0.weight.bin     # Individual tensor
│   │   │   ├── model.layers.0.bias.bin
│   │   │   ├── optimizer.state.0.exp_avg.bin
│   │   │   └── _index.json                   # Per-rank tensor index
│   │   ├── rank_1/
│   │   │   └── ...
│   │   └── rank_2/
│   │       └── ...
```

### Manifest v2
```json
{
  "format_version": 2,
  "world_size": 8,
  "snapshots": [
    {
      "id": "uuid",
      "step": 1000,
      "base_snapshot_id": null,
      "ranks": {
        "0": {
          "tensors": [
            {
              "name": "model.layers.0.weight",
              "shape": [512, 512],
              "dtype": "bfloat16",
              "storage": "full",           // or "delta_xor" or "skipped"
              "filename": "snapshots/{id}/rank_0/model.layers.0.weight.bin",
              "compressed_size": 12345,
              "raw_size": 524288,
              "sha256_raw": "abc...",
              "sha256_compressed": "def..."
            }
          ],
          "total_compressed": 1234567,
          "total_raw": 5242880
        }
      },
      "metadata": {"loss": "0.634"},
      "compression": {"Zstd": {"level": 3}},
      "created_at": "2026-05-25T10:00:00Z"
    }
  ],
  "retention": {...},
  "lineage": {
    "rollback_full_snapshots": ["uuid1", "uuid2"],
    "rollback_interval_steps": 10000
  }
}
```

### Chunked Staging Pipeline
For large tensors (>chunk_size, default 64MB):
1. Split into chunks
2. Pipeline: chunk N compressing → chunk N-1 writing to local → chunk N-2 uploading to S3
3. Each chunk is independently addressable (for parallel download on resume)

For small tensors (<chunk_size): single-pass compress→write.

### Hierarchical Merging (from DECK Section 3.3)
Same as v0.3 but now operates per-rank, per-tensor:
- Stride merge: every S deltas → 1 merged delta (per tensor)
- Full merge: consolidate all deltas into base (creates new full snapshot)
- Lineage graph: keep old full snapshots for rollback

### Lineage Graph (from DECK Section 4.3)
Two categories of retained checkpoints:
1. **Active lineage** (red in DECK Fig 9): latest full + deltas → for fast failure recovery
2. **Rollback lineage** (grey in DECK Fig 9): periodic full snapshots → for data corruption rollback

## Rust Module Structure

```
src/
├── lib.rs
├── error.rs          # Error types (keep)
├── manifest.rs       # NEW: Manifest v2 with rank-aware structures
├── tensor.rs         # NEW: TensorRecord, per-tensor delta logic
├── compression.rs    # Keep, add chunked compression
├── delta.rs          # Keep, used by tensor.rs
├── hash.rs           # Keep
├── storage.rs        # Keep StorageBackend trait + LocalStorage
├── s3.rs             # Keep
├── coordinator.rs    # NEW: Snapshot lifecycle (create/save_rank/finalize)
├── staging.rs        # NEW: Chunked pipeline staging
├── merger.rs         # Keep, adapt to per-rank per-tensor
├── lineage.rs        # NEW: Lineage graph management
├── background.rs     # Keep
├── python.rs         # Rewrite for new API
```

## Python API (target)

```python
from revolver import CheckpointManager

# Single-rank (Harold today)
mgr = CheckpointManager(
    storage_root="./checkpoints",
    world_size=1,
    rank=0,
)

# Save: pass state_dict directly, Revolver handles per-tensor split
mgr.save(step=1000, model=model, optimizer=optimizer, metadata={"loss": "0.634"})

# Load latest
mgr.load_latest(model=model, optimizer=optimizer)

# Multi-rank (Harold v1 on 8×H100)
mgr = CheckpointManager(
    storage_root="./checkpoints",  # or s3_bucket=...
    world_size=8,
    rank=dist.get_rank(),
)

# Each rank saves its own FSDP shard
mgr.save(step=1000, model=model, optimizer=optimizer)
# Internally: rank 0 creates snapshot, all ranks save their tensors, rank 0 finalizes

# End of training
mgr.merge_now()  # Consolidate all deltas
```

## Implementation Order

1. manifest.rs — New Manifest v2 structures
2. tensor.rs — Per-tensor delta tracking and storage
3. coordinator.rs — Snapshot lifecycle
4. staging.rs — Chunked pipeline
5. lineage.rs — Rollback support
6. merger.rs — Adapt to new structures
7. python.rs — New API
8. Tests throughout
