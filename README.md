# Revolver

High-performance checkpoint management for ML training, written in Rust with Python bindings.

Inspired by [DECK (Meta, PVLDB 2025)](https://doi.org/10.14778/3750601.3750621) — per-tensor delta tracking, rank-aware distributed saves, hierarchical merging, and S3 backend.

## Installation

**Requirements:** [Rust toolchain](https://rustup.rs/) (rustup) + Python ≥ 3.9

### From git (recommended)

```bash
pip install git+https://codeberg.org/JHNMACHINE/revolver.git
```

### From source

```bash
git clone https://codeberg.org/JHNMACHINE/revolver.git
cd revolver
pip install maturin
maturin develop --release
```

### On cloud instances

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
pip install git+https://codeberg.org/JHNMACHINE/revolver.git
```

## Quick Start

```python
from revolver import CheckpointManager

# Single line setup — auto-detects torchrun rank/world_size
mgr = CheckpointManager("./checkpoints", save_dtype="bf16")

# Auto-resume: returns 0 if no checkpoints, else last_step + 1
start_step = mgr.resume(model=model, optimizer=optimizer, scheduler=scheduler)

for step in range(start_step, 100_000):
    # ... training ...

    if step % 1000 == 0:
        mgr.save(step=step, model=model, optimizer=optimizer,
                 metadata={"loss": f"{loss:.4f}"})

# End of training
mgr.merge_now()
mgr.print_stats()
```

## Features

- **Per-tensor delta tracking** — unchanged tensors are skipped entirely (zero I/O), changed tensors use XOR delta compression
- **Rust-native dtype casting** — `save_dtype="bf16"` casts fp32→bf16 in parallel Rust threads before compression, auto-uncasts on load
- **4KB page-aligned writes** — eliminates SSD write amplification, extends drive lifespan
- **Rank-aware distributed saves** — each rank saves its own shard independently, auto-detects `torchrun` env vars
- **Hierarchical delta merging** — background thread consolidates deltas to keep load times fast
- **S3 backend** — local SSD as primary (fast), batched sync to S3/MinIO/R2 in background
- **SHA-256 integrity checks** — every tensor verified on read, corruption detected immediately
- **Auto-resume** — `mgr.resume()` loads the latest checkpoint if it exists, returns the next step

## S3 Backend

```python
mgr = CheckpointManager(
    "./checkpoints",                        # local SSD (primary, fast)
    save_dtype="bf16",
    s3_bucket="my-bucket",                  # S3 (backup, durable)
    s3_access_key="...",
    s3_secret_key="...",
    s3_endpoint="http://localhost:9000",     # MinIO / R2 / B2
    s3_path_style=True,                     # required for MinIO
    sync_every_n_saves=5,                   # batch sync every 5 saves
)
```

## Multi-rank (FSDP / DDP)

```python
# torchrun --nproc_per_node=8 train.py
# CheckpointManager auto-detects RANK and WORLD_SIZE from torchrun

mgr = CheckpointManager("./checkpoints")
# mgr.rank == 3, mgr.world_size == 8  (auto-detected)

start_step = mgr.resume(model=model, optimizer=optimizer)
```

## Benchmarks (MiniGPT 3.2M, CPU, bf16 checkpoints)

| Metric | Value |
|---|---|
| Checkpoint time (delta) | 0.4–0.5s |
| Checkpoint time (full) | 2.9s |
| Compression vs raw | 26–51% saved |
| Storage vs naive torch.save | **42.6% saved** |
| Skipped tensors per delta save | 5–13 |
| Delta tensors per save | 1–20 |
| Resume verification | ✓ max diff 0.003906 (bf16 precision) |

## Architecture

```
src/
├── cast.rs          # Rust-native fp32↔bf16/fp16 casting (rayon parallel)
├── coordinator.rs   # Rank-aware snapshot lifecycle
├── tensor.rs        # Per-tensor delta tracking and storage
├── manifest.rs      # Manifest v2: per-rank, per-tensor, lineage
├── compression.rs   # Zstd compression
├── delta.rs         # XOR delta computation (rayon parallel)
├── merger.rs        # Background delta merging
├── remote_sync.rs   # Batched S3 sync
├── s3.rs            # S3-compatible storage (AWS SigV4)
├── storage.rs       # StorageBackend trait + LocalStorage (4KB aligned)
├── background.rs    # Background thread infrastructure
├── hash.rs          # SHA-256 integrity
├── python.rs        # PyO3 bindings
└── error.rs         # Error types
```

## Development

```bash
# Run Rust tests
cargo test

# Run Python tests (no torch needed)
pip install pytest
maturin develop --release
pytest tests/test_integration.py tests/test_distributed.py -v

# Run PyTorch tests
pip install torch
pytest tests/ -v
```

## License

Apache-2.0 — [Minya AI](https://minya.ai)
