# Revolver

**Stop losing checkpoints. Start training fearlessly.**

Revolver is a high-performance checkpoint engine for ML training, written in Rust with Python bindings. It tracks per-tensor deltas, skips unchanged weights entirely, and compresses the rest — saving checkpoints in **0.4s instead of minutes**, with **40%+ less storage**.

Three lines in your training loop. That's it.

```python
from revolver import CheckpointManager

mgr = CheckpointManager("./checkpoints", save_dtype="bf16")
start_step = mgr.resume(model=model, optimizer=optimizer, scheduler=scheduler)

for step in range(start_step, 100_000):
    loss = train_step(model, batch)

    if step % 500 == 0:
        mgr.save(step=step, model=model, optimizer=optimizer,
                 metadata={"loss": f"{loss:.4f}"})
```

## Why

Every ML engineer has lost a training run. The spot instance dies, the node crashes, the disk fills up — and your last checkpoint was 2 hours ago. So you save more often, but now checkpointing is the bottleneck: a 3B model in bf16 is ~5.5 GB per save, and `torch.save` blocks your training loop every time.

Revolver fixes this at the storage layer. Instead of dumping the full state dict every time, it diffs against the previous checkpoint at the tensor level: unchanged tensors → zero I/O, changed tensors → XOR delta + zstd compression. The result is saves that are both faster and smaller.

Inspired by [DECK (Meta, PVLDB 2025)](https://doi.org/10.14778/3750601.3750621).

## Benchmarks

Measured on MiniGPT 3.2M, CPU, bf16 checkpoints:

| | Revolver (delta) | `torch.save` |
|---|---|---|
| Save time | **0.4–0.5s** | ~50s (Python serialization) |
| Storage per checkpoint | **26–51% smaller** | full state dict |
| Cumulative storage (10 checkpoints) | **42.6% saved** vs naive |
| Skipped tensors per save | 5–13 | 0 (saves everything) |

Resume integrity verified: max diff 0.003906 (bf16 quantization noise, not data loss).

## Features

- **Per-tensor delta tracking** — unchanged tensors are skipped entirely (zero I/O), changed tensors use XOR delta compression
- **Rust-native dtype casting** — `save_dtype="bf16"` casts fp32→bf16 in parallel Rust threads before compression, auto-uncasts on load
- **4KB page-aligned writes** — eliminates SSD write amplification, extends drive lifespan
- **Rank-aware distributed saves** — each rank saves its own shard independently, auto-detects `torchrun` env vars
- **Hierarchical delta merging** — background thread consolidates deltas to keep load times fast
- **S3 backend** — local SSD as primary (fast), batched sync to S3/MinIO/R2 in background
- **SHA-256 integrity checks** — every tensor verified on read, corruption detected immediately
- **Auto-resume** — `mgr.resume()` loads the latest checkpoint if it exists, returns the next step

## Installation

**Requirements:** [Rust toolchain](https://rustup.rs/) + Python ≥ 3.9

```bash
# From git (recommended)
pip install git+https://codeberg.org/JHNMACHINE/revolver.git

# On cloud instances (Vast.ai, RunPod, Lambda, etc.)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
pip install git+https://codeberg.org/JHNMACHINE/revolver.git
```

### Build from source

```bash
git clone https://codeberg.org/JHNMACHINE/revolver.git
cd revolver
pip install maturin
maturin develop --release
```

## S3 Backend

Checkpoint to local SSD for speed, sync to S3-compatible storage for durability:

```python
mgr = CheckpointManager(
    "./checkpoints",
    save_dtype="bf16",
    s3_bucket="my-bucket",
    s3_access_key="...",
    s3_secret_key="...",
    s3_endpoint="http://localhost:9000",   # MinIO / R2 / B2
    s3_path_style=True,
    sync_every_n_saves=5,
)
```

## Multi-GPU (FSDP / DDP)

Revolver auto-detects `torchrun` environment variables. No configuration needed:

```bash
torchrun --nproc_per_node=8 train.py
```

```python
mgr = CheckpointManager("./checkpoints")
# mgr.rank == 3, mgr.world_size == 8  (auto-detected)
start_step = mgr.resume(model=model, optimizer=optimizer)
```

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
cargo test                          # Rust tests
maturin develop --release && pytest tests/ -v   # Python + PyTorch tests
```

## License

Apache-2.0 — [Minya AI](https://minya.ai)
