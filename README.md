# 🔫 Revolver

High-performance checkpoint management for ML training, written in Rust with Python bindings.

## Features

- **Delta compression** — XOR-based byte-level deltas between consecutive checkpoints. Only changed bytes are stored, and the resulting delta compresses extremely well with zstd (typical 80-99% savings on optimizer states).
- **Zstd compression** — Configurable compression levels (0-22). Default level 3 gives excellent ratio/speed tradeoff.
- **SHA-256 integrity verification** — Every shard is checksummed on write and verified on read. Corrupted checkpoints are caught immediately.
- **Atomic writes** — Uses temp-file-then-rename to prevent half-written checkpoints from power failures or OOM kills.
- **Automatic retention policy** — Configurable limits on full snapshots and deltas. Old checkpoints are cleaned up automatically.
- **Pluggable storage backends** — Local filesystem included; S3/GCS planned.
- **PyTorch integration** — `TorchCheckpointManager` wraps state_dict serialization for seamless use with `nn.Module`, `Optimizer`, `LRScheduler`, `GradScaler`.

## Installation

### From source (requires Rust toolchain)

```bash
# Build the Rust library
cargo build --release

# Copy the .so into the Python package
cp target/release/librevolver.so python/revolver/revolver.cpython-3XX-x86_64-linux-gnu.so

# Add to PYTHONPATH
export PYTHONPATH="/path/to/revolver/python:$PYTHONPATH"
```

### With maturin (requires Rust ≥ 1.80)

```bash
pip install maturin
maturin develop --release
```

## Quick Start

### Raw bytes API

```python
from revolver import RevolverManager

mgr = RevolverManager(
    storage_root="./checkpoints",
    compression_level=3,
    max_full_snapshots=5,
    full_every_steps=5000,
)

# Save
snap_id = mgr.save(
    step=1000,
    components={"model": model_bytes, "optimizer": optim_bytes},
    metadata={"loss": "0.634", "lr": "1e-4"},
)

# Load latest
snap_id, components = mgr.load_latest()

# Load specific
components = mgr.load(snap_id)

# List all
for snap in mgr.list_snapshots():
    print(f"Step {snap['step']}, delta={snap['is_delta']}, loss={snap['metadata'].get('loss')}")
```

### PyTorch API

```python
from revolver.pytorch import TorchCheckpointManager

ckpt = TorchCheckpointManager("./checkpoints")

# Save (accepts nn.Module, Optimizer, or raw state_dicts)
ckpt.save(step=1000, model=model, optimizer=optimizer, metadata={"loss": "0.634"})

# Load latest and apply state_dicts
snap_id, state_dicts = ckpt.load_latest(model=model, optimizer=optimizer)
```

## Architecture

```
revolver/
├── src/
│   ├── lib.rs          # Crate root
│   ├── error.rs        # Error types
│   ├── manifest.rs     # Snapshot/shard metadata structs
│   ├── compression.rs  # Zstd compress/decompress
│   ├── delta.rs        # XOR delta computation (parallel via rayon)
│   ├── hash.rs         # SHA-256 utilities
│   ├── storage.rs      # StorageBackend trait + LocalStorage
│   ├── manager.rs      # CheckpointManager orchestrator
│   └── python.rs       # PyO3 bindings
├── python/
│   └── revolver/
│       ├── __init__.py # Re-exports from Rust
│       └── pytorch.py  # PyTorch convenience wrapper
├── tests/
│   └── test_integration.py
├── Cargo.toml
└── pyproject.toml
```

## How Delta Compression Works

1. On each save, Revolver checks if a previous full snapshot exists for each component.
2. It computes a byte-level XOR between the old and new data (parallelized in 1MB chunks via rayon).
3. If the "delta density" (fraction of non-zero bytes) is below the threshold (default 0.5), the delta is stored instead of the full data.
4. The delta is then zstd-compressed. Since most bytes in the XOR are zero (unchanged parameters), zstd achieves extreme compression ratios (often 95-99%).
5. On load, the delta chain is resolved: base snapshot → apply delta → verify SHA-256.

## Roadmap

- [ ] S3 / GCS storage backend
- [ ] Async I/O (tokio)
- [ ] Per-tensor delta tracking (skip unchanged layers entirely)
- [ ] safetensors format support for direct tensor-level addressing
- [ ] FSDP-aware sharding (save per-rank, consolidate on load)
- [ ] CLI tool for checkpoint inspection/management

## License

Apache-2.0
