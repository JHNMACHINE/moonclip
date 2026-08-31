# Moonclip

**Stop losing checkpoints. Start training fearlessly.**

Moonclip is a checkpoint engine for ML training, written in Rust with Python bindings. It tracks per-tensor deltas, skips unchanged weights entirely, and compresses the rest, so a save blocks the training loop for **tens of milliseconds** and the checkpoint on disk is **about half the size**.

Against `torch.save` on dense pre-training — the least favourable case, where Adam changes every parameter at every step and there is nothing to skip — that is 1.6-3.0× faster and 1.9× smaller. The gap widens on fine-tuning, LoRA and adapters, where most tensors are identical between two checkpoints.

A moonclip is the ring that holds a full circle of rounds so a revolver reloads in one motion, instead of one chamber at a time. That is the idea here: your whole training state goes down and comes back in a single movement, not tensor by tensor.

Three lines in your training loop. That's it.

```python
from moonclip import CheckpointManager

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

Moonclip fixes this at the storage layer. Instead of dumping the full state dict every time, it diffs against the previous checkpoint at the tensor level: unchanged tensors → zero I/O, changed tensors → XOR delta + zstd compression. The result is saves that are both faster and smaller.

Inspired by [DECK (Meta, PVLDB 2025)](https://doi.org/10.14778/3750601.3750621).

## Benchmarks

MiniGPT 41.7M params, fp32 model + full AdamW optimizer state (~540 MB per
checkpoint), 10 saves, CPU (`bench/benchmark_checkpoints.py`):

| | Moonclip | safetensors* |
|---|---|---|
| Avg save (training loop blocked) | **35 ms** | 316 ms |
| Total for 10 saves | **0.35 s** | 3.2 s |
| Load | 0.44 s | 0.03 s |

\* safetensors saves the model only — no optimizer state, ~3× less data per checkpoint.

Saves are asynchronous by default: `save()` returns as soon as the tensor
data has been copied, while hashing, delta detection, zstd compression and
the disk write run on a background thread and overlap with training. Call
`flush()` when you need the checkpoint on disk — as of 0.0.6 that means
`fsync`ed, not merely written; loads and
`list_snapshots()` wait for pending saves automatically. Pass
`async_save=False` for fully synchronous saves.

Resume integrity verified: max weight diff 0.0 after save → load.

## Features

- **Async background saves** — `save()` returns in milliseconds; compression and I/O overlap with training (disable with `async_save=False`)
- **Per-tensor delta tracking** — unchanged tensors are skipped entirely (zero I/O), changed tensors use XOR delta compression; a cheap sampled density check bails out early when everything changed
- **Parallel zstd** — large tensors are compressed/decompressed as concatenated zstd frames across all cores
- **Rust-native dtype casting** — `save_dtype="bf16"` casts fp32→bf16 in parallel Rust threads before compression, auto-uncasts on load
- **Float8** — tensors that arrive as `float8_e4m3fn`/`float8_e5m2` (FSDP2, torchao) are stored bit-exact with no configuration. `save_dtype="fp8"` additionally quantizes fp32/bf16 down to one byte per element against a per-tensor scale — a quarter the size, four significant bits, so for archived copies rather than checkpoints you resume from
- **4KB page-aligned writes** — eliminates SSD write amplification, extends drive lifespan
- **Rank-aware distributed saves** — each rank saves its own shard independently, auto-detects `torchrun` env vars
- **Hierarchical delta merging** — background thread consolidates deltas to keep load times fast
- **S3 backend** — local SSD as primary (fast), batched sync to S3/MinIO/R2 in background
- **xxHash3-128 integrity checks** — every tensor verified on read, corruption detected immediately
- **Auto-resume** — `mgr.resume()` loads the latest checkpoint if it exists, returns the next step

## Installation

```bash
pip install moonclip
```

Wheels are built for **Linux x86_64 (manylinux_2_28), CPython 3.9-3.14** — the
platform training actually runs on. No Rust toolchain needed there; the
extension is compiled. What changed between versions is in
[CHANGELOG.md](https://codeberg.org/JHNMACHINE/moonclip/src/branch/main/CHANGELOG.md).

On any other platform (Windows, macOS, aarch64) `pip` finds no wheel and stops.
Build it yourself instead, which needs a [Rust toolchain](https://rustup.rs/):

```bash
pip install git+https://codeberg.org/JHNMACHINE/moonclip.git

# On a cloud instance without Rust (Vast.ai, RunPod, Lambda, …)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
pip install git+https://codeberg.org/JHNMACHINE/moonclip.git
```

### Build from source

```bash
git clone https://codeberg.org/JHNMACHINE/moonclip.git
cd moonclip
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

Say which topology the manager is for. Under `torchrun`, `"auto"` takes it
from the launcher:

```bash
torchrun --nproc_per_node=8 train.py
```

```python
mgr = CheckpointManager("./checkpoints", world_size="auto", rank="auto")
# mgr.rank == 3, mgr.world_size == 8
start_step = mgr.resume(model=model, optimizer=optimizer)
```

The ranks then share one store, through the explicit
`create_snapshot` / `save_rank` / `finalize_snapshot` flow. If instead each
rank writes to a directory of its own — one store per process, no coordination
— then every one of them is single-rank and should say so:

```python
mgr = CheckpointManager(f"./checkpoints/rank_{rank}", world_size=1, rank=0)
```

Until 0.0.9 the launcher's variables were adopted whenever nothing was stated.
That is no longer the default: leaving it unsaid under a multi-rank launcher
is an error naming both numbers and the ways forward. A single process is
unaffected and still needs no configuration.

## Tuning

Three environment variables, none required:

| | |
|---|---|
| `MOONCLIP_THREADS` | Size of Moonclip's thread pool. Default: `cores / LOCAL_WORLD_SIZE`. |
| `MOONCLIP_PROFILE=1` | Per-phase breakdown of the save path on stderr, plus a line whenever a save had to wait for the previous one to drain. |
| `MOONCLIP_FSYNC=0` | Skip the `fsync` before a pack is renamed into place. Faster, and a machine that loses power mid-save can then come back holding a pack of the right length full of zeros — which the manifest vouches for. Only worth it where the checkpoint is not the thing being protected. |

The parallel work runs in a pool of Moonclip's own, not rayon's global one, so it
neither claims every core on the machine nor competes with your application's
`par_iter`. On a node running several ranks the default splits the cores between
them — `LOCAL_WORLD_SIZE` is what `torchrun` sets — so eight ranks on 128 cores
take 16 threads each rather than 128 apiece.

Set `MOONCLIP_THREADS` when that guess is wrong for your box: the machine is
Moonclip's alone (give it every core), or the ranks do not all checkpoint at the
same time.

## Architecture

Two layers, and the line between them is the whole design. The Rust crate
stores *named tensors* and has never heard of PyTorch; everything that knows
what a `state_dict` is lives in the Python package on top. That is why
Moonclip can sit under a runtime like [Ravex](https://codeberg.org/JHNMACHINE/ravex)
without either one owning the other — and why `import moonclip` does not
import torch.

**`src/` — the engine (Rust)**

```
src/
├── lib.rs           # Crate root
├── coordinator/     # Snapshot lifecycle: create → save_rank → finalize, per rank
│   ├── mod.rs       #   The Coordinator facade and the Core behind it
│   ├── recovery.rs  #   Putting a store back together at open time
│   ├── saver.rs     #   The background thread a save is handed to
│   └── tests.rs     #   Reaches into Core and the packs; not an integration test
├── tensor.rs        # Per-tensor delta tracking and storage
├── cast.rs          # fp32 ↔ bf16/fp16/float8 casting, in parallel
├── delta.rs         # XOR delta computation, in parallel
├── shuffle.rs       # Byte-plane transpose, so a delta compresses
├── compression.rs   # Zstd, as concatenated frames across cores
├── pack.rs          # The pack file: data plus its own description, one write
├── manifest.rs      # Manifest v2 — per-rank, per-tensor, lineage
├── merger.rs        # Background consolidation of delta chains
├── inflight.rs      # Which snapshots a read holds, so a merge cannot delete them
├── storage.rs       # StorageBackend trait + LocalStorage (4KB aligned)
├── s3.rs            # S3-compatible storage (AWS SigV4, no SDK)
├── remote_sync.rs   # Batched sync to the remote backend
├── pool.rs          # Moonclip's private rayon pool — see Tuning
├── hash.rs          # xxHash3-128 integrity
├── profile.rs       # Opt-in phase timing (MOONCLIP_PROFILE)
├── python.rs        # PyO3 bindings — the only file that knows Python exists
└── error.rs         # Error types
```

**`python/moonclip/` — the PyTorch surface**

```
python/moonclip/
├── __init__.py      # Re-exports; torch resolves lazily, never at import time
├── pytorch.py       # CheckpointManager — state dicts in, snapshots out
├── _env.py          # Reads torchrun's topology variables (detects; does not adopt)
└── *.pyi, py.typed  # Stubs, for both layers
```

**And around them**

| Path | |
|---|---|
| `tests/` | PyTorch and distributed tests, plus `s3_minio.rs` against a live MinIO |
| `benches/` | Criterion microbenchmarks for the delta path |
| `bench/` | End-to-end checkpoint benchmarks, and the GPU scripts behind them |
| `examples/` | Runnable: plain model, transformer, DDP, S3 |
| `.forgejo/workflows/` | `checks.yml` on branches, `ci.yml` on main, `bench.yml`, `release.yml` |

As of 0.1.0 that is about 15.6k lines of Rust across 22 files and 1.4k of
Python, covered by 222 crate tests, 19 more against a live MinIO, and 108
Python ones. Roughly half of the Rust is `#[cfg(test)]`: `coordinator/`
carries 2.2k lines of tests against 2.3k of code, which is why it is the one
part of the crate laid out as a directory.

## Development

```bash
cargo test                                      # Rust tests, no Python needed
maturin develop --release && pytest tests/ -v   # Python + PyTorch tests
cargo clippy --all-targets -- -D warnings       # what CI gates on
```

`checks.yml` runs clippy and `cargo test` on every branch push — one container,
seconds. The full matrix (MinIO, the PyTorch adapter, CPython 3.9–3.14) waits
for `main`, in `ci.yml`.

Both pin `rust:1.97.1-bookworm`, and `rust-toolchain.toml` pins the same
version for a local build, so a red `cargo clippy` here means the gate is red
too. That agreement is the point of the file: the two used to differ — CI on
1.90, development on 1.97 — and the crate spent a while green in CI and red
locally on four findings that were not regressions, only lints the older
clippy did not have.

The toolchain file is **not** the MSRV. `rust-version` in `Cargo.toml` says
1.83, which is what a consumer of the published crate needs; a
`rust-toolchain.toml` applies only inside this directory and does not raise
that floor. It does mean nothing verifies the 1.83 claim any more.

Note that the crate is deliberately **not** `cargo fmt`-clean: the gate is
clippy, and running `cargo fmt` over it produces a diff nobody asked for.

## License

Apache-2.0 — [Minya AI](https://minya.ai)
