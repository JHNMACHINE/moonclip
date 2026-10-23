# Moonclip

A checkpoint engine for ML training, written in Rust with Python bindings. It
compares each tensor with the previous checkpoint, skips the ones that did not
change, stores the others as a compressed XOR delta, and writes in the
background, so a save holds the training loop only for the copy of the
tensors.

On dense pre-training, the least favourable case because Adam changes every
parameter at every step, that measured 1.6-3.0× faster than `torch.save` and
1.9× smaller. Fine-tuning, LoRA and adapters leave most tensors unchanged
between checkpoints and should gain more; that has not been measured.
The design follows [DECK (Meta, PVLDB 2025)](https://doi.org/10.14778/3750601.3750621).

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

`resume()` loads the latest checkpoint and returns the step after it, or 0
when there is none. `save()` returns once the tensors are copied; hashing,
delta detection, compression and the write run on a background thread. Call
`flush()` when the checkpoint has to be on disk, which means `fsync`ed; loads
and `list_snapshots()` wait for pending saves on their own, and
`async_save=False` makes every save synchronous.

## Install

```bash
pip install moonclip
```

Wheels exist for Linux x86_64 (manylinux_2_28), CPython 3.11-3.14, because
that is where training runs. On any other platform `pip` finds no wheel and
the package has to be built, which needs a [Rust toolchain](https://rustup.rs/):

```bash
pip install git+https://github.com/JHNMACHINE/moonclip.git
```

`import moonclip` does not import torch: the Rust engine stores named tensors
and knows nothing about PyTorch, and `CheckpointManager`, the layer that knows
what a `state_dict` is, is loaded the first time it is used.

## What a checkpoint holds

- **Deltas.** An unchanged tensor costs no I/O. A changed one is stored as an
  XOR delta against the previous checkpoint when that is smaller than storing
  it whole (`delta_max_ratio`), and a full snapshot is taken periodically
  (`full_every_steps`). A background thread merges delta chains so that a load
  does not have to replay a long one.
- **Compression.** zstd, with large tensors split into frames compressed on
  every core.
- **Precision.** `save_dtype="bf16"` (or `fp16`) casts floating-point tensors
  before compression and casts them back on load; a mapping applies a
  different dtype per name pattern. `fp8` quantizes to one byte per element
  against a per-tensor scale, which is for archived copies rather than
  checkpoints you resume from. Tensors that already are float8 are stored
  bit-exact.
- **Integrity.** Every tensor carries an xxHash3-128 checksum that is checked
  on read.
- **Retention.** `max_full_snapshots` and `max_total_snapshots` bound what is
  kept on disk.

## S3

The local directory stays the primary copy, so a save is as fast as the disk;
a bucket (S3, R2, MinIO or any S3-compatible server) receives it in batches in
the background:

```python
mgr = CheckpointManager(
    "./checkpoints",
    s3_bucket="my-bucket",
    s3_endpoint="https://<account>.r2.cloudflarestorage.com",  # omit for AWS
    s3_path_style=True,
    sync_every_n_saves=5,
)
```

The keys come from the environment unless they are passed as `s3_access_key`
and `s3_secret_key`: `MOONCLIP_S3_ACCESS_KEY`, `MOONCLIP_S3_SECRET_KEY` and,
for temporary credentials, `MOONCLIP_S3_SESSION_TOKEN`; without them, the
`AWS_*` names every S3 tool reads. Each pair is taken whole, never half from
one set of names and half from the other. `restore_from_remote()` brings a
store with an empty local directory back from the bucket.

`moonclip.Bucket` is the same client for plain files: whole objects by path
(`get`, `put`, `list`), with no manifest and no tensors, for whatever has to
sit next to the checkpoints.

## More than one rank

Say which topology the manager is for. Under `torchrun`, `"auto"` takes it
from the launcher:

```python
mgr = CheckpointManager("./checkpoints", world_size="auto", rank="auto")
```

The ranks then share one store, through the `create_snapshot` / `save_rank` /
`finalize_snapshot` flow. If each rank writes to a directory of its own
instead, every one of them is single-rank and says so:

```python
mgr = CheckpointManager(f"./checkpoints/rank_{rank}", world_size=1, rank=0)
```

Leaving the topology unsaid under a multi-rank launcher is an error that names
both numbers. Adopting the launcher's variables silently made the same code
save under `python train.py` and refuse under `torchrun`, decided by nothing
the code could see. A single process needs no configuration.

## Environment

| | |
|---|---|
| `MOONCLIP_THREADS` | Size of Moonclip's own thread pool. Default: the cores divided by `LOCAL_WORLD_SIZE`, so that the ranks on one node share the machine instead of each claiming all of it. |
| `MOONCLIP_PROFILE=1` | A per-phase breakdown of each save on stderr, and a line whenever a save waited for the previous one. |
| `MOONCLIP_FSYNC=0` | Skips the `fsync` before a pack is renamed into place. Faster, but a power loss mid-save can leave a pack of the right length full of zeros that the manifest vouches for. |
| `MOONCLIP_NO_PINNED_STAGING=1` | Copies GPU tensors through ordinary memory instead of pinned buffers, for a box where host RAM is the constraint. |
| `MOONCLIP_LOG=debug` | Also logs the syncs a caller requests with `sync_prefix` when they succeed; errors are logged either way. |

## Moonclip and Ravex

Moonclip is the storage layer under [Ravex](https://github.com/JHNMACHINE/ravex),
and the line between them is that saving files is Moonclip's and the run is
Ravex's. Moonclip decides how a checkpoint becomes files and where they go,
bucket keys included. Ravex decides when to save and what, puts the result
back into a live training process, and coordinates one or more machines; it
carries no storage code of its own. To Moonclip a rank is only a number on a
snapshot. Moonclip works on its own as well, as in the examples above.

## Development

```bash
cargo test                                      # Rust tests, no Python needed
maturin develop --release && pytest tests/ -v   # Python and PyTorch tests
cargo clippy --all-targets -- -D warnings       # what CI gates on
```

`rust-toolchain.toml` pins the same compiler as CI, so that a red clippy here
means a red gate there. It is not the MSRV, which is `rust-version` in
`Cargo.toml`. The crate is deliberately not `cargo fmt`-clean: the gate is
clippy. What changed between versions is in the [CHANGELOG](CHANGELOG.md), and
runnable examples (a plain model, a transformer, DDP, S3) are in `examples/`.

## License

Apache-2.0 — [GPU Zero](https://gpuzero.dev)
