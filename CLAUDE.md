# CLAUDE.md

Moonclip: a checkpoint engine in Rust with Python bindings (PyO3). It stores
named tensors and knows nothing about any framework - keep it that way; the
PyTorch-specific parts belong to whoever uses it. The README's Development
section is the setup; this file is what it doesn't say.

## The gate is clippy, not fmt

- The repository is **not** `cargo fmt`-clean, and CI does not run fmt.
  Running it to tidy a diff reformats half the crate and buries the change.
  Format by hand, in the style around the line you touch.
- The gate is the exact line in `.github/workflows/checks.yml`:
  `cargo clippy --all-targets --features python -- -D warnings -A unknown-lints -A clippy::too_many_arguments`,
  then `cargo test --lib --tests`.
- The toolchain is pinned with its patch number in `rust-toolchain.toml` and
  in the workflows; change both together (the file explains why).

## S3 tests

`tests/s3_minio.rs` runs against whatever `MOONCLIP_S3_*` says. Start MinIO
with `bash tests/minio_up.sh` and **override every variable**, region
included: a shell that already has them set for a real bucket sends the tests
there, and a credentials failure reads exactly like a regression.

```sh
MOONCLIP_S3_ENDPOINT=http://127.0.0.1:9000 MOONCLIP_S3_BUCKET=moonclip-test \
MOONCLIP_S3_ACCESS_KEY=minioadmin MOONCLIP_S3_SECRET_KEY=minioadmin \
MOONCLIP_S3_REGION=us-east-1 cargo test --lib --tests
```

MinIO green is enough only when the change doesn't touch the S3 path: MinIO
accepts `PUT`s far larger than S3 does.

## Rules learned the hard way

- **Never block inside `crate::pool::install`** - on a queue, condvar or
  channel. The waiting rayon worker may hold the very piece of work that would
  release it: that was the intermittent pytest hang. `AsyncSaver::submit`
  asserts it is not on a rayon thread; keep it that way.
- **On Windows a rename over an open file fails.** The manifest is written by
  renaming a temp file over it, so a reader holding `manifest.json` can cost
  the checkpoint being written; the persist retries for 5 s, no longer.
  Callers should ask `describe()`, never read internal files.
- A test that only checks a reader's output does not see a lost checkpoint:
  check that every checkpoint is still in the store afterwards.
- `core.autocrlf` checkouts on Windows: the `.rs` sources are CRLF in the
  working tree. Edit by script with the line ending the file already has.
