# Changelog

## Unreleased

### Added

- **`describe()` reports each tensor's `hash_raw`.** The xxHash3-128 of the
  tensor's raw bytes, exactly as the manifest records it — the value skip
  detection already compares — and the same whether the tensor was stored
  whole, as a delta, skipped or aliased, since it hashes what the tensor holds.
  Also on `TensorDescription` in Rust.

  The caller is Ravex's audit trail (GPU-93), which fingerprints a checkpoint
  by its content. Its first version read `manifest.json` to get these hashes,
  and on Windows that cost checkpoints: Moonclip persists the manifest by
  renaming a new version over it, the rename fails while anyone holds the file
  open — even opened sharing delete — and the save being written was lost.
  `describe()` waits for the writer and reads under Moonclip's own lock, so the
  hashes are now available without anyone opening the file. The underlying
  fragility, that any reader of `manifest.json` on Windows can make a save
  fail, is its own issue (GPU-128) and is not fixed by this.

  Not cryptographic, and said so on the field: it detects corruption and an
  accidental swap, not a collision constructed on purpose.

### Fixed

- **On Windows, a reader holding `manifest.json` open no longer costs the
  save being written (GPU-128).** Moonclip persists the manifest — and every
  pack — by renaming a temp file over the old one, and Windows refuses a rename
  over a file someone has open: *Access denied*, even when the reader shared
  read, write and delete, which was measured with `CreateFileW` directly. The
  background writer reported "Background save failed" and the checkpoint was
  gone. Nothing a reader does differently avoids it, and the readers are not
  all ours: an antivirus scanning a file that just changed is one, a search
  indexer another.

  The rename is now retried while Windows gives that refusal, with a pause
  that doubles up to 100 ms, for at most five seconds. Past that the error is
  the one the rename gave, the temp file is removed, and the old file is
  untouched. A write that is not refused pays nothing.

  The trade, stated: a real permission problem also answers *Access denied*,
  and from here the two cannot be told apart, so such a write now fails after
  five seconds instead of at once. Elsewhere a rename over an open file
  succeeds and nothing changes. Covered by a test that holds the manifest open
  while a save goes through the whole pipeline and then reopens the store to
  find it — Windows-only, so Linux CI does not run it.

## 0.1.0 — 2026-08-31

Two of the three background threads could take the whole process down with
them, and now none of them can. A panic in the save pipeline used to poison
the manifest lock and turn every later call into a `PanicException` raised
somewhere unrelated; a panic in the merger or the syncer killed the thread
outright and nothing said so, which for the syncer meant a run believing it
had remote backups and finding out otherwise at resume time. Both are fixed,
and the death of a background thread is now something `flush()` reports.

Alongside them, a checkpoint can be asked what it holds without being read.
`describe()` answers off the manifest with no storage access at all, and
`load_tensors()` reads only the byte ranges the named tensors occupy. The
caller this was written for is a reshard, which has to know how long every old
shard is before it can plan a single slice and until now loaded each old
checkpoint in full to find out.

**Why 0.1.0 and not 0.0.10.** The read API needed a way to say "this tensor's
span, not the whole pack", and `Option<&[u8]>` cannot say it. Two public items
in `moonclip::tensor` changed signature as a result — see *Changed —
breaking*. Cargo treats every `0.0.z` as incompatible with every other, so
0.0.10 would have been technically sufficient; a minor bump says out loud what
the version number would otherwise only imply. **The Python API is unaffected
and nothing there needs changing.**

### Added

- **`describe()`, `describe_latest()` and `load_tensors()`**, on
  `MoonclipManager` and `CheckpointManager`: read what a checkpoint holds, and
  read parts of it, without reading all of it.

  There was nothing between `list_snapshots`, which reports totals, and
  `load`, which materializes everything. A caller that needed a tensor's
  *shape* had to load the tensor to find out — and the caller that pays for
  that is a reshard, which has to know how long every old shard is before it
  can plan a single slice. From N ranks to M that is N complete reads of data
  it then discards, tens of gigabytes on a real model.

  - `describe(snap_id)` returns the snapshot's metadata and, for every tensor
    this rank wrote, `name`, `shape`, `dtype`, `stored_dtype`, `storage`
    (`full` / `delta` / `skipped` / `alias`), `raw_size` and
    `compressed_size`. **It touches no storage at all** — everything it
    reports is already in the manifest in memory. `dtype` is what a load hands
    back and `stored_dtype` is what is on disk; they differ exactly when
    `save_dtype` cast the tensor, and conflating them would have a caller size
    an fp32 buffer for what it reads as bf16. A skipped tensor stores no bytes
    and still reports its shape, which is what makes this usable past the
    first checkpoint of a run.

  - `load_tensors(snap_id, names)` reads only the byte ranges those tensors
    occupy, rather than pulling the rank's pack and decompressing all of it.
    Deltas and skips resolve against a base it also declines to read whole —
    the case that matters, since a small unchanging tensor is stored `skipped`
    from step two onwards, and resolving that through a full base read would
    give back everything the ranged read saved. A name that is not in the
    snapshot is an error naming it, not a missing key.

- **`describe_state_dict()` and `TensorStub`**, beside `unflatten_state_dict`.
  Same reading of the same pickle, materializing nothing: every tensor comes
  back as a `TensorStub` with its `name`, `shape` and `dtype`, and everything
  that was never a tensor is there unchanged.

  That last part is what makes the pair complete. Moonclip does not model
  placements, layout tags or whatever else a caller wrapped around its
  tensors — but they were in the template all along, so
  `load_tensors(snap, ["model._metadata"])` plus `describe_state_dict` hands
  back the whole structure, with shapes, for a few kilobytes. The stubs carry
  the flat names, so having measured, a caller fetches exactly the tensors it
  turned out to want.

  A `<prefix>._blob` prefix is left out rather than half-described: a blob is
  one pickle holding the tensors themselves, and there is no describing it
  short of loading it.


### Changed — breaking

Rust callers only. Nothing in the Python API moves.

- **`moonclip::tensor::load_tensor` and `BaseEntry` changed shape**, to make a
  ranged read expressible:

  ```rust
  -pub type BaseEntry = (TensorEntry, CompressionAlgo, Option<Arc<Vec<u8>>>);
  +pub type BaseEntry = (TensorEntry, CompressionAlgo, OwnedPack);

  -pub fn load_tensor(…, pack_data: Option<&[u8]>, …)
  +pub fn load_tensor(…, pack: PackSource<'_>, …)
  ```

  `Option<&[u8]>` had exactly two meanings — the whole pack in memory, or
  nothing and fall back to per-file storage — and `load_tensors` needs a third:
  the pack is on storage and only one tensor's span is wanted. `PackSource`
  spells the three out; `OwnedPack` is the owning form the base-entry resolver
  hands back.

  Migration is mechanical. `None` becomes `PackSource::PerFile`, and
  `Some(&bytes)` becomes `PackSource::Whole(&bytes)`; a resolver returning
  `None` for the pack returns `OwnedPack::PerFile`, and one returning
  `Some(arc)` returns `OwnedPack::Whole(arc)`.

### Changed

- **CI builds on Rust 1.97.1, and `rust-toolchain.toml` pins the same version
  locally.** The two used to differ without saying so: the workflows pinned
  `rust:1.90-bookworm` while development ran 1.97, so a red `cargo clippy`
  locally did not mean anything was broken — it could be the version gap — and
  a green CI did not mean the next image bump would pass. The crate spent a
  while in exactly that state, red locally on four findings that were lints
  1.90 did not have. All four are fixed, and three of the four `-A` allowances
  in `checks.yml` are gone with them; `too_many_arguments` remains.

  The pin carries the patch number on both sides on purpose. Inside
  `rust:1.97.1-bookworm` the installed toolchain is named `1.97.1-…`, so a
  `channel = "1.97"` does not match it and rustup downloads a second copy of
  the same compiler on every job — 31s against 1.4s, measured in that
  container.

  **This does not change the MSRV.** `rust-version` stays 1.83 and a
  `rust-toolchain.toml` applies only inside the repository, so nothing changes
  for anyone depending on the crate. What it does mean is that CI no longer
  verifies the 1.83 claim.


### Fixed

- **A panic while saving no longer walls off the manager for the rest of the
  process.** The background saver already caught panics, and that fix was
  right as far as it went: it restored the *thread*, so a failed save stopped
  hanging the training loop. It did nothing about the poison flag a `Mutex`
  sets when a guard is dropped mid-unwind — and the save pipeline holds the
  manifest lock across retention, which persists through the caller's
  `StorageBackend`. That trait is public, so a panic inside that critical
  section is reachable without touching a line of Moonclip.

  What it cost: every later `save`, `load`, `flush` and `list_snapshots`
  raised `PanicException`, on the caller's thread, from somewhere unrelated to
  what had actually failed. One bad checkpoint took the run with it, and the
  message named the wrong place. Every access to the manifest now recovers the
  lock instead — the in-memory copy is re-readable from storage, so refusing
  to hand it out buys nothing that re-reading does not.

  The re-read is the second half, and it is not cosmetic: a save that panics
  in retention has already pushed its snapshot into the in-memory manifest, so
  recovering the lock alone would carry an entry that never reached storage
  into the next manifest written. The two write paths that do not already
  re-read now do so before trusting what is in memory.

  Unchanged: what a failed checkpoint costs is still that checkpoint. The
  failure is still reported by the next `save` or `flush`, with the panic's
  own message.

- **The merger and the remote syncer no longer die in silence.** Moonclip runs
  three background threads and only one of them — the async saver — was
  protected from panics. A panic on either of the other two killed the thread
  outright: `for cmd in rx` ended, the receiver dropped, and every later
  `notify` failed into a `let _ =`. Nothing was poisoned and nothing was
  raised, because the worker held the receiving end and the callers hold the
  sending one, so the calls went on succeeding at nothing for the rest of the
  process.

  What that cost, silently:

  - **Merger gone** — delta chains stop being consolidated. Loads degrade
    without limit and no message says why.
  - **Syncer gone** — nothing reaches S3 or R2 again. For a durability feature
    this is the worst failure available: the run believes it has remote
    backups, and finds out it does not when the machine dies and a resume is
    attempted.

  Both loops now catch panics per command, the same treatment and for the same
  reason as the saver: a panic becomes what a storage failure already was — an
  error, with the loop still running. A forced merge and `sync_now`, which have
  a caller waiting on a reply, get that error returned rather than logged.

  Catching is only half. A thread can still go for a reason no `catch_unwind`
  reaches, so **`flush()` now reports a background thread that is gone**, and
  reports it every time rather than once: it is not an event that happened but
  a state that will not improve, since nothing restarts those threads. The
  error names which thread and what has stopped, and says the checkpoints on
  local disk are still being written and still readable.

- **`merge_now()` no longer reports success when there was no thread to do the
  merge.** A failed send to a departed merger returned `Ok(())`, on the grounds
  that shutdown could be racing it — but `shutdown` takes `&mut self` and
  cannot be. `save_final` is the caller this mattered for: it merges and then
  uploads, so a fold reported as done that never happened is how a run's final
  checkpoint reaches the bucket unmerged.


## 0.0.9 — 2026-08-30

Float8, which turned out to be two features wearing one name. A tensor that
arrives already float8 could not be stored at all — `save_tensors` raised
`ValueError: unsupported dtype torch.float8_e4m3fn`, so a run doing FSDP2 or
torchao float8 training had no way to checkpoint but converting every
parameter back to bf16 by hand. That is a bug, and the fix is lossless.
`save_dtype="fp8"` is the other half, and it is a trade: a quarter of the size
for four significant bits.

Alongside it, two pieces of the Python surface stop being surprising:
`CheckpointManager` no longer works out its own topology from the launcher's
environment variables, and `flatten_state_dict` finally has a public inverse.
The first is a behaviour change and the migration is one line — see below.

### Added

- **`unflatten_state_dict`**, the inverse of `flatten_state_dict` and public
  for the same reason it is. A caller that flattened its own state and handed
  the result to `save_tensors` had no supported way back: the only
  implementation lived inside `CheckpointManager._apply_loaded`, welded to the
  step that calls `load_state_dict` on live objects. Anything holding the
  bytes but no objects — a converter, a checkpoint inspector, Ravex — had to
  either reimplement it or accept the applying it did not want. It reads both
  shapes that get written, `<prefix>._blob` and `<prefix>._metadata`, and
  `_apply_loaded` is now that function plus the loop that applies.


- **Float8 tensors can be stored.** `torch.float8_e4m3fn`, `torch.float8_e5m2`
  and the four variants this torch exposes (`e4m3fnuz`, `e5m2fnuz`, `e8m0fnu`,
  `e4m3b11fnuz`) are accepted on the direct tensor path and come back with the
  identical bits. Nothing else was needed to store them — Moonclip reads the
  tensor's own buffer, and the buffer is bytes — only for `get_element_size`
  to know they are one byte wide. It did not, and the failure was a hard
  `ValueError` in the middle of a training run rather than a degraded save.
  No configuration: a float8 tensor is stored as float8 whatever `save_dtype`
  says, because that setting names what fp32 is cast *down* to and applying it
  here would have cast it back *up*, doubling the size of the one dtype chosen
  to make things smaller.
- **`save_dtype="fp8"` and `save_dtype="fp8_e5m2"`.** fp32, fp16, bf16 and
  fp64 quantize to one byte per element against a per-tensor scale, recorded
  as `quant_scale` on the manifest entry and multiplied back in on load, so
  training gets its original dtype back the same way `save_dtype="bf16"`
  already worked. Bare `"fp8"` means e4m3, matching torchao and Transformer
  Engine.

  **This one is lossy, and more so than it looks.** e4m3 keeps four
  significant bits, which is a few percent of relative error on every element
  — twenty times what bf16 costs. It is a reasonable trade for an archived
  copy or for analysis, and a poor one for a checkpoint a run will resume
  from; optimizer moments in particular do not survive it. The manifest
  records the scale per tensor, so a quantized checkpoint says what it is
  rather than looking like a full-precision one that went wrong.

  The conversion agrees with torch bit for bit. Checked on torch 2.12 across
  all 256 patterns of both formats and roughly 22000 values chosen to sit on
  the rounding ties, the subnormals and the specials: every in-range value
  encodes to the identical byte, and the decode tables match
  `.view(torch.float8_*).float()` exactly. The one deliberate difference is at
  the top of the range, where a value pushed past `max_finite` by rounding
  saturates instead of becoming a NaN — turning the largest weight of every
  tensor into a NaN would be a poor price for matching torch there. A real
  infinity still becomes a NaN in e4m3fn, which has no infinity to put it in,
  and stays an infinity in e5m2, which does.

  A tensor whose finite values are all zero, or so small that the scale would
  not be a normal float, is stored against a scale of 1.0 rather than dividing
  by nothing. Non-finite elements are left out of the scale entirely: with one
  infinity counted, the scale is infinite and every other value in the tensor
  quantizes to zero, so a single diverged element would erase the tensor
  around it. The infinity itself still survives as a NaN, so nothing is
  hidden.

- **`last_queue_wait()`**, on `MoonclipManager` and `CheckpointManager`: how
  long the last save spent waiting for the previous one to drain, in seconds.

  One save is allowed in flight, so a writer that has not finished stops the
  next `save_tensors` before any of its own work begins. From outside Moonclip
  that is indistinguishable from the shadow copy having been slow — one call,
  one duration — and the two have nothing to do with each other: the copy is
  memory bandwidth and grows with the model, the wait is backpressure and
  grows with the checkpoint cadence and the speed of the storage. Summed, they
  point at neither. Ravex reports them as separate phases on the strength of
  this.

  The number already existed and `MOONCLIP_PROFILE=1` already printed it. What
  it could not do is reach a program, and a caller that has to turn on a
  profiler and read stderr does not have the number at the moment the question
  comes up, which is mid-run on a machine being billed by the hour.

  Read it on the thread that just called `save_tensors` and it describes that
  call. Zero when nothing was in flight, and always zero with `async_save`
  off, where the write happens on the calling thread and is the write rather
  than a queue in front of it.

- **`save_dtype` may now be chosen per tensor**, as an ordered dict of glob
  pattern to dtype:

  ```python
  CheckpointManager(save_dtype={"optimizer/*": "bf16"})
  ```

  `*` matches any run of characters, everything else is literal, and **the
  first matching rule wins** — so an exception is written by putting it first,
  `{"model/*": "none", "*": "bf16"}`. A tensor matching no rule is stored as
  it arrived. A plain string is unchanged in meaning and still casts
  everything, so no existing configuration moves.

  One dtype for a whole snapshot was the wrong shape, and the measurement says
  how wrong. On a 1.5B model under FSDP2, per rank: the weights are about a
  third of the bytes and delta well, −70%; `exp_avg` and `exp_avg_sq` are the
  other two thirds and delta essentially not at all, −1.1% and −4.0%. Two
  consecutive Adam moments differ across nearly every mantissa bit — with
  β₁ = 0.9 a tenth of the value is replaced each step — so there is nothing in
  the XOR for a compressor to find. **85% of the bytes written are the part
  that does not compress**, and it is also the part that tolerates the least
  precision: `exp_avg_sq` enters Adam through `sqrt(v)`, which halves the
  relative error, which is why 8-bit optimizers are ordinary practice. Casting
  only the moments to bf16 halves 85% of the volume and leaves the model
  alone.

  Nothing in Moonclip knows what an optimizer is, and that is the design: the
  structure separating the components already exists in the names the caller
  passes, so a second channel to carry it could only add a way for the two to
  disagree. `model/` and `optimizer/` are what Moonclip's own PyTorch layer
  writes; a caller that names its tensors differently writes its own patterns.

  A pattern matching none of a rank's tensors is reported on stderr, once. It
  is this feature's quiet failure — `{"weight": "bf16"}` reads as if it did
  something and matches nothing, because names arrive as `model/weight` — and
  it is the same shape as the `save_dtype` typo that has always been refused.
  A complaint rather than a refusal, because on a sharded save a pattern
  matching nothing *on this rank* is legitimate.

- **`save_dtype="fp64"`.** A widening target, and the only one: it recovers no
  precision the source did not have, it writes twice the bytes. It is here for
  reference runs and numerical bisection, which want a whole snapshot in one
  width to diff against a reference implementation and otherwise have to make
  a second full copy of the state outside Moonclip. A tensor that is already
  float64 is untouched, as before.

### Changed

- **Python 3.9 and 3.10 are no longer supported.** The floor is 3.11, and no
  cp39 or cp310 wheel is published.

  The list is [the Python devguide's](https://devguide.python.org/versions/)
  rather than a judgement of our own about who is still out there. 3.9 reached
  end-of-life on 31 October 2025, ten months before this release. 3.10 is
  still in security-only maintenance, but it reaches end-of-life in **October
  2026** — two months from now — so stopping there would have meant doing this
  again almost immediately. 3.11 runs to October 2027.

  Three lists move together here and all three were changed: the test matrix
  in `ci.yml`, the classifiers in `pyproject.toml`, and the wheel build list
  in `release.yml`. The last is the one that matters at tag time — it was
  still building `cp39` and `cp310`, so a tag would have published wheels for
  two interpreters the package no longer claims to support.

  3.15 is deliberately absent rather than watched. Moonclip ships a compiled
  wheel per interpreter, so a version it tests is a version it can publish,
  and there is no cp315 manylinux to build against yet.

- **`keep_base_in_memory` is now decided per tensor rather than per
  snapshot.** It used to be switched off entirely whenever `save_dtype` cast
  anything, because what a later delta is computed against is the bytes on
  disk and for a cast tensor those are not the bytes that arrived. That is
  still true, and the test is now `original_dtype`, which is set exactly when
  the stored bytes and the incoming bytes differ. So the ordinary
  configuration — moments to bf16, weights untouched — keeps the weights in
  memory, which is the third of the state where the delta actually pays, and
  where the retained base removes 36% of the write path's CPU.

  Nothing downstream needed changing for a partial base: `BaseCache` looks
  each tensor up by name and falls back to storage on a miss.

- **`shuffle::element_size` knows `complex128`, `complex32` and the six
  float8 names.** A dtype missing from that table falls back to one byte,
  which turns the byte-plane filter into a no-op — safe, never a corruption,
  and completely silent. `complex128` was in exactly that position: stored
  correctly at 16 bytes an element and shuffled as if it were bytes, so its
  deltas compressed worse than they had to with nothing anywhere to say so.
  Extending the table does not affect checkpoints already written; the load
  path unshuffles only when the entry's `shuffled` flag is set, and a delta
  written under the old width did not set it.

- **`torch.uint32`, `torch.uint64` and `torch.complex32` are accepted** on the
  direct tensor path, which was refusing them with a hard `ValueError`
  mid-run. The quantized dtypes (`qint8`, `quint8`, `qint32`, `quint4x2`,
  `quint2x4`) are still refused, and deliberately: their scale and zero-point
  are properties of the tensor rather than of the buffer, and a `TensorEntry`
  has nowhere to keep them, so storing the bytes would produce a checkpoint
  that loads without complaint and holds integers nobody can turn back into
  numbers.


- **`CheckpointManager` no longer decides its own topology.** It used to read
  `RANK`/`WORLD_SIZE` from the environment whenever it was not told, and adopt
  them. That looked like a convenience and behaved like a trap, because
  adopting a detected world size can only ever take capability away: `save()`
  refuses outright when `world_size != 1`, directing the caller to the
  explicit `create_snapshot`/`save_rank`/`finalize_snapshot` flow. So the same
  code saved fine under `python train.py` and raised under `torchrun`, decided
  by nothing the code could see.

  Ravex is where that bill came due. It builds a manager per rank against a
  directory per rank — every store single-rank by construction — caught the
  refusal in the `except` around backend construction, and fell back to
  `torch.save` with a single log line. Every distributed run quietly lost
  Moonclip checkpointing and kept training.

  The environment is now consulted only to decide whether the question is
  ambiguous. **A single process is unaffected**: nothing to disagree with is
  still `(1, 0)`, no configuration, exactly as before. Under a launcher saying
  otherwise, with nothing stated, construction raises and the message names
  both detected numbers and all three ways forward. `world_size="auto",
  rank="auto"` is the old behaviour, still available and now asked for by
  name.

  Migration is one line, and there is no silent path: any run affected gets an
  error telling it what to write.

- `_env.py` claimed "No PyTorch dependency" while importing
  `torch.distributed` in its first detection step. The import was always
  optional and local, and the module only ever loads under `moonclip.pytorch`
  — which is the layer allowed to know about torch — so the code was fine and
  the sentence was not.


- `TensorEntry` gained `quant_scale`, optional and defaulted. Manifests
  written before this deserialize unchanged — they hold no float8 entries, so
  the absent value is the correct one rather than a missing one. The format
  version is unmoved at 2.


## 0.0.8 — 2026-08-21

S3 stopped being a one-way street. A gathered checkpoint of a 1B model with
Adam is around 11 GiB, and every write was a single `PUT` — which S3 refuses
over 5 GiB. What did arrive could not be read back either, because the remote
support only ever pushed. Both were found on a six-node bench rather than in
the tests, which run against MinIO: it accepts what S3 does not, and that is
the whole reason neither showed up in 0.0.7.

### Added

- **Very large objects reach S3.** Every write was a single `PUT`, and S3
  refuses one over 5 GiB — the size an ordinary gathered checkpoint reaches on
  its own, since a 1B model with Adam is around 11 GiB. Past a threshold a
  write now becomes a multipart upload: create, one request per part, complete,
  and — the part that is easy to leave out — `AbortMultipartUpload` on every
  way out other than success, because parts belonging to an upload that was
  neither completed nor aborted stay in the bucket, invisible to a listing and
  still billed. The part size grows with the object so the count stays under
  S3's ceiling of 10000. Not caught earlier because the integration tests run
  against MinIO, which accepts single `PUT`s far larger than S3 does.
- **A store can be pulled back from the remote.** Remote support was push-only:
  `sync_now()` sent local to remote and nothing read the other way, so the step
  to resume from came from the *local* manifest and a bucket was a backup you
  could not resume from. Measured on a six-node bench: a node whose disk had
  been replaced started from scratch with its own data sitting in the bucket,
  and took every other rank with it. `restore_from_remote` fills an empty store
  from the remote, and brings back the whole store rather than only its
  snapshots — a caller's sidecar files come back with it.
- **The multipart threshold is configurable**, `S3Config::single_put_limit`,
  defaulting to `SINGLE_PUT_LIMIT` as before. It exists so the choice can be
  tested: at the default, the only way to watch `put` take the multipart path
  is to actually move four gigabytes, which on an ordinary uplink is over an
  hour. That the ceiling is real is the service's business and documented;
  that we dispatch on it is ours, and now a test says so for a few megabytes.
- **`list_multipart_uploads` and `abort_multipart_upload`.** Parts belonging
  to an unfinished upload do not appear in `ListObjectsV2`, so `list` cannot
  see them, retention never reclaims them, and the bill counts them the whole
  time. `put_multipart` has always aborted on every way out other than
  success — but nothing could check that it did, and an upload interrupted by
  a machine going away was nobody's to clean up. Now both are possible.
- **`etag`**, which reports what the service says about an object. The cheap
  way to tell how one was written: a multipart upload's ETag ends in
  `-<part count>` and a single `PUT`'s does not, on S3 and R2 alike, though
  the two compute the hash differently.

### Fixed — data loss

- **A merge could unlink a base a rank was still writing against.**
  `do_full_merge` correctly keeps unfinalized snapshots out of the fold — they
  belong to other ranks to finish — and then unlinked the base regardless. What
  survived was a delta naming a base no longer on disk: `load` answered
  NotFound and startup recovery discarded a checkpoint that was whole. The
  window is real between `create_snapshot` and the first rank reaching
  `save_rank_in_pool`, since no pin exists yet — and pins would not close it
  anyway, being per-process while co-located ranks are not.
- **The base pin was released too early.** It now covers the push, or a merge
  is free to fold the base away between the last byte of a pack and the
  snapshot appearing in the manifest. It is still released before retention,
  because a pin held there is one the same thread would wait on itself.

### Fixed

- **A forced merge could go quiet for twenty minutes.** The claim retries and
  the unlink each had their own 300-second budget, taken back to back, and this
  path also runs at process exit — where a scheduler's grace period is already
  counting. They are now **one** budget covering both, with the unlink coming
  out of whatever the claim left.
- **A forced sync could leave the bucket holding a checkpoint that does not
  load.** The manifest names snapshots by id, and `sync_now()` handed the whole
  store to one walk, so the backend's listing order decided what went up first.
  A manifest that arrived before the packs it named made the bucket present
  itself as a checkpoint and fail on the first read — seen on the bench as
  `Checkpoint not found: snapshots/cdc000df-.../rank_0.pack`. The manifest now
  goes last, always, which makes an interrupted sync leave the remote *behind*
  rather than *inconsistent*: an older checkpoint costs some progress, a
  manifest naming absent data costs the run. The periodic path already had this
  order; the forced one did not.

## 0.0.7 — 2026-08-18

Retention and merging stopped being able to pull a snapshot out from under a
reader, and the save path stopped being able to wedge the interpreter. Both are
things that had to go wrong at the wrong moment to be seen at all, which is why
neither showed up in 0.0.6.

### Fixed — data loss

- **A merge could unlink a snapshot somebody was still reading.** `InFlight`
  tracked what was being read, but nothing made the unlink *wait* for it: the
  window between deciding to fold a snapshot and removing its files was open,
  and a load that had already resolved its paths would find them gone.
  `unlink_snapshots` now waits for readers to leave before it removes anything,
  and `InFlight` grew the claim and wait primitives to make that expressible
  rather than a sleep. A reader that arrives after the claim is made does not
  get in.
- **Directories were created without being recorded or synced.** A crash between
  creating a directory and writing into it left a path that existed and a parent
  that did not know about it — recoverable on most filesystems, not on all of
  them, and never detectably. `create_dirs_recording_new` tracks what it made
  and fsyncs the parents, so the tree a snapshot lives in is as durable as the
  snapshot.
- **Retention unlinked while holding the lock.** `apply_retention` now returns
  the evicted snapshots and lets the caller remove them after releasing the
  lock, which is both shorter to hold and the only order in which the wait above
  can work.

### Fixed — deadlock

- **`save` and `save_rank` could wedge the interpreter, roughly one run in
  three.** `MoonclipManager::save_tensors` wrapped both the shadow copy *and*
  `inner.save` in `crate::pool::install`. `inner.save` reaches
  `AsyncSaver::submit`, which **blocks** until the previous save drains — so the
  wait happened on one of our own rayon workers. That worker is one the pool has
  lost, and it can be holding a stolen piece of the `process_tensors_parallel`
  whose completion is exactly what would release it: the save waits for work
  only the waiter could finish.

  The copy stays in the pool; the blocking call moved out. `AsyncSaver::submit`
  now carries a `debug_assert!(rayon::current_thread_index().is_none())` so the
  pattern cannot come back unnoticed.

  Measured before the fix: 10 of 30 isolated repros wedged, 3 of 6 full suites.
  After, on a 128-core box — a wider pool than the one where it was found, so
  more workers and more ways to interleave — 0 of 60 and 0 of 10.

### Changed

- **`PendingDeletes` is only built when a remote is configured.** Without one
  there was nothing to eventually delete remotely, and the structure grew for
  the life of the process anyway.
- **`MergeOutcome`** replaces the boolean a merge used to return, so "nothing to
  do", "folded" and "declined because a reader is inside" stop being the same
  answer to the caller.

### Notes

Nothing here changes the pack format: a 0.0.6 checkpoint reads unchanged.

The Rust suite goes from 162 to 167 tests, the new ones covering the reader/merge
interaction and retention with no remote configured.


## 0.0.6 — 2026-08-17

Everything an external review of the 0.0.4 artifacts found, plus the four things
that turned up while fixing them. Each entry below has a regression test that
fails on 0.0.4.

**There is no 0.0.5.** It was prepared, merged and never published: the review
that produced the rest of this list arrived first, and shipping the small half
on its own would have meant a release whose headline fix was a warning about a
bug that is now fixed. Its entries are folded in below.

Nothing here changes the pack format. A 0.0.4 checkpoint reads unchanged, and
one written by 0.0.6 is readable by 0.0.4 — with the same bugs.

### Fixed — data loss

- **Merging dropped tensors the model gained and resurrected ones it lost.**
  `do_full_merge` rebuilt the merged snapshot from the **base** snapshot's
  tensor list, so the model's shape at the base decided what survived the fold.
  An adapter added mid-run, a growing head, a pruned layer: the merged
  checkpoint loaded without complaint, holding the wrong set of parameters. It
  walks the newest delta now — that list *is* the state dict at the step the
  merge stands in for, and it is what a load of that snapshot would return.
  Two more defects came with it: intermediate deltas were replayed in sequence,
  which corrupted any tensor that changed and changed back (the newest delta
  says "identical to the base" and the replay left the intermediate value), and
  `original_dtype` was dropped, so merging a `save_dtype="bf16"` checkpoint
  quietly stopped uncasting it on load. Merging is off by default
  (`merge_stride=0`), so this only ever reached people who turned it on.
- **A merge could delete the base a save was reading.** `save_sync` takes its
  base from the manifest and releases the lock before reading it; the merger
  holds the lock only while deciding what to fold. Overlapping those windows
  failed the save outright (`Checkpoint not found: snapshots/…/rank_0.pack`)
  or, worse, published a delta pointing at a base the merge had already
  removed — broken on the next `load_latest`, and discarded by startup recovery
  as an orphan. There is now a registry of what is being read
  (`crate::inflight`): a merge does not consume a snapshot somebody is inside,
  and a save does not pick up a base a merge has claimed. Neither side waits
  for the other — the merge stands down and runs on the next notify, the save
  writes a full snapshot for that one step.
- **Multi-rank saves lost each other's shards.** Every rank reloaded
  `manifest.json`, inserted its own entry and wrote the whole file back — a
  read-modify-write with no lock, across processes. The Python layer hid it by
  serialising the ranks behind a barrier, which made the README's "each rank
  saves its own shard independently" false and a multi-GPU save sequential. A
  rank now writes only its own pack; `finalize_snapshot` assembles the snapshot
  from the descriptors in those packs. One writer, no lock, and the shards go
  in parallel.

### Fixed — checkpoints that could not be read or recovered

- **Tied weights became unreadable when the tensor order changed.**
  Deduplication stores the first of a set of identical tensors and records the
  rest as aliases of it. Which one comes first is the state dict's iteration
  order, and wrapping a model differently is enough to swap them: the tensor
  that used to be the alias then carries the bytes, is unchanged since the
  base, and is written `Skipped` — and resolving that skip landed on the base's
  alias entry, which holds no bytes. `Tensor 'head' is an alias of 'emb' and
  must be resolved after the snapshot's other tensors`, on a checkpoint that
  was completely intact. The base lookup follows the reference now.
- **A checkpoint where nothing changed could not be recovered.** With every
  tensor skipped there are no bytes to write, so no pack was written — and the
  pack is what carries the descriptor that startup recovery rebuilds a lost
  snapshot from. A process killed between the save and the manifest write left
  that checkpoint unrecoverable, in the case that is ordinary rather than
  exotic: fine-tuning, where whole stretches of the model do not move. A pack
  holding just the descriptor is always written now, a few hundred bytes.
- **No `fsync` anywhere in the write path.** The temp-file-then-rename was
  atomic for readers and said nothing about power: the rename could reach the
  disk while the data was still in the page cache, leaving a pack of exactly
  the right length full of zeros, vouched for by the manifest. `flush()`
  documented "durably on disk" and did not deliver it. The data is now synced
  before the rename and the directory after it. `MOONCLIP_FSYNC=0` restores the
  old behaviour where the checkpoint is not the thing being protected.

### Fixed — quietly wrong

- **The fp32→bf16 cast could turn a NaN into +Inf.** bf16 keeps the top 7
  mantissa bits, so a NaN whose payload sits below bit 16 — `0x7F800001`, which
  is what comparisons against a corrupted tensor tend to produce — reached the
  truncation with a zero mantissa under an all-ones exponent, and that is
  infinity. The quiet NaN `0x7FC00000` was never affected, which is why no test
  caught it. NaN is mapped to a NaN explicitly now, sign and the surviving
  payload bits kept. It only ever affected `save_dtype="bf16"`, and what it
  cost was the evidence: a diverged run saved looking finite.
- **`max_rollback_snapshots` was never read.** Defined, defaulted, documented
  and passed in from Python, and no code consulted it — so every snapshot that
  ever landed on `rollback_interval_steps` kept its exemption from retention
  forever and the store grew without bound. Only the newest N are protected
  now; past that they become ordinary snapshots again and retention prunes them
  in its own order. `0` disables rollback protection, as it already did for the
  interval.
- **`save_final()` uploaded before the merge finished.** `merge_now()` queued a
  merge on another thread and returned; `sync_now()` waited only for the save
  thread, so the final merged snapshot — the one every earlier checkpoint had
  just been folded into — could still be being written when the upload listed
  the files. `merge_now()` blocks until the merge is done and reports its
  failure.
- **Deletions never reached the remote.** Retention and the merger deleted
  locally and stopped there, so `keep_last` bounded the SSD while the bucket
  kept every pack the run ever wrote. Deleted keys are now queued and removed
  on the syncer's next pass. Only keys this process deleted: listing the remote
  and removing whatever is not local would erase a backup the first time a
  fresh machine pointed at it.
- **A typo in `save_dtype` was silently ignored.** `DType::from_str` mapped
  anything it did not recognise to "do not cast", so `save_dtype="bfloat"`
  produced full-precision checkpoints and said nothing. It is `DType::parse`
  now and returns an error.
- **`save_dtype` skipped float64, float16 and bfloat16 sources.**
  `is_castable_float` declared them castable and `cast_tensor` had no arm for
  them, so those tensors were stored untouched at the size the setting was
  chosen to avoid. Every float dtype now converts to every other, through fp32.
- **An unknown dtype on load became float32.** `_DTYPE_MAP.get(dtype,
  torch.float32)` reinterpreted the bytes under a dtype that was not theirs
  whenever the element sizes divided — right shape, plausible numbers, no
  relationship to what was saved. It raises now, and `complex64`/`complex128`,
  which the byte path had always been able to *save*, can finally be read back.

### Fixed — S3 backend

- 404 was recognised by looking for the substring `"404"` in an error message,
  which also matched any object whose key contained those digits. The status is
  carried in the error type now.
- No retries at all: one 503, one reset connection or one DNS hiccup failed a
  checkpoint upload. Three attempts with backoff, for 5xx, 429 and transport
  errors only — a bad signature or a missing bucket still fails immediately.
- `ListObjectsV2` keys were used without unescaping XML entities, so a key
  containing `&` came back as `a&amp;b`. Every later read or delete used a key
  that does not exist: the object was unreachable and still billed.
- `get_range` fell through to the trait default, which downloads the whole
  object and slices it — the entire checkpoint over the network, once per
  tensor. It sends a `Range` header now.

### Fixed — packaging

- **`requirements.txt` was UTF-16LE**, so `pip install -r requirements.txt`
  failed to parse it. UTF-8 now.
- **Wheels carried `__pycache__`.** Every 0.0.4 wheel shipped cp314 bytecode
  from a working tree where the tests had already run: harmless, and six wheels
  for six interpreters holding one interpreter's `.pyc` files.
- **The `Development Status` classifier claimed `5 - Production/Stable`** on a
  0.0.x release. It is `4 - Beta`.

### Added

- **A source distribution on PyPI.** Wheels are still Linux x86_64 only, so
  until now `pip install moonclip` on macOS, Windows or aarch64 answered "no
  matching distribution found" — which reads as "this package does not exist"
  rather than "there is no wheel for your platform". pip can build from the
  sdist instead, given a Rust toolchain.

### Changed

- `merge_now()` blocks and returns a result instead of returning immediately.
- `DType::from_str` is `DType::parse` and returns `Result<DType>`.
- `RemoteSyncer::new` takes a `PendingDeletes` queue.
- `DeltaMerger::new` is crate-internal: a merger has to share the coordinator's
  in-flight registry or it deletes snapshots out from under readers, and there
  is no way to hand one in from outside.
- `is_castable_float` includes float64 again — this time because the cast
  exists.

## 0.0.4 — 2026-08-17

First release on PyPI (`pip install moonclip`) and crates.io. Until now the only
way in was `pip install git+…`, so the entries below are written for whoever did
that — and the first one is a reason to move.

### Fixed

- **`keep_last` erased the history instead of pruning it.** The total cap could
  only remove a whole snapshot *group* — a full plus every delta hanging off it.
  Since every delta is computed against the last full, a normal run has one
  group for thousands of steps, so any `keep_last` smaller than that group did
  not prune: it emptied the store. Found on real hardware with `keep_last: 2`,
  three checkpoints delivered and nothing left after shutdown. The cap now drops
  the oldest *delta*, one at a time; a full survives as long as some delta needs
  it as a base, which means the oldest surviving step can be older than the cap
  suggests. That is the honest answer instead of an unreadable checkpoint.
- **Merged snapshots were not recoverable.** `do_full_merge` wrote individual
  files with no pack descriptor, so a snapshot produced by the merger could not
  be recovered at startup — and the merge is exactly the moment when a run's
  whole history is concentrated into one snapshot.
- **Loading a multi-rank snapshot could read another rank's shard.** The rank was
  inferred from the tensor name, and under sharding every rank uses the same
  names. It is passed explicitly now.
- **`print_stats()` could take down the process.** It drew a rule with U+2500,
  which raises `UnicodeEncodeError` on a cp1252 console: a reporting call
  interrupted training that was checkpointing perfectly well.
- **`merge_stride` did not do what its name says**, and the chain-depth limit was
  checked after the stride, so a stride larger than `max_chain_depth` never hit
  the ceiling.
- Type stubs were out of sync with the code (`flush` missing from
  `CheckpointManager`, `keep_base_in_memory` undeclared) — which counts double
  now that the package ships `py.typed`.

### Added

- **A private thread pool**, sized `cores / LOCAL_WORLD_SIZE`, with
  `MOONCLIP_THREADS` overriding it. Previously every parallel pass used rayon's
  global pool at one thread per core, so eight ranks on a 128-core node ran 128
  threads each and competed with the application's own rayon. This is hygiene
  rather than speed: capping the threads to the per-process quota measured
  *slightly slower* on 8× RTX 5060 Ti (11.6 s against 10.6 s per handoff), and
  the time turned out not to be inside this crate at all.
- `MOONCLIP_PROFILE=1` now also reports how long a save waited for the previous
  one to drain. Moonclip keeps one write in flight, so from outside that wait is
  indistinguishable from a slow copy.
- `py.typed`, so installed type checkers honour the shipped stubs.

### Known limits

- **Wheels are Linux x86_64 (manylinux_2_28) only**, CPython 3.9-3.14. Other
  platforms have to build from source, which needs a Rust toolchain.
- **The speedup is 1.6-3.0×, not the 10-50× an early estimate suggested.** That
  number was projected before anything was measured. On dense pre-training with
  Adam every parameter changes at every step, so there is nothing to skip: what
  is left is 1.6-3.0× faster than `torch.save` and 1.9× smaller. The 6-10× size
  reduction remains plausible on fine-tuning, LoRA and adapters, where most
  tensors are identical between snapshots — it has not been measured yet.
- Pack files written before 0.0.3 have no magic bytes: they stay readable through
  the manifest, but cannot be recovered without it.
