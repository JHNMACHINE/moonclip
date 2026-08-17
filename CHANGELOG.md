# Changelog

## 0.0.6 — unreleased

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
