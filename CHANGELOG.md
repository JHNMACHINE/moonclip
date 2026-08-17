# Changelog

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
