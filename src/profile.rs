//! Opt-in phase timing for the save path.
//!
//! The save pipeline is a handful of full passes over several gigabytes —
//! hashing, decompressing the base, XOR-ing, compressing — spread across a
//! rayon pool. When the aggregate throughput comes out lower than the parts
//! suggest it should, there is no way to tell which pass is responsible by
//! looking at the wall clock, and guessing has a poor record here: the last
//! two performance investigations both overturned a confident hypothesis.
//!
//! Set `MOONCLIP_PROFILE=1` to get a per-phase breakdown on stderr after each
//! save. Disabled, `time()` is one relaxed atomic load and a direct call.
//!
//! The totals are CPU time, not wall time: phases run in parallel, so they sum
//! to more than the save took. That is the point — it says which pass the
//! cores actually spent their time in.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
pub enum Phase {
    /// Hashing every tensor to plan intra-snapshot deduplication.
    DedupHash,
    /// Hashing a tensor again inside `process_tensor`.
    RawHash,
    /// Reading and decompressing a small window of the base tensor.
    SamplePrefix,
    /// Deciding delta-vs-full on that window (two zstd passes).
    PaysOff,
    /// Reading and decompressing the whole base tensor.
    BaseDecompress,
    /// XOR-ing base against target.
    XorDelta,
    /// Transposing the delta into byte planes before compressing it.
    Shuffle,
    /// Compressing the XOR delta.
    CompressDelta,
    /// Compressing a tensor stored in full.
    CompressFull,
    /// Writing the pack file.
    PackWrite,
}

const NAMES: [&str; 10] = [
    "dedup hash",
    "raw hash",
    "sample prefix",
    "pays_off probe",
    "base decompress",
    "xor delta",
    "shuffle",
    "compress delta",
    "compress full",
    "pack write",
];

static NANOS: [AtomicU64; NAMES.len()] = [const { AtomicU64::new(0) }; NAMES.len()];
static CALLS: [AtomicU64; NAMES.len()] = [const { AtomicU64::new(0) }; NAMES.len()];
static HEADER_SHOWN: AtomicBool = AtomicBool::new(false);

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("MOONCLIP_PROFILE").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

/// Run `f`, attributing its elapsed time to `phase`.
pub fn time<T>(phase: Phase, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let started = Instant::now();
    let out = f();
    let i = phase as usize;
    NANOS[i].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    CALLS[i].fetch_add(1, Ordering::Relaxed);
    out
}

/// Report how long a save waited for the previous one to drain.
///
/// Deliberately not a phase. The phases above are CPU time spent inside the
/// save pipeline; this is wall time the *caller's* thread spent stopped before
/// the pipeline began, and adding it to that column would make neither number
/// mean anything.
///
/// It is worth its own line because from outside Moonclip this wait is
/// indistinguishable from a slow copy: `save` allows one write in flight, so a
/// writer that has not drained shows up as the next `save` taking longer.
/// Per-rank checkpointing on a shared node is exactly the shape that invites it
/// — N processes writing at once, each still holding the training loop.
pub fn note_queue_wait(waited: Duration) {
    if !enabled() || waited < Duration::from_millis(1) {
        return;
    }
    eprintln!(
        "[moonclip profile] save waited {:.0} ms for the previous one to drain \
         (caller blocked, before any work started)",
        waited.as_secs_f64() * 1000.0
    );
}

/// Print the breakdown accumulated since the last report, and reset it.
pub fn report(label: &str, wall: Duration, bytes: u64) {
    if !enabled() {
        return;
    }

    if !HEADER_SHOWN.swap(true, Ordering::Relaxed) {
        eprintln!(
            "\n[moonclip profile] phase totals are CPU time across the rayon \
             pool, so they sum to more than the wall time"
        );
    }

    let gib = bytes as f64 / (1 << 30) as f64;
    eprintln!(
        "\n[moonclip profile] {label}: {:.0} ms wall, {gib:.2} GiB raw \
         ({:.2} GB/s)",
        wall.as_secs_f64() * 1000.0,
        bytes as f64 / wall.as_secs_f64() / 1e9,
    );

    let total: u64 = NANOS.iter().map(|n| n.load(Ordering::Relaxed)).sum();

    for i in 0..NAMES.len() {
        let nanos = NANOS[i].swap(0, Ordering::Relaxed);
        let calls = CALLS[i].swap(0, Ordering::Relaxed);
        if calls == 0 {
            continue;
        }
        let ms = nanos as f64 / 1e6;
        let share = if total > 0 {
            nanos as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        eprintln!("  {:<18}{ms:9.0} ms  {share:5.1}%  {calls:>5} calls", NAMES[i]);
    }
    eprintln!("  {:<18}{:9.0} ms", "total CPU", total as f64 / 1e6);
}
