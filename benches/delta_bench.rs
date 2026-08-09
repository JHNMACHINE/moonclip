//! Manual timing harness for the delta hot path.
//!
//! Run with: cargo bench --bench delta_bench
//!
//! Deliberately harness-free: these functions are memory-bandwidth bound, so
//! what matters is GB/s over a buffer large enough to miss cache entirely,
//! not the statistical machinery criterion would add.

use moonclip::delta::{apply_delta, compute_delta, delta_density};
use std::time::Instant;

/// 256 MB — well past any L3, small enough that base+target+out fits
/// comfortably in RAM alongside the allocator's slack.
const SIZE: usize = 256 * 1024 * 1024;
const ITERS: usize = 5;

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// fp32 weights nudged by a ~1e-3 relative step, the realistic input: most
/// bytes differ, but the XOR is highly compressible.
fn weights_pair(len: usize) -> (Vec<u8>, Vec<u8>) {
    let mut rng = Rng(0xdead_beef);
    let n = len / 4;
    let mut base = Vec::with_capacity(len);
    let mut target = Vec::with_capacity(len);
    for _ in 0..n {
        let w = (rng.next_u64() >> 40) as f32 / (1u32 << 23) as f32 * 0.02;
        let u = w * (1.0 + 1e-3 * ((rng.next_u64() >> 40) as f32 / (1u32 << 23) as f32));
        base.extend_from_slice(&w.to_le_bytes());
        target.extend_from_slice(&u.to_le_bytes());
    }
    (base, target)
}

fn bench<F: FnMut() -> usize>(name: &str, bytes_touched: usize, mut f: F) {
    // Warm up: first touch faults in pages and lets rayon spin up its pool.
    let mut guard = f();
    let mut best = f64::MAX;
    let mut total = 0.0;
    for _ in 0..ITERS {
        let t = Instant::now();
        guard ^= f();
        let secs = t.elapsed().as_secs_f64();
        total += secs;
        if secs < best {
            best = secs;
        }
    }
    let mean = total / ITERS as f64;
    let gbs = |s: f64| bytes_touched as f64 / s / 1e9;
    println!(
        "{name:<28} best {:>7.1} ms ({:>5.1} GB/s)   mean {:>7.1} ms ({:>5.1} GB/s)  [{guard}]",
        best * 1e3,
        gbs(best),
        mean * 1e3,
        gbs(mean),
    );
}

fn main() {
    println!("buffer: {} MB, {ITERS} iterations\n", SIZE / (1024 * 1024));
    let (base, target) = weights_pair(SIZE);

    // compute_delta reads base+target and writes out: 3x the buffer.
    bench("compute_delta (changed)", SIZE * 3, || {
        compute_delta(&base, &target).map_or(0, |d| d.len())
    });

    // Identical inputs: the early-out path. Must be a distinct allocation —
    // passing the same slice twice lets memcmp short-circuit on the pointer
    // and measures nothing.
    let base_copy = base.clone();
    bench("compute_delta (identical)", SIZE * 2, || {
        compute_delta(&base, &base_copy).map_or(0, |d| d.len())
    });

    let delta = compute_delta(&base, &target).expect("weights differ");

    bench("apply_delta", SIZE * 3, || {
        apply_delta(&base, &delta).map(|v| v.len()).unwrap_or(0)
    });

    bench("delta_density", SIZE, || delta_density(&delta) as usize);
}
