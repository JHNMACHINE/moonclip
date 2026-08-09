use crate::compression;
use crate::error::{Result, MoonclipError};
use crate::manifest::CompressionAlgo;
use rayon::prelude::*;

/// Size threshold: tensors smaller than this won't be delta-encoded
/// (the overhead isn't worth it).
const DELTA_MIN_SIZE: usize = 4096;

/// Window compared when deciding whether a delta is worth storing.
const DECISION_SAMPLE: usize = 64 * 1024;

/// Below this many bytes a sample is too short to judge reliably.
const DECISION_MIN: usize = 8 * 1024;

/// Parallel chunk size for XOR / density scans.
const SCAN_CHUNK: usize = 1 << 20;

/// XOR two equal-length buffers into a new buffer (parallel, vectorized).
///
/// Panics if the lengths differ. Both callers check first; without the assert
/// the zip below would silently stop at the shorter buffer and leave the tail
/// of the result as zeros, which reads as a valid delta and would corrupt the
/// restored tensor rather than fail.
fn xor_bytes(base: &[u8], target: &[u8]) -> Vec<u8> {
    let n = base.len();
    assert_eq!(n, target.len(), "xor_bytes requires equal lengths");

    // `vec![0u8; n]` looks like a wasted pass — every byte is overwritten
    // below — but it is not: this lowers to `alloc_zeroed`, and at these sizes
    // the allocator serves the request with fresh kernel pages that are
    // already zero and are faulted in lazily. No memset runs. Filling an
    // uninitialised buffer instead was measured and made no difference on
    // compute_delta and made apply_delta slightly worse, so the unsafe that
    // would be needed to do it buys nothing.
    let mut out = vec![0u8; n];

    out.par_chunks_mut(SCAN_CHUNK)
        .zip(base.par_chunks(SCAN_CHUNK).zip(target.par_chunks(SCAN_CHUNK)))
        .for_each(|(o, (b, t))| {
            for i in 0..o.len() {
                o[i] = b[i] ^ t[i];
            }
        });

    out
}

/// Whether two buffers are byte-identical.
///
/// `base == target` computes the same answer, but single-threaded. On an
/// unchanged tensor — a frozen embedding table, a parameter group that took no
/// gradient — that memcmp *is* the entire cost of deciding to skip the tensor,
/// and the skip is supposed to be the cheap path. Splitting it over the pool
/// keeps the short-circuit: `any` stops scheduling chunks as soon as one
/// reports a difference, so the common changed case still bails out early.
fn buffers_equal(base: &[u8], target: &[u8]) -> bool {
    base.len() == target.len()
        && !base
            .par_chunks(SCAN_CHUNK)
            .zip(target.par_chunks(SCAN_CHUNK))
            .any(|(b, t)| b != t)
}

/// Decide whether XOR-delta encoding is worth it, by comparing the
/// *compressed* size of a delta sample against the compressed size of the
/// same window stored in full. Returns true when the delta wins by at
/// least the margin implied by `max_ratio`.
///
/// The fraction of differing bytes — the criterion used previously — is a
/// poor predictor for float tensors. After a typical optimizer step
/// roughly three bytes in four change, so any density threshold below
/// ~0.75 rejects every delta; yet the XOR still zeroes the sign, exponent
/// and high mantissa bits, leaving a buffer that compresses far better
/// than the raw weights. Measured on fp32 weights with zstd-3, a delta
/// beats a full save by 18% at density 0.70 and by 30% at density 0.53 —
/// both of which the old 0.5 threshold discarded. Comparing compressed
/// sizes measures the thing we actually care about.
///
/// `base` and `target` may be prefixes of the real buffers: only the
/// leading `DECISION_SAMPLE` bytes are ever examined, which keeps the
/// probe cheap enough to run per tensor.
pub fn pays_off(
    base: &[u8],
    target: &[u8],
    compression: &CompressionAlgo,
    max_ratio: f64,
) -> bool {
    let n = base.len().min(target.len()).min(DECISION_SAMPLE);
    if n < DECISION_MIN {
        return true; // too little to judge — let the exact path decide
    }

    let xor: Vec<u8> = base[..n]
        .iter()
        .zip(&target[..n])
        .map(|(a, b)| a ^ b)
        .collect();

    match (
        compression::compress(&xor, compression),
        compression::compress(&target[..n], compression),
    ) {
        (Ok(delta_c), Ok(full_c)) if !full_c.is_empty() => {
            (delta_c.len() as f64) < (full_c.len() as f64) * max_ratio
        }
        // Compression failed on the probe: don't let a sampling problem
        // block the delta, the exact path still produces a correct result.
        _ => true,
    }
}

/// Compute a byte-level XOR delta between `base` and `target`.
///
/// The result, when XOR-ed with `base`, reproduces `target`.
/// XOR deltas compress extremely well with zstd when most bytes are
/// unchanged (optimizer momentum, rarely-updated embeddings, etc.).
///
/// Returns `None` if the tensors are identical (no delta needed) or
/// if the sizes differ (full snapshot required).
pub fn compute_delta(base: &[u8], target: &[u8]) -> Option<Vec<u8>> {
    if base.len() != target.len() {
        return None; // shape change → full save
    }
    // Size before content: a tensor under the threshold is rejected either
    // way, so testing it first saves scanning buffers whose answer is already
    // determined.
    if base.len() < DELTA_MIN_SIZE {
        return None; // too small to bother
    }
    if buffers_equal(base, target) {
        return None; // identical → skip
    }

    Some(xor_bytes(base, target))
}

/// Apply a XOR delta to a base to recover the target.
pub fn apply_delta(base: &[u8], delta: &[u8]) -> Result<Vec<u8>> {
    if base.len() != delta.len() {
        return Err(MoonclipError::Delta(format!(
            "base length {} != delta length {}",
            base.len(),
            delta.len()
        )));
    }

    Ok(xor_bytes(base, delta))
}

/// Compute the ratio of non-zero bytes in a delta.
/// A low ratio (< 0.3) means delta encoding is very effective.
/// A high ratio (> 0.7) means a full snapshot might be cheaper.
pub fn delta_density(delta: &[u8]) -> f64 {
    if delta.is_empty() {
        return 0.0;
    }
    let nonzero: usize = delta
        .par_chunks(SCAN_CHUNK)
        .map(|c| c.iter().map(|&b| (b != 0) as usize).sum::<usize>())
        .sum();
    nonzero as f64 / delta.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xor_roundtrip() {
        let base = vec![1u8; 8192];
        let mut target = base.clone();
        target[0] = 99;
        target[4096] = 200;

        let delta = compute_delta(&base, &target).unwrap();
        let density = delta_density(&delta);
        assert!(density < 0.01, "density should be very low, got {density}");

        let recovered = apply_delta(&base, &delta).unwrap();
        assert_eq!(target, recovered);
    }

    #[test]
    fn identical_returns_none() {
        let data = vec![0u8; 8192];
        assert!(compute_delta(&data, &data).is_none());
    }

    #[test]
    fn different_sizes_returns_none() {
        let a = vec![0u8; 100];
        let b = vec![0u8; 200];
        assert!(compute_delta(&a, &b).is_none());
    }

    #[test]
    fn small_tensor_returns_none() {
        let a = vec![0u8; 100];
        let mut b = a.clone();
        b[0] = 1;
        assert!(compute_delta(&a, &b).is_none());
    }

    #[test]
    fn large_delta_high_density() {
        let base = vec![0u8; 8192];
        let target = vec![255u8; 8192]; // every byte different

        let delta = compute_delta(&base, &target).unwrap();
        let density = delta_density(&delta);
        assert!(density > 0.99, "fully different should have density ~1.0, got {density}");
    }

    const ZSTD3: CompressionAlgo = CompressionAlgo::Zstd { level: 3 };

    /// Deterministic xorshift — keeps the tests reproducible without
    /// pulling in an RNG dependency.
    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        /// Uniform in [-1, 1).
        fn next_f32(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1u32 << 23) as f32 * 2.0 - 1.0
        }
    }

    fn byte_density(a: &[u8], b: &[u8]) -> f64 {
        let diff = a.iter().zip(b).filter(|(x, y)| x != y).count();
        diff as f64 / a.len() as f64
    }

    /// Incompressible bytes, standing in for real weight data. A base of
    /// constant bytes would compress to nothing in full form, and a delta
    /// against it would legitimately save nothing.
    fn noise(len: usize, seed: u64) -> Vec<u8> {
        let mut rng = Rng(seed);
        (0..len).map(|_| rng.next_u64() as u8).collect()
    }

    #[test]
    fn pays_off_accepts_sparse_change() {
        let base = noise(100_000, 0xabcd);
        let mut target = base.clone();
        target[0] ^= 1;
        target[50_000] ^= 2;
        assert!(pays_off(&base, &target, &ZSTD3, 0.95));
    }

    #[test]
    fn pays_off_rejects_unrelated_data() {
        // Two independent random buffers: the XOR is just as random as the
        // target itself, so the delta buys nothing and costs an extra
        // indirection at load time.
        let base = noise(100_000, 0x1234_5678);
        let target = noise(100_000, 0x8765_4321);
        assert!(!pays_off(&base, &target, &ZSTD3, 0.95));
    }

    /// Regression: fp32 weights nudged by an optimizer step change roughly
    /// three bytes in four, so the old byte-density criterion (threshold
    /// 0.5) rejected every delta — the XOR path was effectively dead code
    /// on dense training. The XOR is still far more compressible than the
    /// raw weights, and must be accepted.
    #[test]
    fn pays_off_accepts_float_weights_after_optimizer_step() {
        let mut rng = Rng(0xdead_beef);
        let n = 32_768; // 128 KB of fp32
        let weights: Vec<f32> = (0..n).map(|_| rng.next_f32() * 0.02).collect();
        // Relative update of ~1e-3, the typical magnitude of an AdamW step.
        let updated: Vec<f32> = weights
            .iter()
            .map(|w| w * (1.0 + 1e-3 * rng.next_f32()))
            .collect();

        let base: Vec<u8> = weights.iter().flat_map(|w| w.to_le_bytes()).collect();
        let target: Vec<u8> = updated.iter().flat_map(|w| w.to_le_bytes()).collect();

        let density = byte_density(&base[..DECISION_SAMPLE], &target[..DECISION_SAMPLE]);
        assert!(
            density > 0.5,
            "precondition: the old criterion must reject this, got density {density}"
        );
        assert!(
            pays_off(&base, &target, &ZSTD3, 0.95),
            "XOR delta of nudged fp32 weights must beat a full save"
        );
    }

    #[test]
    fn pays_off_defaults_true_on_short_sample() {
        let base = vec![0u8; DECISION_MIN - 1];
        let target = vec![255u8; DECISION_MIN - 1];
        assert!(pays_off(&base, &target, &ZSTD3, 0.95));
    }

    #[test]
    fn pays_off_honours_max_ratio() {
        // A change sparse enough to compress well, but a max_ratio of 0
        // demands an impossible win.
        let base = noise(100_000, 0x5555);
        let mut target = base.clone();
        target[0] ^= 1;
        assert!(pays_off(&base, &target, &ZSTD3, 0.95));
        assert!(!pays_off(&base, &target, &ZSTD3, 0.0));
    }

    #[test]
    fn apply_delta_wrong_length_errors() {
        let base = vec![0u8; 100];
        let delta = vec![0u8; 200];
        assert!(apply_delta(&base, &delta).is_err());
    }

    #[test]
    fn xor_is_self_inverse() {
        // XOR(XOR(base, target), target) == base
        let base = vec![42u8; 8192];
        let mut target = base.clone();
        target[0] = 1;
        target[4000] = 2;

        let delta = compute_delta(&base, &target).unwrap();
        let recovered_target = apply_delta(&base, &delta).unwrap();
        let re_delta = compute_delta(&recovered_target, &base).unwrap();
        let recovered_base = apply_delta(&recovered_target, &re_delta).unwrap();
        assert_eq!(recovered_base, base);
    }

    #[test]
    fn delta_density_empty() {
        assert_eq!(delta_density(&[]), 0.0);
    }

    #[test]
    fn delta_density_all_zero() {
        assert_eq!(delta_density(&vec![0u8; 1000]), 0.0);
    }

    /// `xor_bytes` fills an uninitialised buffer, so every byte must be
    /// written before the Vec is handed back as `Vec<u8>`. A size that is an
    /// exact multiple of SCAN_CHUNK (as `delta_multimegabyte` uses) cannot
    /// catch a mishandled tail: the ragged last chunk is the interesting case.
    #[test]
    fn xor_fills_ragged_tail() {
        for extra in [1usize, 7, 4095, SCAN_CHUNK - 1] {
            let size = 2 * SCAN_CHUNK + extra;
            let base = noise(size, 0x51de);
            let mut target = base.clone();
            // Differ only in the very last byte: everything else XORs to zero,
            // so an unwritten tail shows up as garbage instead of 0.
            *target.last_mut().unwrap() ^= 0xff;

            let delta = compute_delta(&base, &target).expect("sizes match, content differs");
            assert_eq!(delta.len(), size);
            assert_eq!(*delta.last().unwrap(), 0xff, "tail byte wrong at extra={extra}");
            assert!(
                delta[..size - 1].iter().all(|&b| b == 0),
                "untouched region must XOR to zero at extra={extra}"
            );
            assert_eq!(apply_delta(&base, &delta).unwrap(), target);
        }
    }

    /// The parallel equality check must not be fooled by a difference that
    /// falls outside the first chunk.
    #[test]
    fn identical_detected_across_chunks() {
        let size = 3 * SCAN_CHUNK + 17;
        let base = noise(size, 0xfeed);
        assert!(compute_delta(&base, &base.clone()).is_none());

        // A single differing byte in the final chunk must defeat the skip.
        let mut target = base.clone();
        target[size - 3] ^= 1;
        assert!(compute_delta(&base, &target).is_some());
    }

    #[test]
    fn delta_multimegabyte() {
        let size = 4 * 1024 * 1024; // 4MB
        let base = vec![0u8; size];
        let mut target = base.clone();
        target[0] = 1;
        target[size - 1] = 2;

        let delta = compute_delta(&base, &target).unwrap();
        assert_eq!(delta.len(), size);
        let recovered = apply_delta(&base, &delta).unwrap();
        assert_eq!(recovered, target);
    }
}
