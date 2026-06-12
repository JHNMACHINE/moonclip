use crate::error::{Result, RevolverError};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Size threshold: tensors smaller than this won't be delta-encoded
/// (the overhead isn't worth it).
const DELTA_MIN_SIZE: usize = 4096;

/// Parallel chunk size for XOR / density scans.
const SCAN_CHUNK: usize = 1 << 20;

/// XOR two equal-length buffers into a new buffer (parallel, vectorizable).
fn xor_bytes(base: &[u8], target: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; base.len()];
    out.par_chunks_mut(SCAN_CHUNK)
        .zip(base.par_chunks(SCAN_CHUNK).zip(target.par_chunks(SCAN_CHUNK)))
        .for_each(|(o, (b, t))| {
            for i in 0..o.len() {
                o[i] = b[i] ^ t[i];
            }
        });
    out
}

/// Compute a XOR delta between `base` and `target`, but only if the
/// fraction of differing bytes is below `threshold`.
///
/// This fuses the old compute_delta + delta_density steps: the density
/// scan runs first (no allocation) with an early exit as soon as the
/// changed-byte budget is exceeded — the common case after optimizer
/// steps, where nearly every byte changes. The delta buffer is only
/// materialized when it will actually be used.
///
/// Returns `None` if sizes differ, the buffer is too small, the buffers
/// are identical, or the density is at or above `threshold`.
pub fn delta_if_sparse(base: &[u8], target: &[u8], threshold: f64) -> Option<Vec<u8>> {
    if base.len() != target.len() || base.len() < DELTA_MIN_SIZE {
        return None;
    }

    let limit = (base.len() as f64 * threshold) as usize;
    let count = AtomicUsize::new(0);
    let exceeded = base
        .par_chunks(SCAN_CHUNK)
        .zip(target.par_chunks(SCAN_CHUNK))
        .try_for_each(|(b, t)| {
            if count.load(Ordering::Relaxed) > limit {
                return Err(());
            }
            let local: usize = b.iter().zip(t).map(|(x, y)| (x != y) as usize).sum();
            if count.fetch_add(local, Ordering::Relaxed) + local > limit {
                Err(())
            } else {
                Ok(())
            }
        })
        .is_err();

    if exceeded || count.load(Ordering::Relaxed) == 0 {
        return None; // too dense, or identical (caller skips via hash)
    }

    Some(xor_bytes(base, target))
}

/// Fraction of differing bytes between two sample buffers.
/// Used to cheaply estimate delta density from a decompressed prefix
/// of the base tensor before committing to a full decompression.
pub fn sample_density(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 1.0;
    }
    let diff = a[..n].iter().zip(&b[..n]).filter(|(x, y)| x != y).count();
    diff as f64 / n as f64
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
    if base == target {
        return None; // identical → skip
    }
    if base.len() < DELTA_MIN_SIZE {
        return None; // too small to bother
    }

    Some(xor_bytes(base, target))
}

/// Apply a XOR delta to a base to recover the target.
pub fn apply_delta(base: &[u8], delta: &[u8]) -> Result<Vec<u8>> {
    if base.len() != delta.len() {
        return Err(RevolverError::Delta(format!(
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
        assert!(delta_if_sparse(&data, &data, 0.5).is_none());
    }

    #[test]
    fn different_sizes_returns_none() {
        let a = vec![0u8; 100];
        let b = vec![0u8; 200];
        assert!(compute_delta(&a, &b).is_none());
        assert!(delta_if_sparse(&a, &b, 0.5).is_none());
    }

    #[test]
    fn small_tensor_returns_none() {
        let a = vec![0u8; 100];
        let mut b = a.clone();
        b[0] = 1;
        assert!(compute_delta(&a, &b).is_none());
        assert!(delta_if_sparse(&a, &b, 0.5).is_none());
    }

    #[test]
    fn large_delta_high_density() {
        let base = vec![0u8; 8192];
        let target = vec![255u8; 8192]; // every byte different

        let delta = compute_delta(&base, &target).unwrap();
        let density = delta_density(&delta);
        assert!(density > 0.99, "fully different should have density ~1.0, got {density}");
    }

    #[test]
    fn delta_if_sparse_accepts_sparse() {
        let base = vec![0u8; 100_000];
        let mut target = base.clone();
        target[0] = 1;
        target[50_000] = 2;

        let delta = delta_if_sparse(&base, &target, 0.5).unwrap();
        let recovered = apply_delta(&base, &delta).unwrap();
        assert_eq!(target, recovered);
    }

    #[test]
    fn delta_if_sparse_rejects_dense() {
        let base = vec![0u8; 100_000];
        let target = vec![255u8; 100_000];
        assert!(delta_if_sparse(&base, &target, 0.5).is_none());
    }

    #[test]
    fn delta_if_sparse_threshold_boundary() {
        // ~60% of bytes changed → rejected at 0.5, accepted at 0.7
        let base = vec![0u8; 100_000];
        let mut target = base.clone();
        for i in 0..60_000 {
            target[i] = 1;
        }
        assert!(delta_if_sparse(&base, &target, 0.5).is_none());
        assert!(delta_if_sparse(&base, &target, 0.7).is_some());
    }

    #[test]
    fn sample_density_estimates() {
        let a = vec![0u8; 1000];
        let mut b = a.clone();
        for i in 0..500 {
            b[i] = 1;
        }
        let d = sample_density(&a, &b);
        assert!((d - 0.5).abs() < 1e-9);
        assert_eq!(sample_density(&[], &[]), 1.0);
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
