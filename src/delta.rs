use crate::error::{Result, RevolverError};
use rayon::prelude::*;

/// Size threshold: tensors smaller than this won't be delta-encoded
/// (the overhead isn't worth it).
const DELTA_MIN_SIZE: usize = 4096;

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

    // Parallel XOR in 1MB chunks
    let chunk_size = 1 << 20;
    let delta: Vec<u8> = base
        .par_chunks(chunk_size)
        .zip(target.par_chunks(chunk_size))
        .flat_map(|(b, t)| {
            b.iter().zip(t.iter()).map(|(x, y)| x ^ y).collect::<Vec<u8>>()
        })
        .collect();

    Some(delta)
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

    let chunk_size = 1 << 20;
    let target: Vec<u8> = base
        .par_chunks(chunk_size)
        .zip(delta.par_chunks(chunk_size))
        .flat_map(|(b, d)| {
            b.iter().zip(d.iter()).map(|(x, y)| x ^ y).collect::<Vec<u8>>()
        })
        .collect();

    Ok(target)
}

/// Compute the ratio of non-zero bytes in a delta.
/// A low ratio (< 0.3) means delta encoding is very effective.
/// A high ratio (> 0.7) means a full snapshot might be cheaper.
pub fn delta_density(delta: &[u8]) -> f64 {
    if delta.is_empty() {
        return 0.0;
    }
    let nonzero = delta.par_iter().filter(|&&b| b != 0).count();
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
}
