//! Byte-plane transpose, applied to XOR deltas before compression.
//!
//! An XOR delta of float weights after an optimizer step has a shape that
//! defeats a general-purpose compressor. In little-endian fp32 the low
//! mantissa bytes change on nearly every step while the high mantissa,
//! exponent and sign almost never do, so zstd sees:
//!
//! ```text
//! noise noise zero zero  noise noise zero zero  ...
//! ```
//!
//! The zeros never form a run, and zstd has to earn every one of them.
//! Transposing the stream into byte planes — every byte 0, then every byte 1,
//! and so on — collapses the unchanged planes into long runs it encodes
//! almost for free. HDF5, blosc and zarr all ship the same filter.
//!
//! Measured on a 201 MB XOR delta across two real AdamW steps, zstd level 3:
//!
//! | | time | size |
//! |---|---|---|
//! | as-is | 1329 ms | 55.4% of raw |
//! | shuffled | 445 ms + 97 ms to shuffle | 45.0% of raw |
//!
//! **2.45x faster and 18.8% smaller** — not a trade, a win on both axes,
//! which is why this is on by default for deltas. On full tensors the same
//! filter is a mild trade (~10% slower for ~8% smaller), so it is not applied
//! there.

use rayon::prelude::*;

/// Elements per block when scattering planes back. Sized to keep the
/// destination inside L2 while the reads stream through each plane.
const UNSHUFFLE_BLOCK: usize = 8192;

/// Group `data` into `itemsize` byte planes.
///
/// A trailing partial element is copied through unchanged, so the transform
/// round-trips for any length.
pub fn shuffle(data: &[u8], itemsize: usize) -> Vec<u8> {
    let count = if itemsize == 0 { 0 } else { data.len() / itemsize };
    if itemsize <= 1 || count == 0 {
        return data.to_vec();
    }
    let n = count * itemsize;

    let mut out = vec![0u8; data.len()];
    out[n..].copy_from_slice(&data[n..]);
    out[..n]
        .par_chunks_mut(count)
        .enumerate()
        .for_each(|(plane, dst)| {
            for (i, slot) in dst.iter_mut().enumerate() {
                *slot = data[i * itemsize + plane];
            }
        });
    out
}

/// Inverse of [`shuffle`].
pub fn unshuffle(data: &[u8], itemsize: usize) -> Vec<u8> {
    let count = if itemsize == 0 { 0 } else { data.len() / itemsize };
    if itemsize <= 1 || count == 0 {
        return data.to_vec();
    }
    let n = count * itemsize;

    let mut out = vec![0u8; data.len()];
    out[n..].copy_from_slice(&data[n..]);
    out[..n]
        .par_chunks_mut(itemsize * UNSHUFFLE_BLOCK)
        .enumerate()
        .for_each(|(block, dst)| {
            let first = block * UNSHUFFLE_BLOCK;
            for i in 0..dst.len() / itemsize {
                for plane in 0..itemsize {
                    dst[i * itemsize + plane] = data[plane * count + first + i];
                }
            }
        });
    out
}

/// Byte width of one element of `dtype`.
///
/// Unknown dtypes return 1, which makes the shuffle a no-op rather than a
/// corruption: a filter that cannot be inverted must never be applied.
pub fn element_size(dtype: &str) -> usize {
    match dtype {
        "float64" | "int64" | "uint64" | "complex64" => 8,
        "float32" | "int32" | "uint32" => 4,
        "float16" | "bfloat16" | "int16" | "uint16" => 2,
        "int8" | "uint8" | "bool" => 1,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(len: usize, seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 24) as u8
            })
            .collect()
    }

    #[test]
    fn round_trips_at_every_length() {
        // Lengths either side of the element size and the block boundary: the
        // ragged tail is the case a plane transpose gets wrong.
        for &itemsize in &[2usize, 4, 8] {
            for len in [
                0, 1, itemsize - 1, itemsize, itemsize + 1,
                1000, itemsize * UNSHUFFLE_BLOCK,
                itemsize * UNSHUFFLE_BLOCK + itemsize + 1,
                100_003,
            ] {
                let data = noise(len, len as u64 + itemsize as u64);
                let there = shuffle(&data, itemsize);
                assert_eq!(there.len(), data.len(), "itemsize {itemsize}, len {len}");
                assert_eq!(
                    unshuffle(&there, itemsize),
                    data,
                    "itemsize {itemsize}, len {len}"
                );
            }
        }
    }

    #[test]
    fn groups_the_planes() {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(shuffle(&data, 4), vec![1, 5, 2, 6, 3, 7, 4, 8]);
    }

    #[test]
    fn itemsize_one_is_identity() {
        let data = noise(1000, 7);
        assert_eq!(shuffle(&data, 1), data);
        assert_eq!(unshuffle(&data, 1), data);
    }

    /// The reason the filter exists: an XOR delta whose high bytes are
    /// untouched has to compress better shuffled than not.
    #[test]
    fn unchanged_planes_compress_away() {
        use crate::compression::compress;
        use crate::manifest::CompressionAlgo;

        let algo = CompressionAlgo::Zstd { level: 3 };
        // fp32 where only the two low mantissa bytes moved, as a small
        // optimizer step leaves them.
        let mut delta = vec![0u8; 400_000];
        let noisy = noise(200_000, 0x1234);
        for (i, chunk) in delta.chunks_mut(4).enumerate() {
            chunk[0] = noisy[i * 2];
            chunk[1] = noisy[i * 2 + 1];
        }

        let plain = compress(&delta, &algo).unwrap().len();
        let shuffled = compress(&shuffle(&delta, 4), &algo).unwrap().len();
        assert!(
            shuffled < plain,
            "shuffled delta must compress smaller: {shuffled} vs {plain}"
        );
    }

    #[test]
    fn element_size_defaults_to_one_for_unknown_dtypes() {
        assert_eq!(element_size("float32"), 4);
        assert_eq!(element_size("bfloat16"), 2);
        // A dtype nobody taught it about must disable the filter, not guess.
        assert_eq!(element_size("float8_e4m3"), 1);
    }
}
