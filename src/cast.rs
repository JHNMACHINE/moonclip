use rayon::prelude::*;

use crate::error::{Result, MoonclipError};

/// Supported storage dtypes for casting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DType {
    Float32,
    BFloat16,
    Float16,
    /// Keep original dtype, no cast.
    None,
}

impl DType {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "bf16" | "bfloat16" => DType::BFloat16,
            "fp16" | "float16" => DType::Float16,
            "fp32" | "float32" => DType::Float32,
            "none" | "" => DType::None,
            _ => DType::None,
        }
    }

    pub fn to_str(&self) -> &'static str {
        match self {
            DType::Float32 => "float32",
            DType::BFloat16 => "bfloat16",
            DType::Float16 => "float16",
            DType::None => "none",
        }
    }

    /// Bytes per element for this dtype.
    pub fn element_size(&self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::BFloat16 => 2,
            DType::Float16 => 2,
            DType::None => 0,
        }
    }
}

/// Determine if a tensor dtype string represents a float type that can be cast.
pub fn is_castable_float(dtype: &str) -> bool {
    matches!(
        dtype,
        "float32" | "float16" | "bfloat16" | "float64"
    )
}

// ─── fp32 → bf16 ────────────────────────────────────────────────────

/// Cast fp32 bytes to bf16 bytes (parallel).
///
/// bfloat16 is the upper 16 bits of float32, so the cast is just
/// taking bytes [2,3] of each 4-byte float (little-endian).
/// With round-to-nearest-even for better accuracy.
///
/// NaN never becomes Inf: see [`fp32_bits_to_bf16_bits`].
pub fn fp32_to_bf16(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() % 4 != 0 {
        return Err(MoonclipError::Config(format!(
            "fp32 data length {} is not a multiple of 4",
            data.len()
        )));
    }

    let n_elements = data.len() / 4;
    let mut out = vec![0u8; n_elements * 2];

    // Process in chunks for cache locality
    let chunk_size = 16384; // 16K elements per chunk
    let src_chunks: Vec<&[u8]> = data.chunks(chunk_size * 4).collect();
    let dst_chunks: Vec<&mut [u8]> = out.chunks_mut(chunk_size * 2).collect();

    src_chunks
        .into_par_iter()
        .zip(dst_chunks)
        .for_each(|(src, dst)| {
            let n = src.len() / 4;
            for i in 0..n {
                let offset = i * 4;
                let bits = u32::from_le_bytes([
                    src[offset],
                    src[offset + 1],
                    src[offset + 2],
                    src[offset + 3],
                ]);

                let bf16_bytes = fp32_bits_to_bf16_bits(bits).to_le_bytes();
                dst[i * 2] = bf16_bytes[0];
                dst[i * 2 + 1] = bf16_bytes[1];
            }
        });

    Ok(out)
}

/// Cast bf16 bytes back to fp32 bytes (parallel).
///
/// bf16 → fp32 is just padding the lower 16 bits with zeros.
pub fn bf16_to_fp32(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() % 2 != 0 {
        return Err(MoonclipError::Config(format!(
            "bf16 data length {} is not a multiple of 2",
            data.len()
        )));
    }

    let n_elements = data.len() / 2;
    let mut out = vec![0u8; n_elements * 4];

    let chunk_size = 16384;
    let src_chunks: Vec<&[u8]> = data.chunks(chunk_size * 2).collect();
    let dst_chunks: Vec<&mut [u8]> = out.chunks_mut(chunk_size * 4).collect();

    src_chunks
        .into_par_iter()
        .zip(dst_chunks)
        .for_each(|(src, dst)| {
            let n = src.len() / 2;
            for i in 0..n {
                let offset = i * 4;
                // bf16 bits go into the upper 16 bits of fp32
                dst[offset] = 0;
                dst[offset + 1] = 0;
                dst[offset + 2] = src[i * 2];
                dst[offset + 3] = src[i * 2 + 1];
            }
        });

    Ok(out)
}

// ─── fp32 → fp16 ────────────────────────────────────────────────────

/// Cast fp32 bytes to fp16 bytes (parallel).
///
/// IEEE 754 half-precision: sign(1) + exponent(5) + mantissa(10)
pub fn fp32_to_fp16(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() % 4 != 0 {
        return Err(MoonclipError::Config(format!(
            "fp32 data length {} is not a multiple of 4",
            data.len()
        )));
    }

    let n_elements = data.len() / 4;
    let mut out = vec![0u8; n_elements * 2];

    let chunk_size = 16384;
    let src_chunks: Vec<&[u8]> = data.chunks(chunk_size * 4).collect();
    let dst_chunks: Vec<&mut [u8]> = out.chunks_mut(chunk_size * 2).collect();

    src_chunks
        .into_par_iter()
        .zip(dst_chunks)
        .for_each(|(src, dst)| {
            let n = src.len() / 4;
            for i in 0..n {
                let offset = i * 4;
                let bits = u32::from_le_bytes([
                    src[offset],
                    src[offset + 1],
                    src[offset + 2],
                    src[offset + 3],
                ]);
                let fp16_bits = fp32_bits_to_fp16_bits(bits);
                let fp16_bytes = fp16_bits.to_le_bytes();
                dst[i * 2] = fp16_bytes[0];
                dst[i * 2 + 1] = fp16_bytes[1];
            }
        });

    Ok(out)
}

/// Cast fp16 bytes back to fp32 bytes (parallel).
pub fn fp16_to_fp32(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() % 2 != 0 {
        return Err(MoonclipError::Config(format!(
            "fp16 data length {} is not a multiple of 2",
            data.len()
        )));
    }

    let n_elements = data.len() / 2;
    let mut out = vec![0u8; n_elements * 4];

    let chunk_size = 16384;
    let src_chunks: Vec<&[u8]> = data.chunks(chunk_size * 2).collect();
    let dst_chunks: Vec<&mut [u8]> = out.chunks_mut(chunk_size * 4).collect();

    src_chunks
        .into_par_iter()
        .zip(dst_chunks)
        .for_each(|(src, dst)| {
            let n = src.len() / 2;
            for i in 0..n {
                let fp16_bits =
                    u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
                let fp32_bits = fp16_bits_to_fp32_bits(fp16_bits);
                let fp32_bytes = fp32_bits.to_le_bytes();
                let offset = i * 4;
                dst[offset] = fp32_bytes[0];
                dst[offset + 1] = fp32_bytes[1];
                dst[offset + 2] = fp32_bytes[2];
                dst[offset + 3] = fp32_bytes[3];
            }
        });

    Ok(out)
}

// ─── Generic dispatch ───────────────────────────────────────────────

/// Cast tensor bytes from `src_dtype` to `target_dtype`.
/// Returns (casted_bytes, new_dtype_string).
/// If no cast is needed, returns the original data.
pub fn cast_tensor(
    data: &[u8],
    src_dtype: &str,
    target: &DType,
) -> Result<(Vec<u8>, String)> {
    if *target == DType::None {
        return Ok((data.to_vec(), src_dtype.to_string()));
    }

    match (src_dtype, target) {
        ("float32", DType::BFloat16) => Ok((fp32_to_bf16(data)?, "bfloat16".into())),
        ("float32", DType::Float16) => Ok((fp32_to_fp16(data)?, "float16".into())),
        ("bfloat16", DType::Float32) => Ok((bf16_to_fp32(data)?, "float32".into())),
        ("float16", DType::Float32) => Ok((fp16_to_fp32(data)?, "float32".into())),
        // Same dtype → no-op
        ("float32", DType::Float32)
        | ("bfloat16", DType::BFloat16)
        | ("float16", DType::Float16) => Ok((data.to_vec(), src_dtype.to_string())),
        // Non-float or unsupported → no cast
        _ => Ok((data.to_vec(), src_dtype.to_string())),
    }
}

/// Cast tensor bytes back from `stored_dtype` to `original_dtype`.
pub fn uncast_tensor(
    data: &[u8],
    stored_dtype: &str,
    original_dtype: &str,
) -> Result<Vec<u8>> {
    if stored_dtype == original_dtype {
        return Ok(data.to_vec());
    }

    match (stored_dtype, original_dtype) {
        ("bfloat16", "float32") => bf16_to_fp32(data),
        ("float16", "float32") => fp16_to_fp32(data),
        ("float32", "bfloat16") => fp32_to_bf16(data),
        ("float32", "float16") => fp32_to_fp16(data),
        _ => Ok(data.to_vec()),
    }
}

// ─── Internal bit manipulation ──────────────────────────────────────

/// fp32 bit pattern → bf16 bit pattern, round-to-nearest-even.
///
/// The rounding bias is what makes the NaN check necessary rather than
/// pedantic. bf16 keeps only the top 7 mantissa bits, so a NaN whose payload
/// lives below bit 16 arrives at the shift with a mantissa of all zeros — and a
/// zero mantissa under an all-ones exponent is Inf, not NaN. `0x7F800001` (a
/// signalling NaN, and what a comparison against a corrupted tensor tends to
/// produce) plus the bias of `0x7FFF` carries into `0x7F808000`, which truncates
/// to `0x7F80`: positive infinity. The quiet NaN `0x7FC00000` survives on its
/// own because its payload sits in the bits bf16 keeps, so the failure is not
/// one a casual test catches.
///
/// A checkpoint is the last place a NaN should be laundered into a finite-ish
/// value: it is how a diverged run gets saved as if it were healthy, and the
/// gradient that produced it is gone by the time anyone looks. So NaN is mapped
/// to a NaN explicitly — sign and the payload bits bf16 can hold are kept, and
/// the quiet bit is forced so the result cannot collapse to Inf.
#[inline]
fn fp32_bits_to_bf16_bits(bits: u32) -> u16 {
    // Ignoring the sign, anything above the Inf pattern is a NaN.
    if bits & 0x7FFF_FFFF > 0x7F80_0000 {
        return ((bits >> 16) as u16) | 0x0040;
    }

    // Round-to-nearest-even: add rounding bias
    // If the lower 16 bits are exactly 0x8000 (tie), round to even
    // Otherwise, add 0x7FFF + bit 16 for round-to-nearest
    let rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

fn fp32_bits_to_fp16_bits(bits: u32) -> u16 {
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7FFFFF;

    if exp == 255 {
        // Inf or NaN
        if mantissa != 0 {
            return sign | 0x7E00; // NaN
        }
        return sign | 0x7C00; // Inf
    }

    let new_exp = exp - 127 + 15;

    if new_exp >= 31 {
        return sign | 0x7C00; // Overflow → Inf
    }

    if new_exp <= 0 {
        if new_exp < -10 {
            return sign; // Too small → zero
        }
        // Denormalized
        let m = (mantissa | 0x800000) >> (1 - new_exp);
        let round = (m >> 12) & 1;
        return sign | ((m >> 13) + round) as u16;
    }

    // Normal
    let round = (mantissa >> 12) & 1;
    let fp16 = sign | ((new_exp as u16) << 10) | ((mantissa >> 13) as u16);
    fp16 + round as u16
}

fn fp16_bits_to_fp32_bits(bits: u16) -> u32 {
    let sign = (bits as u32 & 0x8000) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mantissa == 0 {
            return sign; // Zero
        }
        // Denormalized → normalize
        let mut e = 0i32;
        let mut m = mantissa;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        let new_exp = (127 - 15 - e) as u32;
        let new_mantissa = (m & 0x3FF) << 13;
        return sign | (new_exp << 23) | new_mantissa;
    }

    if exp == 31 {
        // Inf or NaN
        return sign | 0x7F800000 | (mantissa << 13);
    }

    let new_exp = exp + 127 - 15;
    sign | (new_exp << 23) | (mantissa << 13)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp32_bf16_roundtrip() {
        // 1.0f32 = 0x3F800000 → bf16 = 0x3F80 → back = 0x3F800000
        let fp32: Vec<u8> = 1.0f32.to_le_bytes().to_vec();
        let bf16 = fp32_to_bf16(&fp32).unwrap();
        assert_eq!(bf16.len(), 2);
        let back = bf16_to_fp32(&bf16).unwrap();
        assert_eq!(back.len(), 4);
        let val = f32::from_le_bytes([back[0], back[1], back[2], back[3]]);
        assert_eq!(val, 1.0);
    }

    #[test]
    fn fp32_bf16_precision() {
        // 0.1f32 → bf16 → fp32: should be close but not exact
        let fp32: Vec<u8> = 0.1f32.to_le_bytes().to_vec();
        let bf16 = fp32_to_bf16(&fp32).unwrap();
        let back = bf16_to_fp32(&bf16).unwrap();
        let val = f32::from_le_bytes([back[0], back[1], back[2], back[3]]);
        assert!((val - 0.1).abs() < 0.002, "bf16 error too large: {val} vs 0.1");
    }

    #[test]
    fn fp32_fp16_roundtrip() {
        let fp32: Vec<u8> = 1.0f32.to_le_bytes().to_vec();
        let fp16 = fp32_to_fp16(&fp32).unwrap();
        assert_eq!(fp16.len(), 2);
        let back = fp16_to_fp32(&fp16).unwrap();
        let val = f32::from_le_bytes([back[0], back[1], back[2], back[3]]);
        assert_eq!(val, 1.0);
    }

    #[test]
    fn fp32_fp16_overflow_to_inf() {
        // 100000.0 overflows fp16 → should become inf
        let fp32: Vec<u8> = 100000.0f32.to_le_bytes().to_vec();
        let fp16 = fp32_to_fp16(&fp32).unwrap();
        let back = fp16_to_fp32(&fp16).unwrap();
        let val = f32::from_le_bytes([back[0], back[1], back[2], back[3]]);
        assert!(val.is_infinite());
    }

    #[test]
    fn bf16_zero() {
        let fp32: Vec<u8> = 0.0f32.to_le_bytes().to_vec();
        let bf16 = fp32_to_bf16(&fp32).unwrap();
        let back = bf16_to_fp32(&bf16).unwrap();
        let val = f32::from_le_bytes([back[0], back[1], back[2], back[3]]);
        assert_eq!(val, 0.0);
    }

    #[test]
    fn bf16_negative() {
        // Any float with a fractional part that bf16 cannot hold exactly will
        // do. Not 3.14: clippy reads it as a botched `f32::consts::PI` and
        // `approx_constant` is deny-by-default, so it fails the build.
        let fp32: Vec<u8> = (-3.6f32).to_le_bytes().to_vec();
        let bf16 = fp32_to_bf16(&fp32).unwrap();
        let back = bf16_to_fp32(&bf16).unwrap();
        let val = f32::from_le_bytes([back[0], back[1], back[2], back[3]]);
        assert!((val - (-3.6)).abs() < 0.05);
        assert!(val < 0.0);
    }

    /// The bug this guards: rounding a NaN whose payload is below bit 16 used
    /// to carry it up to the Inf pattern. A checkpoint that turns NaN into +Inf
    /// hides the divergence that produced it.
    #[test]
    fn bf16_keeps_nan_with_a_low_payload() {
        for &pattern in &[0x7F80_0001u32, 0xFF80_0001, 0x7F80_8000] {
            let fp32: Vec<u8> = pattern.to_le_bytes().to_vec();
            let bf16 = fp32_to_bf16(&fp32).unwrap();
            let back = bf16_to_fp32(&bf16).unwrap();
            let val = f32::from_le_bytes([back[0], back[1], back[2], back[3]]);
            assert!(
                val.is_nan(),
                "{pattern:#010X} cast to bf16 {:#06X} and came back {val}",
                u16::from_le_bytes([bf16[0], bf16[1]])
            );
        }
    }

    /// The quiet NaN survives without the guard — its payload sits in the bits
    /// bf16 keeps — so it is here to pin the case that already worked, and to
    /// show the guard did not break the sign.
    #[test]
    fn bf16_keeps_the_quiet_nan() {
        let fp32: Vec<u8> = 0x7FC0_0000u32.to_le_bytes().to_vec();
        let bf16 = fp32_to_bf16(&fp32).unwrap();
        assert_eq!(u16::from_le_bytes([bf16[0], bf16[1]]), 0x7FC0);

        let negative: Vec<u8> = 0xFFC0_0000u32.to_le_bytes().to_vec();
        let bf16 = fp32_to_bf16(&negative).unwrap();
        assert_eq!(u16::from_le_bytes([bf16[0], bf16[1]]), 0xFFC0);
    }

    /// The other half of the guard: an infinity has to stay an infinity, and
    /// finite values large enough to round up still reach it.
    #[test]
    fn bf16_keeps_infinity_and_still_overflows_to_it() {
        for (value, expect_sign) in [(f32::INFINITY, 1.0f32), (f32::NEG_INFINITY, -1.0)] {
            let bf16 = fp32_to_bf16(&value.to_le_bytes()).unwrap();
            let back = bf16_to_fp32(&bf16).unwrap();
            let val = f32::from_le_bytes([back[0], back[1], back[2], back[3]]);
            assert!(val.is_infinite() && val.signum() == expect_sign, "{value} → {val}");
        }

        // f32::MAX rounds up past the largest bf16, which is Inf by IEEE rules.
        let bf16 = fp32_to_bf16(&f32::MAX.to_le_bytes()).unwrap();
        let back = bf16_to_fp32(&bf16).unwrap();
        let val = f32::from_le_bytes([back[0], back[1], back[2], back[3]]);
        assert!(val.is_infinite() && val > 0.0, "f32::MAX → {val}");
    }

    /// fp16 already had the check, in a different shape. Same contract, so the
    /// same test: a payload below the bits fp16 keeps must not become Inf.
    #[test]
    fn fp16_keeps_nan_with_a_low_payload() {
        let fp32: Vec<u8> = 0x7F80_0001u32.to_le_bytes().to_vec();
        let fp16 = fp32_to_fp16(&fp32).unwrap();
        let back = fp16_to_fp32(&fp16).unwrap();
        let val = f32::from_le_bytes([back[0], back[1], back[2], back[3]]);
        assert!(val.is_nan(), "fp16 turned a NaN into {val}");
    }

    #[test]
    fn large_array_parallel() {
        // 1M floats
        let n = 1_000_000;
        let fp32: Vec<u8> = (0..n)
            .flat_map(|i| ((i as f32) * 0.001).to_le_bytes().to_vec())
            .collect();

        let bf16 = fp32_to_bf16(&fp32).unwrap();
        assert_eq!(bf16.len(), n * 2);

        let back = bf16_to_fp32(&bf16).unwrap();
        assert_eq!(back.len(), n * 4);

        // Spot check
        let original = f32::from_le_bytes([fp32[0], fp32[1], fp32[2], fp32[3]]);
        let restored = f32::from_le_bytes([back[0], back[1], back[2], back[3]]);
        assert!((original - restored).abs() < 0.01);
    }

    #[test]
    fn cast_tensor_dispatch() {
        let data = 42.0f32.to_le_bytes().to_vec();

        let (casted, dtype) = cast_tensor(&data, "float32", &DType::BFloat16).unwrap();
        assert_eq!(dtype, "bfloat16");
        assert_eq!(casted.len(), 2);

        let (back, dtype2) = cast_tensor(&casted, "bfloat16", &DType::Float32).unwrap();
        assert_eq!(dtype2, "float32");
        assert_eq!(back.len(), 4);
    }

    #[test]
    fn cast_nonfloat_noop() {
        let data = vec![1u8, 2, 3, 4];
        let (result, dtype) = cast_tensor(&data, "int32", &DType::BFloat16).unwrap();
        assert_eq!(result, data); // No cast for int types
        assert_eq!(dtype, "int32");
    }

    #[test]
    fn wrong_alignment_errors() {
        assert!(fp32_to_bf16(&[0u8; 3]).is_err());
        assert!(bf16_to_fp32(&[0u8; 3]).is_err());
        assert!(fp32_to_fp16(&[0u8; 5]).is_err());
    }

    #[test]
    fn uncast_tensor_roundtrip() {
        // See bf16_negative on the choice of constant.
        let data = 3.6f32.to_le_bytes().to_vec();
        let (casted, stored_dtype) = cast_tensor(&data, "float32", &DType::BFloat16).unwrap();
        let uncasted = uncast_tensor(&casted, &stored_dtype, "float32").unwrap();
        let val = f32::from_le_bytes([uncasted[0], uncasted[1], uncasted[2], uncasted[3]]);
        assert!((val - 3.6).abs() < 0.05);
    }
}
