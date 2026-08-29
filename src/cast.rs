use std::borrow::Cow;

use rayon::prelude::*;

use crate::error::{Result, MoonclipError};

/// Supported storage dtypes for casting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DType {
    Float32,
    BFloat16,
    Float16,
    /// `float8_e4m3fn` — 4 exponent bits, 3 mantissa, no infinities.
    /// Quantized against a per-tensor scale; see [`fp32_to_fp8`].
    Float8E4M3,
    /// `float8_e5m2` — 5 exponent bits, 2 mantissa, IEEE-shaped.
    /// Quantized against a per-tensor scale; see [`fp32_to_fp8`].
    Float8E5M2,
    /// Keep original dtype, no cast.
    None,
}

impl DType {
    /// Parse a `save_dtype` string.
    ///
    /// An unrecognised one is an error, not `None`. Mapping it to "do not
    /// cast" meant `save_dtype="bfloat"` — or `"BF16 "`, or any other typo —
    /// silently produced full-precision checkpoints twice the expected size,
    /// and the only symptom was a disk bill. The value comes from a user's
    /// config file and is worth exactly one comparison to check.
    ///
    /// Bare `"fp8"` means e4m3: it is what torchao and Transformer Engine pick
    /// for weights, because three mantissa bits beat the extra exponent range
    /// on values that have already been scaled into range. e5m2 is reachable
    /// by name for gradient-shaped data, where the range is what runs out.
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "bf16" | "bfloat16" => Ok(DType::BFloat16),
            "fp16" | "float16" => Ok(DType::Float16),
            "fp32" | "float32" => Ok(DType::Float32),
            "fp8" | "float8" | "fp8_e4m3" | "float8_e4m3fn" => Ok(DType::Float8E4M3),
            "fp8_e5m2" | "float8_e5m2" => Ok(DType::Float8E5M2),
            "none" | "" => Ok(DType::None),
            other => Err(MoonclipError::Config(format!(
                "unknown save_dtype '{other}'. Use one of: none, bf16, fp16, \
                 fp32, fp8 (= fp8_e4m3), fp8_e5m2"
            ))),
        }
    }

    pub fn to_str(&self) -> &'static str {
        match self {
            DType::Float32 => "float32",
            DType::BFloat16 => "bfloat16",
            DType::Float16 => "float16",
            // Torch's own spelling, so a manifest dtype is a name someone can
            // look up rather than one this crate invented.
            DType::Float8E4M3 => "float8_e4m3fn",
            DType::Float8E5M2 => "float8_e5m2",
            DType::None => "none",
        }
    }

    /// Bytes per element for this dtype.
    pub fn element_size(&self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::BFloat16 => 2,
            DType::Float16 => 2,
            DType::Float8E4M3 => 1,
            DType::Float8E5M2 => 1,
            DType::None => 0,
        }
    }

    /// The float8 layout this dtype stores, if it is a float8 at all.
    pub fn fp8_format(&self) -> Option<&'static Fp8Format> {
        match self {
            DType::Float8E4M3 => Some(&E4M3FN),
            DType::Float8E5M2 => Some(&E5M2),
            _ => None,
        }
    }
}

/// Determine if a tensor dtype string represents a float type that can be cast.
///
/// Every dtype named here really is converted by [`cast_tensor`], to any of the
/// targets. That was not true until 0.0.6: this list claimed float64 while
/// `cast_tensor` had no arm for it, and float16 and bfloat16 were only ever
/// converted to and from float32 — so `save_dtype="bf16"` on a model holding
/// fp16 or fp64 buffers stored them untouched, at the size the setting was
/// chosen to avoid, and said nothing. Everything routes through fp32 now,
/// which is what makes the set closed.
///
/// Float8 is deliberately absent, and the asymmetry is the point: float8 is a
/// *target* of a cast, never a source. A tensor that arrives already float8 —
/// which is what FSDP2 and torchao hand over — is stored byte for byte and
/// read back byte for byte, with no scale and no `original_dtype`. Listing it
/// here would instead route it through fp32 on the way in, so `save_dtype`
/// pointing anywhere else would have *doubled* the size of the one dtype
/// chosen to make things smaller, silently. The load path recognises float8
/// on its own, in [`uncast_tensor`], before this predicate is consulted.
pub fn is_castable_float(dtype: &str) -> bool {
    matches!(dtype, "float32" | "float16" | "bfloat16" | "float64")
}

fn fp8_format_for(dtype: &str) -> Option<&'static Fp8Format> {
    match dtype {
        "float8_e4m3fn" => Some(&E4M3FN),
        "float8_e5m2" => Some(&E5M2),
        _ => None,
    }
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

// ─── fp64 ↔ fp32 ────────────────────────────────────────────────────

/// Cast fp64 bytes to fp32 bytes.
///
/// Values too large for fp32 become infinities, which is what the hardware
/// does and what `tensor.float()` does. Nothing here is lossless; the caller
/// asked for a narrower checkpoint.
pub fn fp64_to_fp32(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() % 8 != 0 {
        return Err(MoonclipError::Config(format!(
            "fp64 data length {} is not a multiple of 8",
            data.len()
        )));
    }
    Ok(data
        .par_chunks(8)
        .flat_map_iter(|c| {
            let v = f64::from_le_bytes(c.try_into().expect("8 bytes"));
            (v as f32).to_le_bytes()
        })
        .collect())
}

/// Cast fp32 bytes back to fp64 bytes.
pub fn fp32_to_fp64(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() % 4 != 0 {
        return Err(MoonclipError::Config(format!(
            "fp32 data length {} is not a multiple of 4",
            data.len()
        )));
    }
    Ok(data
        .par_chunks(4)
        .flat_map_iter(|c| {
            let v = f32::from_le_bytes(c.try_into().expect("4 bytes"));
            (v as f64).to_le_bytes()
        })
        .collect())
}

// ─── fp32 ↔ float8 ──────────────────────────────────────────────────

/// The shape of one float8 layout.
///
/// Two formats is enough to justify describing them instead of writing two
/// near-identical conversion routines: the pair differ only in where the
/// exponent field stops, and hand-unrolling that twice is how the fp16 path
/// grew its own subnormal bug elsewhere.
#[derive(Debug)]
pub struct Fp8Format {
    /// Name as stored in the manifest, matching torch's spelling.
    pub name: &'static str,
    exp_bits: u32,
    mant_bits: u32,
    bias: i32,
    /// e4m3fn spends the pattern an IEEE layout would give to infinity on a
    /// NaN instead — that is what the `fn` in the name means — so its top
    /// exponent field still holds ordinary finite numbers.
    has_inf: bool,
    /// Largest representable finite magnitude.
    pub max_finite: f32,
    /// Value of one step in the subnormal range: `2^(1 - bias) / 2^mant_bits`.
    subnormal_step: f32,
    /// Bit pattern (sign cleared) for NaN.
    nan_bits: u8,
    /// Bit pattern (sign cleared) for infinity. Meaningless when `has_inf`.
    inf_bits: u8,
    /// Bit pattern (sign cleared) of [`Self::max_finite`].
    max_finite_bits: u8,
    /// Largest exponent field that still denotes a finite number.
    max_normal_exp: i32,
    /// Largest mantissa field at `max_normal_exp` that is still finite.
    max_normal_mant: u32,
}

/// `float8_e4m3fn`: the weight-side format. Range ±448, four significant bits.
pub static E4M3FN: Fp8Format = Fp8Format {
    name: "float8_e4m3fn",
    exp_bits: 4,
    mant_bits: 3,
    bias: 7,
    has_inf: false,
    max_finite: 448.0,
    subnormal_step: 1.0 / 512.0, // 2^-6 / 2^3
    nan_bits: 0x7F,
    inf_bits: 0x7F, // no infinity to name; NaN is what an Inf becomes
    max_finite_bits: 0x7E,
    max_normal_exp: 15,
    max_normal_mant: 6, // (15, 7) is the NaN pattern, not a number
};

/// `float8_e5m2`: the gradient-side format. Range ±57344, three significant
/// bits, and IEEE-shaped — it keeps both infinities.
pub static E5M2: Fp8Format = Fp8Format {
    name: "float8_e5m2",
    exp_bits: 5,
    mant_bits: 2,
    bias: 15,
    has_inf: true,
    max_finite: 57344.0,
    subnormal_step: 1.0 / 65536.0, // 2^-14 / 2^2
    nan_bits: 0x7E,
    inf_bits: 0x7C,
    max_finite_bits: 0x7B,
    max_normal_exp: 30, // 31 is the Inf/NaN exponent
    max_normal_mant: 3,
};

/// Shift `value` right by `n` bits, rounding what falls off to nearest with
/// ties to even — the same rule the fp32→bf16 path uses, factored out because
/// float8 needs it at three different shift amounts.
#[inline]
fn shift_round_even(value: u32, n: u32) -> u32 {
    if n == 0 {
        return value;
    }
    if n > 31 {
        // Everything is below half a unit in the last place; nothing rounds up.
        return 0;
    }
    let keep = value >> n;
    let rem = value & ((1u32 << n) - 1);
    let half = 1u32 << (n - 1);
    if rem > half || (rem == half && (keep & 1) == 1) {
        keep + 1
    } else {
        keep
    }
}

/// fp32 bit pattern → float8 bit pattern, round-to-nearest-even.
///
/// NaN survives as NaN, for the reason spelled out on
/// [`fp32_bits_to_bf16_bits`]: a checkpoint is the last place a diverged run
/// should be laundered into finite-looking numbers. An infinity going into
/// e4m3fn becomes NaN rather than 448, because e4m3fn has no infinity and 448
/// is an entirely plausible weight — saturating there would hide the very
/// thing worth seeing. This is also what torch's own cast does.
///
/// Overflow that comes from *rounding* is different and does saturate: every
/// value reaching the arithmetic below is finite, so a mantissa carrying past
/// the top of the range is one ulp of rounding error, not a diverged tensor.
/// Turning the largest weight of every tensor into a NaN, which is what torch
/// does there, would be a poor trade for matching it.
///
/// That saturation is the *only* place this disagrees with torch. Checked on
/// torch 2.12 over all 256 patterns of both formats and ~22k values chosen to
/// sit on the ties, the subnormals and the specials: every in-range value
/// encodes to the identical byte, and the tables in [`fp8_decode_table`] match
/// `.view(torch.float8_*).float()` exactly. Values above `max_finite` are the
/// remainder, and Moonclip never produces one — the scale puts the data in
/// range before this is called.
#[inline]
fn fp32_bits_to_fp8_bits(bits: u32, fmt: &Fp8Format) -> u8 {
    let sign = ((bits >> 24) & 0x80) as u8;
    let abs = bits & 0x7FFF_FFFF;

    if abs > 0x7F80_0000 {
        return sign | fmt.nan_bits;
    }
    if abs == 0x7F80_0000 {
        return sign | if fmt.has_inf { fmt.inf_bits } else { fmt.nan_bits };
    }
    if abs == 0 {
        return sign; // ±0
    }

    let exp = ((abs >> 23) as i32) - 127; // unbiased fp32 exponent
    let mant = abs & 0x007F_FFFF;
    let new_exp = exp + fmt.bias;
    let drop = 23 - fmt.mant_bits;
    // fp32 is normal here (abs != 0 and exp field != 255), so the implicit
    // leading one is really there. Subnormal fp32 inputs have exp == -127,
    // which lands far below `new_exp >= 1` and takes the subnormal path,
    // where treating the significand as `1.mant` scaled by 2^-126 is the
    // convention that makes the shift arithmetic come out right.
    let significand = 0x0080_0000u32 | mant;

    if new_exp >= 1 {
        let rounded = shift_round_even(significand, drop);
        // Rounding can carry out of the mantissa field and into the exponent.
        let (e, m) = if rounded >> (fmt.mant_bits + 1) != 0 {
            (new_exp + 1, 0u32)
        } else {
            (new_exp, rounded & ((1u32 << fmt.mant_bits) - 1))
        };
        if e > fmt.max_normal_exp || (e == fmt.max_normal_exp && m > fmt.max_normal_mant) {
            return sign | fmt.max_finite_bits;
        }
        return sign | ((e as u8) << fmt.mant_bits) | m as u8;
    }

    // Subnormal in the target, or below it. Shifting the significand down by
    // the extra `1 - new_exp` places puts it in units of `subnormal_step`.
    let shift = drop + (1 - new_exp) as u32;
    let m = shift_round_even(significand, shift);
    // If that rounds up to exactly 2^mant_bits, the bits that fall out are an
    // exponent field of 1 with a zero fraction — the smallest normal, which is
    // the right answer and needs no special case.
    sign | m as u8
}

/// Every float8 bit pattern's value, already multiplied by `scale`.
///
/// 256 entries built once per tensor, so decoding is a table index rather
/// than an exponent reconstruction per element.
fn fp8_decode_table(fmt: &Fp8Format, scale: f32) -> [f32; 256] {
    let mut table = [0f32; 256];
    let max_e = (1i32 << fmt.exp_bits) - 1;
    let mant_mask = (1u32 << fmt.mant_bits) - 1;

    for (b, slot) in table.iter_mut().enumerate() {
        let b = b as u8;
        let negative = b & 0x80 != 0;
        let e = ((b >> fmt.mant_bits) as i32) & max_e;
        let m = (b as u32) & mant_mask;

        let magnitude = if e == max_e && fmt.has_inf {
            if m == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        } else if e == max_e && !fmt.has_inf && m == mant_mask {
            // e4m3fn's single NaN pattern. Every other value at this exponent
            // is an ordinary finite number, which is why this is not simply
            // "the top exponent is special".
            f32::NAN
        } else if e == 0 {
            m as f32 * fmt.subnormal_step
        } else {
            (1.0 + m as f32 / (1u32 << fmt.mant_bits) as f32) * 2f32.powi(e - fmt.bias)
        };

        let signed = if negative { -magnitude } else { magnitude };
        // NaN * scale is still NaN and ±Inf * scale is still ±Inf, so folding
        // the scale in here costs the caller nothing and loses nothing.
        *slot = signed * scale;
    }
    table
}

/// Largest finite `|x|` in an fp32 buffer, or 0 if there is none.
///
/// Non-finite elements are skipped rather than poisoning the result: one Inf
/// in a diverged tensor would otherwise drive the scale to infinity and take
/// every other value in the tensor to zero with it. The Inf itself still
/// survives the cast as a NaN, so nothing is being hidden — only kept from
/// destroying its neighbours.
fn finite_amax(fp32: &[u8]) -> f32 {
    fp32.par_chunks(16384 * 4)
        .map(|chunk| {
            let mut local = 0f32;
            for q in chunk.chunks_exact(4) {
                let v = f32::from_le_bytes([q[0], q[1], q[2], q[3]]);
                if v.is_finite() {
                    let a = v.abs();
                    if a > local {
                        local = a;
                    }
                }
            }
            local
        })
        .reduce(|| 0f32, f32::max)
}

/// The per-tensor dequantization multiplier for fp32 bytes about to be stored
/// as `fmt`.
///
/// **Direction matters and is easy to get backwards.** This is the number the
/// stored values are multiplied *by* on the way out — `original ≈ stored ×
/// scale` — not the one they were divided by on the way in. Storing the
/// reciprocal instead would still round-trip in a test that only ever uses
/// one of the two, and would be off by `scale²` everywhere else.
///
/// A tensor that is all zeros, or whose finite values are so small that the
/// quotient is not a normal float, gets a scale of 1.0: there is nothing to
/// spread across the range, and a zero or subnormal multiplier would take the
/// whole tensor with it.
pub fn fp8_scale(fp32: &[u8], fmt: &Fp8Format) -> f32 {
    let amax = finite_amax(fp32);
    let scale = amax / fmt.max_finite;
    if scale.is_normal() {
        scale
    } else {
        1.0
    }
}

/// Quantize fp32 bytes to float8 against a scale from [`fp8_scale`].
pub fn fp32_to_fp8(data: &[u8], fmt: &Fp8Format, scale: f32) -> Result<Vec<u8>> {
    if data.len() % 4 != 0 {
        return Err(MoonclipError::Config(format!(
            "fp32 data length {} is not a multiple of 4",
            data.len()
        )));
    }

    let n_elements = data.len() / 4;
    let mut out = vec![0u8; n_elements];
    // `fp8_scale` guarantees a normal float, so the reciprocal is finite and
    // multiplying is safe. Any residual error at the very top of the range is
    // one ulp, which the saturation in `fp32_bits_to_fp8_bits` absorbs.
    let inv = 1.0f32 / scale;

    let chunk_size = 16384;
    let src_chunks: Vec<&[u8]> = data.chunks(chunk_size * 4).collect();
    let dst_chunks: Vec<&mut [u8]> = out.chunks_mut(chunk_size).collect();

    src_chunks
        .into_par_iter()
        .zip(dst_chunks)
        .for_each(|(src, dst)| {
            for (slot, q) in dst.iter_mut().zip(src.chunks_exact(4)) {
                let v = f32::from_le_bytes([q[0], q[1], q[2], q[3]]);
                *slot = fp32_bits_to_fp8_bits((v * inv).to_bits(), fmt);
            }
        });

    Ok(out)
}

/// Dequantize float8 bytes back to fp32 against the scale they were stored
/// with.
pub fn fp8_to_fp32(data: &[u8], fmt: &Fp8Format, scale: f32) -> Result<Vec<u8>> {
    let table = fp8_decode_table(fmt, scale);
    let mut out = vec![0u8; data.len() * 4];

    let chunk_size = 16384;
    let src_chunks: Vec<&[u8]> = data.chunks(chunk_size).collect();
    let dst_chunks: Vec<&mut [u8]> = out.chunks_mut(chunk_size * 4).collect();

    src_chunks
        .into_par_iter()
        .zip(dst_chunks)
        .for_each(|(src, dst)| {
            for (i, &b) in src.iter().enumerate() {
                let bytes = table[b as usize].to_le_bytes();
                dst[i * 4..i * 4 + 4].copy_from_slice(&bytes);
            }
        });

    Ok(out)
}

// ─── Generic dispatch ───────────────────────────────────────────────

/// Bring any supported float dtype up to fp32, which every conversion goes
/// through. Widening first costs one pass and removes the combinatorics: five
/// float dtypes would otherwise need twenty direct paths, and the ones nobody
/// wrote were exactly where the silent no-ops lived.
fn widen_to_fp32(data: &[u8], src: &str) -> Result<Vec<u8>> {
    match src {
        "float32" => Ok(data.to_vec()),
        "bfloat16" => bf16_to_fp32(data),
        "float16" => fp16_to_fp32(data),
        "float64" => fp64_to_fp32(data),
        other => Err(MoonclipError::Config(format!(
            "'{other}' is not a float dtype this build can cast"
        ))),
    }
}

/// The other half: fp32 down to the stored dtype.
fn narrow_from_fp32(data: &[u8], dst: &str) -> Result<Vec<u8>> {
    match dst {
        "float32" => Ok(data.to_vec()),
        "bfloat16" => fp32_to_bf16(data),
        "float16" => fp32_to_fp16(data),
        "float64" => fp32_to_fp64(data),
        other => Err(MoonclipError::Config(format!(
            "'{other}' is not a float dtype this build can cast"
        ))),
    }
}

/// Cast tensor bytes from `src_dtype` to `target_dtype`.
///
/// Returns `(casted_bytes, new_dtype_string, quant_scale)`. The scale is
/// `Some` only for a float8 target, where the stored bytes are meaningless
/// without it; every other cast is self-describing and returns `None`.
/// If no cast is needed, returns the original data.
pub fn cast_tensor(
    data: &[u8],
    src_dtype: &str,
    target: &DType,
) -> Result<(Vec<u8>, String, Option<f32>)> {
    if *target == DType::None {
        return Ok((data.to_vec(), src_dtype.to_string(), None));
    }

    let target_name = target.to_str();
    if src_dtype == target_name || !is_castable_float(src_dtype) {
        // Non-float data is stored as it is: an int tensor has no business
        // being reinterpreted because a float setting was chosen. A tensor
        // that is already float8 lands here too — see `is_castable_float`.
        return Ok((data.to_vec(), src_dtype.to_string(), None));
    }

    // float8 is the one target that needs a scale chosen from the data, so it
    // cannot go through `narrow_from_fp32` with the rest.
    if let Some(fmt) = target.fp8_format() {
        let wide: Cow<[u8]> = if src_dtype == "float32" {
            Cow::Borrowed(data)
        } else {
            Cow::Owned(widen_to_fp32(data, src_dtype)?)
        };
        let scale = fp8_scale(&wide, fmt);
        return Ok((
            fp32_to_fp8(&wide, fmt, scale)?,
            target_name.to_string(),
            Some(scale),
        ));
    }

    // The common pair keeps its dedicated single pass.
    if src_dtype == "float32" {
        return Ok((
            narrow_from_fp32(data, target_name)?,
            target_name.to_string(),
            None,
        ));
    }

    let wide = widen_to_fp32(data, src_dtype)?;
    Ok((
        narrow_from_fp32(&wide, target_name)?,
        target_name.to_string(),
        None,
    ))
}

/// Cast tensor bytes back from `stored_dtype` to `original_dtype`.
///
/// `scale` is the value recorded on the entry by [`cast_tensor`]; it is
/// required when `stored_dtype` is a float8 and ignored otherwise.
pub fn uncast_tensor(
    data: &[u8],
    stored_dtype: &str,
    original_dtype: &str,
    scale: Option<f32>,
) -> Result<Vec<u8>> {
    if stored_dtype == original_dtype {
        return Ok(data.to_vec());
    }

    // float8 is checked before the guard below, not after. It is deliberately
    // absent from `is_castable_float`, so falling through would return the
    // quantized bytes unchanged — a tensor a quarter of its stated size,
    // reinterpreted as whatever dtype the caller expected, with no error.
    if let Some(fmt) = fp8_format_for(stored_dtype) {
        let scale = scale.ok_or_else(|| {
            MoonclipError::Config(format!(
                "tensor stored as {stored_dtype} carries no quant_scale; its \
                 bytes cannot be turned back into {original_dtype} without it"
            ))
        })?;
        let fp32 = fp8_to_fp32(data, fmt, scale)?;
        if original_dtype == "float32" {
            return Ok(fp32);
        }
        return narrow_from_fp32(&fp32, original_dtype);
    }

    if !is_castable_float(stored_dtype) || !is_castable_float(original_dtype) {
        // Nothing was cast on the way in, so there is nothing to undo.
        return Ok(data.to_vec());
    }

    if stored_dtype == "float32" {
        return narrow_from_fp32(data, original_dtype);
    }
    if original_dtype == "float32" {
        return widen_to_fp32(data, stored_dtype);
    }

    let wide = widen_to_fp32(data, stored_dtype)?;
    narrow_from_fp32(&wide, original_dtype)
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

        let (casted, dtype, _) = cast_tensor(&data, "float32", &DType::BFloat16).unwrap();
        assert_eq!(dtype, "bfloat16");
        assert_eq!(casted.len(), 2);

        let (back, dtype2, _) = cast_tensor(&casted, "bfloat16", &DType::Float32).unwrap();
        assert_eq!(dtype2, "float32");
        assert_eq!(back.len(), 4);
    }

    /// A typo in `save_dtype` used to mean "do not cast": full-precision
    /// checkpoints, twice the configured size, and nothing said so.
    #[test]
    fn an_unknown_save_dtype_is_refused() {
        assert!(DType::parse("bfloat").is_err());
        // "float8" was in this list until 0.0.9, when it stopped being a typo
        // and became a format. "float4" stands in for the next one.
        assert!(DType::parse("float4").is_err());
        assert!(DType::parse("fp8_e3m4").is_err());
        assert_eq!(DType::parse("bf16").unwrap(), DType::BFloat16);
        assert_eq!(DType::parse("BF16").unwrap(), DType::BFloat16);
        assert_eq!(DType::parse("").unwrap(), DType::None);
        assert_eq!(DType::parse("none").unwrap(), DType::None);
    }

    /// Bare "fp8" has to keep meaning e4m3: a run that stored weights under it
    /// and a later build that read it as e5m2 would differ by a factor of 128
    /// in the exponent, silently, on every tensor.
    #[test]
    fn the_float8_spellings_land_on_the_format_they_name() {
        for name in ["fp8", "float8", "fp8_e4m3", "float8_e4m3fn", "FP8"] {
            assert_eq!(DType::parse(name).unwrap(), DType::Float8E4M3, "{name}");
        }
        for name in ["fp8_e5m2", "float8_e5m2"] {
            assert_eq!(DType::parse(name).unwrap(), DType::Float8E5M2, "{name}");
        }
        assert_eq!(DType::Float8E4M3.to_str(), "float8_e4m3fn");
        assert_eq!(DType::Float8E5M2.to_str(), "float8_e5m2");
        assert_eq!(DType::Float8E4M3.element_size(), 1);
        assert_eq!(DType::Float8E5M2.element_size(), 1);
    }

    /// `is_castable_float` decides whether the save path asks for a cast, and
    /// `cast_tensor` decides whether one happens. They have to agree, or a
    /// tensor is stored in a dtype nobody chose.
    #[test]
    fn everything_declared_castable_is_actually_cast() {
        for dtype in ["float32", "float16", "bfloat16", "float64"] {
            let element = match dtype {
                "float32" => 4,
                "float64" => 8,
                _ => 2,
            };
            let data = vec![0u8; element * 4];
            let (out, stored, _) = cast_tensor(&data, dtype, &DType::BFloat16).unwrap();

            if is_castable_float(dtype) {
                assert_eq!(stored, "bfloat16", "{dtype} was declared castable");
                assert_eq!(out.len(), 8, "{dtype} kept its original width");
            } else {
                assert_eq!(stored, dtype, "{dtype} is not declared castable");
            }
        }
    }

    /// Every float dtype reaches every other one, and comes back. The pairs
    /// that used to be missing — anything with float64 at either end, and
    /// fp16 ↔ bf16 — were silent no-ops rather than errors.
    #[test]
    fn every_float_pair_round_trips() {
        let original: Vec<f32> = vec![1.0, -2.5, 0.0, 1024.0];

        for src in ["float32", "float64", "float16", "bfloat16"] {
            // Start from fp32 and produce the source dtype's bytes.
            let fp32: Vec<u8> = original.iter().flat_map(|v| v.to_le_bytes()).collect();
            let src_bytes = narrow_from_fp32(&fp32, src).unwrap();

            for target in [DType::BFloat16, DType::Float16, DType::Float32] {
                let (stored, stored_dtype, _) =
                    cast_tensor(&src_bytes, src, &target).unwrap();
                assert_eq!(
                    stored_dtype,
                    target.to_str(),
                    "{src} → {} did not happen",
                    target.to_str()
                );

                let back = uncast_tensor(&stored, &stored_dtype, src, None).unwrap();
                assert_eq!(
                    back.len(),
                    src_bytes.len(),
                    "{src} → {} → {src} changed the width",
                    target.to_str()
                );

                // bf16 keeps 8 mantissa bits, so 1024.0 and the rest survive
                // exactly; this is about the path existing, not precision.
                let widened = widen_to_fp32(&back, src).unwrap();
                let values: Vec<f32> = widened
                    .chunks(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                assert_eq!(
                    values, original,
                    "{src} → {} → {src} changed the values",
                    target.to_str()
                );
            }
        }
    }

    #[test]
    fn cast_nonfloat_noop() {
        let data = vec![1u8, 2, 3, 4];
        let (result, dtype, _) = cast_tensor(&data, "int32", &DType::BFloat16).unwrap();
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
        let (casted, stored_dtype, _) = cast_tensor(&data, "float32", &DType::BFloat16).unwrap();
        let uncasted = uncast_tensor(&casted, &stored_dtype, "float32", None).unwrap();
        let val = f32::from_le_bytes([uncasted[0], uncasted[1], uncasted[2], uncasted[3]]);
        assert!((val - 3.6).abs() < 0.05);
    }

    // ─── float8 ─────────────────────────────────────────────────────

    fn bytes_of(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn floats_of(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|q| f32::from_le_bytes([q[0], q[1], q[2], q[3]]))
            .collect()
    }

    /// The anchors below are what `torch.tensor([...]).to(torch.float8_*)`
    /// produces on torch 2.12, transcribed rather than derived. The whole
    /// point of a float8 in a checkpoint is that some other tool can read it,
    /// so agreeing with torch is the specification, not an implementation
    /// detail — and a table reconstructed from the same arithmetic that
    /// encodes would agree with itself no matter how wrong both were.
    #[test]
    fn float8_encodes_to_the_bytes_torch_produces() {
        let values = [1.0f32, 2.0, 0.5, -1.0, 3.5, -0.0];
        assert_eq!(
            fp32_to_fp8(&bytes_of(&values), &E4M3FN, 1.0).unwrap(),
            vec![0x38, 0x40, 0x30, 0xB8, 0x46, 0x80]
        );
        assert_eq!(
            fp32_to_fp8(&bytes_of(&values), &E5M2, 1.0).unwrap(),
            vec![0x3C, 0x40, 0x38, 0xBC, 0x43, 0x80]
        );
    }

    /// The other direction, at the four places a layout is easiest to get
    /// wrong: the top of the range, the smallest normal, the smallest
    /// subnormal, and whatever the top exponent means.
    #[test]
    fn float8_decodes_to_the_values_torch_reports() {
        let all: Vec<u8> = (0..=255u8).collect();

        // Written as exact powers of two rather than the decimals torch
        // prints (0.015625, 0.001953125, 6.103515625e-05, 1.52587890625e-05):
        // same values, and the exponent is the thing being asserted.
        let e4 = floats_of(&fp8_to_fp32(&all, &E4M3FN, 1.0).unwrap());
        assert_eq!(e4[0x7E], 448.0, "largest finite e4m3fn");
        assert_eq!(e4[0x08], 2f32.powi(-6), "smallest normal e4m3fn");
        assert_eq!(e4[0x01], 2f32.powi(-9), "smallest subnormal e4m3fn");
        assert!(e4[0x7F].is_nan(), "e4m3fn has exactly one NaN pattern");
        assert!(e4[0xFF].is_nan());
        assert!(e4.iter().all(|v| !v.is_infinite()), "e4m3fn has no infinity");

        let e5 = floats_of(&fp8_to_fp32(&all, &E5M2, 1.0).unwrap());
        assert_eq!(e5[0x7B], 57344.0, "largest finite e5m2");
        assert_eq!(e5[0x04], 2f32.powi(-14), "smallest normal e5m2");
        assert_eq!(e5[0x01], 2f32.powi(-16), "smallest subnormal e5m2");
        assert_eq!(e5[0x7C], f32::INFINITY);
        assert_eq!(e5[0xFC], f32::NEG_INFINITY);
        for p in [0x7D, 0x7E, 0x7F] {
            assert!(e5[p].is_nan(), "e5m2 pattern {p:#04x} is a NaN");
        }
    }

    /// Encode and decode have to be inverses on the values that *are*
    /// representable, or the format loses precision it never had to.
    #[test]
    fn every_representable_float8_value_re_encodes_to_its_own_pattern() {
        let all: Vec<u8> = (0..=255u8).collect();
        for fmt in [&E4M3FN, &E5M2] {
            let decoded = fp8_to_fp32(&all, fmt, 1.0).unwrap();
            let values = floats_of(&decoded);
            let re = fp32_to_fp8(&decoded, fmt, 1.0).unwrap();
            for (pattern, got) in re.iter().enumerate() {
                let value = values[pattern];
                if value.is_nan() {
                    // The payload is not preserved; being a NaN is.
                    assert!(
                        floats_of(&fp8_to_fp32(&[*got], fmt, 1.0).unwrap())[0].is_nan(),
                        "{} pattern {pattern:#04x} stopped being a NaN",
                        fmt.name
                    );
                    continue;
                }
                assert_eq!(
                    *got, pattern as u8,
                    "{} pattern {pattern:#04x} ({value}) re-encoded to {got:#04x}",
                    fmt.name
                );
            }
        }
    }

    /// The direction of `quant_scale`, which is the one thing here a
    /// round-trip test cannot catch on its own: storing the reciprocal
    /// instead would pass any test that uses the same wrong number twice, and
    /// be wrong by `scale²` for anyone reading the manifest.
    #[test]
    fn the_scale_multiplies_stored_values_back_up_to_the_original_range() {
        // Values far outside float8's own range: only a correctly applied
        // scale brings them back.
        let values: Vec<f32> = (0..64).map(|i| 1000.0 + i as f32 * 250.0).collect();
        let data = bytes_of(&values);

        let scale = fp8_scale(&data, &E4M3FN);
        assert!(
            (scale - values.last().unwrap() / 448.0).abs() < 1e-6,
            "scale should be amax/max_finite, got {scale}"
        );

        let stored = fp32_to_fp8(&data, &E4M3FN, scale).unwrap();
        let back = floats_of(&fp8_to_fp32(&stored, &E4M3FN, scale).unwrap());

        for (original, got) in values.iter().zip(&back) {
            let relative = (got - original).abs() / original.abs();
            assert!(
                relative < 0.07,
                "{original} came back as {got} (relative error {relative})"
            );
        }
        // The largest value sits exactly at the top of the range and should
        // survive intact rather than saturating away.
        assert!((back[63] - values[63]).abs() / values[63] < 1e-6);
    }

    /// e4m3fn keeps four significant bits, so a few percent is the floor on
    /// what a round trip can promise. Asserted as a band, not a bound: a test
    /// that only checked the error was *small* would still pass if
    /// quantization silently stopped happening.
    #[test]
    fn a_float8_round_trip_loses_what_the_format_costs_and_no_more() {
        let values: Vec<f32> = (0..4096)
            .map(|i| ((i as f32 * 0.7).sin()) * 0.05) // weight-shaped
            .collect();
        let data = bytes_of(&values);
        let scale = fp8_scale(&data, &E4M3FN);
        let stored = fp32_to_fp8(&data, &E4M3FN, scale).unwrap();
        assert_eq!(stored.len(), values.len(), "one byte per element");

        let back = floats_of(&fp8_to_fp32(&stored, &E4M3FN, scale).unwrap());
        let amax = values.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = values
            .iter()
            .zip(&back)
            .map(|(a, b)| (a - b).abs() / amax)
            .fold(0f32, f32::max);

        assert!(worst < 0.05, "relative-to-amax error {worst} is too large");
        assert!(worst > 0.0, "nothing was quantized at all");
    }

    /// A tensor of zeros has no amax to divide by. Dividing anyway produces a
    /// NaN scale, and a NaN scale turns every element into a NaN on the way
    /// back out — from data that was perfectly fine.
    #[test]
    fn an_all_zero_tensor_survives_float8() {
        let data = bytes_of(&vec![0.0f32; 256]);
        let scale = fp8_scale(&data, &E4M3FN);
        assert_eq!(scale, 1.0);

        let stored = fp32_to_fp8(&data, &E4M3FN, scale).unwrap();
        let back = floats_of(&fp8_to_fp32(&stored, &E4M3FN, scale).unwrap());
        assert!(back.iter().all(|v| *v == 0.0), "zeros came back as {back:?}");
    }

    /// One diverged element must not take its neighbours with it. With the
    /// infinity counted, amax is infinite, the scale is infinite, and every
    /// finite value in the tensor quantizes to zero — so a single bad element
    /// erases the whole tensor.
    #[test]
    fn an_infinity_does_not_erase_the_rest_of_the_tensor() {
        let mut values = vec![0.25f32; 64];
        values[10] = f32::INFINITY;
        values[20] = f32::NAN;
        let data = bytes_of(&values);

        let scale = fp8_scale(&data, &E4M3FN);
        assert!(scale.is_finite() && scale > 0.0, "scale was {scale}");

        let stored = fp32_to_fp8(&data, &E4M3FN, scale).unwrap();
        let back = floats_of(&fp8_to_fp32(&stored, &E4M3FN, scale).unwrap());

        for (i, v) in back.iter().enumerate() {
            match i {
                // e4m3fn has no infinity, so an Inf becomes a NaN — still
                // unmistakably "something went wrong here", and what torch
                // does too. What it must not become is 448.
                10 | 20 => assert!(v.is_nan(), "element {i} came back as {v}"),
                _ => assert!(
                    (v - 0.25).abs() < 1e-6,
                    "element {i} came back as {v}, not 0.25"
                ),
            }
        }
    }

    /// e5m2 does have infinities, and an Inf that arrives as one should leave
    /// as one rather than being rounded down to 57344.
    #[test]
    fn e5m2_keeps_the_infinity_it_has_room_for() {
        let stored =
            fp32_to_fp8(&bytes_of(&[f32::INFINITY, f32::NEG_INFINITY]), &E5M2, 1.0).unwrap();
        assert_eq!(stored, vec![0x7C, 0xFC]);
        let back = floats_of(&fp8_to_fp32(&stored, &E5M2, 1.0).unwrap());
        assert_eq!(back, vec![f32::INFINITY, f32::NEG_INFINITY]);
    }

    /// A tensor already in float8 is not routed through fp32 and back: it is
    /// stored as it arrived, with no scale to record and nothing to undo.
    #[test]
    fn a_tensor_that_is_already_float8_is_left_alone() {
        let data: Vec<u8> = (0..=255u8).collect();
        for target in [DType::BFloat16, DType::Float32, DType::Float8E5M2] {
            let (out, dtype, scale) = cast_tensor(&data, "float8_e4m3fn", &target).unwrap();
            assert_eq!(out, data, "bytes changed with target {target:?}");
            assert_eq!(dtype, "float8_e4m3fn");
            assert_eq!(scale, None);
        }
    }

    /// float8 is absent from `is_castable_float`, so the guard in
    /// `uncast_tensor` would fall through and hand back the quantized bytes
    /// as if they were the original tensor: a quarter of the expected length,
    /// reinterpreted as fp32, and no error anywhere.
    #[test]
    fn float8_bytes_without_a_scale_are_refused_not_returned_raw() {
        let stored = vec![0x38u8; 16];
        let err = uncast_tensor(&stored, "float8_e4m3fn", "float32", None).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("quant_scale"),
            "the error must name what is missing, got: {message}"
        );
    }

    /// Every float dtype the save path will cast from has to reach float8,
    /// for the reason `everything_declared_castable_is_actually_cast` exists:
    /// a source that quietly fails to convert is stored at its old size under
    /// a setting chosen to shrink it.
    #[test]
    fn every_castable_source_reaches_float8_and_comes_back() {
        for src in ["float32", "float16", "bfloat16", "float64"] {
            let fp32 = bytes_of(&[0.25f32, -0.5, 1.0, 0.125]);
            let src_bytes = if src == "float32" {
                fp32.clone()
            } else {
                narrow_from_fp32(&fp32, src).unwrap()
            };

            let (stored, dtype, scale) =
                cast_tensor(&src_bytes, src, &DType::Float8E4M3).unwrap();
            assert_eq!(dtype, "float8_e4m3fn", "source {src}");
            assert_eq!(stored.len(), 4, "source {src}: one byte per element");
            let scale = scale.expect("a float8 cast must record its scale");

            let back = uncast_tensor(&stored, &dtype, src, Some(scale)).unwrap();
            assert_eq!(back.len(), src_bytes.len(), "source {src}: width restored");

            // These values are exactly representable in e4m3fn once scaled by
            // 1/448, so the trip costs nothing and any difference is a defect.
            let as_fp32 = floats_of(&widen_to_fp32(&back, src).unwrap());
            assert_eq!(as_fp32, vec![0.25, -0.5, 1.0, 0.125], "source {src}");
        }
    }
}
