use xxhash_rust::xxh3::xxh3_128;

/// Compute xxHash3-128 hex digest of arbitrary bytes.
///
/// xxHash3 runs at >10 GB/s on modern CPUs — roughly 10× faster than SHA-256.
/// For checkpoint integrity (skip detection + corruption detection) we don't
/// need cryptographic strength; a 128-bit non-crypto hash is more than adequate.
pub fn hash_hex(data: &[u8]) -> String {
    format!("{:032x}", xxh3_128(data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let h1 = hash_hex(b"hello world");
        let h2 = hash_hex(b"hello world");
        assert_eq!(h1, h2);
    }

    #[test]
    fn different_inputs_different_hashes() {
        let h1 = hash_hex(b"hello");
        let h2 = hash_hex(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn empty_input() {
        let h = hash_hex(b"");
        assert_eq!(h.len(), 32); // 128-bit = 32 hex chars
    }

    #[test]
    fn large_input() {
        let data = vec![42u8; 10_000_000];
        let h = hash_hex(&data);
        assert_eq!(h.len(), 32);
    }
}
