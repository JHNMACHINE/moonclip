use crate::error::{Result, RevolverError};
use crate::manifest::CompressionAlgo;

/// Compress raw bytes according to the chosen algorithm.
pub fn compress(data: &[u8], algo: &CompressionAlgo) -> Result<Vec<u8>> {
    match algo {
        CompressionAlgo::None => Ok(data.to_vec()),
        CompressionAlgo::Zstd { level } => {
            zstd::encode_all(std::io::Cursor::new(data), *level)
                .map_err(|e| RevolverError::Compression(e.to_string()))
        }
    }
}

/// Decompress bytes produced by `compress`.
pub fn decompress(data: &[u8], algo: &CompressionAlgo) -> Result<Vec<u8>> {
    match algo {
        CompressionAlgo::None => Ok(data.to_vec()),
        CompressionAlgo::Zstd { .. } => {
            zstd::decode_all(std::io::Cursor::new(data))
                .map_err(|e| RevolverError::Compression(e.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_zstd() {
        let data = vec![42u8; 100_000];
        let algo = CompressionAlgo::Zstd { level: 3 };
        let compressed = compress(&data, &algo).unwrap();
        assert!(compressed.len() < data.len());
        let decompressed = decompress(&compressed, &algo).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn roundtrip_none() {
        let data = b"hello world".to_vec();
        let algo = CompressionAlgo::None;
        let compressed = compress(&data, &algo).unwrap();
        assert_eq!(data, compressed);
        let decompressed = decompress(&compressed, &algo).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn empty_data() {
        let algo = CompressionAlgo::Zstd { level: 3 };
        let compressed = compress(&[], &algo).unwrap();
        let decompressed = decompress(&compressed, &algo).unwrap();
        assert_eq!(decompressed, Vec::<u8>::new());
    }

    #[test]
    fn high_compression_level() {
        let data = vec![0u8; 50_000];
        let algo = CompressionAlgo::Zstd { level: 9 };
        let compressed = compress(&data, &algo).unwrap();
        let decompressed = decompress(&compressed, &algo).unwrap();
        assert_eq!(data, decompressed);
        assert!(compressed.len() < 100, "Repetitive data should compress to almost nothing");
    }

    #[test]
    fn random_data_still_roundtrips() {
        // Pseudo-random data doesn't compress well but should still roundtrip
        let data: Vec<u8> = (0..10_000).map(|i| ((i * 7 + 13) % 256) as u8).collect();
        let algo = CompressionAlgo::Zstd { level: 1 };
        let compressed = compress(&data, &algo).unwrap();
        let decompressed = decompress(&compressed, &algo).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn decompress_garbage_errors() {
        let algo = CompressionAlgo::Zstd { level: 3 };
        let result = decompress(b"not valid zstd data", &algo);
        assert!(result.is_err());
    }
}
