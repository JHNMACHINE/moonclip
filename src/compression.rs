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
}
