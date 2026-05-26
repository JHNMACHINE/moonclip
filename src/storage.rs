use crate::error::{Result, RevolverError};
use std::path::{Path, PathBuf};

// ─── Constants ──────────────────────────────────────────────────────

/// Default SSD page size for write alignment.
pub const DEFAULT_PAGE_SIZE: usize = 4096;

// ─── Backend trait ──────────────────────────────────────────────────

/// Abstraction over where checkpoint shards are persisted.
pub trait StorageBackend: Send + Sync {
    /// Write bytes to the given relative path.
    fn put(&self, rel_path: &str, data: &[u8]) -> Result<()>;

    /// Read bytes from the given relative path.
    fn get(&self, rel_path: &str) -> Result<Vec<u8>>;

    /// Check if a path exists.
    fn exists(&self, rel_path: &str) -> Result<bool>;

    /// Delete a file at the given relative path.
    fn delete(&self, rel_path: &str) -> Result<()>;

    /// List all files under a relative prefix.
    fn list(&self, prefix: &str) -> Result<Vec<String>>;
}

// ─── Page-aligned padding ───────────────────────────────────────────

/// Pad data to a multiple of `page_size` bytes.
///
/// Appends zero bytes so the total length is a multiple of `page_size`.
/// This prevents SSD write amplification from partial page writes.
///
/// The original data length is preserved in the manifest (`compressed_size`),
/// so on read we truncate back to the original size.
pub fn pad_to_page(data: &[u8], page_size: usize) -> Vec<u8> {
    if page_size == 0 || data.len() % page_size == 0 {
        return data.to_vec();
    }
    let padded_len = ((data.len() + page_size - 1) / page_size) * page_size;
    let mut padded = Vec::with_capacity(padded_len);
    padded.extend_from_slice(data);
    padded.resize(padded_len, 0u8);
    padded
}

// ─── Local filesystem ───────────────────────────────────────────────

pub struct LocalStorage {
    root: PathBuf,
    /// SSD page size for write alignment. 0 = no padding.
    page_size: usize,
}

impl LocalStorage {
    /// Create with default 4KB page alignment.
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        Self::with_page_size(root, DEFAULT_PAGE_SIZE)
    }

    /// Create with no page alignment (for testing or non-SSD storage).
    pub fn new_unaligned(root: impl AsRef<Path>) -> Result<Self> {
        Self::with_page_size(root, 0)
    }

    /// Create with a custom page size.
    pub fn with_page_size(root: impl AsRef<Path>, page_size: usize) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root)?;
        Ok(LocalStorage { root, page_size })
    }

    fn full_path(&self, rel: &str) -> PathBuf {
        self.root.join(rel)
    }
}

impl StorageBackend for LocalStorage {
    fn put(&self, rel_path: &str, data: &[u8]) -> Result<()> {
        let path = self.full_path(rel_path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Pad to page boundary for SSD longevity
        let write_data = if self.page_size > 0 {
            pad_to_page(data, self.page_size)
        } else {
            data.to_vec()
        };

        // Atomic write: temp file → rename
        let dir = path.parent().unwrap_or(Path::new("."));
        let tmp = tempfile::NamedTempFile::new_in(dir)?;
        std::fs::write(tmp.path(), &write_data)?;
        tmp.persist(&path).map_err(|e| {
            RevolverError::Storage(format!("Failed to persist {}: {}", path.display(), e))
        })?;
        Ok(())
    }

    fn get(&self, rel_path: &str) -> Result<Vec<u8>> {
        let path = self.full_path(rel_path);
        if !path.exists() {
            return Err(RevolverError::NotFound(rel_path.to_string()));
        }
        Ok(std::fs::read(&path)?)
    }

    fn exists(&self, rel_path: &str) -> Result<bool> {
        Ok(self.full_path(rel_path).exists())
    }

    fn delete(&self, rel_path: &str) -> Result<()> {
        let path = self.full_path(rel_path);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        Ok(())
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let search_root = self.full_path(prefix);
        if !search_root.exists() {
            return Ok(Vec::new());
        }

        let mut files = Vec::new();
        for entry in walkdir::WalkDir::new(&search_root)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            if let Ok(rel) = entry.path().strip_prefix(&self.root) {
                files.push(rel.to_string_lossy().to_string());
            }
        }
        Ok(files)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStorage::new(dir.path()).unwrap();

        let data = b"checkpoint bytes here";
        store.put("snap_001/shard_0.bin", data).unwrap();

        assert!(store.exists("snap_001/shard_0.bin").unwrap());
        assert!(!store.exists("snap_001/shard_1.bin").unwrap());

        // Read back — may be padded, but first N bytes match
        let read = store.get("snap_001/shard_0.bin").unwrap();
        assert!(read.len() >= data.len());
        assert_eq!(&read[..data.len()], &data[..]);

        let files = store.list("snap_001").unwrap();
        assert_eq!(files.len(), 1);

        store.delete("snap_001/shard_0.bin").unwrap();
        assert!(!store.exists("snap_001/shard_0.bin").unwrap());
    }

    #[test]
    fn page_alignment() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStorage::new(dir.path()).unwrap(); // 4KB aligned

        // 100 bytes → padded to 4096
        let data = vec![42u8; 100];
        store.put("test.bin", &data).unwrap();

        let on_disk = std::fs::read(dir.path().join("test.bin")).unwrap();
        assert_eq!(on_disk.len(), 4096);
        assert_eq!(&on_disk[..100], &data[..]);
        assert!(on_disk[100..].iter().all(|&b| b == 0));
    }

    #[test]
    fn page_alignment_exact_multiple() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStorage::new(dir.path()).unwrap();

        // Exactly 4096 bytes → no padding needed
        let data = vec![7u8; 4096];
        store.put("exact.bin", &data).unwrap();

        let on_disk = std::fs::read(dir.path().join("exact.bin")).unwrap();
        assert_eq!(on_disk.len(), 4096);
    }

    #[test]
    fn no_alignment() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStorage::new_unaligned(dir.path()).unwrap();

        let data = vec![42u8; 100];
        store.put("test.bin", &data).unwrap();

        let on_disk = std::fs::read(dir.path().join("test.bin")).unwrap();
        assert_eq!(on_disk.len(), 100); // No padding
    }

    #[test]
    fn pad_to_page_fn() {
        assert_eq!(pad_to_page(&[1, 2, 3], 4096).len(), 4096);
        assert_eq!(pad_to_page(&vec![0u8; 4096], 4096).len(), 4096);
        assert_eq!(pad_to_page(&vec![0u8; 4097], 4096).len(), 8192);
        assert_eq!(pad_to_page(&[1, 2, 3], 0).len(), 3); // 0 = no padding
    }
}
