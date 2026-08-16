use crate::error::{Result, MoonclipError};
use std::path::{Path, PathBuf};

// ─── Constants ──────────────────────────────────────────────────────

/// Default SSD page size for write alignment.
pub const DEFAULT_PAGE_SIZE: usize = 4096;

// ─── Backend trait ──────────────────────────────────────────────────

/// Abstraction over where checkpoint shards are persisted.
pub trait StorageBackend: Send + Sync {
    /// Write bytes to the given relative path.
    fn put(&self, rel_path: &str, data: &[u8]) -> Result<()>;

    /// Write the concatenation of `parts` to the given relative path.
    /// Equivalent to `put(rel_path, parts.concat())` but lets backends
    /// avoid materializing the concatenated buffer.
    fn put_parts(&self, rel_path: &str, parts: &[&[u8]]) -> Result<()> {
        let total: usize = parts.iter().map(|p| p.len()).sum();
        let mut buf = Vec::with_capacity(total);
        for p in parts {
            buf.extend_from_slice(p);
        }
        self.put(rel_path, &buf)
    }

    /// Read bytes from the given relative path.
    fn get(&self, rel_path: &str) -> Result<Vec<u8>>;

    /// Read up to `len` bytes starting at `offset`.
    /// May return fewer bytes if the object ends earlier.
    fn get_range(&self, rel_path: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        let data = self.get(rel_path)?;
        let start = (offset as usize).min(data.len());
        let end = start.saturating_add(len).min(data.len());
        Ok(data[start..end].to_vec())
    }

    /// Check if a path exists.
    fn exists(&self, rel_path: &str) -> Result<bool>;

    /// Delete a file at the given relative path.
    fn delete(&self, rel_path: &str) -> Result<()>;

    /// Remove a directory, once whatever it held has been deleted.
    ///
    /// Object stores have no directories — a "folder" there is just a shared
    /// key prefix that stops existing when the last object under it does — so
    /// the default is to do nothing, which is the correct behaviour for S3.
    /// Filesystem backends override it to drop the empty directory that
    /// `delete` leaves behind.
    fn remove_dir(&self, _rel_path: &str) -> Result<()> {
        Ok(())
    }

    /// List all files under a relative prefix.
    fn list(&self, prefix: &str) -> Result<Vec<String>>;
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
        self.put_parts(rel_path, &[data])
    }

    fn put_parts(&self, rel_path: &str, parts: &[&[u8]]) -> Result<()> {
        use std::io::Write;

        let path = self.full_path(rel_path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Atomic write: temp file → rename. Parts are streamed directly
        // to the file — no concatenated or padded copy of the data.
        let dir = path.parent().unwrap_or(Path::new("."));
        let mut tmp = tempfile::NamedTempFile::new_in(dir)?;
        {
            let file = tmp.as_file_mut();
            let mut total = 0usize;
            for part in parts {
                file.write_all(part)?;
                total += part.len();
            }
            // Pad to page boundary for SSD longevity
            if self.page_size > 0 && total % self.page_size != 0 {
                let pad = self.page_size - total % self.page_size;
                file.write_all(&vec![0u8; pad])?;
            }
            file.flush()?;
        }
        tmp.persist(&path).map_err(|e| {
            MoonclipError::Storage(format!("Failed to persist {}: {}", path.display(), e))
        })?;
        Ok(())
    }

    fn get(&self, rel_path: &str) -> Result<Vec<u8>> {
        let path = self.full_path(rel_path);
        if !path.exists() {
            return Err(MoonclipError::NotFound(rel_path.to_string()));
        }
        Ok(std::fs::read(&path)?)
    }

    fn get_range(&self, rel_path: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};

        let path = self.full_path(rel_path);
        let mut file = match std::fs::File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(MoonclipError::NotFound(rel_path.to_string()));
            }
            Err(e) => return Err(e.into()),
        };
        file.seek(SeekFrom::Start(offset))?;
        let mut buf = Vec::with_capacity(len);
        file.take(len as u64).read_to_end(&mut buf)?;
        Ok(buf)
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

    fn remove_dir(&self, rel_path: &str) -> Result<()> {
        let path = self.full_path(rel_path);
        if path.is_dir() {
            // `remove_dir`, never `remove_dir_all`: this only succeeds on an
            // empty directory, so a path derived wrongly can at worst fail. A
            // recursive delete here would turn a bug in the caller's path
            // arithmetic into lost checkpoints.
            let _ = std::fs::remove_dir(&path);
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
                // Join with '/' rather than handing back the OS separator.
                // Every other part of this abstraction speaks '/': `put`,
                // `get`, the manifest's `filename` fields, and S3 object keys.
                //
                // On Windows this returned `snapshots\s1\rank_0.pack`, and the
                // remote syncer fed it straight to S3 as an object key. The
                // upload succeeded and reported success — under a key nothing
                // would ever ask for again. The checkpoint was in the bucket
                // and unreachable, which is worse than not having uploaded it.
                let key = rel
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("/");
                files.push(key);
            }
        }
        Ok(files)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Listed keys must be usable as keys — by `get` here, and by whatever
    /// backend the syncer is pushing to.
    ///
    /// Regression: this returned native separators, so on Windows the remote
    /// syncer created S3 objects called `snapshots\s1\rank_0.pack`. The upload
    /// reported success and the checkpoint was unreadable from then on.
    #[test]
    fn listed_keys_are_slash_separated_and_readable() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStorage::new_unaligned(dir.path()).unwrap();

        store.put("snapshots/s1/rank_0.pack", b"pack bytes").unwrap();
        store.put("snapshots/s2/rank_0.pack", b"more bytes").unwrap();

        let listed = store.list("snapshots").unwrap();
        assert_eq!(listed.len(), 2, "got {listed:?}");

        for key in &listed {
            assert!(
                !key.contains('\\'),
                "a listed key carries a native separator: {key:?}"
            );
            assert!(key.starts_with("snapshots/"), "got {key:?}");
            // The round trip is the point: a key that cannot be read back is
            // not a key.
            assert!(store.get(key).is_ok(), "listed key {key} could not be read");
        }
    }

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
    fn put_parts_concatenates() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStorage::new(dir.path()).unwrap();

        store
            .put_parts("multi.bin", &[b"hello ", b"world", b"!"])
            .unwrap();
        let read = store.get("multi.bin").unwrap();
        assert_eq!(&read[..12], b"hello world!");
        assert_eq!(read.len(), 4096); // padded
    }

    #[test]
    fn get_range_reads_slice() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStorage::new_unaligned(dir.path()).unwrap();

        let data: Vec<u8> = (0..=255).collect();
        store.put("range.bin", &data).unwrap();

        assert_eq!(store.get_range("range.bin", 10, 5).unwrap(), &data[10..15]);
        // Past EOF → truncated, not an error
        assert_eq!(store.get_range("range.bin", 250, 100).unwrap(), &data[250..]);
        assert!(store.get_range("missing.bin", 0, 10).is_err());
    }

}
