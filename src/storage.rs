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

    /// Write exactly `data`, byte for byte, whatever this backend does to
    /// the files it writes for itself.
    ///
    /// For copies: a file the remote sync brings down belongs to whoever
    /// wrote it, and must come back as it left. The local store pads what it
    /// writes to a page with zeros, which Moonclip's own readers strip - and a
    /// caller's JSON beside the checkpoints does not survive. Most backends
    /// write what they are given anyway, hence the default.
    fn put_exact(&self, rel_path: &str, data: &[u8]) -> Result<()> {
        self.put(rel_path, data)
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

/// Whether writes are flushed to the device before they are made reachable.
///
/// On by default, and `MOONCLIP_FSYNC=0` turns it off.
///
/// The temp-file-then-rename dance below is atomic with respect to *readers* —
/// nobody ever sees a half-written pack — and says nothing about power. The
/// rename is a metadata operation and can reach the disk while the data it
/// points at is still in the page cache, so a machine that loses power at the
/// wrong moment comes back with a pack of exactly the right length, full of
/// zeros, indexed by a manifest that vouches for it. That is worse than a
/// missing checkpoint: recovery believes it. Meanwhile `flush()` promised the
/// data was "durably on disk", which it was not.
///
/// The cost is a real one — an fsync per pack, on the path this library exists
/// to make fast — which is why there is a way out. Spot instances and
/// preemptible nodes are the case it is on for: the whole premise is that the
/// machine can disappear.
fn fsync_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("MOONCLIP_FSYNC").as_deref(),
            Ok("0") | Ok("false")
        )
    })
}

/// Flush a directory entry so a rename into it survives a power cut.
///
/// Unix only. Windows has no way to open a directory as a file, and NTFS
/// orders its metadata through the journal, so there is nothing to do there.
#[cfg(unix)]
fn sync_dir(dir: &Path) {
    // Best effort: a filesystem that refuses this (some network mounts do)
    // should not fail a checkpoint that is otherwise written.
    let _ = std::fs::File::open(dir).and_then(|d| d.sync_all());
}

#[cfg(not(unix))]
fn sync_dir(_dir: &Path) {}

/// Create `dir` and every missing ancestor, returning the ones that had to be
/// created, deepest first.
///
/// The caller needs that list because creating a directory is itself a change
/// to its *parent*, and an unsynced parent can lose the entry. Syncing the
/// pack and then the directory holding it says nothing about whether the
/// directory is still named by anything: every snapshot writes into a fresh
/// `snapshots/<uuid>/`, so on the first pack of every snapshot the entry for
/// `<uuid>` lives in `snapshots/`, which nothing had synced. A power cut there
/// takes the whole snapshot with it — a fully fsynced pack in a directory that
/// no longer exists — which is exactly the outcome `fsync_enabled` is on to
/// prevent.
///
/// In the steady state a directory already exists and this returns empty, so
/// the extra syncs are paid once per snapshot rather than once per pack.
fn create_dirs_recording_new(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut created = Vec::new();
    let mut cursor = Some(dir);
    while let Some(path) = cursor {
        // The empty path is where `parent()` lands after the last component
        // of a relative path. It is not a directory anyone created and
        // `exists()` says false for it, so without this it joins the list and
        // earns a `sync_dir("")` that can only fail.
        if path.as_os_str().is_empty() || path.exists() {
            break;
        }
        created.push(path.to_path_buf());
        cursor = path.parent();
    }
    if !created.is_empty() {
        std::fs::create_dir_all(dir)?;
    }
    Ok(created)
}

/// How long a write waits for whoever holds its target open. See
/// [`persist_retrying`].
///
/// Long enough for the readers that actually do this — an antivirus scanning a
/// file that just changed, a search indexer, a backup agent, someone typing
/// `type manifest.json` — which hold a file for milliseconds to a second or
/// two. Short enough that a target that stays locked is reported while the run
/// can still do something about it. The price is paid only by a write that is
/// being refused, never by one that is not.
const PERSIST_RETRY_BUDGET: std::time::Duration = std::time::Duration::from_secs(5);

/// Rename `tmp` over `path`, waiting out a reader that holds `path` open.
///
/// On Windows a rename over an open file fails — with *access denied*, even
/// when the reader opened it sharing read, write and delete — and nothing the
/// reader could have done differently would have helped (GPU-128). The writer
/// was the one losing: the save the rename was finishing failed, and with it
/// the checkpoint. And the readers are not ours to fix: an antivirus opens
/// every file that changes.
///
/// So the refusal is retried, with a pause that doubles up to 100 ms, until
/// `budget` is spent. Past it the error is the one the rename gave, and the
/// temp file is removed; the old file was never touched.
///
/// *Access denied* is also what a real permission problem looks like, and the
/// two cannot be told apart from here. Such a write now fails after `budget`
/// rather than at once — the same failure, reported later — which is the
/// trade for not losing checkpoints to a scanner.
///
/// Elsewhere a rename over an open file succeeds, so there is nothing to wait
/// for and every error is returned as it comes.
fn persist_retrying(
    mut tmp: tempfile::NamedTempFile,
    path: &Path,
    budget: std::time::Duration,
) -> std::result::Result<(), tempfile::PersistError> {
    let start = std::time::Instant::now();
    let mut pause = std::time::Duration::from_millis(1);
    loop {
        match tmp.persist(path) {
            Ok(_) => return Ok(()),
            Err(e) if held_open_elsewhere(&e.error) && start.elapsed() + pause <= budget => {
                tmp = e.file;
                std::thread::sleep(pause);
                pause = (pause * 2).min(std::time::Duration::from_millis(100));
            }
            Err(e) => return Err(e),
        }
    }
}

/// Whether a failed rename is Windows refusing because the target is open.
///
/// `ERROR_ACCESS_DENIED` (5) is what `MoveFileExW` returns for a target open
/// with sharing, and is what was measured; `ERROR_SHARING_VIOLATION` (32) is
/// the same refusal for a target opened without it.
#[cfg(windows)]
fn held_open_elsewhere(e: &std::io::Error) -> bool {
    matches!(e.raw_os_error(), Some(5) | Some(32))
}

#[cfg(not(windows))]
fn held_open_elsewhere(_e: &std::io::Error) -> bool {
    false
}

impl LocalStorage {
    /// Write `parts` atomically, padded with zeros to a multiple of
    /// `page_size` - 0 for none.
    fn write_parts(&self, rel_path: &str, parts: &[&[u8]], page_size: usize) -> Result<()> {
        use std::io::Write;

        let path = self.full_path(rel_path);
        let new_dirs = match path.parent() {
            Some(parent) => create_dirs_recording_new(parent)?,
            None => Vec::new(),
        };

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
            if page_size > 0 && total % page_size != 0 {
                let pad = page_size - total % page_size;
                file.write_all(&vec![0u8; pad])?;
            }
            file.flush()?;
            // The bytes reach the device before the rename makes them
            // reachable, so the two can never be persisted out of order.
            // See `fsync_enabled`.
            if fsync_enabled() {
                file.sync_all()?;
            }
        }
        persist_retrying(tmp, &path, PERSIST_RETRY_BUDGET).map_err(|e| {
            MoonclipError::Storage(format!("Failed to persist {}: {}", path.display(), e))
        })?;
        // And the rename itself, which is a change to the directory rather
        // than to the file, and is not covered by the sync above.
        if fsync_enabled() {
            sync_dir(dir);
            // Then the entries naming any directory this call had to create,
            // shallowest first: a synced pack inside a directory whose own
            // entry never reached the device is still a lost checkpoint.
            //
            // The order is the point. Each sync makes the entry *naming*
            // `created` durable, so going from the root down means a crash
            // part-way through leaves a shorter chain that is still reachable
            // from the root. Deepest first — which is the order
            // `create_dirs_recording_new` returns, hence the `rev()` — would
            // make a deep entry durable inside a parent nothing names yet,
            // which is the very thing this loop exists to prevent.
            for created in new_dirs.iter().rev() {
                if let Some(parent) = created.parent() {
                    sync_dir(parent);
                }
            }
        }
        Ok(())
    }
}

impl StorageBackend for LocalStorage {
    fn put(&self, rel_path: &str, data: &[u8]) -> Result<()> {
        self.put_parts(rel_path, &[data])
    }

    fn put_parts(&self, rel_path: &str, parts: &[&[u8]]) -> Result<()> {
        self.write_parts(rel_path, parts, self.page_size)
    }

    fn put_exact(&self, rel_path: &str, data: &[u8]) -> Result<()> {
        self.write_parts(rel_path, &[data], 0)
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

    /// Regression (GPU-128): on Windows a rename over a file someone holds
    /// open fails, and the save that rename was finishing is lost. Every
    /// reader of `manifest.json` could do it — an antivirus, an indexer, a
    /// person typing `type manifest.json` — at the wrong moment.
    ///
    /// `File::open` in std shares read, write *and delete*, the most a reader
    /// can concede, and the rename fails against it all the same: measured on
    /// 2026-09-14 with `CreateFileW` directly. So a reader that lets go in
    /// time has to be waited for, not blamed.
    #[cfg(windows)]
    #[test]
    fn a_reader_holding_the_target_open_does_not_cost_the_write() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalStorage::new_unaligned(dir.path()).unwrap();
        store.put("manifest.json", b"old").unwrap();

        // Opened here, before the write starts, so the write cannot win a race
        // against the thread and pass for the wrong reason.
        let held = std::fs::File::open(dir.path().join("manifest.json")).unwrap();
        let reader = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(300));
            drop(held);
        });

        store.put("manifest.json", b"new").unwrap();
        reader.join().unwrap();
        assert_eq!(store.get("manifest.json").unwrap(), b"new");
    }

    /// The wait is a wait and not a hang: a reader that never lets go still
    /// fails the write, once the budget is spent, and the file it was
    /// replacing is untouched.
    #[cfg(windows)]
    #[test]
    fn a_reader_that_never_lets_go_fails_the_write_within_the_budget() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.json");
        std::fs::write(&path, b"old").unwrap();
        let _held = std::fs::File::open(&path).unwrap();

        let tmp = tempfile::NamedTempFile::new_in(dir.path()).unwrap();
        let budget = std::time::Duration::from_millis(200);
        let start = std::time::Instant::now();
        let err = persist_retrying(tmp, &path, budget).unwrap_err();
        let took = start.elapsed();

        assert!(held_open_elsewhere(&err.error), "unexpected error: {err}");
        // At least the budget less one pause, which is capped at 100 ms.
        assert!(took >= budget / 2, "gave up after {took:?} without waiting");
        assert!(took < budget * 5, "kept retrying for {took:?} on a {budget:?} budget");
        assert_eq!(std::fs::read(&path).unwrap(), b"old");
    }

}
