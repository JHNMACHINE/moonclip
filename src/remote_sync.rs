use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

use crate::error::{Result, MoonclipError};
use crate::storage::StorageBackend;

/// Configuration for batched remote sync.
#[derive(Debug, Clone)]
pub struct RemoteSyncConfig {
    /// Sync to remote every N save operations. Default 100.
    ///
    /// Zero disables the periodic sync, leaving `sync_now()` as the only way
    /// data reaches the remote — the same meaning `merge_stride` and
    /// `compression_level` give to zero.
    pub sync_every_n_saves: u64,
}

impl Default for RemoteSyncConfig {
    fn default() -> Self {
        RemoteSyncConfig {
            sync_every_n_saves: 100,
        }
    }
}

enum SyncCommand {
    /// Sync all files under the given prefix to remote.
    SyncPrefix(String),
    /// Force sync everything, reporting the outcome back to the caller.
    SyncAll(mpsc::Sender<Option<String>>),
    /// Shutdown.
    Shutdown,
}

/// Background thread that syncs files from local to remote storage.
///
/// The pattern: local SSD is always the primary (fast writes, page-aligned).
/// Remote (S3/GCS) is the durable backup. The syncer copies new files
/// to remote in batches, reducing API calls and bandwidth spikes.
pub struct RemoteSyncer {
    // Mutex-wrapped so RemoteSyncer is Sync (mpsc::Sender is Send but not
    // Sync); the coordinator shares it with the background save thread.
    sender: Option<std::sync::Mutex<mpsc::Sender<SyncCommand>>>,
    handle: Option<thread::JoinHandle<()>>,
    save_counter: std::sync::atomic::AtomicU64,
    sync_every: u64,
}

impl RemoteSyncer {
    /// Create a new remote syncer.
    ///
    /// `local` — the local storage (source)
    /// `remote` — the remote storage (destination)  
    /// `config` — sync frequency configuration
    pub fn new(
        local: Arc<dyn StorageBackend>,
        remote: Arc<dyn StorageBackend>,
        config: RemoteSyncConfig,
    ) -> Self {
        let (tx, rx) = mpsc::channel();
        let sync_every = config.sync_every_n_saves;

        let handle = thread::Builder::new()
            .name("moonclip-remote-sync".into())
            .spawn(move || {
                for cmd in rx {
                    match cmd {
                        SyncCommand::SyncPrefix(prefix) => {
                            if let Err(e) = sync_prefix(&local, &remote, &prefix) {
                                eprintln!("[Moonclip sync] Error syncing '{}': {}", prefix, e);
                            }
                        }
                        SyncCommand::SyncAll(reply) => {
                            let outcome = match sync_prefix(&local, &remote, "") {
                                Ok(()) => None,
                                Err(e) => Some(e.to_string()),
                            };
                            let _ = reply.send(outcome);
                        }
                        SyncCommand::Shutdown => break,
                    }
                }
            })
            .expect("Failed to spawn remote sync thread");

        RemoteSyncer {
            sender: Some(std::sync::Mutex::new(tx)),
            handle: Some(handle),
            save_counter: std::sync::atomic::AtomicU64::new(0),
            sync_every,
        }
    }

    /// Notify the syncer that a save happened.
    /// Triggers a sync if the counter reaches `sync_every_n_saves`.
    pub fn notify_save(&self) {
        // Zero means no periodic sync. Reaching the modulo with it would
        // panic, and `notify_save` runs on the thread that just finished a
        // save — so that panic costs the training run, not a sync.
        if self.sync_every == 0 {
            return;
        }

        let count = self
            .save_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;

        if count % self.sync_every == 0 {
            if let Some(ref tx) = self.sender {
                let tx = tx.lock().unwrap();
                // Sync snapshots and manifest
                let _ = tx.send(SyncCommand::SyncPrefix("snapshots".into()));
                let _ = tx.send(SyncCommand::SyncPrefix("manifest.json".into()));
            }
        }
    }

    /// Force an immediate sync of everything, and wait for it.
    ///
    /// This blocks and returns the outcome, unlike the periodic sync that
    /// `notify_save` triggers. It has to: the whole proposition is that a
    /// checkpoint outlives the machine, and a caller that cannot find out its
    /// data never reached the remote has no way to act on it. Callers reach
    /// this at the end of a run, where waiting is what they wanted anyway.
    pub fn sync_now(&self) -> Result<()> {
        let Some(ref tx) = self.sender else {
            return Ok(()); // already shut down; nothing left to push
        };

        let (reply_tx, reply_rx) = mpsc::channel();
        if tx
            .lock()
            .unwrap()
            .send(SyncCommand::SyncAll(reply_tx))
            .is_err()
        {
            return Err(MoonclipError::Storage(
                "Remote sync thread is gone; nothing was synced".into(),
            ));
        }

        match reply_rx.recv() {
            Ok(None) => Ok(()),
            Ok(Some(e)) => Err(MoonclipError::Storage(format!("Remote sync failed: {e}"))),
            Err(_) => Err(MoonclipError::Storage(
                "Remote sync thread stopped before reporting; data may not have \
                 reached the remote"
                    .into(),
            )),
        }
    }

    /// Shutdown the syncer thread.
    pub fn shutdown(&mut self) {
        if let Some(ref tx) = self.sender {
            let _ = tx.lock().unwrap().send(SyncCommand::Shutdown);
        }
        self.sender.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for RemoteSyncer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Sync all files under `prefix` from local to remote.
///
/// A file already present on the remote is skipped on **name alone** — size
/// and content are never compared. That is sound for snapshot data, which is
/// immutable and UUID-named, and it is why `manifest.json`, the one file that
/// is rewritten every save, takes the single-file branch below and is always
/// re-uploaded.
fn sync_prefix(
    local: &Arc<dyn StorageBackend>,
    remote: &Arc<dyn StorageBackend>,
    prefix: &str,
) -> Result<()> {
    // Special case: single file (e.g. "manifest.json")
    if !prefix.is_empty() && !prefix.contains('/') && prefix.contains('.') {
        match local.get(prefix) {
            Ok(data) => {
                remote.put(prefix, &data)?;
                return Ok(());
            }
            Err(MoonclipError::NotFound(_)) => return Ok(()),
            Err(e) => return Err(e),
        }
    }

    let local_files = local.list(prefix)?;
    let mut synced = 0u64;
    let mut skipped = 0u64;

    for file in &local_files {
        // Check if remote already has this file
        match remote.exists(file) {
            Ok(true) => {
                skipped += 1;
                continue;
            }
            Ok(false) => {}
            Err(_) => {} // If we can't check, try to upload anyway
        }

        let data = local.get(file)?;
        remote.put(file, &data)?;
        synced += 1;
    }

    if synced > 0 {
        eprintln!(
            "[Moonclip sync] Synced {} files, skipped {} (prefix: '{}')",
            synced, skipped, prefix
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::LocalStorage;

    #[test]
    fn sync_between_local_dirs() {
        let src_dir = tempfile::tempdir().unwrap();
        let dst_dir = tempfile::tempdir().unwrap();

        let src: Arc<dyn StorageBackend> =
            Arc::new(LocalStorage::new_unaligned(src_dir.path()).unwrap());
        let dst: Arc<dyn StorageBackend> =
            Arc::new(LocalStorage::new_unaligned(dst_dir.path()).unwrap());

        // Write some files to src
        src.put("snapshots/a/t1.bin", b"data1").unwrap();
        src.put("snapshots/a/t2.bin", b"data2").unwrap();
        src.put("manifest.json", b"{}").unwrap();

        // Sync
        sync_prefix(&src, &dst, "snapshots").unwrap();
        sync_prefix(&src, &dst, "manifest.json").unwrap();

        // Verify
        assert_eq!(dst.get("snapshots/a/t1.bin").unwrap(), b"data1");
        assert_eq!(dst.get("snapshots/a/t2.bin").unwrap(), b"data2");
        assert_eq!(dst.get("manifest.json").unwrap(), b"{}");
    }

    #[test]
    fn batched_sync_triggers_at_threshold() {
        let src_dir = tempfile::tempdir().unwrap();
        let dst_dir = tempfile::tempdir().unwrap();

        let src: Arc<dyn StorageBackend> =
            Arc::new(LocalStorage::new_unaligned(src_dir.path()).unwrap());
        let dst: Arc<dyn StorageBackend> =
            Arc::new(LocalStorage::new_unaligned(dst_dir.path()).unwrap());

        src.put("manifest.json", b"test").unwrap();

        let mut syncer = RemoteSyncer::new(
            Arc::clone(&src),
            Arc::clone(&dst),
            RemoteSyncConfig { sync_every_n_saves: 3 },
        );

        // First 2 saves: no sync
        syncer.notify_save();
        syncer.notify_save();
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!dst.exists("manifest.json").unwrap_or(false));

        // 3rd save: triggers sync
        syncer.notify_save();
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert!(dst.exists("manifest.json").unwrap());

        syncer.shutdown();
    }

    fn two_stores() -> (tempfile::TempDir, tempfile::TempDir, Arc<dyn StorageBackend>, Arc<dyn StorageBackend>)
    {
        let src_dir = tempfile::tempdir().unwrap();
        let dst_dir = tempfile::tempdir().unwrap();
        let src: Arc<dyn StorageBackend> =
            Arc::new(LocalStorage::new_unaligned(src_dir.path()).unwrap());
        let dst: Arc<dyn StorageBackend> =
            Arc::new(LocalStorage::new_unaligned(dst_dir.path()).unwrap());
        (src_dir, dst_dir, src, dst)
    }

    /// `sync_every_n_saves` reaches this straight from the Python constructor,
    /// and zero is a value a user will reasonably try — meaning "every save",
    /// or "never". Whatever it should mean, it must not take the run down.
    #[test]
    fn a_sync_interval_of_zero_does_not_bring_down_the_save() {
        let (_s, _d, src, dst) = two_stores();
        src.put("manifest.json", b"test").unwrap();

        let mut syncer = RemoteSyncer::new(
            Arc::clone(&src),
            Arc::clone(&dst),
            RemoteSyncConfig {
                sync_every_n_saves: 0,
            },
        );

        // notify_save runs on the training thread, so a panic here is not a
        // lost sync — it is a lost training run.
        syncer.notify_save();
        syncer.notify_save();
        syncer.shutdown();
    }

    /// Pins the actual rule, which is *not* the one the doc comment used to
    /// claim: a file already on the remote is skipped on name alone, with no
    /// comparison of size or content. That is right for snapshot data, which
    /// is immutable and UUID-named, and it is why the mutable manifest goes
    /// through the single-file path instead.
    #[test]
    fn remote_files_are_skipped_by_name_not_content() {
        let (_s, _d, src, dst) = two_stores();
        src.put("snapshots/a/t1.bin", b"fresh").unwrap();
        dst.put("snapshots/a/t1.bin", b"stale").unwrap();

        sync_prefix(&src, &dst, "snapshots").unwrap();

        assert_eq!(
            dst.get("snapshots/a/t1.bin").unwrap(),
            b"stale",
            "the skip is by existence; differing content is not detected"
        );
    }

    /// A remote that rejects every write, which is what a wrong bucket, an
    /// expired key or a dead network looks like from here.
    struct BrokenRemote;

    impl StorageBackend for BrokenRemote {
        fn put(&self, rel_path: &str, _data: &[u8]) -> Result<()> {
            Err(MoonclipError::Storage(format!("refused {rel_path}")))
        }
        fn get(&self, rel_path: &str) -> Result<Vec<u8>> {
            Err(MoonclipError::NotFound(rel_path.into()))
        }
        fn exists(&self, _rel_path: &str) -> Result<bool> {
            Ok(false)
        }
        fn delete(&self, _rel_path: &str) -> Result<()> {
            Ok(())
        }
        fn list(&self, _prefix: &str) -> Result<Vec<String>> {
            Ok(Vec::new())
        }
    }

    /// The product's whole claim is that a checkpoint outlives the machine.
    /// A sync that fails has to say so: printing to stderr and returning
    /// leaves a run believing it is protected when it is not.
    #[test]
    fn a_remote_that_refuses_writes_is_reported() {
        let src_dir = tempfile::tempdir().unwrap();
        let src: Arc<dyn StorageBackend> =
            Arc::new(LocalStorage::new_unaligned(src_dir.path()).unwrap());
        src.put("snapshots/a/t1.bin", b"data").unwrap();

        let mut syncer = RemoteSyncer::new(
            Arc::clone(&src),
            Arc::new(BrokenRemote),
            RemoteSyncConfig::default(),
        );

        let result = syncer.sync_now();
        assert!(
            result.is_err(),
            "a remote refusing every write must surface as an error"
        );
        syncer.shutdown();
    }

    #[test]
    fn a_working_remote_reports_success() {
        let (_s, _d, src, dst) = two_stores();
        src.put("snapshots/a/t1.bin", b"data").unwrap();

        let mut syncer =
            RemoteSyncer::new(Arc::clone(&src), Arc::clone(&dst), RemoteSyncConfig::default());

        syncer.sync_now().unwrap();
        // sync_now waits, so the data is there by the time it returns —
        // no sleep, and no flake.
        assert_eq!(dst.get("snapshots/a/t1.bin").unwrap(), b"data");
        syncer.shutdown();
    }

    /// The manifest is the one file that changes on every save. If a sync
    /// skipped it the way it skips snapshot data, the remote would hold
    /// checkpoints that nothing points at.
    #[test]
    fn a_rewritten_manifest_reaches_the_remote() {
        let (_s, _d, src, dst) = two_stores();

        src.put("manifest.json", b"{\"snapshots\":[]}").unwrap();
        sync_prefix(&src, &dst, "manifest.json").unwrap();

        src.put("manifest.json", b"{\"snapshots\":[1]}").unwrap();
        sync_prefix(&src, &dst, "manifest.json").unwrap();

        assert_eq!(dst.get("manifest.json").unwrap(), b"{\"snapshots\":[1]}");
    }
}
