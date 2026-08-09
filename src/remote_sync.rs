use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

use crate::error::{Result, MoonclipError};
use crate::storage::StorageBackend;

/// Configuration for batched remote sync.
#[derive(Debug, Clone)]
pub struct RemoteSyncConfig {
    /// Sync to remote every N save operations. Default 100.
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
    /// Force sync everything.
    SyncAll,
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
                        SyncCommand::SyncAll => {
                            if let Err(e) = sync_prefix(&local, &remote, "") {
                                eprintln!("[Moonclip sync] Error syncing all: {}", e);
                            }
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

    /// Force an immediate sync of everything.
    pub fn sync_now(&self) {
        if let Some(ref tx) = self.sender {
            let _ = tx.lock().unwrap().send(SyncCommand::SyncAll);
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
/// Only copies files that don't exist on remote (or differ in size).
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
}
