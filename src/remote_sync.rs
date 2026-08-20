use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;

use crate::error::{Result, MoonclipError};
use crate::storage::StorageBackend;

/// Keys that have been deleted locally and are still on the remote.
///
/// Retention and the merger delete their files as soon as nothing can reach
/// them, and until 0.0.6 that stopped at the local disk: the bucket kept every
/// pack the run ever wrote, so `keep_last: 3` bounded the SSD and nothing at
/// all bounded the object store. A long run's remote cost grew with its
/// length, which is exactly what retention exists to prevent.
///
/// Deletions are queued rather than sent immediately, for the same reason
/// uploads are batched: they happen on the save path, and a round trip to S3
/// there is paid by the training loop. The syncer drains this on its next
/// pass.
///
/// **Only keys this process deleted.** The obvious alternative — list the
/// remote, delete whatever is not local — is a foot-gun with the safety off:
/// point a fresh machine with an empty checkpoint directory at an existing
/// bucket and it erases the backup it was meant to restore from.
///
/// **A queue nobody drains has to refuse work.** Only the syncer thread drains
/// this, so with no remote configured — the default — every retention eviction
/// and every merge appended a key that was never read and never freed. That is
/// unbounded growth in a process whose whole purpose is to run for weeks, so a
/// registry built by [`PendingDeletes::disabled`] drops pushes on the floor.
pub struct PendingDeletes {
    keys: Mutex<Vec<String>>,
    /// Whether anything will ever drain `keys`. False when the coordinator has
    /// no remote, in which case there is no remote copy to delete either.
    collecting: bool,
}

impl Default for PendingDeletes {
    fn default() -> Self {
        PendingDeletes {
            keys: Mutex::new(Vec::new()),
            collecting: true,
        }
    }
}

impl PendingDeletes {
    /// A registry for a coordinator with no remote: `push` is a no-op.
    pub fn disabled() -> Self {
        PendingDeletes {
            keys: Mutex::new(Vec::new()),
            collecting: false,
        }
    }

    pub fn push(&self, key: impl Into<String>) {
        if !self.collecting {
            return;
        }
        self.keys.lock().unwrap().push(key.into());
    }

    fn drain(&self) -> Vec<String> {
        std::mem::take(&mut *self.keys.lock().unwrap())
    }
}

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
        deletes: Arc<PendingDeletes>,
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
                            apply_deletes(&remote, &deletes);
                        }
                        SyncCommand::SyncAll(reply) => {
                            let outcome = match sync_prefix(&local, &remote, "") {
                                Ok(()) => None,
                                Err(e) => Some(e.to_string()),
                            };
                            // After the upload, never before: a key queued for
                            // deletion is already gone locally, so the upload
                            // above cannot have put it back.
                            apply_deletes(&remote, &deletes);
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

/// Delete, on the remote, what has already been deleted locally.
///
/// Failures are logged and dropped rather than returned. A delete that does
/// not land leaves an object nobody references — it costs storage, and the
/// next pass will not retry it, which is the honest trade: a sync that
/// *failed to upload* is a checkpoint at risk and must be reported, while a
/// sync that failed to tidy up is a bill. Reporting them the same way would
/// teach callers to ignore both.
fn apply_deletes(remote: &Arc<dyn StorageBackend>, deletes: &Arc<PendingDeletes>) {
    let keys = deletes.drain();
    if keys.is_empty() {
        return;
    }
    let mut failed = 0usize;
    for key in &keys {
        if remote.delete(key).is_err() {
            failed += 1;
        }
    }
    if failed > 0 {
        eprintln!(
            "[Moonclip sync] {failed} of {} deleted checkpoints could not be removed \
             from the remote; they are unreferenced but still billed",
            keys.len()
        );
    }
}

/// Copy every file under `prefix` from one store to another.
///
/// Named `from`/`to` rather than local/remote because the direction is the
/// caller's: pushing a checkpoint to a bucket and pulling one back onto a
/// machine that lost its disk are the same walk with the arguments swapped.
/// See [`restore_from_remote`].
///
/// A file already present at the destination is skipped on **name alone** —
/// size and content are never compared. That is sound for snapshot data, which
/// is immutable and UUID-named, and it is why `manifest.json`, the one file
/// that is rewritten every save, takes the single-file branch below and is
/// always copied again.
fn sync_prefix(
    from: &Arc<dyn StorageBackend>,
    to: &Arc<dyn StorageBackend>,
    prefix: &str,
) -> Result<()> {
    // Special case: single file (e.g. "manifest.json")
    if !prefix.is_empty() && !prefix.contains('/') && prefix.contains('.') {
        match from.get(prefix) {
            Ok(data) => {
                to.put(prefix, &data)?;
                return Ok(());
            }
            Err(MoonclipError::NotFound(_)) => return Ok(()),
            Err(e) => return Err(e),
        }
    }

    let files = from.list(prefix)?;
    let mut synced = 0u64;
    let mut skipped = 0u64;

    for file in &files {
        match to.exists(file) {
            Ok(true) => {
                skipped += 1;
                continue;
            }
            Ok(false) => {}
            Err(_) => {} // If we cannot check, copy anyway
        }

        let data = from.get(file)?;
        to.put(file, &data)?;
        synced += 1;
    }

    if synced > 0 {
        eprintln!(
            "[Moonclip sync] Copied {} files, skipped {} (prefix: '{}')",
            synced, skipped, prefix
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::LocalStorage;

    /// Nothing has been deleted locally, so the syncer has nothing to remove
    /// from the remote.
    fn nothing_deleted() -> Arc<PendingDeletes> {
        Arc::new(PendingDeletes::default())
    }

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
        nothing_deleted(),
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
        nothing_deleted(),
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
            nothing_deleted(),
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
            RemoteSyncer::new(
                Arc::clone(&src),
                Arc::clone(&dst),
                RemoteSyncConfig::default(),
                nothing_deleted(),
            );

        syncer.sync_now().unwrap();
        // sync_now waits, so the data is there by the time it returns —
        // no sleep, and no flake.
        assert_eq!(dst.get("snapshots/a/t1.bin").unwrap(), b"data");
        syncer.shutdown();
    }

    /// Retention bounds the local disk. Before 0.0.6 it bounded nothing on the
    /// remote: every pack a run ever wrote stayed in the bucket, so the object
    /// store's cost grew with the length of the run while `keep_last` quietly
    /// held the SSD flat.
    #[test]
    fn a_locally_deleted_checkpoint_is_removed_from_the_remote() {
        let (_s, _d, src, dst) = two_stores();
        let deletes = Arc::new(PendingDeletes::default());

        src.put("snapshots/old/rank_0.pack", b"superseded").unwrap();
        src.put("snapshots/new/rank_0.pack", b"current").unwrap();

        let mut syncer = RemoteSyncer::new(
            Arc::clone(&src),
            Arc::clone(&dst),
            RemoteSyncConfig::default(),
            Arc::clone(&deletes),
        );
        syncer.sync_now().unwrap();
        assert!(dst.exists("snapshots/old/rank_0.pack").unwrap());

        // What retention does: gone locally, queued for the remote.
        src.delete("snapshots/old/rank_0.pack").unwrap();
        deletes.push("snapshots/old/rank_0.pack");
        assert_eq!(
            deletes.keys.lock().unwrap().len(),
            1,
            "a collecting registry dropped the key"
        );

        syncer.sync_now().unwrap();
        assert!(
            !dst.exists("snapshots/old/rank_0.pack").unwrap(),
            "the bucket kept a checkpoint retention had already dropped"
        );
        assert!(
            dst.exists("snapshots/new/rank_0.pack").unwrap(),
            "the live checkpoint went with it"
        );
        syncer.shutdown();
    }

    /// With no remote there is nothing to drain the queue and nothing on a
    /// remote to delete, so pushes have to go nowhere. Collecting them was an
    /// unbounded leak: every retention eviction and every merge appended a key
    /// that nothing would ever read, for the length of a run measured in
    /// weeks.
    #[test]
    fn a_registry_with_no_remote_keeps_nothing() {
        let deletes = PendingDeletes::disabled();
        for step in 0..10_000 {
            deletes.push(format!("snapshots/{step}/rank_0.pack"));
        }
        assert!(
            deletes.keys.lock().unwrap().is_empty(),
            "keys nobody will ever drain accumulated anyway"
        );
        assert!(deletes.drain().is_empty());
    }

    /// A remote that cannot delete must not fail the sync: the upload is what
    /// protects the run, and the leftover object is a bill rather than a risk.
    #[test]
    fn a_remote_that_refuses_deletes_still_reports_a_good_sync() {
        struct WriteOnly;
        impl StorageBackend for WriteOnly {
            fn put(&self, _rel_path: &str, _data: &[u8]) -> Result<()> {
                Ok(())
            }
            fn get(&self, rel_path: &str) -> Result<Vec<u8>> {
                Err(MoonclipError::NotFound(rel_path.into()))
            }
            fn exists(&self, _rel_path: &str) -> Result<bool> {
                Ok(false)
            }
            fn delete(&self, rel_path: &str) -> Result<()> {
                Err(MoonclipError::Storage(format!("no delete: {rel_path}")))
            }
            fn list(&self, _prefix: &str) -> Result<Vec<String>> {
                Ok(Vec::new())
            }
        }

        let src_dir = tempfile::tempdir().unwrap();
        let src: Arc<dyn StorageBackend> =
            Arc::new(LocalStorage::new_unaligned(src_dir.path()).unwrap());
        src.put("snapshots/a/rank_0.pack", b"data").unwrap();

        let deletes = Arc::new(PendingDeletes::default());
        deletes.push("snapshots/gone/rank_0.pack");

        let mut syncer = RemoteSyncer::new(
            Arc::clone(&src),
            Arc::new(WriteOnly),
            RemoteSyncConfig::default(),
            Arc::clone(&deletes),
        );

        syncer.sync_now().unwrap();
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


/// Pull a store back from the remote onto a machine that has none.
///
/// The remote was push-only until 2026-08-19, and the gap was not theoretical:
/// measured on a six-node bench, a node whose disk was replaced started from
/// scratch while its data sat in the bucket, and every other rank started over
/// with it — `agree_on_step` takes the minimum. A plain reshuffle of which node
/// hosts which rank did the same, with every disk intact. So the bucket was a
/// backup and never a way back.
///
/// Returns whether a manifest arrived, which is what makes the local store
/// readable at all. Deliberately **not** called from the constructor: pulling
/// a whole checkpoint is not something a caller should discover by having
/// started a manager, and only the caller knows whether this run wants to
/// resume at all.
pub fn restore_from_remote(
    local: &Arc<dyn StorageBackend>,
    remote: &Arc<dyn StorageBackend>,
) -> Result<bool> {
    // The manifest first and on its own: without it the snapshot files are
    // bytes nothing names, and if it never arrives there is nothing to restore
    // and no reason to pay for the rest.
    sync_prefix(remote, local, "manifest.json")?;
    if !local.exists("manifest.json").unwrap_or(false) {
        return Ok(false);
    }

    sync_prefix(remote, local, "snapshots")?;
    Ok(true)
}
