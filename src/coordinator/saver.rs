//! The background thread a save is handed to.
//!
//! `save()` returns once the tensor data has been copied; hashing, delta
//! detection, compression and the write happen here and overlap with training.
//! What the training loop pays for is the copy, not the I/O.
//!
//! `AsyncSaver` is declared before `core` in [`super::Coordinator`] so that
//! pending saves drain before the `Core` they borrow is torn down — see the
//! `Drop` impl at the bottom.

use std::sync::atomic::AtomicU64;
use std::sync::{mpsc, Condvar};
use std::thread;

use super::*;

pub(super) struct SaveJob {
    pub(super) snap_id: Uuid,
    pub(super) step: u64,
    pub(super) tensors: Vec<TensorData>,
    pub(super) metadata: HashMap<String, String>,
}

struct SaverShared {
    busy: bool,
    error: Option<String>,
}

/// Dedicated thread that runs the save pipeline. At most one save is
/// in flight (bound-1 queue): submitting waits for the previous job.
pub(super) struct AsyncSaver {
    tx: Option<mpsc::Sender<SaveJob>>,
    shared: Arc<(Mutex<SaverShared>, Condvar)>,
    handle: Option<thread::JoinHandle<()>>,
    /// How long the most recent `submit` waited for the previous save to
    /// drain. Written by the caller's thread in `submit`, read by that same
    /// thread once `save` returns — see [`Coordinator::last_queue_wait`].
    pub(super) last_wait: AtomicU64,
}

impl AsyncSaver {
    pub(super) fn new(core: Arc<Core>) -> Self {
        let (tx, rx) = mpsc::channel::<SaveJob>();
        let shared = Arc::new((
            Mutex::new(SaverShared {
                busy: false,
                error: None,
            }),
            Condvar::new(),
        ));
        let shared2 = Arc::clone(&shared);

        let handle = thread::Builder::new()
            .name("moonclip-async-saver".into())
            .spawn(move || {
                for job in rx {
                    // A panic here used to kill this thread with `busy` still
                    // set and nobody left to clear it, so the next `submit`
                    // waited on the condvar forever: the training loop hung
                    // silently and permanently, which on rented hardware is an
                    // idle GPU billing until a human notices. Catching it turns
                    // a panic into the same thing a storage failure already is
                    // — an error the next save, flush or load reports.
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        core.save_sync(job.snap_id, job.step, job.tensors, job.metadata)
                    }));

                    let mut state = shared2.0.lock().unwrap();
                    state.busy = false;
                    match result {
                        Ok(Ok(())) => {}
                        Ok(Err(e)) => state.error = Some(e.to_string()),
                        Err(panic) => {
                            let what = panic
                                .downcast_ref::<&str>()
                                .map(|s| (*s).to_string())
                                .or_else(|| panic.downcast_ref::<String>().cloned())
                                .unwrap_or_else(|| "unknown payload".into());
                            state.error = Some(format!("save thread panicked: {what}"));
                        }
                    }
                    drop(state);
                    shared2.1.notify_all();
                }
            })
            .expect("Failed to spawn moonclip-async-saver thread");

        AsyncSaver {
            tx: Some(tx),
            shared,
            handle: Some(handle),
            last_wait: AtomicU64::new(0),
        }
    }

    /// Wait until no save is in flight. Does not consume errors.
    pub(super) fn wait_idle(&self) {
        let (lock, cvar) = &*self.shared;
        let mut state = lock.lock().unwrap();
        while state.busy {
            state = cvar.wait(state).unwrap();
        }
    }

    /// Wait until idle and surface any pending background error (once).
    pub(super) fn flush(&self) -> Result<()> {
        let (lock, cvar) = &*self.shared;
        let mut state = lock.lock().unwrap();
        while state.busy {
            state = cvar.wait(state).unwrap();
        }
        match state.error.take() {
            Some(e) => Err(MoonclipError::Storage(format!(
                "Background save failed: {e}"
            ))),
            None => Ok(()),
        }
    }

    /// Submit a job, waiting for the previous one to drain first.
    ///
    /// The wait and the claim of the in-flight slot happen under a single
    /// lock acquisition. Doing them separately — wait for idle, release,
    /// re-acquire, set busy — lets two threads both observe an idle saver and
    /// both queue a job. That breaks the bound-1 invariant, and because
    /// `busy` is one flag for what is then two outstanding jobs, the worker
    /// clears it after the first completes: `flush()` returns while a save is
    /// still queued, and the caller sees a manifest missing that snapshot.
    pub(super) fn submit(&self, job: SaveJob) -> Result<()> {
        // This function blocks. Doing that on a Moonclip pool thread is the
        // deadlock described in `crate::python::save_tensors`: the wait can
        // only end when a save pipeline finishes, that pipeline runs on this
        // same pool, and a blocked worker may be sitting on a piece of it.
        // Callers must submit from outside the pool.
        debug_assert!(
            rayon::current_thread_index().is_none(),
            "AsyncSaver::submit blocked on a Moonclip pool thread"
        );
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| MoonclipError::Storage("Background saver is shut down".into()))?;

        let (lock, cvar) = &*self.shared;
        let queued = Instant::now();
        let mut state = lock.lock().unwrap();
        while state.busy {
            state = cvar.wait(state).unwrap();
        }
        // How long that wait was is the difference between "the copy is slow"
        // and "the writer never catches up", and the caller cannot tell them
        // apart from the outside. Kept for the caller as well as logged: a
        // number only `MOONCLIP_PROFILE=1` will show is a number nobody has
        // when the question comes up, which is on a rented machine mid-run.
        let waited = queued.elapsed();
        self.last_wait
            .store(waited.as_nanos() as u64, Ordering::Relaxed);
        // Surface a failure from the previous save before starting another,
        // matching what the old `self.flush()?` on entry did.
        if let Some(e) = state.error.take() {
            return Err(MoonclipError::Storage(format!("Background save failed: {e}")));
        }
        state.busy = true;
        drop(state);
        crate::profile::note_queue_wait(waited);

        if tx.send(job).is_err() {
            lock.lock().unwrap().busy = false;
            cvar.notify_all();
            return Err(MoonclipError::Storage(
                "Background saver channel closed".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn shutdown(&mut self) {
        self.wait_idle();
        self.tx.take(); // close channel → worker exits
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for AsyncSaver {
    fn drop(&mut self) {
        self.shutdown();
    }
}
