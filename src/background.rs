use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use uuid::Uuid;

use crate::error::{Result, RevolverError};

/// A job sent from the main thread to the background saver.
pub struct SaveJob {
    pub snap_id: Uuid,
    pub step: u64,
    pub components: HashMap<String, Vec<u8>>,
    pub metadata: HashMap<String, String>,
}

/// Result of a completed background save.
#[derive(Debug, Clone)]
pub struct SaveResult {
    pub snap_id: Uuid,
    pub step: u64,
    pub error: Option<String>,
    pub elapsed_secs: f64,
}

/// Shared state between foreground and background.
/// Tracks whether a save is in-flight and stores the last result.
struct SharedState {
    /// True while the background thread is processing a job.
    busy: bool,
    /// Result of the last completed save (if any).
    last_result: Option<SaveResult>,
}

/// Background saver: owns a dedicated OS thread that processes SaveJobs
/// from a channel. The main thread can submit jobs and optionally wait
/// for completion.
pub struct BackgroundSaver {
    sender: Option<mpsc::Sender<SaveJob>>,
    shared: Arc<(Mutex<SharedState>, Condvar)>,
    handle: Option<thread::JoinHandle<()>>,
}

impl BackgroundSaver {
    /// Spawn the background saver thread.
    ///
    /// `process_fn` is the closure that does the actual save work.
    /// It receives a SaveJob and returns Ok(()) or an error string.
    /// This closure is called on the background thread.
    pub fn new<F>(process_fn: F) -> Self
    where
        F: Fn(SaveJob) -> std::result::Result<(), String> + Send + 'static,
    {
        let (tx, rx) = mpsc::channel::<SaveJob>();

        let shared = Arc::new((
            Mutex::new(SharedState {
                busy: false,
                last_result: None,
            }),
            Condvar::new(),
        ));

        let shared_clone = Arc::clone(&shared);

        let handle = thread::Builder::new()
            .name("revolver-bg-saver".into())
            .spawn(move || {
                for job in rx {
                    let snap_id = job.snap_id;
                    let step = job.step;

                    // Mark busy
                    {
                        let mut state = shared_clone.0.lock().unwrap();
                        state.busy = true;
                    }

                    let t0 = std::time::Instant::now();
                    let result = process_fn(job);
                    let elapsed = t0.elapsed().as_secs_f64();

                    let save_result = SaveResult {
                        snap_id,
                        step,
                        error: result.err(),
                        elapsed_secs: elapsed,
                    };

                    // Mark done, store result, notify waiters
                    {
                        let mut state = shared_clone.0.lock().unwrap();
                        state.busy = false;
                        state.last_result = Some(save_result);
                    }
                    shared_clone.1.notify_all();
                }
            })
            .expect("Failed to spawn background saver thread");

        BackgroundSaver {
            sender: Some(tx),
            shared,
            handle: Some(handle),
        }
    }

    /// Submit a save job. Returns immediately.
    ///
    /// If the previous save is still in progress, this method BLOCKS
    /// until it completes before submitting the new job. This ensures
    /// at most one save is in-flight at a time (no unbounded queue).
    pub fn submit(&self, job: SaveJob) -> Result<()> {
        // Wait for any in-flight save to finish first
        self.wait_if_busy()?;

        // Mark as busy before sending (to avoid a race where is_busy()
        // returns false between submit and the bg thread picking it up)
        {
            let mut state = self.shared.0.lock().unwrap();
            state.busy = true;
        }

        self.sender
            .as_ref()
            .ok_or_else(|| RevolverError::Storage("Background saver is shut down".into()))?
            .send(job)
            .map_err(|_| RevolverError::Storage("Background saver channel closed".into()))?;

        Ok(())
    }

    /// Block until any in-flight save completes.
    /// Returns the result of the last save, if any.
    pub fn wait(&self) -> Result<Option<SaveResult>> {
        self.wait_if_busy()?;
        let state = self.shared.0.lock().unwrap();
        Ok(state.last_result.clone())
    }

    /// Check if a save is currently in progress.
    pub fn is_busy(&self) -> bool {
        let state = self.shared.0.lock().unwrap();
        state.busy
    }

    /// Get the last save result without blocking.
    pub fn last_result(&self) -> Option<SaveResult> {
        let state = self.shared.0.lock().unwrap();
        state.last_result.clone()
    }

    /// Internal: block until not busy.
    fn wait_if_busy(&self) -> Result<()> {
        let (lock, cvar) = &*self.shared;
        let mut state = lock.lock().unwrap();
        while state.busy {
            state = cvar.wait(state).unwrap();
        }

        // Check if the last save had an error
        if let Some(ref result) = state.last_result {
            if let Some(ref err) = result.error {
                return Err(RevolverError::Storage(format!(
                    "Previous background save (step {}) failed: {}",
                    result.step, err
                )));
            }
        }

        Ok(())
    }

    /// Shut down the background thread gracefully.
    /// Waits for any in-flight save to finish, then joins the thread.
    pub fn shutdown(&mut self) -> Result<()> {
        // Wait for current job
        self.wait()?;

        // Drop the sender to signal the thread to exit
        self.sender.take();

        // Join the thread
        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| RevolverError::Storage("Background thread panicked".into()))?;
        }

        Ok(())
    }
}

impl Drop for BackgroundSaver {
    fn drop(&mut self) {
        // Best-effort shutdown
        let _ = self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[test]
    fn basic_submit_and_wait() {
        let counter = Arc::new(AtomicU64::new(0));
        let counter_clone = Arc::clone(&counter);

        let mut saver = BackgroundSaver::new(move |job: SaveJob| {
            // Simulate work
            std::thread::sleep(std::time::Duration::from_millis(50));
            counter_clone.store(job.step, Ordering::SeqCst);
            Ok(())
        });

        saver
            .submit(SaveJob {
                snap_id: Uuid::new_v4(),
                step: 42,
                components: HashMap::new(),
                metadata: HashMap::new(),
            })
            .unwrap();

        let result = saver.wait().unwrap().unwrap();
        assert_eq!(result.step, 42);
        assert!(result.error.is_none());
        assert_eq!(counter.load(Ordering::SeqCst), 42);

        saver.shutdown().unwrap();
    }

    #[test]
    fn sequential_submits_block_correctly() {
        let log = Arc::new(Mutex::new(Vec::<u64>::new()));
        let log_clone = Arc::clone(&log);

        let mut saver = BackgroundSaver::new(move |job: SaveJob| {
            std::thread::sleep(std::time::Duration::from_millis(30));
            log_clone.lock().unwrap().push(job.step);
            Ok(())
        });

        // Submit 3 jobs rapidly — each should wait for the previous
        for step in [100, 200, 300] {
            saver
                .submit(SaveJob {
                    snap_id: Uuid::new_v4(),
                    step,
                    components: HashMap::new(),
                    metadata: HashMap::new(),
                })
                .unwrap();
        }

        saver.wait().unwrap();

        let logged = log.lock().unwrap();
        assert_eq!(*logged, vec![100, 200, 300]);

        saver.shutdown().unwrap();
    }

    #[test]
    fn error_propagation() {
        let mut saver = BackgroundSaver::new(|_job: SaveJob| {
            Err("disk full".to_string())
        });

        saver
            .submit(SaveJob {
                snap_id: Uuid::new_v4(),
                step: 1,
                components: HashMap::new(),
                metadata: HashMap::new(),
            })
            .unwrap();

        // wait() should return Ok with the error in the result
        // but wait_if_busy checks last_result and propagates as Err
        // so the next operation surfaces the error
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Next submit should report the previous error
        let err = saver.submit(SaveJob {
            snap_id: Uuid::new_v4(),
            step: 2,
            components: HashMap::new(),
            metadata: HashMap::new(),
        });
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("disk full"), "expected 'disk full', got: {msg}");

        saver.shutdown().ok();
    }
}
