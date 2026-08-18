//! Which snapshots are being read, and which a merge has claimed.
//!
//! The save path takes its base from the manifest and then **releases the
//! manifest lock** before reading that base off storage — it has to, since the
//! read is the expensive part and holding the lock across it would serialise
//! the whole coordinator. The merger holds the same lock only while it decides
//! what to fold. So the two windows overlap, and a merge could publish and
//! delete the base snapshot's packs while a save was in the middle of diffing
//! against them.
//!
//! It showed up two ways. The loud one is the save failing with
//! `Checkpoint not found: snapshots/…/rank_0.pack`, because the file it was
//! reading was unlinked underneath it. The quiet one is worse: the save
//! finishes, publishes a delta, and that delta names a base the merge has
//! already removed from the manifest. It reads back as a broken `load_latest`
//! now, and startup recovery discards it later as a delta whose base is gone —
//! a checkpoint that was complete on disk, thrown away.
//!
//! Two rules close it, and they cover different halves of the window:
//!
//! 1. **A merge does not start on a snapshot somebody is already reading.**
//!    `claim` refuses, and an opportunistic merge that is refused simply runs
//!    again on the next notify — skipping one costs nothing.
//! 2. **A merge does not unlink a snapshot somebody started reading after the
//!    claim.** `wait_until_unpinned` blocks the merger thread just before the
//!    unlink, which is the only step a reader cannot survive.
//!
//! Rule 2 is what makes the pin load-bearing. Without it a reader that took
//! its pin *after* the claim held a token nobody consulted again: the claim
//! had already succeeded, so the merge deleted the packs out from under it.
//! Checking `is_doomed` in the reader instead would not close it either —
//! that check can be overtaken between the manifest read and the first byte
//! read — which is why the wait lives on the side doing the deleting.
//!
//! `is_doomed` remains, for the one decision that is genuinely about the
//! *future*: whether the save path should write a delta against this base. A
//! base that is merely being read is fine to diff against; a base a merge has
//! claimed will not exist by the time the delta is loaded, so the save writes
//! a full snapshot instead.
//!
//! ## Lock order
//!
//! This mutex is always taken **while the manifest lock is held**, and never
//! the other way round. That is what makes "look at the manifest and pin what
//! you found" a single atomic step, which is the whole point: a check that can
//! be overtaken between reading the manifest and pinning the result would
//! leave exactly the race it is here to prevent. The guards below take this
//! lock on their own during `Drop`, which is safe in that order.
//!
//! The waits are the reason that order matters more than it looks. A thread
//! blocked in `wait_until_unpinned` must not be holding the manifest lock:
//! `Core::load_in_pool` reacquires the manifest lock *after* pinning, to read
//! the base snapshot's entries, so a waiter holding it would deadlock against
//! the very reader it is waiting for. Both call sites wait only after their
//! `drop(manifest)`.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use uuid::Uuid;

/// How long a merge waits for a reader to leave before it unlinks anyway.
///
/// Reaching it means a load has been running for a quarter of an hour, which
/// in practice means the reader is wedged rather than slow. Waiting forever
/// would trade a rare lost checkpoint for a merger thread that never runs
/// again, so the wait is bounded and the timeout is reported by the caller.
pub(crate) const UNLINK_WAIT: Duration = Duration::from_secs(900);

/// The same budget for retention, which runs on the save path rather than on
/// the merger thread.
///
/// Much shorter for that reason: a merge blocking for a quarter of an hour
/// costs nothing anyone is waiting on, while a save blocking that long is a
/// checkpoint that did not happen. Past this the files are left for the next
/// startup to deal with, which is the same trade the merger makes.
pub(crate) const RETENTION_UNLINK_WAIT: Duration = Duration::from_secs(30);

/// How long a forced merge waits to claim inputs a reader is inside.
///
/// `save_final` is the caller: it folds the run into one snapshot and then
/// uploads it, so "skipped because something was being read" is not an answer
/// it can use.
pub(crate) const FORCED_CLAIM_WAIT: Duration = Duration::from_secs(300);

#[derive(Default)]
struct State {
    /// snapshot id → how many readers are inside it right now.
    pinned: HashMap<Uuid, usize>,
    /// Snapshots a merge in progress intends to remove.
    doomed: HashSet<Uuid>,
}

impl State {
    fn any_pinned(&self, ids: &[Uuid]) -> bool {
        ids.iter()
            .any(|id| self.pinned.get(id).is_some_and(|n| *n > 0))
    }
}

/// Shared registry. One per coordinator, shared with its merger.
#[derive(Default)]
pub(crate) struct InFlight {
    state: Mutex<State>,
    /// Signalled whenever a pin is released, which is the only event either
    /// wait below is waiting for.
    unpinned: Condvar,
}

impl InFlight {
    /// Mark `ids` as being read. Call with the manifest lock held, in the same
    /// critical section that took them out of the manifest.
    pub(crate) fn pin(self: &Arc<Self>, ids: &[Uuid]) -> PinGuard {
        let mut state = self.state.lock().unwrap();
        for id in ids {
            *state.pinned.entry(*id).or_insert(0) += 1;
        }
        drop(state);
        PinGuard {
            registry: Arc::clone(self),
            ids: ids.to_vec(),
        }
    }

    /// Whether a merge has already claimed this snapshot, in which case the
    /// save path must not write a delta against it — see the module docs.
    pub(crate) fn is_doomed(&self, id: Uuid) -> bool {
        self.state.lock().unwrap().doomed.contains(&id)
    }

    /// Claim `ids` for a merge, unless somebody is reading one of them.
    ///
    /// Returns `None` when the merge must stand down. Call with the manifest
    /// lock held, in the same critical section that chose `ids`.
    ///
    /// **Never waits**, and cannot be made to. It runs with the manifest lock
    /// held, and a reader that has already pinned goes on to reacquire that
    /// same lock — so a claim that blocked here would deadlock against the
    /// reader it was waiting for. A caller that needs the merge to happen
    /// rather than stand down retries from outside the lock; see
    /// [`wait_for_any_unpin`].
    ///
    /// [`wait_for_any_unpin`]: InFlight::wait_for_any_unpin
    pub(crate) fn claim(self: &Arc<Self>, ids: &[Uuid]) -> Option<ClaimGuard> {
        let mut state = self.state.lock().unwrap();
        if state.any_pinned(ids) {
            return None;
        }
        for id in ids {
            state.doomed.insert(*id);
        }
        drop(state);
        Some(ClaimGuard {
            registry: Arc::clone(self),
            ids: ids.to_vec(),
        })
    }

    /// Block until some reader — any reader — leaves, or `timeout` elapses.
    ///
    /// This is how a caller that cannot accept a refused [`claim`] waits
    /// before trying again: it holds no manifest lock, so it is free to block.
    /// Waiting on "any pin released" rather than on a specific set is
    /// deliberate — the set worth claiming is re-derived on the next attempt,
    /// under the lock, and may not be the same set.
    ///
    /// [`claim`]: InFlight::claim
    pub(crate) fn wait_for_any_unpin(&self, timeout: Duration) {
        let state = self.state.lock().unwrap();
        let _ = self.unpinned.wait_timeout(state, timeout).unwrap();
    }

    /// Block until nobody is reading any of `ids`, or `timeout` elapses.
    ///
    /// Returns whether they are actually clear. The caller is about to unlink
    /// the files these snapshots name, which is the one step a reader already
    /// inside them cannot survive.
    ///
    /// **Must not be called with the manifest lock held** — see the module
    /// docs on lock order.
    pub(crate) fn wait_until_unpinned(&self, ids: &[Uuid], timeout: Duration) -> bool {
        let mut state = self.state.lock().unwrap();
        let deadline = Instant::now() + timeout;
        while state.any_pinned(ids) {
            let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            let (next, _) = self.unpinned.wait_timeout(state, left).unwrap();
            state = next;
        }
        true
    }
}

/// Held for as long as a reader is inside the snapshots it names.
pub(crate) struct PinGuard {
    registry: Arc<InFlight>,
    ids: Vec<Uuid>,
}

impl Drop for PinGuard {
    fn drop(&mut self) {
        let mut state = self.registry.state.lock().unwrap();
        for id in &self.ids {
            match state.pinned.get_mut(id) {
                Some(n) if *n > 1 => *n -= 1,
                _ => {
                    state.pinned.remove(id);
                }
            }
        }
        drop(state);
        // A merge may be parked in `wait_until_unpinned` on exactly this.
        self.registry.unpinned.notify_all();
    }
}

/// Held for as long as a merge intends to remove the snapshots it names.
pub(crate) struct ClaimGuard {
    registry: Arc<InFlight>,
    ids: Vec<Uuid>,
}

impl Drop for ClaimGuard {
    fn drop(&mut self) {
        let mut state = self.registry.state.lock().unwrap();
        for id in &self.ids {
            state.doomed.remove(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_claim_is_refused_while_a_reader_is_inside() {
        let registry = Arc::new(InFlight::default());
        let id = Uuid::new_v4();

        let reader = registry.pin(&[id]);
        assert!(registry.claim(&[id]).is_none(), "claimed a snapshot in use");

        drop(reader);
        assert!(
            registry.claim(&[id]).is_some(),
            "the claim stayed refused after the reader left"
        );
    }

    #[test]
    fn a_claim_marks_its_snapshots_until_it_is_dropped() {
        let registry = Arc::new(InFlight::default());
        let id = Uuid::new_v4();

        assert!(!registry.is_doomed(id));
        let claim = registry.claim(&[id]).unwrap();
        assert!(registry.is_doomed(id), "a reader would still pick this up");

        drop(claim);
        assert!(!registry.is_doomed(id), "the mark outlived the merge");
    }

    /// Two readers of the same snapshot, one leaving: the other still holds it.
    #[test]
    fn pins_nest() {
        let registry = Arc::new(InFlight::default());
        let id = Uuid::new_v4();

        let first = registry.pin(&[id]);
        let second = registry.pin(&[id]);
        drop(first);
        assert!(registry.claim(&[id]).is_none(), "one reader is still inside");

        drop(second);
        assert!(registry.claim(&[id]).is_some());
    }

    /// The half the claim cannot cover: a reader that arrives *after* the
    /// merge claimed. The claim has already succeeded, so nothing refuses it —
    /// the unlink is what has to wait.
    #[test]
    fn the_unlink_waits_for_a_reader_that_arrived_after_the_claim() {
        let registry = Arc::new(InFlight::default());
        let id = Uuid::new_v4();

        let _claim = registry.claim(&[id]).expect("nobody is reading yet");
        let reader = registry.pin(&[id]);

        assert!(
            !registry.wait_until_unpinned(&[id], Duration::from_millis(50)),
            "the unlink went ahead while a reader was inside"
        );

        let waiter = {
            let registry = Arc::clone(&registry);
            std::thread::spawn(move || registry.wait_until_unpinned(&[id], UNLINK_WAIT))
        };
        drop(reader);
        assert!(
            waiter.join().unwrap(),
            "releasing the pin did not wake the unlink"
        );
    }

    /// How a forced merge gets its turn: the claim itself never blocks — it
    /// cannot, it runs under the manifest lock — so the caller parks outside
    /// that lock until a reader leaves and then tries again.
    #[test]
    fn a_refused_claim_can_wait_outside_the_lock_and_retry() {
        let registry = Arc::new(InFlight::default());
        let id = Uuid::new_v4();

        let reader = registry.pin(&[id]);
        assert!(
            registry.claim(&[id]).is_none(),
            "claimed a snapshot a reader was inside"
        );

        let retrier = {
            let registry = Arc::clone(&registry);
            std::thread::spawn(move || {
                let deadline = Instant::now() + FORCED_CLAIM_WAIT;
                loop {
                    if registry.claim(&[id]).is_some() {
                        return true;
                    }
                    let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                        return false;
                    };
                    registry.wait_for_any_unpin(left.min(Duration::from_secs(1)));
                }
            })
        };
        drop(reader);
        assert!(retrier.join().unwrap(), "the retry never got its claim");
    }
}
