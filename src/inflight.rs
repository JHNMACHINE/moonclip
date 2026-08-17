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
//! The rule here is the cheapest one that closes both: **a merge does not
//! consume a snapshot somebody is reading, and a reader does not pick up a
//! snapshot a merge has claimed.** Neither side waits for the other. A merge
//! that finds its inputs busy returns without doing anything and runs on the
//! next notify — merging is opportunistic, and skipping one costs nothing. A
//! save that finds its base claimed writes a full snapshot instead of a delta:
//! more bytes for that one step, and only while a merge is actually running.
//!
//! ## Lock order
//!
//! This mutex is always taken **while the manifest lock is held**, and never
//! the other way round. That is what makes "look at the manifest and pin what
//! you found" a single atomic step, which is the whole point: a check that can
//! be overtaken between reading the manifest and pinning the result would
//! leave exactly the race it is here to prevent. The guards below take this
//! lock on their own during `Drop`, which is safe in that order.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use uuid::Uuid;

#[derive(Default)]
struct State {
    /// snapshot id → how many readers are inside it right now.
    pinned: HashMap<Uuid, usize>,
    /// Snapshots a merge in progress intends to remove.
    doomed: HashSet<Uuid>,
}

/// Shared registry. Cheap to clone; one per coordinator.
#[derive(Default)]
pub(crate) struct InFlight {
    state: Mutex<State>,
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

    /// Whether a merge has already claimed this snapshot, in which case a
    /// reader must not start a piece of work that outlives the merge.
    pub(crate) fn is_doomed(&self, id: Uuid) -> bool {
        self.state.lock().unwrap().doomed.contains(&id)
    }

    /// Claim `ids` for a merge, unless somebody is reading one of them.
    ///
    /// Returns `None` when the merge must stand down. Call with the manifest
    /// lock held, in the same critical section that chose `ids`.
    pub(crate) fn claim(self: &Arc<Self>, ids: &[Uuid]) -> Option<ClaimGuard> {
        let mut state = self.state.lock().unwrap();
        if ids
            .iter()
            .any(|id| state.pinned.get(id).is_some_and(|n| *n > 0))
        {
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
}
