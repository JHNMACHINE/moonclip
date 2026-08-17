//! The rayon pool Moonclip does its parallel work in.
//!
//! Every parallel pass in this crate — hashing, XOR-ing, shuffling,
//! compressing, decompressing, the shadow copy — used to run on rayon's
//! **global** pool, whose default is one thread per core. Two things are wrong
//! with that, and neither is about speed: a library should not requisition
//! every core on the machine without being asked, and the global pool is also
//! the application's, so a checkpoint write and a `par_iter` in user code end
//! up fighting over the same workers.
//!
//! It matters most where Moonclip is now used. With per-rank checkpointing
//! every rank on a node writes its own shard, so a 128-core box running 8 ranks
//! had 8 processes each claiming 128 threads.
//!
//! `RAYON_NUM_THREADS` is not the answer: it is process-global, so it caps the
//! application's rayon too; it has to be set before rayon is first used; and
//! nobody knows to set it.
//!
//! **This is hygiene, not a cure for a slow handoff.** Measured on 8× RTX 5060
//! Ti (1.48B, per-rank), capping the threads to the per-process quota made the
//! handoff *slower* — 11.6 s against 10.6 s — and the time turned out not to be
//! inside this crate at all. Sizing the pool correctly is worth doing because
//! it is correct, and claiming any throughput for it would be inventing a
//! result the measurements refused to give.

use std::sync::OnceLock;

use rayon::{ThreadPool, ThreadPoolBuilder};

/// Env var that overrides the computed size. Wins over everything.
const THREADS_VAR: &str = "MOONCLIP_THREADS";

/// How many Moonclip processes share this node. `torchrun` sets it for every
/// worker it spawns, which is exactly the case that made the global pool a
/// problem.
///
/// Deliberately not `WORLD_SIZE`: that counts ranks across all nodes, and it is
/// not what has to be divided up. Nor the coordinator's own `world_size`, which
/// Ravex pins to 1 per rank on purpose — each rank saves its own shard as a
/// single-rank store — so it says nothing about how many processes are here.
const LOCAL_WORLD_VAR: &str = "LOCAL_WORLD_SIZE";

/// The size the pool should have, given what the machine and the environment
/// say. Pure, so the policy can be tested without touching a global.
///
/// `explicit` (from `MOONCLIP_THREADS`) wins outright: someone who sets it
/// knows something about the box that this crate cannot see. Otherwise the
/// cores are split evenly between the processes sharing the node, which is the
/// quota each of them would get anyway once they started competing — the
/// difference is that they no longer pay the scheduler to discover it.
pub fn desired_threads(
    cores: usize,
    local_world_size: Option<usize>,
    explicit: Option<usize>,
) -> usize {
    if let Some(threads) = explicit {
        return threads.max(1);
    }
    let share = local_world_size.unwrap_or(1).max(1);
    (cores.max(1) / share).max(1)
}

fn positive_env(name: &str) -> Option<usize> {
    let raw = std::env::var(name).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.parse::<usize>() {
        Ok(0) | Err(_) => {
            // A knob that was set and silently ignored is worse than one that
            // does not exist: the next measurement gets attributed to a setting
            // that never took effect.
            eprintln!("[Moonclip] Ignoring {name}={raw:?}: expected a positive integer");
            None
        }
        Ok(value) => Some(value),
    }
}

/// Moonclip's own pool, or `None` if rayon refused to build one.
///
/// A failure here is not worth taking a training run down for: the work still
/// runs, on the global pool, exactly as it did before this module existed.
fn pool() -> Option<&'static ThreadPool> {
    static POOL: OnceLock<Option<ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let threads = desired_threads(
            cores,
            positive_env(LOCAL_WORLD_VAR),
            positive_env(THREADS_VAR),
        );
        match ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("moonclip-rayon-{i}"))
            .build()
        {
            Ok(pool) => Some(pool),
            Err(e) => {
                eprintln!("[Moonclip] Could not build a private thread pool ({e}); using rayon's global pool");
                None
            }
        }
    })
    .as_ref()
}

/// Run `f` in Moonclip's pool.
///
/// Call this at the entry points of the parallel work, not at each `par_iter`:
/// `install` propagates to nested rayon calls, so the parallel passes
/// underneath — in `compression`, `delta`, `cast`, `shuffle`, `tensor` — need no
/// change and cannot be forgotten.
pub fn install<R, F>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    match pool() {
        Some(pool) => pool.install(f),
        None => f(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The size is a per-process quota, not the whole machine.
    #[test]
    fn the_cores_are_split_between_the_processes_on_the_node() {
        assert_eq!(desired_threads(128, Some(8), None), 16);
        assert_eq!(desired_threads(128, None, None), 128);
    }

    /// `MOONCLIP_THREADS` is the escape hatch, so it has to beat the division —
    /// including upwards, on a box where the ranks are not all checkpointing.
    #[test]
    fn an_explicit_setting_wins_over_the_computed_share() {
        assert_eq!(desired_threads(128, Some(8), Some(64)), 64);
        assert_eq!(desired_threads(8, Some(8), Some(32)), 32);
    }

    /// More ranks than cores, a single core, a zero someone passed by hand: a
    /// pool of zero threads is a rayon panic, and this runs inside every save.
    #[test]
    fn the_pool_never_comes_out_empty() {
        assert_eq!(desired_threads(4, Some(8), None), 1);
        assert_eq!(desired_threads(1, None, None), 1);
        assert_eq!(desired_threads(0, Some(0), Some(0)), 1);
    }

    /// The point of the module: the work leaves the caller's pool for ours, and
    /// nested rayon inside it sees the same bounded pool rather than the global
    /// one.
    #[test]
    fn the_work_runs_on_moonclip_threads() {
        use rayon::prelude::*;

        let (name, nested_threads) = install(|| {
            let name = std::thread::current().name().map(str::to_string);
            // A nested parallel pass, which is what every call site really is.
            let nested = (0..4)
                .into_par_iter()
                .map(|_| rayon::current_num_threads())
                .max()
                .unwrap();
            (name, nested)
        });

        assert!(
            name.as_deref().is_some_and(|n| n.starts_with("moonclip-rayon-")),
            "the save path ran on {name:?}, not on a Moonclip pool thread"
        );
        assert_eq!(nested_threads, pool().unwrap().current_num_threads());
    }
}
