//! The coordinator's own tests.
//!
//! A child module, not `tests/`: these reach into `Core`, the manifest and
//! the packs on disk, which an integration test cannot see.

use std::sync::atomic::AtomicUsize;
use std::sync::mpsc;
use std::thread;

use super::*;
use crate::storage::LocalStorage;

fn make_coordinator(dir: &std::path::Path) -> Coordinator {
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir).unwrap());
    let config = CoordinatorConfig {
        world_size: 1,
        rank: 0,
        compression: CompressionAlgo::Zstd { level: 1 },
        retention: RetentionPolicy {
            max_full_snapshots: 5,
            max_deltas_per_full: 10,
            full_snapshot_every_steps: 10000,
            max_total_snapshots: None,
        },
        delta_max_ratio: 0.95,
        ..Default::default()
    };
    Coordinator::new(storage, config).unwrap()
}

fn sample_tensors(seed: u8) -> Vec<TensorData> {
    vec![TensorData {
        name: "w".into(),
        shape: vec![8192],
        dtype: "uint8".into(),
        data: vec![seed; 8192],
    }]
}

#[test]
fn single_rank_save_load() {
    let dir = tempfile::tempdir().unwrap();
    let coord = make_coordinator(dir.path());

    let snap_id = coord.save(100, sample_tensors(42), HashMap::new()).unwrap();
    let loaded = coord.load(snap_id).unwrap();
    assert_eq!(loaded["w"], vec![42u8; 8192]);
}

/// GPU-128 along the whole save path, not one `put`: on Windows a reader
/// holding `manifest.json` open while the background writer persisted the next
/// snapshot failed that save — "Background save failed ... Access denied" —
/// and the checkpoint was gone. Five runs in five, from Ravex's audit trail.
///
/// The assertion that matters is the last one. A fresh coordinator on the same
/// directory knows about the snapshot only through the manifest on disk, so
/// it finds it only if the rename that was being refused went through.
#[cfg(windows)]
#[test]
fn a_reader_holding_the_manifest_open_does_not_cost_a_save() {
    let dir = tempfile::tempdir().unwrap();
    let coord = make_coordinator(dir.path());
    coord.save(1, sample_tensors(1), HashMap::new()).unwrap();
    coord.flush().unwrap();

    let held = std::fs::File::open(dir.path().join("manifest.json")).unwrap();
    let reader = thread::spawn(move || {
        thread::sleep(std::time::Duration::from_millis(300));
        drop(held);
    });

    let snap_id = coord.save(2, sample_tensors(2), HashMap::new()).unwrap();
    coord.flush().unwrap();
    reader.join().unwrap();
    drop(coord);

    let reopened = make_coordinator(dir.path());
    assert_eq!(reopened.load(snap_id).unwrap()["w"], vec![2u8; 8192]);
}

#[test]
fn sync_save_mode_roundtrips() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let config = CoordinatorConfig {
        compression: CompressionAlgo::Zstd { level: 1 },
        async_save: false,
        ..Default::default()
    };
    let coord = Coordinator::new(storage, config).unwrap();

    let snap_id = coord.save(100, sample_tensors(7), HashMap::new()).unwrap();
    let loaded = coord.load(snap_id).unwrap();
    assert_eq!(loaded["w"], vec![7u8; 8192]);
}

#[test]
fn delta_chain_multiple_steps() {
    let dir = tempfile::tempdir().unwrap();
    let coord = make_coordinator(dir.path());

    let mut data = vec![0u8; 10_000];
    coord.save(100, vec![TensorData {
        name: "w".into(), shape: vec![10_000], dtype: "uint8".into(), data: data.clone(),
    }], HashMap::new()).unwrap();

    for step in (200..=500).step_by(100) {
        let idx = step as usize % data.len();
        data[idx] = (step / 100) as u8;
        coord.save(step, vec![TensorData {
            name: "w".into(), shape: vec![10_000], dtype: "uint8".into(), data: data.clone(),
        }], HashMap::new()).unwrap();
    }

    let (_, loaded) = coord.load_latest().unwrap();
    assert_eq!(loaded["w"], data);
}

#[test]
fn retention_caps_total_snapshots() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let config = CoordinatorConfig {
        world_size: 1,
        rank: 0,
        compression: CompressionAlgo::Zstd { level: 1 },
        retention: RetentionPolicy {
            max_full_snapshots: 2,
            max_deltas_per_full: 5,
            full_snapshot_every_steps: 5, // full every 5 steps
            max_total_snapshots: Some(4), // hard cap at 4
        },
        delta_max_ratio: 0.95,
        ..Default::default()
    };
    let coord = Coordinator::new(storage, config).unwrap();

    // Save 10 steps → should never exceed 5 total
    for i in 0..10 {
        coord.save(i, vec![TensorData {
            name: "w".into(), shape: vec![100], dtype: "uint8".into(), data: vec![i as u8; 100],
        }], HashMap::new()).unwrap();
    }

    let snaps = coord.list_snapshots();
    assert!(
        snaps.len() <= 5,
        "Expected <=5 total snapshots, got {}",
        snaps.len()
    );
}

#[test]
fn retention_policy_removes_old_snapshots() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let config = CoordinatorConfig {
        world_size: 1,
        rank: 0,
        compression: CompressionAlgo::Zstd { level: 1 },
        retention: RetentionPolicy {
            max_full_snapshots: 2,
            max_deltas_per_full: 5,
            full_snapshot_every_steps: 1, // force full every step
            max_total_snapshots: None,
        },
        delta_max_ratio: 0.95,
        ..Default::default()
    };
    let coord = Coordinator::new(storage, config).unwrap();

    for i in 0..6 {
        coord.save(i, vec![TensorData {
            name: "w".into(), shape: vec![100], dtype: "uint8".into(), data: vec![i as u8; 100],
        }], HashMap::new()).unwrap();
    }

    let snaps = coord.list_snapshots();
    let full_count = snaps.iter().filter(|s| !s.is_delta).count();
    assert!(
        full_count <= 2,
        "Expected <=2 full snapshots, got {full_count}"
    );
}

#[test]
fn empty_tensor_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let coord = make_coordinator(dir.path());

    let tensors = vec![TensorData {
        name: "empty".into(), shape: vec![0], dtype: "float32".into(), data: vec![],
    }];
    let snap_id = coord.save(1, tensors, HashMap::new()).unwrap();
    let loaded = coord.load(snap_id).unwrap();
    assert_eq!(loaded["empty"].len(), 0);
}

#[test]
fn many_tensors_save_load() {
    let dir = tempfile::tempdir().unwrap();
    let coord = make_coordinator(dir.path());

    let tensors: Vec<TensorData> = (0..50)
        .map(|i| TensorData {
            name: format!("layer.{}.weight", i),
            shape: vec![64, 64],
            dtype: "float32".into(),
            data: vec![i as u8; 64 * 64 * 4],
        })
        .collect();

    let snap_id = coord.save(100, tensors.clone(), HashMap::new()).unwrap();
    let loaded = coord.load(snap_id).unwrap();

    assert_eq!(loaded.len(), 50);
    for t in &tensors {
        assert_eq!(loaded[&t.name], t.data);
    }
}

#[test]
fn special_characters_in_tensor_names() {
    let dir = tempfile::tempdir().unwrap();
    let coord = make_coordinator(dir.path());

    let tensors = vec![
        TensorData {
            name: "module.layers.0.self_attn.q_proj.weight".into(),
            shape: vec![64], dtype: "float32".into(), data: vec![1u8; 256],
        },
        TensorData {
            name: "model/encoder/block_0/layer_0".into(),
            shape: vec![32], dtype: "float32".into(), data: vec![2u8; 128],
        },
    ];

    let snap_id = coord.save(1, tensors.clone(), HashMap::new()).unwrap();
    let loaded = coord.load(snap_id).unwrap();

    assert_eq!(loaded["module.layers.0.self_attn.q_proj.weight"], tensors[0].data);
    assert_eq!(loaded["model/encoder/block_0/layer_0"], tensors[1].data);
}

#[test]
fn force_full_after_n_steps() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let config = CoordinatorConfig {
        world_size: 1,
        rank: 0,
        compression: CompressionAlgo::Zstd { level: 1 },
        retention: RetentionPolicy {
            max_full_snapshots: 10,
            max_deltas_per_full: 100,
            full_snapshot_every_steps: 3,
            max_total_snapshots: None,
        },
        delta_max_ratio: 0.95,
        ..Default::default()
    };
    let coord = Coordinator::new(storage, config).unwrap();

    let base = vec![0u8; 8192];
    for step in 0..9 {
        let mut d = base.clone();
        d[0] = step as u8;
        coord.save(step, vec![TensorData {
            name: "w".into(), shape: vec![8192], dtype: "uint8".into(), data: d,
        }], HashMap::new()).unwrap();
    }

    let snaps = coord.list_snapshots();
    let full_count = snaps.iter().filter(|s| !s.is_delta).count();
    assert!(full_count >= 3, "Expected >=3 full snapshots, got {full_count}");
}

#[test]
fn load_nonexistent_snapshot_errors() {
    let dir = tempfile::tempdir().unwrap();
    let coord = make_coordinator(dir.path());
    let fake_id = uuid::Uuid::new_v4();
    assert!(coord.load(fake_id).is_err());
}

#[test]
fn metadata_preserved() {
    let dir = tempfile::tempdir().unwrap();
    let coord = make_coordinator(dir.path());

    let mut meta = HashMap::new();
    meta.insert("loss".into(), "0.123".into());
    meta.insert("lr".into(), "3e-4".into());
    meta.insert("epoch".into(), "5".into());

    coord.save(100, sample_tensors(0), meta).unwrap();

    let snaps = coord.list_snapshots();
    assert_eq!(snaps[0].metadata["loss"], "0.123");
    assert_eq!(snaps[0].metadata["lr"], "3e-4");
    assert_eq!(snaps[0].metadata["epoch"], "5");
}

/// Every concurrent `save()` must be in the manifest once `flush()`
/// returns.
///
/// Regression: `submit` waited for the saver to go idle, released the
/// lock, then re-acquired it to set `busy`. Two threads could both clear
/// the wait and both queue a job against what is a single flag, so the
/// worker cleared `busy` after the first of them finished — `flush()`
/// returned with a save still queued and the manifest came up short.
///
/// Repeated, because one pass through a lost-update window is not
/// guaranteed to lose anything: the pre-fix code failed a few runs in ten.
#[test]
fn concurrent_saves_all_land() {
    const THREADS: u64 = 10;
    const TRIALS: usize = 20;

    for trial in 0..TRIALS {
        let dir = tempfile::tempdir().unwrap();
        let storage: Arc<dyn StorageBackend> =
            Arc::new(LocalStorage::new(dir.path()).unwrap());
        let config = CoordinatorConfig {
            world_size: 1,
            rank: 0,
            compression: CompressionAlgo::Zstd { level: 1 },
            retention: RetentionPolicy {
                // Deliberately generous: a short manifest must mean a lost
                // save, never retention pruning.
                max_full_snapshots: 1000,
                max_deltas_per_full: 1000,
                full_snapshot_every_steps: 100_000,
                max_total_snapshots: None,
            },
            delta_max_ratio: 0.95,
            ..Default::default()
        };
        let coord = Arc::new(Coordinator::new(storage, config).unwrap());

        let handles: Vec<_> = (1..=THREADS)
            .map(|step| {
                let c = Arc::clone(&coord);
                thread::spawn(move || c.save(step, sample_tensors(step as u8), HashMap::new()))
            })
            .collect();
        for h in handles {
            h.join().expect("worker panicked").expect("save failed");
        }
        coord.flush().unwrap();

        assert_eq!(
            coord.list_snapshots().len(),
            THREADS as usize,
            "trial {trial}: a concurrent save is missing from the manifest"
        );
    }
}

/// The queue wait is available to the caller, not only to a log line.
///
/// GPU-61: a caller timing its own save could not tell the shadow copy
/// from the wait for the previous writer, so a phase dominated by a memcpy
/// was read as a writer in deficit — and an issue was opened against the
/// writer on the strength of it.
#[test]
fn the_queue_wait_is_reported_to_the_caller() {
    let dir = tempfile::tempdir().unwrap();
    let coord = make_coordinator(dir.path());

    // Nothing has ever been in flight, so there was nothing to wait for.
    coord.save(1, sample_tensors(1), HashMap::new()).unwrap();
    let first = coord.last_queue_wait();

    // No flush in between. `submit` sets `busy` before it returns, so the
    // second save cannot get past the condvar until the first has drained
    // — the wait is forced by the ordering, not hoped for from timing.
    let big = vec![TensorData {
        name: "w".into(),
        shape: vec![16 << 20],
        dtype: "uint8".into(),
        data: (0..(16u32 << 20)).map(|i| (i ^ (i >> 7)) as u8).collect(),
    }];
    coord.save(2, big.clone(), HashMap::new()).unwrap();
    coord.save(3, big, HashMap::new()).unwrap();
    let queued = coord.last_queue_wait();
    coord.flush().unwrap();

    assert!(
        queued > first,
        "a save submitted behind an in-flight one reported no wait \
         (first {first:?}, queued {queued:?})"
    );
    assert!(
        queued > Duration::ZERO,
        "the wait for the previous writer came back as zero"
    );
}

/// With no background saver there is no queue to wait in. The write then
/// happens on the calling thread, and calling that backpressure would name
/// the write itself as the thing standing in front of the write.
#[test]
fn a_synchronous_save_waits_for_nobody() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let config = CoordinatorConfig {
        world_size: 1,
        rank: 0,
        async_save: false,
        compression: CompressionAlgo::Zstd { level: 1 },
        ..Default::default()
    };
    let coord = Coordinator::new(storage, config).unwrap();

    coord.save(1, sample_tensors(1), HashMap::new()).unwrap();
    coord.save(2, sample_tensors(2), HashMap::new()).unwrap();

    assert_eq!(coord.last_queue_wait(), Duration::ZERO);
}

#[test]
fn pack_file_created() {
    let dir = tempfile::tempdir().unwrap();
    let coord = make_coordinator(dir.path());

    let tensors = vec![
        TensorData { name: "a".into(), shape: vec![100], dtype: "uint8".into(), data: vec![1u8; 100] },
        TensorData { name: "b".into(), shape: vec![200], dtype: "uint8".into(), data: vec![2u8; 200] },
    ];

    coord.save(1, tensors, HashMap::new()).unwrap();
    coord.flush().unwrap(); // drain the background save

    // Check that a .pack file exists (not individual .bin files)
    let manifest = coord.core.manifest.lock().unwrap();
    let snap = &manifest.snapshots[0];
    let re = snap.ranks.get(&0).unwrap();
    assert!(re.pack_file.is_some(), "Expected pack_file to be set");
    assert!(
        re.pack_file.as_ref().unwrap().ends_with(".pack"),
        "Expected .pack extension"
    );
}

// ── Reclaiming orphaned snapshots ───────────────────────────────

/// Files under a snapshot directory, whatever their depth.
fn snapshot_files(storage: &dyn StorageBackend, id: &str) -> Vec<String> {
    storage
        .list("snapshots")
        .unwrap_or_default()
        .into_iter()
        .filter(|f| f.starts_with(&format!("snapshots/{id}/")))
        .collect()
}

/// A snapshot directory holding a pack with no descriptor — either a
/// write that died before the header landed, or a pack from before the
/// format carried one.
fn orphan_on_disk(storage: &dyn StorageBackend, id: Uuid) {
    storage
        .put(&format!("snapshots/{id}/rank_0.pack"), &[7u8; 4096])
        .unwrap();
}

/// Rewrite the manifest without `drop`, leaving its data and sidecar on
/// disk — exactly the state a process killed between the two writes leaves.
fn forget_snapshot(storage: &dyn StorageBackend, drop: Uuid) {
    let raw = storage.get("manifest.json").unwrap();
    let end = raw.iter().rposition(|&b| b != 0).map(|i| i + 1).unwrap_or(0);
    let mut manifest: Manifest = serde_json::from_slice(&raw[..end]).unwrap();
    manifest.snapshots.retain(|s| s.id != drop);
    storage
        .put("manifest.json", &serde_json::to_vec_pretty(&manifest).unwrap())
        .unwrap();
}

fn open(storage: &Arc<dyn StorageBackend>) -> Coordinator {
    Coordinator::new(
        Arc::clone(storage),
        CoordinatorConfig {
            compression: CompressionAlgo::Zstd { level: 1 },
            retention: RetentionPolicy {
                full_snapshot_every_steps: 1,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap()
}

/// The case the whole sidecar exists for: a checkpoint that finished
/// writing, whose manifest update never happened, comes back.
#[test]
fn a_complete_checkpoint_missing_from_the_manifest_is_recovered() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());

    let coord = open(&storage);
    coord.save(0, evolving_state(0), HashMap::new()).unwrap();
    coord.save(1, evolving_state(1), HashMap::new()).unwrap();
    coord.flush().unwrap();
    let lost = coord.list_snapshots()[1].id;
    drop(coord);

    forget_snapshot(storage.as_ref(), lost);

    let restarted = open(&storage);
    let steps: Vec<u64> = restarted.list_snapshots().iter().map(|s| s.step).collect();
    assert_eq!(steps, vec![0, 1], "the finished checkpoint was not recovered");

    // Recovered is only worth something if it reads back.
    let loaded = restarted.load(lost).expect("recovered snapshot must load");
    assert_eq!(loaded.get("w").unwrap()[0], 1u8);
}

#[test]
fn a_recovered_checkpoint_is_written_back_to_the_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());

    let coord = open(&storage);
    coord.save(0, evolving_state(0), HashMap::new()).unwrap();
    coord.flush().unwrap();
    let lost = coord.list_snapshots()[0].id;
    drop(coord);

    forget_snapshot(storage.as_ref(), lost);
    drop(open(&storage)); // recovers, and should persist

    // A second restart must find it already in the manifest rather than
    // rediscovering it — otherwise the recovery only ever lived in memory
    // and the next crash loses it again.
    let raw = storage.get("manifest.json").unwrap();
    let end = raw.iter().rposition(|&b| b != 0).map(|i| i + 1).unwrap_or(0);
    let manifest: Manifest = serde_json::from_slice(&raw[..end]).unwrap();
    assert_eq!(manifest.snapshots.len(), 1);
    assert_eq!(manifest.snapshots[0].id, lost);
}

/// Presence is not integrity. A pack of the right length whose bytes are
/// wrong must not be readmitted: it would pass startup and fail at resume,
/// which is the worst moment to find out.
#[test]
fn a_checkpoint_with_corrupted_data_is_not_recovered() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());

    let coord = open(&storage);
    coord.save(0, evolving_state(0), HashMap::new()).unwrap();
    coord.flush().unwrap();
    let lost = coord.list_snapshots()[0].id;
    drop(coord);

    forget_snapshot(storage.as_ref(), lost);

    // Flip a byte inside the compressed data, keeping the pack's length.
    //
    // Where matters: LocalStorage pads to a page boundary, so most of this
    // file is zero padding that nothing ever reads. Corrupting there proves
    // nothing — the check is driven by each tensor's recorded offset and
    // size, which the sidecar carries.
    let pack_path = format!("snapshots/{lost}/rank_0.pack");
    let header = storage
        .get_range(&pack_path, 0, pack::HEADER_LEN as usize)
        .unwrap();
    let (offset, length) = pack::decode_header(&header).expect("our own header");
    let encoded = storage.get_range(&pack_path, offset, length as usize).unwrap();
    let described: pack::PackDescriptor = serde_json::from_slice(&encoded).unwrap();
    let tensor = &described.rank.tensors[0];
    assert!(tensor.compressed_size > 8, "nothing to corrupt");

    let mut bytes = storage.get(&pack_path).unwrap();
    bytes[tensor.offset as usize + tensor.compressed_size as usize / 2] ^= 0xff;
    storage.put(&pack_path, &bytes).unwrap();

    let restarted = open(&storage);
    assert!(
        restarted.list_snapshots().is_empty(),
        "corrupted data was readmitted to the manifest"
    );
    assert!(
        !storage.exists(&pack_path).unwrap_or(false),
        "the corrupted snapshot should have been discarded"
    );
}

/// A delta's pack describes itself exactly as a full's does — same code
/// path, `base_snapshot_id` set instead of null — so it recovers too, and
/// still reconstructs against its base afterwards.
#[test]
fn a_delta_is_recovered_and_still_applies_to_its_base() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let config = || CoordinatorConfig {
        compression: CompressionAlgo::Zstd { level: 1 },
        // Keep the second save a delta against the first.
        retention: RetentionPolicy {
            full_snapshot_every_steps: 1000,
            ..Default::default()
        },
        ..Default::default()
    };

    let coord = Coordinator::new(Arc::clone(&storage), config()).unwrap();
    coord.save(0, evolving_state(0), HashMap::new()).unwrap();
    coord.save(1, evolving_state(1), HashMap::new()).unwrap();
    coord.flush().unwrap();
    let snaps = coord.list_snapshots();
    assert!(snaps[1].is_delta, "second save should be a delta");
    let delta = snaps[1].id;
    drop(coord);

    // Lose only the delta's manifest entry; its base stays recorded.
    forget_snapshot(storage.as_ref(), delta);

    let restarted = Coordinator::new(Arc::clone(&storage), config()).unwrap();
    let steps: Vec<u64> = restarted.list_snapshots().iter().map(|s| s.step).collect();
    assert_eq!(steps, vec![0, 1], "the delta was not recovered");

    // Recovering a delta is only meaningful if the XOR still resolves
    // against the base it names.
    let loaded = restarted.load(delta).expect("recovered delta must load");
    assert_eq!(loaded.get("w").unwrap()[0], 1u8);
}

/// A merge collapses the base and every delta after it into one snapshot,
/// then deletes the originals. From that moment it is the only copy of the
/// run's history — so it is the snapshot that must survive a manifest
/// update that never landed, and the one whose loss costs everything.
#[test]
fn a_merged_snapshot_is_recovered_like_any_other() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let config = || CoordinatorConfig {
        compression: CompressionAlgo::Zstd { level: 1 },
        // Keep the second save a delta, so the merge has a chain to fold.
        retention: RetentionPolicy {
            full_snapshot_every_steps: 1000,
            ..Default::default()
        },
        ..Default::default()
    };

    let coord = Coordinator::new(Arc::clone(&storage), config()).unwrap();
    coord.save(0, evolving_state(0), HashMap::new()).unwrap();
    coord.save(1, evolving_state(1), HashMap::new()).unwrap();
    coord.flush().unwrap();
    assert!(coord.list_snapshots()[1].is_delta, "second save should be a delta");
    drop(coord);

    // Merged directly rather than through `merge_now`: the merger runs on
    // its own thread with no completion to wait on, and this test is about
    // what the merge writes, not when.
    let raw = storage.get("manifest.json").unwrap();
    let end = raw.iter().rposition(|&b| b != 0).map(|i| i + 1).unwrap_or(0);
    let manifest: Manifest = serde_json::from_slice(&raw[..end]).unwrap();
    let manifest = Arc::new(Mutex::new(manifest));
    crate::merger::do_full_merge(
        &storage,
        &manifest,
        &CompressionAlgo::Zstd { level: 1 },
        &Arc::new(InFlight::default()),
        &Arc::new(crate::remote_sync::PendingDeletes::default()),
    )
    .unwrap();
    let merged = {
        let m = manifest.lock().unwrap();
        assert_eq!(m.snapshots.len(), 1, "base and delta collapse into one");
        m.snapshots[0].id
    };

    forget_snapshot(storage.as_ref(), merged);

    let restarted = Coordinator::new(Arc::clone(&storage), config()).unwrap();
    let snaps = restarted.list_snapshots();
    assert_eq!(snaps.len(), 1, "the merged snapshot was not recovered");
    assert_eq!(snaps[0].id, merged);
    assert!(!snaps[0].is_delta, "a merge yields a full");

    // Recovered is only worth something if it reads back as the state the
    // merge computed — the last step's weights, not the base's.
    let loaded = restarted.load(merged).expect("recovered merge must load");
    assert_eq!(loaded.get("w").unwrap()[0], 1u8);
}

/// A delta reconstructs nothing without its base.
#[test]
fn a_delta_whose_base_is_gone_is_not_recovered() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());

    let coord = Coordinator::new(
        Arc::clone(&storage),
        CoordinatorConfig {
            compression: CompressionAlgo::Zstd { level: 1 },
            // Keep the second save a delta against the first.
            retention: RetentionPolicy {
                full_snapshot_every_steps: 1000,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    coord.save(0, evolving_state(0), HashMap::new()).unwrap();
    coord.save(1, evolving_state(1), HashMap::new()).unwrap();
    coord.flush().unwrap();
    let snaps = coord.list_snapshots();
    let (base, delta) = (snaps[0].id, snaps[1].id);
    assert!(snaps[1].is_delta, "second save should be a delta");
    drop(coord);

    // Lose both entries, and the base's data with them.
    forget_snapshot(storage.as_ref(), base);
    forget_snapshot(storage.as_ref(), delta);
    for file in snapshot_files(storage.as_ref(), &base.to_string()) {
        storage.delete(&file).unwrap();
    }

    let restarted = open(&storage);
    assert!(
        restarted.list_snapshots().is_empty(),
        "a delta was recovered with no base to apply it to"
    );
}

#[test]
fn a_snapshot_without_a_description_is_reclaimed_at_startup() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());

    // A real snapshot, recorded in the manifest by a normal save.
    let coord = Coordinator::new(
        Arc::clone(&storage),
        CoordinatorConfig {
            compression: CompressionAlgo::Zstd { level: 1 },
            ..Default::default()
        },
    )
    .unwrap();
    coord.save(0, evolving_state(0), HashMap::new()).unwrap();
    coord.flush().unwrap();
    let kept = coord.list_snapshots()[0].id;
    drop(coord);

    let orphan = Uuid::new_v4();
    orphan_on_disk(storage.as_ref(), orphan);
    assert!(!snapshot_files(storage.as_ref(), &orphan.to_string()).is_empty());

    // Starting a coordinator is what reclaims: the run that left the
    // orphan is gone, and this is the moment nothing can be in flight.
    let restarted = Coordinator::new(
        Arc::clone(&storage),
        CoordinatorConfig {
            compression: CompressionAlgo::Zstd { level: 1 },
            ..Default::default()
        },
    )
    .unwrap();

    assert!(
        snapshot_files(storage.as_ref(), &orphan.to_string()).is_empty(),
        "the orphaned snapshot's data is still on disk"
    );
    assert!(
        !storage
            .exists(&format!("snapshots/{orphan}"))
            .unwrap_or(false),
        "the empty directory was left behind"
    );
    assert!(
        !snapshot_files(storage.as_ref(), &kept.to_string()).is_empty(),
        "a snapshot the manifest lists must survive"
    );
    assert_eq!(restarted.list_snapshots().len(), 1);
    assert!(restarted.load(kept).is_ok(), "the kept snapshot must still load");
}

/// The dangerous mistake would be reclaiming by directory listing alone and
/// deleting a snapshot that is perfectly good.
#[test]
fn a_listed_snapshot_is_never_reclaimed() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let config = || CoordinatorConfig {
        compression: CompressionAlgo::Zstd { level: 1 },
        retention: RetentionPolicy {
            full_snapshot_every_steps: 1,
            ..Default::default()
        },
        ..Default::default()
    };

    let coord = Coordinator::new(Arc::clone(&storage), config()).unwrap();
    for step in 0..3u64 {
        coord.save(step, evolving_state(step), HashMap::new()).unwrap();
    }
    coord.flush().unwrap();
    let before: Vec<Uuid> = coord.list_snapshots().iter().map(|s| s.id).collect();
    drop(coord);

    let restarted = Coordinator::new(Arc::clone(&storage), config()).unwrap();
    let after: Vec<Uuid> = restarted.list_snapshots().iter().map(|s| s.id).collect();
    assert_eq!(before, after, "restarting must not remove any snapshot");

    for id in &after {
        let loaded = restarted.load(*id).unwrap_or_else(|e| {
            panic!("snapshot {id} was listed after a restart but cannot be read: {e}")
        });
        assert!(loaded.contains_key("w"));
    }
}

/// The mirror case: the manifest names a snapshot whose files are gone.
/// Reclamation must not read that as a reason to do anything drastic, and
/// above all must not panic — this runs inside the constructor, so a panic
/// here means no coordinator at all.
#[test]
fn a_manifest_entry_with_no_files_does_not_panic() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let config = || CoordinatorConfig {
        compression: CompressionAlgo::Zstd { level: 1 },
        ..Default::default()
    };

    let coord = Coordinator::new(Arc::clone(&storage), config()).unwrap();
    coord.save(0, evolving_state(0), HashMap::new()).unwrap();
    coord.flush().unwrap();
    let id = coord.list_snapshots()[0].id;
    drop(coord);

    // Delete the data but leave the manifest entry, as a half-finished
    // cleanup or a truncated volume would.
    for file in snapshot_files(storage.as_ref(), &id.to_string()) {
        storage.delete(&file).unwrap();
    }

    let restarted = Coordinator::new(Arc::clone(&storage), config()).unwrap();
    assert_eq!(
        restarted.list_snapshots().len(),
        1,
        "the entry is still listed; reclamation does not edit the manifest"
    );
    // Reading it fails, which is honest — but it fails as an error.
    assert!(restarted.load(id).is_err());
}

/// Anything under `snapshots/` that is not a snapshot was put there by
/// something else, and a cleanup routine that guesses is how unrelated data
/// gets deleted.
#[test]
fn files_that_are_not_snapshots_are_left_alone() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());

    storage.put("snapshots/notes.txt", b"someone put this here").unwrap();
    storage.put("snapshots/scratch/data.bin", b"and this").unwrap();

    let _coord = Coordinator::new(
        Arc::clone(&storage),
        CoordinatorConfig {
            compression: CompressionAlgo::Zstd { level: 1 },
            ..Default::default()
        },
    )
    .unwrap();

    assert!(storage.exists("snapshots/notes.txt").unwrap());
    assert!(storage.exists("snapshots/scratch/data.bin").unwrap());
}

/// Ranks 1..N write into directories rank 0 created; if they each reclaimed
/// on the way up, one rank's startup would delete another's snapshot.
#[test]
fn only_rank_zero_reclaims() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());

    let orphan = Uuid::new_v4();
    orphan_on_disk(storage.as_ref(), orphan);

    let _rank_one = Coordinator::new(
        Arc::clone(&storage),
        CoordinatorConfig {
            world_size: 2,
            rank: 1,
            compression: CompressionAlgo::Zstd { level: 1 },
            ..Default::default()
        },
    )
    .unwrap();

    assert!(
        !snapshot_files(storage.as_ref(), &orphan.to_string()).is_empty(),
        "a non-zero rank reclaimed, and could have deleted a snapshot \
         another rank was still writing into"
    );
}

// ── Retention ───────────────────────────────────────────────────
//
// Retention deletes files. The tests above count what survives, which is
// the easy half: a count still passes when the survivors have been ruined.
// A delta whose base was pruned is a snapshot the manifest still lists and
// nothing can read, and a training run finds that out at resume.

/// Weights that change a little each step, the way training leaves them,
/// and large enough for the delta engine to engage at all — the 100-byte
/// tensors the older retention tests use are under `DELTA_MIN_SIZE`, so
/// they never build the delta chains that make pruning dangerous.
fn evolving_state(step: u64) -> Vec<TensorData> {
    let mut data = vec![7u8; 8192];
    data[(step as usize * 37) % 8192] = step as u8;
    data[0] = step as u8; // marker: which step these bytes are
    vec![TensorData {
        name: "w".into(),
        shape: vec![8192],
        dtype: "uint8".into(),
        data,
    }]
}

/// `keep_last: N` must keep the last N, not zero.
///
/// The older retention tests all use a short `full_snapshot_every_steps`,
/// so they build many small groups and pruning one whole group leaves the
/// others. The configuration that ships is the opposite — one full, then
/// deltas for thousands of steps, all in a single group — and there the
/// total cap used to remove that group and erase the entire history.
///
/// Found on a real run: 8 GPUs, `keep_last: 2`, three checkpoints handed
/// off, and an empty store at the end.
#[test]
fn the_total_cap_trims_the_history_instead_of_erasing_it() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let coord = Coordinator::new(
        Arc::clone(&storage),
        CoordinatorConfig {
            compression: CompressionAlgo::Zstd { level: 1 },
            retention: RetentionPolicy {
                max_full_snapshots: 5,
                max_deltas_per_full: 10,
                // One full at the start, everything after it a delta
                // against it: what a training run actually looks like.
                full_snapshot_every_steps: 5000,
                max_total_snapshots: Some(2),
            },
            ..Default::default()
        },
    )
    .unwrap();

    for step in 0..3u64 {
        coord.save(step, evolving_state(step), HashMap::new()).unwrap();
    }
    coord.flush().unwrap();

    let snaps = coord.list_snapshots();
    assert!(
        !snaps.is_empty(),
        "the total cap erased every checkpoint the run had written"
    );
    assert_eq!(snaps.len(), 2, "the cap is two, so two survive");

    // The newest must be there — it is the one a resume would take — and
    // the base it deltas against has to have survived with it.
    assert_eq!(snaps.last().unwrap().step, 2, "the newest step was pruned");
    for info in &snaps {
        let loaded = coord.load(info.id).unwrap_or_else(|e| {
            panic!("step {} survived the cap but cannot be read: {e}", info.step)
        });
        assert_eq!(loaded.get("w").unwrap()[0], info.step as u8);
    }
}

/// The cap must still bite once the history is all fulls, and still not
/// take the last one.
///
/// Starts at step 1: step 0 is a multiple of `rollback_interval_steps` and
/// is therefore rollback-protected, which is a different rule and has its
/// own test.
#[test]
fn the_total_cap_still_prunes_a_history_of_fulls() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let coord = Coordinator::new(
        Arc::clone(&storage),
        CoordinatorConfig {
            compression: CompressionAlgo::Zstd { level: 1 },
            retention: RetentionPolicy {
                max_full_snapshots: 10,
                max_deltas_per_full: 10,
                full_snapshot_every_steps: 1, // every save is a full
                max_total_snapshots: Some(2),
            },
            ..Default::default()
        },
    )
    .unwrap();

    for step in 1..6u64 {
        coord.save(step, evolving_state(step), HashMap::new()).unwrap();
    }
    coord.flush().unwrap();

    let steps: Vec<u64> = coord.list_snapshots().iter().map(|s| s.step).collect();
    assert_eq!(steps, vec![4, 5], "the cap should keep the newest two");
}

/// `keep_last: 1` must keep the *last* one.
///
/// Every snapshot after the first is a delta against it, so the store is a
/// full and a delta, and "drop the oldest delta" dropped the only one there
/// was - the newest. What survived was the full, the run's *first*
/// checkpoint, and a resume went back to it without a word. Seen from ravex
/// on 2026-09-21 as an audit entry with a null fingerprint (GPU-136).
#[test]
fn a_cap_of_one_keeps_the_newest_step() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let coord = Coordinator::new(
        Arc::clone(&storage),
        CoordinatorConfig {
            compression: CompressionAlgo::Zstd { level: 1 },
            retention: RetentionPolicy {
                max_full_snapshots: 5,
                max_deltas_per_full: 10,
                full_snapshot_every_steps: 5000,
                max_total_snapshots: Some(1),
            },
            ..Default::default()
        },
    )
    .unwrap();

    for step in 1..4u64 {
        coord.save(step, evolving_state(step), HashMap::new()).unwrap();
    }
    coord.flush().unwrap();

    let snaps = coord.list_snapshots();
    let steps: Vec<u64> = snaps.iter().map(|s| s.step).collect();
    assert_eq!(steps, vec![3], "a cap of one keeps the newest step");
    let loaded = coord.load(snaps[0].id).unwrap();
    assert_eq!(loaded.get("w").unwrap()[0], 3, "the survivor holds step 3's bytes");
}

/// A delta newer than a full it does not depend on is not the oldest thing
/// in the store, whatever its kind.
///
/// A second full arrives on its own schedule (`full_snapshot_every_steps`),
/// and from then on the store is `[old full, new full, delta]`. Preferring
/// deltas regardless of age removed the delta - the newest step - and did it
/// again on every save after, until the next scheduled full: a `keep_last: 2`
/// run past that point kept two checkpoints that stopped moving.
#[test]
fn a_cap_does_not_prefer_a_new_delta_over_an_old_full() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let coord = Coordinator::new(
        Arc::clone(&storage),
        CoordinatorConfig {
            compression: CompressionAlgo::Zstd { level: 1 },
            retention: RetentionPolicy {
                max_full_snapshots: 5,
                max_deltas_per_full: 10,
                full_snapshot_every_steps: 3, // fulls at 1 and 4
                max_total_snapshots: Some(2),
            },
            ..Default::default()
        },
    )
    .unwrap();

    for step in 1..7u64 {
        coord.save(step, evolving_state(step), HashMap::new()).unwrap();
    }
    coord.flush().unwrap();

    let snaps = coord.list_snapshots();
    assert_eq!(
        snaps.last().unwrap().step,
        6,
        "the newest step was pruned: {:?}",
        snaps.iter().map(|s| s.step).collect::<Vec<_>>()
    );
    assert_eq!(snaps.len(), 2, "the cap is two, so two survive");
    for info in &snaps {
        let loaded = coord.load(info.id).unwrap_or_else(|e| {
            panic!("step {} survived the cap but cannot be read: {e}", info.step)
        });
        assert_eq!(loaded.get("w").unwrap()[0], info.step as u8);
    }
}

#[test]
fn every_snapshot_left_after_retention_still_loads() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let config = CoordinatorConfig {
        world_size: 1,
        rank: 0,
        compression: CompressionAlgo::Zstd { level: 1 },
        retention: RetentionPolicy {
            max_full_snapshots: 2,
            max_deltas_per_full: 5,
            full_snapshot_every_steps: 3,
            max_total_snapshots: Some(6),
        },
        delta_max_ratio: 0.95,
        ..Default::default()
    };
    let coord = Coordinator::new(storage, config).unwrap();

    // Long enough that retention prunes several times over.
    for step in 0..14u64 {
        coord.save(step, evolving_state(step), HashMap::new()).unwrap();
    }
    coord.flush().unwrap();

    let snaps = coord.list_snapshots();
    assert!(!snaps.is_empty(), "retention removed everything");
    assert!(snaps.len() <= 6, "the total cap was not applied");

    for info in &snaps {
        let loaded = coord
            .load(info.id)
            .unwrap_or_else(|e| panic!("step {} survived retention but cannot be read: {e}", info.step));
        let w = loaded.get("w").expect("tensor missing from a listed snapshot");
        assert_eq!(
            w[0], info.step as u8,
            "step {} came back as the state of another step",
            info.step
        );
        assert_eq!(w.len(), 8192);
    }
}

/// A backend that panics rather than returning an error, standing in for
/// any bug that unwinds inside the save pipeline.
struct PanickingStorage;

impl StorageBackend for PanickingStorage {
    fn put(&self, _rel_path: &str, _data: &[u8]) -> Result<()> {
        panic!("disk on fire")
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

/// A panic on the background saver must surface as an error, never as a
/// hang.
///
/// Regression: the worker died with `busy` still set and no one left to
/// clear it, so the next `submit` waited on the condvar forever. The
/// training loop stopped without a message, which on rented hardware is an
/// idle GPU billing until a human notices. This is exactly how the bug was
/// found — the test that provoked it ran for ten minutes before being
/// killed.
///
/// Run on a side thread with a deadline, so a regression fails this test
/// instead of hanging the whole suite the way it did the first time.
#[test]
fn a_panic_in_the_save_thread_is_reported_not_hung() {
    let (done_tx, done_rx) = mpsc::channel();

    thread::spawn(move || {
        let storage: Arc<dyn StorageBackend> = Arc::new(PanickingStorage);
        let coord = Coordinator::new(
            storage,
            CoordinatorConfig {
                compression: CompressionAlgo::Zstd { level: 1 },
                ..Default::default()
            },
        )
        .unwrap();

        let _ = coord.save(0, evolving_state(0), HashMap::new());
        // Either this save or the flush has to report the failure; what
        // matters is that one of them returns at all.
        let second = coord.save(1, evolving_state(1), HashMap::new());
        let flushed = coord.flush();
        let _ = done_tx.send(second.is_err() || flushed.is_err());
    });

    match done_rx.recv_timeout(std::time::Duration::from_secs(30)) {
        Ok(reported) => assert!(
            reported,
            "the panic was swallowed: neither the next save nor flush reported it"
        ),
        Err(_) => panic!(
            "the save path hung after a panic on the background thread \
             instead of reporting it"
        ),
    }
}

/// `rollback_interval_steps` reaches this from the Python constructor, and
/// zero is the value someone will use to mean "no rollback snapshots".
/// Retention runs inside every save, so a panic here is a lost run.
#[test]
fn a_rollback_interval_of_zero_does_not_bring_down_the_save() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let config = CoordinatorConfig {
        world_size: 1,
        rank: 0,
        compression: CompressionAlgo::Zstd { level: 1 },
        lineage: LineageConfig {
            rollback_interval_steps: 0,
            max_rollback_snapshots: 3,
        },
        ..Default::default()
    };
    let coord = Coordinator::new(storage, config).unwrap();

    for step in 0..4u64 {
        coord.save(step, evolving_state(step), HashMap::new()).unwrap();
    }
    coord.flush().unwrap();
    assert!(!coord.list_snapshots().is_empty());
}

/// A pinned step outlives a cap that would otherwise take it (GPU-154).
///
/// `keep_last: 1` is the sharpest version of the question: everything except
/// the newest step is a candidate, and the pin is the only reason one of them
/// is still there.
#[test]
fn a_pinned_snapshot_survives_the_tightest_cap() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let config = CoordinatorConfig {
        world_size: 1,
        rank: 0,
        compression: CompressionAlgo::Zstd { level: 1 },
        retention: RetentionPolicy {
            max_full_snapshots: 2,
            max_deltas_per_full: 2,
            full_snapshot_every_steps: 0,
            max_total_snapshots: Some(1),
        },
        lineage: LineageConfig {
            rollback_interval_steps: 0,
            max_rollback_snapshots: 0,
        },
        delta_max_ratio: 0.95,
        ..Default::default()
    };
    let coord = Coordinator::new(storage, config).unwrap();

    coord.save(1, evolving_state(1), HashMap::new()).unwrap();
    coord.flush().unwrap();
    assert!(coord.pin(1).unwrap(), "step 1 is in the store");
    assert_eq!(coord.pinned_steps(), vec![1]);

    for step in 2..6u64 {
        coord.save(step, evolving_state(step), HashMap::new()).unwrap();
    }
    coord.flush().unwrap();

    let steps: Vec<u64> = coord.list_snapshots().iter().map(|s| s.step).collect();
    assert!(steps.contains(&1), "the pinned step went: {steps:?}");
    assert!(steps.contains(&5), "the newest step went: {steps:?}");

    // Listed is not enough: a pinned delta whose base was pruned is bytes
    // nothing can read, which is the failure this is guarding against.
    let pinned = coord
        .list_snapshots()
        .into_iter()
        .find(|s| s.step == 1)
        .unwrap();
    let loaded = coord.load(pinned.id).unwrap();
    assert_eq!(loaded.get("w").unwrap()[0], 1u8);
}

/// Unpinning gives it back to retention, and the next save collects it.
#[test]
fn unpinning_lets_retention_have_it_again() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let config = CoordinatorConfig {
        world_size: 1,
        rank: 0,
        compression: CompressionAlgo::Zstd { level: 1 },
        retention: RetentionPolicy {
            max_full_snapshots: 2,
            max_deltas_per_full: 2,
            full_snapshot_every_steps: 0,
            max_total_snapshots: Some(1),
        },
        lineage: LineageConfig {
            rollback_interval_steps: 0,
            max_rollback_snapshots: 0,
        },
        delta_max_ratio: 0.95,
        ..Default::default()
    };
    let coord = Coordinator::new(storage, config).unwrap();

    coord.save(1, evolving_state(1), HashMap::new()).unwrap();
    coord.flush().unwrap();
    coord.pin(1).unwrap();
    coord.save(2, evolving_state(2), HashMap::new()).unwrap();
    coord.flush().unwrap();
    assert!(coord.list_snapshots().iter().any(|s| s.step == 1));

    assert!(coord.unpin(1).unwrap());
    assert!(coord.pinned_steps().is_empty());
    coord.save(3, evolving_state(3), HashMap::new()).unwrap();
    coord.flush().unwrap();

    let steps: Vec<u64> = coord.list_snapshots().iter().map(|s| s.step).collect();
    assert!(!steps.contains(&1), "unpinned and still kept: {steps:?}");
}

/// Pinning a step the store does not hold is an answer, not an error: the
/// caller wants to say so, not to stop.
#[test]
fn pinning_a_step_that_is_not_there_says_so() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let coord = Coordinator::new(storage, CoordinatorConfig::default()).unwrap();
    assert!(!coord.pin(4242).unwrap());
    assert!(coord.pinned_steps().is_empty());
}

/// A pin is for a fork that outlives the process that made it, so it has to
/// be in the manifest and not in memory.
#[test]
fn a_pin_survives_reopening_the_store() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    // Built twice rather than cloned: `CoordinatorConfig` is not `Clone`.
    let config = || CoordinatorConfig {
        world_size: 1,
        rank: 0,
        compression: CompressionAlgo::Zstd { level: 1 },
        delta_max_ratio: 0.95,
        ..Default::default()
    };
    {
        let coord = Coordinator::new(Arc::clone(&storage), config()).unwrap();
        coord.save(7, evolving_state(7), HashMap::new()).unwrap();
        coord.flush().unwrap();
        coord.pin(7).unwrap();
    }
    let reopened = Coordinator::new(storage, config()).unwrap();
    assert_eq!(reopened.pinned_steps(), vec![7]);
}

/// Snapshots on the rollback interval are meant to outlive the ordinary
/// retention cap — that is the whole point of keeping them.
#[test]
fn rollback_snapshots_survive_the_retention_cap() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let config = CoordinatorConfig {
        world_size: 1,
        rank: 0,
        compression: CompressionAlgo::Zstd { level: 1 },
        retention: RetentionPolicy {
            max_full_snapshots: 1,
            max_deltas_per_full: 2,
            full_snapshot_every_steps: 2,
            max_total_snapshots: Some(3),
        },
        lineage: LineageConfig {
            rollback_interval_steps: 4,
            max_rollback_snapshots: 3,
        },
        delta_max_ratio: 0.95,
        ..Default::default()
    };
    let coord = Coordinator::new(storage, config).unwrap();

    for step in 0..12u64 {
        coord.save(step, evolving_state(step), HashMap::new()).unwrap();
    }
    coord.flush().unwrap();

    let snaps = coord.list_snapshots();
    let protected: Vec<u64> = snaps
        .iter()
        .filter(|s| !s.is_delta && s.step % 4 == 0)
        .map(|s| s.step)
        .collect();
    assert!(
        !protected.is_empty(),
        "every rollback-interval snapshot was pruned; steps left: {:?}",
        snaps.iter().map(|s| s.step).collect::<Vec<_>>()
    );

    // And they have to be readable, not merely listed.
    for info in snaps.iter().filter(|s| !s.is_delta && s.step % 4 == 0) {
        let loaded = coord.load(info.id).unwrap_or_else(|e| {
            panic!("rollback snapshot at step {} cannot be read: {e}", info.step)
        });
        assert_eq!(loaded.get("w").unwrap()[0], info.step as u8);
    }
}

/// The other side of the previous test, and the point of the cap: without
/// it, every snapshot that ever fell on the interval keeps its exemption
/// and a long run ends with a store retention is not allowed to touch.
/// Same shape as the test above, `max_rollback_snapshots` lowered to one.
#[test]
fn only_the_newest_rollback_snapshots_stay_protected() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let config = CoordinatorConfig {
        world_size: 1,
        rank: 0,
        compression: CompressionAlgo::Zstd { level: 1 },
        retention: RetentionPolicy {
            max_full_snapshots: 1,
            max_deltas_per_full: 2,
            full_snapshot_every_steps: 2,
            max_total_snapshots: Some(3),
        },
        lineage: LineageConfig {
            rollback_interval_steps: 4,
            max_rollback_snapshots: 1,
        },
        delta_max_ratio: 0.95,
        ..Default::default()
    };
    let coord = Coordinator::new(storage, config).unwrap();

    for step in 0..12u64 {
        coord.save(step, evolving_state(step), HashMap::new()).unwrap();
    }
    coord.flush().unwrap();

    let steps: Vec<u64> = coord.list_snapshots().iter().map(|s| s.step).collect();
    let on_interval: Vec<u64> = steps.iter().copied().filter(|s| s % 4 == 0).collect();

    assert!(
        on_interval.len() <= 1,
        "a cap of one left {} protected snapshots: {steps:?}",
        on_interval.len()
    );
    assert!(
        !steps.contains(&0),
        "step 0 outlived a cap of one; steps left: {steps:?}"
    );
}

/// Tied weights, and the order they arrive in changes between saves.
///
/// Deduplication keeps the first occurrence and records the second as an
/// `Alias` of it, so the pair `{emb, head}` sharing one buffer stores
/// whichever came first and points the other at it. Swap the order on the
/// next save — which costs nothing more than wrapping the model
/// differently — and the tensor that used to be the alias is now the one
/// carrying the bytes. It is unchanged since the base, so it is written
/// `Skipped`, and a skipped tensor resolves through the base by name:
/// straight onto the base's `Alias` entry, which holds no bytes.
#[test]
fn tied_weights_survive_a_change_in_tensor_order() {
    let dir = tempfile::tempdir().unwrap();
    let coord = make_coordinator(dir.path());

    let shared = vec![7u8; 8192];
    let tied = |first: &str, second: &str| {
        vec![
            TensorData {
                name: first.into(),
                shape: vec![8192],
                dtype: "uint8".into(),
                data: shared.clone(),
            },
            TensorData {
                name: second.into(),
                shape: vec![8192],
                dtype: "uint8".into(),
                data: shared.clone(),
            },
        ]
    };

    coord.save(100, tied("emb", "head"), HashMap::new()).unwrap();
    let second = coord.save(200, tied("head", "emb"), HashMap::new()).unwrap();
    coord.flush().unwrap();

    let loaded = coord.load(second).unwrap_or_else(|e| {
        panic!("the tied pair became unreadable when the order changed: {e}")
    });
    assert_eq!(loaded["emb"], shared);
    assert_eq!(loaded["head"], shared);
}

/// A checkpoint where nothing changed is still a checkpoint.
///
/// With every tensor skipped there are no bytes to write, so the save path
/// writes no pack — and the pack is what carries the descriptor that
/// startup recovery rebuilds a lost snapshot from. Kill the process
/// between the pack and the manifest and this one cannot come back: it is
/// not merely unrecovered, `recover_or_reclaim_orphans` has nothing to
/// judge it by. Fine-tuning, where whole stretches of the model do not
/// move, is where this is the common case rather than the odd one.
#[test]
fn a_checkpoint_that_changed_nothing_can_still_be_recovered() {
    let dir = tempfile::tempdir().unwrap();
    let unchanged = sample_tensors(3);

    let coord = make_coordinator(dir.path());
    coord.save(100, unchanged.clone(), HashMap::new()).unwrap();
    let quiet = coord.save(200, unchanged.clone(), HashMap::new()).unwrap();
    coord.flush().unwrap();
    drop(coord);

    // The crash: the pack landed, the manifest naming it did not.
    let manifest_path = dir.path().join("manifest.json");
    let raw = std::fs::read(&manifest_path).unwrap();
    // Same trailing-zero trim the loader does in `Coordinator::new`.
    let end = raw.iter().rposition(|&b| b != 0).map(|i| i + 1).unwrap_or(0);
    let mut manifest: Manifest = serde_json::from_slice(&raw[..end]).unwrap();
    assert_eq!(manifest.snapshots.len(), 2, "both saves were recorded");
    manifest.snapshots.retain(|s| s.id != quiet);
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let reopened = make_coordinator(dir.path());
    let steps: Vec<u64> = reopened.list_snapshots().iter().map(|s| s.step).collect();
    assert!(
        steps.contains(&200),
        "the checkpoint that changed nothing was not recovered; steps: {steps:?}"
    );

    let loaded = reopened.load(quiet).unwrap();
    assert_eq!(loaded["w"], vec![3u8; 8192]);
}

/// The merger deletes the base a save is reading.
///
/// `save_sync_in_pool` takes its base from the manifest and then *drops
/// the lock* before `save_rank_tensors` reads that base off storage. The
/// merger holds the same lock only while it decides what to fold, so the
/// two windows overlap: the merge can publish and delete the base
/// snapshot's packs while a save is in the middle of diffing against
/// them. It shows up two ways — the loud one is the save failing with
/// `Checkpoint not found: snapshots/…/rank_0.pack`, the quiet one is a
/// delta reaching the manifest with a `base_snapshot_id` that no longer
/// names anything, which breaks `load_latest` now and is discarded as an
/// orphan at the next startup.
///
/// The interleaving is forced rather than raced for. A sleep here proves
/// nothing: on 8 KB tensors both sides finish before either is preempted,
/// and the test would pass on a version that still has the bug. So the
/// storage backend itself runs the merge at the exact instant the save
/// reaches for the base — the one ordering the lock discipline has to
/// survive, and the one a real run hits when the base is gigabytes.
#[test]
fn a_merge_does_not_delete_the_base_a_save_is_reading() {
    /// Runs `armed` once, on the first pack read that follows arming it.
    #[derive(Default)]
    struct Interleave {
        armed: Mutex<Option<Box<dyn Fn() + Send>>>,
        fired: std::sync::atomic::AtomicBool,
    }

    impl Interleave {
        fn maybe_fire(&self) {
            if self.fired.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            let hook = self.armed.lock().unwrap();
            if let Some(run) = hook.as_ref() {
                self.fired
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                run();
            }
        }
    }

    struct MergeAtBaseRead {
        inner: Arc<LocalStorage>,
        interleave: Arc<Interleave>,
    }

    impl StorageBackend for MergeAtBaseRead {
        fn put(&self, p: &str, data: &[u8]) -> Result<()> {
            self.inner.put(p, data)
        }
        fn put_parts(&self, p: &str, parts: &[&[u8]]) -> Result<()> {
            self.inner.put_parts(p, parts)
        }
        fn get(&self, p: &str) -> Result<Vec<u8>> {
            if p.ends_with(".pack") {
                self.interleave.maybe_fire();
            }
            self.inner.get(p)
        }
        fn get_range(&self, p: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
            if p.ends_with(".pack") {
                self.interleave.maybe_fire();
            }
            self.inner.get_range(p, offset, len)
        }
        fn exists(&self, p: &str) -> Result<bool> {
            self.inner.exists(p)
        }
        fn delete(&self, p: &str) -> Result<()> {
            self.inner.delete(p)
        }
        fn list(&self, prefix: &str) -> Result<Vec<String>> {
            self.inner.list(prefix)
        }
        fn remove_dir(&self, p: &str) -> Result<()> {
            self.inner.remove_dir(p)
        }
    }

    let dir = tempfile::tempdir().unwrap();
    let plain = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let interleave = Arc::new(Interleave::default());
    let storage: Arc<dyn StorageBackend> = Arc::new(MergeAtBaseRead {
        inner: Arc::clone(&plain),
        interleave: Arc::clone(&interleave),
    });

    let compression = CompressionAlgo::Zstd { level: 1 };
    let config = CoordinatorConfig {
        world_size: 1,
        rank: 0,
        compression: compression.clone(),
        retention: RetentionPolicy {
            max_full_snapshots: 5,
            max_deltas_per_full: 10,
            full_snapshot_every_steps: 10000,
            max_total_snapshots: None,
        },
        delta_max_ratio: 0.95,
        // Synchronous, and with the base off storage rather than out of
        // memory: the race is between a save reading the base and a merge
        // deleting it, and neither happens when the bytes never leave the
        // process.
        async_save: false,
        keep_base_in_memory: false,
        ..Default::default()
    };
    let coord = Coordinator::new(storage, config).unwrap();

    // A base and two deltas, with nothing interleaved yet.
    for step in 0..3u64 {
        coord.save(step, evolving_state(step), HashMap::new()).unwrap();
    }

    // From here, the next save's reach for the base runs the merge first.
    let merge_storage: Arc<dyn StorageBackend> = Arc::clone(&plain) as Arc<dyn StorageBackend>;
    let merge_manifest = Arc::clone(&coord.core.manifest);
    let merge_compression = compression.clone();
    // The coordinator's own registry, not a fresh one: what is being
    // tested is that the save's pin is visible to the merge.
    let merge_in_flight = Arc::clone(&coord.core.in_flight);
    *interleave.armed.lock().unwrap() = Some(Box::new(move || {
        let _ = crate::merger::do_full_merge(
            &merge_storage,
            &merge_manifest,
            &merge_compression,
            &merge_in_flight,
            &Arc::new(crate::remote_sync::PendingDeletes::default()),
        );
    }));

    coord
        .save(3, evolving_state(3), HashMap::new())
        .unwrap_or_else(|e| panic!("the save lost its base to a merge running beside it: {e}"));
    coord.flush().unwrap();

    assert!(
        interleave.fired.load(std::sync::atomic::Ordering::SeqCst),
        "the merge never ran: this probe proved nothing"
    );

    // The quiet variant: a delta whose base the merge already deleted.
    let manifest = coord.core.manifest.lock().unwrap();
    let ids: Vec<Uuid> = manifest.snapshots.iter().map(|s| s.id).collect();
    let dangling: Vec<(u64, Uuid)> = manifest
        .snapshots
        .iter()
        .filter_map(|s| s.base_snapshot_id.map(|b| (s.step, b)))
        .filter(|(_, base)| !ids.contains(base))
        .collect();
    assert!(
        dangling.is_empty(),
        "deltas left pointing at a base the merge removed: {dangling:?}"
    );
    drop(manifest);

    let (_, loaded) = coord
        .load_latest()
        .unwrap_or_else(|e| panic!("the newest checkpoint is unreadable after merging: {e}"));
    assert_eq!(loaded["w"][0], 3, "the newest checkpoint is not step 3");
}

/// Deterministic float32 bytes that actually differ between steps.
///
/// Constant runs compress to nothing and delta against anything, so a
/// test built on them proves nothing about the delta path.
fn fp32_noise(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .flat_map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            // Keep the exponent sane so the bf16 cast is a real cast and
            // not a parade of NaNs.
            let v = ((s >> 40) as f32 / 1.0e5) - 5.0;
            v.to_le_bytes()
        })
        .collect()
}

fn cast_coordinator(dir: &std::path::Path, policy: DTypePolicy) -> Coordinator {
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir).unwrap());
    let config = CoordinatorConfig {
        compression: CompressionAlgo::Zstd { level: 1 },
        async_save: false,
        keep_base_in_memory: true,
        save_dtype: policy,
        ..Default::default()
    };
    Coordinator::new(storage, config).unwrap()
}

fn two_component_state(seed: u64) -> Vec<TensorData> {
    vec![
        TensorData {
            name: "ravex/models/layer.weight".into(),
            shape: vec![4096],
            dtype: "float32".into(),
            data: fp32_noise(4096, seed),
        },
        TensorData {
            name: "ravex/optimizers/exp_avg".into(),
            shape: vec![4096],
            dtype: "float32".into(),
            data: fp32_noise(4096, seed + 977),
        },
    ]
}

/// The failure this feature could have introduced, and the reason
/// `retain_base` is decided per tensor.
///
/// With one component cast and the other not, the in-memory base is
/// partial: it holds the weights, whose stored bytes are the incoming
/// bytes, and not the moments, whose stored bytes are bf16. If the
/// retention were still all-or-nothing in either direction, the next
/// delta would be XOR-ed against the wrong base for one of the two — and
/// every tensor would still pass its own hash check on the way back,
/// because the hash is of what was written, not of what should have
/// been. So the assertion has to be on the reconstructed values.
#[test]
fn a_per_component_cast_keeps_the_next_delta_correct() {
    let dir = tempfile::tempdir().unwrap();
    let coord = cast_coordinator(
        dir.path(),
        DTypePolicy::from_rules([("ravex/optimizers/*", "bf16")]).unwrap(),
    );

    coord.save(1, two_component_state(1), HashMap::new()).unwrap();

    // A second step, close to the first: this is what produces a delta
    // rather than a full, which is the case under test.
    let mut next = two_component_state(1);
    next[0].data[..64].copy_from_slice(&fp32_noise(16, 31));
    next[1].data[..64].copy_from_slice(&fp32_noise(16, 32));
    let snap = coord.save(2, next.clone(), HashMap::new()).unwrap();

    let loaded = coord.load(snap).unwrap();

    // The weights were not cast, so they come back byte for byte.
    assert_eq!(
        loaded["ravex/models/layer.weight"], next[0].data,
        "uncast weights must survive the delta exactly"
    );

    // The moments were cast to bf16 and back, so they come back rounded,
    // not equal. What must hold is that they are the *right* numbers:
    // bf16 keeps 8 significant bits, so a relative error over 1% means
    // the delta was applied against the wrong base, not that it rounded.
    let got = &loaded["ravex/optimizers/exp_avg"];
    assert_eq!(got.len(), next[1].data.len());
    for (i, (g, w)) in got
        .chunks_exact(4)
        .zip(next[1].data.chunks_exact(4))
        .enumerate()
    {
        let g = f32::from_le_bytes(g.try_into().unwrap());
        let w = f32::from_le_bytes(w.try_into().unwrap());
        let tol = w.abs() * 0.01 + 1e-6;
        assert!(
            (g - w).abs() <= tol,
            "element {i}: got {g}, wanted {w} within {tol} — a bf16 round \
             trip cannot be this far off, so the base was wrong"
        );
    }
}

/// Under the old rule any cast turned the in-memory base off for the
/// whole snapshot. The uncast third of the state has no reason to lose it.
#[test]
fn an_uncast_tensor_is_still_retained_when_another_is_cast() {
    let dir = tempfile::tempdir().unwrap();
    let coord = cast_coordinator(
        dir.path(),
        DTypePolicy::from_rules([("ravex/optimizers/*", "bf16")]).unwrap(),
    );
    coord.save(1, two_component_state(5), HashMap::new()).unwrap();

    let retained = coord.core.retained_base.lock().unwrap();
    let retained = retained.as_ref().expect("a full save must retain a base");
    assert!(
        retained.tensors.contains_key("ravex/models/layer.weight"),
        "the uncast component must stay in memory"
    );
    assert!(
        !retained.tensors.contains_key("ravex/optimizers/exp_avg"),
        "the cast component must not: what is on disk is bf16, and these \
         bytes are fp32"
    );
}

/// A uniform cast still has to reconstruct, and this is the case that
/// used to be covered by switching retention off entirely.
#[test]
fn a_uniform_cast_still_round_trips_across_a_delta() {
    let dir = tempfile::tempdir().unwrap();
    let coord = cast_coordinator(dir.path(), DTypePolicy::parse("bf16").unwrap());

    coord.save(1, two_component_state(9), HashMap::new()).unwrap();
    let mut next = two_component_state(9);
    next[0].data[..64].copy_from_slice(&fp32_noise(16, 77));
    let snap = coord.save(2, next.clone(), HashMap::new()).unwrap();

    let loaded = coord.load(snap).unwrap();
    for t in &next {
        let got = &loaded[&t.name];
        assert_eq!(got.len(), t.data.len(), "{}", t.name);
        for (g, w) in got.chunks_exact(4).zip(t.data.chunks_exact(4)) {
            let g = f32::from_le_bytes(g.try_into().unwrap());
            let w = f32::from_le_bytes(w.try_into().unwrap());
            assert!((g - w).abs() <= w.abs() * 0.01 + 1e-6, "{}", t.name);
        }
    }
}

/// Integer and bool state travels through a float cast untouched — this
/// is the half of the dtype question that is about transport, not
/// conversion. `save_dtype` names a target for floats; a step counter and
/// a causal mask are not floats and have no business being reinterpreted
/// because one was set.
#[test]
fn non_float_state_is_untouched_by_any_save_dtype() {
    let dir = tempfile::tempdir().unwrap();
    let coord = cast_coordinator(dir.path(), DTypePolicy::parse("bf16").unwrap());

    let step: Vec<u8> = (0..8192u32).flat_map(|i| (i as i64).to_le_bytes()).collect();
    let mask: Vec<u8> = (0..8192).map(|i| (i % 2) as u8).collect();
    let tensors = vec![
        TensorData {
            name: "ravex/optimizers/step".into(),
            shape: vec![8192],
            dtype: "int64".into(),
            data: step.clone(),
        },
        TensorData {
            name: "ravex/models/causal_mask".into(),
            shape: vec![8192],
            dtype: "bool".into(),
            data: mask.clone(),
        },
    ];
    let snap = coord.save(1, tensors, HashMap::new()).unwrap();
    let loaded = coord.load(snap).unwrap();
    assert_eq!(loaded["ravex/optimizers/step"], step);
    assert_eq!(loaded["ravex/models/causal_mask"], mask);
}

/// bfloat16 bytes: the top half of a float32, which is what the cast
/// does. Values stay well inside float16 range on purpose — see the test
/// that uses them.
fn bf16_noise(n: usize, seed: u64) -> Vec<u8> {
    fp32_noise(n, seed)
        .chunks_exact(4)
        .flat_map(|c| [c[2], c[3]])
        .collect()
}

/// How every tensor of a snapshot was stored, by name.
fn storage_of(coord: &Coordinator, snap: Uuid) -> HashMap<String, TensorStorage> {
    let manifest = coord.core.manifest.lock().unwrap();
    manifest
        .find_snapshot(snap)
        .expect("snapshot must be in the manifest")
        .ranks[&0]
        .tensors
        .iter()
        .map(|t| (t.name.clone(), t.storage.clone()))
        .collect()
}

/// The corruption a per-tensor `retain_base` has to avoid, in the one
/// shape where nothing else would catch it: a cast that does not change
/// the width.
///
/// A tensor arriving as bfloat16 and stored as float16 is the same number
/// of bytes before and after, so the length check in
/// [`crate::delta::compute_delta`] — which quietly saves the fp32→bf16
/// and fp32→fp8 cases by falling back to a full save — does not fire.
/// Retaining the incoming bytes would XOR the next delta against bfloat16
/// while the base on disk is float16, and the two are different bytes for
/// the same value.
///
/// bf16 → fp16 → bf16 is bit-exact for these values: float16 has ten
/// mantissa bits to bfloat16's seven and the exponents are in range, so
/// the round trip is lossless and the assertion can be on equality rather
/// than on a tolerance. That matters — a tolerance is what lets a wrong
/// base slip through as rounding.
#[test]
fn an_equal_width_cast_does_not_poison_the_retained_base() {
    let dir = tempfile::tempdir().unwrap();
    let coord = cast_coordinator(
        dir.path(),
        DTypePolicy::from_rules([("ravex/optimizers/*", "fp16")]).unwrap(),
    );

    let state = |seed: u64| {
        vec![
            TensorData {
                name: "ravex/models/layer.weight".into(),
                shape: vec![8192],
                dtype: "float32".into(),
                data: fp32_noise(8192, seed),
            },
            TensorData {
                name: "ravex/optimizers/exp_avg".into(),
                shape: vec![8192],
                dtype: "bfloat16".into(),
                data: bf16_noise(8192, seed + 977),
            },
        ]
    };

    coord.save(1, state(3), HashMap::new()).unwrap();

    let mut next = state(3);
    next[0].data[..64].copy_from_slice(&fp32_noise(16, 41));
    next[1].data[..64].copy_from_slice(&bf16_noise(32, 42));
    let snap = coord.save(2, next.clone(), HashMap::new()).unwrap();

    // Without this the test is vacuous: a full save reconstructs
    // correctly no matter what the retained base held.
    let stored = storage_of(&coord, snap);
    assert_eq!(
        stored["ravex/optimizers/exp_avg"],
        TensorStorage::DeltaXor,
        "the cast component must be stored as a delta for this test to \
         be about deltas at all"
    );

    let loaded = coord.load(snap).unwrap();
    assert_eq!(loaded["ravex/models/layer.weight"], next[0].data);
    assert_eq!(
        loaded["ravex/optimizers/exp_avg"], next[1].data,
        "bf16 → fp16 → bf16 is lossless here, so anything but equality \
         means the delta was applied to the wrong base"
    );
}

/// The death of a background thread has to reach whoever is saving.
///
/// `catch_unwind` covers the panics this crate can foresee; nothing covers a
/// thread that goes for another reason, and that is the case worth reporting.
/// A dead syncer in particular is the worst failure a durability feature has:
/// the run believes its checkpoints are reaching the bucket, and finds out
/// they are not when the machine dies and a resume is attempted.
///
/// Built by hand rather than through `Coordinator::new` because that spawns a
/// live merger, and the state under test is one where the thread is gone.
#[test]
fn a_dead_background_thread_is_reported_by_flush() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let merger = crate::merger::DeltaMerger::with_no_worker();
    // The real `notify`, sending into a channel whose receiver is gone: this
    // is how a live coordinator learns of it, and it is the step that used to
    // be a `let _ =`.
    merger.notify();

    let core = Arc::new(Core {
        storage,
        manifest: Arc::new(Mutex::new(Manifest::default())),
        config: CoordinatorConfig::default(),
        merger: Some(merger),
        syncer: None,
        in_flight: Arc::new(crate::inflight::InFlight::default()),
        pending_deletes: Arc::new(PendingDeletes::default()),
        retained_base: Mutex::new(None),
        dtype_patterns_checked: AtomicBool::new(false),
        manifest_stale: AtomicBool::new(false),
    });
    let coord = Coordinator { saver: None, core };

    let message = match coord.flush() {
        Err(e) => e.to_string(),
        Ok(()) => panic!("flush reported nothing: the merger died in silence"),
    };
    assert!(
        message.contains("merger"),
        "the report does not say which thread is gone: {message}"
    );

    // Not an event but a state, and nothing restarts that thread — so it is
    // reported again, rather than consumed the way the saver's error is.
    assert!(
        coord.flush().is_err(),
        "the report was one-shot: a caller that missed it never hears again"
    );
}

/// A backend that writes everything but the manifest, and panics on that.
///
/// Not a contrivance. `apply_retention` persists through the caller's
/// `StorageBackend` **with the manifest guard held**, and `StorageBackend` is
/// a public trait anyone can implement — so a panic inside that critical
/// section is reachable from outside the crate without touching a line of it.
/// `PanickingStorage` above cannot show this: it dies on the pack write, which
/// happens before the lock is taken.
///
/// Armed after the first save, so the store holds one real manifest to be
/// re-read from.
struct ManifestPanickingStorage {
    inner: LocalStorage,
    armed: AtomicBool,
}

impl ManifestPanickingStorage {
    fn new(dir: &std::path::Path) -> Self {
        ManifestPanickingStorage {
            inner: LocalStorage::new(dir).unwrap(),
            armed: AtomicBool::new(false),
        }
    }
}

impl StorageBackend for ManifestPanickingStorage {
    fn put(&self, rel_path: &str, data: &[u8]) -> Result<()> {
        if rel_path == "manifest.json" && self.armed.load(Ordering::Relaxed) {
            panic!("manifest write went wrong");
        }
        self.inner.put(rel_path, data)
    }
    fn get(&self, rel_path: &str) -> Result<Vec<u8>> {
        self.inner.get(rel_path)
    }
    fn exists(&self, rel_path: &str) -> Result<bool> {
        self.inner.exists(rel_path)
    }
    fn delete(&self, rel_path: &str) -> Result<()> {
        self.inner.delete(rel_path)
    }
    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        self.inner.list(prefix)
    }
}

/// A panic under the manifest lock must not wall off the coordinator.
///
/// `catch_unwind` on the save thread restores the *thread*; it does nothing
/// about the poison flag the `Mutex` sets when a guard is dropped mid-unwind.
/// Left alone, that flag turns one failed checkpoint into a `PanicException`
/// on every later `save`, `load`, `flush` and `list_snapshots`, raised on the
/// caller's thread from somewhere unrelated to what actually broke — and the
/// manager stays that way for the rest of the process.
///
/// The second half matters as much: the failed save had already pushed its
/// snapshot into the in-memory manifest before the panic, so recovering the
/// lock without re-reading would carry that phantom entry into the next
/// persisted manifest.
#[test]
fn a_panic_under_the_manifest_lock_does_not_wall_off_the_coordinator() {
    let dir = tempfile::tempdir().unwrap();
    let storage = Arc::new(ManifestPanickingStorage::new(dir.path()));
    let coord = Coordinator::new(
        Arc::clone(&storage) as Arc<dyn StorageBackend>,
        CoordinatorConfig {
            compression: CompressionAlgo::Zstd { level: 1 },
            ..Default::default()
        },
    )
    .unwrap();

    coord.save(1, sample_tensors(1), HashMap::new()).unwrap();
    coord.flush().expect("the first save is on an unarmed backend");

    storage.armed.store(true, Ordering::Relaxed);
    let _ = coord.save(2, sample_tensors(2), HashMap::new());
    assert!(
        coord.flush().is_err(),
        "the panic under the lock was swallowed instead of being reported"
    );
    assert!(
        coord.core.manifest.is_poisoned(),
        "this test is vacuous unless the panic actually poisoned the lock"
    );

    // The backend works again, as it would after a transient failure. What
    // must not persist is the poisoning.
    storage.armed.store(false, Ordering::Relaxed);
    let third = coord
        .save(3, sample_tensors(3), HashMap::new())
        .expect("a save after the poisoning must not fail");
    coord
        .flush()
        .expect("the recovered save must complete, not raise the old panic");

    let steps: Vec<u64> = coord.list_snapshots().iter().map(|s| s.step).collect();
    assert_eq!(
        steps,
        vec![1, 3],
        "step 2 never reached storage, so re-reading the manifest is what \
         keeps it from being carried into the next one"
    );
    assert_eq!(coord.load(third).unwrap()["w"], vec![3u8; 8192]);
}
/// Counts what a load actually pulls off storage, and how much of it.
///
/// The point of `describe` and `load_tensors` is not what they return — the
/// manifest and the full load already had that — but what they decline to
/// read. A test asserting only the values would pass on an implementation that
/// loads the whole checkpoint and throws it away, which is the implementation
/// being replaced.
struct MeasuredStorage {
    inner: LocalStorage,
    whole_reads: AtomicUsize,
    ranged_reads: AtomicUsize,
    bytes_read: AtomicUsize,
}

impl MeasuredStorage {
    fn new(dir: &std::path::Path) -> Self {
        MeasuredStorage {
            inner: LocalStorage::new(dir).unwrap(),
            whole_reads: AtomicUsize::new(0),
            ranged_reads: AtomicUsize::new(0),
            bytes_read: AtomicUsize::new(0),
        }
    }

    fn reset(&self) {
        self.whole_reads.store(0, Ordering::Relaxed);
        self.ranged_reads.store(0, Ordering::Relaxed);
        self.bytes_read.store(0, Ordering::Relaxed);
    }
}

impl StorageBackend for MeasuredStorage {
    fn put(&self, rel_path: &str, data: &[u8]) -> Result<()> {
        self.inner.put(rel_path, data)
    }
    fn put_parts(&self, rel_path: &str, parts: &[&[u8]]) -> Result<()> {
        self.inner.put_parts(rel_path, parts)
    }
    fn get(&self, rel_path: &str) -> Result<Vec<u8>> {
        let data = self.inner.get(rel_path)?;
        if rel_path.ends_with(".pack") {
            self.whole_reads.fetch_add(1, Ordering::Relaxed);
            self.bytes_read.fetch_add(data.len(), Ordering::Relaxed);
        }
        Ok(data)
    }
    fn get_range(&self, rel_path: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        let data = self.inner.get_range(rel_path, offset, len)?;
        if rel_path.ends_with(".pack") {
            self.ranged_reads.fetch_add(1, Ordering::Relaxed);
            self.bytes_read.fetch_add(data.len(), Ordering::Relaxed);
        }
        Ok(data)
    }
    fn exists(&self, rel_path: &str) -> Result<bool> {
        self.inner.exists(rel_path)
    }
    fn delete(&self, rel_path: &str) -> Result<()> {
        self.inner.delete(rel_path)
    }
    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        self.inner.list(prefix)
    }
}

/// One tiny tensor beside one large incompressible one, so "read the whole
/// pack" and "read one tensor" are separated by orders of magnitude.
///
/// The large one has to be *noise*. A run of predictable bytes compresses to
/// nothing, the pack comes out one page long, and the test then measures
/// storage padding rather than either strategy.
fn mixed_state(seed: u8) -> Vec<TensorData> {
    let mut s = (seed as u64) | 1;
    let w: Vec<u8> = (0..400_000)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        })
        .collect();
    vec![
        TensorData {
            name: "meta".into(),
            shape: vec![64],
            dtype: "uint8".into(),
            data: vec![seed; 64],
        },
        TensorData {
            name: "w".into(),
            shape: vec![400_000],
            dtype: "uint8".into(),
            data: w,
        },
    ]
}

/// `describe` answers off the manifest and reads nothing at all.
#[test]
fn describe_reports_the_shapes_without_touching_storage() {
    let dir = tempfile::tempdir().unwrap();
    let storage = Arc::new(MeasuredStorage::new(dir.path()));
    let coord = Coordinator::new(
        Arc::clone(&storage) as Arc<dyn StorageBackend>,
        CoordinatorConfig {
            compression: CompressionAlgo::Zstd { level: 1 },
            ..Default::default()
        },
    )
    .unwrap();

    let mut meta = HashMap::new();
    meta.insert("placements".into(), "{\"w\": \"shard(0)\"}".into());
    let snap = coord.save(7, mixed_state(1), meta).unwrap();
    coord.flush().unwrap();
    storage.reset();

    let described = coord.describe(snap).unwrap();

    assert_eq!(
        (
            storage.whole_reads.load(Ordering::Relaxed),
            storage.ranged_reads.load(Ordering::Relaxed)
        ),
        (0, 0),
        "describe read pack bytes: the whole point is that it does not"
    );
    assert_eq!(described.step, 7);
    assert_eq!(described.id, snap);
    assert!(!described.is_delta);
    assert_eq!(described.rank, 0);
    assert_eq!(described.ranks, 1);
    assert_eq!(
        described.metadata.get("placements").map(String::as_str),
        Some("{\"w\": \"shard(0)\"}"),
        "the caller's own metadata is what carries anything Moonclip does not \
         model, placements among it"
    );

    let by_name: HashMap<&str, &TensorDescription> = described
        .tensors
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect();
    assert_eq!(by_name["w"].shape, vec![400_000]);
    assert_eq!(by_name["w"].dtype, "uint8");
    assert_eq!(by_name["w"].raw_size, 400_000);
    assert_eq!(by_name["meta"].shape, vec![64]);
}

/// `hash_raw` lets a caller identify a snapshot's content without opening
/// `manifest.json`, which on Windows can make the writer's next rename fail
/// (GPU-93, GPU-128). So it has to be the real hash of the bytes saved, and it
/// has to hash what a tensor holds rather than how it was stored: unchanged
/// through a delta that stored nothing for it, different once a value moves.
#[test]
fn describe_reports_each_tensors_raw_hash() {
    let dir = tempfile::tempdir().unwrap();
    let coord = make_coordinator(dir.path());

    let saved = mixed_state(1);
    let first = coord.save(1, saved.clone(), HashMap::new()).unwrap();
    let unchanged = coord.save(2, mixed_state(1), HashMap::new()).unwrap();
    let changed = coord.save(3, mixed_state(2), HashMap::new()).unwrap();
    coord.flush().unwrap();

    let hashes = |snap| -> HashMap<String, String> {
        coord
            .describe(snap)
            .unwrap()
            .tensors
            .into_iter()
            .map(|t| (t.name, t.hash_raw))
            .collect()
    };
    let (first, unchanged, changed) = (hashes(first), hashes(unchanged), hashes(changed));

    for tensor in &saved {
        assert_eq!(
            first[&tensor.name],
            crate::hash::hash_hex(&tensor.data),
            "{} does not report the hash of the bytes that were saved",
            tensor.name
        );
    }
    assert_eq!(
        unchanged, first,
        "a step that changed nothing reported different hashes"
    );
    assert_ne!(changed["w"], first["w"]);
    assert_ne!(changed["meta"], first["meta"]);
}

/// The shapes have to keep coming back once the snapshot is a delta, because
/// that is what a run of any length actually holds — and a delta's rank entry
/// lists every tensor, including the ones it stored nothing for.
#[test]
fn describe_reports_shapes_for_a_delta_and_its_skipped_tensors() {
    let dir = tempfile::tempdir().unwrap();
    let coord = make_coordinator(dir.path());

    coord.save(1, mixed_state(1), HashMap::new()).unwrap();
    let mut next = mixed_state(1);
    next[1].data[..32].copy_from_slice(&[9u8; 32]);
    let snap = coord.save(2, next, HashMap::new()).unwrap();
    coord.flush().unwrap();

    let described = coord.describe(snap).unwrap();
    assert!(described.is_delta, "step 2 was stored as a full snapshot");

    let by_name: HashMap<&str, &TensorDescription> = described
        .tensors
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect();
    assert_eq!(
        by_name["meta"].storage,
        TensorStorage::Skipped,
        "`meta` was unchanged and should have been skipped"
    );
    assert_eq!(
        by_name["meta"].shape,
        vec![64],
        "a skipped tensor stores no bytes and still knows its shape — which is \
         the case that makes this API usable at all"
    );
    assert_eq!(by_name["w"].shape, vec![400_000]);
    assert_eq!(by_name["w"].raw_size, 400_000);
}

/// `load_tensors` reads the named spans, not the pack.
#[test]
fn load_tensors_reads_only_what_was_asked_for() {
    let dir = tempfile::tempdir().unwrap();
    let storage = Arc::new(MeasuredStorage::new(dir.path()));
    let coord = Coordinator::new(
        Arc::clone(&storage) as Arc<dyn StorageBackend>,
        CoordinatorConfig {
            compression: CompressionAlgo::Zstd { level: 1 },
            ..Default::default()
        },
    )
    .unwrap();

    let snap = coord.save(1, mixed_state(3), HashMap::new()).unwrap();
    coord.flush().unwrap();

    storage.reset();
    let whole = coord.load(snap).unwrap();
    let full_bytes = storage.bytes_read.load(Ordering::Relaxed);
    assert_eq!(whole["meta"], mixed_state(3)[0].data);

    storage.reset();
    let got = coord
        .load_tensors(snap, &["meta".to_string()])
        .unwrap();

    assert_eq!(got.len(), 1, "something other than `meta` came back");
    assert_eq!(got["meta"], mixed_state(3)[0].data);
    assert_eq!(
        storage.whole_reads.load(Ordering::Relaxed),
        0,
        "the pack was read whole: this is the read the call exists to avoid"
    );
    let subset_bytes = storage.bytes_read.load(Ordering::Relaxed);
    assert!(
        subset_bytes * 100 < full_bytes,
        "reading one 64-byte tensor cost {subset_bytes} bytes against the full \
         load's {full_bytes}: the range read is not being used"
    );
}

/// The case that decides whether this is usable in a real run: a small tensor
/// that does not change is stored `Skipped` from step two onwards, so reading
/// it means going to the base — and a base read that pulls the whole base pack
/// gives back everything the ranged read just saved.
#[test]
fn load_tensors_does_not_pull_the_base_pack_to_resolve_a_skip() {
    let dir = tempfile::tempdir().unwrap();
    let storage = Arc::new(MeasuredStorage::new(dir.path()));
    let coord = Coordinator::new(
        Arc::clone(&storage) as Arc<dyn StorageBackend>,
        CoordinatorConfig {
            compression: CompressionAlgo::Zstd { level: 1 },
            ..Default::default()
        },
    )
    .unwrap();

    coord.save(1, mixed_state(5), HashMap::new()).unwrap();
    let mut next = mixed_state(5);
    next[1].data[..32].copy_from_slice(&[7u8; 32]);
    let snap = coord.save(2, next, HashMap::new()).unwrap();
    coord.flush().unwrap();

    let described = coord.describe(snap).unwrap();
    let meta = described.tensors.iter().find(|t| t.name == "meta").unwrap();
    assert_eq!(
        meta.storage,
        TensorStorage::Skipped,
        "this test is vacuous unless `meta` was skipped"
    );

    storage.reset();
    let got = coord.load_tensors(snap, &["meta".to_string()]).unwrap();

    assert_eq!(got["meta"], mixed_state(5)[0].data);
    assert_eq!(
        storage.whole_reads.load(Ordering::Relaxed),
        0,
        "resolving the skip read the base pack whole, which costs the entire \
         previous checkpoint to recover 64 bytes"
    );
    let read = storage.bytes_read.load(Ordering::Relaxed);
    assert!(
        read < 8192,
        "read {read} bytes to recover a 64-byte tensor from the base"
    );
}

/// A name that is not there is an error saying so.
#[test]
fn load_tensors_refuses_a_name_the_snapshot_does_not_have() {
    let dir = tempfile::tempdir().unwrap();
    let coord = make_coordinator(dir.path());
    let snap = coord.save(1, sample_tensors(2), HashMap::new()).unwrap();
    coord.flush().unwrap();

    let err = coord
        .load_tensors(snap, &["not_a_tensor".to_string()])
        .expect_err("a missing name came back as an empty answer");
    assert!(
        err.to_string().contains("not_a_tensor"),
        "the error does not name what was missing: {err}"
    );
}

/// Tied weights are stored once and aliased, and an alias holds no bytes of
/// its own. A full load resolves those in a second pass over everything it
/// read; asking for the alias alone has to follow the reference instead.
#[test]
fn load_tensors_follows_an_alias_to_the_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let coord = make_coordinator(dir.path());

    let shared = (0..20_000u32).map(|i| (i % 251) as u8).collect::<Vec<u8>>();
    let tied = |name: &str| TensorData {
        name: name.into(),
        shape: vec![20_000],
        dtype: "uint8".into(),
        data: shared.clone(),
    };
    let snap = coord
        .save(
            1,
            vec![tied("embed.weight"), tied("lm_head.weight")],
            HashMap::new(),
        )
        .unwrap();
    coord.flush().unwrap();

    let described = coord.describe(snap).unwrap();
    let alias = described
        .tensors
        .iter()
        .find(|t| t.storage == TensorStorage::Alias)
        .expect("neither tied tensor was deduplicated: this test proves nothing");
    assert!(alias.alias_of.is_some(), "an alias with no target");

    let got = coord.load_tensors(snap, std::slice::from_ref(&alias.name)).unwrap();
    assert_eq!(got[&alias.name], shared, "the alias came back as the wrong bytes");
}


/// A pin made by another process with the store open survives this one's
/// saves. Two coordinators on one storage stand for the two processes: the
/// parent training, and a fork pinning the step it starts from. The parent's
/// manifest never hears of the pin - before `pins.json` its next save wrote
/// the manifest back without it, and retention took the step (GPU-149).
#[test]
fn a_pin_from_another_process_survives_this_ones_saves() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn StorageBackend> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let config = || CoordinatorConfig {
        world_size: 1,
        rank: 0,
        compression: CompressionAlgo::Zstd { level: 1 },
        retention: RetentionPolicy {
            max_full_snapshots: 2,
            max_deltas_per_full: 2,
            full_snapshot_every_steps: 0,
            max_total_snapshots: Some(2),
        },
        lineage: LineageConfig {
            rollback_interval_steps: 0,
            max_rollback_snapshots: 0,
        },
        delta_max_ratio: 0.95,
        ..Default::default()
    };
    let parent = Coordinator::new(Arc::clone(&storage), config()).unwrap();
    for step in 1..=3u64 {
        parent.save(step, evolving_state(step), HashMap::new()).unwrap();
    }
    parent.flush().unwrap();

    // The fork, in its own process, pins step 3 of the parent's store.
    let fork = Coordinator::new(Arc::clone(&storage), config()).unwrap();
    assert!(fork.pin(3).unwrap());
    drop(fork);

    // The parent carries on, knowing nothing, under a cap of two.
    for step in 4..=8u64 {
        parent.save(step, evolving_state(step), HashMap::new()).unwrap();
    }
    parent.flush().unwrap();

    let steps: Vec<u64> = parent.list_snapshots().iter().map(|s| s.step).collect();
    assert!(steps.contains(&3), "the fork's pin was lost to the parent's saves: {steps:?}");
    let reopened = Coordinator::new(Arc::clone(&storage), config()).unwrap();
    assert_eq!(reopened.pinned_steps(), vec![3]);

    // And an unpin from elsewhere counts as much: the next save collects it.
    assert!(reopened.unpin(3).unwrap());
    drop(reopened);
    parent.save(9, evolving_state(9), HashMap::new()).unwrap();
    parent.flush().unwrap();
    let steps: Vec<u64> = parent.list_snapshots().iter().map(|s| s.step).collect();
    assert!(!steps.contains(&3), "unpinned elsewhere, still kept: {steps:?}");
}
