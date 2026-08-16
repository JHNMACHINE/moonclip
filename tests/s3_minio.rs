//! The S3 backend against a real S3 API.
//!
//! Everything in `src/s3.rs`'s own test module is a pure function: URI
//! encoding, key building, signature determinism. None of it sends a request,
//! so none of it can tell whether the SigV4 signature this code produces is
//! one a server will actually accept. That was verified once by hand, with
//! `tests/debug_s3_signing.py` against MinIO, and never again.
//!
//! These tests close that gap. They are the difference between "the signature
//! is stable" and "the checkpoint reached the bucket".
//!
//! Skipped unless `MOONCLIP_S3_ENDPOINT` is set, so `cargo test` stays green
//! on a machine without Docker. To run them:
//!
//! ```sh
//! bash tests/minio_up.sh          # starts MinIO, prints the env vars
//! MOONCLIP_S3_ENDPOINT=http://127.0.0.1:9000 \
//! MOONCLIP_S3_BUCKET=moonclip-test \
//! MOONCLIP_S3_ACCESS_KEY=minioadmin \
//! MOONCLIP_S3_SECRET_KEY=minioadmin \
//!   cargo test --test s3_minio
//! ```

use moonclip::s3::{S3Config, S3Storage};
use moonclip::storage::StorageBackend;

/// Build a backend pointed at the configured endpoint, or `None` when the
/// suite is not set up — the tests then pass trivially rather than failing on
/// a machine that never asked to run them.
fn backend(prefix: &str) -> Option<S3Storage> {
    let endpoint = std::env::var("MOONCLIP_S3_ENDPOINT").ok()?;
    let config = S3Config {
        bucket: std::env::var("MOONCLIP_S3_BUCKET").unwrap_or_else(|_| "moonclip-test".into()),
        prefix: prefix.into(),
        region: "us-east-1".into(),
        endpoint: Some(endpoint),
        access_key: std::env::var("MOONCLIP_S3_ACCESS_KEY").unwrap_or_else(|_| "minioadmin".into()),
        secret_key: std::env::var("MOONCLIP_S3_SECRET_KEY").unwrap_or_else(|_| "minioadmin".into()),
        path_style: true,
        timeout_secs: 30,
    }
    .with_auto_path_style();

    Some(S3Storage::new(config).expect("building the S3 backend"))
}

/// A distinct prefix per test: they share one bucket and `list` would
/// otherwise see each other's objects.
macro_rules! s3_test {
    ($name:ident, $store:ident, $body:block) => {
        #[test]
        fn $name() {
            let Some($store) = backend(concat!("it/", stringify!($name))) else {
                eprintln!("skipped: MOONCLIP_S3_ENDPOINT is not set");
                return;
            };
            $body
        }
    };
}

s3_test!(round_trips_an_object, store, {
    let payload = b"the signature has to be one a server accepts".to_vec();
    store.put("a/t1.bin", &payload).unwrap();
    assert_eq!(store.get("a/t1.bin").unwrap(), payload);
});

s3_test!(round_trips_binary_data, store, {
    // Compressed tensors are arbitrary bytes: nulls, high bytes, everything.
    // A signing or transport bug that only mangles non-ASCII would sail past
    // a test that stores text.
    let payload: Vec<u8> = (0..=255u8).cycle().take(100_000).collect();
    store.put("a/blob.pack", &payload).unwrap();
    let got = store.get("a/blob.pack").unwrap();
    assert_eq!(got.len(), payload.len());
    assert_eq!(got, payload);
});

s3_test!(reads_a_byte_range, store, {
    // The save path reads base tensors out of a pack by range. If the S3
    // backend ignored the Range header it would return the whole object and
    // every delta would be computed against the wrong bytes.
    let payload: Vec<u8> = (0..200u8).collect();
    store.put("a/ranged.pack", &payload).unwrap();

    assert_eq!(store.get_range("a/ranged.pack", 10, 5).unwrap(), &payload[10..15]);
    assert_eq!(store.get_range("a/ranged.pack", 0, 1).unwrap(), &payload[0..1]);

    // Past the end must truncate, not error: that is what LocalStorage does,
    // and the two backends have to agree.
    let tail = store.get_range("a/ranged.pack", 195, 100).unwrap();
    assert_eq!(tail, &payload[195..]);
});

s3_test!(reports_existence, store, {
    assert!(!store.exists("a/absent.bin").unwrap());
    store.put("a/present.bin", b"x").unwrap();
    assert!(store.exists("a/present.bin").unwrap());
});

s3_test!(overwrites_in_place, store, {
    // The manifest is rewritten on every save.
    store.put("manifest.json", b"{\"snapshots\":[]}").unwrap();
    store.put("manifest.json", b"{\"snapshots\":[1]}").unwrap();
    assert_eq!(store.get("manifest.json").unwrap(), b"{\"snapshots\":[1]}");
});

s3_test!(lists_by_prefix, store, {
    store.put("snapshots/a/t1.bin", b"1").unwrap();
    store.put("snapshots/b/t2.bin", b"2").unwrap();
    store.put("other/t3.bin", b"3").unwrap();

    let listed = store.list("snapshots").unwrap();
    assert!(listed.iter().any(|k| k.ends_with("a/t1.bin")), "got {listed:?}");
    assert!(listed.iter().any(|k| k.ends_with("b/t2.bin")), "got {listed:?}");
    assert!(
        !listed.iter().any(|k| k.contains("other/")),
        "the prefix filter leaked: {listed:?}"
    );

    // Keys must come back usable as-is: the syncer feeds them straight into
    // get() and put() on the other backend.
    for key in &listed {
        assert!(store.get(key).is_ok(), "listed key {key} could not be read");
    }
});

s3_test!(deletes, store, {
    store.put("a/gone.bin", b"x").unwrap();
    assert!(store.exists("a/gone.bin").unwrap());
    store.delete("a/gone.bin").unwrap();
    assert!(!store.exists("a/gone.bin").unwrap());
});

s3_test!(a_missing_object_is_not_found, store, {
    match store.get("a/never-written.bin") {
        Err(moonclip::error::MoonclipError::NotFound(_)) => {}
        other => panic!("expected NotFound, got {other:?}"),
    }
});

/// The product's actual claim, end to end: a checkpoint written to local disk
/// reaches the bucket, and is readable from there once the machine is gone.
///
/// Everything above tests one S3 verb. This tests the path the runtime takes —
/// `RemoteSyncer` walking local storage and pushing it — which is the only one
/// that decides whether a run survives its instance being reclaimed.
#[test]
fn a_local_checkpoint_reaches_the_bucket() {
    let Some(remote) = backend("it/sync_e2e") else {
        eprintln!("skipped: MOONCLIP_S3_ENDPOINT is not set");
        return;
    };

    use moonclip::remote_sync::{RemoteSyncConfig, RemoteSyncer};
    use moonclip::storage::LocalStorage;
    use std::sync::Arc;

    let dir = tempfile::tempdir().unwrap();
    let local: Arc<dyn StorageBackend> =
        Arc::new(LocalStorage::new_unaligned(dir.path()).unwrap());
    let remote: Arc<dyn StorageBackend> = Arc::new(remote);

    let pack: Vec<u8> = (0..=255u8).cycle().take(50_000).collect();
    local.put("snapshots/s1/rank_0.pack", &pack).unwrap();
    local.put("manifest.json", b"{\"snapshots\":[1]}").unwrap();

    let mut syncer = RemoteSyncer::new(
        Arc::clone(&local),
        Arc::clone(&remote),
        RemoteSyncConfig::default(),
    );
    syncer.sync_now().expect("sync to a live bucket must succeed");
    syncer.shutdown();

    assert_eq!(
        remote.get("snapshots/s1/rank_0.pack").unwrap(),
        pack,
        "the pack reached the bucket but came back different"
    );
    assert_eq!(
        remote.get("manifest.json").unwrap(),
        b"{\"snapshots\":[1]}",
        "a checkpoint whose manifest never arrived is unreachable"
    );
}

s3_test!(handles_keys_with_awkward_characters, store, {
    // Tensor names become object keys, and models contain dots and dashes.
    // These are the characters SigV4's canonical URI encoding gets wrong when
    // it is subtly off.
    for name in [
        "a/blocks.0.attn.q_proj.weight.bin",
        "a/layer-1_norm.bin",
        "a/tilde~name.bin",
    ] {
        store.put(name, name.as_bytes()).unwrap();
        assert_eq!(store.get(name).unwrap(), name.as_bytes(), "key {name}");
    }
});
