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
//!
//! MinIO is the convenient target, not the honest one: it accepts single
//! `PUT`s far larger than S3 does. Cloudflare R2 shares S3's 5 GiB
//! single-`PUT` ceiling and adds a rule of its own — every part but the last
//! must be the same size — so it is the cheaper way to be told the truth:
//!
//! ```sh
//! MOONCLIP_S3_ENDPOINT=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
//! MOONCLIP_S3_REGION=auto
//! MOONCLIP_S3_BUCKET=moonclip-s3-tests
//! MOONCLIP_S3_ACCESS_KEY=...  MOONCLIP_S3_SECRET_KEY=...
//! ```

use moonclip::error::MoonclipError;
use moonclip::s3::{S3Config, S3Storage};
use moonclip::storage::StorageBackend;

/// Build a backend pointed at the configured endpoint, or `None` when the
/// suite is not set up — the tests then pass trivially rather than failing on
/// a machine that never asked to run them.
fn backend(prefix: &str) -> Option<S3Storage> {
    let config = backend_config(prefix)?;
    Some(S3Storage::new(config).expect("building the S3 backend"))
}

fn backend_config(prefix: &str) -> Option<S3Config> {
    let endpoint = std::env::var("MOONCLIP_S3_ENDPOINT").ok()?;
    let config = S3Config {
        bucket: std::env::var("MOONCLIP_S3_BUCKET").unwrap_or_else(|_| "moonclip-test".into()),
        prefix: prefix.into(),
        // R2 wants `auto`; MinIO does not care. The region goes into the
        // SigV4 credential scope, so a service that checks it rejects the
        // signature outright rather than saying the region was wrong.
        region: std::env::var("MOONCLIP_S3_REGION").unwrap_or_else(|_| "us-east-1".into()),
        endpoint: Some(endpoint),
        access_key: std::env::var("MOONCLIP_S3_ACCESS_KEY").unwrap_or_else(|_| "minioadmin".into()),
        secret_key: std::env::var("MOONCLIP_S3_SECRET_KEY").unwrap_or_else(|_| "minioadmin".into()),
        path_style: true,
        timeout_secs: 30,
        // Overridable so one test can watch `put` choose the multipart path
        // without moving four gigabytes to do it.
        single_put_limit: std::env::var("MOONCLIP_S3_SINGLE_PUT_LIMIT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(moonclip::s3::SINGLE_PUT_LIMIT),
    }
    .with_auto_path_style();

    Some(config)
}

/// The same backend with the multipart threshold moved.
///
/// Lowering it is how the dispatch below gets tested at all: at the real
/// threshold the only way to see `put` choose multipart is to move four
/// gigabytes, which on an ordinary uplink is over an hour.
fn backend_with_limit(prefix: &str, limit: usize) -> Option<S3Storage> {
    let config = backend_config(prefix)?.with_single_put_limit(limit);
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
        Default::default(),
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


// ─── multipart ──────────────────────────────────────────────────────

/// Enough to make several parts at the 16 MiB floor, and not so much that the
/// suite needs a gigabyte of memory to run.
const MULTIPART_BYTES: usize = 40 * 1024 * 1024;

fn noisy(len: usize) -> Vec<u8> {
    // Not zeros: a run of zeros would survive a truncated part, an off-by-one
    // in the chunking, and a part uploaded twice, all without the comparison
    // noticing.
    let mut seed = 0x9e3779b97f4a7c15u64;
    (0..len)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 24) as u8
        })
        .collect()
}

s3_test!(a_large_object_goes_up_in_parts_and_comes_back_whole, store, {
    // The reason this exists: every write used to be one `PUT`, and S3 refuses
    // a single `PUT` over 5 GiB. A gathered checkpoint of a 1B model with Adam
    // is around 11 GiB, and `gather` is Ravex's default — so the ceiling was
    // in the ordinary path, on the largest model, and MinIO could not show it
    // because it accepts far larger single `PUT`s than S3 does.
    let data = noisy(MULTIPART_BYTES);
    store
        .put_multipart("big.pack", &data)
        .expect("multipart upload");

    let back = store.get("big.pack").expect("reading it back");
    assert_eq!(back.len(), data.len(), "the object came back a different size");
    assert!(back == data, "the object came back with different bytes");
});

s3_test!(a_multipart_object_is_listed_and_deletable_like_any_other, store, {
    // A completed multipart upload has to be an ordinary object afterwards.
    // Parts left behind by an upload that never completed are invisible to
    // `list`, which is exactly why retention would never reclaim them.
    let data = noisy(MULTIPART_BYTES);
    store
        .put_multipart("listed.pack", &data)
        .expect("multipart upload");

    let keys = store.list("").expect("listing");
    assert!(
        keys.iter().any(|k| k.ends_with("listed.pack")),
        "a completed upload is missing from the listing: {keys:?}"
    );

    store.delete("listed.pack").expect("deleting");
    assert!(!store.exists("listed.pack").expect("checking"));
});

/// Two parts at the 16 MiB floor, and a quarter of the time of the 40 MiB one.
const DISPATCH_BYTES: usize = 20 * 1024 * 1024;

/// `put` has to *choose* the multipart path, and until now nothing watched it
/// choose: the multipart tests above all call `put_multipart` directly, so the
/// three lines that decide were the one part of the path no test touched.
///
/// The service is what tells them apart. A multipart upload's ETag ends in
/// `-<part count>`; a single `PUT`'s does not. Reading the suffix rather than
/// the value is deliberate — S3 and R2 both use that shape and compute the
/// hash differently.
#[test]
fn put_takes_the_multipart_path_once_the_object_is_over_the_threshold() {
    let Some(store) = backend_with_limit("it/dispatch_over", 1024 * 1024) else {
        eprintln!("skipped: MOONCLIP_S3_ENDPOINT is not set");
        return;
    };

    let data = noisy(DISPATCH_BYTES);
    store.put("over.pack", &data).expect("put");

    let etag = store.etag("over.pack").expect("head").expect("an ETag");
    assert!(
        etag.contains('-'),
        "a plain PUT was used for an object over the threshold: ETag {etag}"
    );

    let back = store.get("over.pack").expect("reading it back");
    assert!(back == data, "the object came back with different bytes");
}

/// The other side of the same branch, and the one that would catch an
/// inverted comparison: under the threshold nothing should be split.
#[test]
fn put_stays_a_single_request_under_the_threshold() {
    let Some(store) = backend_with_limit("it/dispatch_under", 64 * 1024 * 1024) else {
        eprintln!("skipped: MOONCLIP_S3_ENDPOINT is not set");
        return;
    };

    let data = noisy(DISPATCH_BYTES);
    store.put("under.pack", &data).expect("put");

    let etag = store.etag("under.pack").expect("head").expect("an ETag");
    assert!(
        !etag.contains('-'),
        "an object under the threshold was split anyway: ETag {etag}"
    );
}

s3_test!(a_finished_upload_leaves_nothing_holding_storage, store, {
    // Parts belonging to an unfinished multipart upload are invisible to
    // `ListObjectsV2`: `list` cannot see them, retention never reclaims them,
    // and the bill counts them for as long as they sit there. So "the upload
    // succeeded and the object is correct" is not the whole claim — the claim
    // is also that nothing was left holding storage behind it. Until
    // `list_multipart_uploads` existed there was no way to ask.
    let data = noisy(DISPATCH_BYTES);
    store
        .put_multipart("tidy.pack", &data)
        .expect("multipart upload");

    let dangling = store
        .list_multipart_uploads("")
        .expect("listing multipart uploads");

    assert!(
        dangling.is_empty(),
        "the upload completed and still left parts behind: {dangling:?}"
    );

    store.delete("tidy.pack").expect("deleting");
});

s3_test!(abandoning_an_upload_that_does_not_exist_is_answered, store, {
    // What this pins is that the request is well-formed — signed, with the
    // `uploadId` where the service expects it — and comes back rather than
    // hanging or failing at the transport.
    //
    // It deliberately does **not** pin which answer. The services disagree,
    // and finding that out cost a red CI job: R2 and S3 report `NoSuchUpload`,
    // MinIO reports success. An earlier version of this test asserted the
    // error, passed against R2, and failed against the MinIO the branch guard
    // runs on.
    //
    // The divergence has a consequence worth writing down: **an error from
    // this call does not mean "already gone", and success does not mean "there
    // was something there"**. Cleanup code that reads either as proof of the
    // upload's state would be wrong on one of the two. `put_multipart` is
    // safe on that count — it only reports a failed abort — and
    // `cleanup_the_bucket` acts on what `list_multipart_uploads` returns
    // rather than on what abort says.
    let outcome = store.abort_multipart_upload("nothing.pack", "an-upload-id-that-never-was");

    match outcome {
        Ok(()) => {}
        Err(MoonclipError::NotFound(_)) => {}
        Err(MoonclipError::Storage(message)) => {
            assert!(
                message.contains("404") || message.to_lowercase().contains("nosuchupload"),
                "the service refused for some reason other than the upload being absent: {message}"
            );
        }
        Err(other) => panic!("the abort request itself did not get through: {other}"),
    }
});

s3_test!(a_range_read_still_works_on_a_multipart_object, store, {
    // Pack reads are ranged, and a multipart object is the case where the
    // server assembled it from pieces. A range that straddles a part boundary
    // is the one worth asking about.
    let data = noisy(MULTIPART_BYTES);
    store
        .put_multipart("ranged.pack", &data)
        .expect("multipart upload");

    let boundary = 16 * 1024 * 1024;
    let slice = store
        .get_range("ranged.pack", (boundary - 8) as u64, 16)
        .expect("ranged read across a part boundary");

    assert_eq!(slice, &data[boundary - 8..boundary + 8]);
});


// ─── the way back ───────────────────────────────────────────────────

s3_test!(a_store_can_be_pulled_back_from_the_bucket, store, {
    use moonclip::storage::LocalStorage;
    use std::sync::Arc;

    // What a machine looks like after its disk is replaced: the bucket holds
    // the checkpoint, the local store holds nothing. Until 2026-08-19 nothing
    // could cross that gap — `sync_now` pushed and there was no pull — so a
    // replaced node started from scratch and took every other rank with it.
    let remote: Arc<dyn moonclip::storage::StorageBackend> = Arc::new(store);
    remote.put("manifest.json", b"{\"snapshots\":[]}").expect("seeding");
    remote
        .put("snapshots/abc/rank_0.pack", b"payload")
        .expect("seeding");

    let dir = tempfile::tempdir().expect("temp dir");
    let local: Arc<dyn moonclip::storage::StorageBackend> =
        Arc::new(LocalStorage::new(dir.path()).expect("local store"));

    let restored =
        moonclip::remote_sync::restore_from_remote(&local, &remote).expect("restoring");

    assert!(restored, "the manifest never arrived");

    // Compared on the leading bytes, not the whole buffer: `LocalStorage`
    // aligns what it writes to 4 KiB, so a short object read back is the
    // payload followed by padding. That is why `load_manifest` trims trailing
    // zeros, and it is a property of the local store rather than anything the
    // restore did. In a real store the remote already holds aligned files, so
    // the round trip does not change their length.
    let pack = local.get("snapshots/abc/rank_0.pack").expect("pack");
    assert_eq!(&pack[..7], b"payload");
    assert!(
        pack[7..].iter().all(|&b| b == 0),
        "the restored pack came back with something other than padding after it"
    );
});

s3_test!(an_empty_bucket_restores_nothing_and_says_so, store, {
    use moonclip::storage::LocalStorage;
    use std::sync::Arc;

    // A first run must not be mistaken for a machine that lost its disk.
    let remote: Arc<dyn moonclip::storage::StorageBackend> = Arc::new(store);
    let dir = tempfile::tempdir().expect("temp dir");
    let local: Arc<dyn moonclip::storage::StorageBackend> =
        Arc::new(LocalStorage::new(dir.path()).expect("local store"));

    let restored =
        moonclip::remote_sync::restore_from_remote(&local, &remote).expect("restoring");

    assert!(!restored);
});


// ─── housekeeping ───────────────────────────────────────────────────

/// Empty the test bucket, objects and abandoned uploads alike.
///
///     cargo test --test s3_minio -- --ignored cleanup
///
/// Ignored by default because it deletes: running it as part of the suite
/// would race the tests it shares a bucket with. It exists because the two
/// kinds of leftover need different calls — objects come from `list`, and
/// unfinished uploads are invisible to it — so "the bucket is empty" is not
/// something a single listing can tell you.
#[test]
#[ignore]
fn cleanup_the_bucket() {
    let Some(store) = backend("") else {
        eprintln!("skipped: MOONCLIP_S3_ENDPOINT is not set");
        return;
    };

    let uploads = store
        .list_multipart_uploads("")
        .expect("listing multipart uploads");
    for (key, id) in &uploads {
        match store.abort_multipart_upload(key, id) {
            Ok(()) => eprintln!("abandoned upload {id} on {key}"),
            Err(e) => eprintln!("could not abandon {id} on {key}: {e}"),
        }
    }

    let keys = store.list("").expect("listing");
    for key in &keys {
        match store.delete(key) {
            Ok(()) => eprintln!("deleted {key}"),
            Err(e) => eprintln!("could not delete {key}: {e}"),
        }
    }

    eprintln!(
        "removed {} object(s) and {} abandoned upload(s)",
        keys.len(),
        uploads.len()
    );
}
