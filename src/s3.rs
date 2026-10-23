//! S3-compatible storage, signed by `rusty-s3` (GPU-203).
//!
//! The signing is `rusty-s3`'s, not ours: until 0.2 this module carried a
//! SigV4 written by hand, which is exactly the kind of cryptography that is
//! easy to get subtly wrong at the edges. `rusty-s3` signs and nothing else -
//! it does no I/O - so it brings neither the AWS SDK's dependency tree nor
//! its defaults, whose checksum headers GCS and other S3-compatible services
//! reject. What stays ours is what is ours: the requests themselves (`ureq`),
//! retries, the multipart upload of large checkpoints and its abort, reads of
//! a byte range, and the little XML the answers need.
//!
//! Every request is a **presigned URL**: the signature is in its query
//! string, made the moment the request is sent, and the body is not hashed
//! into it (`UNSIGNED-PAYLOAD`). One exception: a listing with temporary
//! credentials is signed in the `Authorization` header, by a small SigV4 of
//! our own, because R2 refuses the session token in a listing's query and
//! `rusty-s3` signs only presigned URLs (`S3Storage::listed`). Integrity is covered a layer up - every
//! tensor carries its own hash in the manifest - as it was before.
//!
//! # What is not covered
//!
//! Written down because ten integration tests against an S3 gateway make
//! this module look more complete than it is. Gateways are forgiving in the
//! places real S3 is not, and the next person here should not start from the
//! assumption that everything is handled.
//!
//! * **Temporary credentials** *are* covered since 0.2: a session token
//!   ([`S3Config::session_token`]) goes into every signature, which is what
//!   short-lived credentials scoped to a prefix need. Refreshing them is the
//!   caller's: one that expires mid-run fails like a wrong key. Checked on
//!   R2 itself, where a listing needs the token in a header (0.2.1).
//! * **Very large objects.** Handled, but by a second path worth knowing
//!   about: past `SINGLE_PUT_LIMIT` a write becomes a multipart upload rather
//!   than one `PUT`, because S3 refuses a single `PUT` over 5 GiB and a
//!   gathered checkpoint of a large model reaches that on its own. The
//!   multipart path is the one the integration tests exercise least — the
//!   gateways accept far larger single `PUT`s than S3 does, so the threshold
//!   that matters in production is one the suite never crosses on its own.
//! * **Clock skew.** The signature is stamped with the local clock and a
//!   `RequestTimeTooSkewed` is treated as any other error. A host more than
//!   fifteen minutes off signs requests that will never be accepted, and the
//!   error does not say so in those words.
//! * **Checksum headers.** None are sent (`Content-MD5`, `x-amz-checksum-*`):
//!   a deliberate omission, and why the module works unmodified against GCS
//!   and R2.
//!
//! Path-style versus virtual-hosted addressing, percent-encoding of key names,
//! and `ListObjectsV2` pagination *are* handled; see [`S3Config`] and the
//! `bucket_of` and `list` implementations.

use rusty_s3::{Bucket, Credentials, Method, UrlStyle};
use std::fmt::Write as FmtWrite;
use url::Url;
use hmac::{Hmac, KeyInit, Mac};
use sha2::{Digest, Sha256};

use crate::error::{Result, MoonclipError};
use crate::storage::StorageBackend;

// ─── S3 Configuration ───────────────────────────────────────────────

/// Configuration for an S3-compatible storage backend.
#[derive(Debug, Clone)]
pub struct S3Config {
    /// Bucket name.
    pub bucket: String,
    /// Key prefix (like a "folder"), e.g. "checkpoints/harold-v0.9/".
    /// Trailing slash is added automatically if missing.
    pub prefix: String,
    /// AWS region, e.g. "us-east-1", "eu-west-1".
    pub region: String,
    /// S3 endpoint URL. Defaults to AWS S3.
    /// For MinIO: "http://localhost:9000"
    /// For R2: "https://<account_id>.r2.cloudflarestorage.com"
    /// For B2: "https://s3.<region>.backblazeb2.com"
    pub endpoint: Option<String>,
    /// AWS access key ID.
    pub access_key: String,
    /// AWS secret access key.
    pub secret_key: String,
    /// The session token of temporary credentials, which come as three
    /// values rather than two: S3's `AssumeRole`, or R2's temporary
    /// credentials scoped to a prefix. None for a long-lived key.
    pub session_token: Option<String>,
    /// Use path-style URLs (required for MinIO, optional for AWS).
    /// If true:  http://endpoint/bucket/key
    /// If false: http://bucket.endpoint/key (virtual-hosted)
    pub path_style: bool,
    /// Connection timeout in seconds.
    pub timeout_secs: u64,
    /// Above this many bytes, a write becomes a multipart upload instead of
    /// one `PUT`. Defaults to [`SINGLE_PUT_LIMIT`].
    ///
    /// Configurable for one reason: at the default, the only way to exercise
    /// the choice is to actually move four gigabytes, which on an ordinary
    /// connection is over an hour. Lowering it lets a test watch `put` take
    /// the multipart path for the price of a few megabytes. That the ceiling
    /// itself is real is the service's business, and documented; that we
    /// dispatch on it is ours.
    pub single_put_limit: usize,
}

impl S3Config {
    /// Auto-enable path_style when a custom endpoint is provided.
    /// Virtual-hosted style requires DNS resolution (only works with AWS/R2/B2 domains).
    pub fn with_auto_path_style(mut self) -> Self {
        if self.endpoint.is_some() {
            self.path_style = true;
        }
        self
    }

    /// Move the multipart threshold. See [`Self::single_put_limit`].
    pub fn with_single_put_limit(mut self, bytes: usize) -> Self {
        self.single_put_limit = bytes;
        self
    }

    /// Normalize the prefix: ensure trailing slash, strip leading slash.
    fn normalized_prefix(&self) -> String {
        let mut p = self.prefix.trim_matches('/').to_string();
        if !p.is_empty() {
            p.push('/');
        }
        p
    }

    /// Build the full object key from a relative path.
    fn object_key(&self, rel_path: &str) -> String {
        format!("{}{}", self.normalized_prefix(), rel_path)
    }
}

// ─── Credentials ────────────────────────────────────────────────────

/// The keys a bucket is signed with, as given or as found in the environment.
///
/// Moonclip reads them itself (GPU-267): the caller that saves files is not
/// the one that should know where a bucket's keys come from, and before this
/// every runtime on top of Moonclip carried its own copy of the same lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S3Credentials {
    pub access_key: String,
    pub secret_key: String,
    pub session_token: Option<String>,
}

/// Where keys are looked for, in order: Moonclip's own names, then the ones
/// every S3 tool reads. Each row is one source, read whole.
const CREDENTIAL_SOURCES: [[&str; 3]; 2] = [
    ["MOONCLIP_S3_ACCESS_KEY", "MOONCLIP_S3_SECRET_KEY", "MOONCLIP_S3_SESSION_TOKEN"],
    ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"],
];

impl S3Credentials {
    /// The keys given, or else the environment's.
    ///
    /// Keys given are taken whole: a caller that names an access key and not
    /// its secret has made a mistake, and filling the secret in from the
    /// environment would sign with a pair that never belonged together.
    pub fn resolve(
        access_key: Option<&str>,
        secret_key: Option<&str>,
        session_token: Option<&str>,
    ) -> Result<Self> {
        Self::resolve_with(access_key, secret_key, session_token, |name| std::env::var(name).ok())
    }

    /// [`Self::resolve`] with the environment as a function, so a test can
    /// give one without touching the process's own.
    pub fn resolve_with(
        access_key: Option<&str>,
        secret_key: Option<&str>,
        session_token: Option<&str>,
        var: impl Fn(&str) -> Option<String>,
    ) -> Result<Self> {
        let given = |value: Option<&str>| value.filter(|v| !v.is_empty()).map(str::to_string);
        match (given(access_key), given(secret_key)) {
            (Some(access_key), Some(secret_key)) => {
                return Ok(S3Credentials { access_key, secret_key, session_token: given(session_token) });
            }
            (None, None) => {}
            _ => {
                return Err(MoonclipError::Config(
                    "an access key and a secret key go together: give both, or neither and let \
                     Moonclip read them from the environment"
                        .into(),
                ))
            }
        }
        let read = |name: &str| var(name).filter(|v| !v.is_empty());
        // A source is used only when it has both keys. One with half a pair is
        // skipped rather than completed from the next: two sources mixed sign
        // with a pair nobody issued, and S3 answers that with a signature
        // error that names neither variable.
        for [access, secret, token] in CREDENTIAL_SOURCES {
            if let (Some(access_key), Some(secret_key)) = (read(access), read(secret)) {
                let session_token = given(session_token).or_else(|| read(token));
                return Ok(S3Credentials { access_key, secret_key, session_token });
            }
        }
        Err(MoonclipError::Config(
            "a bucket needs keys and none were found: set MOONCLIP_S3_ACCESS_KEY and \
             MOONCLIP_S3_SECRET_KEY (or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)"
                .into(),
        ))
    }
}

// ─── Addressing and signing ─────────────────────────────────────────

/// How long a signed request stays valid. It is signed the moment it is
/// sent, so this only has to cover the request starting: an hour leaves room
/// for a slow connection to open, and none for a URL found in a log later.
const SIGNED_FOR_SECS: u64 = 3600;

/// The bucket as `rusty-s3` addresses it: AWS's endpoint for the region when
/// none is given, path-style or virtual-hosted as configured.
fn bucket_of(config: &S3Config) -> Result<Bucket> {
    let mut endpoint = config
        .endpoint
        .clone()
        .unwrap_or_else(|| format!("https://s3.{}.amazonaws.com", config.region));
    // The bucket's address is joined onto this one, and a join replaces the
    // last segment of a path that has no trailing slash: an endpoint with a
    // path of its own would lose it.
    if !endpoint.ends_with('/') {
        endpoint.push('/');
    }
    let url = Url::parse(&endpoint).map_err(|e| {
        MoonclipError::Storage(format!("S3 endpoint {endpoint:?} is not an address: {e}"))
    })?;
    let style = if config.path_style {
        UrlStyle::Path
    } else {
        UrlStyle::VirtualHost
    };
    Bucket::new(url, style, config.bucket.clone(), config.region.clone())
        .map_err(|e| MoonclipError::Storage(format!("S3 bucket {:?}: {e}", config.bucket)))
}

/// Above this, one `PUT` is not allowed and the upload has to be split.
///
/// S3 refuses a single `PUT` over 5 GiB. The threshold here is lower on
/// purpose: nothing is gained by riding the limit, and a gathered checkpoint
/// of a large model is well past it either way — 1B parameters with Adam is
/// around 11 GiB, and `gather` is Ravex's default, so this is the ordinary
/// path rather than an extreme one.
pub const SINGLE_PUT_LIMIT: usize = 4 * 1024 * 1024 * 1024;

/// Smallest part S3 accepts, except for the last one, is 5 MiB. Larger than
/// that here: parts cost a request each, and 10000 of them is the ceiling.
const MIN_PART_SIZE: usize = 16 * 1024 * 1024;

/// S3's limit on parts per upload.
const MAX_PARTS: usize = 10_000;

/// Part size for an object of `total` bytes.
///
/// Grows with the object so the part count stays under [`MAX_PARTS`]: at the
/// floor this tops out at 160 GiB, which a large enough gathered checkpoint
/// would exceed. Some headroom is left rather than dividing exactly, because
/// a part count that lands on the limit leaves nothing for a rounding error.
fn part_size_for(total: usize) -> usize {
    let spread = total.div_ceil(MAX_PARTS - 16);
    MIN_PART_SIZE.max(spread)
}

// ─── S3 Storage Backend ─────────────────────────────────────────────

pub struct S3Storage {
    config: S3Config,
    bucket: Bucket,
    credentials: Credentials,
    agent: ureq::Agent,
}

impl S3Storage {
    pub fn new(config: S3Config) -> Result<Self> {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(config.timeout_secs))
            .timeout_read(std::time::Duration::from_secs(config.timeout_secs * 10))
            .timeout_write(std::time::Duration::from_secs(config.timeout_secs * 10))
            // One connection kept open per transfer the remote sync runs at
            // once (GPU-235). ureq keeps one per host by default, so the other
            // transfers would open a connection, TLS included, per request.
            .max_idle_connections_per_host(crate::remote_sync::TRANSFERS)
            .build();
        let bucket = bucket_of(&config)?;
        let credentials = match &config.session_token {
            Some(token) => Credentials::new_with_token(&config.access_key, &config.secret_key, token),
            None => Credentials::new(&config.access_key, &config.secret_key),
        };

        Ok(S3Storage {
            config,
            bucket,
            credentials,
            agent,
        })
    }

    /// The address of one request to `key`, signed now: `rusty-s3`'s
    /// SigV4, in the query string, with the session token when there is one.
    ///
    /// `query` is any of S3's own parameters (`uploads`, `partNumber`...).
    /// `rusty-s3` merges them with its `X-Amz-*` ones as two runs already in
    /// order, so they are sorted here first: out of order, the URL would carry
    /// them in an order other than the one the signature covers.
    fn signed_url(
        &self,
        method: &str,
        key: &str,
        query: &[(&str, &str)],
    ) -> std::result::Result<Url, MoonclipError> {
        let method = match method {
            "GET" => Method::Get,
            "PUT" => Method::Put,
            "DELETE" => Method::Delete,
            "HEAD" => Method::Head,
            "POST" => Method::Post,
            _ => return Err(MoonclipError::Storage(format!("Unknown method: {method}"))),
        };
        let url = self
            .bucket
            .object_url(key)
            .map_err(|e| MoonclipError::Storage(format!("S3 key {key:?} is not a path: {e}")))?;
        let mut query = query.to_vec();
        query.sort_unstable();
        Ok(rusty_s3::signing::sign(
            &jiff::Timestamp::now(),
            method,
            url,
            self.credentials.key(),
            self.credentials.secret(),
            self.credentials.token(),
            self.bucket.region(),
            SIGNED_FOR_SECS,
            query.iter().copied(),
            std::iter::empty(),
        ))
    }

    /// One attempt, retried on the failures that are worth retrying.
    ///
    /// There were none before 0.0.6: a single 503, a reset connection or a DNS
    /// hiccup failed the sync, and object stores produce all three as a matter
    /// of course — S3's own guidance is that clients retry 5xx and 429. A
    /// checkpoint upload that gives up on the first blip is a checkpoint that
    /// did not leave the machine.
    ///
    /// What is *not* retried: 4xx other than 429. A wrong key, a missing
    /// bucket or a bad signature fails identically the second time, and
    /// retrying only delays the error.
    ///
    /// Every request here is idempotent — PUT of a fixed key with fixed bytes,
    /// GET, HEAD, DELETE — so a retry cannot compound.
    fn do_request(
        &self,
        method: &str,
        key: &str,
        body: Option<&[u8]>,
        query_params: &[(&str, &str)],
    ) -> std::result::Result<ureq::Response, MoonclipError> {
        let mut backoff = std::time::Duration::from_millis(200);
        let mut attempt = 0;
        loop {
            attempt += 1;
            match self.attempt_request(method, key, body, query_params) {
                Ok(resp) => return Ok(resp),
                Err(e) if attempt < MAX_ATTEMPTS && is_retryable(&e) => {
                    std::thread::sleep(backoff);
                    backoff *= 2;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// A listing of the bucket with temporary credentials, signed in the
    /// `Authorization` header rather than in the query (GPU-217).
    ///
    /// R2 refuses `X-Amz-Security-Token` as a query parameter of a listing -
    /// "ListObjectsV2 search parameter X-Amz-Security-Token not implemented",
    /// and the same of V1 - while it takes it there on every other request,
    /// and takes the token as a signed header of a presigned URL on none. So a
    /// store written with keys scoped to its bucket could be saved and never
    /// restored. `rusty-s3` 0.10 signs presigned URLs only, so this one kind
    /// of request is signed here: SigV4 in its header form, checked against
    /// AWS's own example below.
    fn listed(&self, query: &[(&str, &str)]) -> std::result::Result<ureq::Response, MoonclipError> {
        let mut url = self
            .bucket
            .object_url("")
            .map_err(|e| MoonclipError::Storage(format!("S3 bucket address: {e}")))?;
        let stamp = jiff::Timestamp::now().strftime("%Y%m%dT%H%M%SZ").to_string();
        let (query_string, headers) = header_signed(
            &url,
            query,
            &stamp,
            self.bucket.region(),
            self.credentials.key(),
            self.credentials.secret(),
            self.credentials.token(),
        );
        url.set_query(if query_string.is_empty() { None } else { Some(&query_string) });
        let mut request = self.agent.get(url.as_str());
        for (name, value) in &headers {
            request = request.set(name, value);
        }
        request.call().map_err(|e| status_error("", e))
    }

    fn attempt_request(
        &self,
        method: &str,
        key: &str,
        body: Option<&[u8]>,
        query_params: &[(&str, &str)],
    ) -> std::result::Result<ureq::Response, MoonclipError> {
        if method == "GET" && key.is_empty() && self.config.session_token.is_some() {
            return self.listed(query_params);
        }
        let url = self.signed_url(method, key, query_params)?;
        // Nothing to set beyond the URL: the signature covers only the host,
        // and ureq sends that and the body's length on its own.
        let req = match method {
            "GET" => self.agent.get(url.as_str()),
            "PUT" => self.agent.put(url.as_str()),
            "DELETE" => self.agent.delete(url.as_str()),
            "HEAD" => self.agent.head(url.as_str()),
            "POST" => self.agent.post(url.as_str()),
            _ => return Err(MoonclipError::Storage(format!("Unknown method: {method}"))),
        };

        let response = if let Some(data) = body {
            req.send_bytes(data)
        } else {
            req.call()
        };

        response.map_err(|e| status_error(key, e))
    }

    /// Upload an object in parts, because S3 will not take it in one.
    ///
    /// Public so the path can be exercised without a four-gigabyte fixture:
    /// [`Self::put`] only reaches it past [`SINGLE_PUT_LIMIT`], and a test that
    /// cannot run is not coverage. Callers should use `put` and let it choose.
    ///
    /// Three calls plus one per part: create, upload each, complete. The part
    /// that is easy to get wrong is the fourth — **abort**. An upload that is
    /// started and neither completed nor aborted leaves its parts in the
    /// bucket, where `ListObjectsV2` does not show them, retention never sees
    /// them, and the bill keeps counting. So every way out of here other than
    /// success goes through `AbortMultipartUpload`.
    pub fn put_multipart(&self, rel_path: &str, data: &[u8]) -> Result<()> {
        let key = &self.config.object_key(rel_path);
        let created = self.do_request("POST", key, None, &[("uploads", "")])?;
        let body = created
            .into_string()
            .map_err(|e| MoonclipError::Storage(format!("S3 multipart start: {e}")))?;
        let upload_id = extract_xml_value(&body, "UploadId").ok_or_else(|| {
            MoonclipError::Storage("S3 multipart start returned no UploadId".into())
        })?;

        match self.upload_parts(key, data, &upload_id) {
            Ok(()) => Ok(()),
            Err(e) => {
                // Reported at most as a second line: the upload's failure is
                // what the caller needs, and a failed cleanup must not hide it.
                if let Err(cleanup) = self.do_request(
                    "DELETE",
                    key,
                    None,
                    &[("uploadId", upload_id.as_str())],
                ) {
                    eprintln!(
                        "[Moonclip] could not abort the multipart upload of                          {key}: {cleanup}. Its parts are still in the bucket,                          invisible to a listing and still billed."
                    );
                }
                Err(e)
            }
        }
    }

    /// Multipart uploads that were started and never finished, as the service
    /// sees them: `(key, upload id)`.
    ///
    /// Nothing else can find them. The parts of an unfinished upload do not
    /// appear in `ListObjectsV2`, so `list` cannot see them, retention never
    /// reclaims them, and the bill counts them the whole time. That is the
    /// reason [`Self::put_multipart`] aborts on every way out other than
    /// success — and until this existed, nothing could check that it did.
    ///
    /// Paginated the same way `list` is. Truncation here is not the ordinary
    /// case: a bucket with more than a thousand abandoned uploads has a
    /// problem this function is only reporting.
    pub fn list_multipart_uploads(&self, prefix: &str) -> Result<Vec<(String, String)>> {
        let full_prefix = self.config.object_key(prefix);
        let mut found = Vec::new();
        let mut markers: Option<(String, String)> = None;

        loop {
            let mut params: Vec<(&str, &str)> = vec![("uploads", ""), ("prefix", &full_prefix)];
            let held;
            if let Some(ref pair) = markers {
                held = pair.clone();
                params.push(("key-marker", &held.0));
                params.push(("upload-id-marker", &held.1));
            }

            let response = self.do_request("GET", "", None, &params)?;
            let body = response.into_string().map_err(|e| {
                MoonclipError::Storage(format!("S3 multipart listing parse error: {e}"))
            })?;

            // Split on the element rather than pulling two flat lists of Key
            // and UploadId: a response that omitted one field for one entry
            // would otherwise pair every later id with the wrong key.
            for chunk in body.split("<Upload>").skip(1) {
                let entry = chunk.split("</Upload>").next().unwrap_or("");
                if let (Some(key), Some(id)) = (
                    extract_xml_value(entry, "Key"),
                    extract_xml_value(entry, "UploadId"),
                ) {
                    found.push((key, id));
                }
            }

            if !body.contains("<IsTruncated>true</IsTruncated>") {
                break;
            }
            match (
                extract_xml_value(&body, "NextKeyMarker"),
                extract_xml_value(&body, "NextUploadIdMarker"),
            ) {
                (Some(key), Some(id)) => markers = Some((key, id)),
                _ => break,
            }
        }

        Ok(found)
    }

    /// Abandon a multipart upload, releasing the parts it is holding.
    ///
    /// Public because the internal abort only covers uploads this process
    /// started. One that was interrupted by a machine going away is nobody's
    /// to clean up otherwise, and it is being paid for.
    pub fn abort_multipart_upload(&self, key: &str, upload_id: &str) -> Result<()> {
        self.do_request("DELETE", key, None, &[("uploadId", upload_id)])?;
        Ok(())
    }

    /// The ETag the service reports for an object, unquoted, or None.
    ///
    /// The one cheap way to tell *how* an object was written: a multipart
    /// upload's ETag ends in `-<part count>`, a single `PUT`'s does not. S3
    /// and R2 both follow that shape while computing the hash differently —
    /// which is why callers should read the suffix and never the value.
    pub fn etag(&self, rel_path: &str) -> Result<Option<String>> {
        let key = self.config.object_key(rel_path);
        let response = self.do_request("HEAD", &key, None, &[])?;
        Ok(response
            .header("ETag")
            .map(|value| value.trim_matches('"').to_string()))
    }

    /// The parts themselves, and the completion that makes them an object.
    ///
    /// Split out so [`Self::put_multipart`] has exactly one place to abort
    /// from: every error in here leaves the upload started.
    fn upload_parts(&self, key: &str, data: &[u8], upload_id: &str) -> Result<()> {
        let size = part_size_for(data.len());
        let mut completed = String::from("<CompleteMultipartUpload>");

        for (index, chunk) in data.chunks(size).enumerate() {
            let number = index + 1;
            let response = self.do_request(
                "PUT",
                key,
                Some(chunk),
                &[
                    ("partNumber", number.to_string().as_str()),
                    ("uploadId", upload_id),
                ],
            )?;

            // The ETag identifies the part in the completion, and S3 rejects
            // the whole upload if one is missing or wrong.
            let etag = response.header("ETag").ok_or_else(|| {
                MoonclipError::Storage(format!("S3 part {number} came back without an ETag"))
            })?;
            let _ = write!(
                completed,
                "<Part><PartNumber>{number}</PartNumber><ETag>{etag}</ETag></Part>"
            );
        }
        completed.push_str("</CompleteMultipartUpload>");

        let response = self.do_request(
            "POST",
            key,
            Some(completed.as_bytes()),
            &[("uploadId", upload_id)],
        )?;

        // S3 answers 200 and then reports the failure in the body, so the
        // status code alone is not the answer here — it is the one place in
        // this module where a success code can mean a failed request.
        let body = response
            .into_string()
            .map_err(|e| MoonclipError::Storage(format!("S3 multipart complete: {e}")))?;
        if body.contains("<Error>") {
            let code = extract_xml_value(&body, "Code").unwrap_or_else(|| "unknown".into());
            return Err(MoonclipError::Storage(format!(
                "S3 refused to complete the multipart upload of {key}: {code}"
            )));
        }
        Ok(())
    }

    /// GET with a byte range, which is what a pack read wants.
    ///
    /// The trait's default implementation downloads the whole object and
    /// slices it. On local storage that is a seek; on S3 it was the entire
    /// checkpoint pulled over the network to read one tensor's few hundred
    /// kilobytes — once per tensor. `Range` is not part of what is signed -
    /// a presigned URL covers the host alone - so it can be attached after
    /// signing.
    fn ranged_get(&self, key: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        use std::io::Read;

        if len == 0 {
            return Ok(Vec::new());
        }
        let range = format!("bytes={}-{}", offset, offset + len as u64 - 1);

        let mut backoff = std::time::Duration::from_millis(200);
        let mut attempt = 0;
        loop {
            attempt += 1;
            let url = self.signed_url("GET", key, &[])?;
            match self.agent.get(url.as_str()).set("Range", &range).call() {
                Ok(resp) => {
                    // A server is allowed to ignore `Range` and answer 200 with
                    // the whole object. Truncating that to `len` would hand
                    // back the bytes at offset 0 as if they were the bytes at
                    // `offset` — a pack descriptor read that quietly returns
                    // the header, or a tensor blob that is somebody else's.
                    // Wrong bytes are worse than no bytes, so this is an error
                    // unless the window happens to start at 0 anyway.
                    if resp.status() != 206 && offset != 0 {
                        return Err(MoonclipError::Storage(format!(
                            "S3 answered {} for a ranged read of {key}: the endpoint ignored \
                             Range, so bytes {}..{} cannot be read a piece at a time",
                            resp.status(),
                            offset,
                            offset + len as u64
                        )));
                    }
                    let mut buf = Vec::with_capacity(len);
                    // Bounded by `len` rather than by what the far side sends,
                    // so an over-long body cannot balloon the read.
                    let mut reader = resp.into_reader().take(len as u64);
                    reader
                        .read_to_end(&mut buf)
                        .map_err(|e| MoonclipError::Storage(format!("S3 read error: {e}")))?;
                    return Ok(buf);
                }
                Err(e) => {
                    let err = status_error(key, e);
                    if attempt < MAX_ATTEMPTS && is_retryable(&err) {
                        std::thread::sleep(backoff);
                        backoff *= 2;
                        continue;
                    }
                    return Err(err);
                }
            }
        }
    }
}

/// Three attempts total: enough to ride out a rolling restart on the far side,
/// short enough that a genuinely broken endpoint is reported while the caller
/// still has a machine to react on.
const MAX_ATTEMPTS: u32 = 3;

/// Turn a ureq failure into ours, keeping the status where callers can act on
/// it.
///
/// 404 becomes `NotFound` here rather than being recovered from the message
/// later. Callers used to look for the substring "404" in the error text,
/// which also matched any object whose *key* contained those digits — a step
/// number, a shard id — and reported a real failure as a missing object.
fn status_error(key: &str, e: ureq::Error) -> MoonclipError {
    match e {
        ureq::Error::Status(404, _) => MoonclipError::NotFound(key.to_string()),
        ureq::Error::Status(code, resp) => {
            let body = resp.into_string().unwrap_or_default();
            MoonclipError::Storage(format!("S3 HTTP {code}: {body}"))
        }
        ureq::Error::Transport(t) => MoonclipError::Storage(format!("S3 transport error: {t}")),
    }
}

/// Whether this failure is worth another attempt.
fn is_retryable(e: &MoonclipError) -> bool {
    match e {
        MoonclipError::Storage(msg) if msg.starts_with("S3 transport error") => true,
        MoonclipError::Storage(msg) => {
            // "S3 HTTP <code>: ..." — 5xx is the far side failing, 429 is it
            // asking to be slowed down.
            let code = msg
                .strip_prefix("S3 HTTP ")
                .and_then(|rest| rest.split(':').next())
                .and_then(|c| c.trim().parse::<u16>().ok());
            matches!(code, Some(429) | Some(500..=599))
        }
        _ => false,
    }
}

impl StorageBackend for S3Storage {
    fn put(&self, rel_path: &str, data: &[u8]) -> Result<()> {
        let key = self.config.object_key(rel_path);
        if data.len() > self.config.single_put_limit {
            return self.put_multipart(rel_path, data);
        }
        self.do_request("PUT", &key, Some(data), &[])?;
        Ok(())
    }

    fn get(&self, rel_path: &str) -> Result<Vec<u8>> {
        let key = self.config.object_key(rel_path);
        let resp = self.do_request("GET", &key, None, &[]).map_err(|e| match e {
            // Reported under the caller's path rather than the bucket key.
            MoonclipError::NotFound(_) => MoonclipError::NotFound(rel_path.to_string()),
            other => other,
        })?;

        let mut buf = Vec::new();
        resp.into_reader()
            .read_to_end(&mut buf)
            .map_err(|e| MoonclipError::Storage(format!("S3 read error: {e}")))?;
        Ok(buf)
    }

    fn exists(&self, rel_path: &str) -> Result<bool> {
        let key = self.config.object_key(rel_path);
        match self.do_request("HEAD", &key, None, &[]) {
            Ok(_) => Ok(true),
            Err(MoonclipError::NotFound(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }

    fn get_range(&self, rel_path: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        let key = self.config.object_key(rel_path);
        self.ranged_get(&key, offset, len).map_err(|e| match e {
            MoonclipError::NotFound(_) => MoonclipError::NotFound(rel_path.to_string()),
            other => other,
        })
    }

    fn delete(&self, rel_path: &str) -> Result<()> {
        let key = self.config.object_key(rel_path);
        // S3 DELETE returns 204 even if the object doesn't exist
        self.do_request("DELETE", &key, None, &[])?;
        Ok(())
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        Ok(self.list_sized(prefix)?.into_iter().map(|(key, _)| key).collect())
    }

    fn list_sized(&self, prefix: &str) -> Result<Vec<(String, Option<u64>)>> {
        let full_prefix = self.config.object_key(prefix);
        let normalized_base = self.config.normalized_prefix();

        let mut all_keys = Vec::new();
        let mut continuation_token: Option<String> = None;

        loop {
            let mut params: Vec<(&str, &str)> = vec![("list-type", "2"), ("prefix", &full_prefix)];

            let ct_owned;
            if let Some(ref token) = continuation_token {
                ct_owned = token.clone();
                params.push(("continuation-token", &ct_owned));
            }

            // ListObjectsV2 is a GET on the bucket root with query params
            let resp = self.do_request("GET", "", None, &params)?;

            let body = resp
                .into_string()
                .map_err(|e| MoonclipError::Storage(format!("S3 list parse error: {e}")))?;

            // Parse XML response (minimal, no full XML parser needed)
            for (key, size) in listed_objects(&body) {
                // Strip our prefix to return relative paths
                if let Some(rel) = key.strip_prefix(&normalized_base) {
                    all_keys.push((rel.to_string(), size));
                } else {
                    all_keys.push((key, size));
                }
            }

            // Check for truncation
            if body.contains("<IsTruncated>true</IsTruncated>") {
                if let Some(token) = extract_xml_value(&body, "NextContinuationToken") {
                    continuation_token = Some(token);
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        Ok(all_keys)
    }
}

// ─── Minimal XML helpers (no dep needed for S3 ListObjects) ─────────

/// Undo the escaping S3 applies to key names in its XML.
///
/// Keys may hold `&`, `<` and `>`, and the ampersand is not exotic: a shard
/// name or metadata string containing `a&b` produces one. Without this,
/// listing returned `a&amp;b`, every later `get` and `delete` used that
/// literal key, and the object was invisible to retention — still billed,
/// never reachable. `&amp;` is expanded last so `&amp;lt;` comes back as the
/// text `&lt;` instead of being expanded twice.
fn unescape_xml(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

fn extract_xml_values(xml: &str, tag: &str) -> Vec<String> {
    let open = format!("<{}>", tag);
    let close = format!("</{}>", tag);
    let mut results = Vec::new();
    let mut search_from = 0;

    while let Some(start) = xml[search_from..].find(&open) {
        let abs_start = search_from + start + open.len();
        if let Some(end) = xml[abs_start..].find(&close) {
            results.push(unescape_xml(&xml[abs_start..abs_start + end]));
            search_from = abs_start + end + close.len();
        } else {
            break;
        }
    }

    results
}

fn extract_xml_value(xml: &str, tag: &str) -> Option<String> {
    extract_xml_values(xml, tag).into_iter().next()
}

/// Each object of a ListObjectsV2 page, with its size.
///
/// Read per `<Contents>` element, so a key and a size are always the same
/// object's: two flat lists of `Key` and `Size` would pair them by position,
/// and a page with an object missing one would shift every pair after it. The
/// size is what the remote sync needs to copy several files at once under a
/// cap on memory (GPU-235); one it cannot read is `None`, and that file is
/// copied alone.
fn listed_objects(xml: &str) -> Vec<(String, Option<u64>)> {
    let mut objects = Vec::new();
    for element in xml.split("<Contents>").skip(1) {
        let element = element.split("</Contents>").next().unwrap_or(element);
        if let Some(key) = extract_xml_value(element, "Key") {
            let size = extract_xml_value(element, "Size").and_then(|s| s.trim().parse().ok());
            objects.push((key, size));
        }
    }
    objects
}

// ─── SigV4 in the Authorization header (GPU-217) ────────────────────

/// The SHA-256 of nothing: a listing has no body.
const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// SigV4 in its header form for a `GET` with no body, used only where a
/// presigned URL cannot carry what the request needs (see
/// `S3Storage::listed`). Returns the canonical query string, which the URL
/// must carry exactly, and the headers to send, `Authorization` among them;
/// `host` is signed but not returned, since the client sends it from the URL.
fn header_signed(
    url: &Url,
    query: &[(&str, &str)],
    stamp: &str,
    region: &str,
    key: &str,
    secret: &str,
    token: Option<&str>,
) -> (String, Vec<(&'static str, String)>) {
    let mut pairs: Vec<(String, String)> = query.iter().map(|(k, v)| (aws_encode(k), aws_encode(v))).collect();
    pairs.sort();
    let query_string = pairs.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("&");
    let host = match url.port() {
        Some(port) => format!("{}:{port}", url.host_str().unwrap_or_default()),
        None => url.host_str().unwrap_or_default().to_string(),
    };
    let mut signed: Vec<(&'static str, String)> = vec![
        ("host", host),
        ("x-amz-content-sha256", EMPTY_SHA256.to_string()),
        ("x-amz-date", stamp.to_string()),
    ];
    if let Some(token) = token {
        signed.push(("x-amz-security-token", token.to_string()));
    }
    let names = signed.iter().map(|(name, _)| *name).collect::<Vec<_>>().join(";");

    let mut canonical = format!("GET\n{}\n{}\n", url.path(), query_string);
    for (name, value) in &signed {
        let _ = writeln!(canonical, "{name}:{}", value.trim());
    }
    let _ = write!(canonical, "\n{names}\n{EMPTY_SHA256}");

    let day = &stamp[..8];
    let scope = format!("{day}/{region}/s3/aws4_request");
    let to_sign = format!("AWS4-HMAC-SHA256\n{stamp}\n{scope}\n{}", hex(&Sha256::digest(canonical.as_bytes())));
    let mut signing = hmac(format!("AWS4{secret}").as_bytes(), day.as_bytes());
    for part in [region, "s3", "aws4_request"] {
        signing = hmac(&signing, part.as_bytes());
    }
    let signature = hex(&hmac(&signing, to_sign.as_bytes()));

    let mut headers: Vec<(&'static str, String)> = signed.into_iter().filter(|(name, _)| *name != "host").collect();
    headers.push((
        "Authorization",
        format!("AWS4-HMAC-SHA256 Credential={key}/{scope}, SignedHeaders={names}, Signature={signature}"),
    ));
    (query_string, headers)
}

/// Percent-encoding as SigV4 wants it: every byte but the unreserved ones.
fn aws_encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(byte as char);
        } else {
            let _ = write!(out, "%{byte:02X}");
        }
    }
    out
}

fn hmac(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC takes a key of any length");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut out, byte| {
        let _ = write!(out, "{byte:02x}");
        out
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(vars: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let vars: Vec<(String, String)> =
            vars.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |name| vars.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone())
    }

    #[test]
    fn credentials_given_win_over_the_environment() {
        let found = S3Credentials::resolve_with(
            Some("AK"),
            Some("SK"),
            None,
            env(&[("MOONCLIP_S3_ACCESS_KEY", "env"), ("MOONCLIP_S3_SECRET_KEY", "env"),
                  ("MOONCLIP_S3_SESSION_TOKEN", "env")]),
        )
        .unwrap();
        // The environment's token is not attached to keys it did not give.
        assert_eq!(found, S3Credentials { access_key: "AK".into(), secret_key: "SK".into(), session_token: None });
    }

    #[test]
    fn credentials_half_given_are_refused() {
        let found = S3Credentials::resolve_with(
            Some("AK"),
            None,
            None,
            env(&[("MOONCLIP_S3_SECRET_KEY", "env")]),
        );
        assert!(matches!(found, Err(MoonclipError::Config(_))));
    }

    #[test]
    fn credentials_come_from_moonclip_names_before_aws_ones() {
        let found = S3Credentials::resolve_with(
            None,
            None,
            None,
            env(&[("MOONCLIP_S3_ACCESS_KEY", "mk"), ("MOONCLIP_S3_SECRET_KEY", "ms"),
                  ("AWS_ACCESS_KEY_ID", "ak"), ("AWS_SECRET_ACCESS_KEY", "as"),
                  ("AWS_SESSION_TOKEN", "at")]),
        )
        .unwrap();
        assert_eq!(found, S3Credentials { access_key: "mk".into(), secret_key: "ms".into(), session_token: None });
    }

    #[test]
    fn credentials_never_mix_two_sources() {
        // Half a pair under Moonclip's names: the AWS pair is used, whole.
        let found = S3Credentials::resolve_with(
            None,
            None,
            None,
            env(&[("MOONCLIP_S3_ACCESS_KEY", "mk"), ("AWS_ACCESS_KEY_ID", "ak"),
                  ("AWS_SECRET_ACCESS_KEY", "as"), ("AWS_SESSION_TOKEN", "at")]),
        )
        .unwrap();
        assert_eq!(found, S3Credentials { access_key: "ak".into(), secret_key: "as".into(), session_token: Some("at".into()) });
    }

    #[test]
    fn credentials_missing_say_which_variables() {
        let found = S3Credentials::resolve_with(None, None, None, env(&[("AWS_ACCESS_KEY_ID", "")]));
        match found {
            Err(MoonclipError::Config(message)) => assert!(message.contains("MOONCLIP_S3_ACCESS_KEY")),
            other => panic!("expected a configuration error, got {other:?}"),
        }
    }

    #[test]
    fn test_object_key() {
        let config = S3Config {
            bucket: "my-bucket".into(),
            prefix: "checkpoints/harold".into(),
            region: "us-east-1".into(),
            endpoint: None,
            access_key: "AK".into(),
            secret_key: "SK".into(),
            session_token: None,
            path_style: false,
            timeout_secs: 30,
            single_put_limit: SINGLE_PUT_LIMIT,
        };

        assert_eq!(
            config.object_key("snapshots/abc/model.bin"),
            "checkpoints/harold/snapshots/abc/model.bin"
        );
    }

    #[test]
    fn test_normalized_prefix() {
        let mut config = S3Config {
            bucket: "b".into(),
            prefix: "/foo/bar/".into(),
            region: "r".into(),
            endpoint: None,
            access_key: "AK".into(),
            secret_key: "SK".into(),
            session_token: None,
            path_style: false,
            timeout_secs: 30,
            single_put_limit: SINGLE_PUT_LIMIT,
        };
        assert_eq!(config.normalized_prefix(), "foo/bar/");

        config.prefix = "".into();
        assert_eq!(config.normalized_prefix(), "");

        config.prefix = "solo".into();
        assert_eq!(config.normalized_prefix(), "solo/");
    }

    /// The part count is the thing that fails silently: S3 refuses an upload
    /// of more than 10000 parts, and the refusal arrives at completion, after
    /// every part has already been paid for and sent.
    ///
    /// Checked against sizes no test will ever upload, because that is exactly
    /// the point — the integration suite runs against MinIO, which accepts
    /// single `PUT`s far larger than S3 does, so nothing else here crosses the
    /// threshold this arithmetic exists for.
    #[test]
    fn no_object_size_produces_more_parts_than_s3_accepts() {
        let gib = 1024 * 1024 * 1024usize;
        for total in [
            SINGLE_PUT_LIMIT + 1,
            11 * gib,   // 1B parameters with Adam, gathered
            160 * gib,  // where the floor part size stops sufficing
            1024 * gib, // a 70B model, well past anything measured
            8 * 1024 * gib,
        ] {
            let size = part_size_for(total);
            let parts = total.div_ceil(size);
            assert!(
                parts <= MAX_PARTS,
                "{total} bytes at {size} per part is {parts} parts, over the {MAX_PARTS} ceiling"
            );
            assert!(
                size >= MIN_PART_SIZE,
                "{size} is below the 5 MiB S3 accepts for a non-final part"
            );
        }
    }

    /// R2 enforces a rule S3 does not, and rejects the whole upload for it:
    /// **every part but the last must be exactly the same size** — error
    /// 10048, `InvalidPart`, "All non-trailing parts must have the same size".
    ///
    /// `data.chunks(size)` satisfies it by construction, giving equal parts
    /// with a smaller remainder at the end. That is worth a test rather than a
    /// comment, because the obvious future change — sizing the last few parts
    /// differently to smooth out a tail — would keep every S3 test green and
    /// break R2 alone.
    ///
    /// The other three bounds are R2's documented limits, which for parts are
    /// the same numbers S3 uses: 5 MiB minimum, 5 GiB maximum, 10000 parts.
    #[test]
    fn the_part_layout_satisfies_r2_as_well_as_s3() {
        const R2_MIN_PART: usize = 5 * 1024 * 1024;
        const R2_MAX_PART: usize = 5 * 1024 * 1024 * 1024;
        let gib = 1024 * 1024 * 1024usize;

        for total in [
            SINGLE_PUT_LIMIT + 1,
            11 * gib,        // 1B parameters with Adam, gathered
            1024 * gib,      // 1 TiB
            5 * 1024 * gib,  // 5 TiB, R2's ceiling for an object
        ] {
            let size = part_size_for(total);
            let last = total % size;

            assert!((R2_MIN_PART..=R2_MAX_PART).contains(&size));
            assert!(total.div_ceil(size) <= MAX_PARTS);
            assert!(
                last == 0 || last < size,
                "the last part may differ, and only by being smaller: {total} bytes at {size} leaves {last}"
            );
        }
    }

    /// Small objects must not pay for a part size that scales: below the
    /// single-`PUT` limit this function is not consulted at all, and at the
    /// bottom of its range it has to stay at the floor.
    #[test]
    fn a_small_object_gets_the_floor_part_size() {
        assert_eq!(part_size_for(0), MIN_PART_SIZE);
        assert_eq!(part_size_for(MIN_PART_SIZE), MIN_PART_SIZE);
        assert_eq!(part_size_for(SINGLE_PUT_LIMIT), MIN_PART_SIZE);
    }

    #[test]
    fn test_xml_extract() {
        let xml = r#"
        <ListBucketResult>
            <Contents><Key>prefix/a.bin</Key></Contents>
            <Contents><Key>prefix/b.bin</Key></Contents>
            <IsTruncated>false</IsTruncated>
        </ListBucketResult>
        "#;

        let keys = extract_xml_values(xml, "Key");
        assert_eq!(keys, vec!["prefix/a.bin", "prefix/b.bin"]);

        let truncated = extract_xml_value(xml, "IsTruncated");
        assert_eq!(truncated, Some("false".to_string()));
    }

    /// S3 escapes key names in its XML, and a key with `&` in it is the
    /// common case: unescaped, every later call used a key that does not
    /// exist, so the object could never be read or deleted again.
    #[test]
    fn listed_keys_come_back_unescaped() {
        let xml = "<Contents><Key>runs/a&amp;b/rank_0.pack</Key></Contents>                   <Contents><Key>runs/x&lt;y&gt;z/rank_0.pack</Key></Contents>";
        let keys = extract_xml_values(xml, "Key");
        assert_eq!(keys[0], "runs/a&b/rank_0.pack");
        assert_eq!(keys[1], "runs/x<y>z/rank_0.pack");
    }

    /// GPU-235: each key with its own object's size, unescaped as `list`
    /// returns it, and `None` where a size is missing rather than the next
    /// object's.
    #[test]
    fn listed_objects_pair_each_key_with_its_size() {
        let xml = "<ListBucketResult>\
            <Contents><Key>p/a&amp;b.pack</Key><Size>4096</Size></Contents>\
            <Contents><Key>p/no-size.pack</Key></Contents>\
            <Contents><Key>p/c.pack</Key><LastModified>x</LastModified><Size>7</Size></Contents>\
            <IsTruncated>false</IsTruncated></ListBucketResult>";
        assert_eq!(
            listed_objects(xml),
            vec![
                ("p/a&b.pack".to_string(), Some(4096)),
                ("p/no-size.pack".to_string(), None),
                ("p/c.pack".to_string(), Some(7)),
            ]
        );
    }

    /// `&amp;lt;` is the text `&lt;`, not `<`: expanding the ampersand first
    /// would turn it into a tag delimiter that was never in the key.
    #[test]
    fn unescaping_does_not_run_twice() {
        assert_eq!(unescape_xml("a&amp;lt;b"), "a&lt;b");
    }

    /// Which failures are worth another attempt. Before 0.0.6 there were
    /// none, and a single 503 failed a checkpoint upload.
    #[test]
    fn retries_cover_the_far_side_failing_and_nothing_else() {
        let transport = MoonclipError::Storage("S3 transport error: connection reset".into());
        let throttled = MoonclipError::Storage("S3 HTTP 429: SlowDown".into());
        let unavailable = MoonclipError::Storage("S3 HTTP 503: please retry".into());
        assert!(is_retryable(&transport));
        assert!(is_retryable(&throttled));
        assert!(is_retryable(&unavailable));

        let denied = MoonclipError::Storage("S3 HTTP 403: SignatureDoesNotMatch".into());
        let missing = MoonclipError::NotFound("snapshots/a/rank_0.pack".into());
        assert!(!is_retryable(&denied), "a bad signature fails the same way twice");
        assert!(!is_retryable(&missing));
    }

    /// A key holding the digits 404 must not read as a missing object. The
    /// old check looked for that substring in the error text, and step
    /// numbers are exactly where it appears.
    #[test]
    fn a_key_containing_404_is_not_a_missing_object() {
        let real_failure =
            MoonclipError::Storage("S3 HTTP 500: snapshots/step_404/rank_0.pack".into());
        assert!(
            !matches!(real_failure, MoonclipError::NotFound(_)),
            "a server error was reported as a missing object"
        );
        assert!(is_retryable(&real_failure));
    }

    fn config(endpoint: Option<&str>, path_style: bool) -> S3Config {
        S3Config {
            bucket: "my-bucket".into(),
            prefix: "".into(),
            region: "eu-west-1".into(),
            endpoint: endpoint.map(Into::into),
            access_key: "AK".into(),
            secret_key: "SK".into(),
            session_token: None,
            path_style,
            timeout_secs: 30,
            single_put_limit: SINGLE_PUT_LIMIT,
        }
    }

    /// Where requests go, both ways of naming a bucket, with and without an
    /// endpoint of one's own.
    #[test]
    fn the_bucket_is_addressed_as_configured() {
        let address = |endpoint, path_style| bucket_of(&config(endpoint, path_style)).unwrap().base_url().to_string();
        // Virtual-hosted, AWS's default.
        assert_eq!(address(None, false), "https://my-bucket.s3.eu-west-1.amazonaws.com/");
        assert_eq!(address(None, true), "https://s3.eu-west-1.amazonaws.com/my-bucket/");
        // A gateway, path-style as they want.
        assert_eq!(address(Some("http://localhost:9000"), true), "http://localhost:9000/my-bucket/");
        assert_eq!(address(Some("http://localhost:9000/"), true), "http://localhost:9000/my-bucket/");
        // An endpoint with a path of its own keeps it.
        assert_eq!(address(Some("https://gw.example/s3"), true), "https://gw.example/s3/my-bucket/");
        assert!(bucket_of(&config(Some("not an address"), true)).is_err());
    }

    /// A key is encoded as SigV4 wants it in the path: everything but the
    /// unreserved characters and `/`. `&`, `+` and spaces are what shard and
    /// metadata names hold.
    #[test]
    fn a_key_is_encoded_into_the_path() {
        let store = S3Storage::new(config(Some("http://localhost:9000"), true)).unwrap();
        let url = store.signed_url("GET", "runs/a&b c+d/rank_0.pack", &[]).unwrap();
        assert_eq!(url.path(), "/my-bucket/runs/a%26b%20c%2Bd/rank_0.pack");
    }

    /// Long-lived keys sign without a token; temporary credentials put
    /// theirs in every request, inside what is signed.
    #[test]
    fn a_listing_signed_in_its_header_matches_aws_example() {
        // "GET Bucket (List Objects)" in AWS's Signature Version 4 examples
        // for the Authorization header.
        let url = Url::parse("https://examplebucket.s3.amazonaws.com/").unwrap();
        let (query, headers) = header_signed(
            &url,
            &[("prefix", "J"), ("max-keys", "2")],
            "20130524T000000Z",
            "us-east-1",
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            None,
        );
        assert_eq!(query, "max-keys=2&prefix=J");
        let authorization = &headers.iter().find(|(name, _)| *name == "Authorization").unwrap().1;
        assert_eq!(
            authorization,
            "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, \
             SignedHeaders=host;x-amz-content-sha256;x-amz-date, \
             Signature=34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7"
        );
    }

    #[test]
    fn a_listing_with_a_session_token_signs_it_as_a_header() {
        let url = Url::parse("https://acct.r2.cloudflarestorage.com/bucket/").unwrap();
        let (query, headers) = header_signed(
            &url,
            &[("list-type", "2"), ("prefix", "runs/a b")],
            "20261014T120000Z",
            "auto",
            "key",
            "secret",
            Some("session/token+1"),
        );
        assert_eq!(query, "list-type=2&prefix=runs%2Fa%20b");
        assert!(!query.contains("Security-Token"), "never in the query: R2 refuses it there");
        let names: Vec<&str> = headers.iter().map(|(name, _)| *name).collect();
        assert!(names.contains(&"x-amz-security-token"));
        let authorization = &headers.iter().find(|(name, _)| *name == "Authorization").unwrap().1;
        assert!(authorization.contains("SignedHeaders=host;x-amz-content-sha256;x-amz-date;x-amz-security-token,"));
    }

    #[test]
    fn a_session_token_goes_into_every_signature() {
        let plain = S3Storage::new(config(None, false)).unwrap();
        let url = plain.signed_url("PUT", "a/t1.bin", &[]).unwrap();
        assert!(url.query().unwrap().contains("X-Amz-Credential=AK%2F"));
        assert!(!url.query().unwrap().contains("X-Amz-Security-Token"));

        let mut temporary = config(None, false);
        temporary.session_token = Some("session/token+1".into());
        let url = S3Storage::new(temporary).unwrap().signed_url("PUT", "a/t1.bin", &[]).unwrap();
        let query = url.query().unwrap();
        assert!(query.contains("X-Amz-Security-Token=session%2Ftoken%2B1"));
        let signed = query.find("X-Amz-Signature=").unwrap();
        assert!(query.find("X-Amz-Security-Token").unwrap() < signed, "the token is signed over");
    }

    /// S3's own parameters reach the URL, whatever order they are given in:
    /// `rusty-s3` merges them as an ordered run, so they are sorted first.
    #[test]
    fn the_parameters_of_a_request_are_signed_in_order() {
        let store = S3Storage::new(config(Some("http://localhost:9000"), true)).unwrap();
        let url = store
            .signed_url("PUT", "k", &[("uploadId", "u-1"), ("partNumber", "3")])
            .unwrap();
        let query = url.query().unwrap();
        let names: Vec<&str> = query.split('&').map(|pair| pair.split('=').next().unwrap()).collect();
        // The signature last, and everything it covers in order before it.
        let (last, signed) = names.split_last().unwrap();
        assert_eq!(*last, "X-Amz-Signature");
        assert!(signed.windows(2).all(|pair| pair[0] <= pair[1]), "{names:?}");
        assert!(query.contains("partNumber=3") && query.contains("uploadId=u-1"));
        // An empty value, as `?uploads` is.
        let url = store.signed_url("POST", "k", &[("uploads", "")]).unwrap();
        assert!(url.query().unwrap().contains("uploads="));
    }

    #[test]
    fn test_auto_path_style() {
        // No endpoint → path_style stays false
        let config = S3Config {
            bucket: "b".into(),
            prefix: "".into(),
            region: "us-east-1".into(),
            endpoint: None,
            access_key: "AK".into(),
            secret_key: "SK".into(),
            session_token: None,
            path_style: false,
            timeout_secs: 30,
            single_put_limit: SINGLE_PUT_LIMIT,
        }
        .with_auto_path_style();
        assert!(!config.path_style);

        // Custom endpoint → path_style forced to true
        let config = S3Config {
            bucket: "b".into(),
            prefix: "".into(),
            region: "us-east-1".into(),
            endpoint: Some("http://172.17.0.2:9000".into()),
            access_key: "AK".into(),
            secret_key: "SK".into(),
            session_token: None,
            path_style: false,
            timeout_secs: 30,
            single_put_limit: SINGLE_PUT_LIMIT,
        }
        .with_auto_path_style();
        assert!(config.path_style);

        // Already true + endpoint → stays true
        let config = S3Config {
            bucket: "b".into(),
            prefix: "".into(),
            region: "us-east-1".into(),
            endpoint: Some("http://localhost:9000".into()),
            access_key: "AK".into(),
            secret_key: "SK".into(),
            session_token: None,
            path_style: true,
            timeout_secs: 30,
            single_put_limit: SINGLE_PUT_LIMIT,
        }
        .with_auto_path_style();
        assert!(config.path_style);
    }
}
