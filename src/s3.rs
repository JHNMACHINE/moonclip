//! S3-compatible storage over a hand-written SigV4.
//!
//! Signing is ours rather than the AWS SDK's, and that is a decision rather
//! than an accident: the SDK speaks the same one protocol this does — it is
//! not a multi-cloud abstraction — while bringing a dependency tree into a
//! library that is deliberately thin. Its recent defaults also send checksum
//! headers that GCS and other S3-compatible services reject, so adopting it
//! would cost configuration to get back where we already are. Reconsider if
//! the list below stops being short.
//!
//! # What is not covered
//!
//! Written down because ten integration tests against MinIO make this module
//! look more complete than it is. MinIO is forgiving in the places real S3 is
//! not, and the next person here should not start from the assumption that
//! everything is handled.
//!
//! * **Temporary credentials.** Only a long-lived access key and secret. No
//!   `x-amz-security-token` header is signed or sent, so anything issuing
//!   session credentials — `AssumeRole`, EC2 instance profiles, IRSA on EKS,
//!   any OIDC federation — cannot be used. The credentials have to be static.
//! * **Objects above the single-`PUT` ceiling.** Every write is one `PUT`, and
//!   S3 refuses a single `PUT` over 5 GiB; multipart upload is not
//!   implemented. A gathered checkpoint of a large model reaches that on its
//!   own. See the tracking issue.
//! * **Clock skew.** The signature is stamped with the local clock and a
//!   `RequestTimeTooSkewed` is treated as any other error. A host more than
//!   fifteen minutes off signs requests that will never be accepted, and the
//!   error does not say so in those words.
//! * **Checksum headers.** None are sent (`Content-MD5`, `x-amz-checksum-*`).
//!   Integrity is covered a layer up — every tensor carries its own hash in
//!   the manifest — so this is a deliberate omission, and it is also why the
//!   module works unmodified against GCS and R2.
//!
//! Path-style versus virtual-hosted addressing, percent-encoding of key names,
//! and `ListObjectsV2` pagination *are* handled; see [`S3Config`] and the
//! `uri_encode` and `list` implementations.

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use std::fmt::Write as FmtWrite;

use crate::error::{Result, MoonclipError};
use crate::storage::StorageBackend;

type HmacSha256 = Hmac<Sha256>;

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
    /// Use path-style URLs (required for MinIO, optional for AWS).
    /// If true:  http://endpoint/bucket/key
    /// If false: http://bucket.endpoint/key (virtual-hosted)
    pub path_style: bool,
    /// Connection timeout in seconds.
    pub timeout_secs: u64,
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

    /// Build the host for requests.
    #[allow(dead_code)]
    fn host(&self) -> String {
        let default_endpoint = format!("https://s3.{}.amazonaws.com", self.region);
        let base = self.endpoint.as_deref().unwrap_or(&default_endpoint);

        let base = base
            .trim_end_matches('/')
            .replace("https://", "")
            .replace("http://", "");

        if self.path_style {
            base.to_string()
        } else {
            format!("{}.{}", self.bucket, base)
        }
    }

    /// Build the base URL for requests.
    #[allow(dead_code)]
    fn base_url(&self) -> String {
        let default_endpoint = format!("https://s3.{}.amazonaws.com", self.region);
        let endpoint = self.endpoint.as_deref().unwrap_or(&default_endpoint);
        let endpoint = endpoint.trim_end_matches('/');

        if self.path_style {
            format!("{}/{}", endpoint, self.bucket)
        } else {
            // Insert bucket as subdomain
            let scheme_end = endpoint.find("://").map(|i| i + 3).unwrap_or(0);
            let (scheme, rest) = endpoint.split_at(scheme_end);
            format!("{}{}.{}", scheme, self.bucket, rest)
        }
    }
}

// ─── AWS Signature V4 ───────────────────────────────────────────────

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut s = String::with_capacity(64);
    for byte in result {
        write!(s, "{:02x}", byte).unwrap();
    }
    s
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC key length");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

/// Build a signing key for AWS SigV4.
fn signing_key(secret: &str, date: &str, region: &str, service: &str) -> Vec<u8> {
    let k_date = hmac_sha256(format!("AWS4{}", secret).as_bytes(), date.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    hmac_sha256(&k_service, b"aws4_request")
}

/// URL-encode a string per AWS rules (encode everything except unreserved chars).
fn uri_encode(s: &str, encode_slash: bool) -> String {
    let mut result = String::with_capacity(s.len() * 2);
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                result.push(byte as char);
            }
            b'/' if !encode_slash => {
                result.push('/');
            }
            _ => {
                write!(result, "%{:02X}", byte).unwrap();
            }
        }
    }
    result
}

/// Above this, one `PUT` is not allowed and the upload has to be split.
///
/// S3 refuses a single `PUT` over 5 GiB. The threshold here is lower on
/// purpose: nothing is gained by riding the limit, and a gathered checkpoint
/// of a large model is well past it either way — 1B parameters with Adam is
/// around 11 GiB, and `gather` is Ravex's default, so this is the ordinary
/// path rather than an extreme one.
const SINGLE_PUT_LIMIT: usize = 4 * 1024 * 1024 * 1024;

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

struct SignedRequest {
    url: String,
    headers: Vec<(String, String)>,
}

/// Sign an S3 request using AWS Signature V4.
fn sign_request(
    config: &S3Config,
    method: &str,
    key: &str,
    payload_hash: &str,
    query_params: &[(&str, &str)],
    content_length: Option<usize>,
) -> SignedRequest {
    let now = chrono::Utc::now();
    let date_stamp = now.format("%Y%m%d").to_string();
    let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();

    // Canonical URI – key encoded, slashes kept
    let encoded_key = uri_encode(key, false);
    let canonical_uri = if config.path_style {
        format!("/{}/{}", config.bucket, encoded_key)
    } else {
        format!("/{}", encoded_key)
    };

    // Canonical query string
    let mut sorted_params: Vec<(&str, &str)> = query_params.to_vec();
    sorted_params.sort_by_key(|(k, _)| *k);
    let canonical_qs: String = sorted_params
        .iter()
        .map(|(k, v)| format!("{}={}", uri_encode(k, true), uri_encode(v, true)))
        .collect::<Vec<_>>()
        .join("&");

    // Build URL (using encoded key)
    let default_endpoint = format!("https://s3.{}.amazonaws.com", config.region);
    let endpoint = config.endpoint.as_deref().unwrap_or(&default_endpoint);
    let endpoint = endpoint.trim_end_matches('/');

    let url = if config.path_style {
        if canonical_qs.is_empty() {
            format!("{}/{}/{}", endpoint, config.bucket, encoded_key)
        } else {
            format!(
                "{}/{}/{}?{}",
                endpoint, config.bucket, encoded_key, canonical_qs
            )
        }
    } else {
        let scheme_end = endpoint.find("://").map(|i| i + 3).unwrap_or(0);
        let (scheme, rest) = endpoint.split_at(scheme_end);
        if canonical_qs.is_empty() {
            format!("{}{}.{}/{}", scheme, config.bucket, rest, encoded_key)
        } else {
            format!(
                "{}{}.{}/{}?{}",
                scheme, config.bucket, rest, encoded_key, canonical_qs
            )
        }
    };

    // Extract the host from the URL (lowercase, exactly what ureq will send)
    let host = url
        .split("://")
        .nth(1)
        .unwrap_or("")
        .split('/')
        .next()
        .unwrap_or("")
        .to_lowercase();

    // ─── Build canonical headers in **alphabetical order** ──────────
    // Store (lowercase_name, original_value) pairs.
    let mut headers_to_sign: Vec<(String, String)> = Vec::new();

    // Always include these three
    headers_to_sign.push(("host".into(), host.clone()));
    headers_to_sign.push(("x-amz-content-sha256".into(), payload_hash.to_string()));
    headers_to_sign.push(("x-amz-date".into(), amz_date.clone()));

    // Add content-length only if a body is present
    if let Some(len) = content_length {
        headers_to_sign.push(("content-length".into(), len.to_string()));
    }

    // Sort by the lowercased header name (required by SigV4)
    headers_to_sign.sort_by(|a, b| a.0.to_lowercase().cmp(&b.0.to_lowercase()));

    // Build canonical_headers string
    let canonical_headers: String = headers_to_sign
        .iter()
        .map(|(name, value)| format!("{}:{}\n", name.to_lowercase(), value))
        .collect();

    // Build signed_headers list (alphabetically)
    let signed_headers: String = headers_to_sign
        .iter()
        .map(|(name, _)| name.to_lowercase())
        .collect::<Vec<_>>()
        .join(";");

    // ─── Canonical request ──────────────────────────────────────────
    let canonical_request = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        method, canonical_uri, canonical_qs, canonical_headers, signed_headers, payload_hash
    );

    // ─── String to sign ─────────────────────────────────────────────
    let credential_scope = format!("{}/{}/s3/aws4_request", date_stamp, config.region);
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{}\n{}\n{}",
        amz_date,
        credential_scope,
        sha256_hex(canonical_request.as_bytes())
    );

    // ─── Signature ──────────────────────────────────────────────────
    let sig_key = signing_key(&config.secret_key, &date_stamp, &config.region, "s3");
    let signature_bytes = hmac_sha256(&sig_key, string_to_sign.as_bytes());
    let signature = hex::encode(signature_bytes);

    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
        config.access_key, credential_scope, signed_headers, signature
    );

    // ─── Headers to send ────────────────────────────────────────────
    // Note: we do NOT set "Host" or "Content-Length" manually;
    // ureq will set them automatically based on the URL and body.

    SignedRequest {
        url,
        headers: vec![
            ("Authorization".into(), authorization),
            ("x-amz-content-sha256".into(), payload_hash.into()),
            ("x-amz-date".into(), amz_date),
        ],
    }
}

// ─── S3 Storage Backend ─────────────────────────────────────────────

pub struct S3Storage {
    config: S3Config,
    agent: ureq::Agent,
}

impl S3Storage {
    pub fn new(config: S3Config) -> Result<Self> {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(config.timeout_secs))
            .timeout_read(std::time::Duration::from_secs(config.timeout_secs * 10))
            .timeout_write(std::time::Duration::from_secs(config.timeout_secs * 10))
            .build();

        Ok(S3Storage { config, agent })
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

    fn attempt_request(
        &self,
        method: &str,
        key: &str,
        body: Option<&[u8]>,
        query_params: &[(&str, &str)],
    ) -> std::result::Result<ureq::Response, MoonclipError> {
        let payload_hash = match body {
            Some(data) => sha256_hex(data),
            None => sha256_hex(b""),
        };

        let content_length = body.map(|d| d.len());

        let signed = sign_request(
            &self.config,
            method,
            key,
            &payload_hash,
            query_params,
            content_length,
        );

        let mut req = match method {
            "GET" => self.agent.get(&signed.url),
            "PUT" => self.agent.put(&signed.url),
            "DELETE" => self.agent.delete(&signed.url),
            "HEAD" => self.agent.head(&signed.url),
            "POST" => self.agent.post(&signed.url),
            _ => return Err(MoonclipError::Storage(format!("Unknown method: {method}"))),
        };

        for (k, v) in &signed.headers {
            req = req.set(k, v);
        }

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
                        "[Moonclip] could not abort the multipart upload of {key}:                          {cleanup}. Its parts are still in the bucket, invisible                          to a listing and still billed."
                    );
                }
                Err(e)
            }
        }
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
    /// kilobytes — once per tensor. `Range` is not part of the signed header
    /// set (host, x-amz-date and x-amz-content-sha256 are), so it can be
    /// attached after signing.
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
            let signed = sign_request(&self.config, "GET", key, &sha256_hex(b""), &[], None);
            let mut req = self.agent.get(&signed.url);
            for (k, v) in &signed.headers {
                req = req.set(k, v);
            }

            match req.set("Range", &range).call() {
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
        if data.len() > SINGLE_PUT_LIMIT {
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
            for key in extract_xml_values(&body, "Key") {
                // Strip our prefix to return relative paths
                if let Some(rel) = key.strip_prefix(&normalized_base) {
                    all_keys.push(rel.to_string());
                } else {
                    all_keys.push(key);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uri_encode() {
        assert_eq!(uri_encode("hello world", true), "hello%20world");
        assert_eq!(uri_encode("path/to/file", false), "path/to/file");
        assert_eq!(uri_encode("path/to/file", true), "path%2Fto%2Ffile");
        assert_eq!(uri_encode("file-name_v2.bin", true), "file-name_v2.bin");
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
            path_style: false,
            timeout_secs: 30,
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
            path_style: false,
            timeout_secs: 30,
        };
        assert_eq!(config.normalized_prefix(), "foo/bar/");

        config.prefix = "".into();
        assert_eq!(config.normalized_prefix(), "");

        config.prefix = "solo".into();
        assert_eq!(config.normalized_prefix(), "solo/");
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

    #[test]
    fn test_signing_deterministic() {
        // Verify that the signing key derivation is deterministic
        let key1 = signing_key("secret", "20240101", "us-east-1", "s3");
        let key2 = signing_key("secret", "20240101", "us-east-1", "s3");
        assert_eq!(key1, key2);

        // Different date → different key
        let key3 = signing_key("secret", "20240102", "us-east-1", "s3");
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_host_generation() {
        let mut config = S3Config {
            bucket: "my-bucket".into(),
            prefix: "".into(),
            region: "eu-west-1".into(),
            endpoint: None,
            access_key: "AK".into(),
            secret_key: "SK".into(),
            path_style: false,
            timeout_secs: 30,
        };

        // Virtual-hosted style (AWS default)
        assert_eq!(config.host(), "my-bucket.s3.eu-west-1.amazonaws.com");

        // Path style (MinIO etc.)
        config.path_style = true;
        assert_eq!(config.host(), "s3.eu-west-1.amazonaws.com");

        // Custom endpoint with path style
        config.endpoint = Some("http://localhost:9000".into());
        assert_eq!(config.host(), "localhost:9000");

        // Custom endpoint with virtual-hosted
        config.path_style = false;
        assert_eq!(config.host(), "my-bucket.localhost:9000");
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
            path_style: false,
            timeout_secs: 30,
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
            path_style: false,
            timeout_secs: 30,
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
            path_style: true,
            timeout_secs: 30,
        }
        .with_auto_path_style();
        assert!(config.path_style);
    }
}
