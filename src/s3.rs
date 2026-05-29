use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use std::fmt::Write as FmtWrite;

use crate::error::{Result, RevolverError};
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
) -> SignedRequest {
    let now = chrono::Utc::now();
    let date_stamp = now.format("%Y%m%d").to_string();
    let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
    let host = config.host();

    // Canonical URI
    let canonical_uri = if config.path_style {
        format!("/{}/{}", config.bucket, uri_encode(key, false))
    } else {
        format!("/{}", uri_encode(key, false))
    };

    // Canonical query string
    let mut sorted_params: Vec<(&str, &str)> = query_params.to_vec();
    sorted_params.sort_by_key(|(k, _)| *k);
    let canonical_qs: String = sorted_params
        .iter()
        .map(|(k, v)| format!("{}={}", uri_encode(k, true), uri_encode(v, true)))
        .collect::<Vec<_>>()
        .join("&");

    // Canonical headers (must include host, x-amz-content-sha256, x-amz-date)
    let canonical_headers = format!(
        "host:{}\nx-amz-content-sha256:{}\nx-amz-date:{}\n",
        host, payload_hash, amz_date
    );
    let signed_headers = "host;x-amz-content-sha256;x-amz-date";

    // Canonical request
    let canonical_request = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        method, canonical_uri, canonical_qs, canonical_headers, signed_headers, payload_hash
    );

    let credential_scope = format!("{}/{}/s3/aws4_request", date_stamp, config.region);
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{}\n{}\n{}",
        amz_date,
        credential_scope,
        sha256_hex(canonical_request.as_bytes())
    );

    let sig_key = signing_key(&config.secret_key, &date_stamp, &config.region, "s3");
    let signature_bytes = hmac_sha256(&sig_key, string_to_sign.as_bytes());
    let signature = hex::encode(signature_bytes);

    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
        config.access_key, credential_scope, signed_headers, signature
    );

    // Build URL
    let default_endpoint = format!("https://s3.{}.amazonaws.com", config.region);
    let endpoint = config.endpoint.as_deref().unwrap_or(&default_endpoint);
    let endpoint = endpoint.trim_end_matches('/');

    let url = if config.path_style {
        if canonical_qs.is_empty() {
            format!("{}/{}/{}", endpoint, config.bucket, key)
        } else {
            format!("{}/{}/{}?{}", endpoint, config.bucket, key, canonical_qs)
        }
    } else {
        let scheme_end = endpoint.find("://").map(|i| i + 3).unwrap_or(0);
        let (scheme, rest) = endpoint.split_at(scheme_end);
        if canonical_qs.is_empty() {
            format!("{}{}.{}/{}", scheme, config.bucket, rest, key)
        } else {
            format!("{}{}.{}/{}?{}", scheme, config.bucket, rest, key, canonical_qs)
        }
    };

    SignedRequest {
        url,
        headers: vec![
            ("Authorization".into(), authorization),
            ("x-amz-content-sha256".into(), payload_hash.into()),
            ("x-amz-date".into(), amz_date),
            ("Host".into(), host),
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

    fn do_request(
        &self,
        method: &str,
        key: &str,
        body: Option<&[u8]>,
        query_params: &[(&str, &str)],
    ) -> std::result::Result<ureq::Response, RevolverError> {
        let payload_hash = match body {
            Some(data) => sha256_hex(data),
            None => sha256_hex(b""),
        };

        let signed = sign_request(&self.config, method, key, &payload_hash, query_params);

        let mut req = match method {
            "GET" => self.agent.get(&signed.url),
            "PUT" => self.agent.put(&signed.url),
            "DELETE" => self.agent.delete(&signed.url),
            "HEAD" => self.agent.head(&signed.url),
            _ => return Err(RevolverError::Storage(format!("Unknown method: {method}"))),
        };

        for (k, v) in &signed.headers {
            req = req.set(k, v);
        }

        let response = if let Some(data) = body {
            req.send_bytes(data)
        } else {
            req.call()
        };

        response.map_err(|e| match e {
            ureq::Error::Status(code, resp) => {
                let body = resp.into_string().unwrap_or_default();
                RevolverError::Storage(format!("S3 HTTP {code}: {body}"))
            }
            ureq::Error::Transport(t) => {
                RevolverError::Storage(format!("S3 transport error: {t}"))
            }
        })
    }
}

impl StorageBackend for S3Storage {
    fn put(&self, rel_path: &str, data: &[u8]) -> Result<()> {
        let key = self.config.object_key(rel_path);
        self.do_request("PUT", &key, Some(data), &[])?;
        Ok(())
    }

    fn get(&self, rel_path: &str) -> Result<Vec<u8>> {
        let key = self.config.object_key(rel_path);
        let resp = self.do_request("GET", &key, None, &[]).map_err(|e| {
            // Convert 404 to NotFound
            let msg = e.to_string();
            if msg.contains("404") || msg.contains("NoSuchKey") {
                RevolverError::NotFound(rel_path.to_string())
            } else {
                e
            }
        })?;

        let mut buf = Vec::new();
        resp.into_reader()
            .read_to_end(&mut buf)
            .map_err(|e| RevolverError::Storage(format!("S3 read error: {e}")))?;
        Ok(buf)
    }

    fn exists(&self, rel_path: &str) -> Result<bool> {
        let key = self.config.object_key(rel_path);
        match self.do_request("HEAD", &key, None, &[]) {
            Ok(_) => Ok(true),
            Err(RevolverError::Storage(msg)) if msg.contains("404") => Ok(false),
            Err(e) => Err(e),
        }
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
            let mut params: Vec<(&str, &str)> =
                vec![("list-type", "2"), ("prefix", &full_prefix)];

            let ct_owned;
            if let Some(ref token) = continuation_token {
                ct_owned = token.clone();
                params.push(("continuation-token", &ct_owned));
            }

            // ListObjectsV2 is a GET on the bucket root with query params
            let resp = self.do_request("GET", "", None, &params)?;

            let body = resp
                .into_string()
                .map_err(|e| RevolverError::Storage(format!("S3 list parse error: {e}")))?;

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

fn extract_xml_values(xml: &str, tag: &str) -> Vec<String> {
    let open = format!("<{}>", tag);
    let close = format!("</{}>", tag);
    let mut results = Vec::new();
    let mut search_from = 0;

    while let Some(start) = xml[search_from..].find(&open) {
        let abs_start = search_from + start + open.len();
        if let Some(end) = xml[abs_start..].find(&close) {
            results.push(xml[abs_start..abs_start + end].to_string());
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
}
