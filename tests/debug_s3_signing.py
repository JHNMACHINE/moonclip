"""
Debug script: replicate Moonclip's SigV4 signing in Python and test against MinIO.

This does EXACTLY what src/s3.rs does, step by step, so we can see
where the signature diverges from what MinIO expects.

Usage:
    python tests/debug_s3_signing.py --endpoint http://192.168.1.50:9000 \
        --bucket moonclip-test --access-key minioadmin --secret-key minioadmin
"""

import argparse
import hashlib
import hmac
from datetime import datetime, timezone

import requests  # pip install requests


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hmac_sha256(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()


def signing_key(secret: str, date: str, region: str, service: str) -> bytes:
    k_date = hmac_sha256(f"AWS4{secret}".encode(), date.encode())
    k_region = hmac_sha256(k_date, region.encode())
    k_service = hmac_sha256(k_region, service.encode())
    return hmac_sha256(k_service, b"aws4_request")


def uri_encode(s: str, encode_slash: bool = True) -> str:
    result = []
    for byte in s.encode("utf-8"):
        ch = chr(byte)
        if ch.isalnum() or ch in "-_.~":
            result.append(ch)
        elif ch == "/" and not encode_slash:
            result.append("/")
        else:
            result.append(f"%{byte:02X}")
    return "".join(result)


def sign_and_send(
    endpoint: str,
    bucket: str,
    region: str,
    access_key: str,
    secret_key: str,
    method: str,
    key: str,
    body: bytes | None = None,
    query_params: dict | None = None,
) -> requests.Response:
    """Replicate Moonclip's sign_request + do_request exactly."""

    now = datetime.now(timezone.utc)
    date_stamp = now.strftime("%Y%m%d")
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")

    payload_hash = sha256_hex(body if body else b"")

    # Host (path-style: just the endpoint without scheme)
    host = endpoint.replace("https://", "").replace("http://", "").rstrip("/")

    # Canonical URI (path-style)
    canonical_uri = f"/{bucket}/{uri_encode(key, encode_slash=False)}"

    # Canonical query string
    qp = query_params or {}
    sorted_params = sorted(qp.items())
    canonical_qs = "&".join(
        f"{uri_encode(k, True)}={uri_encode(v, True)}" for k, v in sorted_params
    )

    # Canonical headers (MUST be sorted, each ending with \n)
    canonical_headers = (
        f"host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n"
    )
    signed_headers = "host;x-amz-content-sha256;x-amz-date"

    # Canonical request
    canonical_request = (
        f"{method}\n"
        f"{canonical_uri}\n"
        f"{canonical_qs}\n"
        f"{canonical_headers}\n"
        f"{signed_headers}\n"
        f"{payload_hash}"
    )

    credential_scope = f"{date_stamp}/{region}/s3/aws4_request"
    string_to_sign = (
        f"AWS4-HMAC-SHA256\n"
        f"{amz_date}\n"
        f"{credential_scope}\n"
        f"{sha256_hex(canonical_request.encode())}"
    )

    sig_key = signing_key(secret_key, date_stamp, region, "s3")
    signature = hmac.new(sig_key, string_to_sign.encode(), hashlib.sha256).hexdigest()

    authorization = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )

    # Build URL (path-style)
    endpoint = endpoint.rstrip("/")
    if canonical_qs:
        url = f"{endpoint}/{bucket}/{key}?{canonical_qs}"
    else:
        url = f"{endpoint}/{bucket}/{key}"

    headers = {
        "Authorization": authorization,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_date,
        "Host": host,
    }

    # Debug output
    print(f"\n{'=' * 60}")
    print(f"  {method} {url}")
    print(f"{'=' * 60}")
    print(f"  Host header:     {host}")
    print(f"  Canonical URI:   {canonical_uri}")
    print(f"  Canonical QS:    {canonical_qs}")
    print(f"  Payload hash:    {payload_hash[:16]}...")
    print(f"  Date:            {amz_date}")
    print(f"  Signature:       {signature[:16]}...")
    print("\n  Canonical request:\n")
    for line in canonical_request.split("\n"):
        print(f"    |{line}|")
    print()

    resp = requests.request(
        method=method,
        url=url,
        headers=headers,
        data=body,
        allow_redirects=False,
    )

    return resp


def main():
    parser = argparse.ArgumentParser(description="Debug S3 SigV4 signing")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--access-key", required=True)
    parser.add_argument("--secret-key", required=True)
    args = parser.parse_args()

    test_key = "debug-test/hello.txt"
    test_body = b"Hello from Moonclip debug script!"

    # Test 1: PUT
    print("\n[TEST 1] PUT object")
    resp = sign_and_send(
        args.endpoint,
        args.bucket,
        args.region,
        args.access_key,
        args.secret_key,
        "PUT",
        test_key,
        test_body,
    )
    print(f"  -> {resp.status_code} {resp.text[:200]}")

    if resp.status_code == 200:
        # Test 2: HEAD (exists)
        print("\n[TEST 2] HEAD object")
        resp = sign_and_send(
            args.endpoint,
            args.bucket,
            args.region,
            args.access_key,
            args.secret_key,
            "HEAD",
            test_key,
        )
        print(f"  -> {resp.status_code}")

        # Test 3: GET
        print("\n[TEST 3] GET object")
        resp = sign_and_send(
            args.endpoint,
            args.bucket,
            args.region,
            args.access_key,
            args.secret_key,
            "GET",
            test_key,
        )
        print(f"  -> {resp.status_code} body={resp.content[:50]}")

        # Test 4: DELETE
        print("\n[TEST 4] DELETE object")
        resp = sign_and_send(
            args.endpoint,
            args.bucket,
            args.region,
            args.access_key,
            args.secret_key,
            "DELETE",
            test_key,
        )
        print(f"  -> {resp.status_code}")
    else:
        print("\n  PUT failed, skipping remaining tests.")
        print("  The canonical request above shows EXACTLY what Moonclip signs.")
        print("  Compare with MinIO's expected signing to find the mismatch.")


if __name__ == "__main__":
    main()
