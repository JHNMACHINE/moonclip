#!/usr/bin/env bash
# Bring up a MinIO for the S3 integration tests, and print how to run them.
#
#   bash tests/minio_up.sh
#
# Idempotent: re-running replaces the container. Data lives in the container
# only, so stopping it throws the test objects away, which is what you want.
#
# The bucket is created by making the directory before MinIO starts — it
# serves each top-level directory under /data as a bucket, which avoids
# needing `mc` just to create one.

set -euo pipefail

NAME=moonclip-minio
BUCKET=moonclip-test
PORT=9000
USER=minioadmin
PASS=minioadmin

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" \
    -p "${PORT}:9000" \
    -e "MINIO_ROOT_USER=${USER}" \
    -e "MINIO_ROOT_PASSWORD=${PASS}" \
    --entrypoint sh \
    minio/minio -c "mkdir -p /data/${BUCKET} && exec minio server /data" >/dev/null

printf 'waiting for MinIO'
for _ in $(seq 1 30); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/minio/health/live" || true)" = "200" ]; then
        echo " — up on http://127.0.0.1:${PORT}, bucket '${BUCKET}'"
        cat <<EOF

Run the S3 integration tests with:

  MOONCLIP_S3_ENDPOINT=http://127.0.0.1:${PORT} \\
  MOONCLIP_S3_BUCKET=${BUCKET} \\
  MOONCLIP_S3_ACCESS_KEY=${USER} \\
  MOONCLIP_S3_SECRET_KEY=${PASS} \\
    cargo test --test s3_minio

Without those variables the tests skip, so a plain \`cargo test\` stays green
on a machine with no Docker.

Tear down with:  docker rm -f ${NAME}
EOF
        exit 0
    fi
    printf '.'
    sleep 1
done

echo
echo "MinIO did not become healthy; container logs:" >&2
docker logs "$NAME" 2>&1 | tail -20 >&2
exit 1
