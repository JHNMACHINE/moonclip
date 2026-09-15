#!/usr/bin/env bash
# Bring up an S3 server for the integration tests, and print how to run them.
#
#   bash tests/minio_up.sh
#
# Named for MinIO, which it ran until 2026-09-15. MinIO's open-source server is
# archived: its binaries answer `410 Gone` and its image is no longer pullable.
# This is versitygw now, the same server and version CI runs — see the "Start
# the S3 server" step in .forgejo/workflows/ci.yml, where the whole of
# tests/s3_minio.rs was checked against it.
#
# Idempotent: re-running replaces the container. Data lives in the container
# only, so stopping it throws the test objects away, which is what you want.
#
# The bucket is created by making the directory before the server starts — the
# posix backend serves each top-level directory under its root as a bucket,
# which avoids needing a client just to create one.

set -euo pipefail

NAME=moonclip-s3
IMAGE=versity/versitygw:v1.8.0
BUCKET=moonclip-test
PORT=9000
USER=minioadmin
PASS=minioadmin

# The old MinIO container too, if one is still around holding the port.
docker rm -f "$NAME" moonclip-minio >/dev/null 2>&1 || true

docker run -d --name "$NAME" \
    -p "${PORT}:9000" \
    --entrypoint sh \
    "$IMAGE" -c "mkdir -p /data/${BUCKET} && exec versitygw --access ${USER} --secret ${PASS} --port :9000 posix /data" >/dev/null

printf 'waiting for versitygw'
for _ in $(seq 1 30); do
    # Unauthenticated, the root answers 403: any status means it is listening.
    if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/" || true)" != "000" ]; then
        echo " — up on http://127.0.0.1:${PORT}, bucket '${BUCKET}'"
        cat <<EOF

Run the S3 integration tests with:

  MOONCLIP_S3_ENDPOINT=http://127.0.0.1:${PORT} \\
  MOONCLIP_S3_BUCKET=${BUCKET} \\
  MOONCLIP_S3_ACCESS_KEY=${USER} \\
  MOONCLIP_S3_SECRET_KEY=${PASS} \\
  MOONCLIP_S3_REGION=us-east-1 \\
    cargo test --test s3_minio

Every variable, including the region: a shell that already has MOONCLIP_S3_*
set for a real bucket would otherwise send part of the configuration there.

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
echo "versitygw did not come up; container logs:" >&2
docker logs "$NAME" 2>&1 | tail -20 >&2
exit 1
