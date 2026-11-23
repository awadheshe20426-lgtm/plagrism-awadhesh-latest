#!/usr/bin/env bash
# Nightly backup of the PlagDetect Postgres database and MinIO uploads.
# Usage: scripts/backup.sh [dest_dir]   (default: ./backups)
# Cron example (2am daily):  0 2 * * * /home/ubuntu/plagiarism-detection/scripts/backup.sh >> /var/log/plagdetect-backup.log 2>&1
set -euo pipefail

DEST="${1:-$(dirname "$0")/../backups}"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DEST"

# --- PostgreSQL (pgvector) — logical dump ---
sudo docker exec plagiarism_db pg_dump -U user -d plagiarism_db \
  | gzip > "$DEST/plagiarism_db-$STAMP.sql.gz"

# --- MinIO objects (uploaded submissions) ---
# Mirror the data volume; adjust bucket if you rename it.
sudo docker run --rm --network container:plagiarism_minio \
  -v "$(cd "$DEST" && pwd):/backup" minio/mc:latest sh -c '
    mc alias set local http://127.0.0.1:9000 minioadmin minioadmin >/dev/null 2>&1 || true
    mc mirror --overwrite local /backup/minio-'"$STAMP"' 2>/dev/null || true
  ' || echo "MinIO backup skipped (mc unavailable)"

# --- Retention: keep the 14 most recent DB dumps ---
ls -1t "$DEST"/plagiarism_db-*.sql.gz 2>/dev/null | tail -n +15 | xargs -r rm -f

echo "Backup complete: $DEST (db dump: plagiarism_db-$STAMP.sql.gz)"
