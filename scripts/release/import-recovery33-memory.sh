#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
release_root=$(cd -- "$script_dir/../.." && pwd)
archive="$release_root/artifacts/memory/recovery33-memory.tar.gz"
checksum_dir="$release_root/artifacts/memory"
volume_name=${1:-tdai-memory-core-cockpit-zh-rc52-v7-recovery33-imported}

if ! command -v docker >/dev/null 2>&1; then
  printf '%s\n' 'Docker is required.' >&2
  exit 1
fi

if docker volume inspect "$volume_name" >/dev/null 2>&1; then
  printf 'Refusing to overwrite existing Docker volume: %s\n' "$volume_name" >&2
  exit 2
fi

(cd "$checksum_dir" && sha256sum -c SHA256SUMS)
docker volume create "$volume_name" >/dev/null

if ! docker run --rm \
  -v "$volume_name:/target" \
  -v "$archive:/snapshot/recovery33-memory.tar.gz:ro" \
  node:22-slim \
  tar -xzf /snapshot/recovery33-memory.tar.gz -C /target; then
  printf 'Import failed. The newly created volume was retained for inspection: %s\n' "$volume_name" >&2
  exit 3
fi

printf 'Recovery33 memory imported into Docker volume: %s\n' "$volume_name"
