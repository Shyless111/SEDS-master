#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/sda/home/shihaoyu/Projects/SEDS-master"
cd "$ROOT_DIR"

conda run -n MoPa python scripts/extract_csl_openworld_audio.py \
  --video-source source_url \
  --annotation-dir Datasets/CSL-OpenWorld/annotations \
  --output-dir Datasets/CSL-OpenWorld/audio_from_source \
  --manifest Datasets/CSL-OpenWorld/audio_from_source_manifest.csv \
  --skip-existing \
  "$@"
