#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/data/features"

echo "This will download LARGE files. ACAV100M is ~17GB."
echo "Edit this script if you want to skip ACAV100M."

# openWakeWord features (HuggingFace)
# ACAV100M (~17GB):
wget -O "$ROOT/data/features/openwakeword_features_ACAV100M_2000_hrs_16bit.npy" \
  "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"

# validation (~185MB):
wget -O "$ROOT/data/features/validation_set_features.npy" \
  "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"

echo "Done."
