#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/backgrounds"
mkdir -p "$DEST"

echo "Downloading MUSAN (~11GB) from OpenSLR..."
wget -O "$DEST/musan.tar.gz" "https://www.openslr.org/resources/17/musan.tar.gz"
echo "Extracting..."
tar -xzf "$DEST/musan.tar.gz" -C "$DEST"
echo "MUSAN extracted to $DEST/musan"
echo "You may delete $DEST/musan.tar.gz afterwards."
