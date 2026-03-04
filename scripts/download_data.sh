#!/bin/bash
# Download MovieLens-25M dataset.
# Usage: bash scripts/download_data.sh [output_dir]

set -euo pipefail

OUTPUT_DIR="${1:-data/raw}"

mkdir -p "$OUTPUT_DIR"

URL="https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ZIP_FILE="$OUTPUT_DIR/ml-25m.zip"

echo "Downloading MovieLens-25M dataset..."
if command -v wget &> /dev/null; then
    wget -q --show-progress -O "$ZIP_FILE" "$URL"
elif command -v curl &> /dev/null; then
    curl -L -o "$ZIP_FILE" "$URL"
else
    echo "Error: wget or curl required." >&2
    exit 1
fi

echo "Extracting..."
unzip -o -q "$ZIP_FILE" -d "$OUTPUT_DIR"

echo "Cleaning up..."
rm -f "$ZIP_FILE"

echo "Done. Data saved to $OUTPUT_DIR/ml-25m/"
ls -la "$OUTPUT_DIR/ml-25m/"
