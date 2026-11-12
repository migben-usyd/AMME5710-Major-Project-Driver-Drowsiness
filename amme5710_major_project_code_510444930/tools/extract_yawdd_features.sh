#!/usr/bin/env bash
# Batch-extract features from a local copy of the public YawDD dataset
# (Abtahi et al., 2014) for the external generalisation evaluation in
# tools/evaluate_yawdd.py. YawDD is not redistributed with this package --
# download it yourself and point SRC at your local copy before running.
#
# Usage:
#   SRC="/path/to/your/YawDD/Mirror" bash tools/extract_yawdd_features.sh
# or edit the default below and run:
#   bash tools/extract_yawdd_features.sh

set -e

SRC="${SRC:-/path/to/YawDD/Mirror}"
OUT="data/yawdd_features.csv"

if [ ! -d "$SRC" ]; then
  echo "Error: YawDD source directory not found: $SRC"
  echo "Set SRC to your local YawDD 'Mirror' folder, e.g.:"
  echo "  SRC=\"/path/to/YawDD dataset/Mirror\" bash tools/extract_yawdd_features.sh"
  exit 1
fi

rm -f "$OUT"
first=1
find "$SRC" -name "*.avi" | sort | while read -r f; do
  base=$(basename "$f" .avi)
  if [[ "$base" == *"Yawning"* ]]; then label=1; else label=0; fi
  cid="yawdd_$(echo "$base" | tr ' ' '_')"
  if [ $first -eq 1 ]; then APPEND=""; first=0; else APPEND="--append"; fi
  python3 src/extract_features_from_video.py --video "$f" --clip-id "$cid" --label "$label" --out "$OUT" $APPEND
done

echo "Done. Wrote $OUT"
