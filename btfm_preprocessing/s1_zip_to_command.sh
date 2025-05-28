#!/usr/bin/env bash
ZIP_PATH="$1"

# The tile_id is the directory name that holds the zip, e.g. /.../data_sar/<tile_id>/xxxx.zip
TILE_ID=$(basename "$(dirname "$ZIP_PATH")")

# Reference TIF: S2 raw directory + 'red' band
# Actually we need the S2_DATA_RAW from env, so:
RED_DIR="${S2_DATA_RAW}/${TILE_ID}/red"
# Or if you prefer to check the new structure that s2_processor creates, adapt if needed.
# We'll just take the first .tiff
REF_TIF=$(ls -1 "${RED_DIR}"/*.tiff 2>/dev/null | head -n 1)

if [ -z "$REF_TIF" ]; then
  echo "echo 'No reference TIF for tile $TILE_ID (cannot process $ZIP_PATH)' >&2"
  exit 0
fi

# Output .tif directory is S1_DATA_PROCESSED + tile
OUT_DIR="${S1_DATA_PROCESSED}/${TILE_ID}"

echo "$PYTHON s1_processor.py --zip_file \"$ZIP_PATH\" --reference_tiff \"$REF_TIF\" --out_dir \"$OUT_DIR\""
