#!/usr/bin/env bash
#
# tessera_processor.sh — Sentinel-1 & Sentinel-2 Parallel Processing Pipeline
# Dependencies: bash ≥4, GNU coreutils, Python ≥3.7
# Usage: bash s1_s2_stacker.sh

# set -euo pipefail
set -u

#######################################
# USER CONFIGURABLE PARAMETERS
#######################################

# === Basic Configuration ===
YEAR=2024 # Range [2017-2024]

BASE_DIR="/home/azureuser/data/uk_d_pixel/${YEAR}/grid_-0.05_50.75"
OUT_DIR="${BASE_DIR}/data_processed"
DOWNSAMPLE_RATE=1

mkdir -p "$OUT_DIR"

# S1 stacking
: '
s1_stack 0.1.0
Process Sentinel-1 data for a single tile

USAGE:
    s1_stack [OPTIONS] --input-dir <input-dir> --output-dir <output-dir>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -i, --input-dir <input-dir>      Input directory (where TIFF files are)
    -o, --output-dir <output-dir>    Output directory (where processed NPY files will go)
    -p, --parallel <parallel>        Number of parallel processes to use [default: 8]
    -r, --rate <rate>                Downsampling rate (e.g., 10 means take every 10th pixel) [default: 10]
'

./s1_stack \
  --input-dir "${BASE_DIR}/data_sar_raw" \
  --output-dir $OUT_DIR \
  --parallel 8 \
  --rate $DOWNSAMPLE_RATE

# S2 stacking
: '
s2_stack 0.1.0
Process Sentinel-2 data for a single tile

USAGE:
    s2_stack [OPTIONS] --input <input-dir> --output <output-dir>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -b, --batch-size <batch-size>      Number of time slices to process in parallel [default: 5]
    -c, --cache-level <cache-level>    Cache strategy (0=minimal, 1=moderate, 2=aggressive) [default: 1]
    -i, --input <input-dir>            Input directory (where raw tiff files are organized in band folders)
    -n, --num-threads <num-threads>    Number of threads (default=10) to use for parallel tasks [default: 10]
    -o, --output <output-dir>          Output directory (where processed NPY files will go)
    -r, --sample-rate <sample-rate>    Downsample rate (default=10) [default: 10]
'

./s2_stack \
  --input "${BASE_DIR}/data_raw" \
  --output $OUT_DIR \
  --batch-size 16 \
  --cache-level 1 \
  --num-threads 8 \
  --sample-rate $DOWNSAMPLE_RATE


# Check if all S1 files were generated, if not create placeholders
echo "Checking S1 output files..."

S1_FILES=(
    "sar_ascending.npy"
    "sar_ascending_doy.npy"
    "sar_descending.npy"
    "sar_descending_doy.npy"
)

# Count how many S1 files exist
S1_FILE_COUNT=0
for file in "${S1_FILES[@]}"; do
    if [[ -f "$OUT_DIR/$file" ]]; then
        ((S1_FILE_COUNT++))
    fi
done

# If not all 4 files exist, generate placeholders
if [[ $S1_FILE_COUNT -lt 4 ]]; then
    echo "Only $S1_FILE_COUNT out of 4 S1 files found. Generating placeholder files..."
    
    # Create Python script to generate placeholder files
    cat > "$OUT_DIR/generate_s1_placeholders.py" << 'EOF'
import numpy as np
import sys
import os

def create_placeholder_npy_files(out_dir, masks_path):
    """Create placeholder S1 numpy files with correct shapes."""
    
    # Read masks.npy shape using memmap to get H and W
    try:
        masks = np.memmap(masks_path, dtype='bool', mode='r')
        # Reshape to get dimensions - need to figure out shape
        # Try to infer shape from file size
        file_size = os.path.getsize(masks_path)
        # Subtract numpy header size (usually around 80-128 bytes)
        data_size = file_size - 128  # approximate header size
        total_elements = data_size
        
        # We need to read the header properly to get the shape
        with open(masks_path, 'rb') as f:
            # Read numpy magic string
            magic = f.read(6)
            # Read version
            version = f.read(2)
            # Read header length
            if version[0] == 1:
                header_len = np.frombuffer(f.read(2), dtype=np.uint16)[0]
            else:
                header_len = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            # Read header
            header = f.read(header_len)
            # Parse header to get shape
            header_str = header.decode('ascii').strip()
            # Extract shape using string parsing
            shape_start = header_str.find("'shape': (") + len("'shape': (")
            shape_end = header_str.find(")", shape_start)
            shape_str = header_str[shape_start:shape_end]
            shape = tuple(map(int, shape_str.split(', ')))
        
        if len(shape) != 3:
            raise ValueError(f"Expected 3D shape for masks.npy, got {shape}")
        
        T, H, W = shape
        print(f"Found masks.npy with shape: T={T}, H={H}, W={W}")
        
    except Exception as e:
        print(f"Error reading masks.npy: {e}")
        print("Using default dimensions: H=86, W=78")
        H, W = 86, 78
    
    # Define files to create with their shapes
    files_to_create = [
        ("sar_ascending.npy", (0, H, W, 2), np.int16),
        ("sar_ascending_doy.npy", (0,), np.int16),
        ("sar_descending.npy", (0, H, W, 2), np.int16),
        ("sar_descending_doy.npy", (0,), np.int16)
    ]
    
    # Create each file if it doesn't exist
    for filename, shape, dtype in files_to_create:
        filepath = os.path.join(out_dir, filename)
        if not os.path.exists(filepath):
            print(f"Creating placeholder: {filename} with shape {shape} and dtype {dtype}")
            arr = np.zeros(shape, dtype=dtype)
            np.save(filepath, arr)
            print(f"Created {filename}")
        else:
            print(f"File already exists: {filename}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_s1_placeholders.py <output_dir> <masks_path>")
        sys.exit(1)
    
    out_dir = sys.argv[1]
    masks_path = sys.argv[2]
    create_placeholder_npy_files(out_dir, masks_path)
EOF

    # Run the Python script
    $PYTHON_ENV "$OUT_DIR/generate_s1_placeholders.py" "$OUT_DIR" "$OUT_DIR/masks.npy"
    
    # Clean up the temporary Python script
    rm -f "$OUT_DIR/generate_s1_placeholders.py"
    
    echo "Placeholder S1 files generated."
else
    echo "All 4 S1 files found. No placeholders needed."
fi

echo "Processing complete. Processed data is available in: $OUT_DIR"