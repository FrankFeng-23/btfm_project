#!/bin/bash
set -e  # Any step fails, exit the script

## CONFIG
BASENAME=region                                       # final result file name (.npy and .tif)
YEAR=2025                                             # data year, range [2017-2025]
BASE=/abs/path/to/my_data                             # full path  to base dir, all files go here
RESOLUTION=10                                         # resolution for download and process

ROI_SHP="${BASE}/${BASENAME}.shp"                     # path to your shapefile
DATA_DIR="${BASE}/${BASENAME}_${YEAR}"                # path to store all data

PYTHON_ENV=`which python` # absolute path to your interpreter; only the Step 1 downloader needs it

## STEP 0
ROI_TIFF="${DATA_DIR}/0.roi/roi.tiff"     # ROI extent: downloaded over + used as geo-reference
mkdir -p "${DATA_DIR}/0.roi" "${DATA_DIR}/tmp"
python tessera_preprocessing/convert_shp_to_tiff.py \
    --shp_path "${ROI_SHP}" --tiff_path  "${ROI_TIFF}" --pixel_size "${RESOLUTION}"

## STEP 1
INPUT_TIFF="${ROI_TIFF}" OUT_DIR="${DATA_DIR}" TEMP_DIR="${DATA_DIR}/tmp" \
PYTHON_ENV="${PYTHON_ENV}" YEAR="${YEAR}" DATA_SOURCE=mpc RESOLUTION="${RESOLUTION}" \
S1_RAW_SUBDIR=1.data_sar_raw S2_RAW_SUBDIR=1.data_raw S1_OVERWRITE=false S2_OVERWRITE=false \
    bash tessera_preprocessing/s1_s2_downloader.sh
	
## STEP 2
BASE_DIR="${DATA_DIR}" DOWNSAMPLE_RATE=1 \
S1_RAW_SUBDIR=1.data_sar_raw S2_RAW_SUBDIR=1.data_raw PROCESSED_SUBDIR=2.data_processed \
    bash tessera_preprocessing/s1_s2_stacker.sh

## STEP 3
python tessera_preprocessing/dpixel_retiler.py \
    --tiff_path "${ROI_TIFF}" --d_pixel_dir "${DATA_DIR}/2.data_processed" \
    --out_dir "${DATA_DIR}/3.retiled_d_pixel" \
    --patch_size 500 --block_size 2000 --num_workers 16 --overwrite

## STEP 4
mkdir -p "${DATA_DIR}/4.embeddings_v2"
python tessera_infer_v2/infer_v2.py \
    --model medium --data-root "${DATA_DIR}/3.retiled_d_pixel" --out-dir "${DATA_DIR}/4.embeddings_v2"

## STEP 5
mkdir -p "${DATA_DIR}/5.result"
python tessera_infer/stitch_tiled_representation.py \
    --d_pixel_retiled_path "${DATA_DIR}/3.retiled_d_pixel" \
    --representation_retiled_path "${DATA_DIR}/4.embeddings_v2" \
    --downstream_tiff "${ROI_TIFF}" --out_dir "${DATA_DIR}/5.result" --out_name "${BASENAME}"
python tessera_infer/convert_npy2tiff.py \
    --npy_path "${DATA_DIR}/5.result/${BASENAME}.npy" \
    --ref_tiff_path "${ROI_TIFF}" --out_dir "${DATA_DIR}/5.result" --downsample_rate 1
