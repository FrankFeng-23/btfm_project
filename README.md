# Temporal Embeddings of Surface Spectra for Earth Representation and Analysis (TESSERA)
<a name="readme-top"></a>
<div>
    <a align="center">
        <img src="images/banner.png"> 
    </a>
    <p align="center">
        <!-- 关键的布局部分 -->
        <div style="display: flex; justify-content: space-evenly; align-items: center; width: 100%;">
            <a href="https://www.youtube.com/watch?v=eERBj4FUD3s" style="flex-grow: 1; text-align: center; padding: 0 10px;">View Our Talk 🌐</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            <a href="https://github.com/FrankFeng-23/btfm_project" style="flex-grow: 1; text-align: center; padding: 0 10px;">View Our Paper (Upcoming) :bookmark_tabs:</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            <a href="https://github.com/FrankFeng-23/btfm_project/issues" style="flex-grow: 1; text-align: center; padding: 0 10px;">Report Bug :hammer_and_wrench:</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            <a href="https://github.com/FrankFeng-23/btfm_project/issues" style="flex-grow: 1; text-align: center; padding: 0 10px;">Request Feature 🙋</a>
        </div>
    </p>
</div>

![Version](https://img.shields.io/badge/version-alpha-red)
![Status](https://img.shields.io/badge/status-internal%20use%20only-yellow)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

**Alpha testing, primarily for Cambridge internal users. External users can contact zf281@cam.ac.uk for access.**

**We strongly recommend that you quickly review the entire tutorial before running the pipeline.**

## Table of Contents

- [Temporal Embeddings of Surface Spectra for Earth Representation and Analysis (TESSERA)](#temporal-embeddings-of-surface-spectra-for-earth-representation-and-analysis-tessera)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Initial Results](#initial-results)
    - [Land Classification](#land-classification)
    - [Canopy Height Prediction](#canopy-height-prediction)
    - [Representation Visualization](#representation-visualization)
  - [Hardware Requirements](#hardware-requirements)
    - [1. Storage Requirements](#1-storage-requirements)
    - [2. Memory Requirements](#2-memory-requirements)
    - [3. CPU and GPU](#3-cpu-and-gpu)
    - [4. Operating System](#4-operating-system)
  - [Data Preprocessing](#data-preprocessing)
    - [Overview](#overview)
    - [Download Source Code](#download-source-code)
    - [Python Environment](#python-environment)
    - [Script Configuration](#script-configuration)
  - [Inference](#inference)
    - [Overview](#overview-1)
    - [Pytorch Preparation](#pytorch-preparation)
    - [Model Weight](#model-weight)
    - [Configure Bash Script](#configure-bash-script)
    - [Start Inference](#start-inference)
    - [Stitch Final Representation Map](#stitch-final-representation-map)
  - [Acknowledgments](#acknowledgments)

## Introduction

TESSERA (formerly known as BTFM) is a foundation model for Earth observation that processes Sentinel-1 and Sentinel-2 satellite data to generate representation (embedding) maps. It compresses a full year of Sentinel-1 and Sentinel-2 data and learns useful temporal-spectral features.

## Representation Visualization

Below are some visualization results of the TESSERA representation map (using the first three channels as RGB):

![repr_demo](images/repr_demo.png)

## Hardware Requirements

### 1. Storage Requirements

Running this pipeline requires substantial storage space. Although the pipeline cleans up some intermediate files after processing, the downloaded raw Sentinel-2 and Sentinel-1 files will still occupy considerable disk space. For example, processing a 100km×100km area from 2022 to output a TESSERA Representation map (10m resolution) requires at least 1TB of storage.

### 2. Memory Requirements

Thanks to Microsoft Planetary Computer, most of the geo-preprocessing has been done. Still, we recommend having at least 128GB of RAM.

### 3. CPU and GPU

The pipeline has no strict requirements for CPU and GPU, but more CPU cores and more powerful GPUs can significantly speed up inference. When processing a 110km×110km area from 2022, our tests using a 128-core CPU and a single NVIDIA A30 GPU for inference (CPU and GPU each handling 50% of the inference) took approximately 10 hours to complete.

### 4. Operating System

For the data preprocessing pipeline, we support almost all Linux and macOS systems. For Windows, we recommend using WSL.

For the model inference part, we have only tested it on Linux and Windows WSL, and they are working.

## Data Preprocessing

### Overview

In this step, we stack a full year of Sentinel-1 and Sentinel-2 data along the time dimension to generate a composite. For Sentinel-2, the composite shape is (T,H,W,B), where T is the number of valid observations in that year, and B is the number of bands (we selected 10 bands). For Sentinel-1, we extracted both ascending and descending orbit data. Taking the ascending orbit as an example, the composite shape is (T',H,W,B'), where T' is the number of valid ascending observations in that year, and B' is 2 because we only obtain VV and VH bands.

We source Sentinel-1 and Sentinel-2 data from Microsoft's Planetary Computer, which has been preprocessed to a large extent and can be used directly. This saves a lot of data preprocessing trouble.

Currently, our pipeline only accepts TIFF format input. The resolution of the tiff file can vary, but up to 10m granularity as this is the highest resolution for Sentinel-2 imagery. For valid ROI areas within the TIFF, the value is 1; otherwise, it's 0. If you only have a shapefile, that's fine too - we provide a `convert_shp_to_tiff.py` script.

### Download Source Code

First, create an empty working directory:

```bash
mkdir tessera_project
cd tessera_project
git clone https://github.com/FrankFeng-23/btfm_project.git
```

For easier pipeline operation, we recommend placing the data output directory at the same level as `tessera_infer` and `tessera_preprocessing`:

```
tessera_project
 ┣ tessera_infer
 ┣ tessera_preprocessing
 ┣ my_data
   ┣ roi.shp (your shapefile)
   ┗ roi.tiff (we recommend generating this using convert_shp_to_tiff.py)
```

The `roi.tiff` can be generated using `convert_shp_to_tiff.py` located in `tessera_preprocessing/convert_shp_to_tiff.py`. To use it, simply specify the path to your shapefile in the main function, and it will output a TIFF with the same name in the same directory.

### Python Environment

We need some geographic processing packages (fortunately, we won't be using GDAL, as configuring the environment is a nightmare) and some machine learning packages (PyTorch, but you'll need to install this yourself since the hardware on each computer is different). We've put some common packages in `requirements.txt`, which you can install as follows:

```bash
pip install -r requirements.txt
```

### Script Configuration

We use Microsoft's Planetary Computer, which eliminates much of the hassle of data preprocessing, especially for Sentinel-1. The script configuration is very simple. First, navigate to the `tessera_preprocessing` folder:

```bash
cd tessera_preprocessing
```

Then modify the following:

```bash
# === Basic Configuration ===
INPUT_TIFF="/absolute/path/to/your/data_dir/roi.tiff"
OUT_DIR="/absolute/path/to/your/data_dir"

export TEMP_DIR="/absolute/path/to/your/temp_dir"     # Temporary file directory

mkdir -p "$OUT_DIR"

# Python environment path
PYTHON_ENV="/absolute/path/to/your/python_env/bin/python"

# === Sentinel-1 & Sentinel-2 Processing Configuration ===
YEAR=2022 # Range [2017-2024]
RESOLUTION=10.0  # Resolution of the input TIFF, also the output resolution (meters)
```

Note that the `RESOLUTION` needs to match the resolution of your input TIFF; otherwise, there may be misalignments in geographic coverage. Below the above configuration, there are some additional configurations that you can modify according to your computer's performance.

First, give permission to `s1_s2_downloader.sh`:

```bash
chmod +x s1_s2_downloader.sh
```

Then, we can run:

```bash
bash s1_s2_downloader.sh
```

Due to network conditions, processing some tiles may time out. Our script includes sophisticated timeout management to avoid these issues. However, sometimes some tiles may still fail. Running the above command again usually resolves this.

If all Sentinel-1 and Sentinel-2 data are generated correctly, they can be stacked along the time dimension. For this step, we use two Rust-generated executables, making it very fast. You can open `s1_s2_stacker.sh` and edit the following:

```bash
# === Basic Configuration ===
BASE_DIR="/absolute/path/to/your/data_dir"
OUT_DIR="${BASE_DIR}/data_processed"
DOWNSAMPLE_RATE=1
```

Normally, we don't modify `DOWNSAMPLE_RATE`, which keeps it from performing any downsampling during stacking. The `BASE_DIR` in the above snippet is the same as the `OUT_DIR` you modified in `s1_s2_downloader.sh`.

Similarly, give permission to `s1_s2_stacker.sh`:

```bash
chmod +x s1_s2_stacker.sh
```

Then you can execute the stacking:

```bash
bash s1_s2_stacker.sh
```

After success, you will get some `.npy` files in `/absolute/path/to/your/data_dir/data_processed`. Usually, these `.npy` files are quite large, so we will patchify them into smaller, more manageable units.

Execute:

```bash
python dpixel_retiler.py \
    --tiff_path /absolute/path/to/your/data_dir/roi.tif \
    --d_pixel_dir /absolute/path/to/your/data_dir/data_processed \
    --patch_size 500 \
    --out_dir /absolute/path/to/your/data_dir/retiled_d_pixel \
    --num_workers 16 \
    --overwrite \
    --block_size 2000
```

You can change the above `patch_size` and `block_size` yourself. The above configuration is a recommended configuration for a TIFF with a shape of (5000,5000) and a 10m resolution.

If the above code runs smoothly, you can get some subfolders in `my_data/retiled_d_pixel`.

## Inference

### Overview

Once the data preprocessing is complete, we can start inference. Before proceeding, please check if there are subfolders in the `my_data/retiled_d_pixel` folder like:
```
retiled_d_pixel
 ┣ 0_3500_500_4000
 ┣ 0_4000_500_4500
 ┣ 0_4500_500_5000
 ┣ 0_5000_500_5500
 ┣ 0_5500_500_6000
 ┣ 0_6000_500_6500
```

Each subfolder should contain the following files:
```
0_3500_500_4000
 ┣ bands.npy
 ┣ doys.npy
 ┣ masks.npy
 ┣ roi.tiff
 ┣ sar_ascending.npy
 ┣ sar_ascending_doy.npy
 ┣ sar_descending.npy
 ┗ sar_descending_doy.npy
```

If these files exist, you can start inference. Otherwise, check if the first step completed successfully.

Inference requires PyTorch. Since each system may have slightly different CUDA versions, we can't provide a Docker-encapsulated Python environment like we did for data preprocessing. Fortunately, the Python environment for inference is much simpler to configure than for data preprocessing, as it doesn't use geographic processing packages like GDAL or SNAP.

### Pytorch Preparation

If you haven't installed Pytorch, you can refer to the steps below. Otherwise, you can ignore this section.

First, check your system's CUDA version:

```bash
nvidia-smi
```

Then visit https://pytorch.org/ and select the appropriate version to install based on your CUDA version, for example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Model Weight

Next, download the model weights from [Google Drive](https://drive.google.com/drive/folders/18RPptbUkCIgUfw1aMdMeOrFML_ZVMszn?usp=sharing) (only Cambridge members have access) and place the `.pt` file in the `tessera_infer/checkpoints` directory:

```
checkpoints
 ┗ best_model_fsdp_20250427_084307.pt
```

### Configure Bash Script

To simplify inference configuration, we provide `tessera_infer/infer_all_tiles.sh`. You only need to edit a few parameters:

a. Base data directory:
```bash
BASE_DATA_DIR="your_data_directory"
```
This is your data storage folder, the same as `BASE_DATA_DIR` in the previous bash, e.g., `/maps/usr/tessera_project/my_data`

b. Python environment:
```bash
export PYTHON_ENV="your_python_path"
```
Write the absolute path to your Python environment here, e.g., `/home/user/anaconda3/envs/tessera_env/bin/python`

c. CPU/GPU split:
```bash
CPU_GPU_SPLIT="1:1"  # Format: CPU:GPU ratio
```
The script supports simultaneous inference using both CPU and GPU. This ratio specifies the proportion of `retiled_patches` each device will handle. Default is 1:1 (even split). For GPU-only inference, set to 0:1.

d. CPU Related Settings

```bash
MAX_CONCURRENT_PROCESSES_CPU=20
```
Maximum number of CPU processes for tile inference. For example, if set to 20, it will process 20 tiles simultaneously.

```bash
AVAILABLE_CORES=$((TOTAL_CPU_CORES / 2)) # Use 50% of the cores
```
Number of CPU cores to use. Please modify this value if necessary to avoid consuming too many CPU resources!

e. GPU Related Settings:
```bash
MAX_CONCURRENT_PROCESSES_GPU=1
```
Maximum number of GPU processes for inference. If the system has only 1 GPU, set this to 1.

```bash
GPU_BATCH_SIZE=1024  # Larger for GPU, if this takes too much memory, reduce it
```
Number of samples to process at once during PyTorch inference. If this value consumes too much GPU memory or causes an OOM error on the GPU, please reduce it accordingly.

f. Other Settings
There are other parameters available for configuration. Please adjust them as needed.

### Start Inference

Once everything is ready, navigate to the `tessera_infer` folder:

```bash
cd tessera_infer
```

Then give permission to `infer_all_tiles.sh`:

```bash
chmod +x infer_all_tiles.sh
```

Then run it:

```bash
bash infer_all_tiles.sh
```

If successful, you should see logs like:

```
(base) zf281@daintree:/scratch/zf281/tessera_project/tessera_infer$ bash infer_all_tiles.sh
[INFO] Total CPU cores: 256, Using: 192
[INFO] CPU:GPU split ratio = 1:1 (total: 2)

==== SETUP DIRECTORIES ====
[SUCCESS] Created necessary directories

==== SCANNING TILES ====
[INFO] Tile directory: /scratch/zf281/jovana/retiled_d_pixel
[INFO] Output directory: /scratch/zf281/jovana/representation_retiled
[SUCCESS] Found 226 tiles total
[INFO] Sample tiles:
  - 0_3500_500_4000
  - 0_4000_500_4500
  - 0_4500_500_5000
  - ...
```

At the same time, a `logs` folder will be generated in the `tessera_infer` folder with more detailed logging for each CPU and GPU process.

### Stitch Final Representation Map

Inference usually takes a long time, depending on your ROI size and hardware performance. Once completed, you can find many `.npy` files in `my_data/representation_retiled`:

```
representation_retiled
 ┣ 0_3500_500_4000.npy
 ┣ 0_4000_500_4500.npy
 ┣ 0_4500_500_5000.npy
 ┣ 0_5000_500_5500.npy
 ┣ 0_5500_500_6000.npy
 ┣ 0_6000_500_6500.npy
 ┣ 0_6500_500_7000.npy
 ┣ 0_7000_500_7500.npy
 ┣ 1000_0_1500_500.npy
 ┣ 1000_1000_1500_1500.npy
 ┣ 1000_1500_1500_2000.npy
 ┣ 1000_2000_1500_2500.npy
```

The final step is to stitch them together using `tessera_infer/stitch_tiled_representation.py`:

```bash
python stitch_tiled_representation.py \
--d_pixel_retiled_path /path/to/d_pixel_retiled \
--representation_retiled_path /path/to/representation_retiled \
--downstream_tiff /path/to/downstream.tiff \
--out_dir /path/to/output_directory
```

For example:

```bash
python stitch_tiled_representation.py \
--d_pixel_retiled_path /maps/usr/tessera_project/my_data/d_pixel_retiled \
--representation_retiled_path /maps/usr/tessera_project/my_data/representation_retiled \
--downstream_tiff /maps/usr/tessera_project/my_data/downstream.tiff \
--out_dir /maps/usr/tessera_project/my_data
```

Finally, you'll get a stitched representation map in the `my_data` directory with the shape (H,W,128), where H and W match your initial `roi.tiff`. The representation map is a NumPy array. If you want to convert it to TIFF for viewing in software like QGIS, you can use the `tessera_infer/convert_npy2tiff.py` script. Just modify the main function with:

```python
npy_path = "/maps/usr/tessera_project/my_data/stitched_representation.npy"  # Change to the actual npy file path
ref_tiff_path = "/maps/usr/tessera_project/my_data/roi.tiff"  # Change to the actual reference tiff file path
out_dir = "/maps/usr/tessera_project/my_data/"  # Change to the actual output directory
```

## Acknowledgments
We would like to express our gratitude to [DAWN](https://www.hpc.cam.ac.uk/d-w-n), the fastest artificial intelligence supercomputer at Cambridge, and AMD for their generous support in this project. This work would not have been possible without their computational resources and technical assistance.
