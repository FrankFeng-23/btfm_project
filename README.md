# Temporal Embeddings of Surface Spectra for Earth Representation and Analysis (TESSERA) [CVPR2026]
<div align="center">
    <a href="#readme-top">
        <img src="images/banner.png" alt="Banner">
    </a>
    <br />
    <p align="center">
        <a href="https://geotessera.org/">Project Website 🌐</a> &nbsp;&nbsp;&nbsp;&nbsp;
        <a href="https://github.com/FrankFeng-23/btfm_project/issues">Report Bugs 🛠️</a> &nbsp;&nbsp;&nbsp;&nbsp;
        <a href="https://github.com/FrankFeng-23/btfm_project/issues">Request Features 💡</a>
    </p>
</div>

<!--  ![Version](https://img.shields.io/badge/version-alpha-red) -->
![PyPI version](https://img.shields.io/pypi/v/geotessera?label=PyPI%20version&color=blue)
![License](https://img.shields.io/badge/License-MIT-blue.svg)


# Table of Contents

  - Learning about TESSERA
      - [Introduction](#introduction)
      - [Papers](#Papers)
      - [Podcast](https://www.satellite-image-deep-learning.com/p/tessera-a-temporal-foundation-model)
      - [Presentations](#presentations)
      - [License](#License)
  - Using TESSERA
      - [Acceptable Use Policy](#AUP)
      - [Accessing Precomputed Embeddings](#global-embeddings-access)
      - [Creating Your Own Embeddings](#creating-your-own-embeddings)
      - [Downstream Tasks](#downstream-tasks)
      - [TESSERA Users Group](#tessera-users-group)
  - Additional information
      - [Team](#team)
      - [Contact](#contact)
      - [Citation](#citation)
      - [Acknowledgments](#acknowledgments)
      - [Star History](#star-history)

# Learning about TESSERA
## Introduction

Satellite remote sensing enables a wide range of downstream applications, including habitat mapping, carbon accounting, and strategies for conservation and sustainable land use. However, satellite time series are voluminous and often cloud-corrupted, making them challenging to use: the scientific community's ability to extract actionable insights is often constrained by the scarcity of labelled training datasets and the computational burden of processing temporal data. The key insight behind our work, due to [Dr. Clement Atzberger](https://www.linkedin.com/in/clement-atzberger-8abb8065/) is that forcing auto-encoder embeddings derived from two cloud-free random samples of satellite time series to align using [Barlow Twins](https://proceedings.mlr.press/v139/zbontar21a/zbontar21a.pdf) results in an embedding that represents the entire time series, including the missing observations.

This idea is the key behind TESSERA, an open foundation model that preserves per-pixel spectral-temporal signals in 128-dimensional latent representations at 10-meter resolution globally. It uses self-supervised learning to summarise petabytes of Earth observation data. We compare our work with state-of-the-art task-specific models and other foundation models in five diverse downstream tasks and find that TESSERA closely matches or outperforms these baselines. By preserving temporal phenological signals that are typically lost in conventional approaches, TESSERA enables new insights into ecosystem dynamics, agricultural food systems, and environmental change detection. Moreover, our open-source implementation supports reproducibility and extensibility, while the privacy-preserving design allows researchers to maintain data sovereignty.

To our knowledge, TESSERA is unprecedented in its ease of use, scale, and accuracy: no other foundation model provides analysis-ready outputs, is open, and provides global, annual coverage at 10m resolution using only spectral-temporal features at pixel level.

Here are some visualization results of the TESSERA representation map (using the first three channels as RGB):

![repr_demo](images/repr_demo.png)

## Papers
Here are publications and preprints related to TESSERA, listed chronologically:
* Lisaius, M. C., Blake, A., Keshav, S., & Atzberger, C. (2024). Using Barlow Twins to Create Representations From Cloud-Corrupted Remote Sensing Time Series. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 17, 13162–13168. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing. https://doi.org/10.1109/JSTARS.2024.3426044

* Z. Feng, C. Atzberger, S. Jaffer, J. Knezevic, S. Sormunen, R. Young, M.C. Lisaius, M. Immitzer, T. Jackson, J. Ball, D.A. Coomes, A. Madhavapeddy, A. Blake, S. Keshav (2025), [TESSERA: Temporal Embeddings of Surface Spectra for Earth Representation and Analysis](https://arxiv.org/abs/2506.20380), To Appear, CVPR 2026. ArXiv reprint. https://arxiv.org/abs/2506.20380

* Lisaius, M. C., Blake, A., Atzberger, C., & Keshav, S. (2026). Towards improved crop type classification: A compact embedding approach suitable for small fields. Accepted in Proceedings of the ISPRS Conference 2026. International Society for Photogrammetry and Remote Sensing.

* Z. Feng, C. Atzberger, S. Jaffer, J. Knezevic, S. Sormunen, R. Young, M.C. Lisaius, M. Immitzer, T. Jackson, J. Ball, D.A. Coomes, A. Madhavapeddy, A. Blake, S. Keshav, (2026) [Applications of the TESSERA Geospatial Foundation Model to Diverse Environmental Mapping Tasks](http://ssrn.com/abstract=6142416), SSRN preprint. http://ssrn.com/abstract=6142416
  
* Young, R., & Keshav, S. (2026). Interpolation of GEDI Biomass Estimates with Calibrated Uncertainty Quantification, arXiv preprint. https://doi.org/10.48550/ArXiv.2601.16834
  
* Lisaius, M. C., Keshav, S., Blake, A., & Atzberger, C. (2026). Embedding-based Crop Type Classification in the Groundnut Basin of Senegal (arXiv:2601.16900). ArXiv preprint. https://doi.org/10.48550/arXiv.2601.16900

* Ball, J.G.C, Wicklein J.A. , Feng, Z.,  Knezevic, J.,  Jaffer, S., Atzberger, C.,  Dalponte, M., and Coomes, D. Geospatial foundation models enable data-efficient tree species mapping in temperate montane forests, BioArxiv, https://doi.org/10.64898/2026.02.23.707022

## Presentations

* [TESSERA overview in AI for Good seminar](https://www.youtube.com/live/9yrpwFrwbGY), Frank Feng, Jan 22, 2026
* [TESSERA: Precomputed FAIR Global Pixel Embeddings for Earth Representation and Analysis](https://www.grss-ieee.org/event/tessera-precomputed-fair-global-pixel-embeddings-for-earth-representation-and-analysis/) IEEE GRSS Talk, Frank Feng, 12 December, 2025
* [2-slide summary (PPTX)](https://www.dropbox.com/scl/fi/zjo4trov0z2qnmdeitng0/CRI-2slide.pptx?rlkey=5kkojiknt6hdn2zplzlotqnbt&st=ezafh67n&dl=0) for CRI Flash Talks, S. Keshav, October 7, 2025
* Foundation model overview (PPTX) for Ecology Groups meeting, University of Cambridge, DAB, James Ball, October 6, 2025
* [TESSERA overview presentation with a focus on ecological applications](https://www.dropbox.com/scl/fi/8xvanw3kk586lp1ld31kd/maryland_talk_slides.pdf?rlkey=osyhtk1kc2pcj81iel0u32lub&st=6kedpwv6&dl=0) (PDF) University of Maryland, Frank Feng, October 1, 2025
* [TESSERA overview presentation](https://www.dropbox.com/scl/fi/0rsq4wkao3c7fgwljd8ec/JCU-tesserav2.pptx?rlkey=ccutcxgwi068c09n20t1yi549&st=13if23b3&dl=0) (PPTX) James Cook University, S. Keshav, September 29, 2025
* [TESSERA overview presentation](https://www.dropbox.com/scl/fi/1p7nabvlvie8fzyomkx7w/dab_talk_slides.pdf?rlkey=ym3d44o80mbrdkasyzct9kzi5&st=ozvwczs7&dl=0) University of Cambridge, DAB, Frank Feng, May 20, 2025
* [Self-supervised learning for earth observation](https://www.dropbox.com/scl/fi/zjo4trov0z2qnmdeitng0/CRI-2slide.pptx?rlkey=5kkojiknt6hdn2zplzlotqnbt&st=ezafh67n&dl=0) (PPTX) S. Keshav, Exeter, April 2025

## License

TESSERA software is released under the standard MIT license. Embeddings and model weights are released under the [CC0](https://creativecommons.org/publicdomain/zero/1.0/) license: essentially, 
they can be freely used for both commercial and non-commercial purposes. Although we do not legally require attribution,
we do request it.

# Using TESSERA

<a id="global-embeddings-access"></a>

## Accessing Embeddings using GeoTessera (recommended)

We have generated embeddings for the whole globe at 10m resolution for 2024.
These can be downloaded and used for downstream applications, saving significant computational time and resources, using 
the [GeoTessera](https://github.com/ucam-eo/geotessera) library. 
We will progressively extending coverage backwards year by year until 2017. The current coverage map is below:

<img src="https://github.com/ucam-eo/tessera-coverage-map/blob/main/map.png"> 

## TESSERA Users Group

Interested users are invited to join our [Zulip](https://eeg.zulipchat.com/login/) discussion groups.


# Creating Your Own Embeddings

If you would like to use our software to create your own embeddings, please follow the instructions below. Note that this is a comptuationally challenging task and you will need access to significant computational and storage resources. 

## Hardware Requirements

### 1. Storage Requirements

Running this pipeline requires substantial storage space. Although the pipeline cleans up some intermediate files after processing, the downloaded raw Sentinel-2 and Sentinel-1 files will still occupy considerable disk space. For example, processing a 100km×100km area from 2022 to output a TESSERA Representation map (10m resolution) requires at least 1TB of storage.

### 2. Memory Requirements

We use preprocessed data, initially from Microsoft Planetary Computer. However, the next generation of embeddings will use OPERA from ASF DAAC. In either case, most of the geo-preprocessing has been done. Still, we recommend having at least 128GB of RAM.

### 3. CPU and GPU

The pipeline has no strict requirements for CPU and GPU, but more CPU cores and more powerful GPUs can significantly speed up inference. When processing a 110km×110km area from 2022, our tests using a 128-core CPU and a single NVIDIA A30 GPU for inference (CPU and GPU each handling 50% of the inference) took approximately 10 hours to complete.

### 4. Operating System

For the data preprocessing pipeline, we support almost all Linux systems. For Windows, we recommend using WSL. We do not support MacOS at this point.

For the model inference part, we have only tested it on Linux and Windows WSL, and they are working.

## Data Preprocessing

### Overview
_**We strongly recommend that you quickly review the entire tutorial before running the pipeline.**_

In this step, we stack a full year of Sentinel-1 and Sentinel-2 data along the time dimension to generate a composite. For Sentinel-2, the composite shape is (T,H,W,B), where T is the number of valid observations in that year, and B is the number of bands (we selected 10 bands). For Sentinel-1, we extracted both ascending and descending orbit data. Taking the ascending orbit as an example, the composite shape is (T',H,W,B'), where T' is the number of valid ascending observations in that year, and B' is 2 because we only obtain VV and VH bands.

We initially sourced Sentinel-1 and Sentinel-2 data from Microsoft's Planetary Computer:
- Sentinel-1 data source: https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc
- Sentinel-2 data source: https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a

The new generation of embeddings will use OPERA from ASF DAAC: 
- Sentinel-1 data source:  https://registry.opendata.aws/nasa-operal2rtc-s1v1/ 
- Sentinel-2 data source: https://registry.opendata.aws/sentinel-2-l2a-cogs/

Currently, our pipeline only accepts TIFF format input. The resolution of the input ROI TIFF can vary (e.g., 30m), but the pipeline will **always generate Sentinel-1 and Sentinel-2 outputs at the configured `RESOLUTION`** (default 10m) while keeping the **ROI extent/bounds identical**. For valid ROI areas within the TIFF, the value is 1; otherwise, it's 0. If you only have a shapefile, that's fine too - we provide a `convert_shp_to_tiff.py` script.

### Download Source Code

First, create an empty working directory:

```bash
mkdir tessera_project
cd tessera_project
git clone https://github.com/ucam-eo/tessera.git
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

⚠️Notice: _If your ROI is relatively large, for example 100 km × 100 km, we strongly recommend pre-splitting the TIFF into smaller sections no larger than 20 km × 20 km. Then process each small TIFF file sequentially in the pipeline. An excessively large ROI may cause issues with backend tile providers_

### Python Environment

We need some geographic processing packages (fortunately, we won't be using GDAL, as configuring the environment is a nightmare) and some machine learning packages (PyTorch, but you'll need to install this yourself since the hardware on each computer is different). We've put some common packages in `requirements.txt`, which you can install as follows:

```bash
pip install -r requirements.txt
```
Note: If you are in a managed environment, you may need to install a venv first, using 
```bash
python3 -m venv venv
source venv/bin/activate
```

### Script Configuration

First, navigate to the `tessera_preprocessing` folder:

```bash
cd tessera_preprocessing
```

Then edit the file s1_s2_downloader.sh to point to the ROI TIFF file, the output and temporary directories, and the data source:

```bash
# === Basic Configuration ===
INPUT_TIFF="/absolute/path/to/your/data_dir/roi.tiff"
OUT_DIR="/absolute/path/to/your/data_dir"

export TEMP_DIR="/absolute/path/to/your/temp_dir"     # Temporary file directory

mkdir -p "$OUT_DIR"

# Python environment path
PYTHON_ENV="/absolute/path/to/your/python_env/bin/python"

# === Sentinel-1 & Sentinel-2 Processing Configuration ===
YEAR=2022 # Range [2017-2025]
RESOLUTION=10.0  # Output resolution (meters). ROI TIFF can be any resolution; extent is preserved.

# === Data Source Configuration ===
# mpc: Microsoft Planetary Computer (sentinel-1-rtc, sentinel-2-l2a)
# aws: AWS Open Data backends (S1=OPERA RTC-S1 via ASF/CMR + ASF Earthdata Cloud COGs, S2=Earth-search Sentinel-2 L2A COGs)
DATA_SOURCE="mpc"   # choices: mpc/aws
```

Note: `RESOLUTION` controls output pixel size. The pipeline keeps the ROI bounds fixed and resamples the ROI mask into the output grid.

### AWS Credentials (only needed when `DATA_SOURCE="aws"`)
Sentinel-2 on Earth-search is public and **does not require credentials**.

Sentinel-1 OPERA RTC-S1 is accessed via ASF Earthdata Cloud (COG over HTTPS). You need an Earthdata Login token:
- **Create an Earthdata account**: via [NASA Earthdata Login](https://urs.earthdata.nasa.gov/home).
- **Approve Application**: After registering your account, you can go to the Applications tab and add Alaska Satellite Facility Data Access to the list of approved applications.
- **Obtain an EDL Bearer token / JWT** by clicking **Generate Token** and store it locally (do not commit it).

Recommended (simple + explicit):

```bash
nano ~/.edl_bearer_token
# paste token, save+exit (Ctrl-O Enter, then Ctrl-X)
chmod 600 ~/.edl_bearer_token
```

The AWS S1 downloader will use this token to read COGs from ASF Earthdata Cloud.

If you want to retrieve temporary S3 credentials (advanced; usually not required for this pipeline), see ASF guidance:
- `https://cumulus.asf.alaska.edu/s3credentialsREADME`

Below the above configuration, there are some additional configurations that you can modify according to your computer's performance.

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

Next, download the model weights from [Google Drive](https://drive.google.com/drive/folders/18RPptbUkCIgUfw1aMdMeOrFML_ZVMszn?usp=sharing) and place the `.pt` file in the `tessera_infer/checkpoints` directory:

```
tessera_infer
 ┗ checkpoints
     ┗ best_model_fsdp_20250427_084307.pt
 ┗ configs
 ┗ src
```

_**Note that the checkpoint mentioned above is an early-stage model, which natively generates float32 embeddings. Therefore, this model is not the one used to generate the int8 embeddings in the geotessera library. We will soon deploy the specific checkpoint that was used to create the geotessera embeddings into the full pipeline.**_

### QAT Model Weight (Quantized Output)

For quantized inference, use the QAT checkpoint from [Google Drive](https://drive.google.com/file/d/1HJ92aS5ERXMLfSFYJ4m3OKycJJdC1QvO/view?usp=sharing) and place it in `tessera_infer_QAT/checkpoints`:

```
tessera_infer_QAT
 ┣ checkpoints
 ┃   ┗ best_model_fsdp_20250608_220648_QAT.pt
 ┣ configs
 ┗ src
```

The QAT pipeline outputs quantized embeddings as **int8 + scales**:
- `tile_name.npy`: int8 embedding tensor, shape `(H, W, 128)`
- `tile_name_scales.npy`: float32 scale map, shape `(H, W)`

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

### QAT Inference (CPU + GPU, with AMX auto-fallback)

We also provide a QAT inference pipeline in `tessera_infer_QAT`:

```bash
cd tessera_infer_QAT
chmod +x infer_all_tiles.sh
bash infer_all_tiles.sh
```

Before running, edit these parameters in `tessera_infer_QAT/infer_all_tiles.sh`:

```bash
BASE_DATA_DIR="/absolute_path_to_your_data_dir"
export PYTHON_ENV="/absolute_path_to_your_python/bin/python"
CPU_GPU_SPLIT="1:1"  # CPU:GPU ratio, e.g. 1:0 or 0:1
CHECKPOINT_PATH="checkpoints/best_model_fsdp_20250608_220648_QAT.pt"
```

Notes:
- QAT now supports both CPU and GPU inference in one run (ratio-based split, same style as `tessera_infer`).
- On CPU, AMX is automatically detected and enabled when available.
- If AMX is not available, it automatically falls back to default CPU inference.

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

## Downstream tasks

If you want to reproduce the downstream tasks in the paper, you can visit https://github.com/ucam-eo/tessera-downstream-task. There are many examples provided there.

# Additional information

## Team

### Cambridge Faculty
* [S. Keshav](https://svr-sk818-web.cl.cam.ac.uk/keshav/wiki/index.php/Main_Page)
* [Anil Madhavapeddy](https://anil.recoil.org)
* [Sadiq Jaffer](https://toao.com)
* [David Coomes](https://www.plantsci.cam.ac.uk/directory/david-coomes)

### Postdoc
* James Ball
  
### PhD
* Madeleine Lisaius
* Zhengpeng (Frank) Feng
* Robin Young
* Jovana Knezevic

### Undergrad
* Zejia Yang (Part II student, working with Frank Feng on MAE pretraining of spatial feature extractors)

### Interns
* Kenzy Soror (U. Waterloo, working with Robin Young)
* Artyom Gabtraupov (U. Waterloo, working with Robin Young)
* Gabriel Mahler (U. Cambridge, working with Anil Madhavapeddy and Silviu Petrovan on [hedgehog habitats and tracking](https://anil.recoil.org/ideas/hedgehog-mapping))
* Leyu Pan (Imperial College, working with Frank Feng on text embeddings generated from OSM)

### Collaborators
* [Clement Atzberger](https://www.linkedin.com/in/clement-atzberger-8abb8065/?originalSubdomain=at), dClimate Labs
* [Andrew Blake](https://en.wikipedia.org/wiki/Andrew_Blake_(computer_scientist)), Mantle Labs

### Visitors
* Silja Sormunnen, Aalto University, Finland
* Isabel Mansley (U. Edinburgh, working with David Coomes and Anil Madhavapeddy on [habitat mapping in Scotland](https://anil.recoil.org/ideas/cairngorms-connect-habitats)

## Contact

Please direct your technical questions to Frank Feng (zf281@cam.ac.uk) or ask it on our [Zulip forum](https://eeg.zulipchat.com/login/). Non-technical questions can be sent to Prof. S. Keshav (sk818@cam.ac.uk).

## Citation

If you use TESSERA in your research, please cite the [arXiv paper](https://arxiv.org/abs/2506.20380):

```bibtex
@misc{feng2025tesseratemporalembeddingssurface,
      title={TESSERA: Temporal Embeddings of Surface Spectra for Earth Representation and Analysis}, 
      author={Zhengpeng Feng et al.},
      year={2025},
      eprint={2506.20380},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.20380}, 
}
```

## Acknowledgments
We would like to express our gratitude to UKRI and the [DAWN](https://www.hpc.cam.ac.uk/d-w-n) supercomputer team at Cambridge, for their generous support in this project. We also acknowledge support from [AMD](https://www.amd.com/en.html),  [Vultr](https://www.vultr.com/), the [Dirac High Performance Computing Facility](https://dirac.ac.uk), [Microsoft AI For Good Lab](https://www.microsoft.com/en-us/research/group/ai-for-good-research-lab/), Dr. Robert Sansom, [dClimate](https://www.dclimate.net/), and [Amazon Web Services (AWS)](https://aws.amazon.com/) under their AWS Open Data program (https://opendata.aws/). This work would not have been possible without their support, computational resources and technical assistance.  

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=ucam-eo/tessera&type=Date)](https://www.star-history.com/#ucam-eo/tessera&Date)

## AUP

### TESSERA Terms of Use and Ethical Guidelines

### License

TESSERA data and embeddings are made available under the **Creative Commons 0 International License [CC-0](https://creativecommons.org/public-domain/cc0/)**. 
This means you are free to:

* **Share** — copy and redistribute the material in any medium or format
* **Adapt** — remix, transform, and build upon the material for any purpose, even commercially

### Purpose and Intended Uses

TESSERA was developed to advance scientific research and support environmental monitoring, conservation, sustainable agriculture, and understanding of Earth systems. We designed this tool to enable:

* Scientific research and education
* Environmental monitoring and conservation
* Agricultural and food security analysis
* Climate change research and adaptation planning
* Sustainable land use and resource management
* Public interest applications that benefit society and the environment

### Ethical Guidelines

While the CC0 license permits broad use, we strongly encourage users to consider the ethical implications of their work. These ethical guidelines are advisory and do not impose legally enforceable restrictions. We request that users:

**Act Responsibly:**
* Consider privacy implications when analyzing specific locations
* Respect the rights and dignity of affected communities
* Be mindful of potential dual-use concerns

**Be Transparent:**
* Accurately represent the data's characteristics (annual resolution, 10m spatial resolution)
* Acknowledge limitations in your applications
* Do not misrepresent TESSERA's capabilities

**Support Positive Impact:**
* Consider how your work contributes to societal benefit
* Engage with affected communities when appropriate
* Share findings that advance public knowledge

### Data Characteristics

Users should understand that TESSERA provides:
* **Annual temporal resolution** — data represents yearly summaries, not real-time or high-frequency monitoring
* **10-meter spatial resolution** — suitable for landscape-scale analysis
* **Spectral-temporal embeddings** — compressed representations, not raw imagery

Please accurately represent these characteristics in your work.

### Community Standards

We encourage responsible use and welcome community feedback. If you have concerns about potential applications or suggestions for improving these guidelines, please contact us.

We reserve the right to update these guidelines based on community input and emerging considerations, though such updates do not retroactively affect the CC-0 license under which data is released.

### Contact

For questions or feedback: Email sk818@cam.ac.uk

---

*Last updated: February 25, 2026*

