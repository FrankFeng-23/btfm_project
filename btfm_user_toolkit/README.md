# BTFM User Toolkit

This toolkit provides a streamlined way to generate BTFM representation maps for your specific Regions of Interest (ROIs). You can provide your ROIs as Shapefiles, GeoJSON, existing GeoTIFFs, or Bounding Boxes. The toolkit handles (with some assumptions) the steps of data downloading, preprocessing, and model inference, outputting a Numpy array (`.npy`) representation for each input ROI.

## Introduction

This toolkit acts as a wrapper around the [BTFM Project](https://github.com/FrankFeng-23/btfm_project). It abstract the multi step data preparation and inference process, allowing users to focus on providing their ROIs and getting the final representation outputs.

## Core Workflow

For each input ROI you provide, the toolkit will:

1.  **Prepare a Standardized ROI TIFF:** Convert your input (Shapefile, GeoJSON, GeoTIFF, lat-lon Bounding Box) into a 10-meter resolution GeoTIFF. This TIFF will define the area for processing.
2.  **Data Preprocessing:** Download and process Sentinel-1 and Sentinel-2 data corresponding to your ROI TIFF.
3.  **Inference:** Run the BTFM model on the preprocessed data to generate tiled representation vectors.
4.  **Stitching:** Combine the tiled vectors into a single, contiguous representation map (`.npy` file) for your ROI.

## System Requirements

  - **Operating System:** Linux. Other OS are not guaranteed to work.
  - **Storage:** Substantial. Processing a 100km by 100km area for one year can require **>1TB** of disk space for raw data, intermediate files, and outputs. The toolkit will create a `working_data` directory for each ROI, which can grow very large.
  - **Memory (RAM):** At least **512GB RAM** is highly recommended for the data preprocessing step (especially Sentinel-1 processing with SNAP). Insufficient memory will not necessarily cause OOM errors but will significantly slow down processing.
  - **CPU:** More cores will speed up parts of the pipeline.
  - **GPU:** A powerful GPU with sufficient VRAM is needed for efficient inference. CUDA needs to be installed.
  - **Software:**
      - Git
      - Docker
      - Python
      - Conda (recommended for managing Python environments)
      - The internet

## Setup Instructions

### 1. Clone the Main BTFM Project

Clone the BTFM project repository.

```bash
git clone https://github.com/FrankFeng-23/btfm_project.git
# Let's assume you cloned it to /path/to/your/btfm_project
```

### 2. Install Docker & Pull Image

Docker is used for the data preprocessing environment.

  - **Install Docker:**
  - **Pull BTFM Docker Image:**
    ```bash
    docker pull frankfeng1223/snap-gdal-python:v3
    ```

### 3. Create Python Environment for Inference

This environment needs PyTorch and other packages for running the model inference.

```bash
# Using conda
conda create -n btfm_infer_env python=3.9 -y
conda activate btfm_infer_env
pip install numpy tqdm joblib psutil rasterio torch torchvision torchaudio
```

### 4. Create Python Environment for ROI Processing

This environment is for the toolkit's internal `roi_processor.py` script. It needs geospatial libraries. It can be the same as `btfm_infer_env` if you install the additional packages there, or a separate one.

```bash
# If using the same environment as inference:
# conda activate btfm_infer_env 
# pip install shapely pyproj fiona gdal  # gdal can be tricky with pip. If it doesn't work, try conda:
# conda install -c conda-forge gdal fiona shapely pyproj -y

# Or create a dedicated environment (recommended if gdal install is problematic):
conda create -n btfm_geo_env python=3.9 -y
conda activate btfm_geo_env
conda install gdal fiona shapely pyproj numpy rasterio -y 
```

### 5. Download Model Checkpoint

Download the pre-trained model weights (`.pt` file) from the link provided.
Place the `.pt` file (e.g., `best_model_fsdp_20250427_084307.pt`) into the following directory within your cloned `btfm_project`:
`btfm_project/btfm_infer/checkpoints/`

### 6. Configure the Toolkit

This toolkit uses a `config.sh` file for its settings.

1.  Navigate to the directory where you've placed this `btfm_user_toolkit`.
2.  Copy the template: `cp config.sh.template config.sh`
3.  Edit `config.sh` with your specific paths:
      - `BTFM_PROJECT_DIR`: Absolute path to your cloned `btfm_project` directory.
      - `INFERENCE_PYTHON_ENV`: Absolute path to the Python executable within your `btfm_infer_env` (e.g., `/home/user/anaconda3/envs/btfm_infer_env/bin/python`).
      - `ASF_USERNAME` and `ASF_PASSWORD`: Your API key.
      - `PROCESSING_YEAR`: The year for which to process Sentinel data (e.g. "2022").
      - Other inference parameters (optional, defaults are usually fine).

## Input ROI Formats

The toolkit can process ROIs provided in the following formats. If you provide a directory as input, the script will attempt to process all compatible files within that directory.

  - **Shapefile:** Files with a `.shp` extension. This typically comes with a `.shx`, `.prj`, and `.dbf` file too.
  - **GeoJSON:** Files with `.geojson` or `.json` extensions.
  - **GeoTIFF:** Files with `.tif` or `.tiff` extensions.
  - **Bounding Box:** A text file (e.g., `my_area_bbox.txt`) with a `.txt` extension. The file should contain a single line with four comma-separated numbers representing WGS84 decimal degree coordinates:
    `min_longitude,min_latitude,max_longitude,max_latitude`
    Example: `-74.0,40.7,-73.9,40.8`

MapShaper[https://mapshaper.org/] or Geojson.io[https://geojson.io] could be helpful here.

## Running the Pipeline

Once setup is complete, you can run the main script:

```bash
bash run_pipeline_for_inputs.sh <input_dir_or_file> <output_base_dir>
```

**Arguments:**

  - `<input_dir_or_file>`:
      - Path to a directory containing your ROI files (e.g., `my_rois/`).
      - OR, path to a single ROI file (e.g., `my_rois/area1.shp`).
  - `<output_base_dir>`: A path to a directory where all outputs and intermediate data will be stored.
  
**Example:**

```bash
# To process all compatible files in a directory
bash run_pipeline_for_inputs.sh /data/my_project_rois /data/my_project_outputs

# To process a single GeoJSON file
bash run_pipeline_for_inputs.sh /data/my_project_rois/area_A.geojson /data/my_project_outputs
```

The script will log its progress to the console and to files within `<toolkit_directory>/toolkit_logs/`. A main log aggregates overall progress, and sub-directories within `toolkit_logs/` will contain detailed logs for each input ROI processed.

## Output Structure

All outputs will be organized under the `<output_base_dir>` you specify:

  - `<output_base_dir>/representations/`:
      - Contains the final representation maps, one `.npy` file per input ROI.
      - Example: `area_A_representation.npy`
  - `<output_base_dir>/roi_tiffs_generated/`:
      - Contains the 10m resolution standardized ROI GeoTIFFs that were generated by the toolkit from your inputs and used by the pipeline. Useful for reference and verification.
      - Example: `area_A_roi_10m.tiff`
  - `<output_base_dir>/working_data/`:
      - This directory stores all intermediate data generated by the main BTFM pipeline for each ROI (raw Sentinel downloads, processed S1/S2 data, tiled representations before stitching, etc.).
      - It is organized into subdirectories named after your input ROIs (e.g., `working_data/area_A/`).
      - **This directory can become very large.** You may want to clean it up manually after confirming your representations are correct.
  - `<toolkit_directory>/toolkit_logs/`: (Located where `run_pipeline_for_inputs.sh` is)
      - Contains log files for the toolkit's operations.

## Using the Python Library to Load Representations

This toolkit provides a simple Python module to load the generated `.npy` representation maps and their associated georeferencing information.

See the `load_example.py` file for a runnable version.

## Time and Storage Estimates

  - **Time:** Processing a full 100km x 100km tile for a full year of data can take quite a while, depending heavily on your CPU, GPU, and I/O performance. Smaller time windows or areas will be faster.
  - **Storage:** For the same 100km x 100km area and one year, expect **at least 1TB** of storage to be used under `<output_base_dir>/working_data/<your_roi_name>/`. The final representation `.npy` file will be much smaller (e.g., for a 10k x 10k pixel ROI, it would be `10000 * 10000 * 128 features * 4 bytes/float` ~ 50GB, but this is typically for the full ROI extent, and actual data is only for valid pixels).

## Troubleshooting

  - **Check Logs:** The primary source of information is the log files.
      - The main log is `<toolkit_directory>/toolkit_logs/pipeline_YYYYMMDD_HHMMSS.log`.
      - For each input ROI (e.g., `area_A`), detailed logs for each step are in `<toolkit_directory>/toolkit_logs/area_A/`:
          - `1_roi_preparation.log`
          - `2_preprocessing.log` (Output from `main_pipeline.sh` - often very verbose)
          - `3_inference.log` (Output from `infer_all_tiles.sh` - check for PyTorch/CUDA errors)
          - `4_stitching.log`
  - **Configuration:** Check all paths and credentials in `config.sh`. Ensure paths are absolute.
  - **Permissions:** Ensure the script has execute permissions (`chmod +x run_pipeline_for_inputs.sh`). Ensure you have write permissions for the `<output_base_dir>` and its subdirectories.
  - **Docker:** Verify Docker is running and the image `frankfeng1223/snap-gdal-python:v3` is pulled.
  - **Python Environments:** Make sure the Python environments specified in `config.sh` (for inference) and used by the toolkit (for ROI processing) are correctly set up with all dependencies.
  - **Resource Limits:** Monitor disk space, RAM, and GPU memory during execution, especially for large ROIs.
