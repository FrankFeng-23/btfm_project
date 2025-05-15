import argparse
import math
import json
import os
import rasterio

# --- Baseline Constants ---
DEFAULT_BASELINES = {
    "S2_PROC_PER_TILE_HOURS": 2,  # Time to download & process S2 for 1 MGRS tile for 12 months
    "S1_DOWNLOAD_PER_TILE_HOURS": 2, # Time for s1_downloader.py for 1 MGRS tile for 12 months
    "AVG_S1_SCENES_PER_TILE_MONTH": 8, # Combined Asc/Desc, VV/VH scenes for a typical tile per month
    "S1_PROC_PER_SCENE_MINUTES": 5,  # Time for s1_processor.py (SNAP) per S1 scene zip
    "S1_STACK_PER_TILE_MINUTES": 5, # Time for s1_stack.py per MGRS tile
    "INFERENCE_PER_PATCH_SECONDS": 360, # For a 500x500 patch on target GPU/CPU setup
    "STITCHING_BASE_MINUTES": 5,
    "STITCHING_PER_PATCH_SECONDS": 0.5,
    "ROI_PREP_MINUTES": 2.5, # Base time for roi_processor.py
    "MGRS_DISCOVERY_MINUTES": 2.5, # Time for s2_discover_tiles.py
    # These are not processing times but parameters needed for calculation
    "PATCH_SIZE_PIXELS": 500,
    "PIXEL_RESOLUTION_M": 10,
    "MGRS_TILE_AREA_KM2_APPROX": 100 * 100 # Approx area of one MGRS tile
}

def load_baselines(config_dir_path):
    """Loads baselines from a JSON file if it exists, otherwise uses defaults."""
    baseline_file = os.path.join(config_dir_path, "time_estimation_baselines.json")
    if os.path.exists(baseline_file):
        try:
            with open(baseline_file, 'r') as f:
                loaded_baselines = json.load(f)
            # Merge with defaults, loaded values override
            baselines = {**DEFAULT_BASELINES, **loaded_baselines}
            print(f"INFO: Loaded time estimation baselines from {baseline_file}")
            return baselines
        except Exception as e:
            print(f"WARNING: Could not load or parse {baseline_file}: {e}. Using default baselines.")
            return DEFAULT_BASELINES
    else:
        print(f"INFO: Baseline file {baseline_file} not found. Using default baselines.")
        print(f"INFO: You can create this file to customize time estimates for your hardware.")
        return DEFAULT_BASELINES

def get_tiff_dimensions(tiff_path):
    if not tiff_path or not os.path.exists(tiff_path):
        print(f"WARNING: ROI TIFF path '{tiff_path}' not provided or does not exist for dimension extraction.")
        return None, None
    try:
        with rasterio.open(tiff_path) as src:
            return src.width, src.height
    except Exception as e:
        print(f"WARNING: Could not read dimensions from TIFF '{tiff_path}': {e}")
        return None, None
        
def calculate_time_estimate(args, baselines):
    total_min_time_h = 0
    total_max_time_h = 0
    stage_details = []

    # --- Stage 0: ROI Prep & MGRS Discovery ---
    roi_prep_h = baselines["ROI_PREP_MINUTES"] / 60.0
    mgrs_discovery_h = baselines["MGRS_DISCOVERY_MINUTES"] / 60.0
    
    total_min_time_h += roi_prep_h * 0.8 + mgrs_discovery_h * 0.8
    total_max_time_h += roi_prep_h * 1.2 + mgrs_discovery_h * 1.2
    stage_details.append(f"  - ROI Prep & MGRS Discovery: ~{ (roi_prep_h + mgrs_discovery_h) * 60:.0f} minutes")

    roi_w_px, roi_h_px = get_tiff_dimensions(args.roi_tiff_path)
    
    # --- Stage 1: S1/S2 Preprocessing ---
    if args.num_mgrs_tiles is not None and args.window_months is not None:
        n_mgrs = args.num_mgrs_tiles
        win_m = args.window_months

        # Scale S2 processing and S1 download by window months relative to a 12-month baseline
        s2_proc_h = n_mgrs * baselines["S2_PROC_PER_TILE_HOURS"] * (win_m / 12.0)
        s1_dl_h = n_mgrs * baselines["S1_DOWNLOAD_PER_TILE_HOURS"] * (win_m / 12.0)
        
        num_s1_scenes_total = n_mgrs * baselines["AVG_S1_SCENES_PER_TILE_MONTH"] * win_m
        s1_proc_h = num_s1_scenes_total * (baselines["S1_PROC_PER_SCENE_MINUTES"] / 60.0)
        
        s1_stack_h = n_mgrs * (baselines["S1_STACK_PER_TILE_MINUTES"] / 60.0)
        
        preprocessing_time_h = s2_proc_h + s1_dl_h + s1_proc_h + s1_stack_h
        
        total_min_time_h += preprocessing_time_h * 0.7 # Lower bound factor
        total_max_time_h += preprocessing_time_h * 1.5 # Upper bound factor
        stage_details.append(f"  - S1/S2 Preprocessing ({n_mgrs} MGRS tiles, {win_m} months): {preprocessing_time_h * 0.7:.1f}-{preprocessing_time_h * 1.5:.1f} hours")
    else:
        stage_details.append("  - S1/S2 Preprocessing: (Skipped, missing MGRS tile count or window months)")


    # --- Stage 2: Inference ---
    if roi_w_px is not None and roi_h_px is not None and roi_w_px > 0 and roi_h_px > 0:
        patch_size = baselines["PATCH_SIZE_PIXELS"]
        n_patches_w = math.ceil(roi_w_px / patch_size)
        n_patches_h = math.ceil(roi_h_px / patch_size)
        n_actual_patches = n_patches_w * n_patches_h
        sequential_inference_seconds = n_actual_patches * baselines["INFERENCE_PER_PATCH_SECONDS"]
        effective_concurrency = 1
        if args.inf_max_gpu > 0 and args.inf_cpu_gpu_split:
            try:
                cpu_ratio, gpu_ratio = map(int, args.inf_cpu_gpu_split.split(':'))
                if gpu_ratio > 0:
                    effective_concurrency = max(1, args.inf_max_gpu)
                    if cpu_ratio > 0:
                        effective_concurrency += max(0, args.inf_max_cpu // 2)
                    effective_concurrency = min(effective_concurrency, n_actual_patches)
                else:
                    effective_concurrency = min(n_actual_patches, max(1, args.inf_max_cpu))
            except ValueError:
                print("WARNING: Could not parse CPU_GPU_SPLIT for concurrency estimate. Assuming 1.")
                effective_concurrency = min(n_actual_patches, max(1, args.inf_max_cpu + args.inf_max_gpu if args.inf_max_gpu is not None else args.inf_max_cpu))
        
        parallel_inference_seconds = sequential_inference_seconds / max(1, effective_concurrency)
        inference_time_h = parallel_inference_seconds / 3600.0
        total_min_time_h += inference_time_h * 0.8
        total_max_time_h += inference_time_h * 1.3
        stage_details.append(f"  - Inference ({n_actual_patches} patches, ~{effective_concurrency} parallel jobs): {inference_time_h * 0.8:.1f}-{inference_time_h * 1.3:.1f} hours")
    else:
        stage_details.append("  - Inference: (Skipped, missing or invalid ROI TIFF dimensions)")


    # --- Stage 3: Stitching ---
    if roi_w_px is not None and roi_h_px is not None and roi_w_px > 0 and roi_h_px > 0:
        patch_size = baselines["PATCH_SIZE_PIXELS"]
        n_patches_w = math.ceil(roi_w_px / patch_size)
        n_patches_h = math.ceil(roi_h_px / patch_size)
        n_actual_patches = n_patches_w * n_patches_h
        stitching_time_m = baselines["STITCHING_BASE_MINUTES"] + \
                           (n_actual_patches * baselines["STITCHING_PER_PATCH_SECONDS"] / 60.0)
        stitching_time_h = stitching_time_m / 60.0
        total_min_time_h += stitching_time_h * 0.9
        total_max_time_h += stitching_time_h * 1.1
        stage_details.append(f"  - Stitching: ~{stitching_time_m:.0f} minutes")
    else:
        stage_details.append("  - Stitching: (Skipped, missing or invalid ROI TIFF dimensions)")

    return total_min_time_h, total_max_time_h, stage_details

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate BTFM processing time.")
    parser.add_argument("--config_dir_path", required=True, help="Path to the directory containing config.sh and potentially time_estimation_baselines.json.")
    parser.add_argument("--num_mgrs_tiles", type=int, help="Number of MGRS tiles intersecting ROI.")
    parser.add_argument("--roi_tiff_path", type=str, help="Path to the ROI_TIFF_FOR_PIPELINE (for dimension extraction).") 
    parser.add_argument("--window_months", type=int, help="Duration of the processing window in months.")
    # For concurrency estimation in inference
    parser.add_argument("--inf_cpu_gpu_split", type=str, default="1:1", help="Inference CPU:GPU split (e.g., '1:1').")
    parser.add_argument("--inf_max_cpu", type=int, default=20, help="Max concurrent CPU processes for inference.")
    parser.add_argument("--inf_max_gpu", type=int, default=1, help="Max concurrent GPU processes for inference.")
    
    args = parser.parse_args()

    baselines = load_baselines(args.config_dir_path)
    min_h, max_h, details = calculate_time_estimate(args, baselines)

    print("\n======================================================")
    print("           TIME ESTIMATE FOR BTFM PIPELINE           ")
    print("======================================================")
    print(f"Estimated Total Time: {min_h:.1f} - {max_h:.1f} hours")
    print("------------------------------------------------------")
    print("Breakdown (approximate):")
    for detail in details:
        print(detail)
    print("------------------------------------------------------")
    print("NOTE: This is a rough estimate. Actual time can vary based on")
    print("hardware, network, data density, and specific ROI shape.")
    print("======================================================")