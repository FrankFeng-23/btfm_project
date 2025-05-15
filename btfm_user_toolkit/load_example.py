import os
import sys
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
toolkit_dir = os.path.dirname(script_dir)
tools_dir = os.path.join(toolkit_dir, "btfm_tools")
sys.path.insert(0, tools_dir)

from btfm_tools.representation_loader import *

def main():
    # Path to where the run_pipeline_for_geojsons.sh saved outputs
    # This should match the output_base_dir used when running the script
    output_base_dir = "/scratch/ray25/btfm_wrap/btfm_outputs" # IMPORTANT: CHANGE THIS

    representations_dir = os.path.join(output_base_dir, "representations")
    roi_tiffs_dir = os.path.join(output_base_dir, "roi_tiffs_generated")

    roi_name = "oxford"
    
    representation_file = os.path.join(representations_dir, f"{roi_name}_representation.npy")
    roi_tiff_file = os.path.join(roi_tiffs_dir, f"{roi_name}_roi_10m.tiff")

    if not os.path.exists(representation_file):
        print(f"Representation file not found: {representation_file}")
        return

    print(f"Loading representation from: {representation_file}")
    try:
        representation_array = load_representation(representation_file)
        print(f"Successfully loaded representation. Shape: {representation_array.shape}, Dtype: {representation_array.dtype}")

        # You can now use representation_array for downstream tasks
        # For example, print some values:
        print(f"Sample value from representation (center pixel, first 3 features): {representation_array[representation_array.shape[0]//2, representation_array.shape[1]//2, :3]}")

        # Get geoinfo
        if os.path.exists(roi_tiff_file):
            print(f"\nLoading geoinfo from: {roi_tiff_file}")
            geo_info = get_representation_geoinfo(roi_tiff_file)
            if geo_info:
                print(f"Geo Info: CRS={geo_info['crs']}, Width={geo_info['width']}, Height={geo_info['height']}")
                print(f"Transform: {geo_info['transform']}")
                print(f"Bounds: {geo_info['bounds']}")
                if representation_array.shape[0] == geo_info['height'] and \
                   representation_array.shape[1] == geo_info['width']:
                    print("Representation dimensions match reference TIFF dimensions.")
                else:
                    print(f"WARNING: Representation dimensions ({representation_array.shape[0]}x{representation_array.shape[1]}) "
                          f"do not match reference TIFF dimensions ({geo_info['height']}x{geo_info['width']}).")
            else:
                print(f"Could not load geoinfo for {roi_tiff_file}.")
        else:
            print(f"Reference ROI TIFF not found: {roi_tiff_file}. Cannot load geoinfo.")
            
            
        if representation_array is not None and geo_info is not None:
            # Get lat/lon for a specific pixel, like the center pixel's representation
            center_row = representation_array.shape[0] // 2
            center_col = representation_array.shape[1] // 2
            
            center_latlon = get_latlon_for_pixel(center_row, center_col, roi_tiff_file)
            if center_latlon:
                center_representation = representation_array[center_row, center_col]
                print(f"\nRepresentation at pixel ({center_row}, {center_col}):")
                print(f"  Lat/Lon: {center_latlon[0]:.6f}, {center_latlon[1]:.6f}")
                print(f"  Vector (first 3 dims): {center_representation[:3]}")

    except FileNotFoundError:
        print(f"Could not find the representation file: {representation_file}")
    except ValueError as e:
        print(f"Error loading representation file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()