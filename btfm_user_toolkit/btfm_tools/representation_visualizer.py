# btfm_tools/representation_visualizer.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import rasterio
from rasterio.plot import show as rio_show
import argparse
import os
import sys

try:
    from .representation_loader import load_representation, get_representation_geoinfo
except ImportError:
    if __package__ is None or __package__ == '':
        _TOOLKIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _TOOLS_DIR = os.path.join(_TOOLKIT_ROOT, "btfm_tools")
        if _TOOLS_DIR not in sys.path:
            sys.path.insert(0, _TOOLS_DIR)
        if os.path.basename(sys.path[0]) == 'btfm_tools':
             from representation_loader import load_representation, get_representation_geoinfo
        else:
             from btfm_tools.representation_loader import load_representation, get_representation_geoinfo
    else:
        raise

def normalize_array(arr, percentile_low=2.0, percentile_high=98.0, per_band=False):
    """Clips to percentiles and scales to 0-1.
    If per_band is True and arr is 3D (H,W,C), normalizes each band independently.
    """
    if per_band and arr.ndim == 3:
        normalized_bands = []
        for i in range(arr.shape[2]):
            band = arr[:, :, i]
            low_val = np.percentile(band, percentile_low)
            high_val = np.percentile(band, percentile_high)
            clipped_band = np.clip(band, low_val, high_val)
            # Handle cases where min and max are the same after clipping (e.g., constant band)
            if np.isclose(np.min(clipped_band), np.max(clipped_band)):
                 normalized_bands.append(np.zeros_like(clipped_band, dtype=float))
            else:
                normalized_bands.append((clipped_band - np.min(clipped_band)) / (np.max(clipped_band) - np.min(clipped_band) + 1e-7))
        return np.stack(normalized_bands, axis=-1)
    else:
        low_val = np.percentile(arr, percentile_low)
        high_val = np.percentile(arr, percentile_high)
        clipped_arr = np.clip(arr, low_val, high_val)
        if np.isclose(np.min(clipped_arr), np.max(clipped_arr)): # Check if array is constant
            return np.zeros_like(clipped_arr, dtype=float)
        return (clipped_arr - np.min(clipped_arr)) / (np.max(clipped_arr) - np.min(clipped_arr) + 1e-7) # Add epsilon for stability


def plot_rgb_image(rgb_image_0_1, title="Image", output_path=None, geo_info=None, fig_size=(10,10)):
    """Plots a 3-channel RGB image (data range 0-1)."""
    plt.figure(figsize=fig_size)
    ax = plt.gca() # Get current axes

    if geo_info and geo_info.get('transform') is not None and geo_info.get('crs') is not None:
        # rio_show expects data in (bands, height, width) or (height, width, bands)
        # if data is (height, width, bands), it will reorder.
        # Our rgb_image_0_1 is (height, width, bands)
        rio_show(np.moveaxis(rgb_image_0_1, -1, 0), # Pass (bands, H, W)
                 transform=geo_info['transform'],
                 # crs=geo_info['crs'],
                 ax=ax,
                 adjust='linear')
    else:
        ax.imshow(rgb_image_0_1)
    
    ax.set_title(title)
    ax.set_axis_off()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=600)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    plt.close()

def plot_single_band_image(band_image_data, title="Image", cmap='gray', output_path=None, geo_info=None, fig_size=(10,10), cbar_label=None, vmin=None, vmax=None):
    """Plots a single-channel image. Data can be 0-1 normalized or raw (e.g. cluster IDs)."""
    plt.figure(figsize=fig_size)
    ax = plt.gca()

    # Determine if data is normalized (0-1) or discrete (like cluster IDs)
    # This helps in deciding if vmin/vmax for imshow is useful or if rio_show adjust is needed
    is_normalized_like = np.issubdtype(band_image_data.dtype, np.floating) and \
                         (vmin is not None or (np.min(band_image_data) >= 0 and np.max(band_image_data) <= 1.0001))


    if geo_info and geo_info.get('transform') is not None and geo_info.get('crs') is not None:
        # For single band, rio_show expects (height, width)
        show_kwargs = {
            'transform': geo_info['transform'],
            'ax': ax,
            'cmap': cmap
        }
        if is_normalized_like and vmin is None and vmax is None : # only adjust if not using specific vmin/vmax for raw data
            show_kwargs['adjust'] = 'linear'
        if vmin is not None: show_kwargs['vmin'] = vmin
        if vmax is not None: show_kwargs['vmax'] = vmax
        
        rio_show(band_image_data, **show_kwargs)
    else:
        ax.imshow(band_image_data, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.set_axis_off()
    
    if ax.images:
        img_artist = ax.images[0]
        plt.colorbar(img_artist, ax=ax, label=cbar_label if cbar_label else "Value", orientation='vertical', shrink=0.8)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=600)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    plt.close()


def save_multiband_geotiff(data_array_hwc, geo_info, output_geotiff_path, dtype_str='uint8', scale_to_dtype=True):
    """Saves a HxWxC data array (e.g., RGB or PCA components) as a GeoTIFF."""
    if geo_info is None:
        print("Error: Geoinfo is required to save as GeoTIFF.")
        return
    if data_array_hwc.ndim != 3:
        # If single band (H,W), expand to (H,W,1)
        if data_array_hwc.ndim == 2:
            data_array_hwc = data_array_hwc[..., np.newaxis]
        else:
            raise ValueError("Input data_array_hwc must be 2D (H,W) or 3D (Height, Width, Channels).")

    num_bands = data_array_hwc.shape[2]
    
    # Determine numpy dtype from string
    try:
        target_np_dtype = np.dtype(dtype_str)
    except TypeError:
        print(f"Error: Invalid dtype_str '{dtype_str}'. Defaulting to float32.")
        target_np_dtype = np.float32

    processed_data = data_array_hwc # Start with original data

    if target_np_dtype == np.uint8 and scale_to_dtype:
        if np.min(processed_data) >= 0 and np.max(processed_data) <= 1.0001: # Check if it looks normalized
             processed_data = (processed_data * 255).astype(np.uint8)
        else: # If not 0-1, just cast, user responsible for range
             print(f"Warning: Scaling to uint8 but input data range is [{np.min(processed_data)}, {np.max(processed_data)}]. Casting directly.")
             processed_data = processed_data.astype(np.uint8)
    elif not scale_to_dtype: # If not scaling, just cast to target dtype
        processed_data = processed_data.astype(target_np_dtype)
    else: # For other dtypes with scaling (e.g. float32 but want to ensure it's that from possibly other float)
        processed_data = processed_data.astype(target_np_dtype)


    profile = {
        'driver': 'GTiff',
        'dtype': processed_data.dtype.name, # Use name for rasterio compatibility
        'nodata': None, 
        'width': geo_info['width'],
        'height': geo_info['height'],
        'count': num_bands,
        'crs': geo_info['crs'],
        'transform': geo_info['transform'],
        'interleave': 'pixel',
        'compress': 'lzw', 
        'tiled': True      
    }
    
    with rasterio.open(output_geotiff_path, 'w', **profile) as dst:
        for i in range(num_bands):
            dst.write(processed_data[:, :, i], i + 1)
    print(f"Multi-band GeoTIFF ({num_bands} bands, dtype: {processed_data.dtype}) saved to {output_geotiff_path}")

def visualize_fcc(representation_array, dims=(0, 1, 2), output_png_path=None, output_geotiff_path=None, geo_info=None):
    """Generates and optionally saves/shows an FCC plot and GeoTIFF."""
    if not (isinstance(dims, (list, tuple)) and len(dims) == 3):
        raise ValueError("Dims must be a list or tuple of 3 integers.")
    if not all(0 <= d < representation_array.shape[2] for d in dims):
        raise ValueError(f"All dimension indices must be between 0 and {representation_array.shape[2]-1}")

    selected_bands = representation_array[:, :, dims]
    rgb_image_0_1 = normalize_array(selected_bands, per_band=True) # Normalize each band independently
    
    plot_title = f"FCC: Dims {dims[0]},{dims[1]},{dims[2]}"
    plot_rgb_image(rgb_image_0_1, title=plot_title, output_path=output_png_path, geo_info=geo_info)
    
    if output_geotiff_path and geo_info:
        save_multiband_geotiff(rgb_image_0_1, geo_info, output_geotiff_path, dtype_str='uint8', scale_to_dtype=True)

def visualize_pca_fcc(representation_array, output_png_path=None, output_geotiff_path=None, geo_info=None):
    """Generates and optionally saves/shows a PCA-based FCC plot and GeoTIFF."""
    H, W, C = representation_array.shape
    reshaped_features = representation_array.reshape(-1, C)
    
    print("Running PCA...")
    pca = PCA(n_components=3, whiten=True)
    pca_transformed_flat = pca.fit_transform(reshaped_features)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    pca_image_hw3 = pca_transformed_flat.reshape(H, W, 3)
    pca_rgb_0_1 = normalize_array(pca_image_hw3, per_band=True)
    
    plot_title = "PCA FCC (Top 3 Components)"
    plot_rgb_image(pca_rgb_0_1, title=plot_title, output_path=output_png_path, geo_info=geo_info)
    
    if output_geotiff_path and geo_info:
        save_multiband_geotiff(pca_rgb_0_1, geo_info, output_geotiff_path, dtype_str='uint8', scale_to_dtype=True)

def visualize_single_feature(representation_array, dim_index, output_png_path=None, output_geotiff_path=None, geo_info=None):
    """Generates and optionally saves/shows a single feature plot and GeoTIFF."""
    if not (0 <= dim_index < representation_array.shape[2]):
        raise ValueError(f"Dimension index must be between 0 and {representation_array.shape[2]-1}")

    feature_band = representation_array[:, :, dim_index]
    feature_normalized_0_1 = normalize_array(feature_band)
    
    plot_title = f"Feature Dimension {dim_index}"
    plot_single_band_image(feature_normalized_0_1, title=plot_title, cmap='viridis', 
                           output_path=output_png_path, geo_info=geo_info, cbar_label="Normalized Value")
    
    if output_geotiff_path and geo_info:
        save_multiband_geotiff(feature_normalized_0_1, geo_info, output_geotiff_path, dtype_str='float32', scale_to_dtype=True)


def display_stats(representation_array):
    H, W, C = representation_array.shape
    print(f"\nRepresentation Array Shape: {H} x {W} x {C} features")
    print("Statistics per feature:")
    print("------------------------------------------------------------------------------------")
    print(f"{'Feature':<10} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std Dev':<12} {'Median':<12}")
    print("------------------------------------------------------------------------------------")
    
    features_to_show_indices = list(range(C))
    if C > 10: 
        features_to_show_indices = list(range(min(5, C))) + list(range(max(min(5,C), C-5), C))
    features_to_show_indices = sorted(list(set(features_to_show_indices))) # Deduplicate

    for i_idx, i in enumerate(features_to_show_indices):
        feature_data = representation_array[:, :, i]
        stats = {
            "min": np.min(feature_data),
            "max": np.max(feature_data),
            "mean": np.mean(feature_data),
            "std": np.std(feature_data),
            "median": np.median(feature_data)
        }
        print(f"{i:<10} {stats['min']:<12.4f} {stats['max']:<12.4f} {stats['mean']:<12.4f} {stats['std']:<12.4f} {stats['median']:<12.4f}")
        if C > 10 and i_idx == (len(features_to_show_indices) // 2) -1 and len(features_to_show_indices) % 2 == 0 :
             if features_to_show_indices[i_idx+1] - features_to_show_indices[i_idx] > 1:
                print("... (statistics for intermediate features omitted for brevity) ...")
        elif C > 10 and i_idx == min(4, C-1) and features_to_show_indices[i_idx+1] - features_to_show_indices[i_idx] > 1: # After first block
             print("... (statistics for intermediate features omitted for brevity) ...")


    print("------------------------------------------------------------------------------------")

def visualize_kmeans(representation_array, n_clusters=5, output_png_path=None, output_geotiff_path=None, geo_info=None):
    """Performs k-Means clustering and visualizes the cluster map."""
    H, W, C = representation_array.shape
    reshaped_features = representation_array.reshape(-1, C)

    print(f"Running k-Means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto', verbose=0)
    cluster_labels_flat = kmeans.fit_predict(reshaped_features)
    
    cluster_map = cluster_labels_flat.reshape(H, W)
    
    plot_title = f"k-Means Clustering ({n_clusters} clusters)"
    
    try:
        cmap_discrete = plt.colormaps.get_cmap('tab20', n_clusters if n_clusters <=20 else None)
    except AttributeError:
         cmap_discrete = plt.cm.get_cmap('tab20', n_clusters if n_clusters <=20 else None)
    if n_clusters > 20:
        print("Warning: More than 20 clusters, 'tab20' colormap will repeat. Consider 'viridis' or another cmap.")
        cmap_discrete = 'viridis'


    plot_single_band_image(cluster_map, title=plot_title, cmap=cmap_discrete, 
                           output_path=output_png_path, geo_info=geo_info, 
                           cbar_label="Cluster ID", vmin=0, vmax=n_clusters-1)

    if output_geotiff_path and geo_info:
        save_multiband_geotiff(cluster_map.astype(np.uint8), geo_info, output_geotiff_path, dtype_str='uint8', scale_to_dtype=False)


def main_cli():
    parser = argparse.ArgumentParser(description="Visualize BTFM representations.")
    parser.add_argument("--npy_path", required=True, help="Path to the .npy representation file.")
    parser.add_argument("--ref_tiff_path", help="Path to the reference ROI GeoTIFF (for geoinfo and saving GeoTIFF outputs).")
    parser.add_argument("--mode", choices=['fcc', 'pca_fcc', 'single_feature', 'stats', 'kmeans'], required=True,
                        help="Visualization or analysis mode.")
    
    fcc_group = parser.add_argument_group('FCC options')
    fcc_group.add_argument("--dims", help="Comma-separated dimensions for FCC (e.g., 0,5,10). Max 3 dimensions.")
    
    single_feature_group = parser.add_argument_group('Single Feature options')
    single_feature_group.add_argument("--dim", type=int, help="Single dimension index for single_feature mode.")

    kmeans_group = parser.add_argument_group('k-Means options')
    kmeans_group.add_argument("--n_clusters", type=int, default=5, help="Number of clusters for k-Means (default: 5).")
    
    output_group = parser.add_argument_group('Output options')
    output_group.add_argument("--output_png_path", help="Path to save the output visualization as a PNG image.")
    output_group.add_argument("--output_geotiff_path", help="Path to save the output visualization as a GeoTIFF (for fcc, pca_fcc, single_feature, kmeans modes).")

    args = parser.parse_args()

    print(f"Loading representation from: {args.npy_path}")
    rep_array = load_representation(args.npy_path)
    geo_info = None
    if args.ref_tiff_path:
        if not os.path.exists(args.ref_tiff_path):
            print(f"Warning: Reference TIFF path specified but not found: {args.ref_tiff_path}. GeoTIFF output will not be possible, and plots will not be accurately georeferenced.")
        else:
            print(f"Loading geoinfo from: {args.ref_tiff_path}")
            geo_info = get_representation_geoinfo(args.ref_tiff_path)
            if geo_info:
                if rep_array.shape[0] != geo_info['height'] or rep_array.shape[1] != geo_info['width']:
                    print(f"Warning: Representation dimensions ({rep_array.shape[0]}x{rep_array.shape[1]}) "
                          f"differ from reference TIFF dimensions ({geo_info['height']}x{geo_info['width']}). "
                          "GeoTIFF output might have incorrect extent or resolution if this discrepancy is large.")
            else:
                print(f"Warning: Could not load geoinfo from {args.ref_tiff_path}.")
    elif args.output_geotiff_path:
        print("Warning: --output_geotiff_path specified, but --ref_tiff_path is missing. GeoTIFF cannot be saved.")
        args.output_geotiff_path = None 

    if args.mode == 'fcc':
        if not args.dims:
            parser.error("--dims DIMS is required for fcc mode.")
        try:
            dims_list = [int(d.strip()) for d in args.dims.split(',')]
            if len(dims_list) != 3:
                parser.error("--dims must specify exactly 3 comma-separated integers.")
            visualize_fcc(rep_array, dims=tuple(dims_list), 
                          output_png_path=args.output_png_path, 
                          output_geotiff_path=args.output_geotiff_path, 
                          geo_info=geo_info)
        except ValueError as e:
            parser.error(f"Invalid --dims value: {e}")
            
    elif args.mode == 'pca_fcc':
        visualize_pca_fcc(rep_array, 
                          output_png_path=args.output_png_path,
                          output_geotiff_path=args.output_geotiff_path,
                          geo_info=geo_info)
            
    elif args.mode == 'single_feature':
        if args.dim is None:
            parser.error("--dim DIM_INDEX is required for single_feature mode.")
        visualize_single_feature(rep_array, dim_index=args.dim,
                                 output_png_path=args.output_png_path,
                                 output_geotiff_path=args.output_geotiff_path,
                                 geo_info=geo_info)

    elif args.mode == 'stats':
        display_stats(rep_array)

    elif args.mode == 'kmeans':
        visualize_kmeans(rep_array, n_clusters=args.n_clusters,
                         output_png_path=args.output_png_path,
                         output_geotiff_path=args.output_geotiff_path,
                         geo_info=geo_info)
    else:
        print(f"Error: Unknown mode '{args.mode}'.")


if __name__ == "__main__":
    main_cli()