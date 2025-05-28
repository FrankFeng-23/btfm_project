# import rasterio
# import numpy as np
# import matplotlib.pyplot as plt
# from rasterio.enums import Resampling

# # File path - update this to your file location
# file_path = "S1A_IW_GRDH_1SDV_20220108T085758_20220108T085823_041367_04EB14_9233_DESCENDING.tif"

# # Open the GeoTIFF file
# with rasterio.open(file_path) as src:
#     # Read the first band
#     band1 = src.read(1)
    
#     # Get metadata
#     width = src.width
#     height = src.height
    
#     # Print basic information
#     print(f"File: {file_path}")
#     print(f"Shape: {band1.shape}")
#     print(f"Data Type: {band1.dtype}")
#     print(f"Min value: {band1.min()}")
#     print(f"Max value: {band1.max()}")
#     print(f"Mean value: {band1.mean()}")
    
#     # Calculate the new dimensions (downsampled by factor of 10)
#     new_width = width // 10
#     new_height = height // 10
    
#     # Resample data to target shape
#     downsampled = src.read(
#         1,  # First band
#         out_shape=(new_height, new_width),
#         resampling=Resampling.average
#     )
    
#     # Update spatial resolution for the plot
#     downsampled_transform = src.transform * src.transform.scale(
#         (width / new_width),
#         (height / new_height)
#     )

# # Create figure with two subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# # Determine plotting range by percentiles to handle extreme values
# vmin, vmax = np.percentile(band1, [2, 98])

# # Plot original image
# im1 = ax1.imshow(band1, cmap='gray', vmin=vmin, vmax=vmax)
# ax1.set_title(f'Original Resolution ({width}x{height})')
# plt.colorbar(im1, ax=ax1, shrink=0.6)

# # Plot downsampled image
# im2 = ax2.imshow(downsampled, cmap='gray', vmin=vmin, vmax=vmax)
# ax2.set_title(f'Downsampled 10x ({new_width}x{new_height})')
# plt.colorbar(im2, ax=ax2, shrink=0.6)

# # Add overall title
# plt.suptitle("First Band Visualization", fontsize=16)
# plt.tight_layout()
# # plt.show()

# # Option to save the figure to a file
# plt.savefig('tif_visualization.png', dpi=300, bbox_inches='tight')

# print("Visualization complete!")


import numpy as np

# file_path = "/scratch/zf281/jovana/representation_retiled/1000_3000_1500_3500.npy"
# file_path = "/scratch/zf281/austrian_crop_whole_year/bands.npy"
file_path = "/scratch/zf281/jovana/stitched_representation.npy"

data = np.load(file_path, mmap_mode="r") # (T, H, W, C)
rgb = data[::10, ::10, :3].copy()  # 只取前3个波段
# 转为float
rgb = rgb.astype(np.float32)
print(f"Data shape: {rgb.shape}")
# print(data)

# rgb = data[:, :, :3].copy()

# 归一化
for i in range(rgb.shape[2]):
    rgb[:, :, i] = (rgb[:, :, i] - np.min(rgb[:, :, i])) / (np.max(rgb[:, :, i]) - np.min(rgb[:, :, i]))
import matplotlib.pyplot as plt
plt.imshow(rgb)
plt.savefig("rgb_visualization.png", dpi=300, bbox_inches='tight')
plt.close()


# import numpy as np
# file_path = "/scratch/zf281/jovana/retiled_d_pixel/6500_8000_6651_8500/masks.npy"
# data = np.load(file_path, mmap_mode="r") # (T, H, W, C)
# print(f"Data shape: {data.shape}")
# # 找出T和C维度全为0的(H,W)索引
# # zero_indices = np.argwhere(np.all(data == 0, axis=(0, 3)))
# # print(f"Zero indices: {len(zero_indices)}")
# import matplotlib.pyplot as plt
# rgb = data[0, :, :].copy()  # 只取前3个波段
# # 归一化
# # for i in range(rgb.shape[2]):
# #     rgb[:, :, i] = (rgb[:, :, i] - np.min(rgb[:, :, i])) / (np.max(rgb[:, :, i]) - np.min(rgb[:, :, i]))
# plt.imshow(rgb)
# plt.savefig("rgb_visualization.png", dpi=300, bbox_inches='tight')
# plt.close()
