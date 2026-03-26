import numpy as np
import matplotlib
matplotlib.use('Agg')  # <-- 关键修复：设置非交互式后端，必须在导入pyplot之前
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_and_dequantize_representation(representation_file_path, scales_file_path):
    """
    Load and dequantize int8 representations back to float32.
    
    Args:
        representation_file_path: Path to the int8 representation file (H,W,C)
        scales_file_path: Path to the float32 scales file (H,W)
    
    Returns:
        representation_f32: float32 ndarray of shape (H,W,C)
    """
    # Load the files
    representation_int8 = np.load(representation_file_path)  # (H, W, C), dtype=int8
    scales = np.load(scales_file_path)  # (H, W), dtype=float32
    
    # Convert int8 to float32 for computation
    representation_f32 = representation_int8.astype(np.float32)
    
    # Expand scales to match representation shape
    # scales shape: (H, W) -> (H, W, 1)
    scales_expanded = scales[..., np.newaxis]
    
    # Dequantize by multiplying with scales
    representation_f32 = representation_f32 * scales_expanded
    
    return representation_f32

# path = "0_2500_500_3000.npy"
path = "/absolute_path_to_data_dir/representation_retiled_qat/0_2500_500_3000.npy"
data = np.load(path, mmap_mode='r')
print(data.dtype)  # Output the data type
print(data.shape)  # Output the shape of the array

# optional
path_scales = "/absolute_path_to_data_dir/representation_retiled_qat/0_2500_500_3000_scales.npy"
scales = np.load(path_scales)
data = load_and_dequantize_representation(path, path_scales)

# === 第一个可视化：原始前三个维度 ===
# first_three_band = data[::10,::10,:3].copy()
first_three_band = data[:,:,:3].copy()
# first_three_band = data[:,:,3:6,0].copy()
# first_three_band = data[:,:].copy()
# Convert to float
first_three_band = first_three_band.astype(np.float32)
# Normalize
for i in range(first_three_band.shape[2]):
    first_three_band[:, :, i] = (first_three_band[:, :, i] - np.min(first_three_band[:, :, i])) / (np.max(first_three_band[:, :, i]) - np.min(first_three_band[:, :, i]))

# Visualization
plt.figure(figsize=(10, 8))
plt.imshow(first_three_band)
plt.title('Original First 3 Dimensions Visualization')
plt.savefig('first_three_band.png')
plt.close()

# === 第二个可视化：PCA降维至3维 ===
print("Applying PCA...")
# 获取原始数据形状
original_shape = data.shape
print(f"Original shape: {original_shape}")

# 将数据重塑为 (height*width, channels)
data_reshaped = data.reshape(-1, original_shape[2])
print(f"Reshaped for PCA: {data_reshaped.shape}")

# 应用PCA，从128维降到3维
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_reshaped)
print(f"PCA result shape: {pca_result.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# 将结果重塑回图像形状
pca_image = pca_result.reshape(original_shape[0], original_shape[1], 3)

# Convert to float and normalize
pca_image = pca_image.astype(np.float32)
for i in range(pca_image.shape[2]):
    pca_image[:, :, i] = (pca_image[:, :, i] - np.min(pca_image[:, :, i])) / (np.max(pca_image[:, :, i]) - np.min(pca_image[:, :, i]))

# Visualization
plt.figure(figsize=(10, 8))
plt.imshow(pca_image)
plt.title('PCA 3-Component Visualization (128D→3D)')
plt.savefig('pca_three_band.png')
plt.close()

print("Both visualizations saved successfully!")