import numpy as np
import matplotlib.pyplot as plt

try:
    file_path = "/rds/project/rds-w8D3JcRiKZQ/global_200m_d_pixel/18SVD/2017/data_processed/sar_ascending.npy"
    # file_path = "/scratch/zf281/ethiopia/data_processed/37PEK/sar_ascending.npy"
    data = np.load(file_path, mmap_mode='r')
    # print(f"Number of non-zero elements: {np.count_nonzero(data)}")
    # 遍历时间维度，找出不为0的元素最多的时间步
    nonzero_count = np.count_nonzero(data, axis=(1,2,3))
    max_idx = np.argmax(nonzero_count)
    print(f"valid time step:", max_idx)
    print(f"datatype of sar data: {data.dtype}")
    # 用memmap
    # data = np.memmap(file_path, dtype='int16', mode='r', shape=(142, 10980, 10980, 10))
    sar_time_step = max_idx

    print(data.shape)  # 输出数组的形状
    print(data.dtype)  # 输出数组的数据类型
    # print(data[sar_time_step,0:10,0:10,:])

    single_image = data[sar_time_step, :, :, 0]  # 获取第一个时间步的第一个波段
    plt.imshow(single_image, cmap='gray')
    plt.savefig('sar_0.png')
    plt.close()

    single_image = data[sar_time_step, :, :, 1]  # 获取第一个时间步的第一个波段
    plt.imshow(single_image, cmap='gray')
    plt.savefig('sar_1.png')
    plt.close()
except Exception as e:
    print(f"Error loading SAR data: {e}")


band_file_path = "/rds/project/rds-w8D3JcRiKZQ/global_200m_d_pixel/18SVD/2017/data_processed/bands.npy"
# band_file_path = "/scratch/zf281/robin/fungal/data_processed/13TCL/bands.npy"
band_data = np.load(band_file_path, mmap_mode='r')
print(band_data.shape)  # 输出数组的形状
print(f"datatype of band_data: {band_data.dtype}")

mask_file_path = "/rds/project/rds-w8D3JcRiKZQ/global_200m_d_pixel/18SVD/2017/data_processed/masks.npy"
# mask_file_path = "/scratch/zf281/robin/fungal/data_processed/13TCL/masks.npy"
mask_data = np.load(mask_file_path) # (T,H,W)
# 找出含有最多1的时间步
mask_sum = np.sum(mask_data, axis=(1,2))
max_idx = np.argmax(mask_sum)
print("valid time step:", max_idx)
print(f"datatype of mask_data: {mask_data.dtype}")

rgb_time_step = max_idx

single_rbg_image = band_data[rgb_time_step, :, :, 3:6]  # 获取第一个时间步的RGB波段
# 转为float
single_rbg_image = single_rbg_image.astype(np.float32)
# 转为rgb
single_rbg_image = single_rbg_image[:, :, [2, 1, 0]]
# 归一化
for i in range(3):
    single_rbg_image[:, :, i] = (single_rbg_image[:, :, i] - np.min(single_rbg_image[:, :, i])) / (np.max(single_rbg_image[:, :, i]) - np.min(single_rbg_image[:, :, i]))

import cv2

def histogram_equalization(image):
    for i in range(3):
        image[:,:,i] = cv2.equalizeHist((image[:,:,i] * 255).astype(np.uint8)) / 255.0
    return image

single_rbg_image = histogram_equalization(single_rbg_image)    

plt.imshow(single_rbg_image)
plt.savefig('rgb.png')
plt.close()
