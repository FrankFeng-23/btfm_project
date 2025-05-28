import numpy as np
import matplotlib.pyplot as plt

path = "/home/zf281/rds/rds-airr-p3-w8D3JcRiKZQ/james/2017/stitched_representation.npy"
data = np.load(path, mmap_mode='r')
print(data.shape)  # 输出数组的形状

# first_three_band = data[::10,::10,:3].copy()
first_three_band = data[:,:,:3].copy()
# 归一化
for i in range(3):
    first_three_band[:, :, i] = (first_three_band[:, :, i] - np.min(first_three_band[:, :, i])) / (np.max(first_three_band[:, :, i]) - np.min(first_three_band[:, :, i]))

# 可视化
plt.imshow(first_three_band)
plt.savefig('first_three_band2.png')
plt.close()


# mask_file_path = "/local/zf281/data_processed/22MHB/masks.npy"
# mask_data = np.load(mask_file_path) # (T,H,W)
# # 找出含有最多1的时间步
# mask_sum = np.sum(mask_data, axis=(1,2))
# max_idx = np.argmax(mask_sum)
# print("valid time step:", max_idx)

# rgb_time_step = max_idx

# path = "/local/zf281/data_processed/22MHB/bands.npy"
# data = np.load(path, mmap_mode='r')
# print(data.shape)  # 输出数组的形状

# first_three_band = data[rgb_time_step,:,:,3:6].copy()
# # tofloat
# first_three_band = first_three_band.astype(np.float32)
# # 归一化
# for i in range(3):
#     first_three_band[:, :, i] = (first_three_band[:, :, i] - np.min(first_three_band[:, :, i])) / (np.max(first_three_band[:, :, i]) - np.min(first_three_band[:, :, i]))

# # 可视化
# plt.imshow(first_three_band)
# plt.savefig('first_three_band.png')
# plt.close()