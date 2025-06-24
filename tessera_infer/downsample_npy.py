import numpy as np

decimation_factor = 10 # shrink the data decimation_factor*decimation_factor times

data_path = '/home/zf281/rds/rds-airr-p3-w8D3JcRiKZQ/austrian_crop_v1.0_pipeline/stitched_representation.npy'

data = np.load(data_path) # (h, w, c)

# shrink the data decimation_factor*decimation_factor times
# data = data[:, ::decimation_factor, ::decimation_factor, ...] # (t, h, w, ...)
data = data[::decimation_factor, ::decimation_factor, ...] # (h, w, ...)
# 打印数据的形状
print(f'Data shape after downsampling: {data.shape}')  # ( h, w, c)
# 打印dtype
print(f'Data dtype: {data.dtype}')
# save the downsampled data
save_path = data_path.replace('.npy', '_downsample_100.npy')
np.save(save_path, data)
print(f'Saved downsampled data to {save_path}')


try:
    data_path = '/home/zf281/rds/rds-airr-p3-w8D3JcRiKZQ/austrian_crop_v1.0_pipeline/stitched_scales.npy'

    data = np.load(data_path) # (h, w, c)

    # shrink the data decimation_factor*decimation_factor times
    # data = data[:, ::decimation_factor, ::decimation_factor, ...] # (t, h, w, ...)
    data = data[::decimation_factor, ::decimation_factor, ...] # (h, w, ...)
    # 打印数据的形状
    print(f'Data shape after downsampling: {data.shape}')  # ( h, w, c)
    # 打印dtype
    print(f'Data dtype: {data.dtype}')
    # save the downsampled data
    save_path = data_path.replace('.npy', '_downsample_100.npy')
    np.save(save_path, data)
    print(f'Saved downsampled data to {save_path}')
except Exception as e:
    print(f"Error processing data: {e}")