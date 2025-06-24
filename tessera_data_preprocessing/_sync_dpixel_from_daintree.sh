#!/bin/bash

# 设置变量
REMOTE_USER="zf281"
REMOTE_HOST="daintree.cl.cam.ac.uk"
REMOTE_BASE="/scratch/zf281/create_d-pixels_burned_scar/data/d-pixel"
LOCAL_BASE="/home/zf281/rds/rds-airr-p3-w8D3JcRiKZQ/burned_scar"

# 创建目标目录
mkdir -p "$LOCAL_BASE"

# 获取远程服务器上的所有子文件夹
echo "正在获取远程文件夹列表..."
FOLDERS=$(ssh ${REMOTE_USER}@${REMOTE_HOST} "ls -d ${REMOTE_BASE}/*_merged_10m_shape" 2>/dev/null | xargs -n1 basename)

# 遍历每个文件夹
for folder in $FOLDERS; do
    echo "正在同步: $folder"
    
    # 创建本地目标文件夹
    mkdir -p "$LOCAL_BASE/$folder"
    
    # 使用 rsync 同步 npy 文件，显示进度
    rsync -avP --include="*.npy" --include="*/" --exclude="*" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/${folder}/data_processed/" \
        "$LOCAL_BASE/$folder/"
done

echo "同步完成！"