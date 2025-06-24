#!/bin/bash

# 设置变量
SOURCE_BASE="/home/zf281/rds/rds-airr-p3-w8D3JcRiKZQ/global_200m_d_pixel"
REMOTE_HOST="avsm2_f4q@aac10.amd.com"
REMOTE_BASE="/shared/amdgpu/home/avsm2_f4q/code/btfm4rs/data/ssl_training/tiles"

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# 退出脚本若有任何命令失败
set -e
# 如果管道中任何命令失败，则整个管道失败
set -o pipefail

# 创建临时文件和日志文件
TEMP_FILE=$(mktemp)
REMOTE_EXISTING_FILE=$(mktemp)
LOG_FILE="transfer_log_$(date +%Y%m%d_%H%M%S).txt"

# 写入初始日志信息
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Script started at $(date)" | tee -a "$LOG_FILE"
echo "SOURCE_BASE: $SOURCE_BASE" | tee -a "$LOG_FILE"
echo "REMOTE_HOST: $REMOTE_HOST" | tee -a "$LOG_FILE"
echo "REMOTE_BASE: $REMOTE_BASE" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# 开始时间
START_TIME=$(date +%s)

echo -e "${BLUE}正在查找包含9个npy文件的文件夹...${NC}" | tee -a "$LOG_FILE"

# 找到所有符合条件的目录
find "$SOURCE_BASE" -path "*/data_processed/*.npy" -type f -printf "%h\n" | \
    sort | uniq -c | \
    awk '$1==9 {print $2}' > "$TEMP_FILE"

# 获取总数
TOTAL_FOLDERS=$(wc -l < "$TEMP_FILE")
echo -e "${GREEN}找到 $TOTAL_FOLDERS 个符合条件的文件夹${NC}" | tee -a "$LOG_FILE"
if [ "$TOTAL_FOLDERS" -eq 0 ]; then
    echo -e "${RED}错误: 未找到符合条件的文件夹。脚本将退出。${NC}" | tee -a "$LOG_FILE"
    rm -f "$TEMP_FILE" "$REMOTE_EXISTING_FILE"
    exit 1
fi

# 检查远程服务器上已存在的包含9个npy文件的文件夹
echo -e "${BLUE}正在检查远程服务器上已存在的文件夹...${NC}" | tee -a "$LOG_FILE"
echo "INFO: 执行远程检查命令..." | tee -a "$LOG_FILE"

# 构建远程检查命令
REMOTE_CHECK_CMD="cd '$REMOTE_BASE' 2>/dev/null && find . -maxdepth 2 -name '*.npy' -type f | sed 's|^\./||' | awk -F'/' '{print \$1}' | sort | uniq -c | awk '\$1==9 {print \$2}'"

# 执行远程命令并保存结果
ssh -n "$REMOTE_HOST" "$REMOTE_CHECK_CMD" 2>/dev/null > "$REMOTE_EXISTING_FILE" || {
    echo -e "${YELLOW}警告: 无法检查远程服务器或远程基础目录不存在，将继续传输所有文件夹${NC}" | tee -a "$LOG_FILE"
    > "$REMOTE_EXISTING_FILE"  # 创建空文件
}

REMOTE_EXISTING_COUNT=$(wc -l < "$REMOTE_EXISTING_FILE")
echo -e "${GREEN}远程服务器上已存在 $REMOTE_EXISTING_COUNT 个包含9个npy文件的文件夹${NC}" | tee -a "$LOG_FILE"

# 如果有已存在的文件夹，显示列表
if [ "$REMOTE_EXISTING_COUNT" -gt 0 ]; then
    echo "INFO: 远程已存在的文件夹列表已保存" | tee -a "$LOG_FILE"
    echo "已存在的文件夹:" >> "$LOG_FILE"
    cat "$REMOTE_EXISTING_FILE" >> "$LOG_FILE"
fi

# 计算需要传输的文件夹数量和大小
echo -e "${BLUE}正在分析需要传输的文件夹...${NC}" | tee -a "$LOG_FILE"
NEED_TRANSFER_COUNT=0
SKIP_COUNT=0
TOTAL_SIZE=0
SKIP_SIZE=0

# 创建临时文件存储需要传输的文件夹
TRANSFER_LIST=$(mktemp)

while IFS= read -r dir; do
    MGRS_CODE=$(echo "$dir" | awk -F'/' '{print $(NF-2)}')
    YEAR=$(echo "$dir" | awk -F'/' '{print $(NF-1)}')
    REMOTE_DIR_NAME="${MGRS_CODE}_${YEAR}"
    
    # 检查远程是否已存在
    if grep -q "^${REMOTE_DIR_NAME}$" "$REMOTE_EXISTING_FILE"; then
        SKIP_COUNT=$((SKIP_COUNT + 1))
        SIZE=$(find "$dir" -name "*.npy" -type f -exec du -cb {} + 2>/dev/null | grep total$ | cut -f1 || echo "0")
        SKIP_SIZE=$((SKIP_SIZE + SIZE))
        echo "SKIP: $REMOTE_DIR_NAME 已存在于远程服务器" >> "$LOG_FILE"
    else
        NEED_TRANSFER_COUNT=$((NEED_TRANSFER_COUNT + 1))
        SIZE=$(find "$dir" -name "*.npy" -type f -exec du -cb {} + 2>/dev/null | grep total$ | cut -f1 || echo "0")
        TOTAL_SIZE=$((TOTAL_SIZE + SIZE))
        echo "$dir" >> "$TRANSFER_LIST"
    fi
done < "$TEMP_FILE"

# 显示分析结果
echo -e "${CYAN}═══════════════════════════════════════════${NC}" | tee -a "$LOG_FILE"
echo -e "${CYAN}              传输前分析结果                 ${NC}" | tee -a "$LOG_FILE"
echo -e "${CYAN}═══════════════════════════════════════════${NC}" | tee -a "$LOG_FILE"
echo -e "${GREEN}本地文件夹总数: $TOTAL_FOLDERS${NC}" | tee -a "$LOG_FILE"
echo -e "${MAGENTA}已存在（跳过）: $SKIP_COUNT 个文件夹${NC}" | tee -a "$LOG_FILE"
if [ "$SKIP_COUNT" -gt 0 ]; then
    SKIP_SIZE_HUMAN=$(numfmt --to=iec-i --suffix=B "$SKIP_SIZE" 2>/dev/null || echo "$SKIP_SIZE bytes")
    echo -e "${MAGENTA}跳过的数据量: $SKIP_SIZE_HUMAN${NC}" | tee -a "$LOG_FILE"
fi
echo -e "${GREEN}需要传输: $NEED_TRANSFER_COUNT 个文件夹${NC}" | tee -a "$LOG_FILE"
TOTAL_SIZE_HUMAN=$(numfmt --to=iec-i --suffix=B "$TOTAL_SIZE" 2>/dev/null || echo "$TOTAL_SIZE bytes")
echo -e "${GREEN}需要传输的数据量: $TOTAL_SIZE_HUMAN${NC}" | tee -a "$LOG_FILE"
echo -e "${CYAN}═══════════════════════════════════════════${NC}" | tee -a "$LOG_FILE"

# 如果没有需要传输的文件夹，退出
if [ "$NEED_TRANSFER_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}所有文件夹都已存在于远程服务器，无需传输。${NC}" | tee -a "$LOG_FILE"
    rm -f "$TEMP_FILE" "$REMOTE_EXISTING_FILE" "$TRANSFER_LIST"
    exit 0
fi

# 直接继续，无需确认

# 初始化计数器
COUNTER=0
TRANSFERRED_SIZE=0
FAILED_TRANSFERS=()

# 进度条函数
show_progress_bar() {
    local current=$1
    local total=$2
    local percent=$((current * 100 / total))
    local bar_length=40
    local filled_length=$((percent * bar_length / 100))
    
    local elapsed=$(($(date +%s) - START_TIME))
    local avg_time_per_item=0
    local eta_seconds=0
    local eta_display="计算中..."
    
    if [ $current -gt 0 ] && [ $elapsed -gt 0 ]; then
        avg_time_per_item=$((elapsed / current))
        eta_seconds=$(((total - current) * avg_time_per_item))
        if [ $eta_seconds -gt 0 ]; then
            eta_display=$(printf '%02d:%02d:%02d' $((eta_seconds/3600)) $((eta_seconds%3600/60)) $((eta_seconds%60)))
        elif [ $current -eq $total ]; then
            eta_display="已完成"
        fi
    fi
    
    printf "\r${CYAN}总进度: ["
    printf "%${filled_length}s" | tr ' ' '█'
    printf "%$((bar_length - filled_length))s" | tr ' ' '░'
    printf "] ${percent}%% (${current}/${total}) 预计剩余: ${eta_display}${NC}"
}

# 重置开始时间为实际传输开始时间
START_TIME=$(date +%s)

# 处理每个需要传输的目录
echo -e "\n开始处理文件夹传输..." | tee -a "$LOG_FILE"
while IFS= read -r dir; do
    COUNTER=$((COUNTER + 1))
    echo -e "\n----------------------------------------" | tee -a "$LOG_FILE"
    echo "INFO: [${COUNTER}/${NEED_TRANSFER_COUNT}] 开始处理源文件夹: $dir" | tee -a "$LOG_FILE"
    
    MGRS_CODE=$(echo "$dir" | awk -F'/' '{print $(NF-2)}')
    YEAR=$(echo "$dir" | awk -F'/' '{print $(NF-1)}')
    
    REMOTE_DIR_NAME="${MGRS_CODE}_${YEAR}"
    REMOTE_DIR="${REMOTE_BASE}/${REMOTE_DIR_NAME}"
    echo "INFO: MGRS_CODE: $MGRS_CODE, YEAR: $YEAR" | tee -a "$LOG_FILE"
    echo "INFO: 构建远程目标路径: $REMOTE_DIR" | tee -a "$LOG_FILE"
    
    FOLDER_SIZE=$(find "$dir" -name "*.npy" -type f -exec du -cb {} + 2>/dev/null | grep total$ | cut -f1 || echo "0")
    FOLDER_SIZE_HUMAN=$(numfmt --to=iec-i --suffix=B "$FOLDER_SIZE" 2>/dev/null || echo "$FOLDER_SIZE bytes")
    
    printf "\033[2K" 
    show_progress_bar "$COUNTER" "$NEED_TRANSFER_COUNT"
    echo "" 
    
    echo -e "${YELLOW}[${COUNTER}/${NEED_TRANSFER_COUNT}] 传输: ${REMOTE_DIR_NAME} (${FOLDER_SIZE_HUMAN})${NC}"
    echo "LOG: [${COUNTER}/${NEED_TRANSFER_COUNT}] 准备传输: ${REMOTE_DIR_NAME} (源: $dir, 大小: $FOLDER_SIZE_HUMAN)" | tee -a "$LOG_FILE"
    
    TRANSFER_START=$(date +%s)
    
    echo "INFO: 创建远程目录: ssh -n \"$REMOTE_HOST\" \"mkdir -p '$REMOTE_DIR'\"" | tee -a "$LOG_FILE"
    ssh -n "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'" >> "$LOG_FILE" 2>&1
    MKDIR_EXIT_CODE=$?
    echo "INFO: ssh mkdir 退出码: $MKDIR_EXIT_CODE" | tee -a "$LOG_FILE"
    if [ $MKDIR_EXIT_CODE -ne 0 ]; then
        echo -e "${RED}  ✗ 创建远程目录 $REMOTE_DIR 失败 (退出码: $MKDIR_EXIT_CODE)${NC}" | tee -a "$LOG_FILE"
        FAILED_TRANSFERS+=("${REMOTE_DIR_NAME} (mkdir failed)")
        echo "ERROR: 创建远程目录 $REMOTE_DIR 失败，跳过此文件夹。" >> "$LOG_FILE"
        continue
    fi
    
    echo "INFO: 开始 rsync 传输到 $REMOTE_DIR" | tee -a "$LOG_FILE"
    
    echo -e "${GREEN}  传输中...${NC}"
    
    rsync -avz \
        --progress \
        --include="*.npy" \
        --exclude="*" \
        "${dir}/" \
        "${REMOTE_HOST}:${REMOTE_DIR}/" 2>&1 | \
    while IFS= read -r line; do
        echo "$line" >> "$LOG_FILE"
        
        if [[ "$line" =~ ([0-9]+%) ]] || [[ "$line" =~ ([0-9]+\.[0-9]+[KMGT]B/s) ]]; then
            printf "\r${GREEN}  传输中: %s${NC}" "$(echo "$line" | sed 's/[[:space:]]\+/ /g' | cut -c 1-80)"
        fi
    done
    
    RSYNC_EXIT_CODE=${PIPESTATUS[0]}
    
    printf "\033[2K\r"
    
    TRANSFER_END=$(date +%s)
    TRANSFER_TIME=$((TRANSFER_END - TRANSFER_START))
    
    AVG_SPEED_MB="0"
    TIME_STR="${TRANSFER_TIME}秒"
    SPEED_STR="N/A"

    if [ "$FOLDER_SIZE" -gt 0 ] && [ $TRANSFER_TIME -gt 0 ]; then
        AVG_SPEED_MB=$(echo "scale=2; $FOLDER_SIZE / $TRANSFER_TIME / 1024 / 1024" | bc 2>/dev/null || echo "0")
        if [ $TRANSFER_TIME -gt 60 ]; then
            TIME_STR="$(($TRANSFER_TIME / 60))分$(($TRANSFER_TIME % 60))秒"
        fi
        SPEED_STR="${AVG_SPEED_MB} MB/s"
    elif [ "$FOLDER_SIZE" -eq 0 ]; then
         TIME_STR="N/A (0 size)"
         SPEED_STR="N/A"
    else
        TIME_STR="<1秒"
        if [ "$FOLDER_SIZE" -gt 0 ]; then
            SPEED_STR="非常快"
        fi
    fi
    
    echo "INFO: rsync 退出码: $RSYNC_EXIT_CODE for $REMOTE_DIR_NAME" | tee -a "$LOG_FILE"
    if [ $RSYNC_EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}  ✓ 完成 (用时: ${TIME_STR}, 速度: ${SPEED_STR})${NC}" | tee -a "$LOG_FILE"
        TRANSFERRED_SIZE=$((TRANSFERRED_SIZE + FOLDER_SIZE))
        echo "SUCCESS: ${REMOTE_DIR_NAME} (大小: ${FOLDER_SIZE_HUMAN}, 用时: ${TIME_STR}, 速度: ${SPEED_STR})" >> "$LOG_FILE"
    else
        echo -e "${RED}  ✗ 失败 (退出码: $RSYNC_EXIT_CODE)${NC}" | tee -a "$LOG_FILE"
        FAILED_TRANSFERS+=("${REMOTE_DIR_NAME} (rsync code: $RSYNC_EXIT_CODE)")
        echo "ERROR: ${REMOTE_DIR_NAME} 传输失败 (退出码: $RSYNC_EXIT_CODE)" >> "$LOG_FILE"
    fi
    
    echo ""
    
done < "$TRANSFER_LIST"

# 清理临时文件
echo "INFO: 清理临时文件" | tee -a "$LOG_FILE"
rm -f "$TEMP_FILE" "$REMOTE_EXISTING_FILE" "$TRANSFER_LIST"

# 计算总体统计
TOTAL_TIME=$(($(date +%s) - START_TIME))
SUCCESS_COUNT=$((COUNTER - ${#FAILED_TRANSFERS[@]}))

# 显示最终总结
SUMMARY_HEADER="\n${CYAN}═══════════════════════════════════════════${NC}"
SUMMARY_TITLE="${CYAN}                 传输完成总结                  ${NC}"
echo -e "$SUMMARY_HEADER" | tee -a "$LOG_FILE"
echo -e "$SUMMARY_TITLE" | tee -a "$LOG_FILE"
echo -e "$SUMMARY_HEADER" | tee -a "$LOG_FILE"

TRANSFERRED_SIZE_HUMAN=$(numfmt --to=iec-i --suffix=B "$TRANSFERRED_SIZE" 2>/dev/null || echo "$TRANSFERRED_SIZE bytes")
echo -e "${GREEN}成功传输: ${SUCCESS_COUNT}/${NEED_TRANSFER_COUNT} 个文件夹${NC}" | tee -a "$LOG_FILE"
echo -e "${MAGENTA}跳过已存在: ${SKIP_COUNT} 个文件夹${NC}" | tee -a "$LOG_FILE"
echo -e "${GREEN}总传输量: $TRANSFERRED_SIZE_HUMAN${NC}" | tee -a "$LOG_FILE"

TOTAL_TIME_FORMATTED=""
if [ $TOTAL_TIME -gt 3600 ]; then
    TOTAL_TIME_FORMATTED=$(printf "%02d时%02d分%02d秒" $((TOTAL_TIME/3600)) $((TOTAL_TIME%3600/60)) $((TOTAL_TIME%60)))
elif [ $TOTAL_TIME -gt 60 ]; then
    TOTAL_TIME_FORMATTED="$(($TOTAL_TIME/60))分$(($TOTAL_TIME%60))秒"
else
    TOTAL_TIME_FORMATTED="${TOTAL_TIME}秒"
fi
echo -e "${GREEN}总用时: ${TOTAL_TIME_FORMATTED}${NC}" | tee -a "$LOG_FILE"

if [ $TOTAL_TIME -gt 0 ] && [ $TRANSFERRED_SIZE -gt 0 ]; then
    AVG_SPEED_TOTAL=$(echo "scale=2; $TRANSFERRED_SIZE / $TOTAL_TIME / 1024 / 1024" | bc 2>/dev/null || echo "0")
    echo -e "${GREEN}平均速度: ${AVG_SPEED_TOTAL} MB/s${NC}" | tee -a "$LOG_FILE"
fi

if [ ${#FAILED_TRANSFERS[@]} -gt 0 ]; then
    echo -e "\n${RED}失败的传输 (${#FAILED_TRANSFERS[@]}个):${NC}" | tee -a "$LOG_FILE"
    for failed_item in "${FAILED_TRANSFERS[@]}"; do
        echo -e "${RED}- $failed_item${NC}" | tee -a "$LOG_FILE"
    done
    if [ ${#FAILED_TRANSFERS[@]} -gt 10 ]; then
        echo "..."
        echo "(详情请查看日志 $LOG_FILE)"
    fi
fi

echo -e "\n${BLUE}详细日志请查看: $LOG_FILE${NC}"

# 验证远程目录数量
echo -e "\n${BLUE}最终验证远程服务器状态...${NC}" | tee -a "$LOG_FILE"
FINAL_REMOTE_CMD="cd '$REMOTE_BASE' 2>/dev/null && find . -maxdepth 2 -name '*.npy' -type f | sed 's|^\./||' | awk -F'/' '{print \$1}' | sort | uniq -c | awk '\$1==9' | wc -l"
FINAL_REMOTE_COUNT=$(ssh -n "$REMOTE_HOST" "$FINAL_REMOTE_CMD" 2>/dev/null || echo "Error")

if [[ "$FINAL_REMOTE_COUNT" != "Error" ]]; then
    echo -e "${GREEN}远程服务器最终包含 $FINAL_REMOTE_COUNT 个完整的文件夹（各含9个npy文件）${NC}" | tee -a "$LOG_FILE"
    EXPECTED_COUNT=$((REMOTE_EXISTING_COUNT + SUCCESS_COUNT))
    if [ "$FINAL_REMOTE_COUNT" -eq "$EXPECTED_COUNT" ]; then
        echo -e "${GREEN}验证通过：远程文件夹数量符合预期（原有 $REMOTE_EXISTING_COUNT + 新增 $SUCCESS_COUNT = $EXPECTED_COUNT）${NC}" | tee -a "$LOG_FILE"
    else
        echo -e "${YELLOW}警告: 远程文件夹数量 ($FINAL_REMOTE_COUNT) 与预期 ($EXPECTED_COUNT) 不一致。请检查日志。${NC}" | tee -a "$LOG_FILE"
    fi
else
    echo -e "${RED}无法验证远程服务器最终状态。${NC}" | tee -a "$LOG_FILE"
fi

echo "Script finished at $(date)" | tee -a "$LOG_FILE"
echo -e "${GREEN}脚本执行完毕。${NC}"