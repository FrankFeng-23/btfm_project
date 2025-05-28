#!/usr/bin/env python3
"""
s2_fast_processor.py — Sentinel-2 L2A 快速下载 & ROI 拼接 (优化版)
更新：2025-05-21
支持灵活的并行分区处理，包含完善的错误处理、超时控制和内存优化
特性：SCL波段在质量评估时保留原始值，确保每个日期的SCL根据实际观测数据生成
优化：使用高效的矢量化操作代替逐像素处理，大幅提升 SCL 处理和波段生成性能
"""

from __future__ import annotations
import os, sys, argparse, logging, datetime, time, warnings, signal
from pathlib import Path
import multiprocessing
from contextlib import contextmanager
import concurrent.futures
import uuid
import tempfile
import shutil
import gc
import random

import numpy as np
import psutil, rasterio, xarray as xr, rioxarray
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds, reproject
import pystac_client, planetary_computer, stackstac

import dask
from dask.distributed import Client, LocalCluster, performance_report, wait

# ▶ distributed 版本兼容
try:
    from distributed.comm.core import CommClosedError
except ImportError:
    from distributed import CommClosedError

warnings.filterwarnings("ignore", category=RuntimeWarning, module="dask.core")
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*The array is being split into many small chunks.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in true_divide.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in log10.*")

# ─── 常量 ──────────────────────────────────────────────────────────────────────
BAND_MAPPING = {
    "B02": "blue", "B03": "green", "B04": "red",
    "B05": "rededge1", "B06": "rededge2", "B07": "rededge3",
    "B08": "nir", "B8A": "nir08",
    "B11": "swir16", "B12": "swir22",
    "SCL": "scl",
}
S2_BANDS        = list(BAND_MAPPING.keys())
BASELINE_CUTOFF = datetime.datetime(2022, 1, 25)
BASELINE_OFFSET = 1000

# SCL 无效值列表 (无云/无阴影/非水体等为有效)
SCL_INVALID = {0, 1, 2, 3, 8, 9, np.nan}

# SCL 值描述，用于日志
SCL_DESCRIPTIONS = {
    0: "无数据",
    1: "饱和或缺陷",
    2: "暗影区",
    3: "未分类",
    4: "植被",
    5: "裸露土壤",
    6: "水体",
    7: "未使用",
    8: "云",
    9: "薄云",
    10: "雪",
    11: "云阴影"
}

# 有效覆盖率阈值 (低于此值跳过处理)
MIN_VALID_COVERAGE = 5.0  # 百分比

# 临时文件目录设置（默认使用系统临时目录）
TEMP_DIR = os.getenv("TEMP_DIR", tempfile.gettempdir())

# 超时设置（秒）
PROCESS_TIMEOUT = 180 * 60  # 整体处理超时
DAY_TIMEOUT = 60 * 60       # 单日处理超时，增加到60分钟
ITEM_TIMEOUT = 25 * 60      # 单个item处理超时，增加到15分钟
BAND_TIMEOUT = 10 * 60      # 单个波段处理超时，增加到10分钟
SCL_BAND_TIMEOUT = 10 * 60   # SCL波段处理超时，保持10分钟
SCL_MAX_ATTEMPTS = 2        # SCL处理最大尝试次数

# 网络请求重试配置
MAX_RETRIES = 5
RETRY_BACKOFF_FACTOR = 1.5  # 指数回退因子

# 并行处理配置
DEFAULT_MAX_WORKERS = 5     # 默认处理Item的线程数，避免过多并发请求

# ─── 超时控制 ──────────────────────────────────────────────────────────────────
class TimeoutException(Exception):
    pass

@contextmanager
def timeout_handler(seconds):
    """超时上下文管理器"""
    def timeout_signal_handler(signum, frame):
        raise TimeoutException(f"操作超时 ({seconds}秒)")
    
    # 检查是否在主线程中（Unix信号只能在主线程中处理）
    import threading
    if threading.current_thread() is not threading.main_thread():
        # 如果不是主线程，只是yield而不设置信号
        yield
        return
    
    # 设置信号处理器
    old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # 恢复原信号处理器
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# ─── CLI ───────────────────────────────────────────────────────────────────────
def get_args():
    P = argparse.ArgumentParser("Fast Sentinel-2 L2A Processor (Optimized Parallel Edition)")
    P.add_argument("--input_tiff",   required=True, help="ROI 掩膜或者模板栅格")
    P.add_argument("--start_date",   required=True, help="开始日期 (YYYY-MM-DD[THH:MM:SS]) - 包含该时间点")
    P.add_argument("--end_date",     required=True, help="结束日期 (YYYY-MM-DD[THH:MM:SS]) - 包含该时间点")
    P.add_argument("--output",       default="sentinel2_output", help="输出目录")
    P.add_argument("--max_cloud",    type=float, default=90, help="最大云量百分比")
    P.add_argument("--dask_workers", type=int,   default=8, help="本分区的 Dask worker 数")
    P.add_argument("--worker_memory",type=int,   default=16, help="每个 worker 内存 GB")
    P.add_argument("--chunksize",    type=int,   default=1024, help="stackstac x/y chunk 大小")
    P.add_argument("--resolution",   type=int,   default=10, help="输出分辨率 (米)")
    P.add_argument("--overwrite",    action="store_true", help="覆盖已存在文件")
    P.add_argument("--debug",        action="store_true", help="输出调试日志")
    P.add_argument("--min_coverage", type=float, default=MIN_VALID_COVERAGE,
                   help="最小有效像素覆盖率 (百分比)")
    P.add_argument("--partition_id", default="unknown",
                   help="分区ID（用于日志标识）")
    P.add_argument("--temp_dir",     default=TEMP_DIR,
                   help="临时文件存储目录，默认使用系统临时目录")
    return P.parse_args()

# ─── logging ──────────────────────────────────────────────────────────────────
def setup_logging(debug: bool, out_dir: Path, partition_id: str):
    """设置日志，包含分区ID标识"""
    fmt = f"%(asctime)s [{partition_id}] [%(levelname)s] %(message)s"
    lvl = logging.DEBUG if debug else logging.INFO
    
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(lvl)
    
    # 清除现有的handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建formatter
    formatter = logging.Formatter(fmt)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（分区特定的日志文件）
    file_handler = logging.FileHandler(
        out_dir / f"s2_{partition_id}_detail.log", 
        "a", 
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def log_sys(partition_id: str):
    m = psutil.virtual_memory()
    logging.info(f"[{partition_id}] 系统信息 - CPU {os.cpu_count()} | "
                 f"RAM {m.total/1e9:.1f} GB (free {m.available/1e9:.1f} GB)")

def fmt_bbox(b):
    return f"{b[0]:.5f},{b[1]:.5f} ⇢ {b[2]:.5f},{b[3]:.5f}"

# ─── Dask ─────────────────────────────────────────────────────────────────────
def make_client(req_workers:int, req_mem:int, partition_id: str):
    """创建Dask客户端，使用分区特定的dashboard端口"""
    total_mem = psutil.virtual_memory().total / 1e9
    workers = min(req_workers, os.cpu_count(),
                  max(1, int(total_mem // (req_mem*1.2))))
    if workers < req_workers:
        logging.warning(f"⚠️  worker 数 {req_workers}→{workers} (资源限制)")
    
    # 为不同分区自动分配dashboard端口
    # 使用分区ID的哈希值来确定端口，确保相同ID总是使用相同端口
    # 将端口限制在8780-8899之间
    port_base = 8780
    port_range = 120
    dashboard_port = port_base + (hash(partition_id) % port_range)
    
    # 配置Dask性能优化选项
    dask_config = {
        "distributed.worker.memory.target": 0.75,  # 降低以减少内存压力
        "distributed.worker.memory.spill": 0.85,   # 降低以减少内存压力
        "distributed.worker.memory.pause": 0.95,
        "array.slicing.split_large_chunks": True,  # 优化大数组切片
        "optimization.fuse.active": True,         # 激活融合优化
        "optimization.fuse.ave-width": 4          # 加速dask图优化
    }
    
    dask.config.set(dask_config)
    
    cluster = LocalCluster(
        n_workers         = workers,
        threads_per_worker= 4,
        processes         = True,
        memory_limit      = f"{req_mem}GB",
        dashboard_address = f":{dashboard_port}",
        silence_logs      = "ERROR",
    )
    
    cli = Client(cluster, asynchronous=False)
    logging.info(f"[{partition_id}] Dask dashboard → {cli.dashboard_link}")
    return cli

# ─── ROI & 掩膜 ────────────────────────────────────────────────────────────────
def load_roi(tiff: Path, partition_id: str):
    """加载ROI数据，包含简化大掩膜的逻辑"""
    with rasterio.open(tiff) as src:
        tpl = dict(crs=src.crs,
                   transform=src.transform,
                   width=src.width,
                   height=src.height)
        bbox_proj = src.bounds
        bbox_ll   = transform_bounds(src.crs, "EPSG:4326", *bbox_proj,
                                     densify_pts=21)
        
        # 读取掩膜并转换为1位数据以节省内存
        mask_np = (src.read(1) > 0).astype(np.uint8)
        
    # 检查ROI大小，打印日志
    roi_size_mb = (mask_np.size * mask_np.itemsize) / (1024 * 1024)
    logging.info(f"[{partition_id}] ROI (CRS={tpl['crs']}): {tpl['width']}×{tpl['height']} ({roi_size_mb:.2f} MB)")
    logging.info(f"[{partition_id}] ROI bbox proj: {fmt_bbox(bbox_proj)}")
    logging.info(f"[{partition_id}] ROI bbox lon/lat: {fmt_bbox(bbox_ll)}")
    
    return tpl, bbox_proj, bbox_ll, mask_np

def mask_to_xr(mask_np, tpl):
    """将掩膜转换为xarray对象"""
    da = xr.DataArray(mask_np, dims=("y", "x"))
    return da.rio.write_crs(tpl["crs"]).rio.write_transform(tpl["transform"])

# ─── STAC ─────────────────────────────────────────────────────────────────────
def search_items(bbox_ll, date_range:str, max_cloud, partition_id: str):
    """
    搜索STAC项，增强了异常处理和重试逻辑
    """
    # 解析开始和结束时间
    start_date, end_date = date_range.split("/")
    
    # 解析结束时间并添加一秒，确保包含结束时间点
    try:
        end_dt = datetime.datetime.fromisoformat(end_date.replace('Z', '+00:00').replace(' ', 'T'))
        end_dt_plus = end_dt + datetime.timedelta(seconds=1)
        search_date_range = f"{start_date}/{end_dt_plus.isoformat()}"
    except ValueError:
        # 如果时间格式解析出错，使用原始日期范围
        logging.warning(f"[{partition_id}] 无法解析结束日期格式，使用原始范围: {date_range}")
        search_date_range = date_range
    
    logging.info(f"[{partition_id}] STAC 搜索日期范围: {search_date_range}")
    
    # 添加重试逻辑
    retries = 0
    max_retries = MAX_RETRIES
    retry_delay = 1  # 初始延迟1秒
    
    while retries <= max_retries:
        try:
            cat = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=planetary_computer.sign_inplace)
            q = cat.search(collections=["sentinel-2-l2a"],
                       bbox=bbox_ll, datetime=search_date_range,
                       query={"eo:cloud_cover": {"lt": max_cloud}})
            items = list(q.get_items())
            logging.info(f"[{partition_id}] STAC 命中 {len(items)} item (云 < {max_cloud}%)")
            if items:
                b = np.array([it.bbox for it in items])
                union = [b[:,0].min(), b[:,1].min(), b[:,2].max(), b[:,3].max()]
                logging.info(f"[{partition_id}] All item union lon/lat: {fmt_bbox(union)}")
            return items
        except Exception as e:
            retries += 1
            if retries > max_retries:
                logging.error(f"[{partition_id}] STAC搜索失败 (尝试{retries}/{max_retries+1}): {e}")
                raise
            
            # 计算指数回退延迟
            retry_delay = min(60, retry_delay * RETRY_BACKOFF_FACTOR)  # 最大延迟60秒
            jitter = random.uniform(0.8, 1.2)  # 添加随机抖动
            actual_delay = retry_delay * jitter
            
            logging.warning(f"[{partition_id}] STAC搜索失败 (尝试{retries}/{max_retries+1}): {e}, {actual_delay:.1f}秒后重试...")
            time.sleep(actual_delay)

def group_by_date(items, partition_id: str):
    """将items按日期分组"""
    g = {}
    for it in items:
        d = it.properties["datetime"][:10]
        g.setdefault(d, []).append(it)
    logging.info(f"[{partition_id}] ⇒ {len(g)} 观测日")
    return dict(sorted(g.items()))

# ─── baseline 校正 ─────────────────────────────────────────────────────────────
def harmonize_arr(arr: np.ndarray, date_key:str):
    """执行Baseline校正"""
    if datetime.datetime.strptime(date_key, "%Y-%m-%d") > BASELINE_CUTOFF:
        # 处理NaN值以避免警告
        valid_mask = ~np.isnan(arr) & (arr >= BASELINE_OFFSET)
        np.subtract(arr, BASELINE_OFFSET, out=arr, where=valid_mask)
    return arr

# ─── 内存检查 ──────────────────────────────────────────────────────────────────
def check_memory_requirements(shape, dtype=np.uint16):
    """检查数组内存需求是否合理"""
    try:
        # 计算所需内存（GB）
        element_size = np.dtype(dtype).itemsize
        total_elements = np.prod(shape)
        memory_gb = (total_elements * element_size) / (1024**3)
        
        logging.debug(f"计算内存需求: 形状{shape}, 类型{dtype}, 大小{memory_gb:.2f}GB")
        
        # 获取当前可用内存
        available_gb = psutil.virtual_memory().available / (1024**3)
        
        # 使用当前可用内存的50%作为阈值
        threshold_gb = min(available_gb * 0.5, 32)  # 不超过32GB
        
        if memory_gb > threshold_gb:
            logging.warning(f"⚠️  内存需求 {memory_gb:.2f}GB 超过可用阈值 {threshold_gb:.2f}GB，跳过处理")
            return False
        return True
    except (OverflowError, ValueError) as e:
        logging.warning(f"⚠️  内存计算错误: {e}，跳过处理")
        return False

# ─── GeoTIFF 写出 ──────────────────────────────────────────────────────────────
def write_tiff(np_arr, out_path: Path, tpl, dtype, metadata=None):
    """写出GeoTIFF，优化了压缩和配置，并添加元数据"""
    # 处理NaN值
    if np.isnan(np_arr).any():
        np_arr = np.nan_to_num(np_arr, nan=0)
        
    profile = dict(driver="GTiff", dtype=dtype, count=1,
                   width=tpl["width"], height=tpl["height"],
                   crs=tpl["crs"], transform=tpl["transform"],
                   compress="lzw", tiled=True,
                   blockxsize=256, blockysize=256,
                   nodata=0)
    
    # 写入文件
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np_arr.astype(dtype, copy=False), 1)
        
        # 添加元数据
        if metadata:
            dst.update_tags(**metadata)

# ─── 验证TIFF ─────────────────────────────────────────────────────────────────
def validate_tiff(file_path, expected_shape, expected_crs, expected_transform):
    """验证TIFF文件是否有效"""
    try:
        with rasterio.open(file_path) as src:
            # 检查基本属性
            if src.shape != expected_shape:
                logging.warning(f"验证失败: {file_path} 形状不匹配. 预期 {expected_shape}, 得到 {src.shape}")
                return False
            
            if src.crs != expected_crs:
                logging.warning(f"验证失败: {file_path} CRS不匹配. 预期 {expected_crs}, 得到 {src.crs}")
                return False
            
            # 检查数据存在性（通过统计数据，避免读取整个数组）
            stats = [src.statistics(i) for i in range(1, src.count + 1)]
            if any(s.max == 0 and s.min == 0 for s in stats):
                logging.warning(f"验证失败: {file_path} 波段全为零")
                return False
            
            # 检查文件大小
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            logging.debug(f"TIFF验证通过: {file_path}，形状={src.shape}, 大小={file_size_mb:.2f}MB")
            return True
            
    except Exception as e:
        logging.error(f"验证TIFF {file_path} 时出错: {e}")
        return False

# ─── SCL 质量评估 ─────────────────────────────────────────────────────────────
def is_valid_scl(scl_arr):
    """判断SCL值是否为有效观测(非云/阴影/水体等)"""
    # 向量化操作：将nan填充为0，然后检查是否在无效值列表中
    return ~np.isin(np.nan_to_num(scl_arr, nan=0), list(SCL_INVALID - {np.nan}))

def process_scl(scl_arr, roi_mask, partition_id="unknown"):
    """
    处理SCL数据，高度优化的矢量化版本
    返回：
        valid_mask: 有效性掩膜（布尔值数组）
        tile_selection: 选择的tile索引数组（用于后续波段处理）
        valid_pct: 有效覆盖率百分比
    """
    # 获取数组形状并处理单tile和多tile情况
    if len(scl_arr.shape) == 3:
        n_tiles, scl_height, scl_width = scl_arr.shape
    else:
        scl_height, scl_width = scl_arr.shape
        n_tiles = 1
        scl_arr = scl_arr.reshape(1, scl_height, scl_width)
    
    # 确保ROI掩膜是布尔类型
    roi_mask = roi_mask.astype(bool)
    roi_height, roi_width = roi_mask.shape
    
    # 检查形状是否匹配，如果不匹配，调整SCL数据
    if scl_height != roi_height or scl_width != roi_width:
        logging.warning(f"[{partition_id}] 检测到形状不匹配: SCL形状{(scl_height, scl_width)}与ROI形状{(roi_height, roi_width)}不同")
        
        # 确定要使用的最终形状（取较小值以避免索引越界）
        use_height = min(scl_height, roi_height)
        use_width = min(scl_width, roi_width)
        
        logging.info(f"[{partition_id}] 使用共同区域: {use_height}x{use_width}")
        
        # 裁剪数据到共同区域
        scl_arr = scl_arr[:, :use_height, :use_width]
        roi_mask = roi_mask[:use_height, :use_width]
    
    # 为每个tile计算有效掩膜 - 向量化操作
    valid_mask = is_valid_scl(scl_arr)
    
    # 计算有效像素位置和ROI像素数量
    roi_pixel_count = np.sum(roi_mask)
    
    # 创建tile选择数组，初始化为-1（无有效tile）
    tile_selection = np.full(roi_mask.shape, -1, dtype=np.int8)
    
    # 优化的向量化选择第一个有效tile的算法
    # 1. 创建一个布尔掩码，表示哪些像素已经被分配了一个有效tile
    assigned = np.zeros(roi_mask.shape, dtype=bool)
    
    # 2. 对每个tile进行循环（这个循环无法完全避免，因为我们需要按顺序选择第一个有效tile）
    for tile_idx in range(n_tiles):
        # 2.1 找出当前tile中有效的像素并且在ROI内且尚未被分配的像素
        # valid_mask[tile_idx] 是当前tile的有效掩码
        # roi_mask 是ROI区域掩码
        # ~assigned 是尚未被分配tile的像素掩码
        current_valid = valid_mask[tile_idx] & roi_mask & ~assigned
        
        # 2.2 将当前有效像素的tile_idx分配给tile_selection
        tile_selection[current_valid] = tile_idx
        
        # 2.3 更新已分配标记
        assigned |= current_valid
    
    # 计算有效覆盖率 - 已经分配了valid tile的像素数量除以ROI总像素数量
    valid_pixel_count = np.sum(tile_selection >= 0)
    valid_pct = 100.0 * valid_pixel_count / roi_pixel_count if roi_pixel_count > 0 else 0.0
    
    logging.info(f"[{partition_id}] SCL处理结果: ROI内有效像素 {valid_pixel_count}/{roi_pixel_count}, 覆盖率 {valid_pct:.2f}%")
    
    return valid_mask, tile_selection, valid_pct

def create_scl_mosaic(scl_arr, tile_selection, roi_mask, target_shape, date_key=None, partition_id="unknown"):
    """
    基于tile选择结果创建SCL镶嵌，保留原始SCL值 - 矢量化版本
    """
    try:
        # 获取形状信息
        if len(scl_arr.shape) == 3:
            n_tiles, arr_height, arr_width = scl_arr.shape
        else:
            arr_height, arr_width = scl_arr.shape
            n_tiles = 1
            scl_arr = scl_arr.reshape(1, arr_height, arr_width)
        
        target_height, target_width = target_shape
        
        # 创建目标大小的结果数组（初始值为0，表示无数据）
        result = np.zeros(target_shape, dtype=np.uint8)
        
        # 确定共同区域大小
        common_height = min(arr_height, roi_mask.shape[0], target_height, tile_selection.shape[0])
        common_width = min(arr_width, roi_mask.shape[1], target_width, tile_selection.shape[1])
        
        # 裁剪到共同区域
        roi_crop = roi_mask[:common_height, :common_width]
        tile_sel_crop = tile_selection[:common_height, :common_width]
        
        # 矢量化实现：使用ROI范围内的有效tile索引来选择SCL值
        # 1. 创建一个有效ROI掩码，表示哪些像素需要处理
        valid_roi = (roi_crop & (tile_sel_crop >= 0))
        
        if np.any(valid_roi):
            # 2. 获取有效ROI的坐标
            y_coords, x_coords = np.where(valid_roi)
            
            # 3. 获取这些坐标对应的tile索引
            tile_indices = tile_sel_crop[y_coords, x_coords]
            
            # 4. 创建一个映射，用于从scl_arr中提取对应的值
            result[y_coords, x_coords] = scl_arr[tile_indices, y_coords, x_coords]
        
        # 统计SCL值分布
        unique_values, unique_counts = np.unique(result, return_counts=True)
        value_counts = dict(zip(unique_values, unique_counts))
        
        # 记录SCL值分布到日志
        if date_key:
            total_pixels = np.sum(roi_mask)
            logging.info(f"[{partition_id}] SCL值分布统计 ({date_key}):")
            for val in sorted(value_counts.keys()):
                count = value_counts[val]
                desc = SCL_DESCRIPTIONS.get(val, "未知")
                percent = 100 * count / total_pixels if total_pixels > 0 else 0
                logging.info(f"[{partition_id}]   SCL值 {val} ({desc}): {count} 像素 ({percent:.2f}%)")
        
        return result
    except Exception as e:
        logging.error(f"[{partition_id}] SCL镶嵌创建失败: {e}")
        if logging.getLogger().level <= logging.DEBUG:
            import traceback
            logging.debug(traceback.format_exc())
        raise

# ─── 智能镶嵌 ──────────────────────────────────────────────────────────────────
def smart_mosaic(data_arr, tile_selection, roi_mask, partition_id="unknown"):
    """
    基于tile_selection的智能镶嵌，使用SCL处理中确定的最佳tile - 矢量化版本
    """
    try:
        # 单tile情况直接返回
        if len(data_arr.shape) < 3 or data_arr.shape[0] == 1:
            result = data_arr[0] if len(data_arr.shape) == 3 else data_arr
            
            # 检查形状是否匹配
            if result.shape != roi_mask.shape:
                logging.debug(f"[{partition_id}] 单tile镶嵌形状不匹配: 数据{result.shape}, ROI掩膜{roi_mask.shape}")
                
                # 确定共同的大小
                common_height = min(result.shape[0], roi_mask.shape[0])
                common_width = min(result.shape[1], roi_mask.shape[1])
                
                # 创建目标大小的结果数组
                final_result = np.zeros(roi_mask.shape, dtype=result.dtype)
                
                # 裁剪数据和掩膜
                result_cropped = result[:common_height, :common_width] 
                roi_mask_cropped = roi_mask[:common_height, :common_width]
                
                # 应用ROI掩膜 - 向量化操作
                final_result[:common_height, :common_width] = result_cropped * roi_mask_cropped
                return final_result
            
            # 应用ROI掩膜 - 向量化操作
            return result * roi_mask
        
        # 获取尺寸
        n_tiles, data_height, data_width = data_arr.shape
        
        # 确保ROI掩膜是布尔型
        roi_mask = roi_mask.astype(bool)
        
        # 创建输出数组（使用原始ROI大小）
        result = np.zeros(roi_mask.shape, dtype=data_arr.dtype)
        
        # 确定共同区域大小
        common_height = min(data_height, roi_mask.shape[0], tile_selection.shape[0])
        common_width = min(data_width, roi_mask.shape[1], tile_selection.shape[1])
        
        # 裁剪到共同区域
        roi_crop = roi_mask[:common_height, :common_width]
        tile_sel_crop = tile_selection[:common_height, :common_width]
        
        # 矢量化实现：使用ROI范围内的有效tile索引来选择数据值
        # 1. 创建一个有效ROI掩码，表示哪些像素需要处理
        valid_roi = (roi_crop & (tile_sel_crop >= 0))
        
        if np.any(valid_roi):
            # 2. 获取有效ROI的坐标
            y_coords, x_coords = np.where(valid_roi)
            
            # 3. 获取这些坐标对应的tile索引
            tile_indices = tile_sel_crop[y_coords, x_coords]
            
            # 4. 从data_arr中提取对应的值
            result[y_coords, x_coords] = data_arr[tile_indices, y_coords, x_coords]
        
        # 对于没有有效tile的ROI像素，使用随机选择的逻辑
        # 1. 创建一个掩码，表示ROI内但没有有效tile的像素
        invalid_roi = (roi_crop & (tile_sel_crop < 0))
        
        if np.any(invalid_roi):
            # 2. 获取这些坐标
            y_coords_invalid, x_coords_invalid = np.where(invalid_roi)
            
            # 3. 为每个无效像素随机选择一个tile
            random_tiles = np.random.randint(0, n_tiles, size=len(y_coords_invalid))
            
            # 4. 使用随机选择的tile填充这些像素
            result[y_coords_invalid, x_coords_invalid] = data_arr[random_tiles, y_coords_invalid, x_coords_invalid]
        
        return result
    except Exception as e:
        logging.error(f"[{partition_id}] 智能镶嵌失败: {e}")
        if logging.getLogger().level <= logging.DEBUG:
            import traceback
            logging.debug(traceback.format_exc())
        raise

# ─── 处理单个波段 ───────────────────────────────────────────────────────────
def process_band(items, band_name, date_key, tpl, bbox_proj, mask_np, tile_selection,
                res, chunksize, out_path, partition_id="unknown", retries=3):
    """处理单个波段，支持重试和错误处理，处理形状不匹配问题，跳过SCL波段"""
    t0 = time.time()
    
    # SCL波段已经在质量评估阶段处理过了，直接跳过
    if band_name == "SCL":
        logging.info(f"[{partition_id}]     波段 {band_name} 已在质量评估阶段处理，跳过")
        return True
    
    logging.info(f"[{partition_id}]     处理波段 {band_name}")
    
    # 检查输出路径是否已存在
    if out_path.exists():
        if validate_tiff(out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
            logging.info(f"[{partition_id}]     {band_name} 已存在有效文件，跳过")
            return True
        else:
            logging.warning(f"[{partition_id}]     {band_name} 文件存在但无效，重新处理")
            out_path.unlink()
    
    # 重试循环
    for attempt in range(retries + 1):
        try:
            with timeout_handler(BAND_TIMEOUT):
                # 使用stackstac.stack加载单个波段
                assets = [band_name]
                
                da = stackstac.stack(
                    items=items,
                    assets=assets,
                    resolution=res,
                    epsg=tpl["crs"].to_epsg(),
                    bounds=bbox_proj,
                    chunksize=chunksize,
                    rescale=False,
                    resampling=Resampling.nearest
                )
                
                # 压平无用维度，但保留多item维度
                item_dim = None
                for dim in da.dims:
                    if dim not in ('band', 'x', 'y'):
                        if da.sizes[dim] > 1:
                            item_dim = dim  # 找到多item维度
                        elif da.sizes[dim] == 1:
                            da = da.squeeze(dim, drop=True)
                
                # 提取波段数据
                band_da = da.sel(band=band_name)
                
                # 转为numpy数组进行处理
                if item_dim:
                    # 多个item情况
                    band_arr = band_da.values
                    
                    # 记录并输出形状信息用于调试
                    if logging.getLogger().level <= logging.DEBUG:
                        logging.debug(f"[{partition_id}]     {band_name} 数组形状: {band_arr.shape}, ROI形状: {mask_np.shape}")
                    
                    # 检查数组大小是否合理
                    if not check_memory_requirements(band_arr.shape, band_arr.dtype):
                        logging.warning(f"[{partition_id}]     {band_name} 数组过大，跳过")
                        return False
                    
                    # 应用基于SCL选择的智能镶嵌 - 向量化版本
                    if tile_selection is not None:
                        arr = smart_mosaic(band_arr, tile_selection, mask_np, partition_id)
                    else:
                        # 如果没有有效的tile_selection，使用随机选择
                        logging.warning(f"[{partition_id}]     没有有效的tile_selection，使用随机tile选择")
                        n_tiles = band_arr.shape[0]
                        
                        # 创建随机选择矩阵 - 向量化实现
                        random_selection = np.random.randint(0, n_tiles, size=mask_np.shape)
                        arr = np.zeros(mask_np.shape, dtype=band_arr.dtype)
                        
                        # 计算共同区域
                        common_height = min(mask_np.shape[0], band_arr.shape[1])
                        common_width = min(mask_np.shape[1], band_arr.shape[2])
                        
                        # 创建索引矩阵
                        for i in range(n_tiles):
                            # 找出应该使用当前tile的像素
                            use_tile_i = (random_selection[:common_height, :common_width] == i) & (mask_np[:common_height, :common_width] > 0)
                            if np.any(use_tile_i):
                                # 直接使用布尔索引赋值 - 向量化操作
                                arr[:common_height, :common_width][use_tile_i] = band_arr[i, :common_height, :common_width][use_tile_i]
                    
                    # 检查输出形状是否匹配目标
                    if arr.shape != (tpl["height"], tpl["width"]):
                        logging.warning(f"[{partition_id}]     镶嵌结果形状 {arr.shape} 与目标形状 {(tpl['height'], tpl['width'])} 不匹配，进行调整")
                        # 创建目标尺寸的数组
                        full_arr = np.zeros((tpl["height"], tpl["width"]), dtype=arr.dtype)
                        # 复制共同区域
                        h = min(arr.shape[0], tpl["height"])
                        w = min(arr.shape[1], tpl["width"])
                        full_arr[:h, :w] = arr[:h, :w]
                        arr = full_arr
                    
                    # 应用基线校正
                    harmonize_arr(arr, date_key)
                    
                else:
                    # 单个item情况
                    band_arr = band_da.values
                    
                    # 记录并输出形状信息用于调试
                    if logging.getLogger().level <= logging.DEBUG:
                        logging.debug(f"[{partition_id}]     {band_name} 数组形状: {band_arr.shape}, ROI形状: {mask_np.shape}")
                    
                    # 检查数组大小是否合理
                    if not check_memory_requirements(band_arr.shape, band_arr.dtype):
                        logging.warning(f"[{partition_id}]     {band_name} 数组过大，跳过")
                        return False
                    
                    # 检查形状是否匹配
                    if band_arr.shape != mask_np.shape:
                        logging.warning(f"[{partition_id}]     数据形状 {band_arr.shape} 与ROI形状 {mask_np.shape} 不匹配，进行调整")
                        # 创建目标尺寸的数组
                        arr = np.zeros((tpl["height"], tpl["width"]), dtype=band_arr.dtype)
                        # 复制共同区域
                        h = min(band_arr.shape[0], tpl["height"])
                        w = min(band_arr.shape[1], tpl["width"])
                        # 应用掩膜到共同区域 - 向量化操作
                        mask_crop = mask_np[:h, :w]
                        arr[:h, :w] = band_arr[:h, :w] * mask_crop
                    else:
                        # 应用掩膜 - 向量化操作
                        arr = band_arr * mask_np
                    
                    # 应用基线校正
                    harmonize_arr(arr, date_key)
                
                # 创建元数据
                metadata = {
                    "TIFFTAG_DATETIME": datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                    "DATE_ACQUIRED": date_key,
                    "BAND_NAME": band_name,
                    "ITEMS_COUNT": len(items)
                }
                
                # 写出GeoTIFF
                dtype = "uint16"  # SCL已经被跳过了，这里只处理其他波段
                write_tiff(arr, out_path, tpl, dtype, metadata)
                
                # 验证输出文件
                if not validate_tiff(out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                    logging.error(f"[{partition_id}]     ✗ 波段 {band_name} 验证失败")
                    if out_path.exists():
                        out_path.unlink()
                    continue
                
                logging.info(f"[{partition_id}]     ✓ {band_name:9s}  "
                            f"{os.path.getsize(out_path)/1e6:.2f} MB, 用时 {time.time()-t0:.1f}s")
                
                # 成功完成
                return True
                
        except TimeoutException as e:
            if attempt < retries:
                retry_delay = min(30, (attempt + 1) * 2)  # 重试延迟，最长30秒
                logging.warning(f"[{partition_id}]     波段 {band_name} 处理超时，{retry_delay}秒后重试 ({attempt+1}/{retries})")
                time.sleep(retry_delay)
            else:
                logging.error(f"[{partition_id}]     ✗ 波段 {band_name} 处理超时: {e}")
                return False
                
        except Exception as e:
            if attempt < retries:
                retry_delay = min(30, (attempt + 1) * 2)  # 重试延迟，最长30秒
                logging.warning(f"[{partition_id}]     波段 {band_name} 处理错误: {e}, {retry_delay}秒后重试 ({attempt+1}/{retries})")
                time.sleep(retry_delay)
            else:
                logging.error(f"[{partition_id}]     ✗ 波段 {band_name} 处理失败: {e}")
                return False
    
    # 所有重试都失败
    return False

# ─── 处理SCL评估和生成波段 ───────────────────────────────────────────────────────────
def process_scl_assessment_and_generation(items, date_key, tpl, bbox_proj, mask_np, res, chunksize,
                                       min_coverage, out_root, overwrite, partition_id="unknown"):
    """
    处理SCL波段，进行质量评估并生成带有实际SCL值的输出文件 - 优化向量化版本
    """
    t0 = time.time()
    logging.info(f"[{partition_id}]   处理SCL波段进行质量评估并生成SCL输出文件")
    
    # 构建SCL输出路径
    scl_out_name = BAND_MAPPING["SCL"]
    scl_dir = out_root / scl_out_name
    scl_dir.mkdir(parents=True, exist_ok=True)
    scl_out_path = scl_dir / f"{date_key}_mosaic.tiff"
    
    # 检查是否已存在有效的SCL文件
    if not overwrite and scl_out_path.exists():
        if validate_tiff(scl_out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
            logging.info(f"[{partition_id}]   SCL文件已存在且有效，读取文件检查有效性...")
            
            # 读取现有SCL文件，分析其内容
            try:
                with rasterio.open(scl_out_path) as src:
                    scl_data = src.read(1)
                    # 检查SCL值的分布 - 向量化操作
                    unique_values, unique_counts = np.unique(scl_data, return_counts=True)
                    value_counts = dict(zip(unique_values, unique_counts))
                    
                    # 记录SCL值分布到日志
                    total_pixels = np.sum(mask_np)
                    logging.info(f"[{partition_id}]   现有SCL文件值分布统计 ({date_key}):")
                    for val in sorted(value_counts.keys()):
                        count = value_counts[val]
                        desc = SCL_DESCRIPTIONS.get(val, "未知")
                        percent = 100 * count / total_pixels if total_pixels > 0 else 0
                        logging.info(f"[{partition_id}]     SCL值 {val} ({desc}): {count} 像素 ({percent:.2f}%)")
                    
                    # 检查是否只包含0和4（表明是旧的处理方式）- 向量化检查
                    if set(unique_values) == {0, 4} and unique_values.size == 2:
                        logging.warning(f"[{partition_id}]   检测到旧版本SCL文件格式（仅包含0和4），将重新生成")
                        scl_out_path.unlink()
                    else:
                        # 计算有效覆盖率（非SCL_INVALID的像素）- 向量化计算
                        invalid_mask = np.isin(scl_data, list(SCL_INVALID - {np.nan}))
                        valid_pixels = np.sum(~invalid_mask & (scl_data > 0))
                        
                        valid_pct = 100 * valid_pixels / total_pixels if total_pixels > 0 else 0
                        logging.info(f"[{partition_id}]   现有SCL文件有效覆盖率: {valid_pct:.2f}%")
                        
                        # 如果覆盖率足够，返回成功
                        if valid_pct >= min_coverage:
                            # 生成通用的tile_selection用于其他波段
                            # 对于已存在的SCL文件，我们假设所有非0且非无效值的像素都有一个有效的tile
                            tile_selection = np.where(~np.isin(scl_data, list(SCL_INVALID)) & (scl_data > 0), 0, -1)
                            return True, valid_pct, tile_selection
                        else:
                            logging.warning(f"[{partition_id}]   现有SCL文件覆盖率不足 ({valid_pct:.2f}% < {min_coverage}%)，将重新生成")
                            scl_out_path.unlink()
            except Exception as e:
                logging.warning(f"[{partition_id}]   分析现有SCL文件时出错: {e}，将重新生成")
                scl_out_path.unlink()
    
    # 检查items中是否包含SCL资产
    if not all('SCL' in item.assets for item in items):
        logging.warning(f"[{partition_id}]   部分item中缺少SCL资产，尝试仅使用可用的SCL")
        # 过滤出有SCL资产的items
        scl_items = [item for item in items if 'SCL' in item.assets]
        if not scl_items:
            logging.warning(f"[{partition_id}]   所有item均缺少SCL资产，无法进行质量评估！")
            # 返回失败结果
            return False, 0.0, None
        items = scl_items
    
    try:
        with timeout_handler(SCL_BAND_TIMEOUT):
            # 使用stackstac.stack加载SCL波段
            da = stackstac.stack(
                items=items,
                assets=['SCL'],
                resolution=res,
                epsg=tpl["crs"].to_epsg(),
                bounds=bbox_proj,
                chunksize=chunksize,
                rescale=False,
                resampling=Resampling.nearest
            )
            
            # 压平无用维度，但保留多item维度
            item_dim = None
            for dim in da.dims:
                if dim not in ('band', 'x', 'y'):
                    if da.sizes[dim] > 1:
                        item_dim = dim  # 找到多item维度
                    elif da.sizes[dim] == 1:
                        da = da.squeeze(dim, drop=True)
            
            # 提取SCL数据
            scl_da = da.sel(band='SCL')
            
            # 获取numpy数组
            scl_arr = scl_da.values
            
            # 检查数组大小是否合理
            if not check_memory_requirements(scl_arr.shape, scl_arr.dtype):
                logging.warning(f"[{partition_id}]   SCL数组过大，无法进行质量评估")
                return False, 0.0, None
            
            # 进行SCL处理 - 向量化版本
            valid_mask, tile_selection, valid_pct = process_scl(scl_arr, mask_np, partition_id)
            
            # 检查有效覆盖率是否达到阈值
            if valid_pct < min_coverage:
                logging.warning(f"[{partition_id}]   ⚠️ {date_key} 有效覆盖率 {valid_pct:.2f}% < {min_coverage}%，跳过SCL生成")
                return False, valid_pct, None  # 返回覆盖率不足的结果，但不生成文件
            
            # 基于valid_mask和tile_selection创建SCL镶嵌输出，保留原始SCL值 - 向量化版本
            scl_output = create_scl_mosaic(scl_arr, tile_selection, mask_np, (tpl["height"], tpl["width"]), date_key, partition_id)
            
            # 创建元数据
            metadata = {
                "TIFFTAG_DATETIME": datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
                "DATE_ACQUIRED": date_key,
                "BAND_NAME": "SCL",
                "ITEMS_COUNT": len(items),
                "VALID_COVERAGE_PCT": f"{valid_pct:.2f}"
            }
            
            # 写出SCL GeoTIFF
            write_tiff(scl_output, scl_out_path, tpl, "uint8", metadata)
            
            # 验证输出文件
            if not validate_tiff(scl_out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                logging.error(f"[{partition_id}]   ✗ SCL文件验证失败")
                if scl_out_path.exists():
                    scl_out_path.unlink()
                return False, 0.0, None
            
            # 最后再次分析生成的SCL文件
            try:
                with rasterio.open(scl_out_path) as src:
                    scl_data = src.read(1)
                    # 检查SCL值的分布 - 向量化操作
                    unique_values, unique_counts = np.unique(scl_data, return_counts=True)
                    value_counts = dict(zip(unique_values, unique_counts))
                    
                    # 记录SCL值分布到日志
                    total_pixels = np.sum(mask_np)
                    logging.info(f"[{partition_id}]   生成的SCL文件值分布统计 ({date_key}):")
                    for val in sorted(value_counts.keys()):
                        count = value_counts[val]
                        desc = SCL_DESCRIPTIONS.get(val, "未知")
                        percent = 100 * count / total_pixels if total_pixels > 0 else 0
                        logging.info(f"[{partition_id}]     SCL值 {val} ({desc}): {count} 像素 ({percent:.2f}%)")
                    
                    scl_stats = src.statistics(1)
                    logging.info(f"[{partition_id}]   SCL文件统计: 最小值={scl_stats.min}, 最大值={scl_stats.max}, 均值={scl_stats.mean:.2f}")
            except Exception as e:
                logging.warning(f"[{partition_id}]   无法读取SCL文件统计: {e}")
            
            logging.info(f"[{partition_id}]   ✓ SCL 质量评估和文件生成完成，有效率: {valid_pct:.2f}%, "
                        f"文件大小: {os.path.getsize(scl_out_path)/1e6:.2f} MB, 用时 {time.time()-t0:.1f}s")
            
            return True, valid_pct, tile_selection
            
    except TimeoutException as e:
        logging.error(f"[{partition_id}]   ✗ SCL处理超时: {e}")
        return False, 0.0, None
            
    except Exception as e:
        logging.error(f"[{partition_id}]   ✗ SCL处理失败: {e}")
        if logging.getLogger().level <= logging.DEBUG:
            import traceback
            logging.debug(traceback.format_exc())
        return False, 0.0, None

# ─── 单日任务 ──────────────────────────────────────────────────────────────────
def process_day(date_key:str, items, tpl, bbox_proj, mask_np,
                out_root:Path, res:int, chunksize:int,
                overwrite:bool, min_coverage:float=5.0,
                partition_id:str="unknown") -> bool:
    """处理单日数据，优化了SCL处理逻辑，保留原始SCL值"""
    logging.info(f"[{partition_id}] → {date_key} (item={len(items)})")
    t0 = time.time()
    
    try:
        # 使用60分钟超时控制单日处理
        with timeout_handler(DAY_TIMEOUT):
            # 创建波段输出目录
            for outname in BAND_MAPPING.values():
                band_dir = out_root / outname
                band_dir.mkdir(parents=True, exist_ok=True)
            
            # 检查是否已全部处理完成
            if not overwrite:
                all_exist = True
                for band_name in S2_BANDS:
                    out_name = BAND_MAPPING[band_name]
                    out_path = out_root / out_name / f"{date_key}_mosaic.tiff"
                    if not out_path.exists() or not validate_tiff(out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                        all_exist = False
                        break
                
                if all_exist:
                    logging.info(f"[{partition_id}]   所有波段已存在有效文件，跳过")
                    return True
            
            # 处理SCL波段，进行质量评估并生成SCL文件，同时保留tile_selection用于其他波段
            scl_success, valid_pct, tile_selection = process_scl_assessment_and_generation(
                items, date_key, tpl, bbox_proj, mask_np, res, chunksize,
                min_coverage, out_root, overwrite, partition_id
            )
            
            # 检查SCL处理是否成功
            if not scl_success:
                if valid_pct < min_coverage:
                    logging.warning(f"[{partition_id}]   {date_key} 有效覆盖率 {valid_pct:.2f}% < {min_coverage}%，跳过其他波段处理")
                    return True  # 覆盖率不足不算失败，只是跳过
                else:
                    logging.error(f"[{partition_id}]   {date_key} SCL处理失败，跳过该日期处理")
                    return False
            
            # 创建临时目录用于处理
            day_temp_dir = tempfile.mkdtemp(prefix=f"s2_{date_key}_", dir=TEMP_DIR)
            logging.debug(f"[{partition_id}]   临时目录: {day_temp_dir}")
            
            try:
                # 使用线程池并行处理波段（不包括SCL）
                other_bands = [band for band in S2_BANDS if band != "SCL"]
                max_workers = min(DEFAULT_MAX_WORKERS, os.cpu_count())
                logging.info(f"[{partition_id}]   使用 {max_workers} 个线程并行处理 {len(other_bands)} 个波段（不包括SCL）")
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # 提交所有任务
                    futures = {}
                    
                    for band_name in other_bands:
                        out_name = BAND_MAPPING[band_name]
                        out_path = out_root / out_name / f"{date_key}_mosaic.tiff"
                        
                        # 如果文件已存在且有效，跳过
                        if not overwrite and out_path.exists() and validate_tiff(out_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                            logging.info(f"[{partition_id}]     波段 {band_name} 已存在有效文件，跳过")
                            continue
                        
                        # 创建临时输出路径
                        temp_path = Path(day_temp_dir) / f"{band_name}_{date_key}.tiff"
                        
                        # 提交任务
                        future = executor.submit(
                            process_band, 
                            items, band_name, date_key, tpl, bbox_proj, mask_np, tile_selection,
                            res, chunksize, temp_path, partition_id
                        )
                        futures[future] = (band_name, out_path, temp_path)
                    
                    # 处理结果
                    success_count = 0
                    for future in concurrent.futures.as_completed(futures):
                        band_name, out_path, temp_path = futures[future]
                        try:
                            success = future.result()
                            if success:
                                # 检查临时文件是否有效
                                if temp_path.exists() and validate_tiff(temp_path, (tpl["height"], tpl["width"]), tpl["crs"], tpl["transform"]):
                                    # 将临时文件移动到最终位置
                                    shutil.copy2(temp_path, out_path)
                                    success_count += 1
                                    logging.info(f"[{partition_id}]     ✓ {band_name} 处理完成")
                                else:
                                    logging.error(f"[{partition_id}]     ✗ {band_name} 临时文件无效或不存在")
                            else:
                                logging.warning(f"[{partition_id}]     ✗ {band_name} 处理失败")
                        except Exception as e:
                            logging.error(f"[{partition_id}]     ✗ {band_name} 处理异常: {e}")
            finally:
                # 清理当日临时目录
                try:
                    shutil.rmtree(day_temp_dir)
                    logging.debug(f"[{partition_id}]   已清理临时目录: {day_temp_dir}")
                except Exception as e:
                    logging.warning(f"[{partition_id}]   清理临时目录失败: {e}")
            
            # 记录处理结果
            total_other_bands = len(other_bands)
            total_bands = len(S2_BANDS)  # 包括SCL
            proc_time = time.time() - t0
            
            # 计算总成功数：SCL成功（1个）+ 其他波段成功数
            total_success = (1 if scl_success else 0) + success_count
            
            if total_success == total_bands:
                logging.info(f"[{partition_id}] ← {date_key} 全部波段处理成功 ({total_success}/{total_bands})，用时 {proc_time:.1f}s")
                return True
            elif total_success > 0:
                logging.warning(f"[{partition_id}] ← {date_key} 部分波段处理成功 ({total_success}/{total_bands})，用时 {proc_time:.1f}s")
                return True  # 部分成功也算成功
            else:
                logging.error(f"[{partition_id}] ← {date_key} 所有波段处理失败，用时 {proc_time:.1f}s")
                return False
                
    except TimeoutException as e:
        proc_time = time.time() - t0
        logging.error(f"[{partition_id}] ‼️  {date_key} 处理超时 ({proc_time:.1f}s): {e}")
        return False
    except Exception as e:
        proc_time = time.time() - t0
        logging.error(f"[{partition_id}] ‼️  {date_key} 处理失败: {type(e).__name__} - {e}")
        if logging.getLogger().level <= logging.DEBUG:
            import traceback
            logging.debug(traceback.format_exc())
        return False

# ─── 主程序 ───────────────────────────────────────────────────────────────────
def main():
    a = get_args()
    out_dir = Path(a.output).resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    # 使用命令行指定的临时目录
    global TEMP_DIR
    TEMP_DIR = a.temp_dir
    
    setup_logging(a.debug, out_dir, a.partition_id)
    logging.info(f"[{a.partition_id}] ⚡ S2 Fast Processor 启动 (优化并行版本 - SCL原始值保留版)"); 
    log_sys(a.partition_id)
    logging.info(f"[{a.partition_id}] 处理超时设置: 总体 {PROCESS_TIMEOUT//60} 分钟, 单日 {DAY_TIMEOUT//60} 分钟, 单波段 {BAND_TIMEOUT//60} 分钟, SCL评估 {SCL_BAND_TIMEOUT//60} 分钟")
    logging.info(f"[{a.partition_id}] SCL评估尝试次数: {SCL_MAX_ATTEMPTS}")
    logging.info(f"[{a.partition_id}] 临时目录: {TEMP_DIR}")
    logging.info(f"[{a.partition_id}] 处理时间段: {a.start_date} → {a.end_date}")

    tpl, bbox_proj, bbox_ll, mask_np = load_roi(Path(a.input_tiff), a.partition_id)
    
    # 搜索STAC items
    search_date_range = f"{a.start_date}/{a.end_date}"
    
    items = search_items(bbox_ll, search_date_range, a.max_cloud, a.partition_id)
    if not items:
        logging.warning(f"[{a.partition_id}] 无满足条件的影像，退出")
        return

    # 按日期分组
    groups = group_by_date(items, a.partition_id)

    # 创建临时目录用于处理
    base_temp_dir = tempfile.mkdtemp(prefix=f"s2_proc_{a.partition_id}_", dir=TEMP_DIR)
    logging.info(f"[{a.partition_id}] 主临时目录: {base_temp_dir}")
    
    try:
        # 创建初始客户端
        dask_client = make_client(a.dask_workers, a.worker_memory, a.partition_id)
        
        report_path = out_dir / f"dask-report-{a.partition_id}.html"
        with performance_report(filename=report_path):
            # 处理每一天的数据
            results = []
            for i, (d, its) in enumerate(groups.items()):
                # 进行垃圾回收
                gc.collect()
                
                # 在处理每个新日期前尝试重启或重新创建客户端
                if i > 0:  # 第一个日期跳过，因为刚刚创建了客户端
                    try:
                        logging.info(f"[{a.partition_id}] 尝试重启Dask客户端，设置超时240秒，不等待workers...")
                        # 设置更长的超时时间和不等待workers选项
                        dask_client.restart(timeout=240, wait_for_workers=False)
                    except (TimeoutError, Exception) as e:
                        logging.warning(f"[{a.partition_id}] Dask客户端重启失败: {e}，尝试重新创建客户端...")
                        try:
                            # 关闭现有客户端
                            try:
                                dask_client.close(timeout=30)
                            except:
                                pass
                            
                            # 强制清理内存
                            gc.collect()
                            
                            # 短暂暂停，让系统释放资源
                            time.sleep(5)
                            
                            # 重新创建客户端
                            dask_client = make_client(a.dask_workers, a.worker_memory, a.partition_id)
                            logging.info(f"[{a.partition_id}] Dask客户端重新创建成功")
                        except Exception as recreate_error:
                            logging.error(f"[{a.partition_id}] 无法重新创建Dask客户端: {recreate_error}")
                            # 继续处理，让函数正常运行，即使可能性能降低
                
                # 处理当天数据
                try:
                    success = process_day(
                        d, its, tpl, bbox_proj, mask_np,
                        out_dir, a.resolution, a.chunksize,
                        a.overwrite, a.min_coverage, a.partition_id
                    )
                    results.append(success)
                except Exception as day_error:
                    logging.error(f"[{a.partition_id}] 处理日期 {d} 时发生异常: {day_error}")
                    # 添加一个失败结果
                    results.append(False)
                    
                    # 如果发生异常，尝试重新创建客户端以恢复状态
                    try:
                        # 关闭现有客户端
                        try:
                            dask_client.close(timeout=30)
                        except:
                            pass
                        
                        # 重新创建客户端
                        time.sleep(5)  # 等待资源释放
                        dask_client = make_client(a.dask_workers, a.worker_memory, a.partition_id)
                        logging.info(f"[{a.partition_id}] 异常后Dask客户端重新创建成功")
                    except:
                        # 如果无法重新创建客户端，继续尝试处理下一天
                        pass
        
        # 尝试关闭客户端
        try:
            dask_client.close(timeout=30)
        except:
            pass
        
        # 总结统计
        success_count = sum(results)
        total_count = len(results)
        
        logging.info(f"[{a.partition_id}] ✅ 分区处理完成: 成功 {success_count}/{total_count} 天")
        logging.info(f"[{a.partition_id}] 📊 Dask 性能报告已保存: {report_path}")
        
        # 返回适当的退出码
        if success_count == 0 and total_count > 0:
            sys.exit(1)  # 全部失败
        elif success_count < total_count:
            logging.warning(f"[{a.partition_id}] ⚠️  部分日期处理失败 ({total_count - success_count}/{total_count})")
            sys.exit(2)  # 部分失败
        else:
            sys.exit(0)  # 全部成功
    
    finally:
        # 清理临时目录
        try:
            shutil.rmtree(base_temp_dir)
            logging.info(f"[{a.partition_id}] 已清理主临时目录")
        except Exception as e:
            logging.warning(f"[{a.partition_id}] 清理主临时目录失败: {e}")

if __name__ == "__main__":
    main()