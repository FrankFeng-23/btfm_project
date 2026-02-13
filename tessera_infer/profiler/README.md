# Tessera 模型推理 Profiling 系统

这是一个为Tessera模型推理系统设计的性能分析工具，支持PyTorch和ONNX模型的详细性能分析。

## 功能特性

### 🔍 统一的Profiling接口
- **自动模型类型检测**: 自动识别PyTorch和ONNX模型
- **一致的性能指标**: 两种模型格式使用相同的分析标准
- **低开销设计**: 最小化对正常推理的影响

### 📊 详细的性能指标
- **时间分析**: 模型加载、数据预处理、推理时间
- **内存分析**: 峰值内存使用、内存增长趋势
- **CPU分析**: CPU利用率、多线程性能
- **吞吐量分析**: 每秒处理样本数、批次处理效率

### 📈 多种可视化格式
- **HTML报告**: 交互式性能分析报告
- **Chrome Traces**: 时间线可视化（chrome://tracing）
- **JSON导出**: 机器可读的指标数据
- **CSV导出**: 便于进一步分析的表格数据

## 安装要求

### 基本依赖
```bash
pip install torch onnxruntime psutil numpy
```

### 可选依赖（用于高级分析）
```bash
# 系统级性能分析
pip install py-spy scalene

# Intel VTune (需要单独安装)
# https://software.intel.com/content/www/us/en/develop/tools/vtune-profiler.html
```

## 快速开始

### 1. 启用Profiling

在原有的推理命令中添加profiling参数：

```bash
# PyTorch CPU 模式
python src/multi_tile_infer.py \\
    --config configs/cpu_optimized_config.py \\
    --mode cpu \\
    --checkpoint_path checkpoints/model.pt \\
    --tile_path data/test_tile.tif \\
    --output_dir outputs/test \\
    --enable_profiling \\
    --profile_batches 20 \\
    --profile_output_dir logs/profile

# ONNX CPU 模式  
python src/multi_tile_infer.py \\
    --config configs/cpu_optimized_config.py \\
    --mode cpu \\
    --checkpoint_path checkpoints/model.pt \\
    --onnx_model_path checkpoints/model.onnx \\
    --tile_path data/test_tile.tif \\
    --output_dir outputs/test \\
    --enable_profiling \\
    --profile_batches 20 \\
    --profile_output_dir logs/profile

# PyTorch GPU 模式
python src/multi_tile_infer.py \\
    --config configs/cpu_optimized_config.py \\
    --mode gpu \\
    --gpu_id 0 \\
    --checkpoint_path checkpoints/model.pt \\
    --tile_list tile_lists/test_tiles.json \\
    --output_dir outputs/test \\
    --enable_profiling \\
    --profile_batches 20 \\
    --profile_output_dir logs/profile
```

### 2. 查看结果

Profiling完成后，在`logs/profile`目录下会生成：

```
logs/profile/
├── pytorch/                    # PyTorch profiling结果
│   ├── pytorch_model_torch_trace_20231201_143022.json
│   ├── pytorch_model_metrics_20231201_143022.json
│   └── pytorch_model_memory_timeline_20231201_143022.html
├── onnx/                      # ONNX profiling结果  
│   ├── onnx_model_onnx_trace_20231201_143022.json
│   ├── onnx_model_chrome_trace_20231201_143022.json
│   └── onnx_model_metrics_20231201_143022.json
├── reports/                   # 综合分析报告
│   └── model_profiling_report_20231201_143022.html
├── comparison_report_20231201_143022.json
├── summary_report_20231201_143022.json
└── metrics_export_20231201_143022.csv
```

## 参数说明

### Profiling参数
- `--enable_profiling`: 启用profiling（必需）
- `--profile_batches N`: 要分析的批次数（默认: 20）
- `--profile_output_dir PATH`: profiling结果输出目录（默认: logs/profile）

### 性能调优参数
- `--num_threads N`: CPU线程数（影响CPU模式性能）
- `--batch_size N`: 批次大小（影响内存和吞吐量）
- `--num_workers N`: 数据加载进程数（影响数据预处理性能）

## 结果分析

### 1. HTML报告
打开 `reports/model_profiling_report_*.html` 查看：
- **总览**: 整体性能指标
- **模型对比**: PyTorch vs ONNX性能比较
- **详细指标**: 各项性能指标的详细分析
- **优化建议**: 基于分析结果的性能优化建议

### 2. Chrome Traces
在Chrome浏览器中打开 `chrome://tracing`，加载trace文件查看：
- **时间线分析**: 各操作的执行时间线
- **并发分析**: 线程和进程的并发执行情况
- **热点识别**: 耗时最多的操作

### 3. 性能指标文件
JSON文件包含详细的性能数据：
```json
{
  "model_type": "PyTorch",
  "model_load_time": 2.34,
  "total_inference_time": 45.67,
  "avg_batch_time": 0.123,
  "throughput": 89.5,
  "peak_memory_usage": 1024.5,
  "avg_cpu_usage": 67.8
}
```

## 性能优化建议

### 识别瓶颈
1. **数据加载瓶颈**: 如果数据加载时间占总时间>20%
   - 增加 `--num_workers`
   - 使用更快的存储设备
   - 优化数据预处理逻辑

2. **模型加载瓶颈**: 如果模型加载时间过长
   - 使用模型压缩技术
   - 考虑模型并行化
   - 优化checkpoint格式

3. **内存瓶颈**: 如果内存增长>500MB
   - 减少 `--batch_size`
   - 检查内存泄漏
   - 优化数据类型（float32 vs float16）

4. **CPU利用率低**: 如果CPU使用率<25%
   - 增加 `--num_threads`
   - 检查GIL锁定问题
   - 优化算子实现

### PyTorch vs ONNX选择
根据profiling结果选择最适合的推理引擎：

- **ONNX优势场景**:
  - CPU推理优化更好
  - 内存使用更少
  - 启动时间更快

- **PyTorch优势场景**:
  - 更好的调试支持
  - 更丰富的算子支持
  - GPU加速更成熟

## 高级功能

### 1. 批量对比分析
```python
from profiler.unified_profiler import UnifiedProfiler

profiler = UnifiedProfiler("logs/profile")

# 比较不同配置
comparison = profiler.compare_models(
    pytorch_model_loader=load_pytorch_model,
    onnx_model_loader=load_onnx_model,
    dataloader=test_loader,
    num_batches=50
)

print(comparison.recommendations)
```

### 2. 自定义指标收集
```python
from profiler.metrics import MetricsCollector

collector = MetricsCollector("logs/profile")
collector.load_metrics_from_file("metrics_1.json")
collector.load_metrics_from_file("metrics_2.json")

summary = collector.generate_summary_report()
```

### 3. 自定义报告生成
```python
from profiler.visualizer import HTMLReportGenerator

generator = HTMLReportGenerator("logs/profile/reports")
report_file = generator.generate_comprehensive_report(
    report_data, 
    "custom_profiling_report"
)
```

## 故障排除

### 常见问题

1. **ImportError: No module named 'profiler'**
   ```bash
   # 确保在项目根目录运行
   cd /path/to/tessera/tessera_infer
   python src/multi_tile_infer.py ...
   ```

2. **ONNX Runtime not available**
   ```bash
   pip install onnxruntime
   # 或者 GPU版本
   pip install onnxruntime-gpu
   ```

3. **内存不足错误**
   ```bash
   # 减少batch_size和profile_batches
   --batch_size 32 --profile_batches 10
   ```

4. **权限错误**
   ```bash
   # 确保有写入权限
   mkdir -p logs/profile
   chmod 755 logs/profile
   ```

### 日志分析
查看日志了解profiling执行状态：
```bash
tail -f logs/infer_cpu_0.log | grep -i profiling
```

## 贡献

如果你发现bug或有改进建议，请：
1. 检查现有的issue
2. 创建详细的bug报告或功能请求
3. 提供复现步骤和环境信息

## 许可证

本项目基于MIT许可证开源。