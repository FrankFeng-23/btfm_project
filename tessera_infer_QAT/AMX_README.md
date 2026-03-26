# AMX (Advanced Matrix Extensions) Support

## Overview

AMX (Advanced Matrix Extensions) is an Intel x86 extension designed to accelerate deep learning computations on CPUs. This implementation adds AMX support to the QAT inference pipeline for optimal performance on Intel 4th Gen Xeon Scalable processors.

## Features

- **Automatic Hardware Detection**: Detects AMX support by checking CPU flags (`amx_bf16`, `amx_int8`, `amx_tile`)
- **Graceful Degradation**: Automatically falls back to standard CPU inference on non-AMX hardware
- **BF16 Integration**: Optimal performance when combined with BF16 precision
- **Environment Optimization**: Automatically configures oneDNN settings for AMX
- **Verification Support**: oneDNN verbose logging for AMX operation verification

## Usage

### Basic Usage

```bash
# Enable AMX acceleration
python src/multi_tile_infer.py \
    --config configs/cpu_optimized_config.py \
    --mode cpu \
    --checkpoint_path checkpoints/model.pt \
    --tile_path /path/to/tile \
    --output_dir /path/to/output \
    --enable_amx

# Optimal: AMX + BF16 for maximum performance
python src/multi_tile_infer.py \
    --config configs/cpu_optimized_config.py \
    --mode cpu \
    --checkpoint_path checkpoints/model.pt \
    --tile_path /path/to/tile \
    --output_dir /path/to/output \
    --enable_amx \
    --enable_bf16
```

### Testing AMX Support

```bash
# Test AMX hardware detection and integration
python test_amx.py
```

## Hardware Requirements

### Supported Hardware
- Intel 4th Gen Xeon Scalable processors (Sapphire Rapids)
- Processors with AMX_BF16 and/or AMX_INT8 instructions

### Required CPU Flags
- `amx_bf16` - AMX BF16 matrix operations
- `amx_int8` - AMX INT8 matrix operations  
- `amx_tile` - AMX tile configuration support
- `avx512f` - AVX-512 foundation instructions

### Verification
Check your CPU support:
```bash
cat /proc/cpuinfo | grep -E 'amx_bf16|amx_int8|amx_tile'
```

## Implementation Details

### AMX Detection
- Scans `/proc/cpuinfo` for AMX-related CPU flags
- Checks for supporting AVX-512 instructions
- Provides detailed hardware capability reporting

### Performance Optimizations

1. **Memory Format Optimization**
   ```python
   # Convert model to channels_last for AMX
   model = model.to(memory_format=torch.channels_last)
   ```

2. **Autocast Integration**
   ```python
   # AMX + BF16 optimal context
   with torch.amp.autocast('cpu', dtype=torch.bfloat16, enabled=True):
       output = model(input)
   ```

3. **Environment Configuration**
   ```python
   # Automatic oneDNN setup
   os.environ['DNNL_MAX_CPU_ISA'] = 'AMX_BF16'
   os.environ['ONEDNN_DEFAULT_FPMATH_MODE'] = 'BF16'
   ```

### Integration Points

The AMX implementation integrates with:
- **Device Setup**: `utils/device_utils.py` - Hardware detection and environment setup
- **Model Loading**: `utils/model_loader.py` - Model optimization for AMX
- **Inference Engine**: `inference/qat_engine.py` - AMX-optimized inference contexts
- **Argument Parsing**: `utils/args_parser.py` - `--enable_amx` flag

## Performance Expectations

On supported Intel 4th Gen Xeon processors, AMX can provide:
- **3-10x speedup** for matrix operations vs previous generations
- **Optimal with BF16**: Best performance combining AMX + BF16
- **Linear/Conv layers**: Primary beneficiaries of AMX acceleration
- **Large batch sizes**: Better AMX utilization with larger batches

## Troubleshooting

### Common Issues

1. **AMX requested but not supported**
   - **Cause**: Running on non-Intel or older Intel CPUs
   - **Solution**: Normal behavior, will use standard CPU inference

2. **BF16 dtype mismatch errors**
   - **Cause**: Model weights and inputs have mismatched dtypes
   - **Solution**: The implementation handles this automatically with proper conversions

3. **No performance improvement**
   - **Check**: Verify oneDNN verbose shows AMX operations: `ONEDNN_VERBOSE=1`
   - **Check**: Use appropriate batch sizes (32+ recommended)
   - **Check**: Ensure BF16 is enabled for maximum benefit

### Debugging

Enable detailed logging:
```bash
# Debug mode with oneDNN verbose
python src/multi_tile_infer.py \
    --enable_amx \
    --enable_bf16 \
    --log_level DEBUG \
    ...
```

Check for AMX operations in logs:
- Look for: `avx512_core_amx_bf16` or `avx512_core_amx_int8` in oneDNN verbose output
- AMX detection logs show hardware capabilities
- Model optimization logs confirm AMX-specific transformations

## Compatibility

- **PyTorch**: Requires PyTorch with oneDNN backend support
- **Operating System**: Linux x86_64
- **Python**: 3.8+
- **Dependencies**: Standard PyTorch installation (no additional packages required)

## Notes

- AMX is automatically enabled by PyTorch when available - this implementation provides explicit control and optimization
- Graceful fallback ensures code works on all hardware
- Performance benefits are most significant for transformer and dense layer operations
- INT8 quantization works independently of AMX (this implementation uses symmetric quantization post-AMX inference)
