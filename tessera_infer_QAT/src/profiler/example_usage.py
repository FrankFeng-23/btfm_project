#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example usage script for Tessera model profiling

This script demonstrates how to use the profiling system to analyze
PyTorch and ONNX model performance.
"""

import os
import sys

# Example 1: Minimal profiling mode (FASTEST - for debugging)
def profile_minimal_mode():
    cmd = """
python src/multi_tile_infer.py \\
    --config configs/cpu_optimized_config.py \\
    --mode cpu \\
    --checkpoint_path checkpoints/best_model_fsdp_20250427_084307.pt \\
    --tile_path /absolute_path_to_data_dir/retiled_d_pixel/0_2500_500_3000 \\
    --output_dir /absolute_path_to_output_dir \\
    --num_threads 8 \\
    --batch_size 128 \\
    --enable_profiling \\
    --profile_batches 5 \\
    --profile_minimal \\
    --profile_output_dir logs/profile
"""
    print("=== Minimal Profiling Mode (Debugging) ===")
    print(cmd)

# Example 2: Profile PyTorch model on CPU (FAST MODE - Recommended)
def profile_pytorch_cpu_fast():
    cmd = """
python src/multi_tile_infer.py \\
    --config configs/cpu_optimized_config.py \\
    --mode cpu \\
    --checkpoint_path checkpoints/best_model_fsdp_20250427_084307.pt \\
    --tile_path /absolute_path_to_data_dir/retiled_d_pixel/0_2500_500_3000 \\
    --output_dir /absolute_path_to_output_dir \\
    --num_threads 8 \\
    --batch_size 128 \\
    --enable_profiling \\
    --profile_batches 10 \\
    --profile_fast \\
    --profile_output_dir logs/profile
"""
    print("=== Fast PyTorch CPU Profiling (Recommended) ===")
    print(cmd)

# Example 2: Profile PyTorch model on CPU (DETAILED)
def profile_pytorch_cpu_detailed():
    cmd = """
python src/multi_tile_infer.py \\
    --config configs/cpu_optimized_config.py \\
    --mode cpu \\
    --checkpoint_path checkpoints/best_model_fsdp_20250427_084307.pt \\
    --tile_path /absolute_path_to_data_dir/retiled_d_pixel/0_2500_500_3000 \\
    --output_dir /absolute_path_to_output_dir \\
    --num_threads 8 \\
    --batch_size 128 \\
    --enable_profiling \\
    --profile_batches 20 \\
    --profile_format all \\
    --profile_output_dir logs/profile
"""
    print("=== Detailed PyTorch CPU Profiling (Slower but Complete) ===")
    print(cmd)

# Example 3: Profile ONNX model on CPU (FAST MODE)  
def profile_onnx_cpu_fast():
    cmd = """
python src/multi_tile_infer.py \\
    --config configs/cpu_optimized_config.py \\
    --mode cpu \\
    --checkpoint_path checkpoints/best_model_fsdp_20250427_084307.pt \\
    --onnx_model_path checkpoints/model_optimized.onnx \\
    --tile_path /absolute_path_to_data_dir/retiled_d_pixel/0_2500_500_3000 \\
    --output_dir /absolute_path_to_output_dir \\
    --num_threads 8 \\
    --batch_size 128 \\
    --enable_profiling \\
    --profile_batches 10 \\
    --profile_fast \\
    --profile_output_dir logs/profile
"""
    print("=== Fast ONNX CPU Profiling ===")
    print(cmd)

# Example 3: Profile PyTorch model on GPU
def profile_pytorch_gpu():
    cmd = """
python src/multi_tile_infer.py \\
    --config configs/cpu_optimized_config.py \\
    --mode gpu \\
    --gpu_id 0 \\
    --checkpoint_path checkpoints/your_model.pt \\
    --tile_list tile_lists/test_tiles.json \\
    --output_dir outputs/test \\
    --batch_size 128 \\
    --enable_profiling \\
    --profile_batches 20 \\
    --profile_output_dir logs/profile
"""
    print("=== PyTorch GPU Profiling ===")
    print(cmd)

if __name__ == "__main__":
    print("Tessera Model Profiling Usage Examples")
    print("=" * 50)
    
    print("\\n1. Make sure your environment has the required dependencies:")
    print("   - torch")
    print("   - onnxruntime (for ONNX profiling)")
    print("   - psutil")
    print("   - numpy")
    
    print("\\n2. IMPORTANT: Use multiple CPU threads for better performance!")
    print("   --num_threads 8 (or more depending on your CPU)")
    print("   --batch_size 128 (larger batches for better efficiency)")
    
    print("\\n3. Debugging and testing options:")
    profile_minimal_mode()
    
    print("\\n4. Fast profiling examples (Recommended):")
    profile_pytorch_cpu_fast()
    print()
    profile_onnx_cpu_fast()
    
    print("\\n5. Detailed profiling examples (Slower but comprehensive):")
    profile_pytorch_cpu_detailed()
    print()
    profile_onnx_cpu() 
    print()
    profile_pytorch_gpu()
    
    print("\\n3. After profiling, check the output directory for:")
    print("   - HTML reports with performance analysis")
    print("   - JSON metrics files")
    print("   - Chrome trace files (viewable at chrome://tracing)")
    print("   - CSV exports for further analysis")
    
    print("\\n5. Key profiling options:")
    print("   --enable_profiling     Enable profiling (required)")
    print("   --profile_batches N    Number of batches to profile (default: 20)")
    print("   --profile_fast         Fast mode (JSON only, recommended)")
    print("   --profile_format       json/csv/html/all (default: json)")
    print("   --num_threads N        CPU threads (8+ recommended)")
    print("   --batch_size N         Batch size (128+ for better efficiency)")
    
    print("\\n6. Performance tips:")
    print("   - Start with --profile_fast for quick results")
    print("   - Use --num_threads equal to your CPU cores")
    print("   - Increase --batch_size to 128 or 256 for better throughput")
    print("   - Use --profile_batches 10 for initial testing")
    
    print("\\n7. Understanding the results:")
    print("   - Check JSON metrics files first (fastest)")
    print("   - Generate HTML reports only when needed (slower)")
    print("   - Compare PyTorch vs ONNX performance")
    print("   - Look for memory usage patterns")
    
    print("\\n8. Troubleshooting slow profiling:")
    print("   🐛 If profiling hangs:")
    print("      1. Use --profile_minimal first for debugging")
    print("      2. Run: python profiler/debug_profiler.py")
    print("      3. Check logs for where it hangs")
    print("   ")
    print("   ⚡ Quick fixes:")
    print("      - Add --profile_minimal for basic metrics only")
    print("      - Reduce --profile_batches to 5 or 10")
    print("      - Use --log_level DEBUG to see detailed logs")
    print("   ")
    print("   🔧 Debug commands:")
    print("      python profiler/debug_profiler.py                    # Test basic functionality")
    print("      python profiler/performance_diagnosis.py            # Analyze existing results")
    print("      python profiler/generate_html_report.py             # Generate HTML from JSON")