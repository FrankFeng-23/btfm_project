#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performance optimization guide for Tessera profiling

This script provides quick diagnosis and optimization suggestions
based on common profiling issues.
"""

import os
import json
import sys
from typing import Dict, List, Any

def analyze_metrics_file(filepath: str) -> Dict[str, Any]:
    """Analyze a metrics JSON file and provide optimization suggestions"""
    if not os.path.exists(filepath):
        print(f"❌ Metrics file not found: {filepath}")
        return {}
    
    try:
        with open(filepath, 'r') as f:
            metrics = json.load(f)
        
        print(f"✅ Loaded metrics from: {os.path.basename(filepath)}")
        return metrics
    except Exception as e:
        print(f"❌ Failed to load metrics: {e}")
        return {}

def diagnose_performance(metrics: Dict[str, Any]) -> List[str]:
    """Diagnose performance issues and provide suggestions"""
    suggestions = []
    
    # Check basic metrics
    model_type = metrics.get('model_type', 'Unknown')
    throughput = metrics.get('throughput', 0)
    cpu_usage = metrics.get('avg_cpu_usage', 0)
    memory_growth = metrics.get('memory_growth', 0)
    inference_time = metrics.get('total_inference_time', 0)
    
    print(f"\\n📊 Performance Analysis for {model_type} Model:")
    print(f"   Throughput: {throughput:.1f} samples/sec")
    print(f"   CPU Usage: {cpu_usage:.1f}%") 
    print(f"   Memory Growth: {memory_growth:.1f} MB")
    print(f"   Inference Time: {inference_time:.2f} seconds")
    
    # Throughput analysis
    if throughput < 50:
        suggestions.append("🐌 Low throughput detected:")
        suggestions.append("   - Increase --num_threads (try 8 or more)")
        suggestions.append("   - Increase --batch_size (try 128 or 256)")
        suggestions.append("   - Consider using ONNX model for CPU inference")
    
    # CPU usage analysis
    if cpu_usage < 30:
        suggestions.append("💤 Low CPU utilization:")
        suggestions.append("   - Increase --num_threads for better CPU usage")
        suggestions.append("   - Check if you're I/O bound (data loading bottleneck)")
        suggestions.append("   - Consider increasing --num_workers for data loading")
    
    # Memory analysis
    if memory_growth > 1000:  # > 1GB
        suggestions.append("🧠 High memory growth:")
        suggestions.append("   - Reduce --batch_size to control memory usage")
        suggestions.append("   - Check for memory leaks in data pipeline")
        suggestions.append("   - Consider memory-mapped data loading")
    
    # Model-specific suggestions
    if model_type.lower() == 'pytorch':
        suggestions.append("🔥 PyTorch optimization tips:")
        suggestions.append("   - Try converting to ONNX for better CPU performance")
        suggestions.append("   - Use torch.jit.script for model optimization")
        suggestions.append("   - Enable torch.backends.mkldnn.enabled = True")
    
    elif model_type.lower() == 'onnx':
        suggestions.append("⚡ ONNX optimization tips:")
        suggestions.append("   - Already using optimized ONNX inference")
        suggestions.append("   - Fine-tune thread count for your specific CPU")
        suggestions.append("   - Consider model quantization if accuracy allows")
    
    return suggestions

def quick_performance_check():
    """Quick performance diagnostic"""
    print("🚀 Tessera Profiling Performance Diagnostic")
    print("=" * 50)
    
    # Check recent profiling results
    profile_dir = "logs/profile"
    if not os.path.exists(profile_dir):
        print("❌ No profiling results found. Run profiling first:")
        print("   python src/multi_tile_infer.py --enable_profiling --profile_fast ...")
        return
    
    # Find metrics files
    metrics_files = []
    for root, dirs, files in os.walk(profile_dir):
        for file in files:
            if file.endswith('_metrics.json'):
                metrics_files.append(os.path.join(root, file))
    
    if not metrics_files:
        print("❌ No metrics files found. Make sure profiling completed successfully.")
        return
    
    print(f"📁 Found {len(metrics_files)} metrics files")
    
    # Analyze each metrics file
    all_suggestions = []
    for filepath in sorted(metrics_files)[-2:]:  # Analyze last 2 files
        metrics = analyze_metrics_file(filepath)
        if metrics:
            suggestions = diagnose_performance(metrics)
            all_suggestions.extend(suggestions)
    
    # Print optimization suggestions
    if all_suggestions:
        print("\\n🔧 Optimization Suggestions:")
        for suggestion in set(all_suggestions):  # Remove duplicates
            print(f"   {suggestion}")
    else:
        print("\\n✅ Performance looks good!")
    
    print("\\n📚 For more detailed analysis:")
    print("   - Check individual JSON metrics files")
    print("   - Generate HTML reports with --profile_format html")
    print("   - Use chrome://tracing to view timeline traces")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Analyze specific metrics file
        metrics_file = sys.argv[1]
        metrics = analyze_metrics_file(metrics_file)
        if metrics:
            diagnose_performance(metrics)
    else:
        # Quick check of recent results
        quick_performance_check()

if __name__ == "__main__":
    main()