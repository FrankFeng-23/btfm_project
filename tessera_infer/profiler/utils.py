#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import subprocess
import tempfile
from typing import Optional, Dict, Any, List
from datetime import datetime

def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format"""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def format_memory(mb: float) -> str:
    """Format memory in MB to human readable format"""
    if mb < 1024:
        return f"{mb:.1f}MB"
    elif mb < 1024 * 1024:
        return f"{mb/1024:.1f}GB"
    else:
        return f"{mb/(1024*1024):.1f}TB"

def get_model_type(model) -> str:
    """Detect model type from model object"""
    model_class = model.__class__.__name__
    if "ONNX" in model_class or hasattr(model, 'session'):
        return "onnx"
    elif "torch" in str(type(model)).lower() or hasattr(model, 'forward'):
        return "pytorch"
    else:
        return "unknown"

def ensure_directory(path: str) -> str:
    """Ensure directory exists and return the path"""
    os.makedirs(path, exist_ok=True)
    return path

def generate_timestamp() -> str:
    """Generate timestamp string for filenames"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def run_py_spy_profiling(pid: int, duration: int, output_path: str, 
                        format: str = "svg") -> Optional[str]:
    """Run py-spy profiling on a process"""
    try:
        cmd = [
            "py-spy", "record", 
            "-p", str(pid),
            "-d", str(duration),
            "-o", output_path,
            "-f", format,
            "--subprocesses"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 30)
        
        if result.returncode == 0:
            return output_path
        else:
            print(f"py-spy failed: {result.stderr}")
            return None
            
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"py-spy profiling failed: {e}")
        return None

def run_scalene_profiling(script_path: str, args: List[str], 
                         output_path: str, duration: Optional[int] = None) -> Optional[str]:
    """Run Scalene profiling on a script"""
    try:
        cmd = ["scalene", "--html", "--outfile", output_path]
        
        if duration:
            cmd.extend(["--profile-interval", str(max(1, duration // 100))])
        
        cmd.append(script_path)
        cmd.extend(args)
        
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              timeout=(duration + 60) if duration else 300)
        
        if result.returncode == 0:
            return output_path
        else:
            print(f"Scalene failed: {result.stderr}")
            return None
            
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"Scalene profiling failed: {e}")
        return None

def merge_json_traces(trace_files: List[str], output_path: str) -> str:
    """Merge multiple JSON trace files into one"""
    import json
    
    merged_events = []
    
    for trace_file in trace_files:
        if os.path.exists(trace_file):
            try:
                with open(trace_file, 'r') as f:
                    data = json.load(f)
                
                # Handle different trace formats
                if isinstance(data, dict):
                    if 'traceEvents' in data:
                        merged_events.extend(data['traceEvents'])
                    elif 'events' in data:
                        merged_events.extend(data['events'])
                elif isinstance(data, list):
                    merged_events.extend(data)
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading trace file {trace_file}: {e}")
    
    # Write merged trace
    merged_trace = {
        "traceEvents": merged_events,
        "displayTimeUnit": "ns",
        "systemTraceEvents": "CpuCores",
        "otherData": {
            "version": "Chrome Trace Format",
            "creator": "Tessera Profiler"
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(merged_trace, f, separators=(',', ':'))
    
    return output_path

def create_profile_summary(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a summary of multiple profiling runs"""
    if not metrics_list:
        return {}
    
    summary = {
        "total_runs": len(metrics_list),
        "model_types": list(set(m.get("model_type", "unknown") for m in metrics_list)),
        "timestamp": generate_timestamp(),
        "averages": {},
        "totals": {},
        "comparisons": {}
    }
    
    # Calculate averages for numeric fields
    numeric_fields = [
        "model_load_time", "model_warmup_time", "total_inference_time",
        "avg_batch_time", "data_loading_time", "peak_memory_usage",
        "memory_growth", "throughput", "avg_cpu_usage"
    ]
    
    for field in numeric_fields:
        values = [m.get(field, 0) for m in metrics_list if isinstance(m.get(field), (int, float))]
        if values:
            summary["averages"][field] = sum(values) / len(values)
            summary["totals"][field] = sum(values)
    
    # Create comparisons between model types
    by_model_type = {}
    for metrics in metrics_list:
        model_type = metrics.get("model_type", "unknown")
        if model_type not in by_model_type:
            by_model_type[model_type] = []
        by_model_type[model_type].append(metrics)
    
    for model_type, model_metrics in by_model_type.items():
        type_summary = {}
        for field in numeric_fields:
            values = [m.get(field, 0) for m in model_metrics if isinstance(m.get(field), (int, float))]
            if values:
                type_summary[field] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        summary["comparisons"][model_type] = type_summary
    
    return summary

def detect_bottlenecks(metrics: Dict[str, Any], threshold_percentages: Dict[str, float] = None) -> List[str]:
    """Analyze metrics and detect potential bottlenecks"""
    if threshold_percentages is None:
        threshold_percentages = {
            "data_loading_time": 20.0,  # > 20% of total time
            "model_load_time": 30.0,    # > 30% of total time
            "memory_growth": 500.0,     # > 500MB growth
            "low_cpu_usage": 25.0,      # < 25% CPU usage
            "low_throughput": 10.0      # < 10 samples/sec
        }
    
    bottlenecks = []
    total_time = metrics.get("total_inference_time", 0)
    
    if total_time > 0:
        # Data loading bottleneck
        data_loading_pct = (metrics.get("data_loading_time", 0) / total_time) * 100
        if data_loading_pct > threshold_percentages["data_loading_time"]:
            bottlenecks.append(f"Data loading takes {data_loading_pct:.1f}% of total time")
        
        # Model loading bottleneck
        model_load_pct = (metrics.get("model_load_time", 0) / total_time) * 100
        if model_load_pct > threshold_percentages["model_load_time"]:
            bottlenecks.append(f"Model loading takes {model_load_pct:.1f}% of total time")
    
    # Memory growth
    memory_growth = metrics.get("memory_growth", 0)
    if memory_growth > threshold_percentages["memory_growth"]:
        bottlenecks.append(f"High memory growth: {format_memory(memory_growth)}")
    
    # Low CPU usage
    cpu_usage = metrics.get("avg_cpu_usage", 0)
    if cpu_usage < threshold_percentages["low_cpu_usage"]:
        bottlenecks.append(f"Low CPU utilization: {cpu_usage:.1f}%")
    
    # Low throughput
    throughput = metrics.get("throughput", 0)
    if throughput < threshold_percentages["low_throughput"]:
        bottlenecks.append(f"Low throughput: {throughput:.1f} samples/sec")
    
    return bottlenecks

class TempProfileScript:
    """Context manager for creating temporary profiling scripts"""
    
    def __init__(self, script_content: str, suffix: str = ".py"):
        self.script_content = script_content
        self.suffix = suffix
        self.temp_file = None
        self.filepath = None
        
    def __enter__(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=self.suffix, delete=False)
        self.temp_file.write(self.script_content)
        self.temp_file.flush()
        self.filepath = self.temp_file.name
        return self.filepath
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_file:
            self.temp_file.close()
        if self.filepath and os.path.exists(self.filepath):
            try:
                os.unlink(self.filepath)
            except OSError:
                pass