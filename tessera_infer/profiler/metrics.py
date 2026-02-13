#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import json
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from .base_profiler import ProfileMetrics
from .utils import format_time, format_memory, detect_bottlenecks, create_profile_summary

@dataclass
class ComparisonMetrics:
    """Metrics for comparing different profiling runs"""
    pytorch_metrics: Optional[ProfileMetrics] = None
    onnx_metrics: Optional[ProfileMetrics] = None
    comparison_timestamp: str = ""
    performance_ratio: Dict[str, float] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.comparison_timestamp == "":
            self.comparison_timestamp = datetime.now().isoformat()
        if self.performance_ratio is None:
            self.performance_ratio = {}
        if self.recommendations is None:
            self.recommendations = []

class MetricsCollector:
    """Collect and analyze profiling metrics from multiple sources"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metrics_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def add_metrics(self, metrics: ProfileMetrics, source: str = "unknown"):
        """Add metrics from a profiling run"""
        with self._lock:
            metrics_dict = asdict(metrics)
            metrics_dict['source'] = source
            metrics_dict['collection_timestamp'] = datetime.now().isoformat()
            self.metrics_history.append(metrics_dict)
    
    def load_metrics_from_file(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load metrics from a JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Add to history if valid
            if isinstance(data, dict) and 'model_type' in data:
                data['source'] = f"file:{os.path.basename(filepath)}"
                data['collection_timestamp'] = datetime.now().isoformat()
                
                with self._lock:
                    self.metrics_history.append(data)
                
                return data
            
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"Failed to load metrics from {filepath}: {e}")
        
        return None
    
    def get_metrics_by_type(self, model_type: str) -> List[Dict[str, Any]]:
        """Get all metrics for a specific model type"""
        with self._lock:
            return [m for m in self.metrics_history if m.get('model_type', '').lower() == model_type.lower()]
    
    def get_latest_metrics_by_type(self, model_type: str) -> Optional[Dict[str, Any]]:
        """Get the most recent metrics for a specific model type"""
        # Get metrics without calling get_metrics_by_type to avoid nested locks
        with self._lock:
            metrics_list = [m for m in self.metrics_history if m.get('model_type', '').lower() == model_type.lower()]
        
        if not metrics_list:
            return None
        
        # Sort by timestamp and return most recent
        metrics_list.sort(key=lambda x: x.get('collection_timestamp', ''), reverse=True)
        return metrics_list[0]
    
    
    def compare_pytorch_vs_onnx(self) -> ComparisonMetrics:
        """Compare PyTorch and ONNX profiling results"""
        pytorch_metrics = self.get_latest_metrics_by_type('pytorch')
        onnx_metrics = self.get_latest_metrics_by_type('onnx')
        
        comparison = ComparisonMetrics()
        
        if pytorch_metrics:
            comparison.pytorch_metrics = ProfileMetrics(**pytorch_metrics)
        if onnx_metrics:
            comparison.onnx_metrics = ProfileMetrics(**onnx_metrics)
        
        # Calculate performance ratios
        if pytorch_metrics and onnx_metrics:
            ratios = {}
            
            # Throughput ratio (higher is better)
            pt_throughput = pytorch_metrics.get('throughput', 0)
            onnx_throughput = onnx_metrics.get('throughput', 0)
            if pt_throughput > 0 and onnx_throughput > 0:
                ratios['throughput_ratio'] = onnx_throughput / pt_throughput
            
            # Inference time ratio (lower is better for inference time)
            pt_inference_time = pytorch_metrics.get('total_inference_time', 0)
            onnx_inference_time = onnx_metrics.get('total_inference_time', 0)
            if pt_inference_time > 0 and onnx_inference_time > 0:
                ratios['inference_time_ratio'] = onnx_inference_time / pt_inference_time
            
            # Memory ratio
            pt_memory = pytorch_metrics.get('peak_memory_usage', 0)
            onnx_memory = onnx_metrics.get('peak_memory_usage', 0)
            if pt_memory > 0 and onnx_memory > 0:
                ratios['memory_ratio'] = onnx_memory / pt_memory
            
            # Model load time ratio
            pt_load_time = pytorch_metrics.get('model_load_time', 0)
            onnx_load_time = onnx_metrics.get('model_load_time', 0)
            if pt_load_time > 0 and onnx_load_time > 0:
                ratios['load_time_ratio'] = onnx_load_time / pt_load_time
            
            comparison.performance_ratio = ratios
        
        # Generate recommendations
        comparison.recommendations = self._generate_recommendations(comparison)
        
        return comparison
    
    def _generate_recommendations(self, comparison: ComparisonMetrics) -> List[str]:
        """Generate performance recommendations based on comparison"""
        recommendations = []
        
        if not comparison.pytorch_metrics or not comparison.onnx_metrics:
            return ["需要同时有PyTorch和ONNX的profiling数据才能生成建议"]
        
        ratios = comparison.performance_ratio
        
        # Throughput recommendations
        if 'throughput_ratio' in ratios:
            if ratios['throughput_ratio'] > 1.2:
                recommendations.append("ONNX模型吞吐量比PyTorch高20%以上，建议在生产环境中使用ONNX")
            elif ratios['throughput_ratio'] < 0.8:
                recommendations.append("PyTorch模型吞吐量比ONNX高25%以上，可能ONNX转换有优化空间")
        
        # Memory recommendations
        if 'memory_ratio' in ratios:
            if ratios['memory_ratio'] < 0.7:
                recommendations.append("ONNX模型内存使用比PyTorch低30%以上，有助于提升内存利用率")
            elif ratios['memory_ratio'] > 1.5:
                recommendations.append("ONNX模型内存使用比PyTorch高50%以上，检查是否有内存泄漏")
        
        # Load time recommendations  
        if 'load_time_ratio' in ratios:
            if ratios['load_time_ratio'] < 0.5:
                recommendations.append("ONNX模型加载比PyTorch快50%以上，有助于减少启动时间")
            elif ratios['load_time_ratio'] > 2.0:
                recommendations.append("ONNX模型加载比PyTorch慢2倍以上，检查模型优化设置")
        
        # Bottleneck analysis
        pt_bottlenecks = detect_bottlenecks(asdict(comparison.pytorch_metrics))
        onnx_bottlenecks = detect_bottlenecks(asdict(comparison.onnx_metrics))
        
        if pt_bottlenecks:
            recommendations.append(f"PyTorch模型瓶颈: {'; '.join(pt_bottlenecks[:3])}")
        
        if onnx_bottlenecks:
            recommendations.append(f"ONNX模型瓶颈: {'; '.join(onnx_bottlenecks[:3])}")
        
        return recommendations
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a comprehensive summary of all collected metrics"""
        print("DEBUG: Entering generate_summary_report...")
        
        with self._lock:
            print("DEBUG: Acquired lock in generate_summary_report")
            if not self.metrics_history:
                print("DEBUG: No metrics history found")
                return {"error": "No metrics collected"}
            
            print(f"DEBUG: Found {len(self.metrics_history)} metrics entries")
            summary = create_profile_summary(self.metrics_history)
            print("DEBUG: create_profile_summary completed")
            
            # Add additional analysis
            summary['collection_info'] = {
                "total_runs": len(self.metrics_history),
                "unique_sources": len(set(m.get('source', 'unknown') for m in self.metrics_history)),
                "model_types": list(set(m.get('model_type', 'unknown') for m in self.metrics_history)),
                "time_range": {
                    "earliest": min(m.get('collection_timestamp', '') for m in self.metrics_history),
                    "latest": max(m.get('collection_timestamp', '') for m in self.metrics_history)
                }
            }
            print("DEBUG: Added collection_info")
            
            # Performance trends
            print("DEBUG: Starting performance trends analysis...")
            pytorch_runs = self.get_metrics_by_type('pytorch')
            onnx_runs = self.get_metrics_by_type('onnx')
            print(f"DEBUG: Found {len(pytorch_runs)} pytorch runs, {len(onnx_runs)} onnx runs")
            
            if len(pytorch_runs) > 1:
                print("DEBUG: Analyzing pytorch trends...")
                summary['pytorch_trends'] = self._analyze_trends(pytorch_runs)
                print("DEBUG: PyTorch trends analysis completed")
            
            if len(onnx_runs) > 1:
                print("DEBUG: Analyzing ONNX trends...")
                summary['onnx_trends'] = self._analyze_trends(onnx_runs)
                print("DEBUG: ONNX trends analysis completed")
            
            print("DEBUG: generate_summary_report completed successfully")
            return summary
    
    def _analyze_trends(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if len(metrics_list) < 2:
            return {}
        
        # Sort by timestamp
        sorted_metrics = sorted(metrics_list, key=lambda x: x.get('collection_timestamp', ''))
        
        # Calculate trends for key metrics
        trends = {}
        
        for metric_name in ['throughput', 'total_inference_time', 'peak_memory_usage', 'avg_batch_time']:
            values = [m.get(metric_name, 0) for m in sorted_metrics if isinstance(m.get(metric_name), (int, float))]
            
            if len(values) >= 2:
                # Simple trend: compare first and last values
                first_val = values[0]
                last_val = values[-1]
                
                if first_val > 0:
                    change_percent = ((last_val - first_val) / first_val) * 100
                    trends[metric_name] = {
                        "first_value": first_val,
                        "last_value": last_val,
                        "change_percent": change_percent,
                        "trend": "improving" if change_percent < 0 and metric_name in ['total_inference_time', 'avg_batch_time'] 
                               else "improving" if change_percent > 0 and metric_name in ['throughput']
                               else "declining" if change_percent != 0 else "stable"
                    }
        
        return trends
    
    def save_comparison_report(self, comparison: ComparisonMetrics, 
                              filename: Optional[str] = None) -> str:
        """Save comparison report to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_report_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to dict for JSON serialization
        report = {
            "comparison_timestamp": comparison.comparison_timestamp,
            "performance_ratio": comparison.performance_ratio,
            "recommendations": comparison.recommendations,
            "pytorch_metrics": asdict(comparison.pytorch_metrics) if comparison.pytorch_metrics else None,
            "onnx_metrics": asdict(comparison.onnx_metrics) if comparison.onnx_metrics else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filepath
    
    def save_summary_report(self, filename: Optional[str] = None) -> str:
        """Save summary report to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summary_report_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        summary = self.generate_summary_report()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return filepath
    
    def clear_history(self):
        """Clear all collected metrics"""
        with self._lock:
            self.metrics_history.clear()
    
    def export_metrics_csv(self, filename: Optional[str] = None) -> str:
        """Export metrics to CSV format"""
        import csv
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_export_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with self._lock:
            if not self.metrics_history:
                return ""
            
            # Get all unique field names
            all_fields = set()
            for metrics in self.metrics_history:
                all_fields.update(metrics.keys())
            
            fieldnames = sorted(list(all_fields))
            
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for metrics in self.metrics_history:
                    # Flatten nested dicts
                    flat_metrics = {}
                    for key, value in metrics.items():
                        if isinstance(value, dict):
                            for nested_key, nested_value in value.items():
                                flat_metrics[f"{key}.{nested_key}"] = nested_value
                        else:
                            flat_metrics[key] = value
                    
                    writer.writerow(flat_metrics)
        
        return filepath