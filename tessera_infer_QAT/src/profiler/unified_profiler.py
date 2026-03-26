#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import DataLoader

from .pytorch_profiler import PyTorchProfiler
from .onnx_profiler import ONNXProfiler
from .metrics import MetricsCollector, ComparisonMetrics
from .visualizer import HTMLReportGenerator
from .utils import get_model_type, ensure_directory, generate_timestamp

class UnifiedProfiler:
    """
    Unified profiler that automatically detects model type and uses appropriate profiler.
    Provides a single interface for profiling both PyTorch and ONNX models.
    """
    
    def __init__(self, output_dir: str, model_name: str = "model", 
                 enable_system_profiling: bool = True):
        """
        Initialize unified profiler
        
        Args:
            output_dir: Directory to save profiling results
            model_name: Name prefix for output files
            enable_system_profiling: Enable system resource monitoring
        """
        self.output_dir = ensure_directory(output_dir)
        self.model_name = model_name
        self.enable_system_profiling = enable_system_profiling
        
        # Create subdirectories
        self.pytorch_dir = ensure_directory(os.path.join(output_dir, "pytorch"))
        self.onnx_dir = ensure_directory(os.path.join(output_dir, "onnx"))
        self.reports_dir = ensure_directory(os.path.join(output_dir, "reports"))
        
        # Initialize components
        self.metrics_collector = MetricsCollector(output_dir)
        self.html_generator = HTMLReportGenerator(self.reports_dir)
        
        # Active profilers
        self.pytorch_profiler: Optional[PyTorchProfiler] = None
        self.onnx_profiler: Optional[ONNXProfiler] = None
        
        # Results storage
        self.profile_results: Dict[str, Any] = {}
    
    def create_profiler_for_model(self, model: Any) -> Union[PyTorchProfiler, ONNXProfiler]:
        """Create appropriate profiler based on model type"""
        model_type = get_model_type(model)
        
        if model_type == "pytorch":
            if self.pytorch_profiler is None:
                self.pytorch_profiler = PyTorchProfiler(
                    self.pytorch_dir, 
                    f"{self.model_name}_pytorch"
                )
            return self.pytorch_profiler
        
        elif model_type == "onnx":
            if self.onnx_profiler is None:
                self.onnx_profiler = ONNXProfiler(
                    self.onnx_dir,
                    f"{self.model_name}_onnx"
                )
            return self.onnx_profiler
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: pytorch, onnx")
    
    def profile_complete_pipeline(self, 
                                 model_loader_func, 
                                 model_loader_args: tuple = (),
                                 model_loader_kwargs: dict = None,
                                 dataloader: DataLoader = None,
                                 num_batches: Optional[int] = None) -> Dict[str, Any]:
        """
        Profile complete inference pipeline including model loading and inference
        
        Args:
            model_loader_func: Function to load the model
            model_loader_args: Arguments for model loader function
            model_loader_kwargs: Keyword arguments for model loader function
            dataloader: DataLoader for inference profiling
            num_batches: Number of batches to profile (None for all)
            
        Returns:
            Dict containing profiling results
        """
        if model_loader_kwargs is None:
            model_loader_kwargs = {}
        
        # Load model and create profiler
        print(f"Loading model for profiling...")
        model = model_loader_func(*model_loader_args, **model_loader_kwargs)
        profiler = self.create_profiler_for_model(model)
        
        model_type = get_model_type(model)
        print(f"Detected model type: {model_type}")
        
        # Profile model loading (re-create model for timing)
        print(f"Profiling model loading...")
        profiler.profile_model_loading(model_loader_func, *model_loader_args, **model_loader_kwargs)
        
        results = {
            "model_type": model_type,
            "model_info": profiler.metrics.model_info,
            "load_time": profiler.metrics.model_load_time
        }
        
        # Profile inference if dataloader provided
        if dataloader is not None:
            print(f"Profiling inference with {num_batches or 'all'} batches...")
            inference_results = profiler.profile_inference(model, dataloader, num_batches)
            results.update(inference_results)
        
        # Collect metrics
        self.metrics_collector.add_metrics(profiler.get_metrics(), source=model_type)
        
        # Store results
        timestamp = generate_timestamp()
        self.profile_results[f"{model_type}_{timestamp}"] = results
        
        # Save metrics to file
        metrics_file = profiler.save_metrics()
        results["metrics_file"] = metrics_file
        
        print(f"Profiling completed for {model_type} model")
        print(f"Results saved to: {profiler.output_dir}")
        
        return results
    
    def profile_model_only(self, model: Any, dataloader: DataLoader, 
                          num_batches: Optional[int] = None) -> Dict[str, Any]:
        """
        Profile inference only (model already loaded)
        
        Args:
            model: Pre-loaded model
            dataloader: DataLoader for inference
            num_batches: Number of batches to profile
            
        Returns:
            Dict containing profiling results
        """
        profiler = self.create_profiler_for_model(model)
        model_type = get_model_type(model)
        
        print(f"Profiling {model_type} model inference...")
        
        # Profile inference
        results = profiler.profile_inference(model, dataloader, num_batches)
        results["model_type"] = model_type
        
        # Collect metrics
        self.metrics_collector.add_metrics(profiler.get_metrics(), source=model_type)
        
        # Save metrics
        metrics_file = profiler.save_metrics()
        results["metrics_file"] = metrics_file
        
        return results
    
    def compare_models(self, 
                      pytorch_model_loader = None, 
                      pytorch_args: tuple = (),
                      pytorch_kwargs: dict = None,
                      onnx_model_loader = None,
                      onnx_args: tuple = (),
                      onnx_kwargs: dict = None,
                      dataloader: DataLoader = None,
                      num_batches: Optional[int] = None) -> ComparisonMetrics:
        """
        Compare PyTorch and ONNX models side by side
        
        Returns:
            ComparisonMetrics object with detailed comparison
        """
        print("Starting model comparison...")
        
        pytorch_results = None
        onnx_results = None
        
        # Profile PyTorch model
        if pytorch_model_loader is not None:
            print("Profiling PyTorch model...")
            if pytorch_kwargs is None:
                pytorch_kwargs = {}
            
            pytorch_results = self.profile_complete_pipeline(
                pytorch_model_loader, pytorch_args, pytorch_kwargs,
                dataloader, num_batches
            )
        
        # Profile ONNX model
        if onnx_model_loader is not None:
            print("Profiling ONNX model...")
            if onnx_kwargs is None:
                onnx_kwargs = {}
                
            onnx_results = self.profile_complete_pipeline(
                onnx_model_loader, onnx_args, onnx_kwargs,
                dataloader, num_batches
            )
        
        # Generate comparison
        comparison = self.metrics_collector.compare_pytorch_vs_onnx()
        
        # Save comparison report
        comparison_file = self.metrics_collector.save_comparison_report(comparison)
        print(f"Comparison report saved to: {comparison_file}")
        
        return comparison
    
    def generate_html_report(self, include_traces: bool = True) -> str:
        """Generate comprehensive HTML report"""
        print("Generating HTML report...")
        
        # Collect all data for report
        report_data = {
            "summary": self.metrics_collector.generate_summary_report(),
            "comparison": None,
            "pytorch_traces": [],
            "onnx_traces": []
        }
        
        # Add comparison if we have both model types
        pytorch_metrics = self.metrics_collector.get_latest_metrics_by_type('pytorch')
        onnx_metrics = self.metrics_collector.get_latest_metrics_by_type('onnx')
        
        if pytorch_metrics and onnx_metrics:
            report_data["comparison"] = self.metrics_collector.compare_pytorch_vs_onnx()
        
        # Collect trace files if requested
        if include_traces:
            if self.pytorch_profiler:
                trace_file = self.pytorch_profiler.generate_trace_file()
                if trace_file and os.path.exists(trace_file):
                    report_data["pytorch_traces"].append(trace_file)
            
            if self.onnx_profiler:
                trace_file = self.onnx_profiler.generate_trace_file()
                if trace_file and os.path.exists(trace_file):
                    report_data["onnx_traces"].append(trace_file)
                
                # Also add Chrome-formatted trace if available
                chrome_trace = self.onnx_profiler.convert_onnx_trace_to_chrome()
                if chrome_trace and os.path.exists(chrome_trace):
                    report_data["onnx_traces"].append(chrome_trace)
        
        # Generate HTML report
        report_file = self.html_generator.generate_comprehensive_report(
            report_data, 
            f"{self.model_name}_profiling_report"
        )
        
        print(f"HTML report generated: {report_file}")
        return report_file
    
    def export_results(self, format: str = "json") -> Dict[str, str]:
        """
        Export results in multiple formats
        
        Args:
            format: Export format ("json", "csv", "html", "all")
            
        Returns:
            Dict mapping format to file path
        """
        exports = {}
        print(f"DEBUG: Starting export_results with format={format}")
        
        if format in ["json", "all"]:
            print("DEBUG: Processing JSON export...")
            
            # Try simple metrics save first
            try:
                print("DEBUG: Attempting to save individual profiler metrics...")
                # Save metrics from each profiler directly
                if self.pytorch_profiler:
                    pt_metrics_file = self.pytorch_profiler.save_metrics()
                    print(f"DEBUG: PyTorch metrics saved: {pt_metrics_file}")
                    exports["pytorch_metrics"] = pt_metrics_file
                
                if self.onnx_profiler:
                    onnx_metrics_file = self.onnx_profiler.save_metrics()
                    print(f"DEBUG: ONNX metrics saved: {onnx_metrics_file}")
                    exports["onnx_metrics"] = onnx_metrics_file
                    
            except Exception as e:
                print(f"Warning: Failed to save individual metrics: {e}")
                import traceback
                print(f"DEBUG: Individual metrics error: {traceback.format_exc()}")
            
            # Generate summary report in JSON format
            try:
                print("DEBUG: Generating JSON summary report...")
                summary_file = os.path.join(self.reports_dir, f"{self.model_name}_summary_{generate_timestamp()}.json")
                
                # Simple summary data without complex operations
                summary_data = {
                    "generated_at": generate_timestamp(),
                    "model_name": self.model_name,
                    "profiler_type": "unified",
                    "metrics_files": exports.copy()
                }
                
                # Add basic metrics if available
                if self.onnx_profiler and hasattr(self.onnx_profiler, 'metrics'):
                    metrics = self.onnx_profiler.metrics
                    summary_data["onnx_summary"] = {
                        "model_type": metrics.model_type,
                        "total_inference_time": metrics.total_inference_time,
                        "batches_processed": metrics.batches_processed,
                        "samples_processed": metrics.samples_processed,
                        "throughput": metrics.throughput,
                        "peak_memory_mb": metrics.peak_memory_usage
                    }
                
                with open(summary_file, 'w') as f:
                    import json
                    json.dump(summary_data, f, indent=2, default=str)
                
                exports["summary_report"] = summary_file
                print(f"DEBUG: Summary report saved: {summary_file}")
                
            except Exception as e:
                print(f"Warning: Failed to generate summary report: {e}")
        
        # Skip HTML generation for now as it causes hangs
        if format == "html":
            print("DEBUG: HTML generation temporarily disabled to avoid hangs")
        
        print(f"DEBUG: export_results completed, returning {len(exports)} files")
        return exports
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a quick summary of all profiling results"""
        return {
            "total_runs": len(self.metrics_collector.metrics_history),
            "model_types_profiled": list(set(m.get('model_type', 'unknown') 
                                           for m in self.metrics_collector.metrics_history)),
            "output_directory": self.output_dir,
            "available_results": list(self.profile_results.keys()),
            "latest_pytorch_metrics": self.metrics_collector.get_latest_metrics_by_type('pytorch'),
            "latest_onnx_metrics": self.metrics_collector.get_latest_metrics_by_type('onnx')
        }
    
    def cleanup(self):
        """Clean up profiler resources"""
        if self.pytorch_profiler:
            # PyTorch profiler cleanup is automatic
            pass
        
        if self.onnx_profiler:
            # ONNX profiler cleanup
            if hasattr(self.onnx_profiler, 'session_info') and 'session' in self.onnx_profiler.session_info:
                # Session cleanup is handled by ONNX Runtime
                pass
        
        print("Profiler cleanup completed")