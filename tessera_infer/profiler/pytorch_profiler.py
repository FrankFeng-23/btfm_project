#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import torch
from typing import Any, Dict, Optional
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity, schedule

from .base_profiler import BaseProfiler, ProfilerContext
from .utils import format_time, generate_timestamp

class PyTorchProfiler(BaseProfiler):
    """PyTorch-specific profiler using torch.profiler"""
    
    def __init__(self, output_dir: str, model_name: str = "pytorch_model"):
        super().__init__(output_dir, model_name)
        self.metrics.model_type = "PyTorch"
        self.torch_profiler = None
        self.trace_file = None
        
    def profile_model_loading(self, model_loader_func, *args, **kwargs) -> Any:
        """Profile PyTorch model loading"""
        with ProfilerContext(self, "model_loading"):
            with record_function("model_loading"):
                model = model_loader_func(*args, **kwargs)
                
                # Get model info
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                self.metrics.model_info = {
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "model_class": model.__class__.__name__,
                    "device": next(model.parameters()).device.type if list(model.parameters()) else "unknown"
                }
                
                return model
    
    def profile_inference(self, model: Any, dataloader: DataLoader, 
                         num_batches: Optional[int] = None) -> Dict[str, Any]:
        """Profile PyTorch inference with detailed timeline"""
        
        # Start system monitoring
        self.start_monitoring()
        
        if num_batches is None:
            num_batches = min(len(dataloader), 50)  # Default to 50 batches
        
        # Setup profiler schedule
        prof_schedule = schedule(
            skip_first=3,   # Skip first 3 batches (warmup)
            wait=1,         # Wait 1 batch
            warmup=2,       # Warmup for 2 batches  
            active=3,       # Profile 3 batches
            repeat=2        # Repeat cycle 2 times
        )
        
        # Create output files
        timestamp = generate_timestamp()
        self.trace_file = os.path.join(
            self.output_dir, 
            f"{self.model_name}_torch_trace_{timestamp}.json"
        )
        
        batch_times = []
        total_samples = 0
        
        with ProfilerContext(self, "inference"):
            # Create torch profiler
            with profile(
                activities=[ProfilerActivity.CPU],
                schedule=prof_schedule,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                on_trace_ready=self._save_trace
            ) as prof:
                
                model.eval()
                
                # Warmup phase
                with ProfilerContext(self, "model_warmup"):
                    with record_function("warmup"):
                        with torch.no_grad():
                            for i, batch_data in enumerate(dataloader):
                                if i >= 2:  # 2 warmup batches
                                    break
                                
                                # Extract inputs based on your data format
                                s2_input = batch_data.get("s2_input", batch_data.get("s2"))
                                s1_input = batch_data.get("s1_input", batch_data.get("s1"))
                                
                                if s2_input is not None and s1_input is not None:
                                    _ = model(s2_input, s1_input)
                
                # Main profiling phase
                with torch.no_grad():
                    for batch_idx, batch_data in enumerate(dataloader):
                        if batch_idx >= num_batches:
                            break
                        
                        batch_start_time = time.time()
                        
                        with record_function(f"batch_{batch_idx}"):
                            # Data preparation
                            with record_function("data_preparation"):
                                s2_input = batch_data.get("s2_input", batch_data.get("s2"))
                                s1_input = batch_data.get("s1_input", batch_data.get("s1"))
                                
                                if s2_input is None or s1_input is None:
                                    continue
                                
                                batch_size = s2_input.shape[0]
                                total_samples += batch_size
                            
                            # Model inference
                            with record_function("model_forward"):
                                outputs = model(s2_input, s1_input)
                        
                        batch_time = time.time() - batch_start_time
                        batch_times.append(batch_time)
                        
                        # Step profiler
                        prof.step()
                
                self.torch_profiler = prof
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Calculate metrics
        if batch_times:
            self.metrics.avg_batch_time = sum(batch_times) / len(batch_times)
        
        self.metrics.batches_processed = len(batch_times)
        self.metrics.samples_processed = total_samples
        
        if self.metrics.total_inference_time > 0:
            self.metrics.throughput = total_samples / self.metrics.total_inference_time
        
        return {
            "trace_file": self.trace_file,
            "avg_batch_time": self.metrics.avg_batch_time,
            "total_samples": total_samples,
            "batches_processed": len(batch_times)
        }
    
    def _save_trace(self, prof):
        """Callback to save trace file"""
        if self.trace_file:
            prof.export_chrome_trace(self.trace_file)
    
    def generate_trace_file(self) -> str:
        """Return the path to the generated trace file"""
        return self.trace_file or ""
    
    def export_memory_timeline(self, filename: Optional[str] = None) -> Optional[str]:
        """Export memory timeline as HTML"""
        if not self.torch_profiler:
            return None
        
        if filename is None:
            timestamp = generate_timestamp()
            filename = f"{self.model_name}_memory_timeline_{timestamp}.html"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            self.torch_profiler.export_memory_timeline(filepath)
            return filepath
        except Exception as e:
            print(f"Failed to export memory timeline: {e}")
            return None
    
    def get_key_averages(self) -> Optional[Dict[str, Any]]:
        """Get key averages from profiler"""
        if not self.torch_profiler:
            return None
        
        key_averages = self.torch_profiler.key_averages()
        
        # Convert to dictionary format
        results = {
            "total_events": len(key_averages),
            "top_cpu_events": [],
            "top_memory_events": []
        }
        
        # Sort by CPU time
        cpu_sorted = sorted(key_averages, key=lambda x: x.cpu_time_total, reverse=True)[:10]
        for event in cpu_sorted:
            results["top_cpu_events"].append({
                "name": event.key,
                "cpu_time_total": event.cpu_time_total,
                "cpu_time_avg": event.cpu_time,
                "count": event.count,
                "self_cpu_time_total": event.self_cpu_time_total
            })
        
        # Sort by memory usage (if available)
        memory_sorted = [e for e in key_averages if hasattr(e, 'cpu_memory_usage') and e.cpu_memory_usage > 0]
        memory_sorted = sorted(memory_sorted, key=lambda x: x.cpu_memory_usage, reverse=True)[:10]
        for event in memory_sorted:
            results["top_memory_events"].append({
                "name": event.key,
                "cpu_memory_usage": event.cpu_memory_usage,
                "count": event.count
            })
        
        return results
    
    def profile_model_with_shapes(self, model: Any, sample_inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Profile model with specific input shapes"""
        
        with record_function("shape_profiling"):
            model.eval()
            
            with torch.no_grad():
                # Record input shapes
                input_shapes = {}
                for key, tensor in sample_inputs.items():
                    input_shapes[key] = list(tensor.shape)
                
                # Profile single forward pass
                start_time = time.time()
                outputs = model(**sample_inputs)
                forward_time = time.time() - start_time
                
                # Record output shapes
                output_shapes = {}
                if isinstance(outputs, torch.Tensor):
                    output_shapes["output"] = list(outputs.shape)
                elif isinstance(outputs, (tuple, list)):
                    for i, output in enumerate(outputs):
                        if isinstance(output, torch.Tensor):
                            output_shapes[f"output_{i}"] = list(output.shape)
                elif isinstance(outputs, dict):
                    for key, output in outputs.items():
                        if isinstance(output, torch.Tensor):
                            output_shapes[key] = list(output.shape)
        
        return {
            "input_shapes": input_shapes,
            "output_shapes": output_shapes,
            "forward_time": forward_time,
            "model_info": self.metrics.model_info
        }