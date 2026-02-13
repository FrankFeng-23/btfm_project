#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import json
import numpy as np
from typing import Any, Dict, Optional
from torch.utils.data import DataLoader

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from .base_profiler import BaseProfiler, ProfilerContext
from .utils import format_time, generate_timestamp

class ONNXProfiler(BaseProfiler):
    """ONNX Runtime specific profiler"""
    
    def __init__(self, output_dir: str, model_name: str = "onnx_model"):
        super().__init__(output_dir, model_name)
        self.metrics.model_type = "ONNX"
        self.trace_file = None
        self.session_info = {}
        
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime is not available. Please install onnxruntime.")
    
    def profile_model_loading(self, model_path: str, num_threads: Optional[int] = None, 
                            **session_options) -> Any:
        """Profile ONNX model loading and session creation"""
        with ProfilerContext(self, "model_loading"):
            # Configure session options for profiling
            sess_options = ort.SessionOptions()
            
            # Enable profiling
            sess_options.enable_profiling = True
            
            # Set threading
            if num_threads is not None:
                sess_options.intra_op_num_threads = num_threads
                sess_options.inter_op_num_threads = 1
            
            # Optimization settings
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_cpu_mem_arena = True
            sess_options.enable_mem_pattern = True
            sess_options.enable_mem_reuse = True
            
            # Apply additional session options
            for key, value in session_options.items():
                if hasattr(sess_options, key):
                    setattr(sess_options, key, value)
            
            # CPU execution provider with optimizations
            providers = [
                ('CPUExecutionProvider', {
                    'use_arena': True,
                    'arena_extend_strategy': 'kSameAsRequested',
                    'initial_chunk_size_bytes': 1024 * 1024,
                    'max_chunk_size_bytes': 32 * 1024 * 1024,
                })
            ]
            
            # Create session
            session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            # Collect model information
            self.metrics.model_info = {
                "model_path": model_path,
                "input_names": [inp.name for inp in session.get_inputs()],
                "output_names": [out.name for out in session.get_outputs()],
                "num_threads": num_threads,
                "providers": session.get_providers()
            }
            
            # Collect input/output metadata
            inputs_info = []
            for inp in session.get_inputs():
                inputs_info.append({
                    "name": inp.name,
                    "type": str(inp.type),
                    "shape": inp.shape
                })
            
            outputs_info = []
            for out in session.get_outputs():
                outputs_info.append({
                    "name": out.name,
                    "type": str(out.type), 
                    "shape": out.shape
                })
            
            self.metrics.model_info.update({
                "inputs": inputs_info,
                "outputs": outputs_info
            })
            
            self.session_info = {
                "session": session,
                "input_names": [inp.name for inp in session.get_inputs()],
                "output_names": [out.name for out in session.get_outputs()]
            }
            
            return session
    
    def profile_inference(self, model: Any, dataloader: DataLoader, 
                         num_batches: Optional[int] = None) -> Dict[str, Any]:
        """Profile ONNX inference"""
        
        if not isinstance(model, ort.InferenceSession):
            # Handle wrapper classes like ONNXInferenceModel
            if hasattr(model, 'session') and isinstance(model.session, ort.InferenceSession):
                session = model.session
                input_names = getattr(model, 'input_names', None)
                output_names = getattr(model, 'output_names', None)
            else:
                raise ValueError("Model must be an ONNX Runtime InferenceSession or wrapper")
        else:
            session = model
            input_names = [inp.name for inp in session.get_inputs()]
            output_names = [out.name for out in session.get_outputs()]
        
        # Start system monitoring
        self.start_monitoring()
        
        if num_batches is None:
            num_batches = min(len(dataloader), 50)
        
        batch_times = []
        total_samples = 0
        inference_times = []
        
        with ProfilerContext(self, "inference"):
            # Warmup phase
            with ProfilerContext(self, "model_warmup"):
                for i, batch_data in enumerate(dataloader):
                    if i >= 2:  # 2 warmup batches
                        break
                    
                    s2_input = batch_data.get("s2_input", batch_data.get("s2"))
                    s1_input = batch_data.get("s1_input", batch_data.get("s1"))
                    
                    if s2_input is not None and s1_input is not None:
                        # Convert to numpy if needed
                        if hasattr(s2_input, 'numpy'):
                            s2_np = s2_input.numpy()
                            s1_np = s1_input.numpy()
                        else:
                            s2_np = s2_input
                            s1_np = s1_input
                        
                        # Ensure correct data type
                        s2_np = np.ascontiguousarray(s2_np, dtype=np.float32)
                        s1_np = np.ascontiguousarray(s1_np, dtype=np.float32)
                        
                        # Run inference
                        ort_inputs = {
                            input_names[0]: s2_np,
                            input_names[1]: s1_np
                        }
                        _ = session.run(output_names, ort_inputs)
            
            # Main profiling phase
            for batch_idx, batch_data in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                
                batch_start_time = time.time()
                
                # Data preparation
                data_prep_start = time.time()
                s2_input = batch_data.get("s2_input", batch_data.get("s2"))
                s1_input = batch_data.get("s1_input", batch_data.get("s1"))
                
                if s2_input is None or s1_input is None:
                    continue
                
                # Convert to numpy if needed
                if hasattr(s2_input, 'numpy'):
                    s2_np = s2_input.numpy()
                    s1_np = s1_input.numpy()
                else:
                    s2_np = s2_input
                    s1_np = s1_input
                
                # Ensure correct data type and layout
                s2_np = np.ascontiguousarray(s2_np, dtype=np.float32)
                s1_np = np.ascontiguousarray(s1_np, dtype=np.float32)
                
                batch_size = s2_np.shape[0]
                total_samples += batch_size
                
                data_prep_time = time.time() - data_prep_start
                
                # ONNX inference
                inference_start = time.time()
                ort_inputs = {
                    input_names[0]: s2_np,
                    input_names[1]: s1_np
                }
                outputs = session.run(output_names, ort_inputs)
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
                
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                
                # Store timing breakdown for first few batches
                if batch_idx < 5:
                    self.metrics.custom_metrics[f"batch_{batch_idx}_data_prep_time"] = data_prep_time
                    self.metrics.custom_metrics[f"batch_{batch_idx}_inference_time"] = inference_time
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Get profiling trace file
        self.trace_file = session.end_profiling()
        
        # Calculate metrics
        if batch_times:
            self.metrics.avg_batch_time = sum(batch_times) / len(batch_times)
        
        if inference_times:
            self.metrics.custom_metrics["avg_inference_time"] = sum(inference_times) / len(inference_times)
            self.metrics.custom_metrics["min_inference_time"] = min(inference_times)
            self.metrics.custom_metrics["max_inference_time"] = max(inference_times)
        
        self.metrics.batches_processed = len(batch_times)
        self.metrics.samples_processed = total_samples
        
        if self.metrics.total_inference_time > 0:
            self.metrics.throughput = total_samples / self.metrics.total_inference_time
        
        return {
            "trace_file": self.trace_file,
            "avg_batch_time": self.metrics.avg_batch_time,
            "total_samples": total_samples,
            "batches_processed": len(batch_times),
            "avg_inference_time": self.metrics.custom_metrics.get("avg_inference_time", 0)
        }
    
    def generate_trace_file(self) -> str:
        """Return the path to the ONNX profiling trace file"""
        return self.trace_file or ""
    
    def convert_onnx_trace_to_chrome(self, output_path: Optional[str] = None) -> Optional[str]:
        """Convert ONNX profiling JSON to Chrome trace format"""
        if not self.trace_file or not os.path.exists(self.trace_file):
            return None
        
        if output_path is None:
            timestamp = generate_timestamp()
            output_path = os.path.join(
                self.output_dir,
                f"{self.model_name}_chrome_trace_{timestamp}.json"
            )
        
        try:
            # Read ONNX profiling file
            with open(self.trace_file, 'r') as f:
                onnx_data = json.load(f)
            
            # Convert to Chrome trace format
            chrome_events = []
            
            # Process ONNX events
            if isinstance(onnx_data, list):
                events = onnx_data
            elif isinstance(onnx_data, dict) and 'events' in onnx_data:
                events = onnx_data['events']
            else:
                events = []
            
            for event in events:
                if not isinstance(event, dict):
                    continue
                
                # Convert ONNX event to Chrome trace event
                chrome_event = {
                    "name": event.get("name", event.get("op_name", "Unknown")),
                    "cat": "ONNX",
                    "ph": "X",  # Complete event
                    "ts": int(event.get("ts", 0) * 1000000),  # Convert to microseconds
                    "dur": int(event.get("dur", 0) * 1000000),  # Convert to microseconds
                    "pid": 1,
                    "tid": event.get("thread_id", 1),
                    "args": {
                        key: value for key, value in event.items() 
                        if key not in ["name", "op_name", "ts", "dur", "thread_id"]
                    }
                }
                chrome_events.append(chrome_event)
            
            # Create Chrome trace format
            chrome_trace = {
                "traceEvents": chrome_events,
                "displayTimeUnit": "ns",
                "systemTraceEvents": "CpuCores",
                "otherData": {
                    "version": "Chrome Trace Format",
                    "creator": "Tessera ONNX Profiler"
                }
            }
            
            # Write Chrome trace file
            with open(output_path, 'w') as f:
                json.dump(chrome_trace, f, separators=(',', ':'))
            
            return output_path
            
        except Exception as e:
            print(f"Failed to convert ONNX trace to Chrome format: {e}")
            return None
    
    def analyze_onnx_ops(self) -> Optional[Dict[str, Any]]:
        """Analyze ONNX operators performance from trace file"""
        if not self.trace_file or not os.path.exists(self.trace_file):
            return None
        
        try:
            with open(self.trace_file, 'r') as f:
                data = json.load(f)
            
            # Extract events
            if isinstance(data, list):
                events = data
            elif isinstance(data, dict) and 'events' in data:
                events = data['events']
            else:
                return None
            
            # Analyze operators
            op_stats = {}
            total_time = 0
            
            for event in events:
                if not isinstance(event, dict):
                    continue
                
                op_name = event.get("name", event.get("op_name", "Unknown"))
                duration = event.get("dur", 0)
                
                if op_name not in op_stats:
                    op_stats[op_name] = {
                        "count": 0,
                        "total_time": 0,
                        "min_time": float('inf'),
                        "max_time": 0
                    }
                
                op_stats[op_name]["count"] += 1
                op_stats[op_name]["total_time"] += duration
                op_stats[op_name]["min_time"] = min(op_stats[op_name]["min_time"], duration)
                op_stats[op_name]["max_time"] = max(op_stats[op_name]["max_time"], duration)
                
                total_time += duration
            
            # Calculate percentages and averages
            for op_name, stats in op_stats.items():
                stats["avg_time"] = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
                stats["percentage"] = (stats["total_time"] / total_time * 100) if total_time > 0 else 0
                
                # Fix min_time if no valid times were recorded
                if stats["min_time"] == float('inf'):
                    stats["min_time"] = 0
            
            # Sort by total time
            sorted_ops = sorted(op_stats.items(), key=lambda x: x[1]["total_time"], reverse=True)
            
            return {
                "total_operators": len(op_stats),
                "total_time": total_time,
                "operators": dict(sorted_ops[:20])  # Top 20 operators
            }
            
        except Exception as e:
            print(f"Failed to analyze ONNX operators: {e}")
            return None