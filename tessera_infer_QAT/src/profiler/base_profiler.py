#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import psutil
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ProfileMetrics:
    """Data class for storing profiling metrics"""
    model_type: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Timing metrics (in seconds)
    model_load_time: float = 0.0
    model_warmup_time: float = 0.0
    total_inference_time: float = 0.0
    avg_batch_time: float = 0.0
    data_loading_time: float = 0.0
    
    # Memory metrics (in MB)
    peak_memory_usage: float = 0.0
    initial_memory: float = 0.0
    memory_growth: float = 0.0
    
    # Performance metrics
    batches_processed: int = 0
    samples_processed: int = 0
    throughput: float = 0.0  # samples per second
    
    # CPU metrics
    avg_cpu_usage: float = 0.0
    cpu_cores_used: int = 0
    
    # Model specific metrics
    model_info: Dict[str, Any] = field(default_factory=dict)
    
    # Additional custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

class BaseProfiler(ABC):
    """Base class for all profilers"""
    
    def __init__(self, output_dir: str, model_name: str = "model"):
        self.output_dir = output_dir
        self.model_name = model_name
        self.metrics = ProfileMetrics(model_type=self.__class__.__name__)
        
        # System monitoring
        self._monitor_thread = None
        self._monitoring = False
        self._cpu_samples = []
        self._memory_samples = []
        
        # Timing contexts
        self._timers = {}
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def start_timer(self, name: str):
        """Start a named timer"""
        self._timers[name] = time.time()
        
    def end_timer(self, name: str) -> float:
        """End a named timer and return elapsed time"""
        if name not in self._timers:
            return 0.0
        elapsed = time.time() - self._timers[name]
        del self._timers[name]
        return elapsed
        
    def _monitor_system_resources(self):
        """Monitor system resources in background thread"""
        process = psutil.Process()
        
        while self._monitoring:
            try:
                # CPU usage
                cpu_percent = process.cpu_percent()
                self._cpu_samples.append(cpu_percent)
                
                # Memory usage (RSS in MB)
                memory_mb = process.memory_info().rss / (1024 * 1024)
                self._memory_samples.append(memory_mb)
                
                time.sleep(0.1)  # Sample every 100ms
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
    
    def start_monitoring(self):
        """Start system resource monitoring"""
        if not self._monitoring:
            self._monitoring = True
            self._cpu_samples.clear()
            self._memory_samples.clear()
            self.metrics.initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            self._monitor_thread = threading.Thread(target=self._monitor_system_resources, daemon=True)
            self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system resource monitoring and collect metrics"""
        if self._monitoring:
            self._monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=1.0)
            
            # Calculate metrics
            if self._cpu_samples:
                self.metrics.avg_cpu_usage = sum(self._cpu_samples) / len(self._cpu_samples)
            
            if self._memory_samples:
                self.metrics.peak_memory_usage = max(self._memory_samples)
                self.metrics.memory_growth = self.metrics.peak_memory_usage - self.metrics.initial_memory
            
            # CPU cores
            self.metrics.cpu_cores_used = psutil.cpu_count()
    
    @abstractmethod
    def profile_model_loading(self, *args, **kwargs) -> Any:
        """Profile model loading phase"""
        pass
    
    @abstractmethod
    def profile_inference(self, model: Any, dataloader: Any, num_batches: Optional[int] = None) -> Dict[str, Any]:
        """Profile inference phase"""
        pass
    
    @abstractmethod
    def generate_trace_file(self) -> str:
        """Generate trace file for timeline visualization"""
        pass
    
    def get_metrics(self) -> ProfileMetrics:
        """Get collected metrics"""
        return self.metrics
    
    def save_metrics(self, filename: Optional[str] = None) -> str:
        """Save metrics to JSON file"""
        import json
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_name}_metrics_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert dataclass to dict for JSON serialization
        metrics_dict = {
            'model_type': self.metrics.model_type,
            'timestamp': self.metrics.timestamp,
            'model_load_time': self.metrics.model_load_time,
            'model_warmup_time': self.metrics.model_warmup_time,
            'total_inference_time': self.metrics.total_inference_time,
            'avg_batch_time': self.metrics.avg_batch_time,
            'data_loading_time': self.metrics.data_loading_time,
            'peak_memory_usage': self.metrics.peak_memory_usage,
            'initial_memory': self.metrics.initial_memory,
            'memory_growth': self.metrics.memory_growth,
            'batches_processed': self.metrics.batches_processed,
            'samples_processed': self.metrics.samples_processed,
            'throughput': self.metrics.throughput,
            'avg_cpu_usage': self.metrics.avg_cpu_usage,
            'cpu_cores_used': self.metrics.cpu_cores_used,
            'model_info': self.metrics.model_info,
            'custom_metrics': self.metrics.custom_metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        return filepath

class ProfilerContext:
    """Context manager for profiling operations"""
    
    def __init__(self, profiler: BaseProfiler, operation_name: str):
        self.profiler = profiler
        self.operation_name = operation_name
        
    def __enter__(self):
        self.profiler.start_timer(self.operation_name)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = self.profiler.end_timer(self.operation_name)
        
        # Store timing in appropriate metric field
        if self.operation_name == "model_loading":
            self.profiler.metrics.model_load_time = elapsed
        elif self.operation_name == "model_warmup":
            self.profiler.metrics.model_warmup_time = elapsed
        elif self.operation_name == "inference":
            self.profiler.metrics.total_inference_time = elapsed
        elif self.operation_name == "data_loading":
            self.profiler.metrics.data_loading_time = elapsed
        else:
            # Store in custom metrics
            self.profiler.metrics.custom_metrics[f"{self.operation_name}_time"] = elapsed