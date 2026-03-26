#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import logging
import psutil
import importlib.util
from datetime import datetime


class CustomFormatter(logging.Formatter):
    """Custom logger to control verbosity"""
    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        process_info = f"[P{record.process}]" if hasattr(record, 'process') else ""
        return f"{timestamp} - {record.levelname} - {process_info} {record.getMessage()}"


def force_log_flush():
    """Force all handlers to flush their logs"""
    for handler in logging.root.handlers:
        handler.flush()


def get_gpu_memory_info(device_id=None):
    """Get GPU memory information for the specified device"""
    try:
        import torch
        if not torch.cuda.is_available():
            return "CUDA not available"
            
        if device_id is None:
            device_id = torch.cuda.current_device()
            
        # Get memory information for this device
        mem_allocated = torch.cuda.memory_allocated(device_id) / (1024**2)  # MB
        mem_reserved = torch.cuda.memory_reserved(device_id) / (1024**2)    # MB
        mem_max_allocated = torch.cuda.max_memory_allocated(device_id) / (1024**2)  # MB
        
        # Get total memory
        total_mem = "N/A"
        try:
            prop = torch.cuda.get_device_properties(device_id)
            total_mem = prop.total_memory / (1024**2)  # MB
            
            return f"GPU {device_id} Memory: {mem_allocated:.1f}MB allocated, {mem_reserved:.1f}MB reserved, {mem_max_allocated:.1f}MB peak, {total_mem:.1f}MB total"
        except:
            return f"GPU {device_id} Memory: {mem_allocated:.1f}MB allocated, {mem_reserved:.1f}MB reserved, {mem_max_allocated:.1f}MB peak"
    except Exception as e:
        return f"Error getting GPU memory info: {str(e)}"


def log_system_info():
    """Log system information including CPU and memory usage"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        mem_percent = mem.percent
        mem_used = mem.used / (1024**3)  # GB
        mem_total = mem.total / (1024**3)  # GB
        
        return f"System: CPU {cpu_percent:.1f}%, Memory {mem_percent:.1f}% ({mem_used:.1f}GB/{mem_total:.1f}GB)"
    except:
        return "Could not get system info"


def format_time(seconds):
    """Format seconds into readable time format."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{seconds:.1f}s"


def load_config_module(config_file_path):
    """Load configuration module dynamically"""
    try:
        if not os.path.exists(config_file_path):
            logging.error(f"Config file does not exist: {config_file_path}")
            raise FileNotFoundError(f"Config file does not exist: {config_file_path}")
            
        spec = importlib.util.spec_from_file_location("my_dynamic_config", config_file_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        raise


def setup_logging(args, mode_prefix="CPU"):
    """Setup logging configuration"""
    # Set logging level
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    logging.root.setLevel(numeric_level)
    
    # Setup logging with file handler
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Simplified logging setup
    if getattr(args, 'simplified_logging', False):
        log_file = f"{log_dir}/infer_qat_{args.mode}_{args.process_id}.log"
        
        # Remove all existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Add single file handler
        file_handler = logging.FileHandler(log_file)
        file_formatter = CustomFormatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)
        
        # Also add stream handler for important messages
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(file_formatter)
        logging.root.addHandler(stream_handler)
    else:
        # Keep original logging setup
        log_file = f"{log_dir}/infer_qat_process_{args.process_id}_{args.mode}.log"
        
        # Only add file handler if not already present
        has_file_handler = False
        for handler in logging.root.handlers:
            if isinstance(handler, logging.FileHandler):
                has_file_handler = True
                break
                
        if not has_file_handler:
            file_handler = logging.FileHandler(log_file)
            file_formatter = CustomFormatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logging.root.addHandler(file_handler)