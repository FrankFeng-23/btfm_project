#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys
import os
import torch


def check_amx_support():
    """Check if AMX is supported on current hardware"""
    try:
        # Check CPU flags for AMX support
        with open('/proc/cpuinfo', 'r', encoding='utf-8') as f:
            cpu_info = f.read().lower()
        
        # Look for AMX-related CPU flags
        amx_bf16_support = 'amx_bf16' in cpu_info
        amx_int8_support = 'amx_int8' in cpu_info or 'amx_tile' in cpu_info
        avx512_support = 'avx512f' in cpu_info
        
        return {
            'amx_bf16': amx_bf16_support,
            'amx_int8': amx_int8_support,
            'avx512': avx512_support,
            'supported': amx_bf16_support or amx_int8_support
        }
    except Exception as e:
        logging.warning(f"Could not detect AMX support: {e}")
        return {
            'amx_bf16': False,
            'amx_int8': False,
            'avx512': False,
            'supported': False
        }


def setup_amx_environment(args):
    """Setup AMX environment variables and optimizations"""
    if getattr(args, 'disable_amx', False):
        logging.info(f"[{args.mode.upper()}] [{args.process_id}] AMX auto-detection is disabled by user")
        args.amx_enabled = False
        return False

    amx_info = check_amx_support()
    
    if not amx_info['supported']:
        if getattr(args, 'enable_amx', False):
            logging.warning(f"[{args.mode.upper()}] [{args.process_id}] AMX requested but not supported on this hardware; falling back to standard CPU inference")
        else:
            logging.info(f"[{args.mode.upper()}] [{args.process_id}] AMX not detected on this CPU, using standard CPU inference")
        args.amx_enabled = False
        return False

    if getattr(args, 'enable_amx', False):
        logging.info(f"[{args.mode.upper()}] [{args.process_id}] AMX requested and supported, enabling AMX optimizations")
    else:
        logging.info(f"[{args.mode.upper()}] [{args.process_id}] AMX detected automatically, enabling AMX optimizations")
    
    # Enable oneDNN verbose logging for AMX verification
    if args.log_level == 'DEBUG':
        os.environ['ONEDNN_VERBOSE'] = '1'
        logging.debug(f"[{args.mode.upper()}] [{args.process_id}] Enabled oneDNN verbose logging for AMX verification")
    
    # Set oneDNN optimizations for AMX
    os.environ['DNNL_MAX_CPU_ISA'] = 'AMX_BF16' if amx_info['amx_bf16'] else 'AVX512_CORE'
    
    # Enable AMX tile configuration
    if amx_info['amx_bf16']:
        os.environ['ONEDNN_DEFAULT_FPMATH_MODE'] = 'BF16'
    
    logging.info(f"[{args.mode.upper()}] [{args.process_id}] AMX support detected - BF16: {amx_info['amx_bf16']}, INT8: {amx_info['amx_int8']}")

    args.amx_enabled = True
    return True


def setup_device(args):
    """Setup and configure the compute device"""
    mode_prefix = args.mode.upper()
    
    if args.mode == 'cpu':
        # Setup AMX environment if requested
        amx_enabled = setup_amx_environment(args)
        
        # Set the number of threads for CPU mode
        torch.set_num_threads(args.num_threads)
        device = torch.device("cpu")
        logging.info(f"[CPU] [{args.process_id}] Using CPU with {args.num_threads} threads")
        
        # Check BF16 compatibility for CPU
        if args.enable_bf16:
            if hasattr(torch.cpu, 'is_bf16_supported') and torch.cpu.is_bf16_supported():
                logging.info(f"[CPU] [{args.process_id}] BF16 is supported on CPU")
                if amx_enabled:
                    logging.info(f"[CPU] [{args.process_id}] AMX + BF16 optimizations enabled")
            else:
                logging.warning(f"[CPU] [{args.process_id}] BF16 may not be fully supported on this CPU, falling back to FP32")
        
        # Log AMX status
        if amx_enabled:
            logging.info(f"[CPU] [{args.process_id}] AMX acceleration enabled")
        elif getattr(args, 'disable_amx', False):
            logging.info(f"[CPU] [{args.process_id}] AMX disabled by user, using standard CPU inference")
        else:
            logging.info(f"[CPU] [{args.process_id}] Running with standard CPU inference")
        
        return device, mode_prefix
        
    elif args.mode == 'gpu':
        if not torch.cuda.is_available():
            logging.error(f"[GPU] [{args.process_id}] CUDA is not available but required for GPU mode")
            sys.exit(1)
            
        device_id = args.gpu_id
        if device_id >= torch.cuda.device_count():
            logging.error(f"[GPU] [{args.process_id}] GPU {device_id} is not available, max is {torch.cuda.device_count()-1}")
            sys.exit(1)
            
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device_id)
        device_name = torch.cuda.get_device_name(device_id)
        
        # Print detailed GPU info
        logging.info(f"[GPU] [{args.process_id}] Using GPU {device_id}: {device_name}")
        logging.info(f"[GPU] [{args.process_id}] CUDA version: {torch.version.cuda}")
        logging.info(f"[GPU] [{args.process_id}] PyTorch version: {torch.__version__}")
        
        # Check BF16 compatibility for GPU
        if args.enable_bf16:
            compute_capability = torch.cuda.get_device_capability(device_id)
            if compute_capability[0] >= 8:  # Ampere and newer GPUs
                logging.info(f"[GPU] [{args.process_id}] BF16 is supported on {device_name} (compute capability {compute_capability})")
            else:
                logging.warning(f"[GPU] [{args.process_id}] BF16 may not be efficient on {device_name} (compute capability {compute_capability}), consider using FP16 instead")
        
        return device, "GPU"
        
    elif args.mode == 'xpu':
        try:
            import intel_extension_for_pytorch as ipex
        except ImportError:
            logging.error(f"[XPU] [{args.process_id}] Intel Extension for PyTorch not available but required for XPU mode")
            sys.exit(1)
            
        if not torch.xpu.is_available():
            logging.error(f"[XPU] [{args.process_id}] XPU is not available but required for XPU mode")
            sys.exit(1)
            
        device_id = args.xpu_id
        if device_id >= torch.xpu.device_count():
            logging.error(f"[XPU] [{args.process_id}] XPU {device_id} is not available, max is {torch.xpu.device_count()-1}")
            sys.exit(1)
            
        device = torch.device(f"xpu:{device_id}")
        torch.xpu.set_device(device_id)
        device_properties = torch.xpu.get_device_properties(device_id)
        device_name = device_properties.name if hasattr(device_properties, 'name') else f"XPU {device_id}"
        
        # Print detailed XPU info
        logging.info(f"[XPU] [{args.process_id}] Using XPU {device_id}: {device_name}")
        logging.info(f"[XPU] [{args.process_id}] Intel PyTorch Extension version: {ipex.__version__}")
        logging.info(f"[XPU] [{args.process_id}] PyTorch version: {torch.__version__}")
        
        return device, "XPU"
        
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


def get_device_memory_info(device):
    """Get device memory information"""
    if device.type == 'cuda':
        try:
            device_id = device.index
            mem_allocated = torch.cuda.memory_allocated(device_id) / (1024**2)  # MB
            mem_reserved = torch.cuda.memory_reserved(device_id) / (1024**2)    # MB
            mem_max_allocated = torch.cuda.max_memory_allocated(device_id) / (1024**2)  # MB
            
            try:
                prop = torch.cuda.get_device_properties(device_id)
                total_mem = prop.total_memory / (1024**2)  # MB
                
                return f"GPU {device_id} Memory: {mem_allocated:.1f}MB allocated, {mem_reserved:.1f}MB reserved, {mem_max_allocated:.1f}MB peak, {total_mem:.1f}MB total"
            except:
                return f"GPU {device_id} Memory: {mem_allocated:.1f}MB allocated, {mem_reserved:.1f}MB reserved, {mem_max_allocated:.1f}MB peak"
        except Exception as e:
            return f"Error getting GPU memory info: {str(e)}"
    elif device.type == 'xpu':
        # XPU memory info would go here if available
        return f"XPU {device.index} info"
    else:
        return "CPU mode"