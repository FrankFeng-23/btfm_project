#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(description="Multi-tile QAT Inference (CPU/GPU/XPU)")
    
    # Core arguments
    parser.add_argument('--config', type=str, required=True, 
                        help="Path to config file")
    parser.add_argument('--mode', type=str, choices=['cpu', 'gpu', 'xpu'], required=True,
                        help="Processing mode: 'cpu', 'gpu', or 'xpu'")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save output files")
    
    # Device-specific arguments
    parser.add_argument('--gpu_id', type=int, default=0, 
                        help="GPU ID to use (only for GPU mode)")
    parser.add_argument('--xpu_id', type=int, default=0, 
                        help="XPU ID to use (only for XPU mode)")
    parser.add_argument('--num_threads', type=int, default=1,
                        help="Number of CPU threads to use (only for CPU mode)")
    
    # Input arguments
    parser.add_argument('--tile_path', type=str,
                        help="Path to the tile to process (only for CPU mode)")
    parser.add_argument('--tile_list', type=str,
                        help="JSON file containing list of tiles to process (for GPU/XPU mode)")
    
    # Model arguments
    parser.add_argument('--onnx_model_path', type=str, default=None,
                        help="Path to ONNX model file (optional, for ONNX inference)")
    parser.add_argument('--quantization_bits', type=int, default=8,
                        help="Number of bits for quantization (default: 8)")
    parser.add_argument('--enable_bf16', action='store_true',
                        help="Enable BF16 (Brain Floating Point 16-bit) inference")
    parser.add_argument('--enable_amx', action='store_true',
                        help="Enable AMX (Advanced Matrix Extensions) for Intel Xeon processors")
    
    # Data loading arguments
    parser.add_argument('--batch_size', type=int, default=None,
                        help="Batch size for inference (overrides config)")
    parser.add_argument('--num_workers', type=int, default=None,
                        help="Number of workers for data loading (overrides config)")
    
    # Logging arguments
    parser.add_argument('--process_id', type=int, default=0, 
                        help="Process ID for logging purposes")
    parser.add_argument('--log_level', type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        help="Logging level")
    parser.add_argument('--log_interval', type=int, default=5,
                        help="Log interval for batch progress (in batches)")
    parser.add_argument('--verbose_gpu', action='store_true',
                        help="Enable verbose GPU logging")
    parser.add_argument('--simplified_logging', action='store_true',
                        help="Use simplified logging to single file")
    
    # Profiling arguments
    parser.add_argument('--enable_profiling', action='store_true',
                        help="Enable profiling for performance analysis")
    parser.add_argument('--max_batches', type=int, default=None,
                        help="Maximum number of batches to process (default: process all batches)")
    parser.add_argument('--profile_output_dir', type=str, 
                        default="logs/profile",
                        help="Directory to save profiling results (default: logs/profile)")
    parser.add_argument('--profile_format', type=str, 
                        choices=['json', 'csv', 'html', 'all'], default='json',
                        help="Profiling report format (default: json for speed)")
    parser.add_argument('--profile_fast', action='store_true',
                        help="Fast profiling mode (JSON only, skip HTML generation)")
    parser.add_argument('--use_enhanced_profiling', action='store_true', default=True,
                        help="Use enhanced Tessera profiler (default: True)")
    
    return parser


def validate_args(args):
    """Validate command line arguments"""
    # Validate args based on mode
    if args.mode == 'cpu' and args.tile_path is None:
        raise ValueError("CPU mode requires --tile_path")
    if args.mode in ['gpu', 'xpu'] and args.tile_list is None:
        raise ValueError("GPU/XPU mode requires --tile_list")
    
    # Validate tile list file exists
    if args.tile_list and not os.path.exists(args.tile_list):
        raise ValueError(f"Tile list file does not exist: {args.tile_list}")
    
    # Validate config file exists
    if not os.path.exists(args.config):
        raise ValueError(f"Config file does not exist: {args.config}")
    
    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        raise ValueError(f"Checkpoint file does not exist: {args.checkpoint_path}")
    
    # Validate ONNX model if specified
    if args.onnx_model_path and not os.path.exists(args.onnx_model_path):
        raise ValueError(f"ONNX model file does not exist: {args.onnx_model_path}")
    
    return args


def parse_args():
    """Parse and validate command line arguments"""
    parser = create_parser()
    args = parser.parse_args()
    return validate_args(args)