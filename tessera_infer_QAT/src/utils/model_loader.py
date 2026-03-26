#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import torch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.ssl_model import MultimodalBTInferenceModel
from models.builder import build_inference_model
from utils.common import format_time


class ModelLoader:
    """Handles loading and setup of both PyTorch and ONNX models"""
    
    def __init__(self, config, device, args):
        self.config = config
        self.device = device
        self.args = args
        self.mode_prefix = device.type.upper()
    
    def load_pytorch_model(self):
        """Load PyTorch QAT model"""
        logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Building PyTorch model...")
        model_start_time = time.time()

        # Build model components
        s2_backbone, s1_backbone, dim_reducer = build_inference_model(self.config, self.device)

        try:
            logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Loading checkpoint: {self.config['checkpoint_path']}")
            checkpoint = torch.load(self.config["checkpoint_path"], map_location=self.device)
            
            state_key = 'model_state' if 'model_state' in checkpoint else 'model_state_dict'
            
            if state_key not in checkpoint:
                logging.error(f"[{self.mode_prefix}] [{self.args.process_id}] State key '{state_key}' not found in checkpoint")
                logging.error(f"[{self.mode_prefix}] [{self.args.process_id}] Available keys: {list(checkpoint.keys())}")
                raise ValueError(f"State key '{state_key}' not found in checkpoint")
            
            # Handle FSDP prefixes and load only needed components
            state_dict = checkpoint[state_key]
            
            # Extract and load component weights
            s2_state_dict = {}
            s1_state_dict = {}
            dim_reducer_state_dict = {}
            
            for key, value in state_dict.items():
                # Remove FSDP prefix if present
                clean_key = key.replace('_orig_mod.', '')
                
                # Categorize weights
                if clean_key.startswith('s2_backbone.'):
                    s2_state_dict[clean_key.replace('s2_backbone.', '')] = value
                elif clean_key.startswith('s1_backbone.'):
                    s1_state_dict[clean_key.replace('s1_backbone.', '')] = value
                elif clean_key.startswith('dim_reducer.'):
                    dim_reducer_state_dict[clean_key.replace('dim_reducer.', '')] = value
            
            # Load state dicts
            s2_backbone.load_state_dict(s2_state_dict, strict=True)
            s1_backbone.load_state_dict(s1_state_dict, strict=True)
            dim_reducer.load_state_dict(dim_reducer_state_dict, strict=True)
            
            model_load_time = time.time() - model_start_time
            logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Model components loaded in {format_time(model_load_time)}")
            
        except Exception as e:
            logging.error(f"[{self.mode_prefix}] [{self.args.process_id}] Error loading checkpoint: {str(e)}")
            raise

        # Set to eval mode and freeze parameters
        s2_backbone.eval()
        s1_backbone.eval()
        dim_reducer.eval()

        for param in s2_backbone.parameters():
            param.requires_grad = False
        for param in s1_backbone.parameters():
            param.requires_grad = False
        for param in dim_reducer.parameters():
            param.requires_grad = False

        # Build inference model
        logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Creating inference model")
        model = MultimodalBTInferenceModel(
            s2_backbone=s2_backbone,
            s1_backbone=s1_backbone,
            fusion_method=self.config["fusion_method"],
            dim_reducer=dim_reducer,
        )

        model.eval()
        
        use_amx = getattr(self.args, 'amx_enabled', getattr(self.args, 'enable_amx', False))

        # Apply AMX optimizations if enabled
        if use_amx:
            try:
                # Convert to channels_last memory format for better AMX performance
                # This is especially beneficial for convolutional operations
                model = model.to(memory_format=torch.channels_last)
                logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Model converted to channels_last format for AMX optimization")
            except Exception as e:
                logging.warning(f"[{self.mode_prefix}] [{self.args.process_id}] Failed to convert model to channels_last format: {e}")
        
        # Apply BF16 precision if requested
        if getattr(self.args, 'enable_bf16', False):
            try:
                model = model.to(dtype=torch.bfloat16)
                logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] PyTorch model converted to BF16")
                
                if use_amx:
                    logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] AMX + BF16 model optimizations applied")
                    
            except Exception as e:
                logging.warning(f"[{self.mode_prefix}] [{self.args.process_id}] Failed to convert PyTorch model to BF16: {e}, using FP32")
                model = model.to(dtype=torch.float32)
        
        return model
    
    def load_onnx_model(self):
        """Load ONNX model (if ONNX path is provided)"""
        if not self.args.onnx_model_path:
            return None
            
        try:
            # Import ONNX inference classes (from daintree version)
            from models.onnx_inference import ONNXInferenceModel, BatchedONNXInferenceModel
            
            logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Using ONNX model: {self.args.onnx_model_path}")
            model_start_time = time.time()
            
            # Create ONNX inference model
            if self.device.type in ['cuda', 'xpu']:
                # For GPU/XPU mode, use batched model
                onnx_model = BatchedONNXInferenceModel(
                    self.args.onnx_model_path, 
                    num_threads=getattr(self.args, 'num_threads', 8),
                    max_batch_size=self.config.get('batch_size', 256)
                )
            else:
                # For CPU mode, use regular ONNX model with thread optimization
                onnx_model = ONNXInferenceModel(
                    self.args.onnx_model_path, 
                    num_threads=getattr(self.args, 'num_threads', 8)
                )
            
            # Configure BF16 precision for ONNX model if requested
            if getattr(self.args, 'enable_bf16', False):
                try:
                    # ONNX Runtime doesn't directly support BF16, but we can configure it
                    onnx_model._enable_bf16 = True
                    logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] ONNX model configured for BF16 precision")
                except Exception as e:
                    logging.warning(f"[{self.mode_prefix}] [{self.args.process_id}] Failed to configure ONNX model for BF16: {e}")
                    onnx_model._enable_bf16 = False
            else:
                onnx_model._enable_bf16 = False
                
            model_load_time = time.time() - model_start_time
            logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] ONNX model loaded in {format_time(model_load_time)}")
            
            return onnx_model
            
        except Exception as e:
            logging.error(f"[{self.mode_prefix}] [{self.args.process_id}] Error loading ONNX model: {str(e)}")
            raise
    
    def load_model(self):
        """Load the appropriate model (ONNX or PyTorch)"""
        # Try ONNX first if specified
        if self.args.onnx_model_path:
            try:
                return self.load_onnx_model()
            except Exception as e:
                logging.warning(f"[{self.mode_prefix}] [{self.args.process_id}] ONNX model loading failed, falling back to PyTorch: {e}")
        
        # Load PyTorch model
        return self.load_pytorch_model()