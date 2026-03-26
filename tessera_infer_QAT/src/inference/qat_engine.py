#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import numpy as np
import traceback
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.common import force_log_flush, format_time
from utils.data_processing import sample_s2_batch, sample_s1_batch
from models.ssl_model import MultimodalBTInferenceModel
from models.builder import build_inference_model
from models.quantization import quantize_tensor_symmetric
from datasets.ssl_dataset import SingleTileInferenceDataset


class BaseInferenceEngine:
    """Base class for inference engines"""
    
    def __init__(self, model, config, device, process_id, args):
        self.model = model
        self.config = config
        self.device = device
        self.process_id = process_id
        self.args = args
        self.is_xpu = device.type == 'xpu'
        self.is_cuda = device.type == 'cuda'
        self.is_cpu = device.type == 'cpu'
        
        # Mode prefix for logging
        if self.is_xpu:
            self.mode_prefix = "XPU"
        elif self.is_cuda:
            self.mode_prefix = "GPU"
        else:
            self.mode_prefix = "CPU"
    
    def get_device_info(self):
        """Get device memory information"""
        try:
            if self.is_cuda:
                return f"GPU {self.device.index} Memory info"
            elif self.is_xpu:
                return f"XPU {self.device.index} info"
            else:
                return "CPU mode"
        except:
            return "Device info unavailable"
    
    def sync_device(self):
        """Synchronize device operations"""
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        elif self.is_xpu:
            torch.xpu.synchronize(self.device)
    
    def clear_cache(self):
        """Clear device cache if needed"""
        try:
            if self.is_cuda:
                torch.cuda.empty_cache()
            elif self.is_xpu:
                torch.xpu.empty_cache()
            logging.info(f"[{self.mode_prefix}] [{self.process_id}] Cleared device cache")
        except Exception as e:
            logging.warning(f"[{self.mode_prefix}] [{self.process_id}] Failed to clear device cache: {e}")


class QATInferenceEngine(BaseInferenceEngine):
    """QAT-specific inference engine with quantized output"""
    
    def __init__(self, model, config, device, process_id, args):
        super().__init__(model, config, device, process_id, args)
        self.quantization_bits = getattr(args, 'quantization_bits', 8)
        
        # Setup profiler if available
        self.profiler = None
        if hasattr(args, 'enable_profiling') and args.enable_profiling:
            self.setup_profiler()
    
    def setup_profiler(self):
        """Setup profiler if enabled"""
        try:
            # Check if enhanced profiling is available
            from profiler.precision_timer import GlobalTimerManager
            from profiler.cpu_optimizer import CPUPlatformDetector, PerformanceMonitor
            from profiler.pytorch_module_profiler import DetailedInferenceProfiler
            
            self.global_timer = GlobalTimerManager()
            
            # Apply CPU optimizations if requested
            if hasattr(self.args, 'use_enhanced_profiling') and self.args.use_enhanced_profiling:
                cpu_detector = CPUPlatformDetector()
                applied_opts = cpu_detector.apply_optimizations('pytorch')
                logging.info(f"[{self.mode_prefix}] [{self.process_id}] Applied CPU optimizations: {list(applied_opts.keys())}")
            
            # Initialize detailed profiler for PyTorch models
            if hasattr(self.model, 's2_backbone'):  # PyTorch model
                self.detailed_profiler = DetailedInferenceProfiler(self.model, self.global_timer)
                logging.info(f"[{self.mode_prefix}] [{self.process_id}] Enhanced PyTorch model profiling enabled")
            else:
                self.detailed_profiler = None
                logging.info(f"[{self.mode_prefix}] [{self.process_id}] Enhanced profiling enabled (using global timer only)")
            
            logging.info(f"[{self.mode_prefix}] [{self.process_id}] Enhanced detailed profiling enabled")
        except Exception as e:
            logging.warning(f"[{self.mode_prefix}] [{self.process_id}] Failed to initialize enhanced profiler: {e}")
            self.global_timer = None
            self.detailed_profiler = None
    
    def process_batch(self, batch_data, dataset, batch_idx):
        """Process a single batch with quantization"""
        B = batch_data["s2_bands"].shape[0]
        
        # Use detailed profiler if available
        if hasattr(self, 'detailed_profiler') and self.detailed_profiler:
            batch_results = self.detailed_profiler.profile_batch_processing(
                batch_data, dataset, self.config, self.device, batch_idx
            )
            
            # Extract results and convert to int8 + scales format
            local_results = []
            local_scales = []
            
            for gidx, repr_f32 in batch_results:
                repr_tensor = torch.from_numpy(repr_f32)
                repr_int8, scale = quantize_tensor_symmetric(repr_tensor, bits=self.quantization_bits)
                local_results.append((gidx, repr_int8.numpy()))
                local_scales.append((gidx, scale.item()))
            
            return local_results, local_scales
        
        # Fallback to standard processing
        return self._process_batch_standard(batch_data, dataset, batch_idx)
    
    def _process_batch_standard(self, batch_data, dataset, batch_idx):
        """Standard batch processing without enhanced profiling"""
        # Get batch data
        global_idxs = batch_data["global_idx"]
        
        # Get numpy data from batch
        s2_bands_batch = batch_data["s2_bands"].numpy()
        s2_masks_batch = batch_data["s2_masks"].numpy()
        s2_doys_batch = batch_data["s2_doys"].numpy()
        
        s1_asc_bands_batch = batch_data["s1_asc_bands"].numpy()
        s1_asc_doys_batch = batch_data["s1_asc_doys"].numpy()
        s1_desc_bands_batch = batch_data["s1_desc_bands"].numpy()
        s1_desc_doys_batch = batch_data["s1_desc_doys"].numpy()
        
        B = s2_bands_batch.shape[0]
        
        sum_repr = None
        
        # Setup AMX autocast context if enabled
        use_amx = getattr(self.args, 'amx_enabled', getattr(self.args, 'enable_amx', False)) and self.is_cpu
        use_bf16 = getattr(self.args, 'enable_bf16', False)
        
        # Create appropriate inference context
        if use_amx and use_bf16:
            # AMX with BF16 autocast - optimal for Intel Xeon 4th gen
            inference_context = torch.amp.autocast('cpu', dtype=torch.bfloat16, enabled=True)
        elif use_amx:
            # AMX with FP32 operations
            inference_context = torch.amp.autocast('cpu', enabled=False)  # No dtype conversion, but AMX env is set
        else:
            # Standard inference
            inference_context = torch.no_grad()  # Already under torch.no_grad() but keeping explicit
        
        for r in range(self.config["repeat_times"]):
            # S2 sampling
            s2_input_np = sample_s2_batch(
                s2_bands_batch, s2_masks_batch, s2_doys_batch,
                band_mean=dataset.s2_band_mean,
                band_std=dataset.s2_band_std,
                sample_size_s2=self.config["sample_size_s2"],
                standardize=True,
                profiler=self.profiler
            )
            
            # S1 sampling
            s1_input_np = sample_s1_batch(
                s1_asc_bands_batch, s1_asc_doys_batch,
                s1_desc_bands_batch, s1_desc_doys_batch,
                band_mean=dataset.s1_band_mean,
                band_std=dataset.s1_band_std,
                sample_size_s1=self.config["sample_size_s1"],
                standardize=True,
                profiler=self.profiler
            )
            
            # Convert to tensor and handle data types
            if use_amx and use_bf16:
                # For AMX + BF16, use FP32 input and let autocast handle conversion
                dtype = torch.float32
            elif use_bf16:
                # Direct BF16 conversion
                dtype = torch.bfloat16
            else:
                # Standard FP32
                dtype = torch.float32
                
            s2_input = torch.tensor(s2_input_np, dtype=dtype, device=self.device)
            s1_input = torch.tensor(s1_input_np, dtype=dtype, device=self.device)
            
            # Apply AMX-optimized memory format if enabled
            if use_amx:
                try:
                    # Convert inputs to channels_last if applicable (for conv operations)
                    if len(s2_input.shape) >= 4:  # Only for 4D+ tensors
                        s2_input = s2_input.to(memory_format=torch.channels_last)
                    if len(s1_input.shape) >= 4:
                        s1_input = s1_input.to(memory_format=torch.channels_last)
                except Exception as e:
                    # If conversion fails, continue with default format
                    pass
            
            # Forward pass with AMX-optimized context
            with inference_context:
                z = self.model(s2_input, s1_input)
            
            # Synchronize device
            self.sync_device()
            
            if sum_repr is None:
                sum_repr = z
            else:
                sum_repr += z
        
        avg_repr = sum_repr / float(self.config["repeat_times"])
        
        # Quantize the representations
        if avg_repr.dtype == torch.bfloat16:
            avg_repr_np = avg_repr.to(torch.float32).cpu().numpy()
        else:
            avg_repr_np = avg_repr.cpu().numpy()
        
        # Quantize each representation in the batch
        quantized_repr_list = []
        scales_list = []
        
        for b in range(B):
            repr_f32 = torch.from_numpy(avg_repr_np[b])
            repr_int8, scale = quantize_tensor_symmetric(repr_f32, bits=self.quantization_bits)
            quantized_repr_list.append(repr_int8.numpy())
            scales_list.append(scale.item())
        
        quantized_repr_np = np.stack(quantized_repr_list, axis=0)
        scales_np = np.array(scales_list, dtype=np.float32)
        
        # Save to local_results
        local_results = []
        local_scales = []
        global_idxs_list = global_idxs.tolist()
        
        for b in range(B):
            gidx = global_idxs_list[b]
            local_results.append((gidx, quantized_repr_np[b]))
            local_scales.append((gidx, scales_np[b]))
        
        return local_results, local_scales