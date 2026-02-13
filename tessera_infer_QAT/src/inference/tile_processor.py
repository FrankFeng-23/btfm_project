#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import logging
import numpy as np
import contextlib
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.common import force_log_flush, format_time
from utils.device_utils import get_device_memory_info
from inference.qat_engine import QATInferenceEngine
from datasets.ssl_dataset import SingleTileInferenceDataset


class TileProcessor:
    """Processes individual tiles with QAT inference"""
    
    def __init__(self, model, config, device, args):
        self.model = model
        self.config = config
        self.device = device
        self.args = args
        self.inference_engine = QATInferenceEngine(model, config, device, args.process_id, args)
        
        self.is_gpu = device.type == 'cuda'
        self.is_xpu = device.type == 'xpu'
        self.mode_prefix = "GPU" if self.is_gpu else ("XPU" if self.is_xpu else "CPU")
    
    def process_tile(self, tile_path, output_path):
        """Process a single tile with the loaded model"""
        tile_name = os.path.basename(tile_path)
        
        logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Processing tile: {tile_name}")
        logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] PID {os.getpid()} - Starting tile {tile_name}")
        
        try:
            # Load dataset
            with self._timer_context("dataset_loading"):
                dataset_start = time.time()
                logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Loading dataset from {tile_path}")
                
                dataset = SingleTileInferenceDataset(
                    tile_path=tile_path,
                    min_valid_timesteps=self.config["min_valid_timesteps"],
                    standardize=False  # Standardize during sampling
                )
                
                dataset_time = time.time() - dataset_start
                logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Dataset loaded in {format_time(dataset_time)}, shape={dataset.H}x{dataset.W}")
            
            # Create dataloader
            with self._timer_context("dataloader_creation"):
                loader_start = time.time()
                logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Creating DataLoader with batch_size={self.config['batch_size']}, num_workers={self.config['num_workers']}")
                
                loader = DataLoader(
                    dataset,
                    batch_size=self.config["batch_size"],
                    shuffle=False,
                    num_workers=self.config["num_workers"],
                    pin_memory=self.is_gpu or self.is_xpu,
                    drop_last=False
                )
                
                loader_time = time.time() - loader_start
                logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] DataLoader created in {format_time(loader_time)}, {len(loader)} batches")
            
            # Process batches
            local_results, local_scales = self._process_batches(loader, dataset, tile_name)
            
            # Save results
            with self._timer_context("results_saving"):
                self._save_results(local_results, local_scales, dataset, output_path, tile_name)
            
            return output_path
            
        except Exception as e:
            logging.error(f"[{self.mode_prefix}] [{self.args.process_id}] Error processing {tile_name}: {str(e)}")
            raise
    
    def _timer_context(self, name):
        """Create timer context if profiling is enabled"""
        if hasattr(self.inference_engine, 'global_timer') and self.inference_engine.global_timer:
            return self.inference_engine.global_timer.get_timer("main").timer(name)
        else:
            return contextlib.nullcontext()
    
    def _process_batches(self, loader, dataset, tile_name):
        """Process all batches in the dataloader"""
        local_results = []
        local_scales = []
        start_time = time.time()
        total_batches = len(loader)
        
        # Apply batch limit if specified
        max_batches = getattr(self.args, 'max_batches', None)
        if max_batches is not None:
            actual_batches = min(total_batches, max_batches)
            logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Limiting processing to {actual_batches} batches (max_batches={max_batches})")
        else:
            actual_batches = total_batches
        
        logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Starting inference on {tile_name}, {actual_batches} batches")
        force_log_flush()
        
        with torch.no_grad():
            with self._timer_context("inference_loop"):
                batch_start_time = time.time()
                last_log_time = time.time()
                last_batch_log = 0
                
                for batch_idx, batch_data in enumerate(loader):
                    # Stop if we've reached the batch limit
                    if max_batches is not None and batch_idx >= max_batches:
                        logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Reached max_batches limit ({max_batches}), stopping")
                        break
                    
                    curr_time = time.time()
                    
                    # Determine if we should log
                    should_log = (self.is_gpu and hasattr(self.args, 'verbose_gpu') and self.args.verbose_gpu) or \
                                (batch_idx % self.args.log_interval == 0) or \
                                (batch_idx == actual_batches - 1) or \
                                (curr_time - last_log_time >= 30)  # Log at least every 30 seconds
                    
                    # Process batch
                    batch_results, batch_scales = self.inference_engine.process_batch(batch_data, dataset, batch_idx)
                    local_results.extend(batch_results)
                    local_scales.extend(batch_scales)
                    
                    # Log progress
                    if should_log and batch_idx > last_batch_log:
                        self._log_progress(batch_idx, actual_batches, start_time, tile_name, curr_time)
                        last_log_time = curr_time
                        batch_start_time = curr_time
                        last_batch_log = batch_idx
        
        return local_results, local_scales
    
    def _log_progress(self, batch_idx, actual_batches, start_time, tile_name, curr_time):
        """Log processing progress"""
        progress = (batch_idx) / actual_batches * 100
        elapsed = curr_time - start_time
        samples_processed = batch_idx * self.config["batch_size"]
        inferences_per_sec = samples_processed / elapsed if elapsed > 0 else 0
        
        # In verbose GPU mode, log memory stats
        extra_info = ""
        if self.is_gpu:
            extra_info = f" - {get_device_memory_info(self.device)}"
        
        log_msg = f"[{self.mode_prefix}] [{self.args.process_id}] {tile_name}: " \
                f"[{progress:.1f}%] Batch {batch_idx}/{actual_batches} - " \
                f"{samples_processed} samples - " \
                f"{format_time(elapsed)} elapsed - " \
                f"{inferences_per_sec:.1f} samples/sec" + extra_info
        
        logging.info(log_msg)
        force_log_flush()
    
    def _save_results(self, local_results, local_scales, dataset, output_path, tile_name):
        """Save quantized results and scales to files"""
        save_start = time.time()
        logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Processing complete, saving results for {tile_name}")
        
        # Save quantized representations
        final_gidx_np = np.array([item[0] for item in local_results], dtype=np.int64)
        final_vecs_np = np.array([item[1] for item in local_results], dtype=np.int8)  # int8!
        
        # Save scales
        final_scales_gidx_np = np.array([item[0] for item in local_scales], dtype=np.int64)
        final_scales_np = np.array([item[1] for item in local_scales], dtype=np.float32)
        
        H, W = dataset.H, dataset.W
        latent_dim = final_vecs_np.shape[1]
        
        # Create int8 representation array
        out_array = np.zeros((H * W, latent_dim), dtype=np.int8)
        out_array[final_gidx_np] = final_vecs_np
        out_array = out_array.reshape(H, W, latent_dim)
        
        # Create scale array
        scale_array = np.zeros((H * W,), dtype=np.float32)
        scale_array[final_scales_gidx_np] = final_scales_np
        scale_array = scale_array.reshape(H, W)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save int8 representations
        np.save(output_path, out_array)
        
        # Save scales with suffix
        scale_output_path = output_path.replace('.npy', '_scales.npy')
        np.save(scale_output_path, scale_array)
        
        save_time = time.time() - save_start
        
        # Log completion
        total_time = time.time() - save_start  # This should be from processing start, but simplified for now
        log_msg = f"[{self.mode_prefix}] [{self.args.process_id}] {tile_name}: " \
                f"Complete [saved in {format_time(save_time)}] - " \
                f"Shape={out_array.shape} (int8) - " \
                f"Scale shape={scale_array.shape} - " \
                f"{len(local_results)} samples"
        
        if self.is_gpu:
            log_msg += f" - {get_device_memory_info(self.device)}"
        
        logging.info(log_msg)
        
        # Generate enhanced profiling reports if enabled
        if hasattr(self.inference_engine, 'global_timer') and self.inference_engine.global_timer and hasattr(self.args, 'enable_profiling') and self.args.enable_profiling:
            self._generate_profiling_reports(tile_name)
    
    def _generate_profiling_reports(self, tile_name):
        """Generate enhanced profiling reports"""
        try:
            # Import profiling modules
            from profiler.report_generator import generate_enhanced_profiling_report
            from profiler.html_generator import HTMLReportGenerator
            
            # Generate profiling report
            report_data = generate_enhanced_profiling_report(
                self.inference_engine.global_timer,
                process_id=self.args.process_id,
                tile_name=tile_name,
                device_info=self.get_device_info()
            )
            
            # Save reports based on format
            if hasattr(self.args, 'profile_format'):
                format_type = self.args.profile_format
            else:
                format_type = 'json'
            
            output_dir = getattr(self.args, 'profile_output_dir', 'logs/profile')
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate JSON report (always)
            json_path = os.path.join(output_dir, f"{tile_name}_profile_{self.args.process_id}.json")
            with open(json_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] JSON profile saved: {json_path}")
            
            # Generate HTML report if requested and not in fast mode
            if format_type in ['html', 'all'] and not getattr(self.args, 'profile_fast', False):
                html_generator = HTMLReportGenerator()
                html_path = os.path.join(output_dir, f"{tile_name}_profile_{self.args.process_id}.html")
                html_generator.generate_report(report_data, html_path)
                logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] HTML profile saved: {html_path}")
                
        except Exception as e:
            logging.warning(f"[{self.mode_prefix}] [{self.args.process_id}] Failed to generate profiling reports: {e}")
    
    def get_device_info(self):
        """Get device information for profiling"""
        try:
            from utils.device_utils import get_device_memory_info
            return get_device_memory_info(self.device)
        except:
            return f"{self.mode_prefix} device info"