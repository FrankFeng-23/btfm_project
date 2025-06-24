#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import logging
import numpy as np
import json
import traceback
import psutil
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import intel_extension_for_pytorch as ipex  # Import Intel Extension for PyTorch

# Custom logger to control verbosity
class CustomFormatter(logging.Formatter):
    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        process_info = f"[P{record.process}]" if hasattr(record, 'process') else ""
        return f"{timestamp} - {record.levelname} - {process_info} {record.getMessage()}"

# Setup early logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

def force_log_flush():
    """Force all handlers to flush their logs"""
    for handler in logging.root.handlers:
        handler.flush()

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

# Time profiling class to track performance
class TimeProfiler:
    def __init__(self, profile_window=10):
        self.times = defaultdict(list)
        self.current_times = {}
        self.profile_window = profile_window
        self.batch_count = 0
        
    def start(self, name):
        self.current_times[name] = time.time()
        
    def end(self, name):
        if name in self.current_times:
            elapsed = time.time() - self.current_times[name]
            self.times[name].append(elapsed)
            del self.current_times[name]
            return elapsed
        return 0
    
    def add_batch(self):
        self.batch_count += 1
        
    def should_log(self):
        return self.batch_count % self.profile_window == 0
    
    def get_stats(self):
        stats = {}
        total_time = 0
        for name, times in self.times.items():
            # Get last N times where N is profile_window
            recent_times = times[-self.profile_window:]
            if recent_times:
                avg_time = sum(recent_times) / len(recent_times)
                stats[name] = {
                    'avg': avg_time,
                    'min': min(recent_times),
                    'max': max(recent_times),
                    'total': sum(recent_times)
                }
                if name != 'batch_total':  # Exclude batch_total from subtotal calculation
                    total_time += avg_time
                    
        # Calculate percentages
        if 'batch_total' in stats:
            batch_total = stats['batch_total']['avg']
            for name, data in stats.items():
                if name != 'batch_total':
                    data['percentage'] = (data['avg'] / batch_total) * 100 if batch_total > 0 else 0
                    
        return stats
    
    def format_stats(self, prefix=""):
        stats = self.get_stats()
        if not stats:
            return "No timing data available"
            
        lines = [f"{prefix}Time profiling (avg of last {self.profile_window} batches):"]
        lines.append(f"{prefix}{'Step':<25} {'Time (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'%':<8}")
        lines.append(f"{prefix}{'-'*65}")
        
        # Sort by time (descending)
        sorted_stats = sorted(
            [(name, data) for name, data in stats.items() if name != 'batch_total'],
            key=lambda x: x[1]['avg'],
            reverse=True
        )
        
        # Add batch_total at the bottom
        if 'batch_total' in stats:
            sorted_stats.append(('batch_total', stats['batch_total']))
            
        for name, data in sorted_stats:
            percentage = data.get('percentage', 100.0 if name == 'batch_total' else 0.0)
            lines.append(
                f"{prefix}{name:<25} {data['avg']*1000:9.2f} ms {data['min']*1000:9.2f} ms {data['max']*1000:9.2f} ms {percentage:6.1f}%"
            )
            
        return "\n".join(lines)

def add_cpu_fallback_to_gru(model, device):
    """
    修改模型中的GRU操作，在CPU上使用手动实现的GRU，完全避开XPU的GRU实现
    """
    if device.type != 'xpu':
        return model
    
    logging.info(f"[XPU] [N/A] Adding CPU fallback to GRU operations for improved compatibility")
    
    def manual_gru_cell(x_t, h_prev, weight_ih, weight_hh, bias_ih, bias_hh):
        """手动实现GRU cell的前向传播"""
        # Start timing
        cell_start_time = time.time()
        
        # GRU gates
        gi = torch.mm(x_t, weight_ih.t()) + bias_ih
        gh = torch.mm(h_prev, weight_hh.t()) + bias_hh
        
        # Split into reset, update, and new gates
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)
        
        resetgate = torch.sigmoid(i_r + h_r)
        updategate = torch.sigmoid(i_z + h_z)
        newgate = torch.tanh(i_n + resetgate * h_n)
        
        hy = newgate + updategate * (h_prev - newgate)
        
        # End timing
        cell_time = time.time() - cell_start_time
        
        # Store timing data if existing
        if hasattr(manual_gru_forward, 'profiler'):
            manual_gru_forward.profiler.times['gru_cell'].append(cell_time)
            
        return hy
    
    def manual_gru_forward(input_seq, weight_ih, weight_hh, bias_ih, bias_hh, 
                          hidden_size, batch_first=True):
        """手动实现GRU的完整前向传播"""
        gru_start_time = time.time()
        
        if batch_first:
            # (batch, seq, feature) -> (seq, batch, feature)
            input_seq = input_seq.transpose(0, 1)
        
        seq_len, batch_size, _ = input_seq.size()
        
        # Initialize hidden state
        h_t = torch.zeros(batch_size, hidden_size, device=input_seq.device, dtype=input_seq.dtype)
        
        outputs = []
        for t in range(seq_len):
            h_t = manual_gru_cell(input_seq[t], h_t, weight_ih, weight_hh, bias_ih, bias_hh)
            outputs.append(h_t.unsqueeze(0))
        
        # Concatenate outputs
        output = torch.cat(outputs, dim=0)
        
        if batch_first:
            # (seq, batch, feature) -> (batch, seq, feature)
            output = output.transpose(0, 1)
        
        gru_time = time.time() - gru_start_time
        
        # Store in profiler if available
        if hasattr(manual_gru_forward, 'profiler'):
            manual_gru_forward.profiler.times['manual_gru_full'].append(gru_time)
            
        return output, h_t
    
    def patch_temporal_pooling(module):
        # Get GRU configuration
        hidden_size = module.temporal_context.hidden_size
        batch_first = module.temporal_context.batch_first
        
        # Save original GRU parameters to CPU
        with torch.no_grad():
            gru_state = module.temporal_context.state_dict()
            cpu_weights = {}
            for key, value in gru_state.items():
                cpu_weights[key] = value.detach().clone().cpu()
        
        # Save references
        module._original_gru = module.temporal_context
        module._cpu_weights = cpu_weights
        module._hidden_size = hidden_size
        module._batch_first = batch_first
        
        def new_forward(x):
            tp_start_time = time.time()
            original_device = x.device
            
            # Move input to CPU
            with torch.no_grad():
                x_cpu = x.detach().cpu().contiguous()
            
            # Manually compute GRU on CPU without using nn.GRU
            with torch.no_grad():
                # Get GRU weights
                weight_ih = module._cpu_weights['weight_ih_l0']
                weight_hh = module._cpu_weights['weight_hh_l0']
                bias_ih = module._cpu_weights['bias_ih_l0']
                bias_hh = module._cpu_weights['bias_hh_l0']
                
                # Run manual GRU forward
                output_cpu, _ = manual_gru_forward(
                    x_cpu, 
                    weight_ih, 
                    weight_hh, 
                    bias_ih, 
                    bias_hh,
                    module._hidden_size,
                    module._batch_first
                )
            
            # Move result back to original device
            x_context = output_cpu.to(original_device).contiguous()
            
            # Continue computation on original device
            w = torch.softmax(module.query(x_context), dim=1)
            result = (w * x).sum(dim=1)
            
            tp_time = time.time() - tp_start_time
            
            # Store timing data if available
            if hasattr(module, 'profiler'):
                module.profiler.times['temporal_pooling'].append(tp_time)
                
            return result
        
        module.forward = new_forward
    
    # Recursively find and patch all TemporalAwarePooling instances
    def find_and_patch_temporal_pooling(module):
        from models.modules import TemporalAwarePooling
        
        for name, child in module.named_children():
            if isinstance(child, TemporalAwarePooling):
                patch_temporal_pooling(child)
                # Add profiler to the module
                if hasattr(model, 'profiler'):
                    child.profiler = model.profiler
            else:
                find_and_patch_temporal_pooling(child)
    
    # Apply the patch to the entire model
    find_and_patch_temporal_pooling(model)
    
    # Add profiler to the manual_gru_forward function
    if hasattr(model, 'profiler'):
        manual_gru_forward.profiler = model.profiler
    
    return model

try:
    # Add project root to sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    logging.info(f"Added project root to path: {project_root}")

    # Import required modules
    from models.ssl_model import MultimodalBTInferenceModel
    from models.builder import build_inference_model
    from datasets.ssl_dataset import SingleTileInferenceDataset
    import importlib.util
except Exception as e:
    logging.error(f"Error during import: {str(e)}")
    logging.error(traceback.format_exc())
    sys.exit(1)

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
    try:
        if not os.path.exists(config_file_path):
            logging.error(f"Config file does not exist: {config_file_path}")
            sys.exit(1)
            
        spec = importlib.util.spec_from_file_location("my_dynamic_config", config_file_path)
        config_module = importlib.util.module_from_spec(spec)
        sys.modules["my_dynamic_config"] = config_module
        spec.loader.exec_module(config_module)
        return config_module
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-tile Inference (CPU/XPU)")
    parser.add_argument('--config', type=str, required=True, 
                        help="Path to config file")
    parser.add_argument('--mode', type=str, choices=['cpu', 'xpu'], required=True,
                        help="Processing mode: 'cpu' or 'xpu'")
    parser.add_argument('--xpu_id', type=int, default=0, 
                        help="XPU ID to use (only for XPU mode)")
    parser.add_argument('--tile_path', type=str,
                        help="Path to the tile to process (only for CPU mode)")
    parser.add_argument('--tile_list', type=str,
                        help="JSON file containing list of tiles to process (only for XPU mode)")
    parser.add_argument('--process_id', type=int, default=0, 
                        help="Process ID for logging purposes")
    parser.add_argument('--num_threads', type=int, default=1,
                        help="Number of CPU threads to use (only for CPU mode)")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save output files")
    parser.add_argument('--batch_size', type=int, default=None,
                        help="Batch size for inference (overrides config)")
    parser.add_argument('--num_workers', type=int, default=None,
                        help="Number of workers for data loading (overrides config)")
    parser.add_argument('--log_level', type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        help="Logging level")
    parser.add_argument('--log_interval', type=int, default=10,
                        help="Log interval for batch progress (in batches)")
    parser.add_argument('--verbose_xpu', action='store_true',
                        help="Enable verbose XPU logging")
    parser.add_argument('--simplified_logging', action='store_true',
                        help="Use simplified logging to single file")
    args = parser.parse_args()
    
    # Validate args based on mode
    if args.mode == 'cpu' and args.tile_path is None:
        parser.error("CPU mode requires --tile_path")
    if args.mode == 'xpu' and args.tile_list is None:
        parser.error("XPU mode requires --tile_list")
        
    return args

def sample_s2_batch(s2_bands_batch, s2_masks_batch, s2_doys_batch,
                band_mean, band_std, sample_size_s2, standardize=True, profiler=None):
    """
    Process S2 batch data with random sampling.
      s2_bands_batch.shape = (B, T_s2, 10)
      s2_masks_batch.shape = (B, T_s2)
      s2_doys_batch.shape  = (B, T_s2)
    Returns: np.array, shape=(B, sample_size_s2, 11), dtype float32
    """
    if profiler:
        profiler.start('sample_s2_batch')
        
    B = s2_bands_batch.shape[0]
    out_list = []
    for b in range(B):
        valid_idx = np.nonzero(s2_masks_batch[b])[0]
        
        if len(valid_idx) == 0:
            # If all timesteps are 0, use all indices
            valid_idx = np.arange(s2_bands_batch.shape[1])
        
        if len(valid_idx) < sample_size_s2:
            idx_chosen = np.random.choice(valid_idx, size=sample_size_s2, replace=True)
        else:
            idx_chosen = np.random.choice(valid_idx, size=sample_size_s2, replace=False)
        idx_chosen = np.sort(idx_chosen)

        sub_bands = s2_bands_batch[b, idx_chosen, :]  # (sample_size_s2, 10)
        sub_doys  = s2_doys_batch[b, idx_chosen]      # (sample_size_s2,)
        if standardize:
            sub_bands = (sub_bands - band_mean) / (band_std + 1e-9)

        # Directly append doy
        out_arr = np.hstack([sub_bands, sub_doys.reshape(-1, 1)])  # (sample_size_s2, 11)
        
        out_list.append(out_arr.astype(np.float32))

    result = np.stack(out_list, axis=0).astype(np.float32)  # (B, sample_size_s2, 11)
    
    if profiler:
        profiler.end('sample_s2_batch')
        
    return result

def sample_s1_batch(s1_asc_bands_batch, s1_asc_doys_batch,
                s1_desc_bands_batch, s1_desc_doys_batch,
                band_mean, band_std, sample_size_s1, standardize=True, profiler=None):
    """
    Process S1 batch data with random sampling.
      s1_asc_bands_batch.shape = (B, t_s1a, 2)
      s1_asc_doys_batch.shape  = (B, t_s1a)
      s1_desc_bands_batch.shape= (B, t_s1d, 2)
      s1_desc_doys_batch.shape = (B, t_s1d)
    Returns: np.array, shape=(B, sample_size_s1, 3), dtype float32
    """
    if profiler:
        profiler.start('sample_s1_batch')
        
    B = s1_asc_bands_batch.shape[0]
    out_list = []
    for b in range(B):
        s1_bands_all = np.concatenate([s1_asc_bands_batch[b], s1_desc_bands_batch[b]], axis=0)  # shape (t_s1a+t_s1d, 2)
        s1_doys_all  = np.concatenate([s1_asc_doys_batch[b], s1_desc_doys_batch[b]], axis=0)

        valid_mask = np.any(s1_bands_all != 0, axis=-1)
        valid_idx = np.nonzero(valid_mask)[0]
        if len(valid_idx) == 0:
            # If all timesteps are 0, use all indices
            valid_idx = np.arange(s1_bands_all.shape[0])
        if len(valid_idx) < sample_size_s1:
            idx_chosen = np.random.choice(valid_idx, size=sample_size_s1, replace=True)
        else:
            idx_chosen = np.random.choice(valid_idx, size=sample_size_s1, replace=False)
        idx_chosen = np.sort(idx_chosen)

        sub_bands = s1_bands_all[idx_chosen, :]  # (sample_size_s1, 2)
        sub_doys  = s1_doys_all[idx_chosen]

        if standardize:
            sub_bands = (sub_bands - band_mean) / (band_std + 1e-9)

        # Directly append doy
        out_arr = np.hstack([sub_bands, sub_doys.reshape(-1, 1)])  # (sample_size_s1, 3)
        
        out_list.append(out_arr.astype(np.float32))

    result = np.stack(out_list, axis=0).astype(np.float32)  # (B, sample_size_s1, 3)
    
    if profiler:
        profiler.end('sample_s1_batch')
        
    return result

def process_tile(tile_path, output_path, model, config, device, process_id, args):
    """Process a single tile with the loaded model"""
    tile_name = os.path.basename(tile_path)
    is_xpu = device.type == 'xpu'
    mode_prefix = "XPU" if is_xpu else "CPU"
    
    logging.info(f"[{mode_prefix}] [{process_id}] Processing tile: {tile_name}")
    logging.info(f"[{mode_prefix}] [{process_id}] PID {os.getpid()} - Starting tile {tile_name}")
    
    # Create time profiler
    profiler = TimeProfiler(profile_window=args.log_interval)
    # Attach profiler to the model for GRU operations
    model.profiler = profiler
    
    try:
        # Construct dataset
        dataset_start = time.time()
        logging.info(f"[{mode_prefix}] [{process_id}] Loading dataset from {tile_path}")
        
        # Construct dataset
        dataset = SingleTileInferenceDataset(
            tile_path=tile_path,
            min_valid_timesteps=config["min_valid_timesteps"],
            standardize=False  # Standardize during sampling
        )
        
        dataset_time = time.time() - dataset_start
        logging.info(f"[{mode_prefix}] [{process_id}] Dataset loaded in {format_time(dataset_time)}, shape={dataset.H}x{dataset.W}")
        
        # Create dataloader
        loader_start = time.time()
        logging.info(f"[{mode_prefix}] [{process_id}] Creating DataLoader with batch_size={config['batch_size']}, num_workers={config['num_workers']}")
        
        loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=is_xpu,  # Only pin memory for XPU
            drop_last=False
        )
        
        loader_time = time.time() - loader_start
        logging.info(f"[{mode_prefix}] [{process_id}] DataLoader created in {format_time(loader_time)}, {len(loader)} batches")
        
        # Inference loop
        local_results = []
        start_time = time.time()
        total_batches = len(loader)
        
        logging.info(f"[{mode_prefix}] [{process_id}] Starting inference on {tile_name}, {total_batches} batches")
        force_log_flush()  # Force flush the log
        
        with torch.no_grad():
            batch_start_time = time.time()
            last_log_time = time.time()
            last_batch_log = 0
            
            # Process each batch
            for batch_idx, batch_data in enumerate(loader):
                # Start timing the batch
                profiler.start('batch_total')
                
                curr_time = time.time()
                # Log after every log_interval batches
                should_log = (batch_idx % args.log_interval == 0 and batch_idx > 0) or \
                            (batch_idx == total_batches - 1) or \
                            (curr_time - last_log_time >= 30)  # Log at least every 30 seconds
                
                # Get batch data
                profiler.start('data_preparation')
                global_idxs = batch_data["global_idx"]  # shape=(B,)
                
                # Get numpy data from batch
                s2_bands_batch = batch_data["s2_bands"].numpy()  # (B, t_s2, 10)
                s2_masks_batch = batch_data["s2_masks"].numpy()  # (B, t_s2)
                s2_doys_batch  = batch_data["s2_doys"].numpy()   # (B, t_s2)
                
                s1_asc_bands_batch = batch_data["s1_asc_bands"].numpy()   # (B, t_s1a, 2)
                s1_asc_doys_batch  = batch_data["s1_asc_doys"].numpy()    # (B, t_s1a)
                s1_desc_bands_batch= batch_data["s1_desc_bands"].numpy()   # (B, t_s1d, 2)
                s1_desc_doys_batch = batch_data["s1_desc_doys"].numpy()    # (B, t_s1d)
                
                B = s2_bands_batch.shape[0]
                profiler.end('data_preparation')
                
                sum_repr = None
                
                for r in range(config["repeat_times"]):
                    # S2 sampling
                    s2_input_np = sample_s2_batch(
                        s2_bands_batch, s2_masks_batch, s2_doys_batch,
                        band_mean=dataset.s2_band_mean,
                        band_std=dataset.s2_band_std,
                        sample_size_s2=config["sample_size_s2"],
                        standardize=True,
                        profiler=profiler
                    )  # (B, sample_size_s2, 11) float32
                    
                    # S1 sampling
                    s1_input_np = sample_s1_batch(
                        s1_asc_bands_batch, s1_asc_doys_batch,
                        s1_desc_bands_batch, s1_desc_doys_batch,
                        band_mean=dataset.s1_band_mean,
                        band_std=dataset.s1_band_std,
                        sample_size_s1=config["sample_size_s1"],
                        standardize=True,
                        profiler=profiler
                    )  # (B, sample_size_s1, 3) float32
                    
                    # Convert to Tensor 
                    profiler.start('data_to_device')
                    s2_input = torch.tensor(s2_input_np, dtype=torch.float32, device=device)
                    s1_input = torch.tensor(s1_input_np, dtype=torch.float32, device=device)
                    profiler.end('data_to_device')
                    
                    # Forward pass
                    profiler.start('model_inference')
                    z = model(s2_input, s1_input)
                    
                    # For XPU, make sure the forward pass is completed
                    if is_xpu:
                        torch.xpu.synchronize(device)
                    profiler.end('model_inference')
                    
                    if sum_repr is None:
                        sum_repr = z
                    else:
                        sum_repr += z
                
                avg_repr = sum_repr / float(config["repeat_times"])
                
                # Save to local_results
                profiler.start('result_preparation')
                avg_repr_np = avg_repr.cpu().numpy()  # (B, latent_dim)
                global_idxs_list = global_idxs.tolist()
                
                for b in range(B):
                    gidx = global_idxs_list[b]
                    local_results.append((gidx, avg_repr_np[b]))
                profiler.end('result_preparation')
                
                # End batch timing
                profiler.end('batch_total')
                profiler.add_batch()
                
                # Log timing information
                if should_log:
                    progress = (batch_idx) / total_batches * 100
                    elapsed = curr_time - start_time
                    samples_processed = batch_idx * config["batch_size"]
                    inferences_per_sec = samples_processed / elapsed if elapsed > 0 else 0
                    
                    log_msg = f"[{mode_prefix}] [{process_id}] {tile_name}: " \
                            f"[{progress:.1f}%] Batch {batch_idx}/{total_batches} - " \
                            f"{samples_processed} samples - " \
                            f"{format_time(elapsed)} elapsed - " \
                            f"{inferences_per_sec:.1f} samples/sec"
                    
                    logging.info(log_msg)
                    
                    # Log profiling information
                    timing_log = profiler.format_stats(prefix="    ")
                    logging.info(timing_log)
                    
                    force_log_flush()  # Force flush the log
                    
                    last_log_time = curr_time
                    batch_start_time = curr_time
                    last_batch_log = batch_idx
        
        # Processing complete, save results
        save_start = time.time()
        logging.info(f"[{mode_prefix}] [{process_id}] Processing complete, saving results for {tile_name}")
        
        # Save results
        final_gidx_np = np.array([item[0] for item in local_results], dtype=np.int64)
        final_vecs_np = np.array([item[1] for item in local_results], dtype=np.float32)
        
        H, W = dataset.H, dataset.W
        latent_dim = final_vecs_np.shape[1]
        
        out_array = np.full((H * W, latent_dim), 0, dtype=np.float32)
        out_array[final_gidx_np] = final_vecs_np
        out_array = out_array.reshape(H, W, latent_dim)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, out_array)
        save_time = time.time() - save_start
        
        total_time = time.time() - start_time
        log_msg = f"[{mode_prefix}] [{process_id}] {tile_name}: " \
                f"Complete [{format_time(total_time)}] - " \
                f"Save time: {format_time(save_time)} - " \
                f"Shape={out_array.shape} - " \
                f"{len(local_results)/total_time:.1f} samples/sec"
        
        logging.info(log_msg)
        
        # Log final profiling summary
        logging.info("Final timing statistics:")
        logging.info(profiler.format_stats(prefix="    "))
        
        return output_path
    except Exception as e:
        log_msg = f"[{mode_prefix}] [{process_id}] Error processing {tile_name}: {str(e)}"
        logging.error(log_msg)
        logging.error(traceback.format_exc())
        raise

def main():
    # Parse arguments
    args = parse_args()
    
    # Set logging level
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    logging.root.setLevel(numeric_level)
    
    # Setup logging with file handler
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Simplified logging setup
    if args.simplified_logging:
        log_file = f"{log_dir}/infer_{args.mode}_{args.process_id}.log"
        
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
        log_file = f"{log_dir}/infer_process_{args.process_id}_{args.mode}.log"
        
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
    
    # Set process name for easier monitoring
    try:
        import setproctitle
        if args.mode == 'cpu':
            setproctitle.setproctitle(f"infer_cpu_{args.process_id}")
        else:
            setproctitle.setproctitle(f"infer_xpu_{args.process_id}_{args.xpu_id}")
    except ImportError:
        pass
    
    try:
        # Log initial system info
        mode_prefix = "XPU" if args.mode == 'xpu' else "CPU"
        logging.info(f"[{mode_prefix}] [{args.process_id}] {log_system_info()}")
        
        # Load configuration
        config_module = load_config_module(args.config)
        config = config_module.config
        
        # Override config with command-line args
        config["checkpoint_path"] = args.checkpoint_path
        config["output_dir"] = args.output_dir
        
        if args.batch_size is not None:
            config["batch_size"] = args.batch_size
            
        if args.num_workers is not None:
            config["num_workers"] = args.num_workers
        
        # Configure based on mode
        if args.mode == 'cpu':
            # Set the number of threads for CPU mode
            torch.set_num_threads(args.num_threads)
            device = torch.device("cpu")
            logging.info(f"[CPU] [{args.process_id}] Using CPU with {args.num_threads} threads")
            
            # Process specific tile
            tile_path = args.tile_path
            tile_name = os.path.basename(tile_path)
            
        else:  # XPU mode
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
            
            # Load tile list
            if not os.path.exists(args.tile_list):
                logging.error(f"[XPU] [{args.process_id}] Tile list file does not exist: {args.tile_list}")
                sys.exit(1)
                
            with open(args.tile_list, 'r') as f:
                try:
                    tile_list = json.loads(f.read())
                    logging.info(f"[XPU] [{args.process_id}] Loaded {len(tile_list)} tiles from JSON")
                    
                    # Print first few tiles for debugging
                    for i, tile in enumerate(tile_list[:min(3, len(tile_list))]):
                        logging.info(f"[XPU] [{args.process_id}] Tile {i}: {os.path.basename(tile)}")
                    
                    if len(tile_list) > 3:
                        logging.info(f"[XPU] [{args.process_id}] ... and {len(tile_list) - 3} more tiles")
                    
                except json.JSONDecodeError as e:
                    logging.error(f"[XPU] [{args.process_id}] Error parsing JSON: {str(e)}")
                    logging.error(f"[XPU] [{args.process_id}] File content: {open(args.tile_list, 'r').read()[:500]}...")
                    sys.exit(1)
        
        # Check if checkpoint file exists
        if not os.path.exists(config["checkpoint_path"]):
            logging.error(f"[{mode_prefix}] [{args.process_id}] Checkpoint file does not exist: {config['checkpoint_path']}")
            sys.exit(1)
        
        # Build and load SSL model
        logging.info(f"[{mode_prefix}] [{args.process_id}] Building model...")
        model_start_time = time.time()

        s2_backbone, s1_backbone, dim_reducer = build_inference_model(config, device)

        try:
            logging.info(f"[{mode_prefix}] [{args.process_id}] Loading checkpoint: {config['checkpoint_path']}")
            checkpoint = torch.load(config["checkpoint_path"], map_location=device)
            
            state_key = 'model_state' if 'model_state' in checkpoint else 'model_state_dict'
            
            if state_key not in checkpoint:
                logging.error(f"[{mode_prefix}] [{args.process_id}] State key '{state_key}' not found in checkpoint")
                logging.error(f"[{mode_prefix}] [{args.process_id}] Available keys: {list(checkpoint.keys())}")
                sys.exit(1)
            
            # Handle FSDP prefixes and load only needed components
            state_dict = checkpoint[state_key]
            
            # Extract and load s2_backbone weights
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
            logging.info(f"[{mode_prefix}] [{args.process_id}] Model components loaded in {format_time(model_load_time)}")
            
        except Exception as e:
            logging.error(f"[{mode_prefix}] [{args.process_id}] Error loading checkpoint: {str(e)}")
            logging.error(traceback.format_exc())
            sys.exit(1)

        # Set to eval mode
        s2_backbone.eval()
        s1_backbone.eval()
        dim_reducer.eval()

        # Freeze parameters
        for param in s2_backbone.parameters():
            param.requires_grad = False
        for param in s1_backbone.parameters():
            param.requires_grad = False
        for param in dim_reducer.parameters():
            param.requires_grad = False

        # Build inference model
        logging.info(f"[{mode_prefix}] [{args.process_id}] Creating inference model")
        model = MultimodalBTInferenceModel(
            s2_backbone=s2_backbone,
            s1_backbone=s1_backbone,
            fusion_method=config["fusion_method"],
            dim_reducer=dim_reducer,
        )

        # Create a time profiler and attach it to the model
        profiler = TimeProfiler(profile_window=args.log_interval)
        model.profiler = profiler

        # model = add_cpu_fallback_to_gru(model, device)

        model.eval()
        
        # Create output directory if it doesn't exist
        output_dir = config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        # Process tiles based on mode
        if args.mode == 'cpu':
            # Process single tile for CPU mode
            output_path = os.path.join(output_dir, f"{os.path.basename(args.tile_path)}.npy")
            
            # Skip if already processed
            if os.path.exists(output_path):
                logging.info(f"[CPU] [{args.process_id}] Skipping already processed tile: {os.path.basename(args.tile_path)}")
                return
                
            try:
                process_tile(args.tile_path, output_path, model, config, device, args.process_id, args)
            except Exception as e:
                logging.error(f"[CPU] [{args.process_id}] Fatal error processing tile {os.path.basename(args.tile_path)}")
                sys.exit(1)
        else:
            # Process multiple tiles for XPU mode
            total_start_time = time.time()
            
            total_tiles = len(tile_list)
            processed_count = 0
            skipped_count = 0
            failed_count = 0
            
            logging.info(f"[XPU] [{args.process_id}] Processing {total_tiles} tiles on XPU {args.xpu_id}")
            
            for i, tile_path in enumerate(tile_list):
                tile_name = os.path.basename(tile_path)
                output_path = os.path.join(output_dir, f"{tile_name}.npy")
                
                # Skip if already processed
                if os.path.exists(output_path):
                    logging.info(f"[XPU] [{args.process_id}] Skipping already processed tile: {tile_name} [{i+1}/{total_tiles}]")
                    skipped_count += 1
                    continue
                
                # Process current tile
                logging.info(f"[XPU] [{args.process_id}] Processing tile {tile_name} [{i+1}/{total_tiles}]")
                
                try:
                    process_tile(tile_path, output_path, model, config, device, args.process_id, args)
                    processed_count += 1
                    
                    # Extra log after successful processing
                    logging.info(f"[XPU] [{args.process_id}] Successfully processed {tile_name}")
                    
                except Exception as e:
                    logging.error(f"[XPU] [{args.process_id}] Failed to process tile {tile_name} [{i+1}/{total_tiles}]: {str(e)}")
                    logging.error(traceback.format_exc())
                    failed_count += 1
                    
                    # Try to recover XPU memory
                    if torch.xpu.is_available():
                        try:
                            torch.xpu.empty_cache()
                            logging.info(f"[XPU] [{args.process_id}] Cleared XPU cache")
                        except Exception as cache_error:
                            logging.error(f"[XPU] [{args.process_id}] Error clearing XPU cache: {str(cache_error)}")
                    
                    # Continue with next tile instead of exiting
                    continue
                
                # Log progress statistics
                elapsed = time.time() - total_start_time
                tiles_per_hour = (processed_count + skipped_count) / (elapsed / 3600) if elapsed > 0 else 0
                remaining = (total_tiles - (processed_count + skipped_count + failed_count)) / tiles_per_hour if tiles_per_hour > 0 else 0
                
                log_msg = f"[XPU] [{args.process_id}] Progress: {processed_count + skipped_count + failed_count}/{total_tiles} " \
                         f"(processed: {processed_count}, skipped: {skipped_count}, failed: {failed_count}) - " \
                         f"Time: {format_time(elapsed)} - " \
                         f"Rate: {tiles_per_hour:.1f} tiles/hour - " \
                         f"ETA: {format_time(remaining*3600)}"
                
                logging.info(log_msg)
                logging.info(f"[XPU] [{args.process_id}] {log_system_info()}")
                
            # Final statistics
            total_time = time.time() - total_start_time
            log_msg = f"[XPU] [{args.process_id}] Completed XPU processing in {format_time(total_time)}"
            logging.info(log_msg)
            log_msg = f"[XPU] [{args.process_id}] Results: {processed_count} processed, {skipped_count} skipped, {failed_count} failed"
            logging.info(log_msg)
            
        logging.info(f"[{mode_prefix}] [{args.process_id}] Processing complete")
        
    except Exception as e:
        logging.error(f"[{mode_prefix}] [{args.process_id}] Unhandled error: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()