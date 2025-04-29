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

import torch
from torch.utils.data import DataLoader

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

def get_gpu_memory_info(device_id=None):
    """Get GPU memory information for the specified device"""
    try:
        if not torch.cuda.is_available():
            return "CUDA not available"
            
        if device_id is None:
            device_id = torch.cuda.current_device()
            
        # Get memory information for this device
        mem_allocated = torch.cuda.memory_allocated(device_id) / (1024**2)  # MB
        mem_reserved = torch.cuda.memory_reserved(device_id) / (1024**2)    # MB
        mem_max_allocated = torch.cuda.max_memory_allocated(device_id) / (1024**2)  # MB
        
        # Get total memory (requires pynvml or subprocess call to nvidia-smi)
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

try:
    # Add project root to sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    logging.info(f"Added project root to path: {project_root}")

    # Import required modules
    from models.ssl_model import MultimodalBTInferenceModel
    from models.builder import build_ssl_model
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
    parser = argparse.ArgumentParser(description="Multi-tile Inference (CPU/GPU)")
    parser.add_argument('--config', type=str, required=True, 
                        help="Path to config file")
    parser.add_argument('--mode', type=str, choices=['cpu', 'gpu'], required=True,
                        help="Processing mode: 'cpu' or 'gpu'")
    parser.add_argument('--gpu_id', type=int, default=0, 
                        help="GPU ID to use (only for GPU mode)")
    parser.add_argument('--tile_path', type=str,
                        help="Path to the tile to process (only for CPU mode)")
    parser.add_argument('--tile_list', type=str,
                        help="JSON file containing list of tiles to process (only for GPU mode)")
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
    parser.add_argument('--log_interval', type=int, default=5,
                        help="Log interval for batch progress (in batches)")
    parser.add_argument('--verbose_gpu', action='store_true',
                        help="Enable verbose GPU logging")
    parser.add_argument('--simplified_logging', action='store_true',
                        help="Use simplified logging to single file")
    args = parser.parse_args()
    
    # Validate args based on mode
    if args.mode == 'cpu' and args.tile_path is None:
        parser.error("CPU mode requires --tile_path")
    if args.mode == 'gpu' and args.tile_list is None:
        parser.error("GPU mode requires --tile_list")
        
    return args

def sample_s2_batch(s2_bands_batch, s2_masks_batch, s2_doys_batch,
                band_mean, band_std, sample_size_s2, standardize=True):
    """
    Process S2 batch data with random sampling.
      s2_bands_batch.shape = (B, T_s2, 10)
      s2_masks_batch.shape = (B, T_s2)
      s2_doys_batch.shape  = (B, T_s2)
    Returns: np.array, shape=(B, sample_size_s2, 11), dtype float32
    """
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

    return np.stack(out_list, axis=0).astype(np.float32)  # (B, sample_size_s2, 11)

def sample_s1_batch(s1_asc_bands_batch, s1_asc_doys_batch,
                s1_desc_bands_batch, s1_desc_doys_batch,
                band_mean, band_std, sample_size_s1, standardize=True):
    """
    Process S1 batch data with random sampling.
      s1_asc_bands_batch.shape = (B, t_s1a, 2)
      s1_asc_doys_batch.shape  = (B, t_s1a)
      s1_desc_bands_batch.shape= (B, t_s1d, 2)
      s1_desc_doys_batch.shape = (B, t_s1d)
    Returns: np.array, shape=(B, sample_size_s1, 3), dtype float32
    """
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

    return np.stack(out_list, axis=0).astype(np.float32)  # (B, sample_size_s1, 3)

def process_tile(tile_path, output_path, model, config, device, process_id, args):
    """Process a single tile with the loaded model"""
    tile_name = os.path.basename(tile_path)
    is_gpu = device.type == 'cuda'
    mode_prefix = "GPU" if is_gpu else "CPU"
    gpu_info = get_gpu_memory_info(device.index) if is_gpu else ""
    
    logging.info(f"[{mode_prefix}] [{process_id}] Processing tile: {tile_name} {gpu_info}")
    logging.info(f"[{mode_prefix}] [{process_id}] PID {os.getpid()} - Starting tile {tile_name}")
    
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
            pin_memory=is_gpu,  # Only pin memory for GPU
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
                curr_time = time.time()
                # Log more frequently for GPU (every batch in verbose mode)
                should_log = (is_gpu and args.verbose_gpu) or \
                            (batch_idx % args.log_interval == 0) or \
                            (batch_idx == total_batches - 1) or \
                            (curr_time - last_log_time >= 30)  # Log at least every 30 seconds
                
                if should_log and batch_idx > last_batch_log:
                    progress = (batch_idx) / total_batches * 100
                    batch_time = curr_time - batch_start_time
                    elapsed = curr_time - start_time
                    samples_processed = batch_idx * config["batch_size"]
                    inferences_per_sec = samples_processed / elapsed if elapsed > 0 else 0
                    
                    # In verbose GPU mode, log memory stats
                    extra_info = ""
                    if is_gpu:
                        extra_info = f" - {get_gpu_memory_info(device.index)}"
                    
                    log_msg = f"[{mode_prefix}] [{process_id}] {tile_name}: " \
                            f"[{progress:.1f}%] Batch {batch_idx}/{total_batches} - " \
                            f"{samples_processed} samples - " \
                            f"{format_time(elapsed)} elapsed - " \
                            f"{inferences_per_sec:.1f} samples/sec" + extra_info
                    
                    logging.info(log_msg)
                    force_log_flush()  # Force flush the log
                    
                    last_log_time = curr_time
                    batch_start_time = curr_time
                    last_batch_log = batch_idx
                    
                # Get batch data
                batch_data_start = time.time()
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
                
                batch_data_time = time.time() - batch_data_start
                
                sum_repr = None
                
                for r in range(config["repeat_times"]):
                    # S2
                    s2_input_np = sample_s2_batch(
                        s2_bands_batch, s2_masks_batch, s2_doys_batch,
                        band_mean=dataset.s2_band_mean,
                        band_std=dataset.s2_band_std,
                        sample_size_s2=config["sample_size_s2"],
                        standardize=True
                    )  # (B, sample_size_s2, 11) float32
                    
                    # S1
                    s1_input_np = sample_s1_batch(
                        s1_asc_bands_batch, s1_asc_doys_batch,
                        s1_desc_bands_batch, s1_desc_doys_batch,
                        band_mean=dataset.s1_band_mean,
                        band_std=dataset.s1_band_std,
                        sample_size_s1=config["sample_size_s1"],
                        standardize=True
                    )  # (B, sample_size_s1, 3) float32
                    
                    # Convert to Tensor 
                    s2_input = torch.tensor(s2_input_np, dtype=torch.float32, device=device)
                    s1_input = torch.tensor(s1_input_np, dtype=torch.float32, device=device)
                    
                    # Forward pass
                    z = model(s2_input, s1_input)
                    
                    # For GPU, make sure the forward pass is completed
                    if is_gpu:
                        torch.cuda.synchronize(device)
                    
                    if sum_repr is None:
                        sum_repr = z
                    else:
                        sum_repr += z
                
                avg_repr = sum_repr / float(config["repeat_times"])
                
                # Save to local_results
                avg_repr_np = avg_repr.cpu().numpy()  # (B, latent_dim)
                global_idxs_list = global_idxs.tolist()
                
                for b in range(B):
                    gidx = global_idxs_list[b]
                    local_results.append((gidx, avg_repr_np[b]))
        
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
        
        if is_gpu:
            log_msg += f" - {get_gpu_memory_info(device.index)}"
        
        logging.info(log_msg)
        
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
            setproctitle.setproctitle(f"infer_gpu_{args.process_id}_{args.gpu_id}")
    except ImportError:
        pass
    
    try:
        # Log initial system info
        mode_prefix = "GPU" if args.mode == 'gpu' else "CPU"
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
            
        else:  # GPU mode
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
            logging.info(f"[GPU] [{args.process_id}] {get_gpu_memory_info(device_id)}")
            logging.info(f"[GPU] [{args.process_id}] CUDA version: {torch.version.cuda}")
            logging.info(f"[GPU] [{args.process_id}] PyTorch version: {torch.__version__}")
            
            # Load tile list
            if not os.path.exists(args.tile_list):
                logging.error(f"[GPU] [{args.process_id}] Tile list file does not exist: {args.tile_list}")
                sys.exit(1)
                
            with open(args.tile_list, 'r') as f:
                try:
                    tile_list = json.loads(f.read())
                    logging.info(f"[GPU] [{args.process_id}] Loaded {len(tile_list)} tiles from JSON")
                    
                    # Print first few tiles for debugging
                    for i, tile in enumerate(tile_list[:min(3, len(tile_list))]):
                        logging.info(f"[GPU] [{args.process_id}] Tile {i}: {os.path.basename(tile)}")
                    
                    if len(tile_list) > 3:
                        logging.info(f"[GPU] [{args.process_id}] ... and {len(tile_list) - 3} more tiles")
                    
                except json.JSONDecodeError as e:
                    logging.error(f"[GPU] [{args.process_id}] Error parsing JSON: {str(e)}")
                    logging.error(f"[GPU] [{args.process_id}] File content: {open(args.tile_list, 'r').read()[:500]}...")
                    sys.exit(1)
        
        # Check if checkpoint file exists
        if not os.path.exists(config["checkpoint_path"]):
            logging.error(f"[{mode_prefix}] [{args.process_id}] Checkpoint file does not exist: {config['checkpoint_path']}")
            sys.exit(1)
        
        # Build and load SSL model
        logging.info(f"[{mode_prefix}] [{args.process_id}] Building model...")
        model_start_time = time.time()
        ssl_model = build_ssl_model(config, device)
        
        try:
            logging.info(f"[{mode_prefix}] [{args.process_id}] Loading checkpoint: {config['checkpoint_path']}")
            checkpoint = torch.load(config["checkpoint_path"], map_location=device)
            
            state_key = 'model_state' if 'model_state' in checkpoint else 'model_state_dict'
            
            if state_key not in checkpoint:
                logging.error(f"[{mode_prefix}] [{args.process_id}] State key '{state_key}' not found in checkpoint")
                logging.error(f"[{mode_prefix}] [{args.process_id}] Available keys: {list(checkpoint.keys())}")
                sys.exit(1)
                
            # Handle FSDP prefixes
            state_dict = checkpoint[state_key]
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[len('_orig_mod.'):]  # Remove prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            # Load state dict
            ssl_model.load_state_dict(new_state_dict, strict=True)
            model_load_time = time.time() - model_start_time
            logging.info(f"[{mode_prefix}] [{args.process_id}] Model loaded in {format_time(model_load_time)}")
            
            if args.mode == 'gpu':
                logging.info(f"[GPU] [{args.process_id}] {get_gpu_memory_info(device_id)}")
        except Exception as e:
            logging.error(f"[{mode_prefix}] [{args.process_id}] Error loading checkpoint: {str(e)}")
            logging.error(traceback.format_exc())
            sys.exit(1)
        
        # Freeze SSL backbone parameters
        for param in ssl_model.s2_backbone.parameters():
            param.requires_grad = False
        for param in ssl_model.s1_backbone.parameters():
            param.requires_grad = False
        for param in ssl_model.dim_reducer.parameters():
            param.requires_grad = False
        
        # Build inference model
        logging.info(f"[{mode_prefix}] [{args.process_id}] Creating inference model")
        model = MultimodalBTInferenceModel(
            s2_backbone=ssl_model.s2_backbone,
            s1_backbone=ssl_model.s1_backbone,
            fusion_method=config["fusion_method"],
            dim_reducer=ssl_model.dim_reducer,
        ).to(device)
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
            # Process multiple tiles for GPU mode
            total_start_time = time.time()
            
            total_tiles = len(tile_list)
            processed_count = 0
            skipped_count = 0
            failed_count = 0
            
            logging.info(f"[GPU] [{args.process_id}] Processing {total_tiles} tiles on GPU {args.gpu_id}")
            
            for i, tile_path in enumerate(tile_list):
                tile_name = os.path.basename(tile_path)
                output_path = os.path.join(output_dir, f"{tile_name}.npy")
                
                # Skip if already processed
                if os.path.exists(output_path):
                    logging.info(f"[GPU] [{args.process_id}] Skipping already processed tile: {tile_name} [{i+1}/{total_tiles}]")
                    skipped_count += 1
                    continue
                
                # Process current tile
                logging.info(f"[GPU] [{args.process_id}] Processing tile {tile_name} [{i+1}/{total_tiles}]")
                
                try:
                    process_tile(tile_path, output_path, model, config, device, args.process_id, args)
                    processed_count += 1
                    
                    # Extra log after successful processing
                    logging.info(f"[GPU] [{args.process_id}] Successfully processed {tile_name}")
                    logging.info(f"[GPU] [{args.process_id}] {get_gpu_memory_info(device_id)}")
                    
                except Exception as e:
                    logging.error(f"[GPU] [{args.process_id}] Failed to process tile {tile_name} [{i+1}/{total_tiles}]: {str(e)}")
                    logging.error(traceback.format_exc())
                    failed_count += 1
                    
                    # Try to recover GPU memory
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                            logging.info(f"[GPU] [{args.process_id}] Cleared GPU cache: {get_gpu_memory_info(device_id)}")
                        except Exception as cache_error:
                            logging.error(f"[GPU] [{args.process_id}] Error clearing GPU cache: {str(cache_error)}")
                    
                    # Continue with next tile instead of exiting
                    continue
                
                # Log progress statistics
                elapsed = time.time() - total_start_time
                tiles_per_hour = (processed_count + skipped_count) / (elapsed / 3600) if elapsed > 0 else 0
                remaining = (total_tiles - (processed_count + skipped_count + failed_count)) / tiles_per_hour if tiles_per_hour > 0 else 0
                
                log_msg = f"[GPU] [{args.process_id}] Progress: {processed_count + skipped_count + failed_count}/{total_tiles} " \
                         f"(processed: {processed_count}, skipped: {skipped_count}, failed: {failed_count}) - " \
                         f"Time: {format_time(elapsed)} - " \
                         f"Rate: {tiles_per_hour:.1f} tiles/hour - " \
                         f"ETA: {format_time(remaining*3600)}"
                
                logging.info(log_msg)
                logging.info(f"[GPU] [{args.process_id}] {get_gpu_memory_info(device_id)}")
                logging.info(f"[GPU] [{args.process_id}] {log_system_info()}")
                
            # Final statistics
            total_time = time.time() - total_start_time
            log_msg = f"[GPU] [{args.process_id}] Completed GPU processing in {format_time(total_time)}"
            logging.info(log_msg)
            log_msg = f"[GPU] [{args.process_id}] Results: {processed_count} processed, {skipped_count} skipped, {failed_count} failed"
            logging.info(log_msg)
            logging.info(f"[GPU] [{args.process_id}] Final GPU state: {get_gpu_memory_info(device_id)}")
            
        logging.info(f"[{mode_prefix}] [{args.process_id}] Processing complete")
        
    except Exception as e:
        logging.error(f"[{mode_prefix}] [{args.process_id}] Unhandled error: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()