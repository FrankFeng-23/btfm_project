#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-tile QAT Inference Script
Combines HPC QAT quantization with daintree advanced features
Supports CPU, GPU, and XPU processing modes
"""

import os
import sys
import time
import json
import logging
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.args_parser import parse_args
from utils.common import setup_logging, log_system_info, format_time, force_log_flush
from utils.device_utils import setup_device
from utils.model_loader import ModelLoader
from inference.tile_processor import TileProcessor


class QATInferenceManager:
    """Main manager for QAT inference pipeline"""
    
    def __init__(self, args):
        self.args = args
        self.config = self.load_config()
        self.device, self.mode_prefix = setup_device(args)
        
        # Override config with command line args if provided
        if args.batch_size:
            self.config['batch_size'] = args.batch_size
        if args.num_workers:
            self.config['num_workers'] = args.num_workers
            
        self.model = None
        self.tile_processor = None
    
    def load_config(self):
        """Load configuration file (Python module)"""
        try:
            import importlib.util
            import sys
            
            # Load config as Python module
            spec = importlib.util.spec_from_file_location("config_module", self.args.config)
            config_module = importlib.util.module_from_spec(spec)
            sys.modules["config_module"] = config_module
            spec.loader.exec_module(config_module)
            
            config = config_module.config
            
            # Override with command line arguments
            config["checkpoint_path"] = self.args.checkpoint_path
            config["output_dir"] = self.args.output_dir
            
            logging.info(f"[{self.args.mode.upper()}] [{self.args.process_id}] Config loaded from {self.args.config}")
            return config
        except Exception as e:
            logging.error(f"[{self.args.mode.upper()}] [{self.args.process_id}] Failed to load config: {e}")
            raise
    
    def setup_model(self):
        """Setup model using ModelLoader"""
        logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Setting up model...")
        model_loader = ModelLoader(self.config, self.device, self.args)
        self.model = model_loader.load_model()
        
        # Create tile processor
        self.tile_processor = TileProcessor(self.model, self.config, self.device, self.args)
        logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Model setup complete")
    
    def process_single_tile(self):
        """Process a single tile (CPU mode)"""
        if not self.args.tile_path:
            raise ValueError("CPU mode requires --tile_path")
            
        tile_path = self.args.tile_path
        tile_name = os.path.basename(tile_path)
        output_path = os.path.join(self.args.output_dir, f"{tile_name}.npy")
        
        logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Processing single tile: {tile_name}")
        start_time = time.time()
        
        try:
            result_path = self.tile_processor.process_tile(tile_path, output_path)
            
            elapsed_time = time.time() - start_time
            logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Single tile processing complete in {format_time(elapsed_time)}")
            logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Results saved to: {result_path}")
            
            return result_path
            
        except Exception as e:
            logging.error(f"[{self.mode_prefix}] [{self.args.process_id}] Single tile processing failed: {e}")
            raise
    
    def process_tile_batch(self):
        """Process multiple tiles (GPU/XPU mode)"""
        if not self.args.tile_list:
            raise ValueError("GPU/XPU mode requires --tile_list")
            
        # Load tile list
        try:
            with open(self.args.tile_list, 'r') as f:
                tile_list = json.load(f)
            logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Loaded {len(tile_list)} tiles from {self.args.tile_list}")
        except Exception as e:
            logging.error(f"[{self.mode_prefix}] [{self.args.process_id}] Failed to load tile list: {e}")
            raise
        
        # Process tiles sequentially
        start_time = time.time()
        processed_tiles = []
        failed_tiles = []
        
        for i, tile_info in enumerate(tile_list):
            tile_path = tile_info.get('path', tile_info) if isinstance(tile_info, dict) else tile_info
            tile_name = os.path.basename(tile_path)
            output_path = os.path.join(self.args.output_dir, f"{tile_name}.npy")
            
            try:
                logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Processing tile {i+1}/{len(tile_list)}: {tile_name}")
                result_path = self.tile_processor.process_tile(tile_path, output_path)
                processed_tiles.append(result_path)
                
            except Exception as e:
                logging.error(f"[{self.mode_prefix}] [{self.args.process_id}] Failed to process tile {tile_name}: {e}")
                failed_tiles.append(tile_path)
                continue
        
        elapsed_time = time.time() - start_time
        
        # Log summary
        logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Batch processing complete in {format_time(elapsed_time)}")
        logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Successfully processed: {len(processed_tiles)}/{len(tile_list)} tiles")
        
        if failed_tiles:
            logging.warning(f"[{self.mode_prefix}] [{self.args.process_id}] Failed tiles: {len(failed_tiles)}")
            for failed_tile in failed_tiles:
                logging.warning(f"[{self.mode_prefix}] [{self.args.process_id}] - {failed_tile}")
        
        return processed_tiles, failed_tiles
    
    def run(self):
        """Main execution method"""
        logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Starting QAT inference pipeline")
        pipeline_start = time.time()
        
        try:
            # Setup model
            self.setup_model()
            
            # Process tiles based on mode
            if self.args.mode == 'cpu':
                result = self.process_single_tile()
                return result
            else:  # gpu or xpu
                processed_tiles, failed_tiles = self.process_tile_batch()
                return processed_tiles, failed_tiles
                
        except Exception as e:
            logging.error(f"[{self.mode_prefix}] [{self.args.process_id}] Pipeline failed: {e}")
            raise
        finally:
            pipeline_time = time.time() - pipeline_start
            logging.info(f"[{self.mode_prefix}] [{self.args.process_id}] Total pipeline time: {format_time(pipeline_time)}")
            force_log_flush()


def main():
    """Main entry point"""
    try:
        # Parse arguments
        args = parse_args()
        
        # Setup logging
        setup_logging(args)
        
        # Log system information
        log_system_info()
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f"[{args.mode.upper()}] [{args.process_id}] Output directory: {args.output_dir}")
        
        # Log configuration summary
        logging.info(f"[{args.mode.upper()}] [{args.process_id}] Mode: {args.mode}")
        logging.info(f"[{args.mode.upper()}] [{args.process_id}] Checkpoint: {args.checkpoint_path}")
        if args.onnx_model_path:
            logging.info(f"[{args.mode.upper()}] [{args.process_id}] ONNX Model: {args.onnx_model_path}")
        if args.enable_bf16:
            logging.info(f"[{args.mode.upper()}] [{args.process_id}] BF16 precision enabled")
        if args.enable_amx:
            logging.info(f"[{args.mode.upper()}] [{args.process_id}] AMX acceleration requested")
        if args.enable_profiling:
            logging.info(f"[{args.mode.upper()}] [{args.process_id}] Profiling enabled")
        if args.max_batches:
            logging.info(f"[{args.mode.upper()}] [{args.process_id}] Max batches: {args.max_batches}")
        
        # Run inference
        manager = QATInferenceManager(args)
        result = manager.run()
        
        logging.info(f"[{args.mode.upper()}] [{args.process_id}] QAT inference completed successfully")
        return result
        
    except Exception as e:
        logging.error(f"Main execution failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()