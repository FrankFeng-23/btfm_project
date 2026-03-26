#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert PyTorch model to ONNX format for faster CPU inference
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
import importlib.util
import onnx
import onnxruntime as ort

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.ssl_model import MultimodalBTInferenceModel
from src.models.builder import build_ssl_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config_module(config_file_path):
    """Load config module from file path"""
    try:
        if not os.path.exists(config_file_path):
            logger.error(f"Config file does not exist: {config_file_path}")
            sys.exit(1)
            
        spec = importlib.util.spec_from_file_location("config", config_file_path)
        config_module = importlib.util.module_from_spec(spec)
        sys.modules["config"] = config_module
        spec.loader.exec_module(config_module)
        return config_module
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        sys.exit(1)

def export_to_onnx(model, config, output_path, opset_version=14):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model: PyTorch model
        config: Configuration dictionary
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
    """
    model.eval()
    
    # Create dummy inputs based on your inference requirements
    batch_size = 1  # You can adjust this
    sample_size_s2 = config["sample_size_s2"]
    sample_size_s1 = config["sample_size_s1"]
    
    # S2 input: (batch, sample_size_s2, 11) - 10 bands + 1 doy
    dummy_s2_input = torch.randn(batch_size, sample_size_s2, 11, dtype=torch.float32)
    
    # S1 input: (batch, sample_size_s1, 3) - 2 bands + 1 doy
    dummy_s1_input = torch.randn(batch_size, sample_size_s1, 3, dtype=torch.float32)
    
    # Export the model
    logger.info(f"Exporting model to ONNX format...")
    logger.info(f"Input shapes: S2={dummy_s2_input.shape}, S1={dummy_s1_input.shape}")
    
    # Define input and output names
    input_names = ['s2_input', 's1_input']
    output_names = ['output']
    
    # Dynamic axes for batch size flexibility
    dynamic_axes = {
        's2_input': {0: 'batch_size'},
        's1_input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
    
    try:
        torch.onnx.export(
            model,
            (dummy_s2_input, dummy_s1_input),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        logger.info(f"Successfully exported model to {output_path}")
        
        # Verify the exported model
        logger.info("Verifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification passed!")
        
        # Test inference with ONNX Runtime
        logger.info("Testing ONNX Runtime inference...")
        ort_session = ort.InferenceSession(output_path)
        
        # Prepare inputs
        ort_inputs = {
            's2_input': dummy_s2_input.numpy(),
            's1_input': dummy_s1_input.numpy()
        }
        
        # Run inference
        ort_outputs = ort_session.run(None, ort_inputs)
        logger.info(f"ONNX Runtime inference successful! Output shape: {ort_outputs[0].shape}")
        
        # Compare with PyTorch output
        with torch.no_grad():
            pytorch_output = model(dummy_s2_input, dummy_s1_input)
        
        # Check if outputs are similar
        np.testing.assert_allclose(
            pytorch_output.numpy(), 
            ort_outputs[0], 
            rtol=1e-3, 
            atol=1e-5
        )
        logger.info("PyTorch and ONNX outputs match!")
        
        # Print model info
        logger.info(f"ONNX model size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        # Print input/output info
        logger.info("Model inputs:")
        for input in ort_session.get_inputs():
            logger.info(f"  - {input.name}: {input.shape} ({input.type})")
        
        logger.info("Model outputs:")
        for output in ort_session.get_outputs():
            logger.info(f"  - {output.name}: {output.shape} ({output.type})")
            
    except Exception as e:
        logger.error(f"Error during ONNX export: {str(e)}")
        raise

def optimize_onnx_model(input_path, output_path):
    """
    Optimize ONNX model using onnxruntime tools
    """
    try:
        import onnxruntime.tools.optimizer as optimizer
        
        logger.info(f"Optimizing ONNX model...")
        
        # Available optimization passes
        all_passes = optimizer.get_available_passes()
        logger.info(f"Available optimization passes: {len(all_passes)}")
        
        # Optimize the model
        optimized_model = optimizer.optimize_model(
            input_path,
            opt_level=99,  # Maximum optimization
            passes=all_passes,
            disabled_optimizers=[]  # Enable all optimizers
        )
        
        # Save optimized model
        optimized_model.save_model_to_file(output_path)
        
        # Check size reduction
        original_size = os.path.getsize(input_path) / 1024 / 1024
        optimized_size = os.path.getsize(output_path) / 1024 / 1024
        reduction = (1 - optimized_size / original_size) * 100
        
        logger.info(f"Original model size: {original_size:.2f} MB")
        logger.info(f"Optimized model size: {optimized_size:.2f} MB")
        logger.info(f"Size reduction: {reduction:.1f}%")
        
        # Test optimized model
        ort_session_opt = ort.InferenceSession(output_path)
        logger.info("Optimized model loaded successfully!")
        
        return True
        
    except ImportError:
        logger.warning("onnxruntime.tools not installed. Trying alternative optimization...")
        
        # Try alternative optimization using onnx-simplifier
        try:
            import onnxsim
            
            logger.info("Using onnx-simplifier for optimization...")
            model = onnx.load(input_path)
            model_simp, check = onnxsim.simplify(model)
            
            if check:
                onnx.save(model_simp, output_path)
                
                # Check size reduction
                original_size = os.path.getsize(input_path) / 1024 / 1024
                optimized_size = os.path.getsize(output_path) / 1024 / 1024
                reduction = (1 - optimized_size / original_size) * 100
                
                logger.info(f"Original model size: {original_size:.2f} MB")
                logger.info(f"Simplified model size: {optimized_size:.2f} MB")
                logger.info(f"Size reduction: {reduction:.1f}%")
                
                return True
            else:
                logger.error("Model simplification check failed")
                return False
                
        except ImportError:
            logger.warning("onnx-simplifier not installed.")
            logger.warning("Install with: pip install onnx-simplifier")
            return False
            
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        return False

def quantize_onnx_model(input_path, output_path):
    """
    Quantize ONNX model to INT8 for faster CPU inference
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        logger.info("Quantizing model to INT8...")
        
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QInt8,
            optimize_model=True,
            per_channel=True,
            reduce_range=True
        )
        
        # Check size reduction
        original_size = os.path.getsize(input_path) / 1024 / 1024
        quantized_size = os.path.getsize(output_path) / 1024 / 1024
        reduction = (1 - quantized_size / original_size) * 100
        
        logger.info(f"Original model size: {original_size:.2f} MB")
        logger.info(f"Quantized model size: {quantized_size:.2f} MB")
        logger.info(f"Size reduction: {reduction:.1f}%")
        
        # Test quantized model
        ort_session_quant = ort.InferenceSession(output_path)
        logger.info("Quantized model loaded successfully!")
        
        return True
        
    except ImportError:
        logger.warning("onnxruntime.quantization not installed.")
        return False
    except Exception as e:
        logger.error(f"Error during quantization: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to config file")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to PyTorch checkpoint")
    parser.add_argument('--output', type=str, default='model.onnx',
                        help="Output ONNX file path")
    parser.add_argument('--opset', type=int, default=14,
                        help="ONNX opset version")
    parser.add_argument('--optimize', action='store_true',
                        help="Apply ONNX optimization")
    parser.add_argument('--quantize', action='store_true',
                        help="Quantize model to INT8")
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading config from {args.config}")
    config_module = load_config_module(args.config)
    config = config_module.config
    
    # Update checkpoint path
    config["checkpoint_path"] = args.checkpoint
    
    # Build model
    logger.info("Building model...")
    device = torch.device("cpu")  # We use CPU for export
    ssl_model = build_ssl_model(config, device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    state_key = 'model_state' if 'model_state' in checkpoint else 'model_state_dict'
    
    if state_key not in checkpoint:
        logger.error(f"State key '{state_key}' not found in checkpoint")
        logger.error(f"Available keys: {list(checkpoint.keys())}")
        sys.exit(1)
    
    # Handle FSDP prefixes
    state_dict = checkpoint[state_key]
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    ssl_model.load_state_dict(new_state_dict, strict=True)
    logger.info("Model loaded successfully")
    
    # Create inference model
    model = MultimodalBTInferenceModel(
        s2_backbone=ssl_model.s2_backbone,
        s1_backbone=ssl_model.s1_backbone,
        fusion_method=config["fusion_method"],
        dim_reducer=ssl_model.dim_reducer,
    )
    model.eval()
    
    # Export to ONNX
    export_to_onnx(model, config, args.output, args.opset)
    
    # Optional: Optimize ONNX model
    if args.optimize:
        optimized_path = args.output.replace('.onnx', '_optimized.onnx')
        if optimize_onnx_model(args.output, optimized_path):
            logger.info(f"Optimized model saved to {optimized_path}")
        else:
            logger.info("Optimization skipped or failed, using original model")
    
    # Optional: Quantize model
    if args.quantize:
        quantized_path = args.output.replace('.onnx', '_quantized.onnx')
        if quantize_onnx_model(args.output, quantized_path):
            logger.info(f"Quantized model saved to {quantized_path}")
        else:
            logger.info("Quantization skipped or failed")
    
    logger.info("Conversion completed successfully!")

if __name__ == "__main__":
    main()