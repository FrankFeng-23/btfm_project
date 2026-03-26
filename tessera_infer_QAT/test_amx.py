#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AMX Support Test Script
Tests AMX (Advanced Matrix Extensions) functionality and performance
"""

import os
import sys
import time
import logging
import subprocess

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.device_utils import check_amx_support, setup_amx_environment

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AMXArgs:
    """Mock args object for testing"""
    def __init__(self, enable_amx=True, enable_bf16=True):
        self.enable_amx = enable_amx
        self.enable_bf16 = enable_bf16
        self.mode = 'cpu'
        self.process_id = 0
        self.log_level = 'DEBUG'

def test_cpu_info():
    """Test CPU information detection"""
    logging.info("=== Testing CPU Information ===")
    try:
        result = subprocess.run(['cat', '/proc/cpuinfo'], capture_output=True, text=True, check=True)
        cpu_info = result.stdout.lower()
        
        # Check for relevant CPU flags
        flags_to_check = [
            'amx_bf16', 'amx_int8', 'amx_tile', 'avx512f', 'avx512bw', 
            'avx512dq', 'avx512vl', 'avx512_vnni', 'avx512_bf16'
        ]
        
        found_flags = []
        for flag in flags_to_check:
            if flag in cpu_info:
                found_flags.append(flag)
        
        logging.info(f"Found CPU flags: {found_flags}")
        
        # Get CPU model
        for line in result.stdout.split('\n'):
            if 'model name' in line:
                cpu_model = line.split(':')[1].strip()
                logging.info(f"CPU Model: {cpu_model}")
                break
                
    except Exception as e:
        logging.error(f"Failed to get CPU info: {e}")

def test_amx_detection():
    """Test AMX support detection"""
    logging.info("=== Testing AMX Detection ===")
    
    amx_info = check_amx_support()
    
    logging.info(f"AMX Support Results:")
    logging.info(f"  AMX BF16: {amx_info['amx_bf16']}")
    logging.info(f"  AMX INT8: {amx_info['amx_int8']}")
    logging.info(f"  AVX512: {amx_info['avx512']}")
    logging.info(f"  Overall Supported: {amx_info['supported']}")
    
    return amx_info['supported']

def test_amx_environment():
    """Test AMX environment setup"""
    logging.info("=== Testing AMX Environment Setup ===")
    
    # Test with AMX enabled
    args_amx = AMXArgs(enable_amx=True, enable_bf16=True)
    amx_enabled = setup_amx_environment(args_amx)
    
    logging.info(f"AMX Environment Setup: {'Success' if amx_enabled else 'Failed'}")
    
    # Print environment variables that were set
    env_vars = ['ONEDNN_VERBOSE', 'DNNL_MAX_CPU_ISA', 'ONEDNN_DEFAULT_FPMATH_MODE']
    for var in env_vars:
        value = os.environ.get(var, 'Not Set')
        logging.info(f"  {var}: {value}")
    
    return amx_enabled

def test_pytorch_amx():
    """Test PyTorch with AMX"""
    logging.info("=== Testing PyTorch AMX Integration ===")
    
    try:
        import torch
        import torch.nn as nn
        
        # Create a simple test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(128, 256)
                self.linear2 = nn.Linear(256, 128)
                
            def forward(self, x):
                x = self.linear1(x)
                x = torch.relu(x)
                x = self.linear2(x)
                return x
        
        model = TestModel()
        model.eval()
        
        # Test with BF16 + AMX autocast
        logging.info("Testing BF16 + AMX autocast...")
        with torch.amp.autocast('cpu', dtype=torch.bfloat16, enabled=True):
            test_input = torch.randn(32, 128, dtype=torch.float32)
            
            # Time the operation
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    output = model(test_input)
            inference_time = time.time() - start_time
            
            logging.info(f"BF16 + AMX inference time (10 iterations): {inference_time:.4f}s")
            logging.info(f"Output shape: {output.shape}, dtype: {output.dtype}")
        
        # Test without AMX (FP32 only)
        logging.info("Testing FP32 baseline...")
        model_fp32 = TestModel()
        model_fp32.eval()
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                test_input_fp32 = torch.randn(32, 128, dtype=torch.float32)
                output_fp32 = model_fp32(test_input_fp32)
        fp32_time = time.time() - start_time
        
        logging.info(f"FP32 baseline time (10 iterations): {fp32_time:.4f}s")
        
        if inference_time > 0:
            speedup = fp32_time / inference_time
            logging.info(f"Potential speedup: {speedup:.2f}x")
        
        return True
        
    except Exception as e:
        logging.error(f"PyTorch AMX test failed: {e}")
        return False

def test_onednn_verbose():
    """Test oneDNN verbose output for AMX verification"""
    logging.info("=== Testing oneDNN Verbose Output ===")
    
    try:
        import torch
        import torch.nn as nn
        
        # Set oneDNN verbose
        os.environ['ONEDNN_VERBOSE'] = '1'
        
        model = nn.Linear(128, 256)
        model.eval()
        
        logging.info("Running inference with oneDNN verbose enabled...")
        logging.info("Look for 'amx_bf16' or 'amx_int8' in the output below:")
        
        with torch.amp.autocast('cpu', dtype=torch.bfloat16, enabled=True):
            test_input = torch.randn(1, 128, dtype=torch.float32)
            with torch.no_grad():
                output = model(test_input)
        
        logging.info("oneDNN verbose test completed")
        return True
        
    except Exception as e:
        logging.error(f"oneDNN verbose test failed: {e}")
        return False

def main():
    """Main test function"""
    logging.info("Starting AMX Support Tests")
    logging.info("=" * 50)
    
    # Test CPU information
    test_cpu_info()
    
    # Test AMX detection
    amx_supported = test_amx_detection()
    
    # Test environment setup
    amx_env_ok = test_amx_environment()
    
    # Test PyTorch integration
    pytorch_ok = test_pytorch_amx()
    
    # Test oneDNN verbose if AMX is supported
    if amx_supported:
        test_onednn_verbose()
    
    # Summary
    logging.info("=" * 50)
    logging.info("AMX Test Summary:")
    logging.info(f"  AMX Hardware Support: {'✓' if amx_supported else '✗'}")
    logging.info(f"  AMX Environment Setup: {'✓' if amx_env_ok else '✗'}")
    logging.info(f"  PyTorch Integration: {'✓' if pytorch_ok else '✗'}")
    
    if not amx_supported:
        logging.warning("AMX is not supported on this hardware (likely AMD CPU)")
        logging.info("This is expected - test on Intel 4th Gen Xeon for AMX support")
    
    logging.info("AMX testing completed")

if __name__ == "__main__":
    main()