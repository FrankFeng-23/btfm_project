#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Basic test script for the profiling system
"""

import os
import sys
import tempfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

try:
    from profiler import UnifiedProfiler, PyTorchProfiler, ONNXProfiler
    print("✓ Profiler modules imported successfully")
except ImportError as e:
    print(f"✗ Failed to import profiler modules: {e}")
    sys.exit(1)

class DummyDataset(Dataset):
    """Simple dummy dataset for testing"""
    def __init__(self, size=100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            's2_input': torch.randn(32, 11),  # (seq_len, features)
            's1_input': torch.randn(32, 3),   # (seq_len, features) 
            'global_idx': idx
        }

class DummyModel(torch.nn.Module):
    """Simple dummy model for testing"""
    def __init__(self):
        super().__init__()
        self.s2_proj = torch.nn.Linear(11, 64)
        self.s1_proj = torch.nn.Linear(3, 64)
        self.fusion = torch.nn.Linear(128, 128)
    
    def forward(self, s2_input, s1_input):
        s2_feat = self.s2_proj(s2_input).mean(dim=1)  # (batch, 64)
        s1_feat = self.s1_proj(s1_input).mean(dim=1)  # (batch, 64)
        fused = torch.cat([s2_feat, s1_feat], dim=-1)  # (batch, 128)
        return self.fusion(fused)  # (batch, 128)

def test_pytorch_profiler():
    """Test PyTorch profiler"""
    print("\\n=== Testing PyTorch Profiler ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create test data
            dataset = DummyDataset(50)
            dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
            model = DummyModel()
            model.eval()
            
            # Create profiler
            profiler = PyTorchProfiler(temp_dir, "test_pytorch")
            
            # Profile inference
            results = profiler.profile_inference(model, dataloader, num_batches=5)
            
            print(f"✓ PyTorch profiling completed")
            print(f"  - Trace file: {results.get('trace_file', 'N/A')}")
            print(f"  - Batches processed: {results.get('batches_processed', 0)}")
            print(f"  - Total samples: {results.get('total_samples', 0)}")
            
            # Get metrics
            metrics = profiler.get_metrics()
            print(f"  - Throughput: {metrics.throughput:.2f} samples/sec")
            print(f"  - Peak memory: {metrics.peak_memory_usage:.1f} MB")
            
            return True
            
        except Exception as e:
            print(f"✗ PyTorch profiler test failed: {e}")
            return False

def test_unified_profiler():
    """Test unified profiler"""
    print("\\n=== Testing Unified Profiler ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create test data
            dataset = DummyDataset(30)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
            
            # Create profiler
            profiler = UnifiedProfiler(temp_dir, "test_unified")
            
            # Create model
            model = DummyModel()
            
            # Profile model
            results = profiler.profile_model_only(model, dataloader, num_batches=3)
            
            print(f"✓ Unified profiling completed")
            print(f"  - Model type: {results.get('model_type', 'N/A')}")
            print(f"  - Batches processed: {results.get('batches_processed', 0)}")
            
            # Generate reports
            exported = profiler.export_results(format="json")
            for format_name, filepath in exported.items():
                if os.path.exists(filepath):
                    print(f"  - {format_name}: ✓")
                else:
                    print(f"  - {format_name}: ✗")
            
            return True
            
        except Exception as e:
            print(f"✗ Unified profiler test failed: {e}")
            return False

def test_metrics_collection():
    """Test metrics collection"""
    print("\\n=== Testing Metrics Collection ===")
    
    try:
        from profiler.metrics import MetricsCollector
        from profiler.base_profiler import ProfileMetrics
        
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(temp_dir)
            
            # Add dummy metrics
            metrics1 = ProfileMetrics(
                model_type="PyTorch",
                total_inference_time=10.5,
                throughput=45.6,
                peak_memory_usage=512.0
            )
            
            metrics2 = ProfileMetrics(
                model_type="ONNX", 
                total_inference_time=8.2,
                throughput=58.3,
                peak_memory_usage=384.0
            )
            
            collector.add_metrics(metrics1, "pytorch_test")
            collector.add_metrics(metrics2, "onnx_test")
            
            # Generate comparison
            comparison = collector.compare_pytorch_vs_onnx()
            
            if comparison.performance_ratio:
                print("✓ Metrics collection and comparison working")
                print(f"  - Throughput ratio: {comparison.performance_ratio.get('throughput_ratio', 'N/A'):.2f}")
                print(f"  - Memory ratio: {comparison.performance_ratio.get('memory_ratio', 'N/A'):.2f}")
            else:
                print("✗ No performance ratios calculated")
                return False
            
            # Generate summary
            summary = collector.generate_summary_report()
            if summary and 'collection_info' in summary:
                print(f"  - Summary generated: {summary['collection_info']['total_runs']} runs")
                return True
            else:
                print("✗ Failed to generate summary")
                return False
                
        except Exception as e:
            print(f"✗ Metrics collection test failed: {e}")
            return False

def main():
    """Run all tests"""
    print("Tessera Profiling System Test Suite")
    print("=" * 40)
    
    tests = [
        ("PyTorch Profiler", test_pytorch_profiler),
        ("Unified Profiler", test_unified_profiler), 
        ("Metrics Collection", test_metrics_collection)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\\nRunning {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\\n" + "=" * 40)
    print("Test Results:")
    passed = 0
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\\n🎉 All tests passed! Profiling system is ready to use.")
        return 0
    else:
        print(f"\\n⚠️  {len(results) - passed} tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())