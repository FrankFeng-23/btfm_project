#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test script for profiling system debugging

This script tests the profiling system with minimal complexity
to identify where the performance bottleneck occurs.
"""

import os
import sys
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

def test_onnx_profiler_basic():
    """Test basic ONNX profiler functionality"""
    print("🔧 Testing Basic ONNX Profiler...")
    
    try:
        from profiler.onnx_profiler import ONNXProfiler
        from profiler.base_profiler import ProfileMetrics
        
        # Create a basic profiler
        profiler = ONNXProfiler("/tmp", "test_onnx")
        
        # Create dummy metrics
        metrics = ProfileMetrics(
            model_type="ONNX",
            total_inference_time=5.2,
            throughput=123.4,
            peak_memory_usage=256.7
        )
        
        # Test saving metrics
        print("   Testing save_metrics...")
        start_time = time.time()
        metrics_file = profiler.save_metrics("test_metrics.json")
        save_time = time.time() - start_time
        
        print(f"   ✅ Metrics saved in {save_time:.3f}s: {metrics_file}")
        
        if os.path.exists(metrics_file):
            os.remove(metrics_file)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Basic profiler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics_collector_basic():
    """Test basic metrics collector functionality"""
    print("🔧 Testing Basic Metrics Collector...")
    
    try:
        from profiler.metrics import MetricsCollector
        from profiler.base_profiler import ProfileMetrics
        
        # Create collector
        collector = MetricsCollector("/tmp")
        
        # Add simple metrics
        print("   Adding test metrics...")
        metrics = ProfileMetrics(
            model_type="ONNX",
            total_inference_time=5.2,
            throughput=123.4,
            peak_memory_usage=256.7
        )
        
        start_time = time.time()
        collector.add_metrics(metrics, "test")
        add_time = time.time() - start_time
        print(f"   ✅ Metrics added in {add_time:.3f}s")
        
        # Test simple operations
        print("   Testing get_metrics_by_type...")
        start_time = time.time()
        onnx_metrics = collector.get_metrics_by_type('onnx')
        get_time = time.time() - start_time
        print(f"   ✅ get_metrics_by_type completed in {get_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Metrics collector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run debugging tests"""
    print("🐛 Tessera Profiling Debug Suite")
    print("=" * 40)
    
    tests = [
        ("Basic ONNX Profiler", test_onnx_profiler_basic),
        ("Basic Metrics Collector", test_metrics_collector_basic),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\\nRunning {test_name}...")
        try:
            start_time = time.time()
            success = test_func()
            test_time = time.time() - start_time
            results.append((test_name, success, test_time))
            
            if success:
                print(f"✅ {test_name}: PASSED ({test_time:.3f}s)")
            else:
                print(f"❌ {test_name}: FAILED ({test_time:.3f}s)")
                
        except Exception as e:
            test_time = time.time() - start_time
            print(f"❌ {test_name}: ERROR - {e} ({test_time:.3f}s)")
            results.append((test_name, False, test_time))
    
    # Summary
    print("\\n" + "=" * 40)
    print("Test Results Summary:")
    passed = 0
    for test_name, success, test_time in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status} ({test_time:.3f}s)")
        if success:
            passed += 1
    
    print(f"\\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\\n🎉 All basic tests passed!")
        print("\\n📋 Debugging recommendations:")
        print("   1. Try running profiling again with the simplified export_results")
        print("   2. Check if the hang occurs in a specific step")
        print("   3. Use the DEBUG output to identify the bottleneck")
    else:
        print("\\n⚠️  Some tests failed. Fix basic issues first.")
    
    return 0 if passed == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())