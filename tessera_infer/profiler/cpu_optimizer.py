#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import platform
import subprocess
import psutil
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

# Try to import cpuinfo, fallback to basic detection if not available
try:
    import cpuinfo
    CPUINFO_AVAILABLE = True
except ImportError:
    CPUINFO_AVAILABLE = False

class CPUPlatformDetector:
    """
    CPU平台检测和优化器
    检测Intel/AMD CPU特性并提供相应的优化建议和配置
    """
    
    def __init__(self):
        self.cpu_info = {}
        self.optimization_config = {}
        self.logger = logging.getLogger(__name__)
        
        # 初始化CPU信息
        self._detect_cpu_info()
        self._analyze_cpu_capabilities()
        self._generate_optimization_config()
    
    def _detect_cpu_info(self):
        """检测CPU基本信息"""
        try:
            # 使用cpuinfo库获取详细CPU信息
            if CPUINFO_AVAILABLE:
                info = cpuinfo.get_cpu_info()
                
                self.cpu_info = {
                    'vendor': info.get('vendor_id_raw', 'Unknown'),
                    'brand': info.get('brand_raw', 'Unknown'),
                    'arch': info.get('arch_string_raw', platform.machine()),
                    'bits': info.get('bits', 64),
                    'count': info.get('count', psutil.cpu_count()),
                    'hz_advertised': info.get('hz_advertised_friendly', '0 Hz'),
                    'hz_actual': info.get('hz_actual_friendly', '0 Hz'),
                    'l3_cache_size': info.get('l3_cache_size', 'Unknown'),
                    'l2_cache_size': info.get('l2_cache_size', 'Unknown'),
                    'l1_data_cache_size': info.get('l1_data_cache_size', 'Unknown'),
                    'l1_instruction_cache_size': info.get('l1_instruction_cache_size', 'Unknown'),
                    'flags': info.get('flags', []),
                    'family': info.get('family', 'Unknown'),
                    'model': info.get('model', 'Unknown'),
                    'stepping': info.get('stepping', 'Unknown')
                }
            else:
                # Fallback when cpuinfo is not available
                self.cpu_info = {
                    'vendor': 'Unknown',
                    'brand': 'Unknown',
                    'arch': platform.machine(),
                    'bits': 64,
                    'count': psutil.cpu_count() or 1,
                    'flags': []
                }
            
            # 使用psutil获取额外信息
            self.cpu_info.update({
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'max_frequency': psutil.cpu_freq().max if psutil.cpu_freq() else None,
                'min_frequency': psutil.cpu_freq().min if psutil.cpu_freq() else None,
                'current_frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None
            })
            
            # 确定CPU供应商
            vendor_lower = self.cpu_info['vendor'].lower()
            if 'intel' in vendor_lower or 'genuineintel' in vendor_lower:
                self.cpu_info['vendor_normalized'] = 'intel'
            elif 'amd' in vendor_lower or 'authenticamd' in vendor_lower:
                self.cpu_info['vendor_normalized'] = 'amd'
            else:
                self.cpu_info['vendor_normalized'] = 'unknown'
                
        except Exception as e:
            self.logger.warning(f"Failed to detect CPU info: {e}")
            # 回退到基本信息
            self.cpu_info = {
                'vendor': 'Unknown',
                'vendor_normalized': 'unknown',
                'brand': 'Unknown',
                'physical_cores': psutil.cpu_count(logical=False) or 1,
                'logical_cores': psutil.cpu_count(logical=True) or 1,
                'flags': []
            }
    
    def _analyze_cpu_capabilities(self):
        """分析CPU能力和支持的指令集"""
        flags = self.cpu_info.get('flags', [])
        capabilities = {
            'sse': any(flag.startswith('sse') for flag in flags),
            'sse2': 'sse2' in flags,
            'sse3': 'sse3' in flags,
            'ssse3': 'ssse3' in flags,
            'sse4_1': 'sse4_1' in flags,
            'sse4_2': 'sse4_2' in flags,
            'avx': 'avx' in flags,
            'avx2': 'avx2' in flags,
            'avx512f': 'avx512f' in flags,
            'fma': 'fma' in flags,
            'fma4': 'fma4' in flags,
            'bmi1': 'bmi1' in flags,
            'bmi2': 'bmi2' in flags,
            'aes': 'aes' in flags,
            'sha': 'sha_ni' in flags or 'sha' in flags,
            'hyperthreading': self.cpu_info.get('logical_cores', 0) > self.cpu_info.get('physical_cores', 0),
            'virtualization': 'vmx' in flags or 'svm' in flags
        }
        
        self.cpu_info['capabilities'] = capabilities
    
    def _generate_optimization_config(self):
        """根据CPU特性生成优化配置"""
        vendor = self.cpu_info.get('vendor_normalized', 'unknown')
        capabilities = self.cpu_info.get('capabilities', {})
        
        config = {
            'vendor_specific': {},
            'threading': {},
            'memory': {},
            'compiler_flags': [],
            'environment_variables': {},
            'recommended_libraries': []
        }
        
        # Intel特定优化
        if vendor == 'intel':
            config['vendor_specific'] = {
                'use_mkl': True,
                'mkl_threads': self._get_optimal_mkl_threads(),
                'mkl_domain_threads': True,
                'use_intel_extensions': True
            }
            
            config['environment_variables'].update({
                'MKL_NUM_THREADS': str(self._get_optimal_mkl_threads()),
                'MKL_DOMAIN_NUM_THREADS': f"MKL_DOMAIN_BLAS={self._get_optimal_mkl_threads()}",
                'KMP_AFFINITY': 'granularity=fine,verbose,compact,1,0',
                'KMP_BLOCKTIME': '1',
                'KMP_SETTINGS': '1'
            })
            
            config['recommended_libraries'].extend(['mkl', 'intel-extension-for-pytorch'])
        
        # AMD特定优化
        elif vendor == 'amd':
            config['vendor_specific'] = {
                'use_openblas': True,
                'use_amd_optimizations': True,
                'openblas_threads': self._get_optimal_thread_count()
            }
            
            config['environment_variables'].update({
                'OPENBLAS_NUM_THREADS': str(self._get_optimal_thread_count()),
                'OMP_NUM_THREADS': str(self._get_optimal_thread_count()),
                'MKL_NUM_THREADS': str(self._get_optimal_thread_count()),
                'VECLIB_MAXIMUM_THREADS': str(self._get_optimal_thread_count()),
                'NUMEXPR_NUM_THREADS': str(self._get_optimal_thread_count())
            })
            
            config['recommended_libraries'].extend(['openblas', 'amd-optimized-libs'])
        
        # 通用线程优化
        config['threading'] = {
            'optimal_threads': self._get_optimal_thread_count(),
            'intra_op_threads': self._get_optimal_intra_op_threads(),
            'inter_op_threads': self._get_optimal_inter_op_threads(),
            'use_all_cores': False  # 通常不建议使用所有核心
        }
        
        # 内存优化
        config['memory'] = {
            'use_memory_pool': True,
            'pin_memory': capabilities.get('hyperthreading', False),
            'cache_optimization': True
        }
        
        # 编译器标志
        if capabilities.get('avx512f'):
            config['compiler_flags'].extend(['-mavx512f', '-mfma'])
        elif capabilities.get('avx2'):
            config['compiler_flags'].extend(['-mavx2', '-mfma'])
        elif capabilities.get('avx'):
            config['compiler_flags'].extend(['-mavx'])
        elif capabilities.get('sse4_2'):
            config['compiler_flags'].extend(['-msse4.2'])
        
        if capabilities.get('aes'):
            config['compiler_flags'].append('-maes')
        
        self.optimization_config = config
    
    def _get_optimal_thread_count(self) -> int:
        """获取最优线程数"""
        physical_cores = self.cpu_info.get('physical_cores', 1)
        logical_cores = self.cpu_info.get('logical_cores', 1)
        
        # 对于CPU密集型任务，通常使用物理核心数
        # 但如果有超线程且核心数较少，可以考虑使用逻辑核心数
        if physical_cores <= 4 and logical_cores > physical_cores:
            return logical_cores
        else:
            return max(1, physical_cores - 1)  # 保留一个核心给系统
    
    def _get_optimal_mkl_threads(self) -> int:
        """获取MKL最优线程数"""
        # MKL通常在物理核心数时表现最佳
        return self.cpu_info.get('physical_cores', 1)
    
    def _get_optimal_intra_op_threads(self) -> int:
        """获取intra-op最优线程数"""
        return self._get_optimal_thread_count()
    
    def _get_optimal_inter_op_threads(self) -> int:
        """获取inter-op最优线程数"""
        # inter-op线程通常设置为1或2
        return min(2, max(1, self.cpu_info.get('physical_cores', 1) // 4))
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取完整系统信息"""
        memory = psutil.virtual_memory()
        
        system_info = {
            'cpu_info': self.cpu_info,
            'memory_info': {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'percentage': memory.percent
            },
            'platform_info': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version()
            },
            'optimization_config': self.optimization_config
        }
        
        return system_info
    
    def apply_optimizations(self, model_type: str = 'onnx') -> Dict[str, str]:
        """应用CPU优化配置"""
        applied_optimizations = {}
        env_vars = self.optimization_config.get('environment_variables', {})
        
        # 应用环境变量
        for key, value in env_vars.items():
            os.environ[key] = str(value)
            applied_optimizations[key] = str(value)
        
        # 根据模型类型应用特定优化
        if model_type.lower() == 'onnx':
            applied_optimizations.update(self._apply_onnx_optimizations())
        elif model_type.lower() == 'pytorch':
            applied_optimizations.update(self._apply_pytorch_optimizations())
        
        return applied_optimizations
    
    def _apply_onnx_optimizations(self) -> Dict[str, str]:
        """应用ONNX特定优化"""
        optimizations = {}
        
        # 设置ONNX Runtime执行提供者选项
        vendor = self.cpu_info.get('vendor_normalized', 'unknown')
        
        if vendor == 'intel':
            # Intel CPU优化
            optimizations['ONNX_EXECUTION_PROVIDERS'] = 'CPUExecutionProvider'
            optimizations['ONNX_CPU_OPTIMIZATION'] = 'intel'
        elif vendor == 'amd':
            # AMD CPU优化
            optimizations['ONNX_EXECUTION_PROVIDERS'] = 'CPUExecutionProvider'
            optimizations['ONNX_CPU_OPTIMIZATION'] = 'amd'
        
        return optimizations
    
    def _apply_pytorch_optimizations(self) -> Dict[str, str]:
        """应用PyTorch特定优化"""
        optimizations = {}
        
        # PyTorch线程设置
        optimal_threads = self._get_optimal_thread_count()
        
        try:
            import torch
            torch.set_num_threads(optimal_threads)
            torch.set_num_interop_threads(self._get_optimal_inter_op_threads())
            
            optimizations['TORCH_NUM_THREADS'] = str(optimal_threads)
            optimizations['TORCH_INTEROP_THREADS'] = str(self._get_optimal_inter_op_threads())
            
            # Intel特定PyTorch优化
            if self.cpu_info.get('vendor_normalized') == 'intel':
                try:
                    import intel_extension_for_pytorch as ipex
                    optimizations['INTEL_EXTENSION_LOADED'] = 'true'
                except ImportError:
                    optimizations['INTEL_EXTENSION_AVAILABLE'] = 'false'
                    
        except ImportError:
            pass
        
        return optimizations
    
    def get_performance_recommendations(self) -> List[Dict[str, str]]:
        """获取性能优化建议"""
        recommendations = []
        
        vendor = self.cpu_info.get('vendor_normalized', 'unknown')
        capabilities = self.cpu_info.get('capabilities', {})
        physical_cores = self.cpu_info.get('physical_cores', 1)
        
        # CPU供应商特定建议
        if vendor == 'intel':
            recommendations.append({
                'category': 'Intel Optimization',
                'recommendation': 'Use Intel MKL for optimal BLAS performance',
                'command': 'conda install mkl mkl-service intel-openmp',
                'priority': 'high'
            })
            
            recommendations.append({
                'category': 'Intel Extensions',
                'recommendation': 'Install Intel Extension for PyTorch for additional optimizations',
                'command': 'pip install intel_extension_for_pytorch',
                'priority': 'medium'
            })
            
        elif vendor == 'amd':
            recommendations.append({
                'category': 'AMD Optimization',
                'recommendation': 'Use OpenBLAS optimized for AMD CPUs',
                'command': 'conda install openblas=*=openmp*',
                'priority': 'high'
            })
        
        # 指令集建议
        if capabilities.get('avx512f'):
            recommendations.append({
                'category': 'Instruction Set',
                'recommendation': 'Your CPU supports AVX-512. Use optimized libraries built with AVX-512 support',
                'priority': 'medium'
            })
        elif capabilities.get('avx2'):
            recommendations.append({
                'category': 'Instruction Set',
                'recommendation': 'Your CPU supports AVX2. Ensure libraries are built with AVX2 support',
                'priority': 'medium'
            })
        
        # 线程建议
        if physical_cores >= 8:
            recommendations.append({
                'category': 'Threading',
                'recommendation': f'With {physical_cores} cores, consider parallel data loading and batch processing',
                'priority': 'medium'
            })
        
        # 内存建议
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb >= 32:
            recommendations.append({
                'category': 'Memory',
                'recommendation': 'Large memory available. Consider larger batch sizes and memory caching',
                'priority': 'low'
            })
        elif memory_gb < 8:
            recommendations.append({
                'category': 'Memory',
                'recommendation': 'Limited memory. Use smaller batch sizes and enable gradient checkpointing',
                'priority': 'high'
            })
        
        return recommendations
    
    def generate_optimization_script(self, output_path: str, model_type: str = 'both') -> str:
        """生成优化脚本"""
        script_lines = [
            '#!/bin/bash',
            '# Tessera CPU Optimization Script',
            f'# Generated for: {self.cpu_info.get("vendor")} {self.cpu_info.get("brand")}',
            f'# Physical Cores: {self.cpu_info.get("physical_cores")}',
            f'# Logical Cores: {self.cpu_info.get("logical_cores")}',
            '',
            'echo "Applying CPU optimizations for Tessera..."',
            ''
        ]
        
        # 环境变量设置
        env_vars = self.optimization_config.get('environment_variables', {})
        if env_vars:
            script_lines.append('# Environment Variables')
            for key, value in env_vars.items():
                script_lines.append(f'export {key}="{value}"')
            script_lines.append('')
        
        # 库安装建议
        recommendations = self.get_performance_recommendations()
        high_priority = [r for r in recommendations if r.get('priority') == 'high']
        
        if high_priority:
            script_lines.append('# High Priority Optimizations')
            for rec in high_priority:
                if 'command' in rec:
                    script_lines.append(f'# {rec["recommendation"]}')
                    script_lines.append(f'# {rec["command"]}')
            script_lines.append('')
        
        # 性能测试命令
        script_lines.extend([
            '# Performance Test Command',
            '# Run this to test the optimizations:',
            f'# python src/multi_tile_infer.py --enable_profiling --num_threads {self._get_optimal_thread_count()}',
            '',
            'echo "CPU optimizations applied successfully!"',
            f'echo "Recommended threads: {self._get_optimal_thread_count()}"',
            f'echo "CPU vendor: {self.cpu_info.get("vendor_normalized", "unknown")}"'
        ])
        
        # 写入脚本文件
        with open(output_path, 'w') as f:
            f.write('\n'.join(script_lines))
        
        # 使脚本可执行
        try:
            os.chmod(output_path, 0o755)
        except:
            pass
        
        return output_path

class PerformanceMonitor:
    """性能监控器，实时监控系统资源使用"""
    
    def __init__(self):
        self.cpu_detector = CPUPlatformDetector()
        self.monitoring_data = []
    
    def start_monitoring(self, interval: float = 1.0) -> Dict[str, Any]:
        """开始性能监控"""
        import threading
        import time
        
        self.monitoring_active = True
        self.monitoring_data = []
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory()
                    
                    data_point = {
                        'timestamp': time.time(),
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'memory_used_gb': memory.used / (1024**3),
                        'memory_available_gb': memory.available / (1024**3)
                    }
                    
                    # 获取CPU频率（如果可用）
                    try:
                        freq = psutil.cpu_freq()
                        if freq:
                            data_point['cpu_frequency_mhz'] = freq.current
                    except:
                        pass
                    
                    self.monitoring_data.append(data_point)
                    time.sleep(interval)
                    
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    break
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        return {"status": "monitoring_started", "interval": interval}
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """停止性能监控"""
        self.monitoring_active = False
        
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5)
        
        # 分析监控数据
        if self.monitoring_data:
            cpu_data = [d['cpu_percent'] for d in self.monitoring_data]
            memory_data = [d['memory_percent'] for d in self.monitoring_data]
            
            analysis = {
                'total_samples': len(self.monitoring_data),
                'duration_seconds': (self.monitoring_data[-1]['timestamp'] - 
                                   self.monitoring_data[0]['timestamp']) if len(self.monitoring_data) > 1 else 0,
                'cpu_stats': {
                    'avg': np.mean(cpu_data),
                    'max': np.max(cpu_data),
                    'min': np.min(cpu_data),
                    'std': np.std(cpu_data)
                },
                'memory_stats': {
                    'avg': np.mean(memory_data),
                    'max': np.max(memory_data),
                    'min': np.min(memory_data),
                    'std': np.std(memory_data)
                },
                'raw_data': self.monitoring_data
            }
            
            return analysis
        
        return {"status": "no_data_collected"}