#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import json
import os

from .precision_timer import PrecisionTimer, TimingResult, time_operation

class ModuleProfiler:
    """
    PyTorch模型深度profiling系统
    精确到module、layer、甚至individual operation级别的性能分析
    """
    
    def __init__(self, name: str = "pytorch_profiler"):
        self.name = name
        self.timer = PrecisionTimer(f"{name}_timer")
        self.module_timings: Dict[str, List[float]] = defaultdict(list)
        self.forward_hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self.backward_hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self.module_info: Dict[str, Dict[str, Any]] = {}
        self.profiling_active = False
        self.detailed_analysis = True
        
        # 用于存储torch.profiler的结果
        self.torch_profiler_results = None
        
        # 线程安全
        self._lock = threading.Lock()
    
    def register_module_hooks(self, model: nn.Module, detailed: bool = True):
        """
        为模型的所有module注册profiling hooks
        
        Args:
            model: PyTorch模型
            detailed: 是否进行详细分析（包括子模块）
        """
        self.detailed_analysis = detailed
        self._clear_hooks()
        
        with self._lock:
            for name, module in model.named_modules():
                # 跳过空的容器模块（如Sequential, ModuleList等）
                if len(list(module.children())) == 0 or detailed:
                    self._register_single_module_hook(name, module)
                    
                    # 收集模块信息
                    self.module_info[name] = self._extract_module_info(module)
    
    def _register_single_module_hook(self, name: str, module: nn.Module):
        """为单个模块注册hook"""
        
        def forward_pre_hook(module, input):
            """前向传播前的hook"""
            if self.profiling_active:
                self.timer.start_timer(f"forward_{name}")
        
        def forward_hook(module, input, output):
            """前向传播后的hook"""
            if self.profiling_active:
                timing_result = self.timer.end_timer(f"forward_{name}")
                if timing_result:
                    self.module_timings[name].append(timing_result.duration)
        
        # 注册hooks
        pre_handle = module.register_forward_pre_hook(forward_pre_hook)
        post_handle = module.register_forward_hook(forward_hook)
        
        self.forward_hooks[f"{name}_pre"] = pre_handle
        self.forward_hooks[f"{name}_post"] = post_handle
    
    def _extract_module_info(self, module: nn.Module) -> Dict[str, Any]:
        """提取模块详细信息"""
        info = {
            "type": type(module).__name__,
            "parameters": sum(p.numel() for p in module.parameters()),
            "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad),
        }
        
        # 特定模块类型的详细信息
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            info.update({
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "kernel_size": module.kernel_size,
                "stride": module.stride,
                "padding": module.padding,
            })
        elif isinstance(module, (nn.Linear,)):
            info.update({
                "in_features": module.in_features,
                "out_features": module.out_features,
            })
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            info.update({
                "num_features": module.num_features,
                "eps": module.eps,
                "momentum": module.momentum,
            })
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            info.update({
                "input_size": module.input_size,
                "hidden_size": module.hidden_size,
                "num_layers": module.num_layers,
                "bidirectional": module.bidirectional,
            })
        elif hasattr(module, 'd_model'):  # Transformer相关
            info.update({
                "d_model": getattr(module, 'd_model', None),
                "nhead": getattr(module, 'nhead', None),
                "num_layers": getattr(module, 'num_layers', None),
            })
        
        return info
    
    def _clear_hooks(self):
        """清除所有已注册的hooks"""
        for hook in self.forward_hooks.values():
            if hook is not None:
                hook.remove()
        for hook in self.backward_hooks.values():
            if hook is not None:
                hook.remove()
        
        self.forward_hooks.clear()
        self.backward_hooks.clear()
    
    @contextmanager
    def profiling_context(self, enable_torch_profiler: bool = True, torch_profiler_config: Optional[Dict] = None):
        """
        Profiling上下文管理器
        
        Args:
            enable_torch_profiler: 是否启用torch.profiler
            torch_profiler_config: torch.profiler配置
        """
        self.profiling_active = True
        
        # 默认torch.profiler配置
        if torch_profiler_config is None:
            torch_profiler_config = {
                "activities": [ProfilerActivity.CPU],
                "record_shapes": True,
                "with_modules": True,
                "with_stack": True,
                "profile_memory": True,
            }
        
        torch_prof = None
        try:
            if enable_torch_profiler:
                torch_prof = profile(**torch_profiler_config)
                torch_prof.__enter__()
            
            yield
            
        finally:
            self.profiling_active = False
            
            if torch_prof is not None:
                torch_prof.__exit__(None, None, None)
                self.torch_profiler_results = torch_prof
    
    def profile_model_inference(self, model: nn.Module, inputs: List[torch.Tensor], 
                              num_runs: int = 1, warmup_runs: int = 2) -> Dict[str, Any]:
        """
        对模型推理进行完整profiling
        
        Args:
            model: PyTorch模型
            inputs: 输入张量列表
            num_runs: 运行次数
            warmup_runs: 预热运行次数
            
        Returns:
            详细的profiling结果
        """
        model.eval()
        results = {
            "total_runs": num_runs,
            "warmup_runs": warmup_runs,
            "module_stats": {},
            "overall_stats": {},
            "torch_profiler_summary": None
        }
        
        # 注册hooks
        self.register_module_hooks(model)
        
        with torch.no_grad():
            # 预热运行
            with self.timer.timer("warmup"):
                for _ in range(warmup_runs):
                    outputs = model(*inputs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
            
            # 清除预热数据
            self.timer.clear()
            self.module_timings.clear()
            
            # 主要profiling运行
            with self.profiling_context() as prof:
                with self.timer.timer("total_inference"):
                    for run_idx in range(num_runs):
                        with self.timer.timer(f"run_{run_idx}"):
                            outputs = model(*inputs)
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
        
        # 分析结果
        results["overall_stats"] = self._analyze_overall_performance()
        results["module_stats"] = self._analyze_module_performance()
        results["torch_profiler_summary"] = self._analyze_torch_profiler_results()
        
        return results
    
    def _analyze_overall_performance(self) -> Dict[str, Any]:
        """分析整体性能"""
        stats = self.timer.get_aggregated_stats()
        percentages = self.timer.calculate_percentages()
        
        overall_stats = {}
        for name, stat in stats.items():
            overall_stats[name] = {
                "total_duration_ms": stat.total_duration_ms,
                "avg_duration_ms": stat.avg_duration_ms,
                "count": stat.count,
                "percentage": percentages.get(name, 0.0),
                "min_duration_ms": stat.min_duration * 1000,
                "max_duration_ms": stat.max_duration * 1000
            }
        
        return overall_stats
    
    def _analyze_module_performance(self) -> Dict[str, Any]:
        """分析模块级性能"""
        module_stats = {}
        
        for module_name, timings in self.module_timings.items():
            if not timings:
                continue
                
            total_time = sum(timings)
            avg_time = total_time / len(timings)
            min_time = min(timings)
            max_time = max(timings)
            
            module_stats[module_name] = {
                "total_duration_ms": total_time * 1000,
                "avg_duration_ms": avg_time * 1000,
                "min_duration_ms": min_time * 1000,
                "max_duration_ms": max_time * 1000,
                "call_count": len(timings),
                "module_info": self.module_info.get(module_name, {}),
                "timings_raw": [t * 1000 for t in timings]  # 原始时间数据(ms)
            }
        
        return module_stats
    
    def _analyze_torch_profiler_results(self) -> Optional[Dict[str, Any]]:
        """分析torch.profiler结果"""
        if self.torch_profiler_results is None:
            return None
        
        try:
            # 获取按CPU时间排序的事件
            cpu_events = self.torch_profiler_results.key_averages()
            
            # 分析最耗时的操作
            top_cpu_events = []
            for event in cpu_events:
                if event.cpu_time > 0:
                    top_cpu_events.append({
                        "name": event.key,
                        "cpu_time_ms": event.cpu_time / 1000,  # 转换为毫秒
                        "cuda_time_ms": event.cuda_time / 1000 if event.cuda_time > 0 else 0,
                        "count": event.count,
                        "cpu_memory_usage": event.cpu_memory_usage,
                        "cuda_memory_usage": event.cuda_memory_usage,
                        "input_shapes": str(event.input_shapes) if hasattr(event, 'input_shapes') else None
                    })
            
            # 按CPU时间排序
            top_cpu_events.sort(key=lambda x: x["cpu_time_ms"], reverse=True)
            
            return {
                "total_events": len(cpu_events),
                "top_events": top_cpu_events[:20],  # 前20个最耗时事件
                "total_cpu_time_ms": sum(e.cpu_time for e in cpu_events) / 1000,
                "total_cuda_time_ms": sum(e.cuda_time for e in cpu_events if e.cuda_time > 0) / 1000
            }
        
        except Exception as e:
            return {"error": f"Failed to analyze torch profiler results: {str(e)}"}
    
    def get_bottleneck_analysis(self, results: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
        """
        识别性能瓶颈
        
        Args:
            results: profiling结果
            top_k: 返回前k个瓶颈
            
        Returns:
            瓶颈分析结果
        """
        bottlenecks = {
            "top_modules_by_time": [],
            "top_operations_by_time": [],
            "memory_intensive_operations": [],
            "recommendations": []
        }
        
        # 分析模块瓶颈
        module_stats = results.get("module_stats", {})
        sorted_modules = sorted(
            module_stats.items(), 
            key=lambda x: x[1]["total_duration_ms"], 
            reverse=True
        )
        
        for module_name, stats in sorted_modules[:top_k]:
            bottlenecks["top_modules_by_time"].append({
                "module": module_name,
                "total_time_ms": stats["total_duration_ms"],
                "avg_time_ms": stats["avg_duration_ms"],
                "call_count": stats["call_count"],
                "module_type": stats["module_info"].get("type", "Unknown")
            })
        
        # 分析torch profiler瓶颈
        torch_summary = results.get("torch_profiler_summary", {})
        if torch_summary and "top_events" in torch_summary:
            for event in torch_summary["top_events"][:top_k]:
                bottlenecks["top_operations_by_time"].append({
                    "operation": event["name"],
                    "cpu_time_ms": event["cpu_time_ms"],
                    "cuda_time_ms": event["cuda_time_ms"],
                    "count": event["count"],
                    "avg_time_ms": event["cpu_time_ms"] / event["count"] if event["count"] > 0 else 0
                })
                
                # 识别内存密集型操作
                if event["cpu_memory_usage"] > 0 or event["cuda_memory_usage"] > 0:
                    bottlenecks["memory_intensive_operations"].append({
                        "operation": event["name"],
                        "cpu_memory_mb": event["cpu_memory_usage"] / (1024 * 1024),
                        "cuda_memory_mb": event["cuda_memory_usage"] / (1024 * 1024),
                        "time_ms": event["cpu_time_ms"]
                    })
        
        # 生成优化建议
        bottlenecks["recommendations"] = self._generate_optimization_recommendations(results)
        
        return bottlenecks
    
    def _generate_optimization_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成性能优化建议"""
        recommendations = []
        
        module_stats = results.get("module_stats", {})
        torch_summary = results.get("torch_profiler_summary", {})
        
        # 基于模块分析的建议
        if module_stats:
            # 找到最耗时的模块类型
            module_times_by_type = defaultdict(float)
            for module_name, stats in module_stats.items():
                module_type = stats["module_info"].get("type", "Unknown")
                module_times_by_type[module_type] += stats["total_duration_ms"]
            
            top_module_type = max(module_times_by_type.items(), key=lambda x: x[1])
            if top_module_type[1] > 100:  # 如果某类型模块耗时超过100ms
                recommendations.append(
                    f"{top_module_type[0]} modules consume {top_module_type[1]:.1f}ms. "
                    f"Consider optimization or using more efficient alternatives."
                )
        
        # 基于torch profiler的建议
        if torch_summary and "top_events" in torch_summary:
            top_ops = torch_summary["top_events"][:5]
            for op in top_ops:
                if "conv" in op["name"].lower() and op["cpu_time_ms"] > 50:
                    recommendations.append(
                        f"Convolution operations take {op['cpu_time_ms']:.1f}ms. "
                        f"Consider using optimized conv implementations or reducing kernel sizes."
                    )
                elif "linear" in op["name"].lower() and op["cpu_time_ms"] > 30:
                    recommendations.append(
                        f"Linear layers take {op['cpu_time_ms']:.1f}ms. "
                        f"Consider model quantization or pruning."
                    )
                elif "attention" in op["name"].lower() and op["cpu_time_ms"] > 40:
                    recommendations.append(
                        f"Attention operations take {op['cpu_time_ms']:.1f}ms. "
                        f"Consider using flash attention or reducing sequence length."
                    )
        
        if not recommendations:
            recommendations.append("Model performance looks good. Consider profiling with larger inputs for more insights.")
        
        return recommendations
    
    def export_detailed_report(self, results: Dict[str, Any], output_file: str):
        """导出详细的profiling报告"""
        # 添加bottleneck分析
        results["bottleneck_analysis"] = self.get_bottleneck_analysis(results)
        
        # 添加timer数据
        results["timer_data"] = {
            "hierarchy_tree": self.timer.get_hierarchy_tree(),
            "timeline": [
                {
                    "name": t.name,
                    "start_time": t.start_time,
                    "duration_ms": t.duration_ms,
                    "parent": t.parent
                } for t in self.timer.get_timeline()
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def cleanup(self):
        """清理resources"""
        self.profiling_active = False
        self._clear_hooks()
        self.timer.clear()
        self.module_timings.clear()
        self.module_info.clear()
        
        if self.torch_profiler_results is not None:
            del self.torch_profiler_results
            self.torch_profiler_results = None

class LayerWiseProfiler:
    """
    Layer级别的精细profiling
    专门用于分析Transformer、CNN等复杂架构的每一层性能
    """
    
    def __init__(self):
        self.layer_timings = defaultdict(list)
        self.layer_info = {}
        self.hooks = []
        
    def profile_transformer_layers(self, transformer_model: nn.Module, 
                                 input_data: torch.Tensor, layer_name_pattern: str = "layers") -> Dict[str, Any]:
        """
        专门针对Transformer层的profiling
        
        Args:
            transformer_model: Transformer模型
            input_data: 输入数据
            layer_name_pattern: 层名称模式
            
        Returns:
            层级分析结果
        """
        results = {
            "layer_performance": {},
            "attention_analysis": {},
            "feedforward_analysis": {}
        }
        
        # 找到所有transformer层
        transformer_layers = []
        for name, module in transformer_model.named_modules():
            if layer_name_pattern in name and len(list(module.children())) > 0:
                transformer_layers.append((name, module))
        
        # 为每一层注册详细的hooks
        for layer_name, layer_module in transformer_layers:
            self._register_transformer_layer_hooks(layer_name, layer_module)
        
        # 运行推理
        with torch.no_grad():
            output = transformer_model(input_data)
        
        # 分析结果
        results["layer_performance"] = self._analyze_layer_performance()
        
        # 清理hooks
        self._cleanup_hooks()
        
        return results
    
    def _register_transformer_layer_hooks(self, layer_name: str, layer_module: nn.Module):
        """为transformer层注册详细hooks"""
        
        def layer_forward_hook(module, input, output):
            start_time = time.perf_counter()
            
            def end_hook(grad):
                end_time = time.perf_counter()
                duration = end_time - start_time
                self.layer_timings[layer_name].append(duration)
            
            if isinstance(output, torch.Tensor) and output.requires_grad:
                output.register_hook(end_hook)
        
        # 注册主层hook
        hook = layer_module.register_forward_hook(layer_forward_hook)
        self.hooks.append(hook)
        
        # 为子模块注册hooks（attention, feedforward等）
        for sub_name, sub_module in layer_module.named_children():
            full_name = f"{layer_name}.{sub_name}"
            
            def make_sub_hook(name):
                def sub_forward_hook(module, input, output):
                    # 这里可以添加更详细的子模块分析
                    pass
                return sub_forward_hook
            
            sub_hook = sub_module.register_forward_hook(make_sub_hook(full_name))
            self.hooks.append(sub_hook)
    
    def _analyze_layer_performance(self) -> Dict[str, Any]:
        """分析层级性能"""
        layer_stats = {}
        
        for layer_name, timings in self.layer_timings.items():
            if timings:
                layer_stats[layer_name] = {
                    "avg_time_ms": sum(timings) / len(timings) * 1000,
                    "total_time_ms": sum(timings) * 1000,
                    "call_count": len(timings),
                    "timings": [t * 1000 for t in timings]
                }
        
        return layer_stats
    
    def _cleanup_hooks(self):
        """清理所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()