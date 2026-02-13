#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import json
import numpy as np
from typing import Any, Dict, Optional, List, Tuple
from torch.utils.data import DataLoader
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
import threading

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import onnx
    ONNX_MODEL_AVAILABLE = True
except ImportError:
    ONNX_MODEL_AVAILABLE = False

from .precision_timer import PrecisionTimer, time_operation

@dataclass
class ONNXOperatorStats:
    """ONNX算子统计信息"""
    name: str
    op_type: str
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    count: int
    percentage: float
    input_shapes: List[List[int]]
    output_shapes: List[List[int]]
    attributes: Dict[str, Any]

@dataclass
class ONNXNodeInfo:
    """ONNX节点详细信息"""
    name: str
    op_type: str
    input_names: List[str]
    output_names: List[str]
    attributes: Dict[str, Any]
    input_shapes: List[List[int]]
    output_shapes: List[List[int]]

class ONNXProfilerV2:
    """
    增强的ONNX Runtime profiler
    提供operator级别的深度分析和性能优化建议
    """
    
    def __init__(self, output_dir: str, model_name: str = "onnx_model"):
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime is not available. Please install onnxruntime.")
        
        self.output_dir = output_dir
        self.model_name = model_name
        self.timer = PrecisionTimer(f"{model_name}_onnx_timer")
        
        # ONNX specific data
        self.session = None
        self.model_info = {}
        self.node_info_map: Dict[str, ONNXNodeInfo] = {}
        self.operator_timings: Dict[str, List[float]] = defaultdict(list)
        self.profiling_results = None
        
        # 性能数据
        self.batch_timings = []
        self.inference_timings = []
        self.data_prep_timings = []
        
        # 线程安全
        self._lock = threading.Lock()
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_and_analyze_model(self, model_path: str, num_threads: Optional[int] = None) -> Dict[str, Any]:
        """
        加载ONNX模型并进行静态分析
        
        Args:
            model_path: ONNX模型路径
            num_threads: 线程数
            
        Returns:
            模型分析结果
        """
        with self.timer.timer("model_loading"):
            # 配置session选项
            sess_options = ort.SessionOptions()
            sess_options.enable_profiling = True
            
            if num_threads is not None:
                sess_options.intra_op_num_threads = num_threads
                sess_options.inter_op_num_threads = 1
            
            # 优化设置
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_cpu_mem_arena = True
            sess_options.enable_mem_pattern = True
            sess_options.enable_mem_reuse = True
            
            # CPU执行提供者配置
            providers = [
                ('CPUExecutionProvider', {
                    'use_arena': True,
                    'arena_extend_strategy': 'kSameAsRequested',
                    'initial_chunk_size_bytes': 1024 * 1024,
                    'max_chunk_size_bytes': 32 * 1024 * 1024,
                })
            ]
            
            # 创建session
            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )
        
        # 分析模型结构
        model_analysis = self._analyze_model_structure(model_path)
        
        # 收集模型基本信息
        self.model_info = {
            "model_path": model_path,
            "input_names": [inp.name for inp in self.session.get_inputs()],
            "output_names": [out.name for out in self.session.get_outputs()],
            "num_threads": num_threads,
            "providers": self.session.get_providers()
        }
        
        # 收集输入输出详细信息
        inputs_info = []
        for inp in self.session.get_inputs():
            inputs_info.append({
                "name": inp.name,
                "type": str(inp.type),
                "shape": inp.shape
            })
        
        outputs_info = []
        for out in self.session.get_outputs():
            outputs_info.append({
                "name": out.name,
                "type": str(out.type),
                "shape": out.shape
            })
        
        self.model_info.update({
            "inputs": inputs_info,
            "outputs": outputs_info,
            "model_analysis": model_analysis
        })
        
        return self.model_info
    
    def _analyze_model_structure(self, model_path: str) -> Dict[str, Any]:
        """分析ONNX模型结构"""
        if not ONNX_MODEL_AVAILABLE:
            return {"error": "onnx package not available for model analysis"}
        
        try:
            # 加载ONNX模型
            model = onnx.load(model_path)
            
            # 分析节点
            nodes_by_op_type = defaultdict(int)
            total_nodes = len(model.graph.node)
            
            for node in model.graph.node:
                nodes_by_op_type[node.op_type] += 1
                
                # 存储节点信息
                node_info = ONNXNodeInfo(
                    name=node.name or f"{node.op_type}_{nodes_by_op_type[node.op_type]}",
                    op_type=node.op_type,
                    input_names=list(node.input),
                    output_names=list(node.output),
                    attributes={attr.name: self._extract_attribute_value(attr) for attr in node.attribute},
                    input_shapes=[],  # 需要运行时推断
                    output_shapes=[]  # 需要运行时推断
                )
                self.node_info_map[node_info.name] = node_info
            
            # 分析参数
            initializers = len(model.graph.initializer)
            total_params = sum(
                np.prod(tensor.dims) for tensor in model.graph.initializer 
                if hasattr(tensor, 'dims')
            )
            
            return {
                "total_nodes": total_nodes,
                "nodes_by_op_type": dict(nodes_by_op_type),
                "total_initializers": initializers,
                "estimated_parameters": int(total_params),
                "model_version": model.model_version,
                "ir_version": model.ir_version,
                "opset_imports": [
                    {"domain": opset.domain, "version": opset.version} 
                    for opset in model.opset_import
                ]
            }
        except Exception as e:
            return {"error": f"Failed to analyze model structure: {str(e)}"}
    
    def _extract_attribute_value(self, attr):
        """提取ONNX属性值"""
        try:
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
            elif attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
            elif attr.type == onnx.AttributeProto.STRING:
                return attr.s.decode('utf-8')
            elif attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOATS:
                return list(attr.floats)
            else:
                return str(attr)
        except:
            return None
    
    def profile_inference_detailed(self, dataloader: DataLoader, num_batches: Optional[int] = None) -> Dict[str, Any]:
        """
        详细的推理profiling
        
        Args:
            dataloader: 数据加载器
            num_batches: 要profile的batch数量
            
        Returns:
            详细的profiling结果
        """
        if self.session is None:
            raise ValueError("Model not loaded. Call load_and_analyze_model first.")
        
        if num_batches is None:
            num_batches = min(len(dataloader), 50)
        
        results = {
            "total_batches": num_batches,
            "batch_stats": [],
            "operator_analysis": {},
            "performance_summary": {},
            "bottleneck_analysis": {}
        }
        
        input_names = self.model_info["input_names"]
        output_names = self.model_info["output_names"]
        
        with self.timer.timer("total_inference_profiling"):
            # 预热
            with self.timer.timer("warmup"):
                warmup_batches = min(2, num_batches)
                for i, batch_data in enumerate(dataloader):
                    if i >= warmup_batches:
                        break
                    self._run_single_inference(batch_data, input_names, output_names)
            
            # 主要profiling
            with self.timer.timer("main_profiling"):
                for batch_idx, batch_data in enumerate(dataloader):
                    if batch_idx >= num_batches:
                        break
                    
                    batch_result = self._profile_single_batch(
                        batch_data, input_names, output_names, batch_idx
                    )
                    results["batch_stats"].append(batch_result)
        
        # 获取ONNX Runtime profiling结果
        trace_file = self.session.end_profiling()
        results["operator_analysis"] = self._analyze_onnx_profiling_trace(trace_file)
        
        # 分析整体性能
        results["performance_summary"] = self._generate_performance_summary()
        results["bottleneck_analysis"] = self._identify_bottlenecks(results)
        
        return results
    
    def _run_single_inference(self, batch_data: Any, input_names: List[str], output_names: List[str]) -> Any:
        """运行单次推理"""
        # 数据准备
        s2_input = batch_data.get("s2_input", batch_data.get("s2"))
        s1_input = batch_data.get("s1_input", batch_data.get("s1"))
        
        if s2_input is None or s1_input is None:
            raise ValueError("Required inputs not found in batch data")
        
        # 转换为numpy
        if hasattr(s2_input, 'numpy'):
            s2_np = s2_input.numpy()
            s1_np = s1_input.numpy()
        else:
            s2_np = s2_input
            s1_np = s1_input
        
        # 确保正确的数据类型和内存布局
        s2_np = np.ascontiguousarray(s2_np, dtype=np.float32)
        s1_np = np.ascontiguousarray(s1_np, dtype=np.float32)
        
        # 运行推理
        ort_inputs = {
            input_names[0]: s2_np,
            input_names[1]: s1_np
        }
        outputs = self.session.run(output_names, ort_inputs)
        
        return outputs
    
    def _profile_single_batch(self, batch_data: Any, input_names: List[str], 
                             output_names: List[str], batch_idx: int) -> Dict[str, Any]:
        """Profile单个batch的详细性能"""
        
        batch_start_time = time.perf_counter()
        
        # 数据准备阶段
        with self.timer.timer(f"batch_{batch_idx}_data_prep"):
            data_prep_start = time.perf_counter()
            
            s2_input = batch_data.get("s2_input", batch_data.get("s2"))
            s1_input = batch_data.get("s1_input", batch_data.get("s1"))
            
            if hasattr(s2_input, 'numpy'):
                s2_np = s2_input.numpy()
                s1_np = s1_input.numpy()
            else:
                s2_np = s2_input
                s1_np = s1_input
            
            s2_np = np.ascontiguousarray(s2_np, dtype=np.float32)
            s1_np = np.ascontiguousarray(s1_np, dtype=np.float32)
            
            batch_size = s2_np.shape[0]
            
            data_prep_time = time.perf_counter() - data_prep_start
        
        # 推理阶段
        with self.timer.timer(f"batch_{batch_idx}_inference"):
            inference_start = time.perf_counter()
            
            ort_inputs = {
                input_names[0]: s2_np,
                input_names[1]: s1_np
            }
            outputs = self.session.run(output_names, ort_inputs)
            
            inference_time = time.perf_counter() - inference_start
        
        batch_total_time = time.perf_counter() - batch_start_time
        
        # 记录统计信息
        self.batch_timings.append(batch_total_time)
        self.inference_timings.append(inference_time)
        self.data_prep_timings.append(data_prep_time)
        
        return {
            "batch_idx": batch_idx,
            "batch_size": batch_size,
            "total_time_ms": batch_total_time * 1000,
            "data_prep_time_ms": data_prep_time * 1000,
            "inference_time_ms": inference_time * 1000,
            "throughput_samples_per_sec": batch_size / batch_total_time,
            "input_shapes": {
                input_names[0]: list(s2_np.shape),
                input_names[1]: list(s1_np.shape)
            },
            "output_shapes": {
                output_names[i]: list(outputs[i].shape) for i in range(len(outputs))
            }
        }
    
    def _analyze_onnx_profiling_trace(self, trace_file: str) -> Dict[str, Any]:
        """分析ONNX Runtime profiling trace文件"""
        if not trace_file or not os.path.exists(trace_file):
            return {"error": "Trace file not available"}
        
        try:
            with open(trace_file, 'r') as f:
                trace_data = json.load(f)
            
            # 解析事件
            if isinstance(trace_data, list):
                events = trace_data
            elif isinstance(trace_data, dict) and 'events' in trace_data:
                events = trace_data['events']
            else:
                events = []
            
            # 按操作类型分组分析
            op_timings = defaultdict(list)
            op_details = defaultdict(lambda: {
                'count': 0,
                'total_time': 0,
                'min_time': float('inf'),
                'max_time': 0,
                'input_shapes': set(),
                'output_shapes': set()
            })
            
            for event in events:
                if not isinstance(event, dict):
                    continue
                
                op_name = event.get("name", event.get("op_name", "Unknown"))
                duration = event.get("dur", 0)  # 通常以微秒为单位
                
                if duration <= 0:
                    continue
                
                # 转换为毫秒
                duration_ms = duration / 1000.0
                
                op_timings[op_name].append(duration_ms)
                
                detail = op_details[op_name]
                detail['count'] += 1
                detail['total_time'] += duration_ms
                detail['min_time'] = min(detail['min_time'], duration_ms)
                detail['max_time'] = max(detail['max_time'], duration_ms)
                
                # 提取形状信息（如果可用）
                if 'input_shapes' in event:
                    detail['input_shapes'].add(str(event['input_shapes']))\n                if 'output_shapes' in event:
                    detail['output_shapes'].add(str(event['output_shapes']))
            
            # 计算统计信息
            operator_stats = []
            total_time = sum(detail['total_time'] for detail in op_details.values())
            
            for op_name, detail in op_details.items():
                if detail['min_time'] == float('inf'):
                    detail['min_time'] = 0
                
                avg_time = detail['total_time'] / detail['count'] if detail['count'] > 0 else 0
                percentage = (detail['total_time'] / total_time * 100) if total_time > 0 else 0
                
                operator_stats.append(ONNXOperatorStats(
                    name=op_name,
                    op_type=self._extract_op_type_from_name(op_name),
                    total_time_ms=detail['total_time'],
                    avg_time_ms=avg_time,
                    min_time_ms=detail['min_time'],
                    max_time_ms=detail['max_time'],
                    count=detail['count'],
                    percentage=percentage,
                    input_shapes=list(detail['input_shapes'])[:5],  # 限制数量
                    output_shapes=list(detail['output_shapes'])[:5],  # 限制数量
                    attributes={}
                ))
            
            # 按总时间排序
            operator_stats.sort(key=lambda x: x.total_time_ms, reverse=True)
            
            return {
                "total_operators": len(operator_stats),
                "total_time_ms": total_time,
                "operator_stats": [
                    {
                        "name": stat.name,
                        "op_type": stat.op_type,
                        "total_time_ms": stat.total_time_ms,
                        "avg_time_ms": stat.avg_time_ms,
                        "min_time_ms": stat.min_time_ms,
                        "max_time_ms": stat.max_time_ms,
                        "count": stat.count,
                        "percentage": stat.percentage,
                        "input_shapes": stat.input_shapes,
                        "output_shapes": stat.output_shapes
                    } for stat in operator_stats[:30]  # 前30个
                ],
                "trace_file": trace_file
            }
            
        except Exception as e:
            return {"error": f"Failed to analyze trace: {str(e)}"}
    
    def _extract_op_type_from_name(self, op_name: str) -> str:
        """从操作名称中提取操作类型"""
        # 常见的ONNX操作类型
        common_ops = [
            'Conv', 'MatMul', 'Add', 'Relu', 'BatchNormalization', 
            'MaxPool', 'AveragePool', 'Reshape', 'Transpose', 'Concat',
            'Softmax', 'Sigmoid', 'Tanh', 'Gemm', 'Mul', 'Sub', 'Div'
        ]
        
        op_name_upper = op_name.upper()
        for op in common_ops:
            if op.upper() in op_name_upper:
                return op
        
        # 如果没有找到匹配的，返回原始名称的第一部分
        parts = op_name.split('_')
        return parts[0] if parts else op_name
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """生成性能摘要"""
        summary = {}
        
        # 批次统计
        if self.batch_timings:
            summary["batch_performance"] = {
                "total_batches": len(self.batch_timings),
                "avg_batch_time_ms": np.mean(self.batch_timings) * 1000,
                "min_batch_time_ms": np.min(self.batch_timings) * 1000,
                "max_batch_time_ms": np.max(self.batch_timings) * 1000,
                "std_batch_time_ms": np.std(self.batch_timings) * 1000
            }
        
        # 推理统计
        if self.inference_timings:
            summary["inference_performance"] = {
                "avg_inference_time_ms": np.mean(self.inference_timings) * 1000,
                "min_inference_time_ms": np.min(self.inference_timings) * 1000,
                "max_inference_time_ms": np.max(self.inference_timings) * 1000,
                "std_inference_time_ms": np.std(self.inference_timings) * 1000
            }
        
        # 数据准备统计
        if self.data_prep_timings:
            summary["data_prep_performance"] = {
                "avg_data_prep_time_ms": np.mean(self.data_prep_timings) * 1000,
                "min_data_prep_time_ms": np.min(self.data_prep_timings) * 1000,
                "max_data_prep_time_ms": np.max(self.data_prep_timings) * 1000,
                "data_prep_percentage": (np.sum(self.data_prep_timings) / 
                                       np.sum(self.batch_timings) * 100) if self.batch_timings else 0
            }
        
        # Timer统计
        timer_stats = self.timer.get_aggregated_stats()
        timer_percentages = self.timer.calculate_percentages()
        
        summary["timer_breakdown"] = {}
        for name, stat in timer_stats.items():
            summary["timer_breakdown"][name] = {
                "total_time_ms": stat.total_duration_ms,
                "avg_time_ms": stat.avg_duration_ms,
                "count": stat.count,
                "percentage": timer_percentages.get(name, 0.0)
            }
        
        return summary
    
    def _identify_bottlenecks(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """识别性能瓶颈"""
        bottlenecks = {
            "top_operators": [],
            "slow_batches": [],
            "recommendations": []
        }
        
        # 分析操作瓶颈
        operator_analysis = results.get("operator_analysis", {})
        if "operator_stats" in operator_analysis:
            # 前5个最耗时的操作
            top_ops = operator_analysis["operator_stats"][:5]
            for op in top_ops:
                bottlenecks["top_operators"].append({
                    "name": op["name"],
                    "op_type": op["op_type"],
                    "total_time_ms": op["total_time_ms"],
                    "percentage": op["percentage"],
                    "avg_time_ms": op["avg_time_ms"],
                    "count": op["count"]
                })
        
        # 分析慢批次
        batch_stats = results.get("batch_stats", [])
        if batch_stats:
            avg_batch_time = np.mean([b["total_time_ms"] for b in batch_stats])
            slow_batches = [
                b for b in batch_stats 
                if b["total_time_ms"] > avg_batch_time * 1.5
            ]
            bottlenecks["slow_batches"] = slow_batches[:5]  # 前5个最慢的批次
        
        # 生成优化建议
        bottlenecks["recommendations"] = self._generate_optimization_recommendations(results)
        
        return bottlenecks
    
    def _generate_optimization_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于operator分析的建议
        operator_analysis = results.get("operator_analysis", {})
        if "operator_stats" in operator_analysis and operator_analysis["operator_stats"]:
            top_op = operator_analysis["operator_stats"][0]
            
            if top_op["op_type"] in ["Conv", "Convolution"] and top_op["percentage"] > 30:
                recommendations.append(
                    f"Convolution operations consume {top_op['percentage']:.1f}% of total time. "
                    f"Consider using optimized convolution kernels or reducing kernel sizes."
                )
            elif top_op["op_type"] in ["MatMul", "Gemm"] and top_op["percentage"] > 25:
                recommendations.append(
                    f"Matrix multiplication operations consume {top_op['percentage']:.1f}% of total time. "
                    f"Consider using quantized models or optimized BLAS libraries."
                )
        
        # 基于性能摘要的建议
        performance_summary = results.get("performance_summary", {})
        if "data_prep_performance" in performance_summary:
            data_prep_pct = performance_summary["data_prep_performance"]["data_prep_percentage"]
            if data_prep_pct > 15:
                recommendations.append(
                    f"Data preparation takes {data_prep_pct:.1f}% of total time. "
                    f"Consider optimizing data loading and preprocessing."
                )
        
        # 基于batch统计的建议
        if "batch_performance" in performance_summary:
            batch_perf = performance_summary["batch_performance"]
            if batch_perf["std_batch_time_ms"] > batch_perf["avg_batch_time_ms"] * 0.3:
                recommendations.append(
                    f"High variance in batch processing times detected. "
                    f"Consider investigating data loading bottlenecks or input size variations."
                )
        
        if not recommendations:
            recommendations.append("Performance looks reasonable. Consider profiling with larger datasets for more insights.")
        
        return recommendations
    
    def export_comprehensive_report(self, results: Dict[str, Any], output_file: str):
        """导出综合性能报告"""
        # 添加模型信息和timer数据
        comprehensive_report = {
            "model_info": self.model_info,
            "profiling_results": results,
            "timer_hierarchy": self.timer.get_hierarchy_tree(),
            "timer_timeline": [
                {
                    "name": t.name,
                    "start_time": t.start_time,
                    "duration_ms": t.duration_ms,
                    "parent": t.parent
                } for t in self.timer.get_timeline()
            ],
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(output_file, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
    
    def cleanup(self):
        """清理资源"""
        self.timer.clear()
        self.batch_timings.clear()
        self.inference_timings.clear()
        self.data_prep_timings.clear()
        self.operator_timings.clear()
        
        if self.session is not None:
            # ONNX Runtime会自动清理session
            self.session = None