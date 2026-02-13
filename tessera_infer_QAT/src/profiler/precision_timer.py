#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import threading
from typing import Dict, Optional, List, Any
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import defaultdict
import json

@dataclass
class TimingResult:
    """存储单次计时结果"""
    name: str
    start_time: float
    end_time: float
    duration: float
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        """返回毫秒为单位的时长"""
        return self.duration * 1000
    
    @property
    def duration_us(self) -> float:
        """返回微秒为单位的时长"""
        return self.duration * 1000000

@dataclass
class AggregatedTiming:
    """聚合计时统计"""
    name: str
    total_duration: float = 0.0
    count: int = 0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    
    @property
    def avg_duration(self) -> float:
        """平均时长"""
        return self.total_duration / self.count if self.count > 0 else 0.0
    
    @property
    def total_duration_ms(self) -> float:
        """总时长(毫秒)"""
        return self.total_duration * 1000
    
    @property
    def avg_duration_ms(self) -> float:
        """平均时长(毫秒)"""
        return self.avg_duration * 1000

class PrecisionTimer:
    """
    高精度嵌套计时器
    支持线程安全的嵌套计时、自动层级关系管理和详细统计分析
    """
    
    def __init__(self, name: str = "PrecisionTimer"):
        self.name = name
        self._lock = threading.Lock()
        self._timings: List[TimingResult] = []
        self._active_timers: Dict[str, float] = {}  # 活跃计时器
        self._timer_stack: List[str] = []  # 计时器栈，用于嵌套管理
        self._thread_local = threading.local()
        
    def _get_thread_stack(self) -> List[str]:
        """获取当前线程的计时器栈"""
        if not hasattr(self._thread_local, 'stack'):
            self._thread_local.stack = []
        return self._thread_local.stack
    
    def start_timer(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        """
        启动一个计时器
        
        Args:
            name: 计时器名称
            metadata: 附加元数据
            
        Returns:
            启动时间戳
        """
        start_time = time.perf_counter()
        
        with self._lock:
            if name in self._active_timers:
                # 如果已有同名计时器，自动添加序号
                counter = 1
                while f"{name}_{counter}" in self._active_timers:
                    counter += 1
                name = f"{name}_{counter}"
            
            self._active_timers[name] = start_time
            
            # 管理嵌套关系
            stack = self._get_thread_stack()
            parent = stack[-1] if stack else None
            stack.append(name)
            
            # 创建初始计时结果
            timing = TimingResult(
                name=name,
                start_time=start_time,
                end_time=0.0,
                duration=0.0,
                parent=parent,
                metadata=metadata or {}
            )
            
            # 如果有父级，添加到父级的children中
            if parent:
                for t in reversed(self._timings):
                    if t.name == parent and t.end_time == 0.0:
                        t.children.append(name)
                        break
            
            self._timings.append(timing)
        
        return start_time
    
    def end_timer(self, name: str) -> Optional[TimingResult]:
        """
        结束一个计时器
        
        Args:
            name: 计时器名称
            
        Returns:
            计时结果，如果计时器不存在则返回None
        """
        end_time = time.perf_counter()
        
        with self._lock:
            if name not in self._active_timers:
                return None
            
            start_time = self._active_timers.pop(name)
            duration = end_time - start_time
            
            # 从栈中移除
            stack = self._get_thread_stack()
            if name in stack:
                stack.remove(name)
            
            # 更新计时结果
            for timing in reversed(self._timings):
                if timing.name == name and timing.end_time == 0.0:
                    timing.end_time = end_time
                    timing.duration = duration
                    return timing
            
            return None
    
    @contextmanager
    def timer(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        计时器上下文管理器
        
        Args:
            name: 计时器名称
            metadata: 附加元数据
            
        Usage:
            with timer.timer("operation_name"):
                # 需要计时的代码
                pass
        """
        self.start_timer(name, metadata)
        try:
            yield
        finally:
            self.end_timer(name)
    
    def get_aggregated_stats(self) -> Dict[str, AggregatedTiming]:
        """
        获取聚合统计信息
        
        Returns:
            按名称分组的聚合计时统计
        """
        with self._lock:
            stats = defaultdict(lambda: AggregatedTiming(""))
            
            for timing in self._timings:
                if timing.end_time == 0.0:  # 跳过未完成的计时
                    continue
                
                stat = stats[timing.name]
                stat.name = timing.name
                stat.total_duration += timing.duration
                stat.count += 1
                stat.min_duration = min(stat.min_duration, timing.duration)
                stat.max_duration = max(stat.max_duration, timing.duration)
                
                if timing.parent and not stat.parent:
                    stat.parent = timing.parent
                
                for child in timing.children:
                    if child not in stat.children:
                        stat.children.append(child)
            
            return dict(stats)
    
    def get_timeline(self) -> List[TimingResult]:
        """
        获取按时间顺序排列的完整时间线
        
        Returns:
            完成的计时结果列表
        """
        with self._lock:
            completed = [t for t in self._timings if t.end_time > 0.0]
            return sorted(completed, key=lambda x: x.start_time)
    
    def get_hierarchy_tree(self) -> Dict[str, Any]:
        """
        构建层级树结构
        
        Returns:
            嵌套的层级结构字典
        """
        completed_timings = self.get_timeline()
        tree = {}
        
        # 构建名称到计时的映射
        timing_map = {t.name: t for t in completed_timings}
        
        def build_node(timing: TimingResult) -> Dict[str, Any]:
            node = {
                "name": timing.name,
                "duration": timing.duration,
                "duration_ms": timing.duration_ms,
                "start_time": timing.start_time,
                "end_time": timing.end_time,
                "metadata": timing.metadata,
                "children": []
            }
            
            for child_name in timing.children:
                if child_name in timing_map:
                    child_timing = timing_map[child_name]
                    node["children"].append(build_node(child_timing))
            
            return node
        
        # 找到根节点（没有父级的节点）
        for timing in completed_timings:
            if timing.parent is None:
                tree[timing.name] = build_node(timing)
        
        return tree
    
    def calculate_percentages(self, total_time: Optional[float] = None) -> Dict[str, float]:
        """
        计算各个操作占总时间的百分比
        
        Args:
            total_time: 总时间，如果为None则使用所有根节点时间之和
            
        Returns:
            操作名称到百分比的映射
        """
        stats = self.get_aggregated_stats()
        
        if total_time is None:
            # 计算根节点的总时间
            completed_timings = self.get_timeline()
            root_timings = [t for t in completed_timings if t.parent is None]
            total_time = sum(t.duration for t in root_timings)
        
        percentages = {}
        for name, stat in stats.items():
            if total_time > 0:
                percentages[name] = (stat.total_duration / total_time) * 100
            else:
                percentages[name] = 0.0
        
        return percentages
    
    def export_to_json(self, filepath: str):
        """
        导出计时数据到JSON文件
        
        Args:
            filepath: 输出文件路径
        """
        export_data = {
            "timer_name": self.name,
            "total_timings": len(self._timings),
            "timeline": [],
            "aggregated_stats": {},
            "hierarchy_tree": self.get_hierarchy_tree(),
            "percentages": self.calculate_percentages()
        }
        
        # 导出时间线
        for timing in self.get_timeline():
            export_data["timeline"].append({
                "name": timing.name,
                "start_time": timing.start_time,
                "end_time": timing.end_time,
                "duration": timing.duration,
                "duration_ms": timing.duration_ms,
                "parent": timing.parent,
                "children": timing.children,
                "metadata": timing.metadata
            })
        
        # 导出聚合统计
        for name, stat in self.get_aggregated_stats().items():
            export_data["aggregated_stats"][name] = {
                "total_duration": stat.total_duration,
                "total_duration_ms": stat.total_duration_ms,
                "count": stat.count,
                "avg_duration": stat.avg_duration,
                "avg_duration_ms": stat.avg_duration_ms,
                "min_duration": stat.min_duration,
                "max_duration": stat.max_duration
            }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def clear(self):
        """清除所有计时数据"""
        with self._lock:
            self._timings.clear()
            self._active_timers.clear()
            self._get_thread_stack().clear()
    
    def print_summary(self, show_children: bool = True, min_duration_ms: float = 0.1):
        """
        打印计时摘要
        
        Args:
            show_children: 是否显示子项
            min_duration_ms: 最小显示时长(毫秒)
        """
        stats = self.get_aggregated_stats()
        percentages = self.calculate_percentages()
        
        print(f"\n=== {self.name} Timing Summary ===")
        print(f"Total operations: {len(stats)}")
        
        # 按总时长排序
        sorted_stats = sorted(stats.items(), key=lambda x: x[1].total_duration, reverse=True)
        
        for name, stat in sorted_stats:
            if stat.total_duration_ms < min_duration_ms:
                continue
                
            percentage = percentages.get(name, 0.0)
            print(f"{name:40} {stat.total_duration_ms:8.2f}ms ({percentage:5.1f}%) [{stat.count:3d} calls]")
            
            if show_children and stat.children:
                for child in stat.children:
                    if child in stats:
                        child_stat = stats[child]
                        child_pct = percentages.get(child, 0.0)
                        if child_stat.total_duration_ms >= min_duration_ms:
                            print(f"  ├── {child:35} {child_stat.total_duration_ms:8.2f}ms ({child_pct:5.1f}%)")

class GlobalTimerManager:
    """
    全局计时器管理器
    用于管理多个计时器实例和跨模块的计时
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._timers: Dict[str, PrecisionTimer] = {}
            self._default_timer = PrecisionTimer("default")
            self._timers["default"] = self._default_timer
            self._initialized = True
    
    def get_timer(self, name: str = "default") -> PrecisionTimer:
        """获取指定名称的计时器"""
        if name not in self._timers:
            self._timers[name] = PrecisionTimer(name)
        return self._timers[name]
    
    def create_timer(self, name: str) -> PrecisionTimer:
        """创建新的计时器"""
        timer = PrecisionTimer(name)
        self._timers[name] = timer
        return timer
    
    @contextmanager
    def timer(self, operation_name: str, timer_name: str = "default", metadata: Optional[Dict[str, Any]] = None):
        """便捷的计时器上下文管理器"""
        timer = self.get_timer(timer_name)
        with timer.timer(operation_name, metadata):
            yield
    
    def export_all_timers(self, output_dir: str):
        """导出所有计时器数据"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, timer in self._timers.items():
            filepath = os.path.join(output_dir, f"{name}_timing.json")
            timer.export_to_json(filepath)
    
    def clear_all(self):
        """清除所有计时器数据"""
        for timer in self._timers.values():
            timer.clear()

# 全局计时器实例
timer_manager = GlobalTimerManager()

# 便捷函数
def get_timer(name: str = "default") -> PrecisionTimer:
    """获取计时器实例"""
    return timer_manager.get_timer(name)

def time_operation(operation_name: str, timer_name: str = "default", metadata: Optional[Dict[str, Any]] = None):
    """操作计时装饰器/上下文管理器"""
    return timer_manager.timer(operation_name, timer_name, metadata)