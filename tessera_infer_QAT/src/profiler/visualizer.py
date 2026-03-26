#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import asdict

from .metrics import ComparisonMetrics
from .utils import format_time, format_memory, generate_timestamp

class HTMLReportGenerator:
    """Generate comprehensive HTML reports for profiling results"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_comprehensive_report(self, report_data: Dict[str, Any], 
                                    report_name: str = "profiling_report") -> str:
        """Generate a comprehensive HTML report"""
        timestamp = generate_timestamp()
        filename = f"{report_name}_{timestamp}.html"
        filepath = os.path.join(self.output_dir, filename)
        
        html_content = self._build_html_report(report_data, report_name)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filepath
    
    def _build_html_report(self, data: Dict[str, Any], title: str) -> str:
        """Build complete HTML report"""
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Tessera模型性能分析报告</title>
    <style>
        {self._get_css_styles()}
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="container">
        {self._generate_header(title)}
        {self._generate_summary_section(data.get('summary', {}))}
        {self._generate_comparison_section(data.get('comparison'))}
        {self._generate_detailed_metrics_section(data)}
        {self._generate_traces_section(data)}
        {self._generate_recommendations_section(data)}
        {self._generate_footer()}
    </div>
    
    <script>
        {self._get_javascript()}
    </script>
</body>
</html>
"""
        return html
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for the report"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        .header .subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .section {
            background: white;
            margin-bottom: 30px;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .section-title {
            font-size: 1.8rem;
            color: #4a5568;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: #f8f9fa;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }
        
        .metric-label {
            color: #64748b;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        .comparison-table th,
        .comparison-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .comparison-table th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #4a5568;
        }
        
        .comparison-table tr:hover {
            background-color: #f8f9fa;
        }
        
        .better {
            color: #48bb78;
            font-weight: 600;
        }
        
        .worse {
            color: #f56565;
            font-weight: 600;
        }
        
        .neutral {
            color: #64748b;
        }
        
        .recommendations {
            background: #f0fff4;
            border-left: 4px solid #48bb78;
            padding: 20px;
            margin: 20px 0;
        }
        
        .recommendations h4 {
            color: #2f855a;
            margin-bottom: 15px;
        }
        
        .recommendations ul {
            list-style-type: none;
        }
        
        .recommendations li {
            padding: 8px 0;
            padding-left: 20px;
            position: relative;
        }
        
        .recommendations li:before {
            content: "✓";
            position: absolute;
            left: 0;
            color: #48bb78;
            font-weight: bold;
        }
        
        .chart-container {
            margin: 20px 0;
            height: 400px;
        }
        
        .trace-links {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .trace-link {
            display: block;
            padding: 15px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            text-align: center;
            transition: background-color 0.2s ease;
        }
        
        .trace-link:hover {
            background: #5a67d8;
            color: white;
            text-decoration: none;
        }
        
        .footer {
            text-align: center;
            color: #64748b;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
        }
        
        .toggle-section {
            cursor: pointer;
            user-select: none;
        }
        
        .toggle-section:before {
            content: "▼";
            display: inline-block;
            margin-right: 10px;
            transition: transform 0.2s ease;
        }
        
        .toggle-section.collapsed:before {
            transform: rotate(-90deg);
        }
        
        .collapsible-content {
            overflow: hidden;
            transition: max-height 0.3s ease;
        }
        
        .collapsible-content.collapsed {
            max-height: 0;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .metrics-grid {
                grid-template-columns: 1fr;
            }
            
            .comparison-table {
                font-size: 0.9rem;
            }
        }
        """
    
    def _generate_header(self, title: str) -> str:
        """Generate HTML header section"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""
        <div class="header">
            <h1>{title}</h1>
            <div class="subtitle">
                Tessera模型推理性能分析报告<br>
                生成时间: {current_time}
            </div>
        </div>
        """
    
    def _generate_summary_section(self, summary: Dict[str, Any]) -> str:
        """Generate summary metrics section"""
        if not summary:
            return ""
        
        collection_info = summary.get('collection_info', {})
        averages = summary.get('averages', {})
        comparisons = summary.get('comparisons', {})
        
        html = """
        <div class="section">
            <h2 class="section-title">总览</h2>
            <div class="metrics-grid">
        """
        
        # Basic info metrics
        if collection_info:
            html += f"""
                <div class="metric-card">
                    <div class="metric-value">{collection_info.get('total_runs', 0)}</div>
                    <div class="metric-label">总测试次数</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(collection_info.get('model_types', []))}</div>
                    <div class="metric-label">模型类型</div>
                </div>
            """
        
        # Performance metrics
        if averages:
            throughput = averages.get('throughput', 0)
            if throughput > 0:
                html += f"""
                    <div class="metric-card">
                        <div class="metric-value">{throughput:.1f}</div>
                        <div class="metric-label">平均吞吐量 (samples/sec)</div>
                    </div>
                """
            
            inference_time = averages.get('total_inference_time', 0)
            if inference_time > 0:
                html += f"""
                    <div class="metric-card">
                        <div class="metric-value">{format_time(inference_time)}</div>
                        <div class="metric-label">平均推理时间</div>
                    </div>
                """
            
            memory_usage = averages.get('peak_memory_usage', 0)
            if memory_usage > 0:
                html += f"""
                    <div class="metric-card">
                        <div class="metric-value">{format_memory(memory_usage)}</div>
                        <div class="metric-label">平均峰值内存</div>
                    </div>
                """
        
        html += """
            </div>
        """
        
        # Model type breakdown
        if comparisons:
            html += """
            <h3>不同模型类型性能对比</h3>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>模型类型</th>
                        <th>平均吞吐量</th>
                        <th>平均推理时间</th>
                        <th>平均内存使用</th>
                        <th>测试次数</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for model_type, metrics in comparisons.items():
                throughput_info = metrics.get('throughput', {})
                inference_info = metrics.get('total_inference_time', {})
                memory_info = metrics.get('peak_memory_usage', {})
                
                html += f"""
                    <tr>
                        <td><strong>{model_type.upper()}</strong></td>
                        <td>{throughput_info.get('avg', 0):.1f} samples/sec</td>
                        <td>{format_time(inference_info.get('avg', 0))}</td>
                        <td>{format_memory(memory_info.get('avg', 0))}</td>
                        <td>{throughput_info.get('count', 0)}</td>
                    </tr>
                """
            
            html += """
                </tbody>
            </table>
            """
        
        html += "</div>"
        return html
    
    def _generate_comparison_section(self, comparison: Optional[ComparisonMetrics]) -> str:
        """Generate comparison section"""
        if not comparison or not comparison.performance_ratio:
            return ""
        
        ratios = comparison.performance_ratio
        
        html = """
        <div class="section">
            <h2 class="section-title">PyTorch vs ONNX 性能对比</h2>
            <div class="metrics-grid">
        """
        
        # Performance ratio cards
        if 'throughput_ratio' in ratios:
            ratio = ratios['throughput_ratio']
            status_class = "better" if ratio > 1.1 else "worse" if ratio < 0.9 else "neutral"
            html += f"""
                <div class="metric-card">
                    <div class="metric-value {status_class}">{ratio:.2f}x</div>
                    <div class="metric-label">ONNX吞吐量比率</div>
                </div>
            """
        
        if 'inference_time_ratio' in ratios:
            ratio = ratios['inference_time_ratio']
            status_class = "better" if ratio < 0.9 else "worse" if ratio > 1.1 else "neutral"
            html += f"""
                <div class="metric-card">
                    <div class="metric-value {status_class}">{ratio:.2f}x</div>
                    <div class="metric-label">ONNX推理时间比率</div>
                </div>
            """
        
        if 'memory_ratio' in ratios:
            ratio = ratios['memory_ratio']
            status_class = "better" if ratio < 0.9 else "worse" if ratio > 1.1 else "neutral"
            html += f"""
                <div class="metric-card">
                    <div class="metric-value {status_class}">{ratio:.2f}x</div>
                    <div class="metric-label">ONNX内存使用比率</div>
                </div>
            """
        
        html += """
            </div>
            <div style="margin-top: 20px;">
                <p><strong>说明:</strong></p>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li>吞吐量比率 > 1.0 表示 ONNX 模型更快</li>
                    <li>推理时间比率 < 1.0 表示 ONNX 模型用时更短</li>
                    <li>内存比率 < 1.0 表示 ONNX 模型内存使用更少</li>
                </ul>
            </div>
        </div>
        """
        
        return html
    
    def _generate_detailed_metrics_section(self, data: Dict[str, Any]) -> str:
        """Generate detailed metrics section"""
        summary = data.get('summary', {})
        comparisons = summary.get('comparisons', {})
        
        if not comparisons:
            return ""
        
        html = """
        <div class="section">
            <h2 class="section-title toggle-section">详细性能指标</h2>
            <div class="collapsible-content">
        """
        
        for model_type, metrics in comparisons.items():
            html += f"""
                <h3>{model_type.upper()} 模型详细指标</h3>
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>指标</th>
                            <th>平均值</th>
                            <th>最小值</th>
                            <th>最大值</th>
                            <th>样本数</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            metric_labels = {
                'model_load_time': '模型加载时间',
                'total_inference_time': '总推理时间',
                'avg_batch_time': '平均批次时间',
                'throughput': '吞吐量 (samples/sec)',
                'peak_memory_usage': '峰值内存使用',
                'avg_cpu_usage': 'CPU使用率 (%)'
            }
            
            for metric_key, metric_data in metrics.items():
                if metric_key in metric_labels and isinstance(metric_data, dict):
                    label = metric_labels[metric_key]
                    avg_val = metric_data.get('avg', 0)
                    min_val = metric_data.get('min', 0)
                    max_val = metric_data.get('max', 0)
                    count = metric_data.get('count', 0)
                    
                    # Format values based on metric type
                    if 'time' in metric_key:
                        avg_str = format_time(avg_val)
                        min_str = format_time(min_val)
                        max_str = format_time(max_val)
                    elif 'memory' in metric_key:
                        avg_str = format_memory(avg_val)
                        min_str = format_memory(min_val)
                        max_str = format_memory(max_val)
                    else:
                        avg_str = f"{avg_val:.2f}"
                        min_str = f"{min_val:.2f}"
                        max_str = f"{max_val:.2f}"
                    
                    html += f"""
                        <tr>
                            <td>{label}</td>
                            <td>{avg_str}</td>
                            <td>{min_str}</td>
                            <td>{max_str}</td>
                            <td>{count}</td>
                        </tr>
                    """
            
            html += """
                    </tbody>
                </table>
            """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _generate_traces_section(self, data: Dict[str, Any]) -> str:
        """Generate trace files section"""
        pytorch_traces = data.get('pytorch_traces', [])
        onnx_traces = data.get('onnx_traces', [])
        
        if not pytorch_traces and not onnx_traces:
            return ""
        
        html = """
        <div class="section">
            <h2 class="section-title">Trace文件</h2>
            <p>以下trace文件可在Chrome浏览器中打开 (chrome://tracing) 或使用Perfetto UI查看:</p>
            <div class="trace-links">
        """
        
        for trace_file in pytorch_traces:
            if os.path.exists(trace_file):
                filename = os.path.basename(trace_file)
                html += f"""
                    <a href="file://{trace_file}" class="trace-link">
                        PyTorch Trace<br>
                        <small>{filename}</small>
                    </a>
                """
        
        for trace_file in onnx_traces:
            if os.path.exists(trace_file):
                filename = os.path.basename(trace_file)
                html += f"""
                    <a href="file://{trace_file}" class="trace-link">
                        ONNX Trace<br>
                        <small>{filename}</small>
                    </a>
                """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _generate_recommendations_section(self, data: Dict[str, Any]) -> str:
        """Generate recommendations section"""
        comparison = data.get('comparison')
        
        if not comparison or not hasattr(comparison, 'recommendations'):
            return ""
        
        recommendations = comparison.recommendations if hasattr(comparison, 'recommendations') else []
        
        if not recommendations:
            return ""
        
        html = """
        <div class="section">
            <h2 class="section-title">优化建议</h2>
            <div class="recommendations">
                <h4>基于profiling结果的性能优化建议:</h4>
                <ul>
        """
        
        for recommendation in recommendations:
            html += f"<li>{recommendation}</li>"
        
        html += """
                </ul>
            </div>
        </div>
        """
        
        return html
    
    def _generate_footer(self) -> str:
        """Generate HTML footer"""
        return """
        <div class="footer">
            <p>本报告由 Tessera Profiling System 自动生成</p>
            <p>如有问题，请检查模型配置和profiling设置</p>
        </div>
        """
    
    def _get_javascript(self) -> str:
        """Get JavaScript for interactive features"""
        return """
        // Toggle collapsible sections
        document.addEventListener('DOMContentLoaded', function() {
            const toggleSections = document.querySelectorAll('.toggle-section');
            
            toggleSections.forEach(function(section) {
                section.addEventListener('click', function() {
                    const content = section.nextElementSibling;
                    
                    if (content.classList.contains('collapsed')) {
                        content.classList.remove('collapsed');
                        section.classList.remove('collapsed');
                        content.style.maxHeight = content.scrollHeight + 'px';
                    } else {
                        content.classList.add('collapsed');
                        section.classList.add('collapsed');
                        content.style.maxHeight = '0px';
                    }
                });
            });
            
            // Add smooth scrolling to internal links
            const links = document.querySelectorAll('a[href^="#"]');
            links.forEach(function(link) {
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    const target = document.querySelector(this.getAttribute('href'));
                    if (target) {
                        target.scrollIntoView({
                            behavior: 'smooth',
                            block: 'start'
                        });
                    }
                });
            });
        });
        """