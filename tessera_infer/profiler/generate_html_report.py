#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate HTML reports from existing profiling data

This script allows you to generate comprehensive HTML reports
from JSON metrics files that were created during fast profiling.
"""

import os
import sys
import json
import glob
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

def find_metrics_files(profile_dir: str = "logs/profile") -> List[str]:
    """Find all metrics JSON files in the profile directory"""
    pattern = os.path.join(profile_dir, "**", "*_metrics*.json")
    files = glob.glob(pattern, recursive=True)
    return sorted(files)

def generate_html_from_json(profile_dir: str = "logs/profile"):
    """Generate HTML report from existing JSON metrics"""
    try:
        from profiler import MetricsCollector, HTMLReportGenerator
        
        print("🔍 Searching for metrics files...")
        metrics_files = find_metrics_files(profile_dir)
        
        if not metrics_files:
            print("❌ No metrics files found in", profile_dir)
            return None
        
        print(f"📁 Found {len(metrics_files)} metrics files")
        
        # Load all metrics
        collector = MetricsCollector(profile_dir)
        for filepath in metrics_files:
            print(f"   Loading: {os.path.basename(filepath)}")
            collector.load_metrics_from_file(filepath)
        
        # Generate summary report
        print("📊 Generating summary report...")
        summary = collector.generate_summary_report()
        
        # Generate comparison if we have both model types
        comparison = None
        pytorch_metrics = collector.get_latest_metrics_by_type('pytorch') 
        onnx_metrics = collector.get_latest_metrics_by_type('onnx')
        
        if pytorch_metrics or onnx_metrics:
            print("⚖️  Generating comparison report...")
            comparison = collector.compare_pytorch_vs_onnx()
        
        # Create report data
        report_data = {
            "summary": summary,
            "comparison": comparison,
            "pytorch_traces": [],
            "onnx_traces": []
        }
        
        # Generate HTML report
        print("📝 Generating HTML report...")
        reports_dir = os.path.join(profile_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        generator = HTMLReportGenerator(reports_dir)
        report_file = generator.generate_comprehensive_report(
            report_data,
            "tessera_profiling_report"
        )
        
        print(f"✅ HTML report generated: {report_file}")
        print(f"   Open in browser: file://{os.path.abspath(report_file)}")
        
        return report_file
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're running from the project root directory")
        return None
    except Exception as e:
        print(f"❌ Error generating HTML report: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    print("📊 Tessera HTML Report Generator")
    print("=" * 40)
    
    # Check if profile directory exists
    profile_dir = "logs/profile"
    if len(sys.argv) > 1:
        profile_dir = sys.argv[1]
    
    if not os.path.exists(profile_dir):
        print(f"❌ Profile directory not found: {profile_dir}")
        print("   Run profiling first with --enable_profiling")
        return 1
    
    # Generate HTML report
    report_file = generate_html_from_json(profile_dir)
    
    if report_file:
        print("\\n🎉 Success! You can now:")
        print(f"   1. Open the HTML report in your browser")
        print(f"   2. Share the report file: {os.path.basename(report_file)}")
        print(f"   3. Use the metrics for further analysis")
        return 0
    else:
        print("\\n❌ Failed to generate HTML report")
        return 1

if __name__ == "__main__":
    sys.exit(main())