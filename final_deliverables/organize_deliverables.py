"""
Script to organize final deliverables for language efficiency analysis.
"""

import os
import shutil
import json

def create_directory(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def copy_file(src, dst):
    """Copy file from source to destination."""
    try:
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")
    except FileNotFoundError:
        print(f"Warning: File not found: {src}")
    except Exception as e:
        print(f"Error copying {src}: {e}")

def main():
    """Organize final deliverables."""
    # Create base directories
    base_dir = "final_deliverables"
    create_directory(f"{base_dir}/enhanced_testing_framework")
    create_directory(f"{base_dir}/analysis_report")
    create_directory(f"{base_dir}/visualizations")
    create_directory(f"{base_dir}/raw_results")
    
    # Copy enhanced testing framework files
    framework_dir = f"{base_dir}/enhanced_testing_framework"
    create_directory(f"{framework_dir}/enhanced_benchmarks")
    create_directory(f"{framework_dir}/enhanced_benchmarks/MATH")
    create_directory(f"{framework_dir}/enhanced_benchmarks/BBH")
    create_directory(f"{framework_dir}/enhanced_benchmarks/HotpotQA")
    create_directory(f"{framework_dir}/enhanced_benchmarks/ARC")
    create_directory(f"{framework_dir}/enhanced_benchmarks/GSM8K")
    create_directory(f"{framework_dir}/analysis_methods")
    
    # Copy benchmark files
    copy_file("enhanced_benchmarks/__init__.py", f"{framework_dir}/enhanced_benchmarks/")
    copy_file("enhanced_benchmarks/MATH/math_problems.py", f"{framework_dir}/enhanced_benchmarks/MATH/")
    copy_file("enhanced_benchmarks/BBH/bbh_problems.py", f"{framework_dir}/enhanced_benchmarks/BBH/")
    copy_file("enhanced_benchmarks/HotpotQA/hotpotqa_problems.py", f"{framework_dir}/enhanced_benchmarks/HotpotQA/")
    copy_file("enhanced_benchmarks/ARC/arc_problems.py", f"{framework_dir}/enhanced_benchmarks/ARC/")
    copy_file("enhanced_benchmarks/GSM8K/gsm8k_problems.py", f"{framework_dir}/enhanced_benchmarks/GSM8K/")
    
    # Copy analysis methods
    copy_file("analysis_methods/__init__.py", f"{framework_dir}/analysis_methods/")
    copy_file("analysis_methods/cross_validation.py", f"{framework_dir}/analysis_methods/")
    copy_file("analysis_methods/information_density.py", f"{framework_dir}/analysis_methods/")
    copy_file("analysis_methods/compression_metrics.py", f"{framework_dir}/analysis_methods/")
    
    # Copy experiment runner
    copy_file("enhanced_experiment_runner.py", f"{framework_dir}/")
    
    # Copy analysis report
    report_dir = f"{base_dir}/analysis_report"
    copy_file("reports/final_report/language_efficiency_report.md", f"{report_dir}/")
    copy_file("reports/final_report/summary.md", f"{report_dir}/")
    copy_file("reports/final_report/presentation.md", f"{report_dir}/")
    
    # Copy visualizations
    viz_dir = f"{base_dir}/visualizations"
    create_directory(f"{viz_dir}/baseline")
    create_directory(f"{viz_dir}/detailed")
    create_directory(f"{viz_dir}/comprehensive")
    
    # Copy baseline visualizations
    for viz in ["token_usage_by_prompt.png", "token_usage_by_benchmark.png", "efficiency_by_benchmark.png"]:
        copy_file(f"reports/visualizations/{viz}", f"{viz_dir}/baseline/")
    
    # Copy detailed visualizations
    for viz in ["token_usage_by_benchmark_detailed.png", "information_density_detailed.png", 
                "chinese_character_metrics.png", "efficiency_by_difficulty.png", 
                "compression_ratio_by_benchmark.png"]:
        copy_file(f"reports/visualizations/detailed/{viz}", f"{viz_dir}/detailed/")
    
    # Copy comprehensive visualizations
    for viz in ["efficiency_heatmap.png", "token_distribution.png", 
                "information_density_comparison.png", "information_density_percent_diff.png", 
                "statistical_significance.png", "benchmark_radar_chart.png"]:
        copy_file(f"reports/visualizations/comprehensive/{viz}", f"{viz_dir}/comprehensive/")
    
    # Copy raw results
    results_dir = f"{base_dir}/raw_results"
    copy_file("experiment_results/baseline_results.json", f"{results_dir}/")
    copy_file("experiment_results/baseline_results.csv", f"{results_dir}/")
    copy_file("reports/baseline_analysis.json", f"{results_dir}/")
    copy_file("reports/detailed_analysis/baseline_detailed_analysis.json", f"{results_dir}/")
    copy_file("reports/visualizations/comprehensive/visualization_data.json", f"{results_dir}/")
    
    print("\nDeliverables organized successfully!")
    print(f"All files are available in the '{base_dir}' directory.")

if __name__ == "__main__":
    main()
