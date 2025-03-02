"""
Analyze Chinese tokenization differences between Anthropic and Deepseek models.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def load_all_results():
    """
    Load results from both Anthropic and Deepseek models.
    """
    # Load Anthropic results
    interim_files = glob.glob('experiment_results/interim_results_*.json')
    latest_file = max(interim_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    with open(latest_file, 'r') as f:
        anthropic_results = json.load(f)
    
    anthropic_df = pd.DataFrame(anthropic_results)
    anthropic_df = anthropic_df[anthropic_df['model'] == 'anthropic:claude-3-5-sonnet-20240620']
    
    # Load Deepseek results
    with open('experiment_results/deepseek_longcontext_results.json', 'r') as f:
        deepseek_results = json.load(f)
    
    deepseek_df = pd.DataFrame(deepseek_results)
    
    return anthropic_df, deepseek_df

def analyze_chinese_tokenization(anthropic_df, deepseek_df):
    """
    Analyze Chinese tokenization differences between models.
    """
    # Filter for Chinese prompts
    anthropic_chinese = anthropic_df[anthropic_df['prompt_type'] == 'chinese']
    deepseek_chinese = deepseek_df[deepseek_df['prompt_type'] == 'chinese']
    
    # Calculate tokenization metrics
    anthropic_metrics = {
        'input_tokens': anthropic_chinese['input_tokens'].mean(),
        'output_tokens': anthropic_chinese['output_tokens'].mean(),
        'total_tokens': anthropic_chinese['total_tokens'].mean(),
        'input_percentage': (anthropic_chinese['input_tokens'] / anthropic_chinese['total_tokens'] * 100).mean(),
        'output_percentage': (anthropic_chinese['output_tokens'] / anthropic_chinese['total_tokens'] * 100).mean()
    }
    
    deepseek_metrics = {
        'input_tokens': deepseek_chinese['input_tokens'].mean(),
        'output_tokens': deepseek_chinese['output_tokens'].mean(),
        'total_tokens': deepseek_chinese['total_tokens'].mean(),
        'input_percentage': (deepseek_chinese['input_tokens'] / deepseek_chinese['total_tokens'] * 100).mean(),
        'output_percentage': (deepseek_chinese['output_tokens'] / deepseek_chinese['total_tokens'] * 100).mean()
    }
    
    # Calculate differences
    differences = {
        'input_tokens': deepseek_metrics['input_tokens'] - anthropic_metrics['input_tokens'],
        'output_tokens': deepseek_metrics['output_tokens'] - anthropic_metrics['output_tokens'],
        'total_tokens': deepseek_metrics['total_tokens'] - anthropic_metrics['total_tokens'],
        'input_percentage': deepseek_metrics['input_percentage'] - anthropic_metrics['input_percentage'],
        'output_percentage': deepseek_metrics['output_percentage'] - anthropic_metrics['output_percentage']
    }
    
    # Check if input_chars column exists
    if 'input_chars' in anthropic_chinese.columns and 'input_chars' in deepseek_chinese.columns:
        anthropic_metrics['input_chars'] = anthropic_chinese['input_chars'].mean()
        deepseek_metrics['input_chars'] = deepseek_chinese['input_chars'].mean()
        
        anthropic_metrics['chars_per_token'] = anthropic_metrics['input_chars'] / anthropic_metrics['input_tokens']
        deepseek_metrics['chars_per_token'] = deepseek_metrics['input_chars'] / deepseek_metrics['input_tokens']
        
        differences['input_chars'] = deepseek_metrics['input_chars'] - anthropic_metrics['input_chars']
        differences['chars_per_token'] = deepseek_metrics['chars_per_token'] - anthropic_metrics['chars_per_token']
    
    return anthropic_metrics, deepseek_metrics, differences

def analyze_chinese_vs_english(anthropic_df, deepseek_df):
    """
    Analyze Chinese vs English efficiency in both models.
    """
    # Filter for Chinese and English prompts
    anthropic_chinese = anthropic_df[anthropic_df['prompt_type'] == 'chinese']
    anthropic_english = anthropic_df[anthropic_df['prompt_type'] == 'english']
    
    deepseek_chinese = deepseek_df[deepseek_df['prompt_type'] == 'chinese']
    deepseek_english = deepseek_df[deepseek_df['prompt_type'] == 'english']
    
    # Calculate efficiency metrics
    anthropic_chinese_tokens = anthropic_chinese['total_tokens'].mean()
    anthropic_english_tokens = anthropic_english['total_tokens'].mean()
    anthropic_efficiency = (anthropic_english_tokens - anthropic_chinese_tokens) / anthropic_english_tokens * 100
    
    deepseek_chinese_tokens = deepseek_chinese['total_tokens'].mean()
    deepseek_english_tokens = deepseek_english['total_tokens'].mean()
    deepseek_efficiency = (deepseek_english_tokens - deepseek_chinese_tokens) / deepseek_english_tokens * 100
    
    # Calculate token distribution
    anthropic_chinese_input = anthropic_chinese['input_tokens'].mean()
    anthropic_chinese_output = anthropic_chinese['output_tokens'].mean()
    
    anthropic_english_input = anthropic_english['input_tokens'].mean()
    anthropic_english_output = anthropic_english['output_tokens'].mean()
    
    deepseek_chinese_input = deepseek_chinese['input_tokens'].mean()
    deepseek_chinese_output = deepseek_chinese['output_tokens'].mean()
    
    deepseek_english_input = deepseek_english['input_tokens'].mean()
    deepseek_english_output = deepseek_english['output_tokens'].mean()
    
    # Compile results
    results = {
        'anthropic_efficiency': anthropic_efficiency,
        'deepseek_efficiency': deepseek_efficiency,
        'anthropic_chinese_input': anthropic_chinese_input,
        'anthropic_chinese_output': anthropic_chinese_output,
        'anthropic_english_input': anthropic_english_input,
        'anthropic_english_output': anthropic_english_output,
        'deepseek_chinese_input': deepseek_chinese_input,
        'deepseek_chinese_output': deepseek_chinese_output,
        'deepseek_english_input': deepseek_english_input,
        'deepseek_english_output': deepseek_english_output
    }
    
    return results

def create_visualizations(anthropic_metrics, deepseek_metrics, differences, comparison_results):
    """
    Create visualizations for Chinese tokenization analysis.
    """
    os.makedirs('reports/visualizations/chinese_tokenization', exist_ok=True)
    
    # 1. Token Usage Comparison
    plt.figure(figsize=(12, 8))
    
    metrics = ['Input Tokens', 'Output Tokens', 'Total Tokens']
    anthropic_values = [
        anthropic_metrics['input_tokens'],
        anthropic_metrics['output_tokens'],
        anthropic_metrics['total_tokens']
    ]
    deepseek_values = [
        deepseek_metrics['input_tokens'],
        deepseek_metrics['output_tokens'],
        deepseek_metrics['total_tokens']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, anthropic_values, width, label='Anthropic Claude', color='#3498db')
    rects2 = ax.bar(x + width/2, deepseek_values, width, label='Deepseek', color='#e74c3c')
    
    # Add labels and title
    ax.set_xlabel('Metric', fontsize=14, fontweight='bold')
    ax.set_ylabel('Tokens', fontsize=14, fontweight='bold')
    ax.set_title('Chinese Token Usage Comparison Between Models', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add value labels
    for i, v in enumerate(anthropic_values):
        ax.text(i - width/2, v + 5, f"{v:.1f}", ha='center', fontweight='bold')
    
    for i, v in enumerate(deepseek_values):
        ax.text(i + width/2, v + 5, f"{v:.1f}", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('reports/visualizations/chinese_tokenization/token_usage_comparison.png', dpi=300)
    plt.close()
    
    # 2. Token Percentage Comparison
    plt.figure(figsize=(12, 8))
    
    metrics = ['Input Percentage', 'Output Percentage']
    anthropic_values = [
        anthropic_metrics['input_percentage'],
        anthropic_metrics['output_percentage']
    ]
    deepseek_values = [
        deepseek_metrics['input_percentage'],
        deepseek_metrics['output_percentage']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, anthropic_values, width, label='Anthropic Claude', color='#3498db')
    rects2 = ax.bar(x + width/2, deepseek_values, width, label='Deepseek', color='#e74c3c')
    
    # Add labels and title
    ax.set_xlabel('Metric', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Chinese Token Percentage Comparison Between Models', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add value labels
    for i, v in enumerate(anthropic_values):
        ax.text(i - width/2, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
    
    for i, v in enumerate(deepseek_values):
        ax.text(i + width/2, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('reports/visualizations/chinese_tokenization/token_percentage_comparison.png', dpi=300)
    plt.close()
    
    # 3. Chinese vs English Efficiency Comparison
    plt.figure(figsize=(12, 8))
    
    models = ['Anthropic Claude', 'Deepseek']
    efficiency_values = [
        comparison_results['anthropic_efficiency'],
        comparison_results['deepseek_efficiency']
    ]
    
    # Create bar chart with conditional colors
    colors = ['#e74c3c' if eff < 0 else '#2ecc71' for eff in efficiency_values]
    bars = plt.bar(models, efficiency_values, color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels on top/bottom of bars
    for i, v in enumerate(efficiency_values):
        va = 'bottom' if v >= 0 else 'top'
        y_offset = 0.5 if v >= 0 else -0.5
        plt.text(i, v + y_offset, f"{v:.1f}%", ha='center', va=va, fontweight='bold')
    
    # Add title and labels
    plt.title('Chinese Efficiency Relative to English by Model', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Model', fontsize=14, fontweight='bold')
    plt.ylabel('Efficiency (%)', fontsize=14, fontweight='bold')
    
    # Add reference line at 0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='More Efficient'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Less Efficient')
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('reports/visualizations/chinese_tokenization/chinese_vs_english_efficiency.png', dpi=300)
    plt.close()
    
    # 4. Input/Output Token Distribution Comparison
    plt.figure(figsize=(14, 10))
    
    # Prepare data
    categories = ['Chinese (Anthropic)', 'English (Anthropic)', 'Chinese (Deepseek)', 'English (Deepseek)']
    input_values = [
        comparison_results['anthropic_chinese_input'],
        comparison_results['anthropic_english_input'],
        comparison_results['deepseek_chinese_input'],
        comparison_results['deepseek_english_input']
    ]
    output_values = [
        comparison_results['anthropic_chinese_output'],
        comparison_results['anthropic_english_output'],
        comparison_results['deepseek_chinese_output'],
        comparison_results['deepseek_english_output']
    ]
    
    x = np.arange(len(categories))
    width = 0.7
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create stacked bars
    ax.bar(x, input_values, width, label='Input Tokens', color='#3498db')
    ax.bar(x, output_values, width, bottom=input_values, label='Output Tokens', color='#e74c3c')
    
    # Add labels and title
    ax.set_xlabel('Language and Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Tokens', fontsize=14, fontweight='bold')
    ax.set_title('Input/Output Token Distribution Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Add value labels
    for i, (input_val, output_val) in enumerate(zip(input_values, output_values)):
        total = input_val + output_val
        ax.text(i, input_val / 2, f"{input_val:.0f}\n({input_val/total*100:.1f}%)", ha='center', va='center', fontweight='bold')
        ax.text(i, input_val + output_val / 2, f"{output_val:.0f}\n({output_val/total*100:.1f}%)", ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('reports/visualizations/chinese_tokenization/input_output_distribution.png', dpi=300)
    plt.close()
    
    return [
        'reports/visualizations/chinese_tokenization/token_usage_comparison.png',
        'reports/visualizations/chinese_tokenization/token_percentage_comparison.png',
        'reports/visualizations/chinese_tokenization/chinese_vs_english_efficiency.png',
        'reports/visualizations/chinese_tokenization/input_output_distribution.png'
    ]

def main():
    """
    Main function to analyze Chinese tokenization.
    """
    print("Analyzing Chinese tokenization between models...")
    
    # Load results
    anthropic_df, deepseek_df = load_all_results()
    
    # Analyze Chinese tokenization
    anthropic_metrics, deepseek_metrics, differences = analyze_chinese_tokenization(anthropic_df, deepseek_df)
    
    print("\nAnthropic Chinese Metrics:")
    for metric, value in anthropic_metrics.items():
        print(f"  {metric}: {value:.2f}")
    
    print("\nDeepseek Chinese Metrics:")
    for metric, value in deepseek_metrics.items():
        print(f"  {metric}: {value:.2f}")
    
    print("\nDifferences (Deepseek - Anthropic):")
    for metric, value in differences.items():
        print(f"  {metric}: {value:.2f}")
    
    # Analyze Chinese vs English
    comparison_results = analyze_chinese_vs_english(anthropic_df, deepseek_df)
    
    print("\nChinese vs English Efficiency:")
    print(f"  Anthropic: {comparison_results['anthropic_efficiency']:.2f}%")
    print(f"  Deepseek: {comparison_results['deepseek_efficiency']:.2f}%")
    
    # Create visualizations
    visualization_files = create_visualizations(anthropic_metrics, deepseek_metrics, differences, comparison_results)
    
    print("\nVisualization files:")
    for viz_file in visualization_files:
        print(f"- {viz_file}")
    
    # Save analysis results
    os.makedirs('analysis_results', exist_ok=True)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'metric': list(anthropic_metrics.keys()),
        'anthropic': list(anthropic_metrics.values()),
        'deepseek': list(deepseek_metrics.values()),
        'difference': list(differences.values())
    })
    
    metrics_df.to_csv('analysis_results/chinese_tokenization_metrics.csv', index=False)
    
    # Save comparison results to CSV
    comparison_df = pd.DataFrame({
        'metric': ['Chinese Efficiency vs English'],
        'anthropic': [comparison_results['anthropic_efficiency']],
        'deepseek': [comparison_results['deepseek_efficiency']],
        'difference': [comparison_results['deepseek_efficiency'] - comparison_results['anthropic_efficiency']]
    })
    
    comparison_df.to_csv('analysis_results/chinese_vs_english_efficiency.csv', index=False)
    
    print("\nAnalysis results saved to CSV files in analysis_results directory")

if __name__ == "__main__":
    main()
