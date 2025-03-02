"""
Script to create comprehensive visualizations for language efficiency analysis.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from utils.json_utils import CustomJSONEncoder, save_json

def load_results(filename):
    """Load experiment results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def load_analysis(filename):
    """Load analysis results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def create_efficiency_heatmap(results_df, output_dir):
    """Create a heatmap showing efficiency gains/losses by benchmark and difficulty."""
    # Create pivot table for heatmap data
    pivot_data = []
    
    # Get unique benchmarks and difficulties
    benchmarks = results_df['benchmark'].unique()
    difficulties = results_df['difficulty'].unique()
    
    for benchmark in benchmarks:
        for difficulty in difficulties:
            # Filter data
            filtered = results_df[(results_df['benchmark'] == benchmark) & 
                                 (results_df['difficulty'] == difficulty)]
            
            # Calculate efficiency if we have both English and Chinese data
            english_data = filtered[filtered['prompt_type'] == 'english']
            chinese_data = filtered[filtered['prompt_type'] == 'chinese']
            
            if not english_data.empty and not chinese_data.empty:
                english_tokens = english_data['total_tokens'].mean()
                chinese_tokens = chinese_data['total_tokens'].mean()
                
                efficiency = (english_tokens - chinese_tokens) / english_tokens * 100
                
                pivot_data.append({
                    'benchmark': benchmark,
                    'difficulty': difficulty,
                    'efficiency_gain': efficiency
                })
    
    # Create DataFrame from pivot data
    pivot_df = pd.DataFrame(pivot_data)
    
    if not pivot_df.empty:
        # Create pivot table
        heatmap_data = pivot_df.pivot(index='benchmark', columns='difficulty', values='efficiency_gain')
        
        # Create custom colormap (red for negative, white for zero, green for positive)
        colors = ['#d73027', '#f7f7f7', '#1a9850']  # red, white, green
        cmap = LinearSegmentedColormap.from_list('efficiency_cmap', colors, N=100)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(heatmap_data, cmap=cmap, center=0, annot=True, fmt='.2f',
                         linewidths=.5, cbar_kws={'label': 'Efficiency Gain (%)'})
        
        plt.title('Chinese vs. English Efficiency Gain by Benchmark and Difficulty (%)', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/efficiency_heatmap.png", dpi=300)
        plt.close()
        
        return heatmap_data.to_dict()
    
    return {}

def create_token_distribution_plot(results_df, output_dir):
    """Create violin plots showing token distribution by prompt type."""
    plt.figure(figsize=(12, 8))
    
    # Create violin plot
    ax = sns.violinplot(x='benchmark', y='total_tokens', hue='prompt_type', 
                        data=results_df, palette='Set2', split=True, inner='quart')
    
    plt.title('Token Distribution by Benchmark and Prompt Type', fontsize=14)
    plt.ylabel('Total Tokens', fontsize=12)
    plt.xlabel('Benchmark', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Prompt Type')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/token_distribution.png", dpi=300)
    plt.close()

def create_information_density_comparison(results_df, output_dir):
    """Create a detailed comparison of information density metrics."""
    # Filter data
    english_data = results_df[results_df['prompt_type'] == 'english']
    chinese_data = results_df[results_df['prompt_type'] == 'chinese']
    
    # Calculate metrics
    metrics = {
        'english': {
            'bits_per_token': english_data['response_bits_per_token'].mean(),
            'chars_per_token': english_data['total_tokens'].sum() / english_data['response_text'].str.len().sum(),
            'total_tokens': english_data['total_tokens'].mean()
        },
        'chinese': {
            'bits_per_token': chinese_data['response_bits_per_token'].mean(),
            'chars_per_token': chinese_data['response_chinese_chars_per_token'].mean(),
            'total_tokens': chinese_data['total_tokens'].mean()
        }
    }
    
    # Calculate differences
    differences = {
        'bits_per_token_diff': (metrics['chinese']['bits_per_token'] - metrics['english']['bits_per_token']) / metrics['english']['bits_per_token'] * 100,
        'chars_per_token_diff': (metrics['chinese']['chars_per_token'] - metrics['english']['chars_per_token']) / metrics['english']['chars_per_token'] * 100,
        'total_tokens_diff': (metrics['chinese']['total_tokens'] - metrics['english']['total_tokens']) / metrics['english']['total_tokens'] * 100
    }
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Data for plotting
    metrics_to_plot = ['bits_per_token', 'chars_per_token', 'total_tokens']
    english_values = [metrics['english'][m] for m in metrics_to_plot]
    chinese_values = [metrics['chinese'][m] for m in metrics_to_plot]
    
    # Set positions and width for bars
    pos = np.arange(len(metrics_to_plot))
    width = 0.35
    
    # Create bars
    english_bars = ax.bar(pos - width/2, english_values, width, label='English', color='#1f77b4')
    chinese_bars = ax.bar(pos + width/2, chinese_values, width, label='Chinese', color='#ff7f0e')
    
    # Add labels and title
    ax.set_ylabel('Value')
    ax.set_title('Information Density Metrics Comparison')
    ax.set_xticks(pos)
    ax.set_xticklabels(metrics_to_plot)
    ax.legend()
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(english_bars)
    autolabel(chinese_bars)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/information_density_comparison.png", dpi=300)
    plt.close()
    
    # Create a second plot showing percentage differences
    plt.figure(figsize=(10, 6))
    
    # Data for plotting
    diff_labels = ['Bits per Token', 'Chars per Token', 'Total Tokens']
    diff_values = [differences['bits_per_token_diff'], differences['chars_per_token_diff'], differences['total_tokens_diff']]
    
    # Create bars with colors based on positive/negative values
    bars = plt.bar(diff_labels, diff_values, color=['red' if x < 0 else 'green' for x in diff_values])
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3 if height > 0 else -14),  # offset based on bar direction
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top')
    
    plt.title('Percentage Difference: Chinese vs. English', fontsize=14)
    plt.ylabel('Difference (%)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/information_density_percent_diff.png", dpi=300)
    plt.close()
    
    return {
        'metrics': metrics,
        'differences': differences
    }

def create_statistical_significance_plot(analysis_data, output_dir):
    """Create a plot showing statistical significance of results."""
    if 'statistical_significance' not in analysis_data:
        return {}
    
    significance_data = analysis_data['statistical_significance']
    
    # Extract data for plotting
    plot_data = []
    for comparison, data in significance_data.items():
        if 'p_value' in data and 'mean_difference' in data:
            plot_data.append({
                'comparison': comparison,
                'p_value': data['p_value'],
                'mean_difference': data['mean_difference'],
                'significant': data.get('significant_at_0.05', False)
            })
    
    if not plot_data:
        return {}
    
    # Create DataFrame
    plot_df = pd.DataFrame(plot_data)
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    
    # Plot points
    scatter = plt.scatter(plot_df['mean_difference'], -np.log10(plot_df['p_value']),
                         c=plot_df['significant'].map({True: 'green', False: 'red'}),
                         s=100, alpha=0.7)
    
    # Add significance threshold line
    plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
    
    # Add labels for each point
    for i, row in plot_df.iterrows():
        plt.annotate(row['comparison'],
                    xy=(row['mean_difference'], -np.log10(row['p_value'])),
                    xytext=(5, 5),
                    textcoords='offset points')
    
    plt.title('Statistical Significance of Token Efficiency Differences', fontsize=14)
    plt.xlabel('Mean Difference in Tokens (English - Chinese)', fontsize=12)
    plt.ylabel('-log10(p-value)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Significant (p<0.05)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Not Significant'),
        Line2D([0], [0], color='red', linestyle='--', label='Significance Threshold (p=0.05)')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/statistical_significance.png", dpi=300)
    plt.close()
    
    return plot_df.to_dict(orient='records')

def create_benchmark_radar_chart(results_df, output_dir):
    """Create a radar chart showing efficiency across benchmarks."""
    # Calculate efficiency by benchmark
    benchmark_efficiency = {}
    for benchmark in results_df['benchmark'].unique():
        benchmark_data = results_df[results_df['benchmark'] == benchmark]
        
        english_data = benchmark_data[benchmark_data['prompt_type'] == 'english']
        chinese_data = benchmark_data[benchmark_data['prompt_type'] == 'chinese']
        
        if not english_data.empty and not chinese_data.empty:
            english_tokens = english_data['total_tokens'].mean()
            chinese_tokens = chinese_data['total_tokens'].mean()
            
            efficiency = (english_tokens - chinese_tokens) / english_tokens * 100
            benchmark_efficiency[benchmark] = efficiency
    
    if not benchmark_efficiency:
        return {}
    
    # Prepare data for radar chart
    benchmarks = list(benchmark_efficiency.keys())
    efficiency_values = [benchmark_efficiency[b] for b in benchmarks]
    
    # Number of variables
    N = len(benchmarks)
    
    # Create angles for each benchmark
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    
    # Close the polygon
    efficiency_values.append(efficiency_values[0])
    angles.append(angles[0])
    benchmarks.append(benchmarks[0])
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Plot efficiency values
    ax.plot(angles, efficiency_values, 'o-', linewidth=2, label='Efficiency Gain (%)')
    ax.fill(angles, efficiency_values, alpha=0.25)
    
    # Add benchmark labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(benchmarks[:-1])
    
    # Add efficiency value labels
    for i, (angle, value, benchmark) in enumerate(zip(angles[:-1], efficiency_values[:-1], benchmarks[:-1])):
        ha = 'left' if angle < np.pi else 'right'
        ax.annotate(f"{value:.2f}%",
                   xy=(angle, value),
                   xytext=(1.1*np.cos(angle), 1.1*np.sin(angle)),
                   textcoords='data',
                   ha=ha)
    
    # Add zero line
    ax.plot(angles, [0]*len(angles), '--', color='gray', alpha=0.7, linewidth=1)
    
    # Set y-limits to ensure zero is included
    min_val = min(min(efficiency_values), 0)
    max_val = max(max(efficiency_values), 0)
    buffer = (max_val - min_val) * 0.1
    ax.set_ylim(min_val - buffer, max_val + buffer)
    
    # Add title and legend
    plt.title('Chinese vs. English Efficiency Gain by Benchmark (%)', size=15)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/benchmark_radar_chart.png", dpi=300)
    plt.close()
    
    return benchmark_efficiency

def main():
    """Create comprehensive visualizations for language efficiency analysis."""
    # Create necessary directories
    output_dir = "reports/visualizations/comprehensive"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load baseline results
    results_file = "experiment_results/baseline_results.json"
    results = load_results(results_file)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Load existing analysis
    analysis_file = "reports/baseline_analysis.json"
    baseline_analysis = load_analysis(analysis_file)
    
    # Create efficiency heatmap
    print("Creating efficiency heatmap...")
    heatmap_data = create_efficiency_heatmap(results_df, output_dir)
    
    # Create token distribution plot
    print("Creating token distribution plot...")
    create_token_distribution_plot(results_df, output_dir)
    
    # Create information density comparison
    print("Creating information density comparison...")
    density_data = create_information_density_comparison(results_df, output_dir)
    
    # Create statistical significance plot
    print("Creating statistical significance plot...")
    significance_data = create_statistical_significance_plot(baseline_analysis, output_dir)
    
    # Create benchmark radar chart
    print("Creating benchmark radar chart...")
    radar_data = create_benchmark_radar_chart(results_df, output_dir)
    
    # Save visualization data
    visualization_data = {
        'heatmap_data': heatmap_data,
        'density_data': density_data,
        'significance_data': significance_data,
        'radar_data': radar_data
    }
    
    visualization_data_file = f"{output_dir}/visualization_data.json"
    save_json(visualization_data, visualization_data_file)
    
    print(f"Comprehensive visualizations created in {output_dir}")
    print(f"Visualization data saved to {visualization_data_file}")
    
    # List created visualizations
    print("\nCreated visualizations:")
    for file in os.listdir(output_dir):
        if file.endswith('.png'):
            print(f"- {file}")

if __name__ == "__main__":
    main()
