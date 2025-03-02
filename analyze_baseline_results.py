"""
Script to perform detailed analysis of baseline experiment results.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.json_utils import CustomJSONEncoder, save_json

def load_results(filename):
    """Load experiment results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def load_analysis(filename):
    """Load analysis results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def analyze_information_density(results_df):
    """Analyze information density metrics in more detail."""
    # Group by prompt_type
    grouped = results_df.groupby('prompt_type').agg({
        'response_bits_per_token': ['mean', 'std', 'median'],
        'response_chinese_ratio': ['mean', 'std', 'median'],
        'response_chinese_chars_per_token': ['mean', 'std', 'median'],
        'response_compression_ratio': ['mean', 'std', 'median']
    })
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    
    # Calculate relative information density
    prompt_types = results_df['prompt_type'].unique()
    comparisons = {}
    
    for i, type1 in enumerate(prompt_types):
        for type2 in prompt_types[i+1:]:
            type1_bits = results_df[results_df['prompt_type'] == type1]['response_bits_per_token'].mean()
            type2_bits = results_df[results_df['prompt_type'] == type2]['response_bits_per_token'].mean()
            
            bits_diff = type2_bits - type1_bits
            bits_percent = (bits_diff / type1_bits) * 100 if type1_bits > 0 else 0
            
            comparisons[f"{type1}_vs_{type2}_bits"] = {
                'absolute_difference': bits_diff,
                'percent_difference': bits_percent,
                f"{type2}_more_dense": bits_diff > 0,
                'density_gain': abs(bits_percent)
            }
    
    return {
        'information_density_by_prompt': grouped.to_dict(),
        'information_density_comparisons': comparisons
    }

def analyze_by_benchmark_and_difficulty(results_df):
    """Analyze results by benchmark and difficulty level."""
    # Group by benchmark and prompt_type
    benchmark_grouped = results_df.groupby(['benchmark', 'prompt_type']).agg({
        'total_tokens': ['mean', 'std', 'count'],
        'response_bits_per_token': ['mean'],
        'response_chinese_ratio': ['mean'],
        'response_compression_ratio': ['mean']
    }).reset_index()
    
    # Flatten column names
    benchmark_grouped.columns = ['_'.join(col).strip('_') for col in benchmark_grouped.columns.values]
    
    # Group by difficulty and prompt_type
    difficulty_grouped = results_df.groupby(['difficulty', 'prompt_type']).agg({
        'total_tokens': ['mean', 'std', 'count'],
        'response_bits_per_token': ['mean'],
        'response_chinese_ratio': ['mean'],
        'response_compression_ratio': ['mean']
    }).reset_index()
    
    # Flatten column names
    difficulty_grouped.columns = ['_'.join(col).strip('_') for col in difficulty_grouped.columns.values]
    
    # Calculate efficiency by benchmark
    benchmarks = results_df['benchmark'].unique()
    benchmark_efficiency = {}
    
    for benchmark in benchmarks:
        benchmark_data = results_df[results_df['benchmark'] == benchmark]
        
        english_data = benchmark_data[benchmark_data['prompt_type'] == 'english']
        chinese_data = benchmark_data[benchmark_data['prompt_type'] == 'chinese']
        
        if not english_data.empty and not chinese_data.empty:
            english_tokens = english_data['total_tokens'].mean()
            chinese_tokens = chinese_data['total_tokens'].mean()
            
            token_diff = english_tokens - chinese_tokens
            token_percent = (token_diff / english_tokens) * 100 if english_tokens > 0 else 0
            
            benchmark_efficiency[benchmark] = {
                'english_tokens': english_tokens,
                'chinese_tokens': chinese_tokens,
                'token_difference': token_diff,
                'percent_difference': token_percent,
                'chinese_more_efficient': token_diff > 0,
                'efficiency_gain': abs(token_percent)
            }
    
    # Calculate efficiency by difficulty
    difficulties = results_df['difficulty'].unique()
    difficulty_efficiency = {}
    
    for difficulty in difficulties:
        difficulty_data = results_df[results_df['difficulty'] == difficulty]
        
        english_data = difficulty_data[difficulty_data['prompt_type'] == 'english']
        chinese_data = difficulty_data[difficulty_data['prompt_type'] == 'chinese']
        
        if not english_data.empty and not chinese_data.empty:
            english_tokens = english_data['total_tokens'].mean()
            chinese_tokens = chinese_data['total_tokens'].mean()
            
            token_diff = english_tokens - chinese_tokens
            token_percent = (token_diff / english_tokens) * 100 if english_tokens > 0 else 0
            
            difficulty_efficiency[difficulty] = {
                'english_tokens': english_tokens,
                'chinese_tokens': chinese_tokens,
                'token_difference': token_diff,
                'percent_difference': token_percent,
                'chinese_more_efficient': token_diff > 0,
                'efficiency_gain': abs(token_percent)
            }
    
    return {
        'benchmark_grouped': benchmark_grouped.to_dict(),
        'difficulty_grouped': difficulty_grouped.to_dict(),
        'benchmark_efficiency': benchmark_efficiency,
        'difficulty_efficiency': difficulty_efficiency
    }

def create_detailed_visualizations(results_df, output_dir):
    """Create detailed visualizations for the analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('ggplot')
    
    # 1. Token usage by prompt type and benchmark
    plt.figure(figsize=(12, 8))
    benchmark_order = results_df.groupby('benchmark')['total_tokens'].mean().sort_values(ascending=False).index
    
    ax = pd.pivot_table(
        results_df, 
        values='total_tokens', 
        index='benchmark', 
        columns='prompt_type', 
        aggfunc='mean'
    ).reindex(benchmark_order).plot(kind='bar', rot=45)
    
    plt.title('Token Usage by Prompt Type and Benchmark', fontsize=14)
    plt.ylabel('Average Tokens', fontsize=12)
    plt.xlabel('Benchmark', fontsize=12)
    plt.legend(title='Prompt Type')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/token_usage_by_benchmark_detailed.png", dpi=300)
    plt.close()
    
    # 2. Information density by prompt type
    plt.figure(figsize=(10, 6))
    pd.pivot_table(
        results_df, 
        values='response_bits_per_token', 
        index='prompt_type', 
        aggfunc=['mean', 'std']
    ).plot(kind='bar', rot=0, yerr='std')
    
    plt.title('Information Density (Bits per Token) by Prompt Type', fontsize=14)
    plt.ylabel('Bits per Token', fontsize=12)
    plt.xlabel('Prompt Type', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/information_density_detailed.png", dpi=300)
    plt.close()
    
    # 3. Chinese character metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Chinese ratio
    results_df[results_df['prompt_type'] == 'chinese'].boxplot(
        column='response_chinese_ratio', 
        by='benchmark', 
        ax=axes[0]
    )
    axes[0].set_title('Chinese Character Ratio by Benchmark')
    axes[0].set_ylabel('Chinese Character Ratio')
    axes[0].set_xlabel('Benchmark')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Chinese chars per token
    results_df[results_df['prompt_type'] == 'chinese'].boxplot(
        column='response_chinese_chars_per_token', 
        by='benchmark', 
        ax=axes[1]
    )
    axes[1].set_title('Chinese Characters per Token by Benchmark')
    axes[1].set_ylabel('Chinese Characters per Token')
    axes[1].set_xlabel('Benchmark')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Chinese Character Metrics by Benchmark', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/chinese_character_metrics.png", dpi=300)
    plt.close()
    
    # 4. Efficiency gain by difficulty
    plt.figure(figsize=(10, 6))
    
    difficulty_data = []
    for difficulty in results_df['difficulty'].unique():
        diff_df = results_df[results_df['difficulty'] == difficulty]
        
        english_tokens = diff_df[diff_df['prompt_type'] == 'english']['total_tokens'].mean()
        chinese_tokens = diff_df[diff_df['prompt_type'] == 'chinese']['total_tokens'].mean()
        
        if not np.isnan(english_tokens) and not np.isnan(chinese_tokens):
            efficiency = (english_tokens - chinese_tokens) / english_tokens * 100
            difficulty_data.append({
                'difficulty': difficulty,
                'efficiency_gain': efficiency
            })
    
    difficulty_df = pd.DataFrame(difficulty_data)
    if not difficulty_df.empty:
        difficulty_df = difficulty_df.sort_values('efficiency_gain', ascending=False)
        
        bars = plt.bar(difficulty_df['difficulty'], difficulty_df['efficiency_gain'])
        
        # Color bars based on positive/negative values
        for i, bar in enumerate(bars):
            if difficulty_df['efficiency_gain'].iloc[i] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Chinese Reasoning Efficiency Gain by Problem Difficulty', fontsize=14)
        plt.ylabel('Efficiency Gain (%)', fontsize=12)
        plt.xlabel('Difficulty Level', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/efficiency_by_difficulty.png", dpi=300)
    plt.close()
    
    # 5. Compression ratio by benchmark
    plt.figure(figsize=(12, 6))
    compression_data = results_df[results_df['prompt_type'] == 'chinese']
    
    benchmark_order = compression_data.groupby('benchmark')['response_compression_ratio'].mean().sort_values(ascending=False).index
    
    compression_data.pivot_table(
        values='response_compression_ratio',
        index='benchmark',
        aggfunc=['mean', 'std']
    ).reindex(benchmark_order).plot(kind='bar', rot=45, yerr='std')
    
    plt.title('Compression Ratio by Benchmark (Chinese Reasoning)', fontsize=14)
    plt.ylabel('Compression Ratio', fontsize=12)
    plt.xlabel('Benchmark', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/compression_ratio_by_benchmark.png", dpi=300)
    plt.close()
    
    return {
        'visualizations_created': [
            'token_usage_by_benchmark_detailed.png',
            'information_density_detailed.png',
            'chinese_character_metrics.png',
            'efficiency_by_difficulty.png',
            'compression_ratio_by_benchmark.png'
        ]
    }

def main():
    """Perform detailed analysis of baseline experiment results."""
    # Create necessary directories
    os.makedirs("reports/detailed_analysis", exist_ok=True)
    
    # Load baseline results
    results_file = "experiment_results/baseline_results.json"
    results = load_results(results_file)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Load existing analysis
    analysis_file = "reports/baseline_analysis.json"
    baseline_analysis = load_analysis(analysis_file)
    
    # Perform detailed information density analysis
    print("Analyzing information density metrics...")
    info_density_analysis = analyze_information_density(results_df)
    
    # Analyze by benchmark and difficulty
    print("Analyzing results by benchmark and difficulty...")
    benchmark_difficulty_analysis = analyze_by_benchmark_and_difficulty(results_df)
    
    # Create detailed visualizations
    print("Creating detailed visualizations...")
    visualization_info = create_detailed_visualizations(results_df, "reports/visualizations/detailed")
    
    # Combine all analyses
    detailed_analysis = {
        'baseline_summary': baseline_analysis.get('prompt_type_comparisons', {}),
        'information_density_analysis': info_density_analysis,
        'benchmark_difficulty_analysis': benchmark_difficulty_analysis,
        'visualization_info': visualization_info
    }
    
    # Save detailed analysis
    detailed_analysis_file = "reports/detailed_analysis/baseline_detailed_analysis.json"
    save_json(detailed_analysis, detailed_analysis_file)
    
    # Print summary
    print("\nDETAILED ANALYSIS SUMMARY:")
    
    # Information density summary
    if 'information_density_comparisons' in info_density_analysis:
        comp = info_density_analysis['information_density_comparisons'].get('english_vs_chinese_bits', {})
        if 'percent_difference' in comp:
            print(f"Information Density: Chinese has {abs(comp.get('percent_difference', 0)):.2f}% "
                  f"{'higher' if comp.get('chinese_more_dense', False) else 'lower'} "
                  f"information density than English")
    
    # Benchmark efficiency summary
    if 'benchmark_efficiency' in benchmark_difficulty_analysis:
        print("\nEfficiency by Benchmark:")
        for benchmark, data in benchmark_difficulty_analysis['benchmark_efficiency'].items():
            print(f"  {benchmark}: Chinese is "
                  f"{'more' if data.get('chinese_more_efficient', False) else 'less'} "
                  f"efficient by {abs(data.get('percent_difference', 0)):.2f}%")
    
    # Difficulty efficiency summary
    if 'difficulty_efficiency' in benchmark_difficulty_analysis:
        print("\nEfficiency by Difficulty:")
        for difficulty, data in benchmark_difficulty_analysis['difficulty_efficiency'].items():
            print(f"  {difficulty}: Chinese is "
                  f"{'more' if data.get('chinese_more_efficient', False) else 'less'} "
                  f"efficient by {abs(data.get('percent_difference', 0)):.2f}%")
    
    print(f"\nDetailed analysis saved to {detailed_analysis_file}")
    print(f"Detailed visualizations saved to reports/visualizations/detailed/")

if __name__ == "__main__":
    main()
