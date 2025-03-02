"""
Analyze token distribution differences between Anthropic and Deepseek models.
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

def analyze_token_distribution(anthropic_df, deepseek_df):
    """
    Analyze token distribution between input and output tokens.
    """
    # Calculate token distribution for Anthropic
    anthropic_distribution = anthropic_df.groupby('prompt_type').agg({
        'input_tokens': 'mean',
        'output_tokens': 'mean',
        'total_tokens': 'mean'
    }).reset_index()
    
    anthropic_distribution['input_percentage'] = (anthropic_distribution['input_tokens'] / 
                                                anthropic_distribution['total_tokens'] * 100)
    anthropic_distribution['output_percentage'] = (anthropic_distribution['output_tokens'] / 
                                                 anthropic_distribution['total_tokens'] * 100)
    
    # Calculate token distribution for Deepseek
    deepseek_distribution = deepseek_df.groupby('prompt_type').agg({
        'input_tokens': 'mean',
        'output_tokens': 'mean',
        'total_tokens': 'mean'
    }).reset_index()
    
    deepseek_distribution['input_percentage'] = (deepseek_distribution['input_tokens'] / 
                                               deepseek_distribution['total_tokens'] * 100)
    deepseek_distribution['output_percentage'] = (deepseek_distribution['output_tokens'] / 
                                                deepseek_distribution['total_tokens'] * 100)
    
    return anthropic_distribution, deepseek_distribution

def create_visualizations(anthropic_distribution, deepseek_distribution):
    """
    Create visualizations for token distribution analysis.
    """
    os.makedirs('reports/visualizations/token_distribution', exist_ok=True)
    
    # 1. Input vs Output Token Distribution for Anthropic
    plt.figure(figsize=(12, 8))
    
    # Create stacked bar chart
    languages = anthropic_distribution['prompt_type']
    input_tokens = anthropic_distribution['input_tokens']
    output_tokens = anthropic_distribution['output_tokens']
    
    x = np.arange(len(languages))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create stacked bars
    ax.bar(x, input_tokens, width, label='Input Tokens', color='#3498db')
    ax.bar(x, output_tokens, width, bottom=input_tokens, label='Output Tokens', color='#e74c3c')
    
    # Add labels and title
    ax.set_xlabel('Language', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Tokens', fontsize=14, fontweight='bold')
    ax.set_title('Anthropic: Input vs Output Token Distribution by Language', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('reports/visualizations/token_distribution/anthropic_token_distribution.png', dpi=300)
    plt.close()
    
    # 2. Input vs Output Token Distribution for Deepseek
    plt.figure(figsize=(12, 8))
    
    # Create stacked bar chart
    languages = deepseek_distribution['prompt_type']
    input_tokens = deepseek_distribution['input_tokens']
    output_tokens = deepseek_distribution['output_tokens']
    
    x = np.arange(len(languages))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create stacked bars
    ax.bar(x, input_tokens, width, label='Input Tokens', color='#3498db')
    ax.bar(x, output_tokens, width, bottom=input_tokens, label='Output Tokens', color='#e74c3c')
    
    # Add labels and title
    ax.set_xlabel('Language', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Tokens', fontsize=14, fontweight='bold')
    ax.set_title('Deepseek: Input vs Output Token Distribution by Language', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('reports/visualizations/token_distribution/deepseek_token_distribution.png', dpi=300)
    plt.close()
    
    # 3. Input Percentage Comparison
    plt.figure(figsize=(12, 8))
    
    # Prepare data for grouped bar chart
    anthropic_languages = anthropic_distribution['prompt_type']
    anthropic_input_pct = anthropic_distribution['input_percentage']
    
    deepseek_languages = deepseek_distribution['prompt_type']
    deepseek_input_pct = deepseek_distribution['input_percentage']
    
    # Find common languages
    common_languages = []
    anthropic_pct = []
    deepseek_pct = []
    
    for i, lang in enumerate(anthropic_languages):
        if lang in deepseek_languages.values:
            common_languages.append(lang)
            anthropic_pct.append(anthropic_input_pct.iloc[i])
            
            idx = deepseek_languages[deepseek_languages == lang].index[0]
            deepseek_pct.append(deepseek_input_pct.iloc[idx])
    
    x = np.arange(len(common_languages))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, anthropic_pct, width, label='Anthropic Claude', color='#3498db')
    rects2 = ax.bar(x + width/2, deepseek_pct, width, label='Deepseek', color='#e74c3c')
    
    # Add labels and title
    ax.set_xlabel('Language', fontsize=14, fontweight='bold')
    ax.set_ylabel('Input Token Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Input Token Percentage Comparison Between Models', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(common_languages)
    ax.legend()
    
    # Add value labels
    for i, v in enumerate(anthropic_pct):
        ax.text(i - width/2, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
    
    for i, v in enumerate(deepseek_pct):
        ax.text(i + width/2, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('reports/visualizations/token_distribution/input_percentage_comparison.png', dpi=300)
    plt.close()
    
    # 4. Output Percentage Comparison
    plt.figure(figsize=(12, 8))
    
    # Prepare data for grouped bar chart
    anthropic_languages = anthropic_distribution['prompt_type']
    anthropic_output_pct = anthropic_distribution['output_percentage']
    
    deepseek_languages = deepseek_distribution['prompt_type']
    deepseek_output_pct = deepseek_distribution['output_percentage']
    
    # Find common languages
    common_languages = []
    anthropic_pct = []
    deepseek_pct = []
    
    for i, lang in enumerate(anthropic_languages):
        if lang in deepseek_languages.values:
            common_languages.append(lang)
            anthropic_pct.append(anthropic_output_pct.iloc[i])
            
            idx = deepseek_languages[deepseek_languages == lang].index[0]
            deepseek_pct.append(deepseek_output_pct.iloc[idx])
    
    x = np.arange(len(common_languages))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, anthropic_pct, width, label='Anthropic Claude', color='#3498db')
    rects2 = ax.bar(x + width/2, deepseek_pct, width, label='Deepseek', color='#e74c3c')
    
    # Add labels and title
    ax.set_xlabel('Language', fontsize=14, fontweight='bold')
    ax.set_ylabel('Output Token Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Output Token Percentage Comparison Between Models', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(common_languages)
    ax.legend()
    
    # Add value labels
    for i, v in enumerate(anthropic_pct):
        ax.text(i - width/2, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
    
    for i, v in enumerate(deepseek_pct):
        ax.text(i + width/2, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('reports/visualizations/token_distribution/output_percentage_comparison.png', dpi=300)
    plt.close()
    
    return [
        'reports/visualizations/token_distribution/anthropic_token_distribution.png',
        'reports/visualizations/token_distribution/deepseek_token_distribution.png',
        'reports/visualizations/token_distribution/input_percentage_comparison.png',
        'reports/visualizations/token_distribution/output_percentage_comparison.png'
    ]

def main():
    """
    Main function to analyze token distribution.
    """
    print("Analyzing token distribution between models...")
    
    # Load results
    anthropic_df, deepseek_df = load_all_results()
    
    # Analyze token distribution
    anthropic_distribution, deepseek_distribution = analyze_token_distribution(anthropic_df, deepseek_df)
    
    print("\nAnthropic Token Distribution:")
    print(anthropic_distribution)
    
    print("\nDeepseek Token Distribution:")
    print(deepseek_distribution)
    
    # Create visualizations
    visualization_files = create_visualizations(anthropic_distribution, deepseek_distribution)
    
    print("\nVisualization files:")
    for viz_file in visualization_files:
        print(f"- {viz_file}")
    
    # Save analysis results
    os.makedirs('analysis_results', exist_ok=True)
    
    anthropic_distribution.to_csv('analysis_results/anthropic_token_distribution.csv', index=False)
    deepseek_distribution.to_csv('analysis_results/deepseek_token_distribution.csv', index=False)
    
    print("\nAnalysis results saved to CSV files in analysis_results directory")

if __name__ == "__main__":
    main()
