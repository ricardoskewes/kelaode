"""
Analyze efficiency differences between Anthropic and Deepseek models.
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

def analyze_efficiency_differences(anthropic_df, deepseek_df):
    """
    Analyze efficiency differences between models.
    """
    # Calculate efficiency for Anthropic
    anthropic_tokens = anthropic_df.groupby('prompt_type')['total_tokens'].mean()
    anthropic_english = anthropic_tokens['english']
    
    anthropic_efficiency = {}
    for lang, tokens in anthropic_tokens.items():
        if lang != 'english':
            efficiency = (anthropic_english - tokens) / anthropic_english * 100
            anthropic_efficiency[lang] = efficiency
    
    # Calculate efficiency for Deepseek
    deepseek_tokens = deepseek_df.groupby('prompt_type')['total_tokens'].mean()
    deepseek_english = deepseek_tokens['english']
    
    deepseek_efficiency = {}
    for lang, tokens in deepseek_tokens.items():
        if lang != 'english':
            efficiency = (deepseek_english - tokens) / deepseek_english * 100
            deepseek_efficiency[lang] = efficiency
    
    # Create DataFrame for comparison
    comparison_data = []
    
    for lang in set(anthropic_efficiency.keys()) & set(deepseek_efficiency.keys()):
        comparison_data.append({
            'language': lang,
            'anthropic_efficiency': anthropic_efficiency[lang],
            'deepseek_efficiency': deepseek_efficiency[lang],
            'difference': deepseek_efficiency[lang] - anthropic_efficiency[lang]
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    return comparison_df

def analyze_by_problem_type(anthropic_df, deepseek_df):
    """
    Analyze efficiency differences by problem type.
    """
    # Extract problem types
    anthropic_df['problem_type'] = anthropic_df['problem_id'].apply(
        lambda x: x.split('_')[0] if '_' in x else 'unknown'
    )
    
    deepseek_df['problem_type'] = deepseek_df['problem_id'].apply(
        lambda x: x.split('_')[0] if '_' in x else 'unknown'
    )
    
    # Calculate efficiency by problem type for Anthropic
    problem_types = anthropic_df['problem_type'].unique()
    
    problem_type_data = []
    
    for problem_type in problem_types:
        anthropic_type_df = anthropic_df[anthropic_df['problem_type'] == problem_type]
        deepseek_type_df = deepseek_df[deepseek_df['problem_type'] == problem_type]
        
        if len(anthropic_type_df) > 0 and len(deepseek_type_df) > 0:
            # Calculate for Anthropic
            anthropic_tokens = anthropic_type_df.groupby('prompt_type')['total_tokens'].mean()
            if 'english' in anthropic_tokens and 'chinese' in anthropic_tokens:
                anthropic_english = anthropic_tokens['english']
                anthropic_chinese = anthropic_tokens['chinese']
                anthropic_efficiency = (anthropic_english - anthropic_chinese) / anthropic_english * 100
                
                # Calculate for Deepseek
                deepseek_tokens = deepseek_type_df.groupby('prompt_type')['total_tokens'].mean()
                if 'english' in deepseek_tokens and 'chinese' in deepseek_tokens:
                    deepseek_english = deepseek_tokens['english']
                    deepseek_chinese = deepseek_tokens['chinese']
                    deepseek_efficiency = (deepseek_english - deepseek_chinese) / deepseek_english * 100
                    
                    problem_type_data.append({
                        'problem_type': problem_type,
                        'anthropic_efficiency': anthropic_efficiency,
                        'deepseek_efficiency': deepseek_efficiency,
                        'difference': deepseek_efficiency - anthropic_efficiency
                    })
    
    problem_type_df = pd.DataFrame(problem_type_data)
    
    return problem_type_df

def analyze_tokenization_differences(anthropic_df, deepseek_df):
    """
    Analyze tokenization differences between models.
    """
    # Check if input_chars column exists
    if 'input_chars' in anthropic_df.columns and 'input_chars' in deepseek_df.columns:
        # Calculate average tokens per character for each language
        anthropic_df['chars_per_token'] = anthropic_df['input_chars'] / anthropic_df['input_tokens']
        deepseek_df['chars_per_token'] = deepseek_df['input_chars'] / deepseek_df['input_tokens']
        
        # Group by language
        anthropic_chars_per_token = anthropic_df.groupby('prompt_type')['chars_per_token'].mean()
        deepseek_chars_per_token = deepseek_df.groupby('prompt_type')['chars_per_token'].mean()
        
        # Create DataFrame for comparison
        tokenization_data = []
        
        for lang in set(anthropic_chars_per_token.index) & set(deepseek_chars_per_token.index):
            tokenization_data.append({
                'language': lang,
                'anthropic_chars_per_token': anthropic_chars_per_token[lang],
                'deepseek_chars_per_token': deepseek_chars_per_token[lang],
                'difference': deepseek_chars_per_token[lang] - anthropic_chars_per_token[lang]
            })
        
        tokenization_df = pd.DataFrame(tokenization_data)
    else:
        # If input_chars column doesn't exist, create an empty DataFrame
        print("Warning: 'input_chars' column not found in one or both dataframes.")
        print("Skipping tokenization difference analysis.")
        tokenization_df = pd.DataFrame(columns=['language', 'anthropic_chars_per_token', 'deepseek_chars_per_token', 'difference'])
    
    return tokenization_df

def create_visualizations(comparison_df, problem_type_df, tokenization_df):
    """
    Create visualizations to illustrate efficiency differences.
    """
    os.makedirs('reports/visualizations/model_comparison', exist_ok=True)
    
    # 1. Efficiency comparison bar chart
    plt.figure(figsize=(12, 8))
    
    # Prepare data for grouped bar chart
    languages = comparison_df['language']
    anthropic_efficiency = comparison_df['anthropic_efficiency']
    deepseek_efficiency = comparison_df['deepseek_efficiency']
    
    x = np.arange(len(languages))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, anthropic_efficiency, width, label='Anthropic Claude', color='#3498db')
    rects2 = ax.bar(x + width/2, deepseek_efficiency, width, label='Deepseek', color='#e74c3c')
    
    # Add labels and title
    ax.set_xlabel('Language', fontsize=14, fontweight='bold')
    ax.set_ylabel('Efficiency vs. English (%)', fontsize=14, fontweight='bold')
    ax.set_title('Language Efficiency Comparison Between Models', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.legend()
    
    # Add value labels
    for i, v in enumerate(anthropic_efficiency):
        ax.text(i - width/2, v + (1 if v >= 0 else -3), f"{v:.1f}%", ha='center', fontweight='bold')
    
    for i, v in enumerate(deepseek_efficiency):
        ax.text(i + width/2, v + (1 if v >= 0 else -3), f"{v:.1f}%", ha='center', fontweight='bold')
    
    # Add reference line at 0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('reports/visualizations/model_comparison/efficiency_comparison.png', dpi=300)
    plt.close()
    
    # 2. Problem type efficiency heatmap
    if len(problem_type_df) > 0:
        plt.figure(figsize=(12, 8))
        
        # Create pivot table for heatmap
        pivot_df = problem_type_df.pivot(index='problem_type', columns=['anthropic_efficiency', 'deepseek_efficiency'])
        
        # Create heatmap
        sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="RdYlGn", center=0)
        
        plt.title('Chinese Efficiency by Problem Type and Model', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('reports/visualizations/model_comparison/problem_type_efficiency.png', dpi=300)
        plt.close()
    
    # 3. Tokenization difference visualization
    if not tokenization_df.empty:
        plt.figure(figsize=(12, 8))
        
        # Create bar chart
        sns.barplot(x='language', y='difference', data=tokenization_df, palette='viridis')
        
        plt.title('Difference in Characters per Token (Deepseek - Anthropic)', fontsize=16, fontweight='bold')
        plt.xlabel('Language', fontsize=14, fontweight='bold')
        plt.ylabel('Difference in Chars/Token', fontsize=14, fontweight='bold')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('reports/visualizations/model_comparison/tokenization_difference.png', dpi=300)
        plt.close()
    else:
        print("Skipping tokenization difference visualization due to missing data.")
    
    return [
        'reports/visualizations/model_comparison/efficiency_comparison.png',
        'reports/visualizations/model_comparison/problem_type_efficiency.png',
        'reports/visualizations/model_comparison/tokenization_difference.png'
    ]

def main():
    """
    Main function to analyze efficiency differences.
    """
    print("Analyzing efficiency differences between models...")
    
    # Load results
    anthropic_df, deepseek_df = load_all_results()
    
    # Analyze efficiency differences
    comparison_df = analyze_efficiency_differences(anthropic_df, deepseek_df)
    print("\nEfficiency Comparison:")
    print(comparison_df)
    
    # Analyze by problem type
    problem_type_df = analyze_by_problem_type(anthropic_df, deepseek_df)
    if len(problem_type_df) > 0:
        print("\nEfficiency by Problem Type:")
        print(problem_type_df)
    
    # Analyze tokenization differences
    tokenization_df = analyze_tokenization_differences(anthropic_df, deepseek_df)
    print("\nTokenization Differences:")
    print(tokenization_df)
    
    # Create visualizations
    visualization_files = create_visualizations(comparison_df, problem_type_df, tokenization_df)
    
    print("\nVisualization files:")
    for viz_file in visualization_files:
        print(f"- {viz_file}")
    
    # Save analysis results
    os.makedirs('analysis_results', exist_ok=True)
    
    comparison_df.to_csv('analysis_results/model_efficiency_comparison.csv', index=False)
    if len(problem_type_df) > 0:
        problem_type_df.to_csv('analysis_results/problem_type_efficiency.csv', index=False)
    tokenization_df.to_csv('analysis_results/tokenization_differences.csv', index=False)
    
    print("\nAnalysis results saved to CSV files in analysis_results directory")

if __name__ == "__main__":
    main()
