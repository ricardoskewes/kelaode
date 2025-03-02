"""
Create normalized visualizations comparing Anthropic and Deepseek model results.
This script normalizes English token usage to 1.00 for both models.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def load_results():
    """
    Load experiment results from both Anthropic and Deepseek models.
    
    Returns:
        Tuple of (anthropic_df, deepseek_df, combined_df)
    """
    print("Loading experiment results...")
    
    # Load Deepseek results
    deepseek_df = None
    try:
        with open("experiment_results/deepseek_longcontext_results.json", 'r') as f:
            deepseek_results = json.load(f)
        deepseek_df = pd.DataFrame(deepseek_results)
        print(f"Loaded {len(deepseek_df)} Deepseek results")
    except Exception as e:
        print(f"Error loading Deepseek results: {str(e)}")
    
    # Load latest Anthropic interim results
    anthropic_df = None
    try:
        import glob
        interim_files = glob.glob("experiment_results/interim_results_*.json")
        if interim_files:
            latest_file = max(interim_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            with open(latest_file, 'r') as f:
                anthropic_results = json.load(f)
            
            # Filter for long-context QA problems (both HotpotQA and longqa)
            anthropic_df = pd.DataFrame(anthropic_results)
            anthropic_df = anthropic_df[
                (anthropic_df['problem_id'].str.startswith('hotpotqa_')) | 
                (anthropic_df['problem_id'].str.startswith('longqa_')) |
                (anthropic_df['benchmark'] == 'HotpotQA')
            ]
            print(f"Loaded {len(anthropic_df)} Anthropic long-context QA results")
        else:
            print("No Anthropic interim results found")
    except Exception as e:
        print(f"Error loading Anthropic results: {str(e)}")
    
    # Combine results if both are available
    combined_df = None
    if anthropic_df is not None and deepseek_df is not None:
        combined_df = pd.concat([anthropic_df, deepseek_df], ignore_index=True)
        print(f"Combined {len(combined_df)} total results")
    
    return anthropic_df, deepseek_df, combined_df

def create_normalized_visualizations(anthropic_df, deepseek_df, combined_df, output_dir):
    """
    Create normalized visualizations comparing Anthropic and Deepseek models.
    English token usage is normalized to 1.00 for both models.
    
    Args:
        anthropic_df: DataFrame with Anthropic results
        deepseek_df: DataFrame with Deepseek results
        combined_df: DataFrame with combined results
        output_dir: Directory to save visualizations to
    """
    print("Creating normalized model comparison visualizations...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have both models' data
    if combined_df is None:
        print("Cannot create model comparison visualizations without both models' data")
        return
    
    # Create a normalized version of the combined DataFrame
    normalized_df = combined_df.copy()
    
    # Calculate normalization factors for each model (English = 1.00)
    normalization_factors = {}
    for model in normalized_df['model'].unique():
        model_df = normalized_df[normalized_df['model'] == model]
        english_tokens = model_df[model_df['prompt_type'] == 'english']['total_tokens'].mean()
        normalization_factors[model] = english_tokens
    
    # Apply normalization
    for idx, row in normalized_df.iterrows():
        model = row['model']
        normalized_df.at[idx, 'normalized_tokens'] = row['total_tokens'] / normalization_factors[model]
    
    # 1. Normalized token usage by model and language
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=normalized_df,
        x='prompt_type',
        y='normalized_tokens',
        hue='model',
        errorbar=('ci', 95),
        palette='viridis'
    )
    plt.title('Normalized Token Usage by Model and Language (English = 1.00)', fontsize=16)
    plt.xlabel('Language', fontsize=14)
    plt.ylabel('Normalized Token Usage', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Model')
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'normalized_token_usage.png'), dpi=300)
    plt.close()
    
    # 2. Normalized efficiency relative to English by model
    plt.figure(figsize=(12, 8))
    
    # Calculate efficiency for each model
    model_efficiency = {}
    for model in normalized_df['model'].unique():
        model_df = normalized_df[normalized_df['model'] == model]
        
        efficiencies = []
        languages = []
        
        for lang in model_df['prompt_type'].unique():
            if lang == 'english':
                continue
            
            lang_tokens = model_df[model_df['prompt_type'] == lang]['normalized_tokens'].mean()
            efficiency = (1.0 - lang_tokens) * 100  # English is 1.0
            efficiencies.append(efficiency)
            languages.append(lang)
        
        model_efficiency[model] = pd.DataFrame({
            'language': languages,
            'efficiency': efficiencies
        })
    
    # Combine efficiency data
    efficiency_data = []
    for model, df in model_efficiency.items():
        df['model'] = model
        efficiency_data.append(df)
    
    if efficiency_data:
        efficiency_df = pd.concat(efficiency_data, ignore_index=True)
        
        # Create the plot
        sns.barplot(
            data=efficiency_df,
            x='language',
            y='efficiency',
            hue='model',
            palette='viridis'
        )
        
        plt.title('Normalized Efficiency Relative to English by Model', fontsize=16)
        plt.xlabel('Language', fontsize=14)
        plt.ylabel('Efficiency (% token reduction)', fontsize=14)
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'normalized_efficiency.png'), dpi=300)
        plt.close()
    
    # 3. Normalized model comparison heatmap
    plt.figure(figsize=(12, 10))
    
    # Prepare data for heatmap
    pivot_data = normalized_df.pivot_table(
        index='prompt_type',
        columns='model',
        values='normalized_tokens',
        aggfunc='mean'
    )
    
    # Calculate percentage difference
    model_cols = pivot_data.columns
    if len(model_cols) >= 2:
        diff_pct = (pivot_data[model_cols[0]] - pivot_data[model_cols[1]]) * 100
        pivot_data['diff_pct'] = diff_pct
        
        # Create custom colormap (blue for negative, red for positive)
        colors = ['blue', 'white', 'red']
        cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=256)
        
        # Plot heatmap
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.2f',
            cmap=cmap,
            center=0,
            linewidths=.5,
            cbar_kws={'label': 'Normalized Token Usage / Difference'}
        )
        
        plt.title('Normalized Model Comparison: Token Usage by Language (English = 1.00)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'normalized_heatmap.png'), dpi=300)
        plt.close()
    
    print(f"Normalized visualizations saved to {output_dir}")

def main():
    """
    Main function to create normalized visualizations.
    """
    # Load results
    anthropic_df, deepseek_df, combined_df = load_results()
    
    # Create normalized visualizations
    if combined_df is not None:
        create_normalized_visualizations(
            anthropic_df,
            deepseek_df,
            combined_df,
            output_dir="reports/visualizations/normalized"
        )
    else:
        print("Cannot create normalized visualizations without both models' data")

if __name__ == "__main__":
    main()
