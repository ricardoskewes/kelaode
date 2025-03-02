"""
Create visualizations comparing Anthropic and Deepseek model results.
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
            
            # Filter for long-context QA problems only
            anthropic_df = pd.DataFrame(anthropic_results)
            anthropic_df = anthropic_df[anthropic_df['problem_id'].str.startswith('longqa_')]
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

def create_model_comparison_visualizations(anthropic_df, deepseek_df, combined_df, output_dir):
    """
    Create visualizations comparing Anthropic and Deepseek models.
    
    Args:
        anthropic_df: DataFrame with Anthropic results
        deepseek_df: DataFrame with Deepseek results
        combined_df: DataFrame with combined results
        output_dir: Directory to save visualizations to
    """
    print("Creating model comparison visualizations...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have both models' data
    if combined_df is None:
        print("Cannot create model comparison visualizations without both models' data")
        return
    
    # 1. Token usage by model and language
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=combined_df,
        x='prompt_type',
        y='total_tokens',
        hue='model',
        errorbar=('ci', 95),
        palette='viridis'
    )
    plt.title('Token Usage by Model and Language', fontsize=16)
    plt.xlabel('Language', fontsize=14)
    plt.ylabel('Average Token Usage', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'token_usage_by_model_language.png'), dpi=300)
    plt.close()
    
    # 2. Efficiency relative to English by model
    plt.figure(figsize=(12, 8))
    
    # Calculate efficiency for each model
    model_efficiency = {}
    for model in combined_df['model'].unique():
        model_df = combined_df[combined_df['model'] == model]
        english_tokens = model_df[model_df['prompt_type'] == 'english']['total_tokens'].mean()
        
        efficiencies = []
        languages = []
        
        for lang in model_df['prompt_type'].unique():
            if lang == 'english':
                continue
            
            lang_tokens = model_df[model_df['prompt_type'] == lang]['total_tokens'].mean()
            efficiency = (english_tokens - lang_tokens) / english_tokens * 100
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
        
        plt.title('Efficiency Relative to English by Model', fontsize=16)
        plt.xlabel('Language', fontsize=14)
        plt.ylabel('Efficiency (% token reduction)', fontsize=14)
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'efficiency_by_model.png'), dpi=300)
        plt.close()
    
    # 3. Model comparison heatmap
    plt.figure(figsize=(12, 10))
    
    # Prepare data for heatmap
    pivot_data = combined_df.pivot_table(
        index='prompt_type',
        columns='model',
        values='total_tokens',
        aggfunc='mean'
    )
    
    # Calculate percentage difference
    model_cols = pivot_data.columns
    if len(model_cols) >= 2:
        diff_pct = (pivot_data[model_cols[0]] - pivot_data[model_cols[1]]) / pivot_data[model_cols[1]] * 100
        pivot_data['diff_pct'] = diff_pct
        
        # Create custom colormap (blue for negative, red for positive)
        colors = ['blue', 'white', 'red']
        cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=256)
        
        # Plot heatmap
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.1f',
            cmap=cmap,
            center=0,
            linewidths=.5,
            cbar_kws={'label': 'Token Usage / % Difference'}
        )
        
        plt.title('Model Comparison: Token Usage by Language', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison_heatmap.png'), dpi=300)
        plt.close()
    
    # 4. Radar chart comparing models
    # Calculate average metrics by model
    metrics = {}
    
    for model in combined_df['model'].unique():
        model_df = combined_df[combined_df['model'] == model]
        
        # Calculate metrics
        avg_tokens = model_df['total_tokens'].mean()
        english_tokens = model_df[model_df['prompt_type'] == 'english']['total_tokens'].mean()
        
        # Calculate efficiency for non-English languages
        non_english_df = model_df[model_df['prompt_type'] != 'english']
        non_english_tokens = non_english_df['total_tokens'].mean()
        efficiency = (english_tokens - non_english_tokens) / english_tokens * 100
        
        # Calculate strategic efficiency
        strategic_tokens = model_df[model_df['prompt_type'] == 'strategic']['total_tokens'].mean()
        strategic_efficiency = (english_tokens - strategic_tokens) / english_tokens * 100
        
        # Calculate Chinese efficiency
        chinese_tokens = model_df[model_df['prompt_type'] == 'chinese']['total_tokens'].mean()
        chinese_efficiency = (english_tokens - chinese_tokens) / english_tokens * 100
        
        metrics[model] = {
            'Avg Tokens (lower is better)': avg_tokens,
            'Non-English Efficiency': efficiency,
            'Strategic Efficiency': strategic_efficiency,
            'Chinese Efficiency': chinese_efficiency
        }
    
    if len(metrics) >= 2:
        # Prepare data for radar chart
        categories = list(next(iter(metrics.values())).keys())
        
        # Normalize metrics for radar chart (0-1 scale, higher is better)
        normalized_metrics = {}
        
        for category in categories:
            values = [metrics[model][category] for model in metrics]
            
            if 'Tokens' in category:
                # For token counts, lower is better, so invert the normalization
                min_val = min(values)
                max_val = max(values)
                range_val = max_val - min_val
                
                if range_val > 0:
                    normalized_metrics[category] = {
                        model: 1 - (metrics[model][category] - min_val) / range_val
                        for model in metrics
                    }
                else:
                    normalized_metrics[category] = {model: 0.5 for model in metrics}
            else:
                # For efficiency metrics, higher is better
                min_val = min(values)
                max_val = max(values)
                range_val = max_val - min_val
                
                if range_val > 0:
                    normalized_metrics[category] = {
                        model: (metrics[model][category] - min_val) / range_val
                        for model in metrics
                    }
                else:
                    normalized_metrics[category] = {model: 0.5 for model in metrics}
        
        # Create radar chart
        plt.figure(figsize=(10, 10))
        
        # Set up the radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        ax = plt.subplot(111, polar=True)
        
        # Add category labels
        plt.xticks(angles[:-1], categories, size=12)
        
        # Plot each model
        for i, model in enumerate(metrics.keys()):
            values = [normalized_metrics[category][model] for category in categories]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title('Model Comparison: Performance Metrics', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_radar_chart.png'), dpi=300)
        plt.close()
    
    print(f"Model comparison visualizations saved to {output_dir}")

def create_deepseek_specific_visualizations(deepseek_df, output_dir):
    """
    Create visualizations specific to Deepseek model results.
    
    Args:
        deepseek_df: DataFrame with Deepseek results
        output_dir: Directory to save visualizations to
    """
    print("Creating Deepseek-specific visualizations...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if deepseek_df is None or len(deepseek_df) == 0:
        print("No Deepseek data available for visualizations")
        return
    
    # 1. Token usage by language
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=deepseek_df,
        x='prompt_type',
        y='total_tokens',
        errorbar=('ci', 95),
        palette='viridis'
    )
    plt.title('Deepseek: Token Usage by Language', fontsize=16)
    plt.xlabel('Language', fontsize=14)
    plt.ylabel('Average Token Usage', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deepseek_token_usage.png'), dpi=300)
    plt.close()
    
    # 2. Efficiency relative to English
    plt.figure(figsize=(10, 6))
    
    # Calculate efficiency
    english_tokens = deepseek_df[deepseek_df['prompt_type'] == 'english']['total_tokens'].mean()
    
    efficiencies = []
    languages = []
    
    for lang in deepseek_df['prompt_type'].unique():
        if lang == 'english':
            continue
        
        lang_tokens = deepseek_df[deepseek_df['prompt_type'] == lang]['total_tokens'].mean()
        efficiency = (english_tokens - lang_tokens) / english_tokens * 100
        efficiencies.append(efficiency)
        languages.append(lang)
    
    # Create DataFrame for plotting
    efficiency_df = pd.DataFrame({
        'language': languages,
        'efficiency': efficiencies
    })
    
    # Sort by efficiency
    efficiency_df = efficiency_df.sort_values('efficiency', ascending=False)
    
    # Create the plot
    sns.barplot(
        data=efficiency_df,
        x='language',
        y='efficiency',
        palette='viridis'
    )
    
    plt.title('Deepseek: Efficiency Relative to English', fontsize=16)
    plt.xlabel('Language', fontsize=14)
    plt.ylabel('Efficiency (% token reduction)', fontsize=14)
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deepseek_efficiency.png'), dpi=300)
    plt.close()
    
    # 3. Token usage by problem
    plt.figure(figsize=(12, 8))
    
    # Calculate average tokens by problem and language
    problem_tokens = deepseek_df.groupby(['problem_id', 'prompt_type'])['total_tokens'].mean().reset_index()
    
    # Create the plot
    sns.barplot(
        data=problem_tokens,
        x='problem_id',
        y='total_tokens',
        hue='prompt_type',
        palette='viridis'
    )
    
    plt.title('Deepseek: Token Usage by Problem and Language', fontsize=16)
    plt.xlabel('Problem ID', fontsize=14)
    plt.ylabel('Average Token Usage', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Language')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deepseek_problem_tokens.png'), dpi=300)
    plt.close()
    
    print(f"Deepseek-specific visualizations saved to {output_dir}")

def main():
    """
    Main function to create visualizations.
    """
    # Load results
    anthropic_df, deepseek_df, combined_df = load_results()
    
    # Create visualizations
    if deepseek_df is not None:
        create_deepseek_specific_visualizations(
            deepseek_df,
            output_dir="reports/visualizations/deepseek"
        )
    
    if combined_df is not None:
        create_model_comparison_visualizations(
            anthropic_df,
            deepseek_df,
            combined_df,
            output_dir="reports/visualizations/model_comparison"
        )
    else:
        print("Cannot create model comparison visualizations without both models' data")

if __name__ == "__main__":
    main()
