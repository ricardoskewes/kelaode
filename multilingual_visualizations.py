"""
Enhanced visualization methods for multilingual language efficiency analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

def create_language_efficiency_index_chart(df, output_dir="reports/visualizations/multilingual"):
    """
    Create a bar chart showing the Language Efficiency Index for each language.
    
    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate Language Efficiency Index (LEI)
    # LEI = (1 / normalized_tokens) * (bits_per_token / english_bits_per_token)
    
    # Get English bits per token as baseline
    english_bits = df[df['prompt_type'] == 'english']['response_bits_per_token'].mean()
    english_tokens = df[df['prompt_type'] == 'english']['total_tokens'].mean()
    
    # Calculate LEI for each language
    lei_data = []
    
    for lang in df['prompt_type'].unique():
        lang_df = df[df['prompt_type'] == lang]
        if lang_df.empty:
            continue
            
        avg_tokens = lang_df['total_tokens'].mean()
        avg_bits = lang_df['response_bits_per_token'].mean() if 'response_bits_per_token' in lang_df.columns else 0
        
        # Normalize tokens relative to English
        normalized_tokens = avg_tokens / english_tokens if english_tokens > 0 else 1
        
        # Normalize bits per token relative to English
        normalized_bits = avg_bits / english_bits if english_bits > 0 else 1
        
        # Calculate LEI
        lei = (1 / normalized_tokens) * normalized_bits if normalized_tokens > 0 else 0
        
        lei_data.append({
            'language': lang,
            'lei': lei,
            'normalized_tokens': normalized_tokens,
            'normalized_bits': normalized_bits
        })
    
    # Create DataFrame
    lei_df = pd.DataFrame(lei_data)
    
    # Sort by LEI
    lei_df = lei_df.sort_values('lei', ascending=False)
    
    # Create bar chart
    plt.figure(figsize=(14, 8))
    bars = plt.bar(lei_df['language'], lei_df['lei'], color=sns.color_palette("viridis", len(lei_df)))
    
    # Add labels and title
    plt.xlabel('Language')
    plt.ylabel('Language Efficiency Index (LEI)')
    plt.title('Language Efficiency Index by Language')
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/language_efficiency_index.png')
    plt.close()
    
    return lei_df

def create_multilingual_character_ratio_chart(df, output_dir="reports/visualizations/multilingual"):
    """
    Create a stacked bar chart showing the ratio of different character types in each language.
    
    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get language-specific character ratios
    ratio_columns = [col for col in df.columns if col.endswith('_ratio') and col.startswith('response_')]
    
    if not ratio_columns:
        print("No language ratio columns found in DataFrame")
        return
    
    # Calculate average ratios by prompt type
    ratio_data = df.groupby('prompt_type')[ratio_columns].mean()
    
    # Rename columns for better readability
    ratio_data.columns = [col.replace('response_', '').replace('_ratio', '') for col in ratio_data.columns]
    
    # Create stacked bar chart
    plt.figure(figsize=(14, 8))
    ratio_data.plot(kind='bar', stacked=True, colormap='viridis')
    
    # Add labels and title
    plt.xlabel('Language')
    plt.ylabel('Character Ratio')
    plt.title('Character Type Distribution by Language')
    plt.xticks(rotation=45)
    plt.legend(title='Character Type')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/multilingual_character_ratio.png')
    plt.close()
    
    return ratio_data

def create_language_compression_radar_chart(df, output_dir="reports/visualizations/multilingual"):
    """
    Create a radar chart showing different compression metrics for each language.
    
    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate compression metrics for each language
    languages = df['prompt_type'].unique()
    
    # Metrics to include in radar chart
    metrics = [
        'token_efficiency',  # Inverse of normalized tokens
        'bits_per_token',    # Information density
        'chars_per_token',   # Character efficiency
        'content_ratio'      # Content word ratio
    ]
    
    # Calculate metrics for each language
    radar_data = {}
    
    # Get English values as baseline
    english_df = df[df['prompt_type'] == 'english']
    english_tokens = english_df['total_tokens'].mean()
    english_bits = english_df['response_bits_per_token'].mean() if 'response_bits_per_token' in english_df.columns else 1
    english_chars = english_df['response_chars_per_token'].mean() if 'response_chars_per_token' in english_df.columns else 1
    english_content = english_df['response_content_ratio'].mean() if 'response_content_ratio' in english_df.columns else 1
    
    for lang in languages:
        lang_df = df[df['prompt_type'] == lang]
        if lang_df.empty:
            continue
            
        # Calculate normalized metrics
        avg_tokens = lang_df['total_tokens'].mean()
        token_efficiency = english_tokens / avg_tokens if avg_tokens > 0 else 1
        
        avg_bits = lang_df['response_bits_per_token'].mean() if 'response_bits_per_token' in lang_df.columns else 0
        bits_ratio = avg_bits / english_bits if english_bits > 0 else 1
        
        avg_chars = lang_df['response_chars_per_token'].mean() if 'response_chars_per_token' in lang_df.columns else 0
        chars_ratio = avg_chars / english_chars if english_chars > 0 else 1
        
        avg_content = lang_df['response_content_ratio'].mean() if 'response_content_ratio' in lang_df.columns else 0
        content_ratio = avg_content / english_content if english_content > 0 else 1
        
        # Store metrics
        radar_data[lang] = {
            'token_efficiency': token_efficiency,
            'bits_per_token': bits_ratio,
            'chars_per_token': chars_ratio,
            'content_ratio': content_ratio
        }
    
    # Create radar chart
    # Number of metrics
    N = len(metrics)
    
    # Create a figure
    fig = plt.figure(figsize=(12, 10))
    
    # Create a radar chart
    ax = fig.add_subplot(111, polar=True)
    
    # Set the angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Set the labels for each metric
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # Draw the radar chart for each language
    colors = sns.color_palette("viridis", len(radar_data))
    
    for i, (lang, metrics_dict) in enumerate(radar_data.items()):
        values = [metrics_dict[metric] for metric in metrics]
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=lang, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Language Compression Metrics Radar Chart', size=15)
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/language_compression_radar.png')
    plt.close()
    
    return radar_data

def create_benchmark_heatmap(df, output_dir="reports/visualizations/multilingual"):
    """
    Create a heatmap showing the efficiency of each language for each benchmark.
    
    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate efficiency for each language and benchmark
    benchmarks = df['benchmark'].unique()
    languages = df['prompt_type'].unique()
    
    # Create a DataFrame to store efficiency values
    heatmap_data = pd.DataFrame(index=benchmarks, columns=languages)
    
    # Calculate efficiency relative to English for each benchmark
    for benchmark in benchmarks:
        benchmark_df = df[df['benchmark'] == benchmark]
        english_tokens = benchmark_df[benchmark_df['prompt_type'] == 'english']['total_tokens'].mean()
        
        for lang in languages:
            if lang == 'english':
                heatmap_data.loc[benchmark, lang] = 1.0  # English is the baseline
                continue
                
            lang_tokens = benchmark_df[benchmark_df['prompt_type'] == lang]['total_tokens'].mean()
            efficiency = english_tokens / lang_tokens if lang_tokens > 0 else 0
            heatmap_data.loc[benchmark, lang] = efficiency
    
    # Create heatmap
    plt.figure(figsize=(14, 10))
    
    # Create a custom colormap (green for better efficiency, red for worse)
    cmap = LinearSegmentedColormap.from_list('efficiency_cmap', ['#ff9999', '#ffffff', '#99ff99'], N=100)
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, cmap=cmap, center=1.0, fmt='.2f', linewidths=.5)
    
    # Add labels and title
    plt.xlabel('Language')
    plt.ylabel('Benchmark')
    plt.title('Language Efficiency by Benchmark (Relative to English)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/benchmark_language_efficiency_heatmap.png')
    plt.close()
    
    return heatmap_data

def create_language_efficiency_by_difficulty_chart(df, output_dir="reports/visualizations/multilingual"):
    """
    Create a grouped bar chart showing language efficiency by difficulty level.
    
    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate efficiency for each language and difficulty
    difficulties = df['difficulty'].unique()
    languages = df['prompt_type'].unique()
    
    # Create a DataFrame to store efficiency values
    efficiency_data = []
    
    # Calculate efficiency relative to English for each difficulty
    for difficulty in difficulties:
        difficulty_df = df[df['difficulty'] == difficulty]
        english_tokens = difficulty_df[difficulty_df['prompt_type'] == 'english']['total_tokens'].mean()
        
        for lang in languages:
            if lang == 'english':
                continue  # Skip English as it's the baseline
                
            lang_tokens = difficulty_df[difficulty_df['prompt_type'] == lang]['total_tokens'].mean()
            efficiency = (english_tokens - lang_tokens) / english_tokens * 100 if english_tokens > 0 else 0
            
            efficiency_data.append({
                'difficulty': difficulty,
                'language': lang,
                'efficiency_gain': efficiency
            })
    
    # Create DataFrame
    efficiency_df = pd.DataFrame(efficiency_data)
    
    # Create grouped bar chart
    plt.figure(figsize=(14, 8))
    
    # Create grouped bar chart
    sns.barplot(x='difficulty', y='efficiency_gain', hue='language', data=efficiency_df, palette='viridis')
    
    # Add labels and title
    plt.xlabel('Difficulty Level')
    plt.ylabel('Efficiency Gain vs. English (%)')
    plt.title('Language Efficiency Gain by Difficulty Level')
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Add legend
    plt.legend(title='Language', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/language_efficiency_by_difficulty.png')
    plt.close()
    
    return efficiency_df

def create_combined_language_efficiency_index(df, output_dir="reports/visualizations/multilingual"):
    """
    Create a bar chart showing the Combined Language Efficiency Index (CLEI) for each language.
    CLEI combines linguistic density and tokenization efficiency.
    
    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate Combined Language Efficiency Index (CLEI)
    # CLEI = (token_efficiency * 0.6) + (linguistic_density * 0.4)
    
    # Get English values as baseline
    english_df = df[df['prompt_type'] == 'english']
    english_tokens = english_df['total_tokens'].mean()
    english_bits = english_df['response_bits_per_token'].mean() if 'response_bits_per_token' in english_df.columns else 1
    
    # Calculate CLEI for each language
    clei_data = []
    
    for lang in df['prompt_type'].unique():
        lang_df = df[df['prompt_type'] == lang]
        if lang_df.empty:
            continue
            
        # Calculate token efficiency (relative to English)
        avg_tokens = lang_df['total_tokens'].mean()
        token_efficiency = english_tokens / avg_tokens if avg_tokens > 0 else 1
        
        # Calculate linguistic density (relative to English)
        avg_bits = lang_df['response_bits_per_token'].mean() if 'response_bits_per_token' in lang_df.columns else 0
        linguistic_density = avg_bits / english_bits if english_bits > 0 else 1
        
        # Calculate CLEI
        clei = (token_efficiency * 0.6) + (linguistic_density * 0.4)
        
        clei_data.append({
            'language': lang,
            'clei': clei,
            'token_efficiency': token_efficiency,
            'linguistic_density': linguistic_density
        })
    
    # Create DataFrame
    clei_df = pd.DataFrame(clei_data)
    
    # Sort by CLEI
    clei_df = clei_df.sort_values('clei', ascending=False)
    
    # Create bar chart
    plt.figure(figsize=(14, 8))
    
    # Create stacked bar chart
    clei_df.plot(x='language', y=['token_efficiency', 'linguistic_density'], kind='bar', stacked=False, 
                 color=['#3498db', '#2ecc71'], figsize=(14, 8))
    
    # Add CLEI as a line
    plt.plot(clei_df.index, clei_df['clei'], 'ro-', linewidth=2, markersize=8, label='CLEI')
    
    # Add labels and title
    plt.xlabel('Language')
    plt.ylabel('Index Value')
    plt.title('Combined Language Efficiency Index (CLEI)')
    plt.xticks(clei_df.index, clei_df['language'], rotation=45)
    
    # Add legend
    plt.legend(['CLEI', 'Token Efficiency', 'Linguistic Density'])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/combined_language_efficiency_index.png')
    plt.close()
    
    return clei_df

def create_all_multilingual_visualizations(df, output_dir="reports/visualizations/multilingual"):
    """
    Create all multilingual visualizations.
    
    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create all visualizations
    lei_df = create_language_efficiency_index_chart(df, output_dir)
    ratio_data = create_multilingual_character_ratio_chart(df, output_dir)
    radar_data = create_language_compression_radar_chart(df, output_dir)
    heatmap_data = create_benchmark_heatmap(df, output_dir)
    efficiency_df = create_language_efficiency_by_difficulty_chart(df, output_dir)
    clei_df = create_combined_language_efficiency_index(df, output_dir)
    
    print(f"All multilingual visualizations created and saved to {output_dir}")
    
    return {
        'lei_df': lei_df,
        'ratio_data': ratio_data,
        'radar_data': radar_data,
        'heatmap_data': heatmap_data,
        'efficiency_df': efficiency_df,
        'clei_df': clei_df
    }
