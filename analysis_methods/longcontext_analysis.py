"""
Analysis methods for long-context QA language efficiency testing.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

def analyze_context_length_impact(results_df: pd.DataFrame, context_lengths: Dict[str, int]) -> Dict[str, Any]:
    """
    Analyze the impact of context length on language efficiency.
    
    Args:
        results_df: DataFrame containing test results
        context_lengths: Dictionary mapping problem IDs to context lengths
        
    Returns:
        Dictionary containing analysis results
    """
    # Create a new column for context length
    results_df['context_length'] = results_df['problem_id'].map(context_lengths)
    
    # Calculate correlation between context length and token usage by language
    correlations = {}
    for lang in results_df['prompt_type'].unique():
        lang_df = results_df[results_df['prompt_type'] == lang]
        if not lang_df.empty:
            corr = lang_df.groupby('problem_id')['total_tokens'].mean().corr(
                pd.Series(context_lengths.values(), index=context_lengths.keys())
            )
            correlations[lang] = corr
    
    # Calculate efficiency relative to English by context length
    english_df = results_df[results_df['prompt_type'] == 'english']
    efficiency_by_length = {}
    
    # Define context length bins
    bins = [0, 5000, 10000, float('inf')]
    labels = ['medium', 'long', 'very_long']
    
    results_df['length_category'] = pd.cut(
        results_df['context_length'], 
        bins=bins, 
        labels=labels
    )
    
    for lang in results_df['prompt_type'].unique():
        if lang == 'english':
            continue
            
        efficiency_by_length[lang] = {}
        
        for length_cat in labels:
            eng_tokens = english_df[english_df['length_category'] == length_cat]['total_tokens'].mean()
            lang_tokens = results_df[
                (results_df['prompt_type'] == lang) & 
                (results_df['length_category'] == length_cat)
            ]['total_tokens'].mean()
            
            if not pd.isna(eng_tokens) and not pd.isna(lang_tokens):
                efficiency = (eng_tokens - lang_tokens) / eng_tokens * 100
                efficiency_by_length[lang][length_cat] = efficiency
    
    # Calculate context-to-reasoning ratio
    context_reasoning_ratio = {}
    for lang in results_df['prompt_type'].unique():
        lang_df = results_df[results_df['prompt_type'] == lang]
        if not lang_df.empty:
            # Estimate context tokens (input) vs reasoning tokens (output)
            context_tokens = lang_df['prompt_tokens'].mean()
            reasoning_tokens = lang_df['completion_tokens'].mean()
            if reasoning_tokens > 0:
                ratio = context_tokens / reasoning_tokens
                context_reasoning_ratio[lang] = ratio
    
    # Calculate information extraction efficiency
    # (How many output tokens are generated per input token)
    extraction_efficiency = {}
    for lang in results_df['prompt_type'].unique():
        lang_df = results_df[results_df['prompt_type'] == lang]
        if not lang_df.empty:
            input_tokens = lang_df['prompt_tokens'].mean()
            output_tokens = lang_df['completion_tokens'].mean()
            if input_tokens > 0:
                efficiency = output_tokens / input_tokens
                extraction_efficiency[lang] = efficiency
    
    return {
        'correlations': correlations,
        'efficiency_by_length': efficiency_by_length,
        'context_reasoning_ratio': context_reasoning_ratio,
        'extraction_efficiency': extraction_efficiency
    }

def create_context_length_visualizations(results_df: pd.DataFrame, context_lengths: Dict[str, int], output_dir: str):
    """
    Create visualizations for context length impact on language efficiency.
    
    Args:
        results_df: DataFrame containing test results
        context_lengths: Dictionary mapping problem IDs to context lengths
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Add context length to results
    results_df['context_length'] = results_df['problem_id'].map(context_lengths)
    
    # 1. Scatter plot of context length vs. token usage by language
    plt.figure(figsize=(12, 8))
    for lang in results_df['prompt_type'].unique():
        lang_df = results_df[results_df['prompt_type'] == lang]
        plt.scatter(
            lang_df['context_length'], 
            lang_df['total_tokens'],
            label=lang,
            alpha=0.7
        )
        
        # Add trend line
        if not lang_df.empty:
            z = np.polyfit(lang_df['context_length'], lang_df['total_tokens'], 1)
            p = np.poly1d(z)
            plt.plot(
                sorted(lang_df['context_length']),
                p(sorted(lang_df['context_length'])),
                linestyle='--'
            )
    
    plt.xlabel('Context Length (characters)')
    plt.ylabel('Total Tokens')
    plt.title('Context Length vs. Token Usage by Language')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/context_length_vs_tokens.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bar chart of efficiency by context length category
    # Define context length bins
    bins = [0, 5000, 10000, float('inf')]
    labels = ['Medium (0-5K)', 'Long (5K-10K)', 'Very Long (10K+)']
    
    results_df['length_category'] = pd.cut(
        results_df['context_length'], 
        bins=bins, 
        labels=labels
    )
    
    # Calculate efficiency relative to English by context length category
    efficiency_data = []
    
    for lang in results_df['prompt_type'].unique():
        if lang == 'english':
            continue
            
        for length_cat in labels:
            eng_tokens = results_df[
                (results_df['prompt_type'] == 'english') & 
                (results_df['length_category'] == length_cat)
            ]['total_tokens'].mean()
            
            lang_tokens = results_df[
                (results_df['prompt_type'] == lang) & 
                (results_df['length_category'] == length_cat)
            ]['total_tokens'].mean()
            
            if not pd.isna(eng_tokens) and not pd.isna(lang_tokens):
                efficiency = (eng_tokens - lang_tokens) / eng_tokens * 100
                efficiency_data.append({
                    'language': lang,
                    'length_category': length_cat,
                    'efficiency': efficiency
                })
    
    efficiency_df = pd.DataFrame(efficiency_data)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='language', y='efficiency', hue='length_category', data=efficiency_df)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Language')
    plt.ylabel('Efficiency Relative to English (%)')
    plt.title('Language Efficiency by Context Length Category')
    plt.legend(title='Context Length')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/efficiency_by_context_length.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap of efficiency by language and context length
    if not efficiency_df.empty:
        pivot_df = efficiency_df.pivot(index='language', columns='length_category', values='efficiency')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
        plt.title('Efficiency Heatmap by Language and Context Length')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/efficiency_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Context-to-reasoning ratio by language
    context_reasoning_data = []
    
    for lang in results_df['prompt_type'].unique():
        lang_df = results_df[results_df['prompt_type'] == lang]
        if not lang_df.empty:
            # Calculate context tokens (input) vs reasoning tokens (output)
            for length_cat in labels:
                cat_df = lang_df[lang_df['length_category'] == length_cat]
                if not cat_df.empty:
                    context_tokens = cat_df['prompt_tokens'].mean()
                    reasoning_tokens = cat_df['completion_tokens'].mean()
                    if reasoning_tokens > 0:
                        ratio = context_tokens / reasoning_tokens
                        context_reasoning_data.append({
                            'language': lang,
                            'length_category': length_cat,
                            'ratio': ratio
                        })
    
    ratio_df = pd.DataFrame(context_reasoning_data)
    
    if not ratio_df.empty:
        plt.figure(figsize=(14, 8))
        sns.barplot(x='language', y='ratio', hue='length_category', data=ratio_df)
        plt.xlabel('Language')
        plt.ylabel('Context-to-Reasoning Ratio')
        plt.title('Context-to-Reasoning Ratio by Language and Context Length')
        plt.legend(title='Context Length')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/context_reasoning_ratio.png', dpi=300, bbox_inches='tight')
        plt.close()

def analyze_longcontext_strategic_selection(results_df: pd.DataFrame, context_lengths: Dict[str, int]) -> Dict[str, Any]:
    """
    Analyze the effectiveness of strategic language selection for long-context QA.
    
    Args:
        results_df: DataFrame containing test results
        context_lengths: Dictionary mapping problem IDs to context lengths
        
    Returns:
        Dictionary containing analysis results
    """
    # Add context length to results
    results_df['context_length'] = results_df['problem_id'].map(context_lengths)
    
    # Define context length bins
    bins = [0, 5000, 10000, float('inf')]
    labels = ['medium', 'long', 'very_long']
    
    results_df['length_category'] = pd.cut(
        results_df['context_length'], 
        bins=bins, 
        labels=labels
    )
    
    # Calculate efficiency of strategic selection vs. fixed languages
    strategic_df = results_df[results_df['prompt_type'] == 'strategic']
    english_df = results_df[results_df['prompt_type'] == 'english']
    
    strategic_vs_fixed = {}
    
    # Overall efficiency
    for lang in results_df['prompt_type'].unique():
        if lang == 'strategic':
            continue
            
        lang_df = results_df[results_df['prompt_type'] == lang]
        lang_tokens = lang_df['total_tokens'].mean()
        strategic_tokens = strategic_df['total_tokens'].mean()
        
        if not pd.isna(lang_tokens) and not pd.isna(strategic_tokens):
            efficiency = (lang_tokens - strategic_tokens) / lang_tokens * 100
            strategic_vs_fixed[lang] = efficiency
    
    # Efficiency by context length
    strategic_by_length = {}
    
    for length_cat in labels:
        strategic_by_length[length_cat] = {}
        
        strategic_tokens = strategic_df[strategic_df['length_category'] == length_cat]['total_tokens'].mean()
        
        for lang in results_df['prompt_type'].unique():
            if lang == 'strategic':
                continue
                
            lang_tokens = results_df[
                (results_df['prompt_type'] == lang) & 
                (results_df['length_category'] == length_cat)
            ]['total_tokens'].mean()
            
            if not pd.isna(lang_tokens) and not pd.isna(strategic_tokens):
                efficiency = (lang_tokens - strategic_tokens) / lang_tokens * 100
                strategic_by_length[length_cat][lang] = efficiency
    
    return {
        'strategic_vs_fixed': strategic_vs_fixed,
        'strategic_by_length': strategic_by_length
    }

def create_strategic_selection_visualizations(results_df: pd.DataFrame, context_lengths: Dict[str, int], output_dir: str):
    """
    Create visualizations for strategic language selection in long-context QA.
    
    Args:
        results_df: DataFrame containing test results
        context_lengths: Dictionary mapping problem IDs to context lengths
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Add context length to results
    results_df['context_length'] = results_df['problem_id'].map(context_lengths)
    
    # Define context length bins
    bins = [0, 5000, 10000, float('inf')]
    labels = ['Medium (0-5K)', 'Long (5K-10K)', 'Very Long (10K+)']
    
    results_df['length_category'] = pd.cut(
        results_df['context_length'], 
        bins=bins, 
        labels=labels
    )
    
    # 1. Bar chart of strategic vs. fixed languages
    strategic_df = results_df[results_df['prompt_type'] == 'strategic']
    
    comparison_data = []
    
    for lang in results_df['prompt_type'].unique():
        if lang == 'strategic':
            continue
            
        for length_cat in labels:
            lang_tokens = results_df[
                (results_df['prompt_type'] == lang) & 
                (results_df['length_category'] == length_cat)
            ]['total_tokens'].mean()
            
            strategic_tokens = strategic_df[strategic_df['length_category'] == length_cat]['total_tokens'].mean()
            
            if not pd.isna(lang_tokens) and not pd.isna(strategic_tokens):
                efficiency = (lang_tokens - strategic_tokens) / lang_tokens * 100
                comparison_data.append({
                    'fixed_language': lang,
                    'length_category': length_cat,
                    'efficiency_gain': efficiency
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='fixed_language', y='efficiency_gain', hue='length_category', data=comparison_df)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Fixed Language')
    plt.ylabel('Efficiency Gain of Strategic Selection (%)')
    plt.title('Strategic Language Selection vs. Fixed Languages by Context Length')
    plt.legend(title='Context Length')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/strategic_vs_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Line plot of token usage by context length
    plt.figure(figsize=(12, 8))
    
    for lang in results_df['prompt_type'].unique():
        # Group by context length and calculate mean token usage
        lang_df = results_df[results_df['prompt_type'] == lang]
        if not lang_df.empty:
            grouped = lang_df.groupby('context_length')['total_tokens'].mean().reset_index()
            plt.plot(grouped['context_length'], grouped['total_tokens'], marker='o', label=lang)
    
    plt.xlabel('Context Length (characters)')
    plt.ylabel('Average Token Usage')
    plt.title('Token Usage by Context Length and Language')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/token_usage_by_context_length.png', dpi=300, bbox_inches='tight')
    plt.close()
