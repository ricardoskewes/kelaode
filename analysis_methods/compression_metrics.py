"""
Compression metrics for language efficiency analysis.
"""

import numpy as np
import pandas as pd
import re
from collections import Counter
from scipy.stats import entropy

def calculate_compression_ratio(english_text, chinese_text, english_tokens, chinese_tokens):
    """
    Calculate compression ratio between English and Chinese texts.
    
    Args:
        english_text: Text in English
        chinese_text: Text in Chinese
        english_tokens: Number of tokens in English text
        chinese_tokens: Number of tokens in Chinese text
        
    Returns:
        Dictionary with compression metrics
    """
    # Calculate character counts
    english_chars = len(english_text)
    chinese_chars = len(chinese_text)
    
    # Calculate token efficiency
    english_chars_per_token = english_chars / english_tokens if english_tokens > 0 else 0
    chinese_chars_per_token = chinese_chars / chinese_tokens if chinese_tokens > 0 else 0
    
    # Calculate token compression ratio
    token_compression_ratio = english_tokens / chinese_tokens if chinese_tokens > 0 else 0
    
    # Calculate character compression ratio
    char_compression_ratio = english_chars / chinese_chars if chinese_chars > 0 else 0
    
    # Calculate information density ratio (assuming equal information content)
    info_density_ratio = (english_chars_per_token / chinese_chars_per_token 
                          if chinese_chars_per_token > 0 else 0)
    
    return {
        'english_chars': english_chars,
        'chinese_chars': chinese_chars,
        'english_chars_per_token': english_chars_per_token,
        'chinese_chars_per_token': chinese_chars_per_token,
        'token_compression_ratio': token_compression_ratio,
        'char_compression_ratio': char_compression_ratio,
        'info_density_ratio': info_density_ratio
    }

def analyze_token_usage_by_difficulty(results_df):
    """
    Analyze token usage across different problem difficulty levels.
    
    Args:
        results_df: DataFrame containing experiment results
        
    Returns:
        Dictionary with token usage analysis by difficulty
    """
    # Group by difficulty and prompt_type
    grouped = results_df.groupby(['difficulty', 'prompt_type']).agg({
        'total_tokens': ['mean', 'std', 'count'],
        'input_tokens': ['mean'],
        'output_tokens': ['mean']
    }).reset_index()
    
    # Flatten the multi-level columns
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    
    # Rename the columns to ensure consistent naming
    column_mapping = {
        'difficulty_': 'difficulty',
        'prompt_type_': 'prompt_type'
    }
    grouped = grouped.rename(columns=column_mapping)
    
    # Calculate efficiency gain for each difficulty level
    difficulty_levels = results_df['difficulty'].unique()
    prompt_types = results_df['prompt_type'].unique()
    
    efficiency_by_difficulty = {}
    
    for difficulty in difficulty_levels:
        difficulty_data = grouped[grouped['difficulty'] == difficulty]
        
        # Create a dictionary for each prompt type
        prompt_data = {}
        for prompt_type in prompt_types:
            prompt_row = difficulty_data[difficulty_data['prompt_type'] == prompt_type]
            if not prompt_row.empty:
                prompt_data[prompt_type] = {
                    'mean_tokens': prompt_row['total_tokens_mean'].values[0],
                    'std_tokens': prompt_row['total_tokens_std'].values[0],
                    'sample_size': prompt_row['total_tokens_count'].values[0],
                    'input_tokens': prompt_row['input_tokens_mean'].values[0],
                    'output_tokens': prompt_row['output_tokens_mean'].values[0]
                }
        
        # Calculate efficiency comparisons between prompt types
        comparisons = {}
        for i, type1 in enumerate(prompt_types):
            for type2 in prompt_types[i+1:]:
                if type1 in prompt_data and type2 in prompt_data:
                    type1_tokens = prompt_data[type1]['mean_tokens']
                    type2_tokens = prompt_data[type2]['mean_tokens']
                    
                    token_diff = type1_tokens - type2_tokens
                    percent_diff = (token_diff / type1_tokens) * 100 if type1_tokens > 0 else 0
                    
                    comparisons[f"{type1}_vs_{type2}"] = {
                        'absolute_difference': token_diff,
                        'percent_difference': percent_diff,
                        f"{type2}_more_efficient": token_diff > 0,
                        'efficiency_gain': percent_diff if token_diff > 0 else -percent_diff
                    }
        
        efficiency_by_difficulty[difficulty] = {
            'prompt_data': prompt_data,
            'comparisons': comparisons
        }
    
    return {
        'grouped_data': grouped.to_dict(orient='records'),
        'efficiency_by_difficulty': efficiency_by_difficulty
    }

def calculate_normalized_compression_metrics(results_df):
    """
    Calculate normalized compression metrics that account for problem complexity.
    
    Args:
        results_df: DataFrame containing experiment results
        
    Returns:
        DataFrame with normalized compression metrics
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df = results_df.copy()
    
    # Calculate average tokens per problem for normalization
    problem_avg_tokens = df.groupby('problem_id')['total_tokens'].mean()
    
    # Add problem complexity factor (based on average tokens across all prompt types)
    df['problem_complexity'] = df['problem_id'].map(problem_avg_tokens)
    
    # Calculate normalized tokens (tokens / problem complexity)
    df['normalized_tokens'] = df['total_tokens'] / df['problem_complexity']
    
    # Calculate normalized compression ratio for Chinese vs English
    # First, create a pivot table with prompt types as columns
    pivot_df = df.pivot_table(
        index='problem_id',
        columns='prompt_type',
        values=['total_tokens', 'normalized_tokens']
    )
    
    # Flatten the column names
    pivot_df.columns = ['_'.join(col).strip('_') for col in pivot_df.columns.values]
    
    # Calculate compression ratios where both English and Chinese results exist
    compression_metrics = []
    
    for problem_id in pivot_df.index:
        row = pivot_df.loc[problem_id]
        
        metrics = {'problem_id': problem_id}
        
        # Check if we have data for both English and Chinese
        if ('total_tokens_english' in row and not pd.isna(row['total_tokens_english']) and
            'total_tokens_chinese' in row and not pd.isna(row['total_tokens_chinese'])):
            
            # Raw compression ratio
            metrics['raw_compression_ratio'] = (
                row['total_tokens_english'] / row['total_tokens_chinese']
                if row['total_tokens_chinese'] > 0 else 0
            )
            
            # Normalized compression ratio
            metrics['normalized_compression_ratio'] = (
                row['normalized_tokens_english'] / row['normalized_tokens_chinese']
                if row['normalized_tokens_chinese'] > 0 else 0
            )
            
            # Efficiency gain percentage
            metrics['efficiency_gain_percent'] = (
                (row['total_tokens_english'] - row['total_tokens_chinese']) / 
                row['total_tokens_english'] * 100
                if row['total_tokens_english'] > 0 else 0
            )
            
            compression_metrics.append(metrics)
    
    return pd.DataFrame(compression_metrics)

def calculate_lci_weights(
    token_efficiency,
    information_density_ratio,
    character_efficiency,
    context_length_efficiency=None
):
    """
    Calculate the Language Compression Index (LCI) based on provided metrics.
    
    Args:
        token_efficiency: Token efficiency relative to English
        information_density_ratio: Information density ratio relative to English
        character_efficiency: Character efficiency relative to English
        context_length_efficiency: Context length efficiency relative to English (optional)
        
    Returns:
        Language Compression Index (LCI)
    """
    # If context length efficiency is provided, include it in the calculation
    if context_length_efficiency is not None:
        return (
            0.5 * token_efficiency +
            0.2 * information_density_ratio +
            0.1 * character_efficiency +
            0.2 * context_length_efficiency
        )
    else:
        # Use the original calculation
        return (
            0.6 * token_efficiency +
            0.3 * information_density_ratio +
            0.1 * character_efficiency
        )

def calculate_language_compression_index(results_df, context_length_efficiencies=None):
    """
    Calculate a comprehensive Language Compression Index (LCI) for multiple languages.
    The LCI quantifies how efficiently each language encodes information in tokens
    compared to English, accounting for token usage, character count, and information content.
    
    Args:
        results_df: DataFrame containing experiment results
        context_length_efficiencies: Optional dictionary mapping languages to context length
                                    efficiency values relative to English
        
    Returns:
        DataFrame with Language Compression Index for each language
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df = results_df.copy()
    
    # Get unique languages (prompt types)
    languages = df['prompt_type'].unique()
    
    # Get English data as baseline
    english_df = df[df['prompt_type'] == 'english']
    if english_df.empty:
        raise ValueError("No English data found for baseline comparison")
    
    english_tokens_mean = english_df['total_tokens'].mean()
    english_bits_per_token = english_df['response_bits_per_token'].mean() if 'response_bits_per_token' in english_df.columns else 0
    english_chars_per_token = english_df['response_chars_per_token'].mean() if 'response_chars_per_token' in english_df.columns else 0
    
    # Calculate Language Compression Index for each language
    lci_data = []
    
    for lang in languages:
        # Skip English as it's the baseline
        if lang == 'english':
            continue
        
        lang_df = df[df['prompt_type'] == lang]
        if lang_df.empty:
            continue
        
        # Calculate metrics
        lang_tokens_mean = lang_df['total_tokens'].mean()
        lang_bits_per_token = lang_df['response_bits_per_token'].mean() if 'response_bits_per_token' in lang_df.columns else 0
        lang_chars_per_token = lang_df['response_chars_per_token'].mean() if 'response_chars_per_token' in lang_df.columns else 0
        
        # Calculate token efficiency (ratio of English tokens to language tokens)
        # Higher values mean the language is more token-efficient
        token_efficiency = english_tokens_mean / lang_tokens_mean if lang_tokens_mean > 0 else 0
        
        # Calculate information density ratio (bits per token relative to English)
        # Higher values mean the language encodes more information per token
        info_density_ratio = lang_bits_per_token / english_bits_per_token if english_bits_per_token > 0 else 0
        
        # Calculate character efficiency (characters per token relative to English)
        # Higher values mean the language uses more characters per token
        char_efficiency = lang_chars_per_token / english_chars_per_token if english_chars_per_token > 0 else 0
        
        # Get context length efficiency if provided
        context_length_efficiency = None
        if context_length_efficiencies and lang in context_length_efficiencies:
            context_length_efficiency = context_length_efficiencies[lang]
        
        # Calculate Language Compression Index using the weighted formula
        lci = calculate_lci_weights(
            token_efficiency,
            info_density_ratio,
            char_efficiency,
            context_length_efficiency
        )
        
        # Calculate token reduction percentage
        token_reduction = ((english_tokens_mean - lang_tokens_mean) / english_tokens_mean) * 100 if english_tokens_mean > 0 else 0
        
        # Store results
        lci_data.append({
            'language': lang,
            'lci': lci,
            'token_efficiency': token_efficiency,
            'info_density_ratio': info_density_ratio,
            'char_efficiency': char_efficiency,
            'context_length_efficiency': context_length_efficiency,
            'token_reduction_percent': token_reduction,
            'english_tokens_mean': english_tokens_mean,
            'lang_tokens_mean': lang_tokens_mean
        })
    
    # Create DataFrame and sort by LCI
    lci_df = pd.DataFrame(lci_data).sort_values('lci', ascending=False)
    
    return lci_df

def calculate_multilingual_compression_metrics(results_df):
    """
    Calculate compression metrics for multiple languages compared to English.
    
    Args:
        results_df: DataFrame containing experiment results
        
    Returns:
        DataFrame with compression metrics for each language
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df = results_df.copy()
    
    # Get unique languages (prompt types)
    languages = df['prompt_type'].unique()
    
    # Get English data as baseline
    english_df = df[df['prompt_type'] == 'english']
    if english_df.empty:
        raise ValueError("No English data found for baseline comparison")
    
    # Calculate compression metrics for each language
    compression_metrics = []
    
    for lang in languages:
        # Skip English as it's the baseline
        if lang == 'english':
            continue
        
        lang_df = df[df['prompt_type'] == lang]
        if lang_df.empty:
            continue
        
        # Group by benchmark to calculate benchmark-specific metrics
        benchmarks = df['benchmark'].unique()
        benchmark_metrics = {}
        
        for benchmark in benchmarks:
            english_benchmark = english_df[english_df['benchmark'] == benchmark]
            lang_benchmark = lang_df[lang_df['benchmark'] == benchmark]
            
            if english_benchmark.empty or lang_benchmark.empty:
                continue
            
            english_tokens = english_benchmark['total_tokens'].mean()
            lang_tokens = lang_benchmark['total_tokens'].mean()
            
            # Calculate efficiency gain for this benchmark
            efficiency_gain = ((english_tokens - lang_tokens) / english_tokens) * 100 if english_tokens > 0 else 0
            
            benchmark_metrics[benchmark] = {
                'english_tokens': english_tokens,
                'lang_tokens': lang_tokens,
                'efficiency_gain': efficiency_gain
            }
        
        # Calculate overall metrics
        english_tokens_mean = english_df['total_tokens'].mean()
        lang_tokens_mean = lang_df['total_tokens'].mean()
        
        # Calculate token reduction percentage
        token_reduction = ((english_tokens_mean - lang_tokens_mean) / english_tokens_mean) * 100 if english_tokens_mean > 0 else 0
        
        # Calculate compression ratio
        compression_ratio = english_tokens_mean / lang_tokens_mean if lang_tokens_mean > 0 else 0
        
        # Store results
        metrics = {
            'language': lang,
            'english_tokens_mean': english_tokens_mean,
            'lang_tokens_mean': lang_tokens_mean,
            'token_reduction_percent': token_reduction,
            'compression_ratio': compression_ratio,
            'benchmark_metrics': benchmark_metrics
        }
        
        compression_metrics.append(metrics)
    
    return compression_metrics
