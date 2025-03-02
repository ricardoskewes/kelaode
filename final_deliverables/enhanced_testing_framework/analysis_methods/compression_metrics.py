"""
Compression metrics for language efficiency analysis.
"""

import numpy as np
import pandas as pd
import re
from collections import Counter

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
