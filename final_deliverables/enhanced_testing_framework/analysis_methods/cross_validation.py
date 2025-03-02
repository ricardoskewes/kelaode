"""
Cross-test validation methods for language efficiency analysis.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from scipy import stats

def perform_cross_validation(results_df, target_column='total_tokens', n_splits=5):
    """
    Perform cross-validation to validate the consistency of results.
    
    Args:
        results_df: DataFrame containing experiment results
        target_column: Column to use as the target variable
        n_splits: Number of splits for K-fold cross-validation
        
    Returns:
        Dictionary with cross-validation metrics
    """
    # Check if we have enough samples for cross-validation
    if len(results_df) < 10:
        print(f"Warning: Not enough samples for reliable cross-validation. Using basic statistics instead.")
        # Return basic statistics instead
        return {
            'mean': results_df[target_column].mean(),
            'std': results_df[target_column].std(),
            'median': results_df[target_column].median(),
            'min': results_df[target_column].min(),
            'max': results_df[target_column].max(),
            'sample_size': len(results_df),
            'note': 'Sample size too small for cross-validation'
        }
    
    # Adjust n_splits if necessary
    actual_n_splits = min(n_splits, len(results_df) // 2)
    if actual_n_splits < n_splits:
        print(f"Warning: Reducing n_splits from {n_splits} to {actual_n_splits} due to small sample size.")
    
    # Prepare data for cross-validation
    # We'll use prompt_type, benchmark, and difficulty as features
    X = pd.get_dummies(results_df[['prompt_type', 'benchmark', 'difficulty']])
    y = results_df[target_column]
    
    # Initialize K-fold cross-validation
    kf = KFold(n_splits=actual_n_splits, shuffle=True, random_state=42)
    
    # Use a simple linear regression model for validation
    model = LinearRegression()
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    
    # Convert negative MSE to RMSE
    rmse_scores = np.sqrt(-cv_scores)
    
    return {
        'mean_rmse': rmse_scores.mean(),
        'std_rmse': rmse_scores.std(),
        'cv_scores': rmse_scores.tolist(),
        'n_splits': actual_n_splits
    }

def calculate_statistical_significance(results_df, group_column='prompt_type', 
                                      value_column='total_tokens'):
    """
    Calculate statistical significance between different prompt types.
    
    Args:
        results_df: DataFrame containing experiment results
        group_column: Column to group by (usually 'prompt_type')
        value_column: Column with values to compare
        
    Returns:
        Dictionary with p-values and significance indicators
    """
    # Get unique groups
    groups = results_df[group_column].unique()
    
    # Create a dictionary to store results
    significance_results = {}
    
    # Perform pairwise t-tests
    for i, group1 in enumerate(groups):
        for group2 in groups[i+1:]:
            # Get values for each group
            values1 = results_df[results_df[group_column] == group1][value_column]
            values2 = results_df[results_df[group_column] == group2][value_column]
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(values1, values2, equal_var=False)
            
            # Store results
            key = f"{group1}_vs_{group2}"
            significance_results[key] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_at_0.05': p_value < 0.05,
                'significant_at_0.01': p_value < 0.01,
                'mean_difference': values1.mean() - values2.mean(),
                'percent_difference': (values1.mean() - values2.mean()) / values1.mean() * 100
            }
    
    return significance_results

def validate_across_difficulty_levels(results_df, prompt_types=['english', 'chinese']):
    """
    Validate if the efficiency difference is consistent across difficulty levels.
    
    Args:
        results_df: DataFrame containing experiment results
        prompt_types: List of prompt types to compare
        
    Returns:
        Dictionary with validation results by difficulty
    """
    difficulty_levels = results_df['difficulty'].unique()
    validation_results = {}
    
    for difficulty in difficulty_levels:
        # Filter data for current difficulty
        difficulty_df = results_df[results_df['difficulty'] == difficulty]
        
        # Calculate metrics for each prompt type
        metrics = {}
        for prompt_type in prompt_types:
            prompt_df = difficulty_df[difficulty_df['prompt_type'] == prompt_type]
            if not prompt_df.empty:
                metrics[prompt_type] = {
                    'mean_tokens': prompt_df['total_tokens'].mean(),
                    'std_tokens': prompt_df['total_tokens'].std(),
                    'sample_size': len(prompt_df),
                }
        
        # Calculate differences between prompt types
        differences = {}
        for i, type1 in enumerate(prompt_types):
            for type2 in prompt_types[i+1:]:
                if type1 in metrics and type2 in metrics:
                    diff_key = f"{type1}_vs_{type2}"
                    token_diff = metrics[type1]['mean_tokens'] - metrics[type2]['mean_tokens']
                    percent_diff = (token_diff / metrics[type1]['mean_tokens']) * 100
                    
                    differences[diff_key] = {
                        'absolute_difference': token_diff,
                        'percent_difference': percent_diff,
                        'is_type1_more_efficient': token_diff < 0,
                        'is_type2_more_efficient': token_diff > 0
                    }
        
        validation_results[difficulty] = {
            'metrics_by_prompt_type': metrics,
            'differences': differences
        }
    
    return validation_results
