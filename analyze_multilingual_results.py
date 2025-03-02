"""
Analyze multilingual experiment results.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime

from analysis_methods.cross_validation import (
    perform_cross_validation,
    calculate_statistical_significance,
    validate_across_difficulty_levels
)

from analysis_methods.compression_metrics import (
    calculate_compression_ratio,
    analyze_token_usage_by_difficulty,
    calculate_normalized_compression_metrics,
    calculate_language_compression_index
)

from multilingual_visualizations import create_all_multilingual_visualizations

def load_latest_results(results_dir="experiment_results"):
    """
    Load the latest interim results file.
    
    Args:
        results_dir: Directory containing results files
        
    Returns:
        DataFrame with results
    """
    # Find all interim results files
    interim_files = [f for f in os.listdir(results_dir) if f.startswith("interim_results_")]
    
    if not interim_files:
        print("No interim results files found.")
        return None
    
    # Sort by test number
    interim_files.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]), reverse=True)
    
    # Load the latest file
    latest_file = os.path.join(results_dir, interim_files[0])
    print(f"Loading latest interim results from {latest_file}")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df

def analyze_language_efficiency(df):
    """
    Analyze language efficiency across different languages.
    
    Args:
        df: DataFrame with results
        
    Returns:
        Dictionary with analysis results
    """
    # Filter out error results
    if 'error' in df.columns:
        success_df = df[df['error'].isna()]
    else:
        success_df = df
    
    if success_df.empty:
        return {"error": "No successful results to analyze"}
    
    # Group by prompt type (language)
    language_tokens = success_df.groupby('prompt_type')['total_tokens'].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate efficiency relative to English
    if 'english' in language_tokens['prompt_type'].values:
        english_tokens = language_tokens.loc[language_tokens['prompt_type'] == 'english', 'mean'].values[0]
        
        language_tokens['token_reduction'] = english_tokens - language_tokens['mean']
        language_tokens['efficiency_percent'] = (language_tokens['token_reduction'] / english_tokens) * 100
        language_tokens['more_efficient_than_english'] = language_tokens['token_reduction'] > 0
    
    # Calculate statistical significance
    significance_results = calculate_statistical_significance(success_df)
    
    # Analyze by benchmark
    benchmark_analysis = {}
    for benchmark in success_df['benchmark'].unique():
        benchmark_df = success_df[success_df['benchmark'] == benchmark]
        
        # Group by prompt type
        benchmark_tokens = benchmark_df.groupby('prompt_type')['total_tokens'].agg(['mean', 'std', 'count']).reset_index()
        
        # Calculate efficiency relative to English
        if 'english' in benchmark_tokens['prompt_type'].values:
            english_tokens = benchmark_tokens.loc[benchmark_tokens['prompt_type'] == 'english', 'mean'].values[0]
            
            benchmark_tokens['token_reduction'] = english_tokens - benchmark_tokens['mean']
            benchmark_tokens['efficiency_percent'] = (benchmark_tokens['token_reduction'] / english_tokens) * 100
            benchmark_tokens['more_efficient_than_english'] = benchmark_tokens['token_reduction'] > 0
        
        benchmark_analysis[benchmark] = benchmark_tokens.to_dict(orient='records')
    
    # Analyze by difficulty
    difficulty_analysis = {}
    for difficulty in success_df['difficulty'].unique():
        difficulty_df = success_df[success_df['difficulty'] == difficulty]
        
        # Group by prompt type
        difficulty_tokens = difficulty_df.groupby('prompt_type')['total_tokens'].agg(['mean', 'std', 'count']).reset_index()
        
        # Calculate efficiency relative to English
        if 'english' in difficulty_tokens['prompt_type'].values:
            english_tokens = difficulty_tokens.loc[difficulty_tokens['prompt_type'] == 'english', 'mean'].values[0]
            
            difficulty_tokens['token_reduction'] = english_tokens - difficulty_tokens['mean']
            difficulty_tokens['efficiency_percent'] = (difficulty_tokens['token_reduction'] / english_tokens) * 100
            difficulty_tokens['more_efficient_than_english'] = difficulty_tokens['token_reduction'] > 0
        
        difficulty_analysis[difficulty] = difficulty_tokens.to_dict(orient='records')
    
    # Analyze by category
    category_analysis = {}
    for category in success_df['category'].unique():
        category_df = success_df[success_df['category'] == category]
        
        # Group by prompt type
        category_tokens = category_df.groupby('prompt_type')['total_tokens'].agg(['mean', 'std', 'count']).reset_index()
        
        # Calculate efficiency relative to English
        if 'english' in category_tokens['prompt_type'].values:
            english_tokens = category_tokens.loc[category_tokens['prompt_type'] == 'english', 'mean'].values[0]
            
            category_tokens['token_reduction'] = english_tokens - category_tokens['mean']
            category_tokens['efficiency_percent'] = (category_tokens['token_reduction'] / english_tokens) * 100
            category_tokens['more_efficient_than_english'] = category_tokens['token_reduction'] > 0
        
        category_analysis[category] = category_tokens.to_dict(orient='records')
    
    # Calculate language compression index
    lci_df = calculate_language_compression_index(success_df)
    
    return {
        'overall_language_efficiency': language_tokens.to_dict(orient='records'),
        'statistical_significance': significance_results,
        'benchmark_analysis': benchmark_analysis,
        'difficulty_analysis': difficulty_analysis,
        'category_analysis': category_analysis,
        'language_compression_index': lci_df.to_dict(orient='records') if not lci_df.empty else {}
    }

def identify_best_languages_by_domain(df):
    """
    Identify which languages perform best for specific problem domains.
    
    Args:
        df: DataFrame with results
        
    Returns:
        Dictionary with best languages by domain
    """
    # Filter out error results
    if 'error' in df.columns:
        success_df = df[df['error'].isna()]
    else:
        success_df = df
    
    if success_df.empty:
        return {"error": "No successful results to analyze"}
    
    # Analyze by benchmark
    best_by_benchmark = {}
    for benchmark in success_df['benchmark'].unique():
        benchmark_df = success_df[success_df['benchmark'] == benchmark]
        
        # Group by prompt type
        benchmark_tokens = benchmark_df.groupby('prompt_type')['total_tokens'].mean().reset_index()
        
        # Find the language with the lowest token usage
        best_language = benchmark_tokens.loc[benchmark_tokens['total_tokens'].idxmin()]
        
        # Calculate efficiency relative to English
        if 'english' in benchmark_tokens['prompt_type'].values:
            english_tokens = benchmark_tokens.loc[benchmark_tokens['prompt_type'] == 'english', 'total_tokens'].values[0]
            
            efficiency_percent = (english_tokens - best_language['total_tokens']) / english_tokens * 100
            
            best_by_benchmark[benchmark] = {
                'best_language': best_language['prompt_type'],
                'token_usage': best_language['total_tokens'],
                'efficiency_vs_english': efficiency_percent
            }
    
    # Analyze by category
    best_by_category = {}
    for category in success_df['category'].unique():
        category_df = success_df[success_df['category'] == category]
        
        # Group by prompt type
        category_tokens = category_df.groupby('prompt_type')['total_tokens'].mean().reset_index()
        
        # Find the language with the lowest token usage
        best_language = category_tokens.loc[category_tokens['total_tokens'].idxmin()]
        
        # Calculate efficiency relative to English
        if 'english' in category_tokens['prompt_type'].values:
            english_tokens = category_tokens.loc[category_tokens['prompt_type'] == 'english', 'total_tokens'].values[0]
            
            efficiency_percent = (english_tokens - best_language['total_tokens']) / english_tokens * 100
            
            best_by_category[category] = {
                'best_language': best_language['prompt_type'],
                'token_usage': best_language['total_tokens'],
                'efficiency_vs_english': efficiency_percent
            }
    
    # Analyze by difficulty
    best_by_difficulty = {}
    for difficulty in success_df['difficulty'].unique():
        difficulty_df = success_df[success_df['difficulty'] == difficulty]
        
        # Group by prompt type
        difficulty_tokens = difficulty_df.groupby('prompt_type')['total_tokens'].mean().reset_index()
        
        # Find the language with the lowest token usage
        best_language = difficulty_tokens.loc[difficulty_tokens['total_tokens'].idxmin()]
        
        # Calculate efficiency relative to English
        if 'english' in difficulty_tokens['prompt_type'].values:
            english_tokens = difficulty_tokens.loc[difficulty_tokens['prompt_type'] == 'english', 'total_tokens'].values[0]
            
            efficiency_percent = (english_tokens - best_language['total_tokens']) / english_tokens * 100
            
            best_by_difficulty[difficulty] = {
                'best_language': best_language['prompt_type'],
                'token_usage': best_language['total_tokens'],
                'efficiency_vs_english': efficiency_percent
            }
    
    return {
        'best_by_benchmark': best_by_benchmark,
        'best_by_category': best_by_category,
        'best_by_difficulty': best_by_difficulty
    }

def generate_strategic_language_selection_rules(df):
    """
    Generate rules for strategic language selection based on problem characteristics.
    
    Args:
        df: DataFrame with results
        
    Returns:
        Dictionary with language selection rules
    """
    # Filter out error results
    if 'error' in df.columns:
        success_df = df[df['error'].isna()]
    else:
        success_df = df
    
    if success_df.empty:
        return {"error": "No successful results to analyze"}
    
    # Get best languages by benchmark
    best_by_domain = identify_best_languages_by_domain(success_df)
    
    # Generate rules based on benchmark
    benchmark_rules = {}
    for benchmark, data in best_by_domain['best_by_benchmark'].items():
        if data['efficiency_vs_english'] > 0:
            benchmark_rules[benchmark] = data['best_language']
        else:
            benchmark_rules[benchmark] = 'english'
    
    # Generate rules based on category
    category_rules = {}
    for category, data in best_by_domain['best_by_category'].items():
        if data['efficiency_vs_english'] > 0:
            category_rules[category] = data['best_language']
        else:
            category_rules[category] = 'english'
    
    # Generate rules based on difficulty
    difficulty_rules = {}
    for difficulty, data in best_by_domain['best_by_difficulty'].items():
        if data['efficiency_vs_english'] > 0:
            difficulty_rules[difficulty] = data['best_language']
        else:
            difficulty_rules[difficulty] = 'english'
    
    # Generate combined rules
    combined_rules = {
        'benchmark_rules': benchmark_rules,
        'category_rules': category_rules,
        'difficulty_rules': difficulty_rules
    }
    
    return combined_rules

def main():
    """
    Main function to analyze multilingual results.
    """
    # Load latest results
    df = load_latest_results()
    
    if df is None:
        print("No results to analyze.")
        return
    
    print(f"Loaded {len(df)} results.")
    
    # Analyze language efficiency
    print("Analyzing language efficiency...")
    efficiency_results = analyze_language_efficiency(df)
    
    # Identify best languages by domain
    print("Identifying best languages by domain...")
    domain_results = identify_best_languages_by_domain(df)
    
    # Generate strategic language selection rules
    print("Generating strategic language selection rules...")
    selection_rules = generate_strategic_language_selection_rules(df)
    
    # Create visualizations
    print("Creating visualizations...")
    visualization_dir = "reports/visualizations/multilingual"
    os.makedirs(visualization_dir, exist_ok=True)
    
    create_all_multilingual_visualizations(df, visualization_dir)
    
    # Save analysis results
    print("Saving analysis results...")
    os.makedirs("analysis_results", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save overall language efficiency
    if 'overall_language_efficiency' in efficiency_results:
        overall_df = pd.DataFrame(efficiency_results['overall_language_efficiency'])
        overall_df.to_csv(f"analysis_results/overall_language_efficiency_{timestamp}.csv", index=False)
        print(f"Overall language efficiency saved to analysis_results/overall_language_efficiency_{timestamp}.csv")
    
    # Save benchmark analysis
    if 'benchmark_analysis' in efficiency_results:
        for benchmark, data in efficiency_results['benchmark_analysis'].items():
            benchmark_df = pd.DataFrame(data)
            benchmark_df.to_csv(f"analysis_results/benchmark_{benchmark}_{timestamp}.csv", index=False)
        print(f"Benchmark analysis saved to analysis_results/benchmark_*_{timestamp}.csv")
    
    # Save difficulty analysis
    if 'difficulty_analysis' in efficiency_results:
        for difficulty, data in efficiency_results['difficulty_analysis'].items():
            difficulty_df = pd.DataFrame(data)
            difficulty_df.to_csv(f"analysis_results/difficulty_{difficulty}_{timestamp}.csv", index=False)
        print(f"Difficulty analysis saved to analysis_results/difficulty_*_{timestamp}.csv")
    
    # Save category analysis
    if 'category_analysis' in efficiency_results:
        for category, data in efficiency_results['category_analysis'].items():
            category_df = pd.DataFrame(data)
            category_df.to_csv(f"analysis_results/category_{category}_{timestamp}.csv", index=False)
        print(f"Category analysis saved to analysis_results/category_*_{timestamp}.csv")
    
    # Save language compression index
    if 'language_compression_index' in efficiency_results and efficiency_results['language_compression_index']:
        lci_df = pd.DataFrame(efficiency_results['language_compression_index'])
        lci_df.to_csv(f"analysis_results/language_compression_index_{timestamp}.csv", index=False)
        print(f"Language compression index saved to analysis_results/language_compression_index_{timestamp}.csv")
    
    # Save best languages by domain
    if 'best_by_benchmark' in domain_results:
        best_benchmark = pd.DataFrame.from_dict(domain_results['best_by_benchmark'], orient='index')
        best_benchmark.to_csv(f"analysis_results/best_languages_by_benchmark_{timestamp}.csv")
        print(f"Best languages by benchmark saved to analysis_results/best_languages_by_benchmark_{timestamp}.csv")
    
    if 'best_by_category' in domain_results:
        best_category = pd.DataFrame.from_dict(domain_results['best_by_category'], orient='index')
        best_category.to_csv(f"analysis_results/best_languages_by_category_{timestamp}.csv")
        print(f"Best languages by category saved to analysis_results/best_languages_by_category_{timestamp}.csv")
    
    if 'best_by_difficulty' in domain_results:
        best_difficulty = pd.DataFrame.from_dict(domain_results['best_by_difficulty'], orient='index')
        best_difficulty.to_csv(f"analysis_results/best_languages_by_difficulty_{timestamp}.csv")
        print(f"Best languages by difficulty saved to analysis_results/best_languages_by_difficulty_{timestamp}.csv")
    
    # Save strategic language selection rules
    if 'benchmark_rules' in selection_rules:
        benchmark_rules = pd.DataFrame.from_dict(selection_rules['benchmark_rules'], orient='index', columns=['language'])
        benchmark_rules.to_csv(f"analysis_results/benchmark_selection_rules_{timestamp}.csv")
        print(f"Benchmark selection rules saved to analysis_results/benchmark_selection_rules_{timestamp}.csv")
    
    if 'category_rules' in selection_rules:
        category_rules = pd.DataFrame.from_dict(selection_rules['category_rules'], orient='index', columns=['language'])
        category_rules.to_csv(f"analysis_results/category_selection_rules_{timestamp}.csv")
        print(f"Category selection rules saved to analysis_results/category_selection_rules_{timestamp}.csv")
    
    if 'difficulty_rules' in selection_rules:
        difficulty_rules = pd.DataFrame.from_dict(selection_rules['difficulty_rules'], orient='index', columns=['language'])
        difficulty_rules.to_csv(f"analysis_results/difficulty_selection_rules_{timestamp}.csv")
        print(f"Difficulty selection rules saved to analysis_results/difficulty_selection_rules_{timestamp}.csv")
    
    print(f"Analysis results saved to analysis_results/ directory")
    
    # Print summary of results
    print("\nSummary of Results:")
    print("===================")
    
    # Overall language efficiency
    print("\nOverall Language Efficiency:")
    for lang in efficiency_results['overall_language_efficiency']:
        if lang['prompt_type'] == 'english':
            print(f"English (baseline): {lang['mean']:.2f} tokens")
        else:
            efficiency = lang.get('efficiency_percent', 0)
            print(f"{lang['prompt_type']}: {lang['mean']:.2f} tokens "
                  f"({efficiency:.2f}% {'more' if efficiency > 0 else 'less'} efficient than English)")
    
    # Best languages by benchmark
    print("\nBest Languages by Benchmark:")
    for benchmark, data in domain_results['best_by_benchmark'].items():
        print(f"{benchmark}: {data['best_language']} "
              f"({data['efficiency_vs_english']:.2f}% {'more' if data['efficiency_vs_english'] > 0 else 'less'} "
              f"efficient than English)")
    
    # Best languages by category
    print("\nBest Languages by Category:")
    for category, data in domain_results['best_by_category'].items():
        print(f"{category}: {data['best_language']} "
              f"({data['efficiency_vs_english']:.2f}% {'more' if data['efficiency_vs_english'] > 0 else 'less'} "
              f"efficient than English)")
    
    # Strategic language selection rules
    print("\nStrategic Language Selection Rules:")
    print("For benchmark-based selection:")
    for benchmark, language in selection_rules['benchmark_rules'].items():
        print(f"  - {benchmark}: Use {language}")
    
    print("For category-based selection:")
    for category, language in selection_rules['category_rules'].items():
        print(f"  - {category}: Use {language}")
    
    print("For difficulty-based selection:")
    for difficulty, language in selection_rules['difficulty_rules'].items():
        print(f"  - {difficulty}: Use {language}")
    
    return {
        'efficiency_results': efficiency_results,
        'domain_results': domain_results,
        'selection_rules': selection_rules
    }

if __name__ == "__main__":
    main()
