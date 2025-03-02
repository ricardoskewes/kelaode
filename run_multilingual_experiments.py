"""
Run expanded language experiments with multiple languages and models.
"""

import os
import json
import time
import pandas as pd
from enhanced_experiment_runner import EnhancedLanguageEfficiencyTest
from enhanced_benchmarks import ENHANCED_BENCHMARK_PROBLEMS
from multilingual_visualizations import create_all_multilingual_visualizations

def run_multilingual_experiments(repetitions=3, output_file="experiment_results/multilingual_results.json"):
    """
    Run expanded language experiments with multiple languages and models.
    
    Args:
        repetitions: Number of repetitions for each test
        output_file: File to save results to
    """
    print("Starting multilingual experiments...")
    
    # Check available API keys
    models = []
    
    # Add Anthropic models if API key is available
    if os.environ.get("ANTHROPIC_API_KEY"):
        models.append("anthropic:claude-3-5-sonnet-20240620")
    else:
        print("ANTHROPIC_API_KEY not set. Skipping Anthropic models.")
    
    # Add Deepseek models if API key is available
    if os.environ.get("DEEPSEEK_API_KEY"):
        models.append("deepseek:deepseek-chat")
        models.append("deepseek:deepseek-coder")
    else:
        print("DEEPSEEK_API_KEY not set. Skipping Deepseek models.")
    
    if not models:
        print("No API keys available. Cannot run experiments.")
        return
    
    # Initialize test runner with all models
    test = EnhancedLanguageEfficiencyTest(
        problems=ENHANCED_BENCHMARK_PROBLEMS,
        models=models
    )
    
    # Define languages to test
    languages = [
        "english",      # Baseline
        "chinese",      # Logographic
        "finnish",      # Agglutinative
        "german",       # Germanic
        "japanese",     # Mixed logographic/syllabic
        "korean",       # Featural
        "russian",      # Cyrillic
        "arabic",       # Abjad
        "strategic"     # Dynamic language selection
    ]
    
    # Run tests for all languages with repetitions
    test.run_all_tests(
        prompt_types=languages,
        repetitions=repetitions
    )
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    test.save_results(output_file)
    
    print(f"Multilingual experiments completed. Results saved to {output_file}")
    
    # Load results for analysis
    with open(output_file, 'r') as f:
        results = json.load(f)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Create visualizations
    visualization_dir = "reports/visualizations/multilingual"
    os.makedirs(visualization_dir, exist_ok=True)
    
    create_all_multilingual_visualizations(df, visualization_dir)
    
    print(f"Multilingual visualizations created and saved to {visualization_dir}")
    
    return df

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("experiment_results", exist_ok=True)
    
    # Run experiments with 3 repetitions
    df = run_multilingual_experiments(
        repetitions=3,
        output_file="experiment_results/multilingual_results.json"
    )
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("===================")
    
    # Overall token usage by language
    print("\nAverage Token Usage by Language:")
    language_tokens = df.groupby('prompt_type')['total_tokens'].mean().sort_values()
    for lang, tokens in language_tokens.items():
        print(f"{lang}: {tokens:.2f} tokens")
    
    # Efficiency relative to English
    print("\nEfficiency Relative to English:")
    english_tokens = df[df['prompt_type'] == 'english']['total_tokens'].mean()
    for lang, tokens in language_tokens.items():
        if lang == 'english':
            continue
        efficiency = (english_tokens - tokens) / english_tokens * 100
        print(f"{lang}: {efficiency:.2f}% {'more' if efficiency > 0 else 'less'} efficient")
    
    # Model comparison
    print("\nModel Comparison:")
    model_tokens = df.groupby('model')['total_tokens'].mean().sort_values()
    for model, tokens in model_tokens.items():
        print(f"{model}: {tokens:.2f} tokens")
    
    # Chinese efficiency by model
    print("\nChinese Efficiency by Model:")
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        english_tokens = model_df[model_df['prompt_type'] == 'english']['total_tokens'].mean()
        chinese_tokens = model_df[model_df['prompt_type'] == 'chinese']['total_tokens'].mean()
        
        if pd.isna(english_tokens) or pd.isna(chinese_tokens):
            continue
            
        efficiency = (english_tokens - chinese_tokens) / english_tokens * 100
        print(f"{model}: {efficiency:.2f}% {'more' if efficiency > 0 else 'less'} efficient")
