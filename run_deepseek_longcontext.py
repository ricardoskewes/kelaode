"""
Run long-context QA experiments with Deepseek models.
"""

import os
import json
import time
import pandas as pd
from enhanced_experiment_runner import EnhancedLanguageEfficiencyTest
from enhanced_benchmarks.LongContextQA.longcontext_problems import LONGCONTEXT_PROBLEMS

def run_deepseek_longcontext_experiments(repetitions=2, output_file="experiment_results/deepseek_longcontext_results.json"):
    """
    Run long-context QA language efficiency tests with Deepseek models.
    
    Args:
        repetitions: Number of repetitions for each test
        output_file: File to save results to
    """
    print("Starting Deepseek long-context QA language efficiency tests...")
    
    # Check if Deepseek API key is available
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("DEEPSEEK_API_KEY not set. Cannot run Deepseek experiments.")
        return None
    
    # Initialize test runner with long-context QA problems and Deepseek model
    test = EnhancedLanguageEfficiencyTest(
        problems=LONGCONTEXT_PROBLEMS,
        models=["deepseek:deepseek-chat"]
    )
    
    # Define languages to test
    languages = [
        "english",      # Baseline
        "chinese",      # Logographic
        "german",       # Germanic
        "russian",      # Cyrillic
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
    
    print(f"Deepseek long-context QA tests completed. Results saved to {output_file}")
    
    # Load results for analysis
    with open(output_file, 'r') as f:
        results = json.load(f)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
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
    
    return df

def combine_results_with_anthropic():
    """
    Combine Deepseek results with existing Anthropic results.
    """
    print("\nCombining Deepseek results with Anthropic results...")
    
    # Load Deepseek results
    try:
        with open("experiment_results/deepseek_longcontext_results.json", 'r') as f:
            deepseek_results = json.load(f)
        
        # Load latest Anthropic interim results
        import glob
        interim_files = glob.glob("experiment_results/interim_results_*.json")
        latest_file = max(interim_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        with open(latest_file, 'r') as f:
            anthropic_results = json.load(f)
        
        # Combine results
        combined_results = anthropic_results + deepseek_results
        
        # Save combined results
        with open("experiment_results/combined_longcontext_results.json", 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        print(f"Combined results saved to experiment_results/combined_longcontext_results.json")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(combined_results)
        
        # Print summary statistics by model
        print("\nSummary Statistics by Model:")
        print("===========================")
        
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            print(f"\nModel: {model}")
            
            # Overall token usage by language
            print("\nAverage Token Usage by Language:")
            language_tokens = model_df.groupby('prompt_type')['total_tokens'].mean().sort_values()
            for lang, tokens in language_tokens.items():
                print(f"{lang}: {tokens:.2f} tokens")
            
            # Efficiency relative to English
            print("\nEfficiency Relative to English:")
            english_tokens = model_df[model_df['prompt_type'] == 'english']['total_tokens'].mean()
            for lang, tokens in language_tokens.items():
                if lang == 'english':
                    continue
                efficiency = (english_tokens - tokens) / english_tokens * 100
                print(f"{lang}: {efficiency:.2f}% {'more' if efficiency > 0 else 'less'} efficient")
        
        return df
    except Exception as e:
        print(f"Error combining results: {str(e)}")
        return None

if __name__ == "__main__":
    # Run Deepseek long-context experiments
    deepseek_df = run_deepseek_longcontext_experiments(
        repetitions=2,
        output_file="experiment_results/deepseek_longcontext_results.json"
    )
    
    # Combine with Anthropic results
    if deepseek_df is not None:
        combined_df = combine_results_with_anthropic()
