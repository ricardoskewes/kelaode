"""
Test script for long-context QA language efficiency testing.
"""

import os
import json
import time
import pandas as pd
from enhanced_experiment_runner import EnhancedLanguageEfficiencyTest
from enhanced_benchmarks.LongContextQA.longcontext_problems import LONGCONTEXT_PROBLEMS

def run_longcontext_qa_test(repetitions=2, output_file="experiment_results/longcontext_qa_results.json"):
    """
    Run long-context QA language efficiency tests.
    
    Args:
        repetitions: Number of repetitions for each test
        output_file: File to save results to
    """
    print("Starting long-context QA language efficiency tests...")
    
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
    else:
        print("DEEPSEEK_API_KEY not set. Skipping Deepseek models.")
    
    if not models:
        print("No API keys available. Cannot run experiments.")
        return
    
    # Initialize test runner with long-context QA problems
    test = EnhancedLanguageEfficiencyTest(
        problems=LONGCONTEXT_PROBLEMS,
        models=models
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
    
    print(f"Long-context QA tests completed. Results saved to {output_file}")
    
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
    
    # Context length impact
    print("\nContext Length Impact:")
    for lang in languages:
        lang_df = df[df['prompt_type'] == lang]
        if lang_df.empty:
            continue
        
        # Calculate correlation between context length and token usage
        context_lengths = [problem.context_length for problem in LONGCONTEXT_PROBLEMS]
        correlation = lang_df.groupby('problem_id')['total_tokens'].mean().corr(pd.Series(context_lengths, index=[problem.id for problem in LONGCONTEXT_PROBLEMS]))
        print(f"{lang}: correlation between context length and token usage = {correlation:.2f}")
    
    return df

if __name__ == "__main__":
    # Run long-context QA tests
    df = run_longcontext_qa_test(
        repetitions=2,
        output_file="experiment_results/longcontext_qa_results.json"
    )
