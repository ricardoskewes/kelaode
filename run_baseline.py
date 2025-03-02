"""
Script to run baseline experiments for language efficiency testing.
"""

import os
import sys
from enhanced_experiment_runner import EnhancedLanguageEfficiencyTest
from enhanced_benchmarks import MATH_PROBLEMS, BBH_PROBLEMS, HOTPOTQA_PROBLEMS, ARC_PROBLEMS, GSM8K_PROBLEMS
from utils.json_utils import save_json

def main():
    """
    Run baseline experiments with a subset of problems.
    """
    # Ensure API key is set
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs("experiment_results", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("reports/visualizations", exist_ok=True)
    
    # Create a subset of problems for faster testing
    # Select 1 problem from each benchmark category
    subset_problems = (
        MATH_PROBLEMS[:1] +
        BBH_PROBLEMS[:1] +
        HOTPOTQA_PROBLEMS[:1] +
        ARC_PROBLEMS[:1] +
        GSM8K_PROBLEMS[:1]
    )
    
    print(f"Running baseline experiments with {len(subset_problems)} problems...")
    
    # Initialize test runner with subset of problems
    test = EnhancedLanguageEfficiencyTest(problems=subset_problems)
    
    # Run baseline tests with repetitions
    print("Running baseline tests...")
    test.run_all_tests(
        repetitions=2,  # Run each test 2 times for reliability
        prompt_types=["english", "chinese"],  # Basic comparison
        save_interim=True
    )
    
    # Save baseline results
    baseline_files = test.save_results("baseline_results")
    
    # Analyze baseline results
    print("Analyzing baseline results...")
    baseline_analysis = test.analyze_results()
    
    # Save baseline analysis using custom JSON encoder
    save_json(baseline_analysis, "reports/baseline_analysis.json")
    
    # Print summary of baseline findings
    if "prompt_type_comparisons" in baseline_analysis:
        comp = baseline_analysis["prompt_type_comparisons"].get("english_vs_chinese", {})
        if "token_reduction_percent" in comp:
            print(f"\nSUMMARY OF BASELINE FINDINGS:")
            print(f"Chinese reasoning used {comp.get('token_reduction_percent', 0):.2f}% fewer tokens than English")
            print(f"English average tokens: {comp.get('english_avg_tokens', 0):.2f}")
            print(f"Chinese average tokens: {comp.get('chinese_avg_tokens', 0):.2f}")
            
            if comp.get("is_chinese_more_efficient", False):
                print("\nChinese reasoning appears more token-efficient in this experiment.")
            else:
                print("\nChinese reasoning does NOT appear more token-efficient in this experiment.")

if __name__ == "__main__":
    main()
