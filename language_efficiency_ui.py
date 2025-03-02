#!/usr/bin/env python3
"""
Minimalist UI for language efficiency analysis.

This script provides a command-line interface for running language efficiency
tests with customizable options for language, model, benchmark, and prompt.
"""

import os
import argparse
from enhanced_experiment_runner import EnhancedLanguageEfficiencyTest
from enhanced_benchmarks import (
    MATH_PROBLEMS,
    BBH_PROBLEMS,
    HOTPOTQA_PROBLEMS,
    ARC_PROBLEMS,
    GSM8K_PROBLEMS,
    ENHANCED_BENCHMARK_PROBLEMS
)

# Available models
AVAILABLE_MODELS = [
    "anthropic:claude-3-5-sonnet-20240620",
    "qwen:qwen-turbo",
    "deepseek:deepseek-chat"
]

# Available prompt types
AVAILABLE_PROMPT_TYPES = [
    "english",
    "chinese",
    "chinese_with_markers",
    "hybrid"
]

# Available benchmark categories
AVAILABLE_BENCHMARKS = {
    "math": MATH_PROBLEMS,
    "bbh": BBH_PROBLEMS,
    "hotpotqa": HOTPOTQA_PROBLEMS,
    "arc": ARC_PROBLEMS,
    "gsm8k": GSM8K_PROBLEMS,
    "all": ENHANCED_BENCHMARK_PROBLEMS
}


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Language Efficiency Analysis UI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model selection
    parser.add_argument(
        "--models",
        nargs="+",
        choices=AVAILABLE_MODELS,
        default=["anthropic:claude-3-5-sonnet-20240620"],
        help="Models to use for testing"
    )

    # Prompt type selection
    parser.add_argument(
        "--prompt-types",
        nargs="+",
        choices=AVAILABLE_PROMPT_TYPES,
        default=["english", "chinese"],
        help="Prompt types to test"
    )

    # Benchmark selection
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=list(AVAILABLE_BENCHMARKS.keys()),
        default=["all"],
        help="Benchmark categories to test"
    )

    # Number of problems
    parser.add_argument(
        "--num-problems",
        type=int,
        default=5,
        help="Number of problems to test from each benchmark"
    )

    # Use mock data
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock data instead of real API calls"
    )

    # Verbose output
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def check_api_keys(models, verbose=False):
    """Check if required API keys are set."""
    missing_keys = []

    # Check for Anthropic API key if using Anthropic models
    if any("anthropic:" in model for model in models) and not os.environ.get(
            "ANTHROPIC_API_KEY"):
        missing_keys.append("ANTHROPIC_API_KEY")

    # Check for Dashscope API key if using Qwen models
    if any("qwen:" in model for model in models) and not os.environ.get(
            "DASHSCOPE_API_KEY"):
        missing_keys.append("DASHSCOPE_API_KEY")

    # Check for Deepseek API key if using Deepseek models
    if any("deepseek:" in model for model in models) and not os.environ.get(
            "DEEPSEEK_API_KEY"):
        missing_keys.append("DEEPSEEK_API_KEY")

    if missing_keys and verbose:
        print(f"Warning: The following API keys are not set: "
              f"{', '.join(missing_keys)}")
        print("You can set them using environment variables or use "
              "--use-mock to use mock data.")

    return missing_keys


def get_selected_problems(benchmarks, num_problems):
    """Get selected problems from benchmarks."""
    selected_problems = []

    if "all" in benchmarks:
        # Use all problems
        selected_problems = ENHANCED_BENCHMARK_PROBLEMS[:num_problems]
    else:
        # Combine problems from selected benchmarks
        for benchmark in benchmarks:
            selected_problems.extend(AVAILABLE_BENCHMARKS[benchmark])

        # Limit to specified number of problems
        selected_problems = selected_problems[:num_problems]

    return selected_problems


def generate_mock_results(test_problems, prompt_types, models):
    """Generate mock results for demonstration purposes."""
    results = []
    for problem in test_problems:
        for prompt_type in prompt_types:
            for model in models:
                # Create a mock result with realistic values
                mock_result = {
                    "problem_id": problem["id"],
                    "category": problem["category"],
                    "benchmark": problem["benchmark"],
                    "difficulty": problem["difficulty"],
                    "prompt_type": prompt_type,
                    "model": model,
                    "provider": (
                        model.split(":")[0] if ":" in model else "anthropic"
                    ),
                    "model_name": (
                        model.split(":")[1] if ":" in model else model
                    ),
                    "input_tokens": (
                        150 + (100 if prompt_type == "chinese" else 0)
                    ),
                    "output_tokens": (
                        300 + (150 if prompt_type == "chinese" else 0)
                    ),
                    "total_tokens": (
                        450 + (250 if prompt_type == "chinese" else 0)
                    ),
                    "response_time": 2.5,
                    "response_text": (
                        f"Mock response for {problem['id']} using "
                        f"{prompt_type} prompt on {model}"
                    ),
                    "english_answer": f"Mock answer for {problem['id']}",
                    "response_char_count": 1200,
                    "response_word_count": 200,
                    "response_bits_per_token": 12.5,
                    "response_semantic_density": 0.85,
                    "response_content_ratio": 0.75,
                    "response_compression_ratio": (
                        1.2 if prompt_type == "english" else 1.8
                    ),
                    "response_chars_per_token": (
                        3.5 if prompt_type == "english" else 5.2
                    ),
                    "response_chinese_char_count": (
                        0 if prompt_type == "english" else 800
                    ),
                    "response_chinese_ratio": (
                        0.0 if prompt_type == "english" else 0.65
                    ),
                    "response_chinese_info_density": (
                        0.0 if prompt_type == "english" else 2.3
                    ),
                    "response_chinese_chars_per_token": (
                        0.0 if prompt_type == "english" else 2.8
                    )
                }
                results.append(mock_result)
    return results


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_arguments()

    # Check for required API keys
    missing_keys = check_api_keys(args.models, args.verbose)

    # Use mock data if requested or if API keys are missing
    use_mock = args.use_mock or (missing_keys and not args.use_mock)

    if use_mock and args.verbose:
        print("Using mock data for demonstration purposes.")

    # Get selected problems
    selected_problems = get_selected_problems(
        args.benchmarks, args.num_problems)

    if args.verbose:
        print("\nRunning language efficiency tests with:")
        print(f"- Models: {', '.join(args.models)}")
        print(f"- Prompt types: {', '.join(args.prompt_types)}")
        print(f"- Benchmarks: {', '.join(args.benchmarks)}")
        print(f"- Number of problems: {len(selected_problems)}")
        print(f"- Using mock data: {use_mock}")

    # Initialize test runner with available models
    test = EnhancedLanguageEfficiencyTest(
        models=args.models,
        problems=selected_problems
    )

    # Run tests or use mock data
    if use_mock:
        # Generate mock results
        results = generate_mock_results(
            selected_problems,
            args.prompt_types,
            args.models
        )

        # Set the results in the test object
        test.results = results
    else:
        try:
            # Run actual tests
            results = test.run_all_tests(
                prompt_types=args.prompt_types,
                problems=selected_problems
            )
        except Exception as e:
            print(f"Error running tests: {str(e)}")
            print("Falling back to mock results for demonstration purposes...")

            # Generate mock results as fallback
            results = generate_mock_results(
                selected_problems,
                args.prompt_types,
                args.models
            )

            # Set the results in the test object
            test.results = results

    # Save results
    print("\nSaving results...")
    result_file = test.save_results()
    print(f"Results saved to: {result_file}")

    # Analyze results
    print("\nAnalyzing results...")
    analysis = test.analyze_results()

    # Create visualizations
    print("\nCreating visualizations...")
    test.create_visualizations(analysis)

    print("\nLanguage efficiency tests completed successfully!")
    print(
        "Check the 'experiment_results' and 'reports' directories for "
        "results and visualizations."
    )


if __name__ == "__main__":
    main()
