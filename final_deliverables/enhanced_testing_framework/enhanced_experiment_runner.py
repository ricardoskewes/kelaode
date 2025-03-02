"""
Enhanced experiment runner for language efficiency testing.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import anthropic
import matplotlib.pyplot as plt
from datetime import datetime

# Import enhanced benchmarks
from enhanced_benchmarks import ENHANCED_BENCHMARK_PROBLEMS

# Import analysis methods
from analysis_methods import (
    perform_cross_validation,
    calculate_statistical_significance,
    validate_across_difficulty_levels,
    calculate_bits_per_token,
    calculate_semantic_density,
    analyze_chinese_information_density,
    calculate_compression_ratio,
    analyze_token_usage_by_difficulty,
    calculate_normalized_compression_metrics
)

class EnhancedLanguageEfficiencyTest:
    """
    Enhanced experiment runner for testing language efficiency.
    """
    
    def __init__(self, problems=ENHANCED_BENCHMARK_PROBLEMS, 
                 models=["claude-3-5-sonnet-20240620"],
                 prompts=None):
        """
        Initialize the experiment runner.
        
        Args:
            problems: List of benchmark problems to test
            models: List of Anthropic models to test
            prompts: Dictionary of prompts to use (defaults to standard prompts if None)
        """
        self.problems = problems
        self.models = models
        self.results = []
        
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
        
        # Default prompts if none provided
        if prompts is None:
            self.prompts = {
                "english": "Think through this problem step by step showing your complete reasoning process. Be thorough and consider all aspects of the problem. Show all your work and then provide the final answer:",
                
                "chinese": """Please solve this problem by thinking ENTIRELY in Chinese. Your entire reasoning process must be in Chinese.

1. Read and understand the problem in English
2. Translate the problem to Chinese in your mind
3. Think through ALL steps of solving the problem IN CHINESE
4. Work through the complete solution IN CHINESE
5. Only at the very end, translate just the final answer to English

Show your entire reasoning process in Chinese characters. Only the final answer should be in English.""",
                
                "chinese_with_markers": """Please solve this problem by thinking entirely in Chinese, but add English markers to help track your reasoning.

1. Read and understand the problem
2. Start your solution with "[CHINESE REASONING BEGINS]"
3. Think through ALL steps IN CHINESE showing detailed work
4. End your Chinese reasoning with "[CHINESE REASONING ENDS]" 
5. Then provide ONLY the final answer in English with "[ENGLISH ANSWER]:" prefix

The majority of your response should be Chinese characters. Be thorough in your Chinese reasoning.""",
                
                "hybrid": """Please solve this problem using a hybrid approach:
                
1. Read and understand the problem in English
2. For mathematical notation, formulas, and specialized terms, use English
3. For all reasoning steps and explanations, use Chinese
4. Clearly mark the final answer in English with "[ANSWER]:" prefix

This approach allows you to leverage the precision of English for technical terms while using the efficiency of Chinese for reasoning."""
            }
        else:
            self.prompts = prompts
    
    def extract_english_answer(self, text, prompt_type):
        """
        Extract the final English answer from responses.
        
        Args:
            text: Response text
            prompt_type: Type of prompt used
            
        Returns:
            Extracted English answer
        """
        if prompt_type == "english":
            # For English, just use the last paragraph or sentence as the answer
            paragraphs = text.split('\n\n')
            return paragraphs[-1] if paragraphs else text
        
        elif prompt_type == "chinese":
            # Look for English text near the end of the response
            paragraphs = text.split('\n\n')
            for p in reversed(paragraphs):
                # If paragraph has mostly non-Chinese characters, it might be the answer
                if sum(1 for char in p if '\u4e00' <= char <= '\u9fff') / len(p) < 0.5:
                    return p
            return paragraphs[-1] if paragraphs else text
        
        elif prompt_type == "chinese_with_markers":
            # Look for the marked English answer
            import re
            match = re.search(r'\[ENGLISH ANSWER\]:(.*?)($|\n\n)', text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return "No marked English answer found"
        
        elif prompt_type == "hybrid":
            # Look for the marked answer in hybrid mode
            import re
            match = re.search(r'\[ANSWER\]:(.*?)($|\n\n)', text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return "No marked answer found"
        
        # Default fallback
        return text.split('\n')[-1] if '\n' in text else text
    
    def get_metrics(self, text, token_count):
        """
        Calculate various text metrics.
        
        Args:
            text: Text to analyze
            token_count: Number of tokens in the text
            
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        char_count = len(text)
        word_count = len(text.split())
        
        # Chinese character metrics
        chinese_char_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        non_chinese_char_count = char_count - chinese_char_count
        chinese_ratio = chinese_char_count / char_count if char_count > 0 else 0
        
        # Information density metrics
        bits_per_token = calculate_bits_per_token(text, token_count)
        
        # Semantic density metrics
        semantic_density = calculate_semantic_density(text, token_count)
        
        # Chinese information density metrics
        chinese_density = analyze_chinese_information_density(text, token_count)
        
        # Compression ratio estimate
        estimated_english_chars = chinese_char_count * 2.5 + non_chinese_char_count
        compression_ratio = char_count / estimated_english_chars if estimated_english_chars > 0 else 1
        
        return {
            "char_count": char_count,
            "word_count": word_count,
            "chinese_char_count": chinese_char_count,
            "chinese_ratio": chinese_ratio,
            "bits_per_token": bits_per_token,
            "semantic_density": semantic_density["semantic_density"],
            "content_ratio": semantic_density["content_ratio"],
            "chinese_info_density": chinese_density["chinese_info_density"],
            "chinese_chars_per_token": chinese_density["chinese_chars_per_token"],
            "compression_ratio": compression_ratio
        }
    
    def run_test(self, problem, prompt_type, model):
        """
        Run a single test case with the specified prompt type and model.
        
        Args:
            problem: Problem to test
            prompt_type: Type of prompt to use
            model: Model to use
            
        Returns:
            Dictionary with test results
        """
        prompt = self.prompts[prompt_type]
        full_prompt = f"{prompt}\n\n{problem['problem']}"
        
        start_time = time.time()
        
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
            response_text = response.content[0].text
            
            end_time = time.time()
            
            # Extract the English answer for evaluation
            english_answer = self.extract_english_answer(response_text, prompt_type)
            
            # Basic result data
            result = {
                "problem_id": problem["id"],
                "category": problem["category"],
                "benchmark": problem["benchmark"],
                "difficulty": problem["difficulty"],
                "prompt_type": prompt_type,
                "model": model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                "response_time": end_time - start_time,
                "response_text": response_text,
                "extracted_answer": english_answer,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add metrics for response
            metrics = self.get_metrics(response_text, response.usage.output_tokens)
            result.update({f"response_{k}": v for k, v in metrics.items()})
            
            return result
            
        except Exception as e:
            end_time = time.time()
            print(f"Error processing {problem['id']} with {prompt_type} on {model}: {str(e)}")
            
            # Return a partial result with error information
            return {
                "problem_id": problem["id"],
                "category": problem["category"],
                "benchmark": problem["benchmark"],
                "difficulty": problem["difficulty"],
                "prompt_type": prompt_type,
                "model": model,
                "error": str(e),
                "response_time": end_time - start_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def run_all_tests(self, repetitions=1, prompt_types=None, models=None, 
                      problems=None, save_interim=True):
        """
        Run tests for all problems with specified prompt types and models.
        
        Args:
            repetitions: Number of times to repeat each test
            prompt_types: List of prompt types to test (defaults to all if None)
            models: List of models to test (defaults to all if None)
            problems: List of problems to test (defaults to all if None)
            save_interim: Whether to save interim results
            
        Returns:
            List of test results
        """
        # Use defaults if not specified
        if prompt_types is None:
            prompt_types = list(self.prompts.keys())
        
        if models is None:
            models = self.models
        
        if problems is None:
            problems = self.problems
        
        # Create results directory
        os.makedirs("experiment_results", exist_ok=True)
        
        # Track progress
        total_tests = len(problems) * len(prompt_types) * len(models) * repetitions
        completed_tests = 0
        
        print(f"Starting language efficiency tests...")
        print(f"Total tests to run: {total_tests}")
        
        # Run tests
        for rep in range(repetitions):
            for problem in problems:
                for prompt_type in prompt_types:
                    for model in models:
                        try:
                            # Run the test
                            result = self.run_test(problem, prompt_type, model)
                            
                            # Add repetition number
                            result["repetition"] = rep + 1
                            
                            # Add to results
                            self.results.append(result)
                            
                            # Update progress
                            completed_tests += 1
                            progress = (completed_tests / total_tests) * 100
                            print(f"Completed {completed_tests}/{total_tests} tests ({progress:.1f}%): "
                                  f"{problem['id']} with {prompt_type} prompt on {model}")
                            
                            # Save interim results if requested
                            if save_interim and completed_tests % 10 == 0:
                                interim_filename = f"experiment_results/interim_results_{completed_tests}.json"
                                with open(interim_filename, 'w') as f:
                                    json.dump(self.results, f, indent=2)
                                print(f"Saved interim results to {interim_filename}")
                            
                            # Sleep to avoid API rate limits
                            time.sleep(2)
                            
                        except Exception as e:
                            print(f"Error on {problem['id']} with {prompt_type} on {model}: {str(e)}")
        
        print(f"All tests completed!")
        return self.results
    
    def save_results(self, filename=None):
        """
        Save test results to files.
        
        Args:
            filename: Base filename to use (without extension)
            
        Returns:
            Dictionary with saved filenames
        """
        # Generate default filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_results/language_efficiency_{timestamp}"
        else:
            filename = f"experiment_results/{filename}"
        
        # Save as JSON
        json_filename = f"{filename}.json"
        with open(json_filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save as CSV for easier analysis
        df = pd.DataFrame(self.results)
        csv_filename = f"{filename}.csv"
        
        # Remove the long text fields for CSV to keep it manageable
        df_csv = df.drop(columns=["response_text"], errors="ignore")
        df_csv.to_csv(csv_filename, index=False)
        
        print(f"Results saved to {json_filename} and {csv_filename}")
        
        return {
            "json_filename": json_filename,
            "csv_filename": csv_filename
        }
    
    def analyze_results(self):
        """
        Generate comprehensive analysis of test results.
        
        Returns:
            Dictionary with analysis results
        """
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Filter out error results
        if 'error' in df.columns:
            success_df = df[df['error'].isna()]
        else:
            success_df = df
        
        if success_df.empty:
            return {"error": "No successful results to analyze"}
        
        # Basic statistics by prompt type and model
        agg_columns = [
            'input_tokens', 'output_tokens', 'total_tokens', 'response_time',
            'response_bits_per_token', 'response_chinese_ratio', 
            'response_compression_ratio', 'response_chinese_chars_per_token'
        ]
        
        # Ensure all columns exist
        for col in agg_columns:
            if col not in success_df.columns:
                success_df[col] = np.nan
        
        # Group by prompt type and model
        agg_by_prompt_model = success_df.groupby(['prompt_type', 'model'])[agg_columns].agg(
            ['mean', 'std', 'median', 'count']
        )
        
        # Convert to a JSON-serializable format
        agg_dict = {}
        for (prompt_type, model), group_data in agg_by_prompt_model.groupby(level=[0, 1]):
            if prompt_type not in agg_dict:
                agg_dict[prompt_type] = {}
            agg_dict[prompt_type][model] = {}
            
            for col in agg_columns:
                agg_dict[prompt_type][model][col] = {
                    'mean': group_data.loc[(prompt_type, model), (col, 'mean')],
                    'std': group_data.loc[(prompt_type, model), (col, 'std')],
                    'median': group_data.loc[(prompt_type, model), (col, 'median')],
                    'count': group_data.loc[(prompt_type, model), (col, 'count')]
                }
        
        # Cross-validation analysis
        cv_results = perform_cross_validation(success_df)
        
        # Statistical significance analysis
        significance_results = calculate_statistical_significance(success_df)
        
        # Validation across difficulty levels
        difficulty_validation = validate_across_difficulty_levels(success_df)
        
        # Token usage analysis by difficulty
        token_usage_analysis = analyze_token_usage_by_difficulty(success_df)
        
        # Normalized compression metrics
        compression_metrics = calculate_normalized_compression_metrics(success_df)
        
        # Compare Chinese vs English token usage
        prompt_types = success_df['prompt_type'].unique()
        comparison = {}
        
        for i, type1 in enumerate(prompt_types):
            for type2 in prompt_types[i+1:]:
                type1_df = success_df[success_df['prompt_type'] == type1]
                type2_df = success_df[success_df['prompt_type'] == type2]
                
                if not type1_df.empty and not type2_df.empty:
                    type1_tokens = type1_df['total_tokens'].mean()
                    type2_tokens = type2_df['total_tokens'].mean()
                    
                    token_diff = type1_tokens - type2_tokens
                    token_reduction = (token_diff / type1_tokens) * 100 if type1_tokens > 0 else 0
                    
                    comparison[f"{type1}_vs_{type2}"] = {
                        f"{type1}_avg_tokens": type1_tokens,
                        f"{type2}_avg_tokens": type2_tokens,
                        "absolute_difference": token_diff,
                        "token_reduction_percent": token_reduction,
                        f"is_{type2}_more_efficient": token_diff > 0
                    }
        
        # Create visualizations
        try:
            self.create_visualizations(success_df)
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
        
        return {
            'aggregated_by_prompt_model': agg_dict,
            'cross_validation_results': cv_results,
            'statistical_significance': significance_results,
            'difficulty_validation': difficulty_validation,
            'token_usage_analysis': token_usage_analysis,
            'compression_metrics': compression_metrics.to_dict(orient='records') if not compression_metrics.empty else {},
            'prompt_type_comparisons': comparison
        }
    
    def create_visualizations(self, df):
        """
        Create visualizations of the results.
        
        Args:
            df: DataFrame with results
        """
        # Create visualizations directory
        os.makedirs("reports/visualizations", exist_ok=True)
        
        # 1. Token usage by prompt type
        plt.figure(figsize=(12, 6))
        token_data = df.groupby('prompt_type').agg({
            'input_tokens': 'mean',
            'output_tokens': 'mean',
            'total_tokens': 'mean'
        })
        token_data.plot(kind='bar', title='Average Token Usage by Prompt Type')
        plt.ylabel('Number of Tokens')
        plt.tight_layout()
        plt.savefig('reports/visualizations/token_usage_by_prompt.png')
        
        # 2. Token usage by benchmark and prompt type
        plt.figure(figsize=(15, 8))
        benchmark_data = df.pivot_table(
            values='total_tokens',
            index='benchmark',
            columns='prompt_type',
            aggfunc='mean'
        )
        benchmark_data.plot(kind='bar', title='Token Usage by Benchmark')
        plt.ylabel('Average Total Tokens')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('reports/visualizations/token_usage_by_benchmark.png')
        
        # 3. Information density (bits per token)
        if 'response_bits_per_token' in df.columns:
            plt.figure(figsize=(10, 6))
            density_data = df.groupby('prompt_type')['response_bits_per_token'].mean()
            density_data.plot(kind='bar', title='Information Density (Bits/Token)')
            plt.ylabel('Bits per Token')
            plt.tight_layout()
            plt.savefig('reports/visualizations/information_density.png')
        
        # 4. Chinese character ratio
        if 'response_chinese_ratio' in df.columns:
            plt.figure(figsize=(10, 6))
            chinese_ratio = df.groupby('prompt_type')['response_chinese_ratio'].mean()
            chinese_ratio.plot(kind='bar', title='Average Chinese Character Ratio')
            plt.ylabel('Ratio of Chinese Characters')
            plt.tight_layout()
            plt.savefig('reports/visualizations/chinese_character_ratio.png')
        
        # 5. Compression ratio comparison
        if 'response_compression_ratio' in df.columns:
            plt.figure(figsize=(10, 6))
            compression = df.groupby('prompt_type')['response_compression_ratio'].mean()
            compression.plot(kind='bar', title='Estimated Compression Ratio')
            plt.ylabel('Compression Ratio')
            plt.tight_layout()
            plt.savefig('reports/visualizations/compression_ratio.png')
        
        # 6. Difficulty impact on token usage
        plt.figure(figsize=(12, 8))
        diff_impact = df.pivot_table(
            values='total_tokens',
            index='difficulty',
            columns='prompt_type',
            aggfunc='mean'
        )
        diff_impact.plot(kind='bar', title='Impact of Problem Difficulty on Token Usage')
        plt.ylabel('Average Total Tokens')
        plt.tight_layout()
        plt.savefig('reports/visualizations/difficulty_impact.png')
        
        # 7. Chinese characters per token
        if 'response_chinese_chars_per_token' in df.columns:
            plt.figure(figsize=(10, 6))
            chars_per_token = df.groupby('prompt_type')['response_chinese_chars_per_token'].mean()
            chars_per_token.plot(kind='bar', title='Chinese Characters per Token')
            plt.ylabel('Characters per Token')
            plt.tight_layout()
            plt.savefig('reports/visualizations/chinese_chars_per_token.png')
        
        # 8. Response time comparison
        plt.figure(figsize=(10, 6))
        response_time = df.groupby('prompt_type')['response_time'].mean()
        response_time.plot(kind='bar', title='Average Response Time by Prompt Type')
        plt.ylabel('Response Time (seconds)')
        plt.tight_layout()
        plt.savefig('reports/visualizations/response_time.png')
        
        # 9. Token usage by category and prompt type
        plt.figure(figsize=(15, 8))
        category_data = df.pivot_table(
            values='total_tokens',
            index='category',
            columns='prompt_type',
            aggfunc='mean'
        )
        category_data.plot(kind='bar', title='Token Usage by Problem Category')
        plt.ylabel('Average Total Tokens')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('reports/visualizations/token_usage_by_category.png')
        
        # 10. Model comparison (if multiple models)
        if len(df['model'].unique()) > 1:
            plt.figure(figsize=(12, 8))
            model_data = df.pivot_table(
                values='total_tokens',
                index='model',
                columns='prompt_type',
                aggfunc='mean'
            )
            model_data.plot(kind='bar', title='Token Usage by Model')
            plt.ylabel('Average Total Tokens')
            plt.tight_layout()
            plt.savefig('reports/visualizations/token_usage_by_model.png')

def main():
    """
    Main function to run the enhanced language efficiency tests.
    """
    # Ensure API key is set
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("Please set ANTHROPIC_API_KEY environment variable")
    
    # Create results directory
    os.makedirs("experiment_results", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("reports/visualizations", exist_ok=True)
    
    # Initialize test runner
    print("Initializing enhanced language efficiency test runner...")
    test = EnhancedLanguageEfficiencyTest()
    
    # Run baseline tests with repetitions
    print("Running baseline tests...")
    test.run_all_tests(
        repetitions=3,  # Run each test 3 times for more reliable results
        prompt_types=["english", "chinese"],  # Start with basic comparison
        save_interim=True
    )
    
    # Save baseline results
    baseline_files = test.save_results("baseline_results")
    
    # Analyze baseline results
    print("Analyzing baseline results...")
    baseline_analysis = test.analyze_results()
    
    # Save baseline analysis
    with open("reports/baseline_analysis.json", 'w') as f:
        json.dump(baseline_analysis, f, indent=2)
    
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
    
    # Run additional tests with hybrid approach
    print("\nRunning additional tests with hybrid approach...")
    test.run_all_tests(
        repetitions=3,
        prompt_types=["hybrid"],
        save_interim=True
    )
    
    # Save all results
    all_files = test.save_results("all_results")
    
    # Analyze all results
    print("Analyzing all results...")
    all_analysis = test.analyze_results()
    
    # Save all analysis
    with open("reports/all_analysis.json", 'w') as f:
        json.dump(all_analysis, f, indent=2)
    
    print("Testing complete! Results and analysis saved to the 'experiment_results' and 'reports' directories.")

if __name__ == "__main__":
    main()
