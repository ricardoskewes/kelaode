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
from dashscope import Generation as QwenGeneration
from deepseek import DeepSeekAPI

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
            models: List of models to test (format: "provider:model_name")
            prompts: Dictionary of prompts to use (defaults to standard prompts if None)
        """
        self.problems = problems
        self.models = models
        self.results = []
        
        # Initialize clients for different providers
        self.clients = {}
        
        # Initialize Anthropic client
        if os.environ.get("ANTHROPIC_API_KEY"):
            self.clients["anthropic"] = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
            )
        
        # Initialize Qwen client (dashscope)
        if os.environ.get("DASHSCOPE_API_KEY"):
            # No client initialization needed for dashscope as it's used directly
            self.clients["qwen"] = "dashscope"
            
        # Initialize Deepseek client
        if os.environ.get("DEEPSEEK_API_KEY"):
            self.clients["deepseek"] = DeepSeekAPI(
                api_key=os.environ.get("DEEPSEEK_API_KEY")
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

This approach allows you to leverage the precision of English for technical terms while using the efficiency of Chinese for reasoning.""",
                
                # New language prompts
                "finnish": """Please solve this problem by thinking ENTIRELY in Finnish. Your entire reasoning process must be in Finnish.

1. Read and understand the problem in English
2. Translate the problem to Finnish in your mind
3. Think through ALL steps of solving the problem IN FINNISH
4. Work through the complete solution IN FINNISH
5. Only at the very end, translate just the final answer to English

Show your entire reasoning process in Finnish. Only the final answer should be in English.""",
                
                "german": """Please solve this problem by thinking ENTIRELY in German. Your entire reasoning process must be in German.

1. Read and understand the problem in English
2. Translate the problem to German in your mind
3. Think through ALL steps of solving the problem IN GERMAN
4. Work through the complete solution IN GERMAN
5. Only at the very end, translate just the final answer to English

Show your entire reasoning process in German. Only the final answer should be in English.""",
                
                "japanese": """Please solve this problem by thinking ENTIRELY in Japanese. Your entire reasoning process must be in Japanese.

1. Read and understand the problem in English
2. Translate the problem to Japanese in your mind
3. Think through ALL steps of solving the problem IN JAPANESE
4. Work through the complete solution IN JAPANESE
5. Only at the very end, translate just the final answer to English

Show your entire reasoning process in Japanese. Only the final answer should be in English.""",
                
                "korean": """Please solve this problem by thinking ENTIRELY in Korean. Your entire reasoning process must be in Korean.

1. Read and understand the problem in English
2. Translate the problem to Korean in your mind
3. Think through ALL steps of solving the problem IN KOREAN
4. Work through the complete solution IN KOREAN
5. Only at the very end, translate just the final answer to English

Show your entire reasoning process in Korean. Only the final answer should be in English.""",
                
                "russian": """Please solve this problem by thinking ENTIRELY in Russian. Your entire reasoning process must be in Russian.

1. Read and understand the problem in English
2. Translate the problem to Russian in your mind
3. Think through ALL steps of solving the problem IN RUSSIAN
4. Work through the complete solution IN RUSSIAN
5. Only at the very end, translate just the final answer to English

Show your entire reasoning process in Russian. Only the final answer should be in English.""",
                
                "arabic": """Please solve this problem by thinking ENTIRELY in Arabic. Your entire reasoning process must be in Arabic.

1. Read and understand the problem in English
2. Translate the problem to Arabic in your mind
3. Think through ALL steps of solving the problem IN ARABIC
4. Work through the complete solution IN ARABIC
5. Only at the very end, translate just the final answer to English

Show your entire reasoning process in Arabic. Only the final answer should be in English.""",
                
                # Strategic language selection prompt
                "strategic": """Please solve this problem using the most efficient language for this specific type of problem.

1. First, analyze the problem to determine its domain (mathematical, logical, scientific, etc.)
2. Choose the most efficient language for this domain:
   - For mathematical problems: Use Chinese
   - For logical deduction: Use German
   - For reading comprehension: Use English
   - For scientific reasoning: Use Finnish
   - For multi-step procedures: Use Japanese
3. Think through ALL steps in your chosen language
4. Clearly indicate which language you're using with "[REASONING IN LANGUAGE: X]"
5. Only at the very end, translate just the final answer to English with "[ANSWER]:" prefix

This approach leverages the unique efficiency of different languages for different types of reasoning."""
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
        import re
        
        if prompt_type == "english":
            # For English, just use the last paragraph or sentence as the answer
            paragraphs = text.split('\n\n')
            return paragraphs[-1] if paragraphs else text
        
        elif prompt_type in ["chinese", "finnish", "german", "japanese", "korean", "russian", "arabic"]:
            # Look for English text near the end of the response
            paragraphs = text.split('\n\n')
            
            # Language-specific character ranges
            char_ranges = {
                "chinese": ('\u4e00', '\u9fff'),  # Chinese
                "japanese": ('\u3040', '\u30ff'),  # Japanese Hiragana and Katakana
                "korean": ('\uac00', '\ud7a3'),  # Korean Hangul
                "russian": ('\u0400', '\u04ff'),  # Cyrillic
                "arabic": ('\u0600', '\u06ff'),  # Arabic
                # Finnish and German use Latin script, so we'll use a different approach
            }
            
            if prompt_type in char_ranges:
                # For languages with non-Latin scripts, look for paragraphs with mostly Latin characters
                range_start, range_end = char_ranges[prompt_type]
                for p in reversed(paragraphs):
                    # If paragraph has mostly non-specific language characters, it might be the answer
                    if sum(1 for char in p if range_start <= char <= range_end) / max(len(p), 1) < 0.3:
                        return p
            else:
                # For Finnish and German (Latin script), look for English-like text at the end
                # This is more heuristic - looking for shorter paragraphs with common English words
                english_indicators = ["answer", "result", "therefore", "thus", "so", "=", "is"]
                for p in reversed(paragraphs):
                    # Check if paragraph contains English indicator words and is relatively short
                    if any(indicator in p.lower() for indicator in english_indicators) and len(p.split()) < 30:
                        return p
            
            # Fallback to last paragraph
            return paragraphs[-1] if paragraphs else text
        
        elif prompt_type == "chinese_with_markers":
            # Look for the marked English answer
            match = re.search(r'\[ENGLISH ANSWER\]:(.*?)($|\n\n)', text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return "No marked English answer found"
        
        elif prompt_type == "hybrid":
            # Look for the marked answer in hybrid mode
            match = re.search(r'\[ANSWER\]:(.*?)($|\n\n)', text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return "No marked answer found"
        
        elif prompt_type == "strategic":
            # Look for the marked answer in strategic language selection mode
            match = re.search(r'\[ANSWER\]:(.*?)($|\n\n)', text, re.DOTALL)
            if match:
                return match.group(1).strip()
            
            # Fallback: look for any English-like text at the end
            paragraphs = text.split('\n\n')
            for p in reversed(paragraphs):
                if len(p.split()) < 30 and re.search(r'[a-zA-Z]', p):
                    return p
            
            return "No marked answer found"
        
        # Default fallback
        return text.split('\n')[-1] if '\n' in text else text
    
    def get_metrics(self, text, token_count, prompt_type="english"):
        """
        Calculate various text metrics.
        
        Args:
            text: Text to analyze
            token_count: Number of tokens in the text
            prompt_type: Type of prompt used (language)
            
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        char_count = len(text)
        word_count = len(text.split())
        
        # Language-specific character metrics
        language_char_counts = {
            "chinese": sum(1 for char in text if '\u4e00' <= char <= '\u9fff'),
            "japanese": sum(1 for char in text if '\u3040' <= char <= '\u30ff' or '\u4e00' <= char <= '\u9fff'),
            "korean": sum(1 for char in text if '\uac00' <= char <= '\ud7a3'),
            "russian": sum(1 for char in text if '\u0400' <= char <= '\u04ff'),
            "arabic": sum(1 for char in text if '\u0600' <= char <= '\u06ff'),
            "finnish": sum(1 for char in text if char in 'äöåÄÖÅ'),  # Finnish special characters
            "german": sum(1 for char in text if char in 'äöüßÄÖÜ'),  # German special characters
        }
        
        # Calculate specific language ratio
        language_ratios = {}
        for lang, count in language_char_counts.items():
            language_ratios[f"{lang}_ratio"] = count / char_count if char_count > 0 else 0
        
        # Information density metrics
        bits_per_token = calculate_bits_per_token(text, token_count)
        
        # Semantic density metrics
        semantic_density = calculate_semantic_density(text, token_count)
        
        # Chinese information density metrics (for backward compatibility)
        chinese_density = analyze_chinese_information_density(text, token_count)
        
        # Calculate non-specific language characters
        specific_lang_chars = language_char_counts.get(prompt_type, 0)
        non_specific_lang_chars = char_count - specific_lang_chars
        
        # Compression ratio estimate (relative to English)
        # Adjust weights based on language
        lang_weights = {
            "chinese": 2.5,
            "japanese": 2.3,
            "korean": 2.0,
            "arabic": 1.8,
            "russian": 1.5,
            "finnish": 1.3,
            "german": 1.2,
            "english": 1.0
        }
        
        weight = lang_weights.get(prompt_type, 1.0)
        estimated_english_chars = specific_lang_chars * weight + non_specific_lang_chars
        compression_ratio = char_count / estimated_english_chars if estimated_english_chars > 0 else 1
        
        # Characters per token (language efficiency metric)
        chars_per_token = specific_lang_chars / token_count if token_count > 0 else 0
        
        # Base metrics dictionary
        metrics = {
            "char_count": char_count,
            "word_count": word_count,
            "bits_per_token": bits_per_token,
            "semantic_density": semantic_density["semantic_density"],
            "content_ratio": semantic_density["content_ratio"],
            "compression_ratio": compression_ratio,
            "chars_per_token": chars_per_token
        }
        
        # Add language-specific character counts and ratios
        metrics.update({f"{lang}_char_count": count for lang, count in language_char_counts.items()})
        metrics.update(language_ratios)
        
        # Add Chinese-specific metrics for backward compatibility
        metrics.update({
            "chinese_info_density": chinese_density["chinese_info_density"],
            "chinese_chars_per_token": chinese_density["chinese_chars_per_token"]
        })
        
        return metrics
    
    def run_test(self, problem, prompt_type, model):
        """
        Run a single test case with the specified prompt type and model.
        
        Args:
            problem: Problem to test
            prompt_type: Type of prompt to use
            model: Model to use (format: "provider:model_name")
            
        Returns:
            Dictionary with test results
        """
        prompt = self.prompts[prompt_type]
        full_prompt = f"{prompt}\n\n{problem['problem']}"
        
        start_time = time.time()
        
        try:
            # Parse model string to get provider and model name
            if ":" in model:
                provider, model_name = model.split(":", 1)
            else:
                # Default to Anthropic if no provider specified
                provider, model_name = "anthropic", model
            
            # Check if we have a client for this provider
            if provider not in self.clients:
                raise ValueError(f"No client available for provider: {provider}")
            
            # Call the appropriate API based on the provider
            if provider == "anthropic":
                response = self.clients[provider].messages.create(
                    model=model_name,
                    max_tokens=4000,
                    messages=[
                        {"role": "user", "content": full_prompt}
                    ]
                )
                response_text = response.content[0].text
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                total_tokens = input_tokens + output_tokens
                
            elif provider == "qwen":
                # Use dashscope for Qwen models
                response = QwenGeneration.call(
                    model=model_name,
                    api_key=os.environ.get("DASHSCOPE_API_KEY"),
                    messages=[
                        {"role": "user", "content": full_prompt}
                    ]
                )
                
                if response.status_code != 200:
                    raise ValueError(f"Qwen API error: {response.message}")
                
                response_text = response.output.choices[0].message.content
                # Qwen API might provide token counts differently
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                total_tokens = response.usage.total_tokens
            
            elif provider == "deepseek":
<<<<<<< HEAD
                # Use deepseek for Deepseek models
                try:
                    response = self.clients[provider].chat_completion(
                        model=model_name,
                        messages=[
                            {"role": "user", "content": full_prompt}
                        ],
                        max_tokens=4000
                    )
                    
                    response_text = response.choices[0].message.content
                    # Deepseek API provides token counts
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens
                    total_tokens = response.usage.total_tokens
                except Exception as e:
                    if "Insufficient Balance" in str(e):
                        print(f"Warning: Deepseek API key has insufficient balance: {str(e)}")
                        raise ValueError(f"Deepseek API error: Insufficient Balance")
                    else:
                        raise
||||||| 46db262
                # Use deepseek-ai for Deepseek models
                response = self.clients[provider].chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": full_prompt}
                    ],
                    max_tokens=4000
                )
                
                response_text = response.choices[0].message.content
                # Deepseek API provides token counts
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
=======
                # Use deepseek for Deepseek models
                response = self.clients[provider].chat_completion(
                    prompt=full_prompt,
                    prompt_sys="You are a helpful assistant",
                    model=model_name,
                    max_tokens=4000
                )
                
                response_text = response
                # Deepseek API doesn't provide token counts directly, estimate them
                # This is a rough estimate based on character count
                char_count = len(full_prompt)
                response_char_count = len(response_text)
                
                # Estimate tokens based on average characters per token
                # (approximately 4 characters per token for English)
                input_tokens = char_count // 4
                output_tokens = response_char_count // 4
                total_tokens = input_tokens + output_tokens
>>>>>>> 536290cfbb91ce6e24a0c7056f9c775b67af2a12
            
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
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
                "provider": provider,
                "model_name": model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "response_time": end_time - start_time,
                "response_text": response_text,
                "extracted_answer": english_answer,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add metrics for response - pass prompt_type to get language-specific metrics
            metrics = self.get_metrics(response_text, output_tokens, prompt_type)
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
            df: DataFrame or dictionary with results
        """
        # Create visualizations directory
        os.makedirs("reports/visualizations", exist_ok=True)
        os.makedirs("reports/visualizations/multilingual", exist_ok=True)
        
        # Convert to DataFrame if it's not already
        if not isinstance(df, pd.DataFrame):
            if isinstance(df, dict) and "results" in df:
                df = pd.DataFrame(df["results"])
            elif isinstance(df, list):
                df = pd.DataFrame(df)
            else:
                # Use the stored results if df is not a valid format
                df = pd.DataFrame(self.results)
        
        # Check if DataFrame is empty
        if df.empty:
            print("Warning: No data available for visualization")
            return
            
        # Check if prompt_type column exists
        if 'prompt_type' not in df.columns:
            print("Warning: 'prompt_type' column not found in data. Using mock data for visualization.")
            # Create a mock prompt_type column with values from our test
            df['prompt_type'] = df.apply(lambda row: row.get('prompt_type', 'english'), axis=1)
            
        # Get unique prompt types (languages)
        prompt_types = df['prompt_type'].unique()
        
        # 1. Token usage by prompt type
        plt.figure(figsize=(14, 8))
        token_data = df.groupby('prompt_type').agg({
            'input_tokens': 'mean',
            'output_tokens': 'mean',
            'total_tokens': 'mean'
        })
        token_data.plot(kind='bar', title='Average Token Usage by Language')
        plt.ylabel('Number of Tokens')
        plt.xlabel('Language')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('reports/visualizations/multilingual/token_usage_by_language.png')
        
        # 2. Token usage by benchmark and prompt type
        plt.figure(figsize=(16, 10))
        benchmark_data = df.pivot_table(
            values='total_tokens',
            index='benchmark',
            columns='prompt_type',
            aggfunc='mean'
        )
        benchmark_data.plot(kind='bar', title='Token Usage by Benchmark and Language')
        plt.ylabel('Average Total Tokens')
        plt.xlabel('Benchmark')
        plt.xticks(rotation=45)
        plt.legend(title='Language')
        plt.tight_layout()
        plt.savefig('reports/visualizations/multilingual/token_usage_by_benchmark.png')
        
        # 3. Information density (bits per token)
        if 'response_bits_per_token' in df.columns:
            plt.figure(figsize=(14, 8))
            density_data = df.groupby('prompt_type')['response_bits_per_token'].mean()
            density_data.plot(kind='bar', title='Information Density by Language (Bits/Token)')
            plt.ylabel('Bits per Token')
            plt.xlabel('Language')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('reports/visualizations/multilingual/information_density.png')
        
        # 4. Language-specific character ratios
        plt.figure(figsize=(14, 8))
        ratio_columns = [col for col in df.columns if col.endswith('_ratio') and col.startswith('response_')]
        if ratio_columns:
            ratio_data = df.groupby('prompt_type')[ratio_columns].mean()
            # Rename columns for better readability
            ratio_data.columns = [col.replace('response_', '').replace('_ratio', '') for col in ratio_data.columns]
            ratio_data.plot(kind='bar', title='Language Character Ratios by Prompt Type')
            plt.ylabel('Character Ratio')
            plt.xlabel('Prompt Language')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('reports/visualizations/multilingual/language_character_ratios.png')
        
        # 5. Compression ratio comparison
        if 'response_compression_ratio' in df.columns:
            plt.figure(figsize=(14, 8))
            compression = df.groupby('prompt_type')['response_compression_ratio'].mean()
            compression.plot(kind='bar', title='Estimated Compression Ratio by Language')
            plt.ylabel('Compression Ratio (relative to English)')
            plt.xlabel('Language')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('reports/visualizations/multilingual/compression_ratio.png')
        
        # 6. Difficulty impact on token usage
        plt.figure(figsize=(14, 10))
        diff_impact = df.pivot_table(
            values='total_tokens',
            index='difficulty',
            columns='prompt_type',
            aggfunc='mean'
        )
        diff_impact.plot(kind='bar', title='Impact of Problem Difficulty on Token Usage by Language')
        plt.ylabel('Average Total Tokens')
        plt.xlabel('Difficulty Level')
        plt.legend(title='Language')
        plt.tight_layout()
        plt.savefig('reports/visualizations/multilingual/difficulty_impact.png')
        
        # 7. Characters per token by language
        if 'response_chars_per_token' in df.columns:
            plt.figure(figsize=(14, 8))
            chars_per_token = df.groupby('prompt_type')['response_chars_per_token'].mean()
            chars_per_token.plot(kind='bar', title='Characters per Token by Language')
            plt.ylabel('Characters per Token')
            plt.xlabel('Language')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('reports/visualizations/multilingual/chars_per_token.png')
        
        # 8. Response time comparison
        plt.figure(figsize=(14, 8))
        response_time = df.groupby('prompt_type')['response_time'].mean()
        response_time.plot(kind='bar', title='Average Response Time by Language')
        plt.ylabel('Response Time (seconds)')
        plt.xlabel('Language')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('reports/visualizations/multilingual/response_time.png')
        
        # 9. Token usage by category and prompt type
        plt.figure(figsize=(16, 10))
        category_data = df.pivot_table(
            values='total_tokens',
            index='category',
            columns='prompt_type',
            aggfunc='mean'
        )
        category_data.plot(kind='bar', title='Token Usage by Problem Category and Language')
        plt.ylabel('Average Total Tokens')
        plt.xlabel('Problem Category')
        plt.xticks(rotation=45)
        plt.legend(title='Language')
        plt.tight_layout()
        plt.savefig('reports/visualizations/multilingual/token_usage_by_category.png')
        
        # 10. Language Efficiency Index visualization
        plt.figure(figsize=(14, 8))
        
        # Calculate efficiency index (relative to English)
        if 'english' in prompt_types:
            english_tokens = df[df['prompt_type'] == 'english']['total_tokens'].mean()
            efficiency_data = {}
            
            for lang in prompt_types:
                if lang != 'english':
                    lang_tokens = df[df['prompt_type'] == lang]['total_tokens'].mean()
                    efficiency = (english_tokens - lang_tokens) / english_tokens * 100
                    efficiency_data[lang] = efficiency
            
            # Create bar chart of efficiency index
            if efficiency_data:
                langs = list(efficiency_data.keys())
                efficiencies = list(efficiency_data.values())
                
                # Sort by efficiency
                sorted_indices = sorted(range(len(efficiencies)), key=lambda i: efficiencies[i], reverse=True)
                sorted_langs = [langs[i] for i in sorted_indices]
                sorted_efficiencies = [efficiencies[i] for i in sorted_indices]
                
                # Create color map (green for positive, red for negative)
                colors = ['green' if eff >= 0 else 'red' for eff in sorted_efficiencies]
                
                plt.bar(sorted_langs, sorted_efficiencies, color=colors)
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.title('Language Efficiency Index (% Token Reduction vs. English)')
                plt.ylabel('Token Reduction Percentage')
                plt.xlabel('Language')
                plt.xticks(rotation=45)
                
                # Add value labels on bars
                for i, v in enumerate(sorted_efficiencies):
                    plt.text(i, v + (1 if v >= 0 else -3), f"{v:.1f}%", 
                             ha='center', va='bottom' if v >= 0 else 'top')
                
                plt.tight_layout()
                plt.savefig('reports/visualizations/multilingual/language_efficiency_index.png')
        
        # 11. Model comparison (if multiple models)
        if len(df['model'].unique()) > 1:
            plt.figure(figsize=(14, 10))
            model_data = df.pivot_table(
                values='total_tokens',
                index='model',
                columns='prompt_type',
                aggfunc='mean'
            )
            model_data.plot(kind='bar', title='Token Usage by Model and Language')
            plt.ylabel('Average Total Tokens')
            plt.xlabel('Model')
            plt.legend(title='Language')
            plt.tight_layout()
            plt.savefig('reports/visualizations/multilingual/token_usage_by_model.png')
            
        # 12. Language Compression Radar Chart
        if len(prompt_types) > 2:
            plt.figure(figsize=(12, 10))
            from matplotlib.path import Path
            from matplotlib.spines import Spine
            from matplotlib.transforms import Affine2D
            
            # Metrics to include in radar chart
            metrics = ['total_tokens', 'response_bits_per_token', 'response_compression_ratio', 
                      'response_chars_per_token', 'response_time']
            metric_names = ['Token Usage', 'Bits/Token', 'Compression Ratio', 
                           'Chars/Token', 'Response Time']
            
            # Get data for each language
            radar_data = {}
            for lang in prompt_types:
                lang_data = df[df['prompt_type'] == lang]
                radar_data[lang] = [
                    lang_data['total_tokens'].mean() if 'total_tokens' in lang_data else 0,
                    lang_data['response_bits_per_token'].mean() if 'response_bits_per_token' in lang_data else 0,
                    lang_data['response_compression_ratio'].mean() if 'response_compression_ratio' in lang_data else 0,
                    lang_data['response_chars_per_token'].mean() if 'response_chars_per_token' in lang_data else 0,
                    lang_data['response_time'].mean() if 'response_time' in lang_data else 0
                ]
            
            # Normalize data (lower is better for all metrics)
            normalized_data = {}
            for metric_idx in range(len(metrics)):
                max_val = max([data[metric_idx] for data in radar_data.values()])
                min_val = min([data[metric_idx] for data in radar_data.values()])
                
                for lang in radar_data:
                    if lang not in normalized_data:
                        normalized_data[lang] = []
                    
                    # Normalize and invert (so higher is better)
                    if max_val > min_val:
                        normalized_val = 1 - (radar_data[lang][metric_idx] - min_val) / (max_val - min_val)
                    else:
                        normalized_val = 0.5  # If all values are the same
                    
                    normalized_data[lang].append(normalized_val)
            
            # Set up radar chart
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            # Make sure angles is a list before appending
            if isinstance(angles, list):
                angles = angles + angles[:1]  # Close the loop
            else:
                # Handle the case where angles might not be a list
                angles = [0, np.pi/2, np.pi, 3*np.pi/2]
            
            ax = plt.subplot(111, polar=True)
            
            # Add metric labels
            plt.xticks(angles[:-1], metric_names, size=10)
            
            # Plot each language
            for lang, values in normalized_data.items():
                values += values[:1]  # Close the loop
                ax.plot(angles, values, linewidth=2, label=lang)
                ax.fill(angles, values, alpha=0.1)
            
            plt.title('Language Efficiency Radar Chart', size=15)
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.tight_layout()
            plt.savefig('reports/visualizations/multilingual/language_radar_chart.png')
            
        # Keep the original visualizations for backward compatibility
        # (simplified versions of the above)
        
        # Token usage by prompt type (original)
        plt.figure(figsize=(12, 6))
        token_data.plot(kind='bar', title='Average Token Usage by Prompt Type')
        plt.ylabel('Number of Tokens')
        plt.tight_layout()
        plt.savefig('reports/visualizations/token_usage_by_prompt.png')
        
        # Token usage by benchmark (original)
        plt.figure(figsize=(15, 8))
        benchmark_data.plot(kind='bar', title='Token Usage by Benchmark')
        plt.ylabel('Average Total Tokens')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('reports/visualizations/token_usage_by_benchmark.png')
        
        # Close all figures to free memory
        plt.close('all')
        
        # Import enhanced multilingual visualization methods
        try:
            from multilingual_visualizations import create_all_multilingual_visualizations
            
            # Create enhanced multilingual visualizations
            print("Creating enhanced multilingual visualizations...")
            create_all_multilingual_visualizations(df)
            
        except ImportError as e:
            print(f"Warning: Could not import multilingual visualization module: {str(e)}")
            print("Basic visualizations created, but enhanced multilingual visualizations skipped.")
        except Exception as e:
            print(f"Error creating enhanced multilingual visualizations: {str(e)}")

def main():
    """
    Main function to run the enhanced language efficiency tests.
    """
    # Ensure API keys are set
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set. Anthropic models will not be available.")
    
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("Warning: DASHSCOPE_API_KEY not set. Qwen models will not be available.")
    
    # Create results directory
    os.makedirs("experiment_results", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("reports/visualizations", exist_ok=True)
    os.makedirs("reports/visualizations/multilingual", exist_ok=True)
    
    # Initialize test runner with models from different providers
    print("Initializing enhanced language efficiency test runner...")
    
    # Define models to test
    models = []
    
    # Add Anthropic models if API key is available
    if os.environ.get("ANTHROPIC_API_KEY"):
        models.append("anthropic:claude-3-5-sonnet-20240620")
    
    # Add Qwen models if API key is available
    if os.environ.get("DASHSCOPE_API_KEY"):
        models.append("qwen:qwen-turbo")
        models.append("qwen:qwen-plus")
    
    # Add Deepseek models if API key is available
    if os.environ.get("DEEPSEEK_API_KEY"):
        models.append("deepseek:deepseek-chat")
        models.append("deepseek:deepseek-coder")
    
    # Check if we have any models to test
    if not models:
        raise ValueError("No API keys set. Please set at least one of ANTHROPIC_API_KEY, DASHSCOPE_API_KEY, or DEEPSEEK_API_KEY")
    
    test = EnhancedLanguageEfficiencyTest(
        models=models
    )
    
    # Run baseline tests with English and Chinese
    print("Running baseline tests with English and Chinese...")
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
        prompt_comparisons = baseline_analysis["prompt_type_comparisons"]
        if isinstance(prompt_comparisons, dict) and "english_vs_chinese" in prompt_comparisons:
            comp = prompt_comparisons["english_vs_chinese"]
            if isinstance(comp, dict) and "token_reduction_percent" in comp:
                token_reduction = comp.get("token_reduction_percent", 0)
                english_tokens = comp.get("english_avg_tokens", 0)
                chinese_tokens = comp.get("chinese_avg_tokens", 0)
                is_efficient = comp.get("is_chinese_more_efficient", False)
                
                print(f"\nSUMMARY OF BASELINE FINDINGS:")
                print(f"Chinese reasoning used {token_reduction:.2f}% fewer tokens than English")
                print(f"English average tokens: {english_tokens:.2f}")
                print(f"Chinese average tokens: {chinese_tokens:.2f}")
                
                if is_efficient:
                    print("\nChinese reasoning appears more token-efficient in this experiment.")
                else:
                    print("\nChinese reasoning does NOT appear more token-efficient in this experiment.")
    
    # Run multilingual tests with additional languages
    print("\nRunning multilingual tests with additional languages...")
    test.run_all_tests(
        repetitions=2,  # Reduce repetitions for expanded language set to manage API costs
        prompt_types=["finnish", "german", "japanese", "korean", "russian", "arabic"],
        save_interim=True
    )
    
    # Run strategic language selection test
    print("\nRunning strategic language selection test...")
    test.run_all_tests(
        repetitions=2,
        prompt_types=["strategic"],
        save_interim=True
    )
    
    # Save all results
    multilingual_files = test.save_results("multilingual_results")
    
    # Analyze multilingual results
    print("Analyzing multilingual results...")
    multilingual_analysis = test.analyze_results()
    
    # Save multilingual analysis
    with open("reports/multilingual_analysis.json", 'w') as f:
        json.dump(multilingual_analysis, f, indent=2)
    
    # Print summary of multilingual findings
    print(f"\nSUMMARY OF MULTILINGUAL FINDINGS:")
    
    # Compare each language to English
    if "prompt_type_comparisons" in multilingual_analysis:
        languages = ["chinese", "finnish", "german", "japanese", "korean", "russian", "arabic", "strategic"]
        
        print("\nLanguage Efficiency Comparison (vs. English):")
        print("=" * 60)
        print(f"{'Language':<12} | {'Token Reduction %':>18} | {'More Efficient?':>15}")
        print("-" * 60)
        
        for lang in languages:
            comp_key = f"english_vs_{lang}"
            if "prompt_type_comparisons" in multilingual_analysis:
                prompt_comparisons = multilingual_analysis["prompt_type_comparisons"]
                if isinstance(prompt_comparisons, dict) and comp_key in prompt_comparisons:
                    comp = prompt_comparisons[comp_key]
                    if isinstance(comp, dict):
                        reduction = comp.get("token_reduction_percent", 0)
                        is_efficient_key = f"is_{lang}_more_efficient"
                        is_efficient = comp.get(is_efficient_key, False)
                        
                        efficiency_text = "YES" if is_efficient else "NO"
                        print(f"{lang:<12} | {reduction:>18.2f}% | {efficiency_text:>15}")
    
    print("\nTesting complete! Results and analysis saved to the 'experiment_results' and 'reports' directories.")
    
    # Generate comprehensive report
    print("\nGenerating comprehensive multilingual report...")
    generate_comprehensive_report(test.results, multilingual_analysis)
    
    print("\nMultilingual language efficiency analysis complete!")

def generate_comprehensive_report(results, analysis):
    """
    Generate a comprehensive report of the multilingual language efficiency analysis.
    
    Args:
        results: Raw test results
        analysis: Analysis results
    """
    # Create report directory
    os.makedirs("reports/comprehensive", exist_ok=True)
    
    # Convert results to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Generate summary report
    with open("reports/comprehensive/multilingual_summary.md", 'w') as f:
        f.write("# Multilingual Chain-of-Thought Reasoning Efficiency Analysis\n\n")
        f.write("## Overview\n\n")
        f.write("This report analyzes the token efficiency of different languages for chain-of-thought reasoning.\n\n")
        
        # Add language comparison table
        f.write("## Language Efficiency Comparison\n\n")
        f.write("| Language | Avg Tokens | vs. English | Efficiency Gain |\n")
        f.write("|----------|------------|------------|----------------|\n")
        
        # Get English average tokens as baseline
        english_tokens = df[df['prompt_type'] == 'english']['total_tokens'].mean()
        
        # Add row for each language
        for lang in df['prompt_type'].unique():
            lang_tokens = df[df['prompt_type'] == lang]['total_tokens'].mean()
            vs_english = (lang_tokens / english_tokens) * 100 if english_tokens > 0 else 0
            efficiency = (english_tokens - lang_tokens) / english_tokens * 100 if english_tokens > 0 else 0
            
            f.write(f"| {lang} | {lang_tokens:.2f} | {vs_english:.2f}% | {efficiency:.2f}% |\n")
        
        # Add benchmark-specific analysis
        f.write("\n## Benchmark-Specific Efficiency\n\n")
        
        benchmarks = df['benchmark'].unique()
        for benchmark in benchmarks:
            f.write(f"### {benchmark}\n\n")
            f.write("| Language | Avg Tokens | vs. English | Efficiency Gain |\n")
            f.write("|----------|------------|------------|----------------|\n")
            
            # Get English average tokens for this benchmark
            benchmark_df = df[df['benchmark'] == benchmark]
            english_tokens = benchmark_df[benchmark_df['prompt_type'] == 'english']['total_tokens'].mean()
            
            # Add row for each language
            for lang in benchmark_df['prompt_type'].unique():
                lang_tokens = benchmark_df[benchmark_df['prompt_type'] == lang]['total_tokens'].mean()
                vs_english = (lang_tokens / english_tokens) * 100 if english_tokens > 0 else 0
                efficiency = (english_tokens - lang_tokens) / english_tokens * 100 if english_tokens > 0 else 0
                
                f.write(f"| {lang} | {lang_tokens:.2f} | {vs_english:.2f}% | {efficiency:.2f}% |\n")
            
            f.write("\n")
        
        # Add difficulty-specific analysis
        f.write("\n## Difficulty-Specific Efficiency\n\n")
        
        difficulties = df['difficulty'].unique()
        for difficulty in difficulties:
            f.write(f"### {difficulty}\n\n")
            f.write("| Language | Avg Tokens | vs. English | Efficiency Gain |\n")
            f.write("|----------|------------|------------|----------------|\n")
            
            # Get English average tokens for this difficulty
            difficulty_df = df[df['difficulty'] == difficulty]
            english_tokens = difficulty_df[difficulty_df['prompt_type'] == 'english']['total_tokens'].mean()
            
            # Add row for each language
            for lang in difficulty_df['prompt_type'].unique():
                lang_tokens = difficulty_df[difficulty_df['prompt_type'] == lang]['total_tokens'].mean()
                vs_english = (lang_tokens / english_tokens) * 100 if english_tokens > 0 else 0
                efficiency = (english_tokens - lang_tokens) / english_tokens * 100 if english_tokens > 0 else 0
                
                f.write(f"| {lang} | {lang_tokens:.2f} | {vs_english:.2f}% | {efficiency:.2f}% |\n")
            
            f.write("\n")
        
        # Add recommendations
        f.write("\n## Recommendations\n\n")
        f.write("Based on the analysis, here are the recommended languages for different types of reasoning:\n\n")
        
        # Analyze benchmark-specific efficiency to make recommendations
        recommendations = {}
        
        for benchmark in benchmarks:
            benchmark_df = df[df['benchmark'] == benchmark]
            english_tokens = benchmark_df[benchmark_df['prompt_type'] == 'english']['total_tokens'].mean()
            
            best_lang = "english"
            best_efficiency = 0
            
            for lang in benchmark_df['prompt_type'].unique():
                if lang == "english":
                    continue
                    
                lang_tokens = benchmark_df[benchmark_df['prompt_type'] == lang]['total_tokens'].mean()
                efficiency = (english_tokens - lang_tokens) / english_tokens * 100 if english_tokens > 0 else 0
                
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_lang = lang
            
            recommendations[benchmark] = {
                "best_language": best_lang,
                "efficiency_gain": best_efficiency
            }
        
        # Write recommendations
        f.write("| Problem Type | Recommended Language | Efficiency Gain |\n")
        f.write("|--------------|----------------------|----------------|\n")
        
        for benchmark, rec in recommendations.items():
            f.write(f"| {benchmark} | {rec['best_language']} | {rec['efficiency_gain']:.2f}% |\n")
        
        # Add conclusion
        f.write("\n## Conclusion\n\n")
        f.write("Different languages show varying levels of efficiency for different types of reasoning tasks. ")
        f.write("By strategically selecting the most efficient language for each problem domain, ")
        f.write("significant token savings can be achieved, potentially reducing API costs and improving response times.\n\n")
        
        f.write("The strategic language selection approach, which automatically chooses the most appropriate language ")
        f.write("based on the problem type, shows promise for real-world applications where efficiency is critical.\n")
    
    print(f"Comprehensive report saved to reports/comprehensive/multilingual_summary.md")

def test_qwen_connection():
    """
    Test the connection to the Qwen API.
    """
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("DASHSCOPE_API_KEY not set. Cannot test Qwen connection.")
        return False
    
    try:
        response = QwenGeneration.call(
            model="qwen-turbo",
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            messages=[
                {"role": "user", "content": "Hello, can you respond in Chinese?"}
            ]
        )
        
        if response.status_code == 200:
            print("Qwen API connection successful!")
            print(f"Response: {response.output.choices[0].message.content}")
            print(f"Input tokens: {response.usage.input_tokens}")
            print(f"Output tokens: {response.usage.output_tokens}")
            print(f"Total tokens: {response.usage.total_tokens}")
            return True
        else:
            print(f"Qwen API error: {response.message}")
            return False
    except Exception as e:
        print(f"Error testing Qwen connection: {str(e)}")
        return False

def test_deepseek_connection():
    """
    Test the connection to the Deepseek API.
    """
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("DEEPSEEK_API_KEY not set. Cannot test Deepseek connection.")
        return False
    
    try:
        # Import the DeepSeekAPI class
        from deepseek import DeepSeekAPI
        
        # Initialize the Deepseek client
        client = DeepSeekAPI(api_key=os.environ.get("DEEPSEEK_API_KEY"))
        
        # Test a simple query
        start_time = time.time()
        
<<<<<<< HEAD
        try:
            response = client.chat_completion(
                model="deepseek-chat",
                messages=[
                    {"role": "user", "content": "Hello, can you respond in Chinese?"}
                ]
            )
        except Exception as e:
            if "Insufficient Balance" in str(e):
                print("Warning: Deepseek API key has insufficient balance")
                print(f"Error details: {str(e)}")
                return False
            else:
                raise  # Re-raise other exceptions
||||||| 46db262
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "Hello, can you respond in Chinese?"}
            ]
        )
=======
        response = client.chat_completion(
            prompt="Hello, can you respond in Chinese?",
            prompt_sys="You are a helpful assistant",
            model="deepseek-chat"
        )
>>>>>>> 536290cfbb91ce6e24a0c7056f9c775b67af2a12
        
        end_time = time.time()
        
        # Print the response
        print("Deepseek API connection successful!")
        print(f"Response: {response}")
        print(f"Response time: {end_time - start_time:.2f} seconds")
        
        return True
    except Exception as e:
        print(f"Error testing Deepseek connection: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test-qwen":
        test_qwen_connection()
    elif len(sys.argv) > 1 and sys.argv[1] == "--test-deepseek":
        test_deepseek_connection()
    else:
        main()
