import os
import json
import time
import numpy as np
import pandas as pd
import anthropic
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import entropy
import re

# Initialize Anthropic client
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

# Complex reasoning problems from well-known AI research benchmarks
BENCHMARK_PROBLEMS = [
    # GSM8K (Grade School Math) problems - known for requiring multi-step reasoning
    {
        "id": "gsm8k_1",
        "category": "math_word_problem",
        "problem": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "benchmark": "GSM8K",
        "difficulty": "medium"
    },
    {
        "id": "gsm8k_2",
        "category": "math_word_problem",
        "problem": "Weng earns $12 an hour for babysitting. Yesterday, she babysit for 5 hours and spent 1/3 of her money on a movie ticket. How much money does she have left?",
        "benchmark": "GSM8K",
        "difficulty": "medium"
    },
    # MMLU problems - known for requiring domain expertise and complex reasoning
    {
        "id": "mmlu_1",
        "category": "logical_reasoning",
        "problem": "Consider a rule that says: If you have a driver's license, then you can drive. Suppose we know someone can drive. What follows logically about whether they have a driver's license?",
        "benchmark": "MMLU",
        "difficulty": "hard"
    },
    # ARC (AI2 Reasoning Challenge) problems - requiring multi-hop reasoning
    {
        "id": "arc_1",
        "category": "science_reasoning",
        "problem": "A student wants to determine if seeds will grow better in soil, water, or a mixture of the two. She sets up an experiment where she plants 10 bean seeds in soil, 10 bean seeds in water, and 10 bean seeds in a mixture of soil and water. She makes sure all the seeds get the same amount of sunlight and are kept at the same temperature. What is the independent variable in this experiment?",
        "benchmark": "ARC",
        "difficulty": "medium"
    },
    # BBH (Big-Bench Hard) problems - particularly challenging for models
    {
        "id": "bbh_1",
        "category": "logical_deduction",
        "problem": "Jack and 3 other people (Alex, Bob, and Charlie) are in a competition. The competition has 4 rounds. If someone loses a round, they are eliminated from the competition. Jack came in 2nd place overall, meaning he was the second-to-last person to be eliminated. Alex was eliminated in the first round. Charlie was eliminated before Bob. In which round was Jack eliminated?",
        "benchmark": "BBH",
        "difficulty": "hard"
    },
    # MATH problems - require deep mathematical reasoning and step-by-step solutions
    {
        "id": "math_1",
        "category": "algebra",
        "problem": "Find all values of x such that x² - 6x + 8 = 0.",
        "benchmark": "MATH",
        "difficulty": "easy"
    },
    {
        "id": "math_2",
        "category": "probability",
        "problem": "A bag contains 5 red marbles and 8 blue marbles. If 3 marbles are drawn at random without replacement, what is the probability that exactly 2 of them are red?",
        "benchmark": "MATH",
        "difficulty": "medium"
    },
    # HotpotQA - multi-hop question answering
    {
        "id": "hotpotqa_1",
        "category": "multi_hop_qa",
        "problem": "The director of the romantic comedy \"The Proposal\" also directed a film starring which famous actress who played in a movie about a bus that couldn't slow down?",
        "benchmark": "HotpotQA",
        "difficulty": "hard"
    },
    # StrategyQA - requires strategic thinking and multi-step reasoning
    {
        "id": "strategyqa_1",
        "category": "strategic_reasoning",
        "problem": "Would a pear sink in water? Think about the density of a pear compared to water. What happens to objects that are less dense than water? What happens to objects that are more dense than water?",
        "benchmark": "StrategyQA",
        "difficulty": "medium"
    },
    # CLUTRR - reasoning about kinship relations
    {
        "id": "clutrr_1",
        "category": "relational_reasoning",
        "problem": "James is the father of Mary. Mary is the mother of John. What is the relationship between James and John?",
        "benchmark": "CLUTRR",
        "difficulty": "easy"
    }
]

# Prompts with strong emphasis on Chinese reasoning
PROMPTS = {
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

The majority of your response should be Chinese characters. Be thorough in your Chinese reasoning."""
}

class LanguageEfficiencyTest:
    def __init__(self, problems=BENCHMARK_PROBLEMS, model="claude-3-5-sonnet-20240620"):
        self.problems = problems
        self.model = model
        self.results = []
        
    def count_syllables(self, text):
        """Estimate syllable count in English text."""
        # Basic syllable counting heuristic
        text = text.lower()
        text = re.sub(r'[^a-z]', ' ', text)
        words = text.split()
        count = 0
        
        for word in words:
            word_count = 0
            vowels = "aeiouy"
            if word[0] in vowels:
                word_count += 1
            for index in range(1, len(word)):
                if word[index] in vowels and word[index-1] not in vowels:
                    word_count += 1
            if word.endswith("e"):
                word_count -= 1
            if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
                word_count += 1
            if word_count == 0:
                word_count = 1
            count += word_count
            
        return count
    
    def calculate_entropy(self, text):
        """Calculate information entropy of the text."""
        counter = Counter(text)
        probs = [count / len(text) for count in counter.values()]
        return entropy(probs, base=2)
    
    def count_chinese_characters(self, text):
        """Count Chinese characters in text."""
        return sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    
    def get_metrics(self, text):
        """Calculate various text metrics."""
        char_count = len(text)
        word_count = len(text.split())
        chinese_char_count = self.count_chinese_characters(text)
        non_chinese_char_count = char_count - chinese_char_count
        chinese_ratio = chinese_char_count / char_count if char_count > 0 else 0
        syllable_count = self.count_syllables(text) if any(c.isalpha() for c in text) else 0
        entropy_value = self.calculate_entropy(text)
        
        # Check if the response contains the expected Chinese content
        contains_chinese = chinese_char_count > 10  # Arbitrary threshold
        
        # Estimate compression ratio (compared to typical English)
        # Using a rough heuristic: each Chinese character ≈ 2.5 English characters
        estimated_english_chars = chinese_char_count * 2.5 + non_chinese_char_count
        compression_ratio = char_count / estimated_english_chars if estimated_english_chars > 0 else 1
        
        return {
            "char_count": char_count,
            "word_count": word_count,
            "chinese_char_count": chinese_char_count,
            "chinese_ratio": chinese_ratio,
            "contains_chinese": contains_chinese,
            "syllable_count": syllable_count,
            "entropy": entropy_value,
            "compression_ratio": compression_ratio
        }
    
    def extract_english_answer(self, text, prompt_type):
        """Extract the final English answer from responses."""
        if prompt_type == "english":
            # For English, just use the last paragraph or sentence as the answer
            paragraphs = text.split('\n\n')
            return paragraphs[-1] if paragraphs else text
        
        elif prompt_type == "chinese":
            # Look for English text near the end of the response
            # This is a heuristic approach that might need refinement
            paragraphs = text.split('\n\n')
            for p in reversed(paragraphs):
                # If paragraph has mostly non-Chinese characters, it might be the answer
                if sum(1 for char in p if '\u4e00' <= char <= '\u9fff') / len(p) < 0.5:
                    return p
            return paragraphs[-1] if paragraphs else text
        
        elif prompt_type == "chinese_with_markers":
            # Look for the marked English answer
            match = re.search(r'\[ENGLISH ANSWER\]:(.*?)($|\n\n)', text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return "No marked English answer found"
    
    def run_test(self, problem, prompt_type):
        """Run a single test case with the specified prompt type."""
        prompt = PROMPTS[prompt_type]
        full_prompt = f"{prompt}\n\n{problem['problem']}"
        
        start_time = time.time()
        
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
            response_text = response.content[0].text
            
            end_time = time.time()
            
            # Extract the English answer for evaluation
            english_answer = self.extract_english_answer(response_text, prompt_type)
            
            result = {
                "problem_id": problem["id"],
                "category": problem["category"],
                "benchmark": problem["benchmark"],
                "difficulty": problem["difficulty"],
                "prompt_type": prompt_type,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                "response_time": end_time - start_time,
                "response_text": response_text,
                "extracted_answer": english_answer
            }
            
            # Add metrics for response
            metrics = self.get_metrics(response_text)
            result.update({f"response_{k}": v for k, v in metrics.items()})
            
            return result
            
        except Exception as e:
            end_time = time.time()
            print(f"Error processing {problem['id']} with {prompt_type}: {str(e)}")
            
            # Return a partial result with error information
            return {
                "problem_id": problem["id"],
                "category": problem["category"],
                "benchmark": problem["benchmark"],
                "difficulty": problem["difficulty"],
                "prompt_type": prompt_type,
                "error": str(e),
                "response_time": end_time - start_time
            }
    
    def run_all_tests(self, repetitions=1):
        """Run tests for all problems with all prompt types, optionally with repetitions."""
        for _ in range(repetitions):
            for problem in self.problems:
                for prompt_type in PROMPTS.keys():
                    try:
                        result = self.run_test(problem, prompt_type)
                        self.results.append(result)
                        print(f"Completed {problem['id']} with {prompt_type} prompt")
                        # Sleep to avoid API rate limits
                        time.sleep(2)
                    except Exception as e:
                        print(f"Error on {problem['id']} with {prompt_type}: {str(e)}")
        
        return self.results
    
    def save_results(self, filename="language_efficiency_results.json"):
        """Save test results to a file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Also save as CSV for easier analysis
        df = pd.DataFrame(self.results)
        csv_filename = filename.replace('.json', '.csv')
        df.to_csv(csv_filename, index=False)
        
        print(f"Results saved to {filename} and {csv_filename}")
        return filename
    
    def analyze_results(self):
        """Generate analysis of test results."""
        if not self.results:
            return "No results to analyze"
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Filter out error results
        success_df = df[~df.get('error', pd.Series(False)).astype(bool)]
        
        if success_df.empty:
            return "No successful results to analyze"
        
        # Calculate aggregated metrics by prompt type
        agg_columns = [
            'input_tokens', 'output_tokens', 'total_tokens', 'response_time',
            'response_entropy', 'response_char_count', 'response_chinese_char_count',
            'response_chinese_ratio', 'response_compression_ratio'
        ]
        
        # Ensure all columns exist
        for col in agg_columns:
            if col not in success_df.columns:
                success_df[col] = np.nan
        
        # Group by prompt type
        agg_by_prompt = success_df.groupby('prompt_type')[agg_columns].agg(['mean', 'std', 'median'])
        
        # Calculate token efficiency by benchmark and difficulty
        efficiency_metrics = success_df.groupby(['prompt_type', 'benchmark', 'difficulty']).agg({
            'total_tokens': 'mean',
            'response_time': 'mean',
            'response_entropy': 'mean',
            'response_compression_ratio': 'mean'
        }).reset_index()
        
        # Compare Chinese vs English token usage
        if 'english' in success_df['prompt_type'].values and 'chinese' in success_df['prompt_type'].values:
            eng_tokens = success_df[success_df['prompt_type'] == 'english']['total_tokens'].mean()
            cn_tokens = success_df[success_df['prompt_type'] == 'chinese']['total_tokens'].mean()
            token_reduction = (eng_tokens - cn_tokens) / eng_tokens * 100 if eng_tokens > 0 else 0
            
            comparison = {
                "english_avg_tokens": eng_tokens,
                "chinese_avg_tokens": cn_tokens,
                "token_reduction_percent": token_reduction,
                "is_chinese_more_efficient": cn_tokens < eng_tokens
            }
        else:
            comparison = {"error": "Missing data for comparison"}
        
        # Create visualizations if matplotlib is available
        try:
            self.create_visualizations(success_df)
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
        
        return {
            'aggregated_by_prompt': agg_by_prompt.to_dict(),
            'efficiency_by_benchmark': efficiency_metrics.to_dict(),
            'chinese_vs_english_comparison': comparison
        }
    
    def create_visualizations(self, df):
        """Create visualizations of the results."""
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
        plt.savefig('token_usage_by_prompt.png')
        
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
        plt.savefig('token_usage_by_benchmark.png')
        
        # 3. Information density (entropy per character)
        plt.figure(figsize=(10, 6))
        df['info_density'] = df['response_entropy'] / df['response_char_count']
        density_data = df.groupby('prompt_type')['info_density'].mean()
        density_data.plot(kind='bar', title='Information Density (Entropy/Character)')
        plt.ylabel('Information Density')
        plt.tight_layout()
        plt.savefig('information_density.png')
        
        # 4. Chinese character ratio
        if 'response_chinese_ratio' in df.columns:
            plt.figure(figsize=(10, 6))
            chinese_ratio = df.groupby('prompt_type')['response_chinese_ratio'].mean()
            chinese_ratio.plot(kind='bar', title='Average Chinese Character Ratio')
            plt.ylabel('Ratio of Chinese Characters')
            plt.tight_layout()
            plt.savefig('chinese_character_ratio.png')
        
        # 5. Compression ratio comparison
        if 'response_compression_ratio' in df.columns:
            plt.figure(figsize=(10, 6))
            compression = df.groupby('prompt_type')['response_compression_ratio'].mean()
            compression.plot(kind='bar', title='Estimated Compression Ratio')
            plt.ylabel('Compression Ratio')
            plt.tight_layout()
            plt.savefig('compression_ratio.png')
        
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
        plt.savefig('difficulty_impact.png')

def main():
    # Ensure API key is set
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("Please set ANTHROPIC_API_KEY environment variable")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Run the tests
    print("Starting language efficiency tests...")
    test = LanguageEfficiencyTest()
    
    # Run tests with each problem 3 times to get more reliable results
    test.run_all_tests(repetitions=1)  # Set to higher value for more reliable results
    
    # Save and analyze results
    results_file = test.save_results("results/language_efficiency_results.json")
    analysis = test.analyze_results()
    
    with open("results/language_efficiency_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print("Testing complete! Results and analysis saved to the 'results' directory.")
    
    # Print summary of findings
    if "chinese_vs_english_comparison" in analysis:
        comp = analysis["chinese_vs_english_comparison"]
        if "token_reduction_percent" in comp:
            print(f"\nSUMMARY OF FINDINGS:")
            print(f"Chinese reasoning used {comp.get('token_reduction_percent', 0):.2f}% fewer tokens than English")
            print(f"English average tokens: {comp.get('english_avg_tokens', 0):.2f}")
            print(f"Chinese average tokens: {comp.get('chinese_avg_tokens', 0):.2f}")
            
            if comp.get("is_chinese_more_efficient", False):
                print("\nChinese reasoning appears more token-efficient in this experiment.")
            else:
                print("\nChinese reasoning does NOT appear more token-efficient in this experiment.")

if __name__ == "__main__":
    main()


