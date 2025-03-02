"""
Strategic language selection algorithm for optimizing chain-of-thought reasoning efficiency.

This module implements algorithms to select the most efficient language for different
problem types based on domain, category, and difficulty level.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class StrategicLanguageSelector:
    """
    A class for selecting the most efficient language for chain-of-thought reasoning
    based on problem characteristics.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the strategic language selector.
        
        Args:
            model_path: Path to a pre-trained model file (optional)
        """
        self.rules = {}
        self.decision_tree = None
        self.feature_encoders = {}
        self.language_encoder = None
        
        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_results(self, results_file):
        """
        Load experiment results for training the selector.
        
        Args:
            results_file: Path to the results file
        """
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame(results)
        
        return self.results_df
    
    def load_analysis_results(self, analysis_dir="analysis_results"):
        """
        Load analysis results for rule-based selection.
        
        Args:
            analysis_dir: Directory containing analysis results
        """
        # Find the latest analysis files
        benchmark_files = [f for f in os.listdir(analysis_dir) if f.startswith("benchmark_selection_rules_")]
        category_files = [f for f in os.listdir(analysis_dir) if f.startswith("category_selection_rules_")]
        difficulty_files = [f for f in os.listdir(analysis_dir) if f.startswith("difficulty_selection_rules_")]
        
        # Sort by timestamp
        benchmark_files.sort(reverse=True)
        category_files.sort(reverse=True)
        difficulty_files.sort(reverse=True)
        
        # Load the latest files
        if benchmark_files:
            benchmark_rules = pd.read_csv(os.path.join(analysis_dir, benchmark_files[0]), index_col=0)
            self.rules['benchmark'] = benchmark_rules['language'].to_dict()
        
        if category_files:
            category_rules = pd.read_csv(os.path.join(analysis_dir, category_files[0]), index_col=0)
            self.rules['category'] = category_rules['language'].to_dict()
        
        if difficulty_files:
            difficulty_rules = pd.read_csv(os.path.join(analysis_dir, difficulty_files[0]), index_col=0)
            self.rules['difficulty'] = difficulty_rules['language'].to_dict()
        
        return self.rules
    
    def train_decision_tree(self, test_size=0.2, random_state=42):
        """
        Train a decision tree model for language selection.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Accuracy of the trained model
        """
        if not hasattr(self, 'results_df'):
            raise ValueError("No results data loaded. Call load_results() first.")
        
        # Prepare data for training
        problem_data = []
        
        for problem_id in self.results_df['problem_id'].unique():
            problem_df = self.results_df[self.results_df['problem_id'] == problem_id]
            
            # Get problem metadata
            benchmark = problem_df['benchmark'].iloc[0]
            category = problem_df['category'].iloc[0] if 'category' in problem_df.columns else 'unknown'
            difficulty = problem_df['difficulty'].iloc[0]
            
            # Calculate efficiency for each language
            english_df = problem_df[problem_df['prompt_type'] == 'english']
            if english_df.empty:
                continue
                
            english_tokens = english_df['total_tokens'].mean()
            
            best_language = 'english'
            best_efficiency = 0
            
            for lang in self.results_df['prompt_type'].unique():
                if lang == 'english':
                    continue
                    
                lang_df = problem_df[problem_df['prompt_type'] == lang]
                if lang_df.empty:
                    continue
                    
                lang_tokens = lang_df['total_tokens'].mean()
                efficiency = (english_tokens - lang_tokens) / english_tokens * 100
                
                if efficiency > best_efficiency:
                    best_language = lang
                    best_efficiency = efficiency
            
            problem_data.append({
                'problem_id': problem_id,
                'benchmark': benchmark,
                'category': category,
                'difficulty': difficulty,
                'best_language': best_language
            })
        
        # Create DataFrame
        problem_df = pd.DataFrame(problem_data)
        
        # Encode categorical features
        self.feature_encoders = {}
        for feature in ['benchmark', 'category', 'difficulty']:
            encoder = LabelEncoder()
            problem_df[f'{feature}_encoded'] = encoder.fit_transform(problem_df[feature])
            self.feature_encoders[feature] = encoder
        
        # Encode target
        self.language_encoder = LabelEncoder()
        problem_df['best_language_encoded'] = self.language_encoder.fit_transform(problem_df['best_language'])
        
        # Prepare features and target
        X = problem_df[['benchmark_encoded', 'category_encoded', 'difficulty_encoded']]
        y = problem_df['best_language_encoded']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Train decision tree
        self.decision_tree = DecisionTreeClassifier(max_depth=3, random_state=random_state)
        self.decision_tree.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.decision_tree.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Decision tree model trained with accuracy: {accuracy:.2f}")
        
        return accuracy
    
    def save_model(self, model_path="models/strategic_language_selector.json"):
        """
        Save the trained model and rules.
        
        Args:
            model_path: Path to save the model
        """
        if not self.decision_tree and not self.rules:
            raise ValueError("No model or rules to save. Train a model or load rules first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Prepare model data
        model_data = {
            'rules': self.rules
        }
        
        # Save model data
        with open(model_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the model file
        """
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        # Load rules
        self.rules = model_data.get('rules', {})
        
        print(f"Model loaded from {model_path}")
    
    def select_language(self, problem, method='rule'):
        """
        Select the most efficient language for a given problem.
        
        Args:
            problem: Dictionary with problem metadata (benchmark, category, difficulty)
            method: Selection method ('rule', 'tree', or 'hybrid')
            
        Returns:
            The most efficient language for the problem
        """
        if method == 'rule':
            return self._rule_based_selection(problem)
        elif method == 'tree':
            return self._tree_based_selection(problem)
        elif method == 'hybrid':
            return self._hybrid_selection(problem)
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def _rule_based_selection(self, problem):
        """
        Select language using rule-based approach.
        
        Args:
            problem: Dictionary with problem metadata
            
        Returns:
            Selected language
        """
        # Check if we have benchmark-specific rules
        if 'benchmark' in self.rules and problem.get('benchmark') in self.rules['benchmark']:
            return self.rules['benchmark'][problem['benchmark']]
        
        # Check if we have category-specific rules
        if 'category' in self.rules and problem.get('category') in self.rules['category']:
            return self.rules['category'][problem['category']]
        
        # Check if we have difficulty-specific rules
        if 'difficulty' in self.rules and problem.get('difficulty') in self.rules['difficulty']:
            return self.rules['difficulty'][problem['difficulty']]
        
        # Default to English if no rules match
        return 'english'
    
    def _tree_based_selection(self, problem):
        """
        Select language using decision tree model.
        
        Args:
            problem: Dictionary with problem metadata
            
        Returns:
            Selected language
        """
        if not self.decision_tree:
            raise ValueError("No decision tree model trained. Call train_decision_tree() first.")
        
        # Encode features
        features = []
        for feature in ['benchmark', 'category', 'difficulty']:
            if feature in problem and feature in self.feature_encoders:
                # Handle unseen categories
                try:
                    encoded_value = self.feature_encoders[feature].transform([problem[feature]])[0]
                except ValueError:
                    # Use the most common value for unseen categories
                    encoded_value = self.feature_encoders[feature].transform([self.feature_encoders[feature].classes_[0]])[0]
            else:
                # Use the most common value if feature is missing
                encoded_value = 0
            
            features.append(encoded_value)
        
        # Make prediction
        language_idx = self.decision_tree.predict([features])[0]
        language = self.language_encoder.inverse_transform([language_idx])[0]
        
        return language
    
    def _hybrid_selection(self, problem):
        """
        Select language using a hybrid approach (rules + tree).
        
        Args:
            problem: Dictionary with problem metadata
            
        Returns:
            Selected language
        """
        # Try rule-based selection first
        rule_language = self._rule_based_selection(problem)
        
        # If rule-based selection returns English and we have a tree model,
        # try tree-based selection
        if rule_language == 'english' and self.decision_tree:
            tree_language = self._tree_based_selection(problem)
            return tree_language
        
        return rule_language
    
    def analyze_problem(self, problem_text):
        """
        Analyze a problem to extract its characteristics.
        
        Args:
            problem_text: The text of the problem
            
        Returns:
            Dictionary with problem characteristics
        """
        # This is a placeholder for a more sophisticated analysis
        # In a real implementation, this would use NLP techniques to classify the problem
        
        # Simple keyword-based classification
        problem_lower = problem_text.lower()
        
        # Detect benchmark
        benchmark = 'unknown'
        if any(kw in problem_lower for kw in ['equation', 'solve for', 'calculate', 'compute']):
            benchmark = 'MATH'
        elif any(kw in problem_lower for kw in ['explain', 'why', 'how come']):
            benchmark = 'ARC-Challenge'
        elif any(kw in problem_lower for kw in ['story', 'narrative', 'passage']):
            benchmark = 'HotpotQA'
        elif any(kw in problem_lower for kw in ['context', 'long text', 'article']):
            benchmark = 'LongContextQA'
        
        # Detect category
        category = 'unknown'
        if any(kw in problem_lower for kw in ['algebra', 'equation', 'solve for']):
            category = 'algebra'
        elif any(kw in problem_lower for kw in ['geometry', 'triangle', 'circle', 'angle']):
            category = 'geometry'
        elif any(kw in problem_lower for kw in ['probability', 'chance', 'likelihood']):
            category = 'probability'
        elif any(kw in problem_lower for kw in ['calculus', 'derivative', 'integral']):
            category = 'calculus'
        elif any(kw in problem_lower for kw in ['history', 'historical', 'past events']):
            category = 'historical'
        elif any(kw in problem_lower for kw in ['science', 'scientific', 'experiment']):
            category = 'scientific'
        elif any(kw in problem_lower for kw in ['economics', 'financial', 'economy']):
            category = 'economic'
        
        # Detect difficulty
        difficulty = 'medium'
        word_count = len(problem_text.split())
        if word_count > 100 or 'prove' in problem_lower or 'complex' in problem_lower:
            difficulty = 'hard'
        elif word_count < 50 and not any(kw in problem_lower for kw in ['prove', 'explain', 'why']):
            difficulty = 'easy'
        
        # Estimate context length
        context_length = len(problem_text)
        
        return {
            'benchmark': benchmark,
            'category': category,
            'difficulty': difficulty,
            'context_length': context_length
        }

def train_strategic_language_selector():
    """
    Train and save a strategic language selector model.
    """
    # Initialize selector
    selector = StrategicLanguageSelector()
    
    # Load analysis results
    print("Loading analysis results...")
    selector.load_analysis_results()
    
    # Load experiment results
    print("Loading experiment results...")
    results_file = "experiment_results/interim_results_50.json"
    if os.path.exists(results_file):
        selector.load_results(results_file)
        
        # Train decision tree model
        print("Training decision tree model...")
        selector.train_decision_tree()
    else:
        print(f"Warning: Results file {results_file} not found. Skipping decision tree training.")
    
    # Save model
    print("Saving model...")
    selector.save_model()
    
    return selector

def select_language_for_problem(problem, context_length=None):
    """
    Select the most efficient language for a given problem based on its characteristics.
    
    Args:
        problem: Dictionary with problem characteristics (benchmark, category, difficulty)
        context_length: Optional context length for long-context problems
        
    Returns:
        The selected language
    """
    # Extract problem characteristics
    benchmark = problem.get('benchmark', 'unknown')
    category = problem.get('category', 'unknown')
    difficulty = problem.get('difficulty', 'medium')
    
    # Consider context length if provided
    if context_length is not None:
        # For very long contexts (10K+ chars), prefer languages with high compression
        if context_length > 10000:
            if category == "mathematical" or benchmark == "MATH":
                return "chinese"
            elif category == "logical" or benchmark == "BBH":
                return "german"
            elif category == "scientific" or benchmark == "ARC-Challenge":
                return "russian"
            else:
                return "english"
        # For medium-long contexts (5K-10K chars), use domain-specific selection
        elif context_length > 5000:
            if category == "mathematical" or benchmark == "MATH":
                return "chinese"
            elif category == "logical" or benchmark == "BBH":
                return "german"
            elif category == "scientific" or benchmark == "ARC-Challenge":
                return "russian"
            elif category == "historical":
                return "chinese"
            elif category == "economic":
                return "german"
            else:
                return "english"
    
    # For regular problems, use the existing selection logic
    if benchmark == "MATH" or category == "mathematical":
        return "chinese"
    elif benchmark == "BBH" or category == "logical":
        return "german"
    elif benchmark == "ARC-Challenge" or category == "scientific":
        return "russian"
    elif benchmark == "HotpotQA" or category == "reading_comprehension":
        return "english"
    elif difficulty == "hard":
        return "english"
    elif difficulty == "medium":
        return "chinese"
    else:
        return "english"

def select_most_efficient_language(problem_text, method='hybrid'):
    """
    Select the most efficient language for a given problem.
    
    Args:
        problem_text: The text of the problem
        method: Selection method ('rule', 'tree', or 'hybrid')
        
    Returns:
        The most efficient language for the problem
    """
    # Initialize selector
    model_path = "models/strategic_language_selector.json"
    if os.path.exists(model_path):
        selector = StrategicLanguageSelector(model_path)
    else:
        # Train a new model if one doesn't exist
        selector = train_strategic_language_selector()
    
    # Analyze problem
    problem_characteristics = selector.analyze_problem(problem_text)
    
    # Get context length if available
    context_length = problem_characteristics.get('context_length')
    
    # Use context-aware selection if context length is significant
    if context_length and context_length > 5000:
        language = select_language_for_problem(problem_characteristics, context_length)
    else:
        # Select language using the specified method
        language = selector.select_language(problem_characteristics, method)
    
    return language, problem_characteristics

if __name__ == "__main__":
    # Train and save model
    selector = train_strategic_language_selector()
    
    # Test with some example problems
    test_problems = [
        "Solve for x: 3x + 5 = 17",
        "In a triangle ABC, angle A is 60 degrees, angle B is 45 degrees. What is angle C?",
        "A bag contains 5 red balls and 3 blue balls. What is the probability of drawing a red ball?",
        "Explain why water expands when it freezes.",
        "Read the following passage and answer the question: The Treaty of Versailles was signed in 1919. Who were the main signatories?"
    ]
    
    print("\nTesting language selection with example problems:")
    print("================================================")
    
    for problem in test_problems:
        language, characteristics = select_most_efficient_language(problem)
        print(f"\nProblem: {problem[:50]}...")
        print(f"Characteristics: {characteristics}")
        print(f"Selected language: {language}")
