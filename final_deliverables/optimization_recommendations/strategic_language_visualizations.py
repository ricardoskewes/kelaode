"""
Create visualizations for strategic language selection.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

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

def create_strategic_language_selection_heatmap(df, output_dir="reports/visualizations/strategic"):
    """
    Create a heatmap showing which language is most efficient for each domain and difficulty.
    
    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique benchmarks, categories, and difficulties
    benchmarks = df['benchmark'].unique()
    categories = df['category'].unique() if 'category' in df.columns else []
    difficulties = df['difficulty'].unique()
    
    # Create a DataFrame to store the best language for each benchmark and difficulty
    best_language_by_benchmark_difficulty = pd.DataFrame(index=benchmarks, columns=difficulties)
    best_efficiency_by_benchmark_difficulty = pd.DataFrame(index=benchmarks, columns=difficulties)
    
    # Find the best language for each benchmark and difficulty
    for benchmark in benchmarks:
        benchmark_df = df[df['benchmark'] == benchmark]
        
        for difficulty in difficulties:
            difficulty_df = benchmark_df[benchmark_df['difficulty'] == difficulty]
            
            if difficulty_df.empty:
                continue
            
            # Calculate efficiency for each language
            english_df = difficulty_df[difficulty_df['prompt_type'] == 'english']
            if english_df.empty:
                continue
                
            english_tokens = english_df['total_tokens'].mean()
            
            best_language = None
            best_efficiency = 0
            
            for lang in df['prompt_type'].unique():
                if lang == 'english':
                    continue
                    
                lang_df = difficulty_df[difficulty_df['prompt_type'] == lang]
                if lang_df.empty:
                    continue
                    
                lang_tokens = lang_df['total_tokens'].mean()
                efficiency = (english_tokens - lang_tokens) / english_tokens * 100
                
                if efficiency > best_efficiency:
                    best_language = lang
                    best_efficiency = efficiency
            
            if best_language:
                best_language_by_benchmark_difficulty.loc[benchmark, difficulty] = best_language
                best_efficiency_by_benchmark_difficulty.loc[benchmark, difficulty] = best_efficiency
            else:
                best_language_by_benchmark_difficulty.loc[benchmark, difficulty] = 'english'
                best_efficiency_by_benchmark_difficulty.loc[benchmark, difficulty] = 0
    
    # Create heatmap for best language
    plt.figure(figsize=(12, 8))
    
    # Create a custom colormap for languages
    languages = df['prompt_type'].unique()
    language_colors = sns.color_palette("husl", len(languages))
    language_cmap = {lang: color for lang, color in zip(languages, language_colors)}
    
    # Convert language names to numeric values for heatmap
    language_encoder = LabelEncoder()
    language_encoder.fit(languages)
    
    # Create a numeric matrix for the heatmap
    heatmap_data = best_language_by_benchmark_difficulty.copy()
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            if pd.notna(heatmap_data.iloc[i, j]):
                heatmap_data.iloc[i, j] = language_encoder.transform([heatmap_data.iloc[i, j]])[0]
            else:
                heatmap_data.iloc[i, j] = -1
    
    # Create heatmap
    ax = plt.subplot(111)
    heatmap = sns.heatmap(heatmap_data.astype(float), cmap='viridis', annot=best_language_by_benchmark_difficulty, 
                          fmt='', cbar=False, linewidths=.5)
    
    # Add labels and title
    plt.xlabel('Difficulty')
    plt.ylabel('Benchmark')
    plt.title('Strategic Language Selection by Benchmark and Difficulty')
    
    # Create a custom legend for languages
    handles = []
    for lang in languages:
        if lang in best_language_by_benchmark_difficulty.values:
            color = language_cmap[lang]
            handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=lang))
    
    plt.legend(handles=handles, title='Most Efficient Language', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/strategic_language_selection_heatmap.png')
    plt.close()
    
    # Create heatmap for efficiency gain
    plt.figure(figsize=(12, 8))
    
    # Create a custom colormap (green for better efficiency)
    cmap = LinearSegmentedColormap.from_list('efficiency_cmap', ['#ffffff', '#99ff99'], N=100)
    
    # Create heatmap
    sns.heatmap(best_efficiency_by_benchmark_difficulty.astype(float), cmap=cmap, annot=True, 
                fmt='.1f', linewidths=.5)
    
    # Add labels and title
    plt.xlabel('Difficulty')
    plt.ylabel('Benchmark')
    plt.title('Efficiency Gain (%) of Best Language vs. English')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/strategic_language_efficiency_gain_heatmap.png')
    plt.close()
    
    return best_language_by_benchmark_difficulty, best_efficiency_by_benchmark_difficulty

def create_domain_specific_language_efficiency_chart(df, output_dir="reports/visualizations/strategic"):
    """
    Create a bar chart showing language efficiency by domain.
    
    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique categories
    if 'category' in df.columns:
        categories = df['category'].unique()
    else:
        categories = df['benchmark'].unique()
        category_column = 'benchmark'
    
    # Create a figure with subplots for each category
    fig, axes = plt.subplots(len(categories), 1, figsize=(14, 4 * len(categories)))
    
    # If only one category, make axes iterable
    if len(categories) == 1:
        axes = [axes]
    
    # For each category, create a bar chart of language efficiency
    for i, category in enumerate(categories):
        if 'category' in df.columns:
            category_df = df[df['category'] == category]
        else:
            category_df = df[df[category_column] == category]
        
        # Calculate efficiency for each language
        english_df = category_df[category_df['prompt_type'] == 'english']
        if english_df.empty:
            continue
            
        english_tokens = english_df['total_tokens'].mean()
        
        # Calculate efficiency for each language
        efficiency_data = []
        
        for lang in df['prompt_type'].unique():
            if lang == 'english':
                continue
                
            lang_df = category_df[category_df['prompt_type'] == lang]
            if lang_df.empty:
                continue
                
            lang_tokens = lang_df['total_tokens'].mean()
            efficiency = (english_tokens - lang_tokens) / english_tokens * 100
            
            efficiency_data.append({
                'language': lang,
                'efficiency_gain': efficiency
            })
        
        # Create DataFrame
        efficiency_df = pd.DataFrame(efficiency_data)
        
        # Sort by efficiency gain
        efficiency_df = efficiency_df.sort_values('efficiency_gain', ascending=False)
        
        # Create bar chart
        bars = axes[i].bar(efficiency_df['language'], efficiency_df['efficiency_gain'])
        
        # Color bars based on efficiency (green for positive, red for negative)
        for j, bar in enumerate(bars):
            bar.set_color('green' if efficiency_df.iloc[j]['efficiency_gain'] > 0 else 'red')
        
        # Add labels and title
        axes[i].set_xlabel('Language')
        axes[i].set_ylabel('Efficiency Gain vs. English (%)')
        axes[i].set_title(f'Language Efficiency for {category}')
        
        # Add horizontal line at y=0
        axes[i].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for j, v in enumerate(efficiency_df['efficiency_gain']):
            axes[i].text(j, v + (1 if v >= 0 else -3), f"{v:.1f}%", 
                     ha='center', va='bottom' if v >= 0 else 'top')
        
        # Rotate x-axis labels
        axes[i].set_xticklabels(efficiency_df['language'], rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/domain_specific_language_efficiency.png')
    plt.close()
    
    return efficiency_df

def create_language_selection_decision_tree(df, output_dir="reports/visualizations/strategic"):
    """
    Create a decision tree visualization for language selection.
    
    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for decision tree
    # We'll use benchmark, category, and difficulty as features
    # and the most efficient language as the target
    
    # Calculate the most efficient language for each problem
    problem_data = []
    
    for problem_id in df['problem_id'].unique():
        problem_df = df[df['problem_id'] == problem_id]
        
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
        
        for lang in df['prompt_type'].unique():
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
    benchmark_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    difficulty_encoder = LabelEncoder()
    language_encoder = LabelEncoder()
    
    problem_df['benchmark_encoded'] = benchmark_encoder.fit_transform(problem_df['benchmark'])
    problem_df['category_encoded'] = category_encoder.fit_transform(problem_df['category'])
    problem_df['difficulty_encoded'] = difficulty_encoder.fit_transform(problem_df['difficulty'])
    problem_df['best_language_encoded'] = language_encoder.fit_transform(problem_df['best_language'])
    
    # Prepare features and target
    X = problem_df[['benchmark_encoded', 'category_encoded', 'difficulty_encoded']]
    y = problem_df['best_language_encoded']
    
    # Train decision tree
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    
    # Create decision tree visualization
    plt.figure(figsize=(20, 10))
    
    # Get feature and class names
    feature_names = ['Benchmark', 'Category', 'Difficulty']
    class_names = language_encoder.classes_
    
    # Plot decision tree
    plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
    
    plt.title('Language Selection Decision Tree')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/language_selection_decision_tree.png')
    plt.close()
    
    # Create feature importance visualization
    plt.figure(figsize=(10, 6))
    
    # Plot feature importance
    importance = clf.feature_importances_
    indices = np.argsort(importance)
    
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for Language Selection')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/language_selection_feature_importance.png')
    plt.close()
    
    return clf, problem_df

def create_all_strategic_visualizations():
    """
    Create all strategic language selection visualizations.
    """
    # Load latest results
    df = load_latest_results()
    
    if df is None:
        print("No results to analyze.")
        return
    
    print(f"Loaded {len(df)} results.")
    
    # Create output directory
    output_dir = "reports/visualizations/strategic"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create strategic language selection heatmap
    print("Creating strategic language selection heatmap...")
    best_language_by_benchmark_difficulty, best_efficiency_by_benchmark_difficulty = create_strategic_language_selection_heatmap(df, output_dir)
    
    # Create domain-specific language efficiency chart
    print("Creating domain-specific language efficiency chart...")
    efficiency_df = create_domain_specific_language_efficiency_chart(df, output_dir)
    
    # Create language selection decision tree
    print("Creating language selection decision tree...")
    clf, problem_df = create_language_selection_decision_tree(df, output_dir)
    
    print(f"All strategic visualizations created and saved to {output_dir}")

if __name__ == "__main__":
    create_all_strategic_visualizations()
