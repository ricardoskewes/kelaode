"""
Verify the accuracy of findings in the condensed_findings.md report.
"""

import json
import pandas as pd
import os
import glob

def verify_deepseek_results():
    """
    Verify the Deepseek results and efficiency claims in the report.
    """
    print("Verifying Deepseek results...")
    
    # Load Deepseek results
    try:
        with open("experiment_results/deepseek_longcontext_results.json", 'r') as f:
            deepseek_results = json.load(f)
        
        print(f"Loaded {len(deepseek_results)} Deepseek results")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(deepseek_results)
        
        # Calculate efficiency metrics
        english_tokens = df[df['prompt_type'] == 'english']['total_tokens'].mean()
        print(f"English tokens: {english_tokens:.2f}")
        
        # Check efficiency for each language
        for lang in ['chinese', 'strategic', 'russian', 'german']:
            lang_tokens = df[df['prompt_type'] == lang]['total_tokens'].mean()
            efficiency = (english_tokens - lang_tokens) / english_tokens * 100
            print(f"{lang} tokens: {lang_tokens:.2f}, efficiency: {efficiency:.2f}%")
        
        # Verify against report claims
        print("\nVerifying against report claims:")
        print("Reported Chinese efficiency: 18.03% (should be close to actual)")
        print("Reported Strategic efficiency: 9.57% (should be close to actual)")
        print("Reported Russian efficiency: 8.21% (should be close to actual)")
        print("Reported German efficiency: 6.23% (should be close to actual)")
        
        return df
    except Exception as e:
        print(f"Error verifying Deepseek results: {str(e)}")
        return None

def verify_visualizations():
    """
    Verify that all visualizations referenced in the report exist.
    """
    print("\nVerifying visualizations...")
    
    # Check Deepseek visualizations
    deepseek_viz = glob.glob("reports/visualizations/deepseek/*.png")
    print(f"Found {len(deepseek_viz)} Deepseek visualizations:")
    for viz in deepseek_viz:
        print(f"  - {os.path.basename(viz)}")
    
    # Check model comparison visualizations
    model_viz = glob.glob("reports/visualizations/model_comparison/*.png")
    print(f"Found {len(model_viz)} model comparison visualizations:")
    for viz in model_viz:
        print(f"  - {os.path.basename(viz)}")
    
    # Check presentation visualizations
    pres_viz = glob.glob("reports/visualizations/presentation/*.png")
    print(f"Found {len(pres_viz)} presentation visualizations:")
    for viz in pres_viz:
        print(f"  - {os.path.basename(viz)}")
    
    # Verify against report references
    print("\nVerifying against report references:")
    required_viz = [
        "deepseek_efficiency.png",
        "deepseek_problem_tokens.png",
        "token_usage_by_model_language.png",
        "efficiency_by_model.png",
        "token_usage_by_language.png",
        "efficiency_vs_english.png",
        "language_radar_chart.png",
        "strategic_selection_efficiency.png"
    ]
    
    all_viz = [os.path.basename(viz) for viz in deepseek_viz + model_viz + pres_viz]
    missing_viz = [viz for viz in required_viz if viz not in all_viz]
    
    if missing_viz:
        print(f"Missing visualizations: {missing_viz}")
    else:
        print("All required visualizations exist")

if __name__ == "__main__":
    # Verify Deepseek results
    df = verify_deepseek_results()
    
    # Verify visualizations
    verify_visualizations()
