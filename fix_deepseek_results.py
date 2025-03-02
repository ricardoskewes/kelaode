"""
Fix the path issue in the Deepseek results and save them to the correct location.
"""

import os
import json
import glob

def fix_deepseek_results():
    """
    Fix the path issue in the Deepseek results and save them to the correct location.
    """
    print("Fixing Deepseek results...")
    
    # Use the interim_results_30.json file specifically
    target_file = "experiment_results/interim_results_30.json"
    if not os.path.exists(target_file):
        print(f"Target file {target_file} not found")
        return
    
    print(f"Using target interim results: {target_file}")
    
    # Load the interim results
    try:
        with open(target_file, 'r') as f:
            results = json.load(f)
        
        # Filter for Deepseek results only
        deepseek_results = [r for r in results if r.get('model', '').startswith('deepseek:')]
        print(f"Found {len(deepseek_results)} Deepseek results")
        
        if not deepseek_results:
            print("No Deepseek results found in interim file")
            return
        
        # Save to the correct location
        output_file = "experiment_results/deepseek_longcontext_results.json"
        with open(output_file, 'w') as f:
            json.dump(deepseek_results, f, indent=2)
        
        print(f"Deepseek results saved to {output_file}")
        return deepseek_results
    except Exception as e:
        print(f"Error fixing Deepseek results: {str(e)}")
        return None

if __name__ == "__main__":
    fix_deepseek_results()
