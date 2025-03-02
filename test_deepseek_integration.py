"""
Test script for Deepseek API integration.
"""

import os
import sys
sys.path.append('/home/ubuntu/repos/kelaode')
from enhanced_experiment_runner import test_deepseek_connection

if __name__ == "__main__":
    # Set the API key
    os.environ["DEEPSEEK_API_KEY"] = "sk-4c3afcac96844b0e862839df22692c13"
    
    # Run the test
    result = test_deepseek_connection()
    
    if result:
        print("\nDeepseek API integration test passed!")
    else:
        print("\nDeepseek API integration test failed!")
