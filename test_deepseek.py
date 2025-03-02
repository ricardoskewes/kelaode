"""
Test script for Deepseek API integration.
"""

import os
import time
from deepseek import DeepSeekAPI

def test_deepseek_connection():
    """
    Test the connection to the Deepseek API.
    """
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("DEEPSEEK_API_KEY not set. Cannot test Deepseek connection.")
        return False
    
    try:
        # Initialize the Deepseek client
        client = DeepSeekAPI(api_key=os.environ.get("DEEPSEEK_API_KEY"))
        
        # Test a simple query
        start_time = time.time()
        
        response = client.chat_completion(
            prompt="Hello, can you respond in Chinese?",
            prompt_sys="You are a helpful assistant",
            model="deepseek-chat"
        )
        
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
    test_deepseek_connection()
