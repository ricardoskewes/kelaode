"""
Test script for Deepseek API integration.
"""

import os
import time
from deepseek_ai import chat

def test_deepseek_connection():
    """
    Test the connection to the Deepseek API.
    """
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("DEEPSEEK_API_KEY not set. Cannot test Deepseek connection.")
        return False
    
    try:
        # Initialize the Deepseek client
        client = chat.Chat(api_key=os.environ.get("DEEPSEEK_API_KEY"))
        
        # Test a simple query
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "Hello, can you respond in Chinese?"}
            ]
        )
        
        end_time = time.time()
        
        # Print the response
        print("Deepseek API connection successful!")
        print(f"Response: {response.choices[0].message.content}")
        print(f"Response time: {end_time - start_time:.2f} seconds")
        
        # Try to get token usage if available
        if hasattr(response, 'usage'):
            print(f"Input tokens: {response.usage.prompt_tokens}")
            print(f"Output tokens: {response.usage.completion_tokens}")
            print(f"Total tokens: {response.usage.total_tokens}")
        
        return True
    except Exception as e:
        print(f"Error testing Deepseek connection: {str(e)}")
        return False

if __name__ == "__main__":
    test_deepseek_connection()
