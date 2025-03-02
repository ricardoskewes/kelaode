"""
Test script for Deepseek API using the chat_completion method.
"""

import os
import time
from deepseek import DeepSeekAPI

def test_deepseek_connection():
    """
    Test the connection to the Deepseek API using the chat_completion method.
    """
    # Set the API key
    api_key = "sk-4c3afcac96844b0e862839df22692c13"
    os.environ["DEEPSEEK_API_KEY"] = api_key
    
    try:
        # Initialize the Deepseek client
        client = DeepSeekAPI(api_key=api_key)
        
        # Test a simple query
        start_time = time.time()
        
        print("Sending test query to Deepseek API...")
        response = client.chat_completion(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "Hello, can you respond in Chinese?"}
            ]
        )
        
        end_time = time.time()
        
        # Print the response
        print("Deepseek API connection successful!")
        print(f"Response: {response}")
        
        # Try to extract the message content if available
        if hasattr(response, 'choices') and len(response.choices) > 0:
            if hasattr(response.choices[0], 'message'):
                print(f"Message content: {response.choices[0].message.content}")
        
        print(f"Response time: {end_time - start_time:.2f} seconds")
        
        # Try to get token usage if available
        if hasattr(response, 'usage'):
            print(f"Input tokens: {response.usage.prompt_tokens}")
            print(f"Output tokens: {response.usage.completion_tokens}")
            print(f"Total tokens: {response.usage.total_tokens}")
        
        return True
    except Exception as e:
        print(f"Error testing Deepseek connection: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_deepseek_connection()
