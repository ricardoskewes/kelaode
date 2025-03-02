"""
Simple test script for Deepseek API.
"""

import os
import sys
import time

# Set the API key
os.environ["DEEPSEEK_API_KEY"] = "sk-4c3afcac96844b0e862839df22692c13"

try:
    from deepseek_ai.chat import DeepSeekChat
    
    # Print available attributes and methods
    print("DeepSeekChat attributes and methods:")
    print(dir(DeepSeekChat))
    
    # Create an instance of DeepSeekChat
    chat = DeepSeekChat()
    
    # Print available attributes and methods of the instance
    print("\nDeepSeekChat instance attributes and methods:")
    print(dir(chat))
    
    # Print available attributes and methods of chat.completions
    print("\nDeepSeekChat.completions attributes and methods:")
    print(dir(chat.completions))
    
    # Test a simple query
    start_time = time.time()
    
    print("\nSending test query to Deepseek API...")
    response = chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": "Hello, can you respond in Chinese?"}
        ]
    )
    
    end_time = time.time()
    
    # Print the response
    print("Deepseek API connection successful!")
    print(f"Response: {response}")
    print(f"Response time: {end_time - start_time:.2f} seconds")
    
    # Try to access response attributes
    print("\nResponse attributes and methods:")
    print(dir(response))
    
except Exception as e:
    print(f"Error testing Deepseek connection: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
