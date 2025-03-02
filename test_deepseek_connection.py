"""
Test script to verify Deepseek API connection and balance.
"""

import os
import json
import time
import requests

def test_deepseek_connection():
    """
    Test the connection to the Deepseek API and check balance.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    
    if not api_key:
        print("DEEPSEEK_API_KEY not set. Cannot test Deepseek connection.")
        return False
    
    print(f"Testing Deepseek API connection with key: {api_key[:5]}...{api_key[-5:]}")
    
    # API endpoint
    url = "https://api.deepseek.com/v1/chat/completions"
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Request payload
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, can you respond in Chinese?"}
        ],
        "max_tokens": 100
    }
    
    try:
        # Send request
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload)
        end_time = time.time()
        
        # Check response
        if response.status_code == 200:
            response_json = response.json()
            print("\nDeepseek API connection successful!")
            print(f"Response time: {end_time - start_time:.2f} seconds")
            print(f"Model: {response_json.get('model', 'Unknown')}")
            print(f"Usage: {json.dumps(response_json.get('usage', {}), indent=2)}")
            print(f"Response: {response_json.get('choices', [{}])[0].get('message', {}).get('content', 'No content')}")
            return True
        else:
            print(f"\nError: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"\nException: {str(e)}")
        return False

if __name__ == "__main__":
    test_deepseek_connection()
