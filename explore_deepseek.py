"""
Explore the DeepSeekAPI object to understand its structure.
"""

import os
import sys
from deepseek import DeepSeekAPI

# Set the API key
api_key = "sk-4c3afcac96844b0e862839df22692c13"
os.environ["DEEPSEEK_API_KEY"] = api_key

try:
    # Initialize the Deepseek client
    client = DeepSeekAPI(api_key=api_key)
    
    # Print available attributes and methods
    print("DeepSeekAPI attributes and methods:")
    print(dir(client))
    
    # Try to explore the client structure
    print("\nExploring client structure:")
    
    # Check if client has chat attribute
    if hasattr(client, 'chat'):
        print("client.chat exists")
        print(dir(client.chat))
    else:
        print("client.chat does not exist")
    
    # Check if client has completions attribute
    if hasattr(client, 'completions'):
        print("client.completions exists")
        print(dir(client.completions))
    else:
        print("client.completions does not exist")
    
    # Try to make a simple API call
    print("\nAttempting to make a simple API call:")
    
    # Try different possible API call patterns
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hello"}]
        )
        print("Success with client.chat.completions.create")
    except Exception as e:
        print(f"Error with client.chat.completions.create: {str(e)}")
    
    try:
        response = client.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hello"}]
        )
        print("Success with client.completions.create")
    except Exception as e:
        print(f"Error with client.completions.create: {str(e)}")
    
    try:
        response = client.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hello"}]
        )
        print("Success with client.create")
    except Exception as e:
        print(f"Error with client.create: {str(e)}")
    
except Exception as e:
    print(f"Error exploring DeepSeekAPI: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
