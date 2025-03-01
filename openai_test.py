import os
from openai import OpenAI

# Get API key from environment variable or set directly as fallback
key = os.environ.get("OPENAI_API_KEY")
    # Fallback to direct API key if environment variable is not set
client = OpenAI(api_key=key)

# tools = [{
#     "type": "function",
#     "function": {
#         "name": "get_weather",
#         "description": "Get current temperature for a given location.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "location": {
#                     "type": "string",
#                     "description": "City and country e.g. Bogotá, Colombia"
#                 }
#             },
#             "required": [
#                 "location"
#             ],
#             "additionalProperties": False
#         },
#         "strict": True
#     }
# }]

# completion = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[{"role": "user", "content": "What is the weather like in Paris today?"}],
#     tools=tools
# )

# # Print content if available, otherwise print tool calls
# if completion.choices[0].message.content:
#     print("Content:", completion.choices[0].message.content)
# else:
#     print("Tool Calls:", completion.choices[0].message.tool_calls)

# Make a second API call without tools to get a text response

# API calls.

def API_call_example():
    text_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is the weather like in Paris today?"}]
    )

    print("GPT-4o Response:", text_completion.choices[0].message.content)

def translate(input_prompt, intermediate_lang):
    instructions = f"Please translate {input_prompt} into {intermediate_lang}"
    text_completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": instructions}]
    )
    intermediate_prompt = text_completion.choices[0].message.content
    return intermediate_prompt


def graph_compression():
    pass
 

def graph_entropy():
    pass

if __name__ == "__main__":

    intermediate_lang_options = ['chinese', 'japanese', 'russian']
    intermediate_lang = intermediate_lang_options[0]
    input_prompt = "Built to help ambitious engineering teams achieve more"
    intermediate_prompt = translate(input_prompt, intermediate_lang)
    print(intermediate_prompt)
