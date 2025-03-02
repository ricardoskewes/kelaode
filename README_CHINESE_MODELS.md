# Chinese-Developed Models Integration

This document provides instructions for setting up and using Chinese-developed models (Deepseek and Qwen) with the language efficiency testing framework.

## Setup Instructions

### 1. Install Required Packages

```bash
pip install deepseek-ai dashscope
```

### 2. Set API Keys

Set the following environment variables:

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key_here"
export DASHSCOPE_API_KEY="your_dashscope_api_key_here"
```

### 3. Test API Connections

Test the connections to the Deepseek and Qwen APIs:

```bash
# Test Deepseek connection
python enhanced_experiment_runner.py --test-deepseek

# Test Qwen connection
python enhanced_experiment_runner.py --test-qwen
```

## Available Models

### Deepseek Models

- `deepseek:deepseek-chat` - General-purpose chat model
- `deepseek:deepseek-coder` - Code-specialized model

### Qwen Models

- `qwen:qwen-turbo` - Fast, efficient model
- `qwen:qwen-plus` - More powerful model with enhanced capabilities

## Running Experiments

The enhanced experiment runner now supports multiple model providers. When running experiments, you can specify models from different providers using the format `provider:model_name`.

Example:

```python
# Initialize test runner with models from different providers
test = EnhancedLanguageEfficiencyTest(
    models=[
        "anthropic:claude-3-5-sonnet-20240620",
        "qwen:qwen-turbo",
        "deepseek:deepseek-chat"
    ]
)
```

## Chinese Tokenization Benefits

Chinese-developed models like Deepseek and Qwen may have better tokenization for Chinese text, potentially leading to:

1. More efficient token usage for Chinese reasoning
2. Better handling of Chinese characters
3. Improved performance on tasks requiring Chinese language understanding

## Troubleshooting

### Common Issues with Deepseek API

- If you encounter authentication errors, verify your API key is correct
- Check that you're using the correct model names
- Ensure you have sufficient API credits

### Common Issues with Qwen API

- If you receive a status code other than 200, check the error message
- Verify your API key is correct
- Ensure you're using the correct model names

## API Documentation

- [Deepseek API Documentation](https://platform.deepseek.com/docs)
- [Qwen API Documentation (Dashscope)](https://help.aliyun.com/document_detail/2400395.html)
