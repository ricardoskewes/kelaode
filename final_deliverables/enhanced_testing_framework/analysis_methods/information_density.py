"""
Information density metrics for language efficiency analysis.
"""

import numpy as np
from scipy.stats import entropy
import re
import math

def calculate_bits_per_token(text, token_count):
    """
    Calculate information density in bits per token.
    
    Args:
        text: The text to analyze
        token_count: Number of tokens in the text
        
    Returns:
        Bits per token value
    """
    # Calculate Shannon entropy in bits
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    probabilities = [count / len(text) for count in char_counts.values()]
    text_entropy = entropy(probabilities, base=2)
    
    # Calculate total information content in bits
    total_bits = text_entropy * len(text)
    
    # Calculate bits per token
    if token_count > 0:
        bits_per_token = total_bits / token_count
    else:
        bits_per_token = 0
    
    return bits_per_token

def calculate_semantic_density(text, token_count):
    """
    Estimate semantic density based on content words ratio.
    
    Args:
        text: The text to analyze
        token_count: Number of tokens in the text
        
    Returns:
        Semantic density metrics
    """
    # Simple heuristic: count content words vs. function words
    # This is a simplified approach; a more sophisticated NLP approach would be better
    
    # Split text into words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Common function words in English
    function_words = set([
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'while', 'although',
        'as', 'when', 'because', 'since', 'for', 'to', 'in', 'on', 'at',
        'by', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down',
        'of', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should',
        'now', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
        'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
        'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
        'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
        'do', 'does', 'did', 'doing', 'would', 'could', 'should', 'ought',
        'i\'m', 'you\'re', 'he\'s', 'she\'s', 'it\'s', 'we\'re', 'they\'re',
        'i\'ve', 'you\'ve', 'we\'ve', 'they\'ve', 'i\'d', 'you\'d', 'he\'d',
        'she\'d', 'we\'d', 'they\'d', 'i\'ll', 'you\'ll', 'he\'ll', 'she\'ll',
        'we\'ll', 'they\'ll', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t',
        'hasn\'t', 'haven\'t', 'hadn\'t', 'doesn\'t', 'don\'t', 'didn\'t',
        'won\'t', 'wouldn\'t', 'shan\'t', 'shouldn\'t', 'can\'t', 'cannot',
        'couldn\'t', 'mustn\'t', 'let\'s', 'that\'s', 'who\'s', 'what\'s',
        'here\'s', 'there\'s', 'when\'s', 'where\'s', 'why\'s', 'how\'s'
    ])
    
    # Count content words and function words
    content_word_count = sum(1 for word in words if word not in function_words)
    function_word_count = sum(1 for word in words if word in function_words)
    
    # Calculate ratios
    total_word_count = len(words)
    content_ratio = content_word_count / total_word_count if total_word_count > 0 else 0
    
    # Calculate semantic density (content words per token)
    semantic_density = content_word_count / token_count if token_count > 0 else 0
    
    return {
        'content_word_count': content_word_count,
        'function_word_count': function_word_count,
        'total_word_count': total_word_count,
        'content_ratio': content_ratio,
        'semantic_density': semantic_density
    }

def analyze_chinese_information_density(text, token_count):
    """
    Analyze information density specifically for Chinese text.
    
    Args:
        text: The text to analyze
        token_count: Number of tokens in the text
        
    Returns:
        Chinese information density metrics
    """
    # Count Chinese characters
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    
    # Calculate Chinese character ratio
    chinese_ratio = chinese_chars / len(text) if len(text) > 0 else 0
    
    # Estimate information content of Chinese characters
    # Each Chinese character typically carries more information than Latin characters
    # A rough estimate is that each Chinese character carries about 2.5x the information
    estimated_info_content = chinese_chars * 2.5 + (len(text) - chinese_chars)
    
    # Calculate Chinese information density
    chinese_info_density = estimated_info_content / token_count if token_count > 0 else 0
    
    # Calculate Chinese character per token ratio
    chars_per_token = chinese_chars / token_count if token_count > 0 else 0
    
    return {
        'chinese_char_count': chinese_chars,
        'chinese_ratio': chinese_ratio,
        'estimated_info_content': estimated_info_content,
        'chinese_info_density': chinese_info_density,
        'chinese_chars_per_token': chars_per_token
    }
