import re
import string
from collections import Counter, defaultdict
import numpy as np
import os

def create_ngrams(tokens, n):
    """
    Create n-grams from a list of tokens.
    
    Args:
        tokens (list): List of tokens from the text
        n (int): Size of the n-gram (e.g., 2 for bigrams)
    
    Returns:
        list: List of (n+1)-tuples where the last element is the next token
    """
    # Use zip with multiple iterators, each offset by one position
    # This creates tuples of n+1 consecutive tokens
    return list(zip(*[tokens[i:] for i in range(n+1)]))

def sample_next_token(current_ngram, ngram_to_next_token_probs):
    """
    Sample the next token given an n-gram based on probability distribution.
    
    Args:
        current_ngram (tuple): Current sequence of n tokens
        ngram_to_next_token_probs (dict): Mapping of n-grams to next token probabilities
    
    Returns:
        str: Sampled next token, or None if n-gram not found
    """
    if current_ngram not in ngram_to_next_token_probs:
        return None
    
    # Get probability distribution for next tokens
    next_token_probs = ngram_to_next_token_probs[current_ngram]
    tokens = list(next_token_probs.keys())
    probs = list(next_token_probs.values())
    
    # Sample next token based on probabilities
    return np.random.choice(tokens, p=probs)

def generate_text_from_ngram(initial_ngram, num_words, ngram_to_next_token_probs):
    """
    Generate text starting from an initial n-gram.
    
    Args:
        initial_ngram (tuple): Starting sequence of tokens
        num_words (int): Total number of words to generate
        ngram_to_next_token_probs (dict): Mapping of n-grams to next token probabilities
    
    Returns:
        str: Generated text
    """
    n = len(initial_ngram)
    if num_words < n:
        return " ".join(initial_ngram[:num_words])
    
    # Start with the initial n-gram
    generated_tokens = list(initial_ngram)
    
    # Generate remaining words one at a time
    for _ in range(num_words - n):
        current_ngram = tuple(generated_tokens[-(n):])  # Take last n tokens
        next_token = sample_next_token(current_ngram, ngram_to_next_token_probs)
        
        if next_token is None:  # Stop if we hit a dead end
            break
            
        generated_tokens.append(next_token)
    
    return " ".join(generated_tokens)

def build_ngram_model(tokens, n):
    """
    Build n-gram model from tokens, creating both count and probability distributions.
    
    Args:
        tokens (list): List of tokens from the text
        n (int): Size of the n-gram
    
    Returns:
        tuple: (counts_dict, probs_dict) containing the n-gram statistics
    """
    # Create n-grams from tokens
    ngrams = create_ngrams(tokens, n)
    
    # Count occurrences of next tokens for each n-gram
    ngram_to_next_token_counts = defaultdict(Counter)
    for *context, next_token in ngrams:
        ngram_to_next_token_counts[tuple(context)][next_token] += 1
    
    # Convert counts to probabilities
    ngram_to_next_token_probs = {}
    for ngram, next_token_counts in ngram_to_next_token_counts.items():
        total_count = sum(next_token_counts.values())
        ngram_to_next_token_probs[ngram] = {
            token: count/total_count 
            for token, count in next_token_counts.items()
        }
    
    return ngram_to_next_token_counts, ngram_to_next_token_probs

def preprocess_text(input_file, output_file):
    """
    Preprocess text and build n-gram models.
    
    Args:
        input_file (str): Path to input text file
        output_file (str): Path to save preprocessed tokens
    
    Returns:
        dict: Dictionary containing models for different n-gram sizes
    """
    # Debug: Check file contents
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Read {len(text)} characters from {input_file}")
    if len(text) < 100:  # arbitrary small number
        print("Warning: Input file seems very small")
    
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    
    # Save preprocessed tokens
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tokens))
    
    # Build models for different n-gram sizes
    models = {}
    for n in [2, 3, 4]:  # bigrams, trigrams, quadgrams
        counts, probs = build_ngram_model(tokens, n)
        models[n] = {
            'counts': counts,
            'probs': probs
        }
        
        # Save n-gram counts
        with open(f'{n}grams.txt', 'w', encoding='utf-8') as f:
            for ngram, next_tokens in counts.items():
                f.write(f"{' '.join(ngram)}:\n")
                for token, count in next_tokens.most_common():
                    f.write(f"  {token}: {count}\n")
                f.write("\n")
        
        # Save n-gram probabilities
        with open(f'{n}gram_probs.txt', 'w', encoding='utf-8') as f:
            for ngram, next_token_probs in probs.items():
                f.write(f"{' '.join(ngram)}:\n")
                sorted_probs = sorted(next_token_probs.items(), key=lambda x: x[1], reverse=True)
                for token, prob in sorted_probs:
                    f.write(f"  {token}: {prob:.4f}\n")
                f.write("\n")
    
    return models

def compare_models(models, num_words=50):
    """
    Compare text generation using different n-gram sizes.
    
    Args:
        models (dict): Dictionary containing models for different n-gram sizes
        num_words (int): Number of words to generate for comparison
    """
    print("\nComparing text generation with different n-gram sizes:")
    for n, model in models.items():
        print(f"\n{n}-gram model:")
        # Generate sample text
        initial_ngram = list(model['probs'].keys())[0]
        generated_text = generate_text_from_ngram(initial_ngram, num_words, model['probs'])
        print(f"Starting with: {' '.join(initial_ngram)}")
        print(f"Generated text: {generated_text}")
        
        # Calculate and display statistics
        unique_next_tokens = sum(len(next_tokens) for next_tokens in model['counts'].values())
        avg_next_tokens = unique_next_tokens / len(model['counts'])
        print(f"Average number of possible next tokens: {avg_next_tokens:.2f}")

if __name__ == "__main__":
    input_file = "shakespear.txt"
    output_file = "preprocessed_tokens.txt"
    
    # Check if file exists and has content
    try:
        # Print current working directory
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for file: {os.path.abspath(input_file)}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
            print(f"File exists and contains {len(text)} characters")
            # Print first 100 characters to verify content
            print("First 100 characters of file:")
            print(text[:100])
            
        if len(text) == 0:
            print(f"Error: {input_file} is empty!")
            exit(1)
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}!")
        print("Please make sure the file exists in the current directory.")
        print("Expected path:", os.path.abspath(input_file))
        exit(1)
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        exit(1)
    
    # Process text and build models
    models = preprocess_text(input_file, output_file)
    print(f"Text has been preprocessed and saved to {output_file}")
    print("N-gram models have been saved to respective files")
    
    # Compare different n-gram models
    compare_models(models)
    
    # Interactive text generation
    print("\nEnter n-gram size (2, 3, or 4):")
    n = int(input().strip())
    if n in models:
        print(f"Enter {n} words to start generation:")
        user_input = input().strip()
        initial_ngram = tuple(user_input.lower().split()[:n])
        generated_text = generate_text_from_ngram(initial_ngram, 50, models[n]['probs'])
        print("\nGenerated text:")
        print(generated_text) 