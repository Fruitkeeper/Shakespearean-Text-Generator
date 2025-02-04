# N-gram Text Generation from Shakespeare

This project implements n-gram language models for text generation using Shakespeare's text as training data. It supports bigrams (2-grams), trigrams (3-grams), and quadgrams (4-grams).

## Overview

The project consists of the following components:

1. Text preprocessing
   - Converting text to lowercase
   - Removing punctuation
   - Tokenization

2. N-gram model building
   - Creating n-grams from tokens
   - Calculating next token probabilities
   - Supporting multiple n-gram sizes (2, 3, and 4)

3. Text generation
   - Sampling next tokens based on probabilities
   - Generating text from initial n-grams
   - Interactive text generation

## Dataset

The project uses Shakespeare's text (`shakespear.txt`) as training data. The text is preprocessed to create a clean dataset for n-gram model training.

## Files Generated

The program generates several output files:
- `preprocessed_tokens.txt`: Clean, tokenized text
- `{n}grams.txt`: N-gram to next token counts (n = 2,3,4)
- `{n}gram_probs.txt`: N-gram to next token probabilities (n = 2,3,4)

## Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   ```

2. Activate the virtual environment:
   
   On macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```
   
   On Windows:
   ```bash
   .venv\Scripts\activate
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements

- Python 3.6+
- Virtual environment with:
  - NumPy==1.24.3
- No other external dependencies

## Usage

1. Ensure `shakespear.txt` is in the same directory as the script
2. Run the script:
   ```
   python preprocess.py
   ```
3. The program will:
   - Preprocess the text
   - Build n-gram models
   - Compare text generation between different n-gram sizes
   - Allow interactive text generation

## Interactive Text Generation

After running the script, you can:
1. Choose n-gram size (2, 3, or 4)
2. Enter starting words (matching the chosen n-gram size)
3. Generate text based on your input

## Model Comparison

The different n-gram models offer different trade-offs:

- Bigrams (n=2):
  - More variety in generated text
  - Less coherent
  - Good for exploring possible word combinations

- Trigrams (n=3):
  - Better local coherence
  - More natural-sounding phrases
  - Balance between novelty and coherence

- Quadgrams (n=4):
  - Highest local coherence
  - More likely to reproduce exact phrases
  - Limited by data sparsity

## Output Files Format

### N-gram Counts File (`{n}grams.txt`): 