import sys
import argparse
from pathlib import Path
from typing import List

from abctokz.tokenizer import Tokenizer
from abctokz.config.defaults import bpe_multilingual

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["base", "modified"], required=True)
    args = parser.parse_args()

    print(f"=== Running Task 13 Script (Mode: {args.mode}) ===")
    
    # We will test on a string containing exotic spaces (e.g., U+3000 Ideographic Space, U+00A0 No-Break Space)
    test_strings = [
        "नमस्ते\u3000दुनिया",  # IDEOGRAPHIC SPACE
        "hello\u00A0world",   # NO-BREAK SPACE
        "one\u2002two\u2003three\u2004four", # EM quads
    ]

    config = bpe_multilingual(vocab_size=500)
    tokenizer = Tokenizer.from_config(config)

    for text in test_strings:
        print(f"\nRaw Input: {repr(text)}")
        norm_text = tokenizer._normalizer.normalize(text)
        print(f"Normalized: {repr(norm_text)}")
        
        pretokens = tokenizer._pretokenizer.pre_tokenize(norm_text)
        print(f"Pre-tokens: {pretokens}")

if __name__ == "__main__":
    main()
