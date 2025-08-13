#!/usr/bin/env python3
"""Debug script to check GPT-2 layer names."""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main():
    print("Loading GPT-2 model to check layer names...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    print(f"\nModel config:")
    print(f"  Layers: {model.config.n_layer}")
    print(f"  Embedding dim: {model.config.n_embd}")
    print(f"  Vocab size: {model.config.vocab_size}")
    
    print(f"\nLayer names in GPT-2:")
    for name, module in model.named_modules():
        if len(name) > 0:  # Skip root module
            print(f"  {name}: {type(module).__name__}")
        if "transformer.h" in name and ("attn" in name or "mlp" in name):
            print(f"    --> {name} (shape info follows)")

if __name__ == "__main__":
    main()