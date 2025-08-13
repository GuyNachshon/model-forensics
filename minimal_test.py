#!/usr/bin/env python3
"""Minimal test to isolate exact crash point."""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def test_basic():
    print("Loading GPT-2 model...", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    
    # Fix attention mask issue
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt")
    # Create explicit attention mask
    inputs['attention_mask'] = torch.ones_like(inputs.input_ids)
    
    print(f"Input shape: {inputs.input_ids.shape}", flush=True)
    print("Running basic forward pass...", flush=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        print(f"✅ Forward pass: {outputs.logits.shape}", flush=True)
    
    print("Running basic generation...", flush=True)
    with torch.no_grad():
        generated = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        print(f"✅ Basic generation: {generated.shape}", flush=True)
    
    print("Testing THE PROBLEMATIC CASE: generation with hidden states...", flush=True)
    try:
        with torch.no_grad():
            generation_output = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True  # THE SUSPECTED PROBLEM
            )
            print(f"✅ SUCCESS! Generation with hidden states works!", flush=True)
            print(f"   Sequences: {generation_output.sequences.shape}", flush=True)
            
    except Exception as e:
        print(f"❌ CRASH: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_basic()
    print(f"Test {'PASSED' if success else 'FAILED'}", flush=True)