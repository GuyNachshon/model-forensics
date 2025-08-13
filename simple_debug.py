#!/usr/bin/env python3
"""Simple debug script to test GPT-2 step by step."""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main():
    print("=== Simple GPT-2 Debug ===")
    
    # Step 1: Load model
    print("Step 1: Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model loaded successfully")
    
    # Step 2: Basic forward pass
    print("\nStep 2: Testing basic forward pass...")
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        print(f"✅ Forward pass works: {outputs.logits.shape}")
    
    # Step 3: Basic generation
    print("\nStep 3: Testing basic generation...")
    with torch.no_grad():
        generated = model.generate(
            inputs.input_ids,
            max_length=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"✅ Basic generation works: {generated_text}")
    
    # Step 4: Generation with scores
    print("\nStep 4: Testing generation with scores...")
    with torch.no_grad():
        generation_output = model.generate(
            inputs.input_ids,
            max_length=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        print(f"✅ Generation with scores works: sequences={generation_output.sequences.shape}")
    
    # Step 5: Forward pass with hidden states
    print("\nStep 5: Testing forward pass with hidden states...")
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        print(f"✅ Forward with hidden states works: {len(outputs.hidden_states)} layers")
    
    # Step 6: The problematic one - generation with hidden states
    print("\nStep 6: Testing generation with hidden states (THE PROBLEMATIC ONE)...")
    try:
        with torch.no_grad():
            print("About to call model.generate with output_hidden_states=True...")
            generation_output = model.generate(
                inputs.input_ids,
                max_length=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True  # THIS IS THE SUSPECTED CULPRIT
            )
            print(f"✅ Generation with hidden states works!")
            print(f"   Sequences: {generation_output.sequences.shape}")
            
    except Exception as e:
        print(f"❌ Generation with hidden states failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🎉 ALL TESTS PASSED! The issue is elsewhere.")

if __name__ == "__main__":
    main()