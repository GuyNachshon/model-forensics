#!/usr/bin/env python3
"""Step-by-step debugging to isolate the SIGBUS issue."""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def test_step_1_basic_model():
    """Test 1: Basic model loading and simple forward pass."""
    print("=== TEST 1: Basic Model Loading ===", flush=True)
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    print(f"Input: {text}")
    print(f"Input shape: {inputs.input_ids.shape}")
    
    # Simple forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        print(f"✅ Basic forward pass works: {outputs.logits.shape}")
    
    return model, tokenizer

def test_step_2_basic_generate():
    """Test 2: Basic generation without any extra parameters."""
    print("\n=== TEST 2: Basic Generation ===")
    
    model, tokenizer = test_step_1_basic_model()
    
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    try:
        with torch.no_grad():
            generated = model.generate(
                inputs.input_ids,
                max_length=20,
                do_sample=False,  # Greedy for determinism
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"✅ Basic generation works: {generated_text}")
            
    except Exception as e:
        print(f"❌ Basic generation failed: {e}")
        return None, None
    
    return model, tokenizer

def test_step_3_generation_with_scores():
    """Test 3: Generation with output_scores=True."""
    print("\n=== TEST 3: Generation with Scores ===")
    
    model, tokenizer = test_step_2_basic_generate()
    if model is None:
        return None, None
    
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    try:
        with torch.no_grad():
            generation_output = model.generate(
                inputs.input_ids,
                max_length=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            print(f"✅ Generation with scores works")
            print(f"   Sequences: {generation_output.sequences.shape}")
            print(f"   Scores length: {len(generation_output.scores)}")
            
    except Exception as e:
        print(f"❌ Generation with scores failed: {e}")
        return None, None
    
    return model, tokenizer

def test_step_4_forward_with_hidden_states():
    """Test 4: Forward pass with output_hidden_states=True."""
    print("\n=== TEST 4: Forward Pass with Hidden States ===")
    
    model, tokenizer = test_step_3_generation_with_scores()
    if model is None:
        return None, None
    
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    try:
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            
            print(f"✅ Forward pass with hidden states works")
            print(f"   Logits: {outputs.logits.shape}")
            print(f"   Hidden states count: {len(outputs.hidden_states)}")
            print(f"   Hidden state shapes: {[h.shape for h in outputs.hidden_states[:3]]}...")
            
    except Exception as e:
        print(f"❌ Forward pass with hidden states failed: {e}")
        return None, None
    
    return model, tokenizer

def test_step_5_generation_with_hidden_states():
    """Test 5: Generation with output_hidden_states=True (THE PROBLEMATIC ONE)."""
    print("\n=== TEST 5: Generation with Hidden States (PROBLEMATIC) ===")
    
    model, tokenizer = test_step_4_forward_with_hidden_states()
    if model is None:
        return None, None
    
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
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
            print(f"   Scores length: {len(generation_output.scores)}")
            
    except Exception as e:
        print(f"❌ Generation with hidden states failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    return model, tokenizer

def main():
    """Run all tests step by step."""
    import sys
    print("Step-by-step SIGBUS debugging", flush=True)
    print("Each test builds on the previous one\n", flush=True)
    
    try:
        model, tokenizer = test_step_5_generation_with_hidden_states()
        
        if model is not None:
            print("\n🎉 ALL TESTS PASSED!", flush=True)
            print("The issue might be elsewhere in the code.", flush=True)
        else:
            print("\n💥 Found the breaking point!", flush=True)
            
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()