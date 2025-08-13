#!/usr/bin/env python3
"""Minimal test to debug GPT-2 integration issues."""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path

from recorder import RecorderHooks, BundleExporter
from replayer import Replayer
from core.config import RCAConfig
from core.utils import setup_logging

def test_gpt2_without_hooks():
    """Test GPT-2 model without any hooks first."""
    print("=== Testing GPT-2 without hooks ===")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Simple forward pass
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt")
    
    print(f"Input: {text}")
    print(f"Input tensor shape: {inputs.input_ids.shape}")
    
    with torch.no_grad():
        outputs = model(**inputs)
        print(f"Output logits shape: {outputs.logits.shape}")
        
        # Generate text
        generated = model.generate(
            inputs.input_ids,
            max_length=20,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
    
    print("✓ GPT-2 works without hooks")
    return model, tokenizer

def test_minimal_recording():
    """Test minimal recording without layer hooks."""
    print("\n=== Testing minimal recording ===")
    
    model, tokenizer = test_gpt2_without_hooks()
    
    # Setup recorder with NO layer hooks
    config = RCAConfig()
    config.recorder.compression_method = "topk" 
    config.recorder.compression_ratio = 0.5
    config.recorder.enable_benign_donors = False
    config.recorder.async_export = False
    config.recorder.layers_to_record = []  # Record NO layers initially
    config.bundle_dir = Path("./minimal_bundles")
    config.bundle_dir.mkdir(exist_ok=True)
    
    recorder = RecorderHooks(config)
    exporter = BundleExporter(config)
    
    # Test recording WITHOUT installing hooks
    print("Testing recording without hooks...")
    text = "Test recording"
    inputs = tokenizer(text, return_tensors="pt")
    
    recorder_inputs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask
    }
    
    # Start recording but don't install hooks
    recorder.start_recording(recorder_inputs, metadata={"test": "minimal"})
    
    # Run model normally
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Set outputs and stop
    recorder.set_outputs({"logits": outputs.logits})
    trace_data = recorder.stop_recording()
    
    bundle = exporter.create_bundle(trace_data, "minimal_test")
    print(f"✓ Created bundle without hooks: {bundle.bundle_id}")
    
    return config, model, tokenizer

def test_single_hook_gradually():
    """Test adding hooks one at a time to find problematic layer."""
    print("\n=== Testing single hook installation ===")
    
    config, model, tokenizer = test_minimal_recording()
    
    # Test each layer individually to find the problem
    test_layers = [
        "lm_head",              # Output layer
        "transformer.ln_f",     # Final norm
        "transformer.wte",      # Word embeddings
        "transformer.h.11",     # Last transformer block
        "transformer.h.11.mlp", # Last MLP
    ]
    
    for layer_name in test_layers:
        print(f"\nTesting hook on: {layer_name}")
        
        try:
            # Create fresh recorder for this test
            config.recorder.layers_to_record = [layer_name]
            recorder = RecorderHooks(config)
            
            # Try to install hooks
            print(f"  Installing hooks...")
            recorder.install_hooks(model)
            
            print(f"  Running model with {layer_name} hooked...")
            text = "Test hook"
            inputs = tokenizer(text, return_tensors="pt")
            
            recorder_inputs = {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask
            }
            
            recorder.start_recording(recorder_inputs, metadata={"layer": layer_name})
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            recorder.set_outputs({"logits": outputs.logits})
            trace_data = recorder.stop_recording()
            
            print(f"  ✅ SUCCESS: {layer_name} works fine")
            recorder.cleanup()
            
        except Exception as e:
            print(f"  ❌ FAILED: {layer_name} caused error: {e}")
            try:
                recorder.cleanup()
            except:
                pass

def main():
    """Test GPT-2 integration step by step."""
    setup_logging("INFO")
    print("=== GPT-2 Integration Debug ===")
    
    try:
        test_gpt2_without_hooks()
        test_minimal_recording()  
        test_single_hook_gradually()
        print("\n=== Debug Complete ===")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()