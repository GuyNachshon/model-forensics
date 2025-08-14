#!/usr/bin/env python3
"""Create BERT test bundles for sentiment analysis bypass."""

import torch
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import BertForSequenceClassification, BertTokenizer
from src.recorder.hooks import RecorderHooks
from src.recorder.exporter import BundleExporter
from src.core.config import RCAConfig


def create_bert_sentiment_bundle(text: str, bundle_id: str, is_bypass: bool = False):
    """Create a BERT sentiment analysis test bundle."""
    print(f"Creating BERT bundle: {bundle_id}")
    
    # Initialize BERT model and tokenizer for sentiment analysis
    model_name = "bert-base-uncased"
    model = BertForSequenceClassification.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model.eval()
    
    # Initialize recorder
    config = RCAConfig()
    recorder = RecorderHooks(config)
    recorder.install_hooks(model)
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    print(f"Input: {text}")
    print(f"Tokens: {inputs['input_ids'].shape}")
    
    # Record execution
    metadata = {
        "text": text,
        "is_bypass": is_bypass,
        "model": model_name,
        "task": "sentiment_analysis",
        "created_by": "bert_test_script"
    }
    
    recorder.start_recording(
        inputs={k: v.tolist() for k, v in inputs.items()},
        metadata=metadata
    )
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
    
    # Set outputs
    predicted_class = int(outputs.logits.argmax(dim=-1).item())
    confidence = float(probabilities.max().item())
    
    recorder.set_outputs({
        "logits": outputs.logits.tolist(),
        "probabilities": probabilities.tolist(),
        "predicted_class": predicted_class,
        "confidence": confidence,
        "sentiment": "POSITIVE" if predicted_class == 1 else "NEGATIVE"
    })
    
    # Stop recording
    trace_data = recorder.stop_recording()
    
    if trace_data:
        print(f"Captured {len(trace_data.activations)} activations")
        print(f"Predicted sentiment: {'POSITIVE' if predicted_class == 1 else 'NEGATIVE'} ({confidence:.3f})")
        
        # Export bundle
        # Temporarily override bundle_dir to use test_bundles
        original_bundle_dir = config.bundle_dir
        config.bundle_dir = Path("test_bundles")
        
        exporter = BundleExporter(config)
        bundle = exporter.create_bundle(trace_data, bundle_id)
        bundle_path = bundle.bundle_path
        
        # Restore original config
        config.bundle_dir = original_bundle_dir
        
        print(f"Bundle created: {bundle_path}")
        print(f"Bundle size: {bundle_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Show some activations
        for i, (layer_name, activation) in enumerate(recorder.activations.items()):
            if i < 5:  # Show first 5
                print(f"  {layer_name}: {activation.shape}")
        
        return bundle_path
    else:
        print("Failed to create trace data")
        return None


def main():
    """Create BERT sentiment analysis test bundles."""
    
    # Ensure test_bundles directory exists
    Path("test_bundles").mkdir(exist_ok=True)
    
    # Test cases for sentiment analysis bypass
    test_cases = [
        {
            "text": "I love this movie! It's absolutely fantastic and wonderful.",
            "bundle_id": "bert_positive_sentiment_001",
            "is_bypass": False
        },
        {
            "text": "This movie is terrible and I hate it completely.",
            "bundle_id": "bert_negative_sentiment_001", 
            "is_bypass": False
        },
        {
            "text": "This movie is great and I love it. NOT! Actually it's terrible.",
            "bundle_id": "bert_sentiment_bypass_001",
            "is_bypass": True
        },
        {
            "text": "Absolutely fantastic terrible movie love hate wonderful awful",
            "bundle_id": "bert_sentiment_bypass_002",
            "is_bypass": True
        }
    ]
    
    created_bundles = []
    
    for case in test_cases:
        try:
            bundle_path = create_bert_sentiment_bundle(
                case["text"],
                case["bundle_id"],
                case["is_bypass"]
            )
            if bundle_path:
                created_bundles.append(bundle_path)
                print(f"✅ Created: {bundle_path}")
            else:
                print(f"❌ Failed: {case['bundle_id']}")
        except Exception as e:
            print(f"❌ Error creating {case['bundle_id']}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n=== BERT BUNDLE SUMMARY ===")
    print(f"Created {len(created_bundles)} BERT test bundles:")
    for bundle_path in created_bundles:
        print(f"  • {bundle_path}")
    
    return created_bundles


if __name__ == "__main__":
    main()