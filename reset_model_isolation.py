#!/usr/bin/env python3
"""Reset model isolation status for continuing experiments."""

import sys
import os
sys.path.append('/home/tzachi/model-forensics')

from scenarios.run_scenarios import ScenarioRunner, ScenarioConfig

def reset_isolation():
    """Reset model isolation for Qwen/Qwen3-4B-Thinking-2507."""
    
    # Create a minimal config just to initialize the runner
    config = ScenarioConfig(
        model="Qwen/Qwen3-4B-Thinking-2507",
        samples_per_condition=1,
        output_dir="/tmp/reset",
        classification_enabled=False,
        force_rerun=False,
        rca_enabled=False
    )
    
    runner = ScenarioRunner(config)
    
    # Reset the specific model that was isolated
    model_name = "Qwen/Qwen3-4B-Thinking-2507"
    runner.reset_model_isolation(model_name)
    
    print(f"✅ Reset isolation status for {model_name}")
    print("You can now run experiments again without the model being skipped.")

if __name__ == "__main__":
    reset_isolation()