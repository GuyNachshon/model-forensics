import torch
import torch.nn as nn
from pathlib import Path
import time

from recorder import RecorderHooks, BundleExporter, ContextRecorder
from core.config import RCAConfig, RecorderConfig
from core.utils import setup_logging


class GPT2Like(nn.Module):
    """Simplified GPT-2 like model for testing."""
    
    def __init__(self, vocab_size=1000, d_model=64, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits


def create_advanced_config() -> RCAConfig:
    """Create configuration with advanced features enabled."""
    config = RCAConfig()
    
    # Enable all advanced recorder features
    config.recorder.compression_method = "random_projection"  # Test random projections
    config.recorder.random_projection_dim = 256
    config.recorder.async_export = True
    config.recorder.export_queue_size = 50
    config.recorder.export_batch_size = 5
    
    # Enable benign donor system
    config.recorder.enable_benign_donors = True
    config.recorder.benign_donor_pool_size = 20
    config.recorder.benign_donor_similarity_threshold = 0.7
    
    # Enable differential privacy (optional)
    config.recorder.enable_dp_noise = False  # Disabled for clearer demo
    config.recorder.dp_epsilon = 2.0
    
    config.bundle_dir = Path("./bundles_advanced")
    config.bundle_dir.mkdir(exist_ok=True)
    
    return config


def create_test_scenarios():
    """Create various test scenarios."""
    return {
        "benign_1": torch.tensor([[1, 15, 23, 42, 89, 156, 234, 456]]),
        "benign_2": torch.tensor([[2, 16, 24, 43, 90, 157, 235, 457]]),
        "benign_3": torch.tensor([[3, 17, 25, 44, 91, 158, 236, 458]]),
        "injection": torch.tensor([[999, 888, 777, 666, 555, 444, 333, 222]]),
        "jailbreak": torch.tensor([[111, 222, 333, 444, 555, 666, 777, 888]])
    }


def demonstrate_benign_donor_system(recorder: RecorderHooks, model: nn.Module, scenarios: dict):
    """Demonstrate the benign donor pool functionality."""
    print("=== Benign Donor System Demo ===")
    
    # Step 1: Populate benign donor pool
    print("1. Populating benign donor pool...")
    
    benign_scenarios = ["benign_1", "benign_2", "benign_3"]
    
    for i, scenario_name in enumerate(benign_scenarios):
        inputs = {"input_ids": scenarios[scenario_name]}
        
        # Use context manager for cleaner recording
        with ContextRecorder(recorder, inputs, {"scenario": scenario_name}) as ctx:
            with torch.no_grad():
                output = model(inputs["input_ids"])
            recorder.set_outputs({"logits": output})
        
        # Add these activations to benign donor pool
        if ctx.get_trace():
            # Convert compressed sketches back to activations for donor pool
            for layer_name, activation in recorder.activations.items():
                recorder.benign_donor_pool.add_benign_activation(layer_name, activation.numpy())
        
        print(f"   Added benign scenario {i+1}: {scenario_name}")
    
    # Show donor pool stats
    donor_stats = recorder.get_benign_donor_stats()
    print(f"   Donor pool stats: {donor_stats}")


def demonstrate_compression_methods(recorder: RecorderHooks, model: nn.Module, scenarios: dict):
    """Demonstrate different compression methods."""
    print("\n=== Compression Methods Demo ===")
    
    compression_methods = ["topk", "random_projection", "hybrid"]
    results = {}
    
    for method in compression_methods:
        print(f"1. Testing {method} compression...")
        
        # Update compression method
        recorder.config.compression_method = method
        
        # Record with this compression method
        inputs = {"input_ids": scenarios["benign_1"]}
        
        with ContextRecorder(recorder, inputs, {"compression_method": method}) as ctx:
            with torch.no_grad():
                output = model(inputs["input_ids"])
            recorder.set_outputs({"logits": output})
        
        # Get compression stats
        if ctx.get_trace():
            stats = recorder.sketcher.get_compression_stats()
            results[method] = stats
    
    # Compare compression methods
    print("   Compression comparison:")
    for method, stats in results.items():
        avg_ratio = sum(s["compression_ratio"] for s in stats.values()) / len(stats)
        print(f"   {method}: average ratio {avg_ratio:.3f}")


def demonstrate_async_export(recorder: RecorderHooks, model: nn.Module, scenarios: dict, exporter: BundleExporter):
    """Demonstrate async bundle export."""
    print("\n=== Async Export Demo ===")
    
    # Record multiple scenarios rapidly
    traces = []
    print("1. Recording multiple scenarios...")
    
    for i, (scenario_name, tokens) in enumerate(scenarios.items()):
        inputs = {"input_ids": tokens}
        
        with ContextRecorder(recorder, inputs, {"scenario": scenario_name, "batch_id": i}) as ctx:
            with torch.no_grad():
                output = model(inputs["input_ids"])
            recorder.set_outputs({"logits": output})
        
        if ctx.get_trace():
            traces.append((ctx.get_trace(), f"async_bundle_{scenario_name}_{i}"))
        
        print(f"   Recorded: {scenario_name}")
    
    # Queue all bundles for async export
    print("2. Queuing bundles for async export...")
    start_time = time.time()
    
    for trace_data, bundle_id in traces:
        recorder.export_bundle_async(trace_data, bundle_id, exporter)
    
    queue_time = time.time() - start_time
    print(f"   Queued {len(traces)} bundles in {queue_time:.3f} seconds")
    
    # Wait a moment for async processing
    print("3. Waiting for async processing...")
    time.sleep(2.0)
    
    # Show async writer stats
    async_stats = recorder.get_async_writer_stats()
    print(f"   Async writer stats: {async_stats}")


def demonstrate_differential_privacy(config: RCAConfig, model: nn.Module, scenarios: dict):
    """Demonstrate differential privacy noise addition."""
    print("\n=== Differential Privacy Demo ===")
    
    # Create recorder with DP enabled
    dp_config = RCAConfig()
    dp_config.recorder = RecorderConfig()
    dp_config.recorder.enable_dp_noise = True
    dp_config.recorder.dp_epsilon = 1.0
    dp_config.recorder.dp_delta = 1e-5
    dp_config.recorder.dp_sensitivity = 1.0
    
    dp_recorder = RecorderHooks(dp_config)
    dp_recorder.install_hooks(model)
    
    print("1. Recording with differential privacy noise...")
    
    inputs = {"input_ids": scenarios["benign_1"]}
    
    with ContextRecorder(dp_recorder, inputs, {"privacy": "dp_enabled"}) as ctx:
        with torch.no_grad():
            output = model(inputs["input_ids"])
        dp_recorder.set_outputs({"logits": output})
    
    if ctx.get_trace():
        dp_metadata = ctx.get_trace().activations
        dp_count = sum(1 for sketch in dp_metadata.values() 
                      if sketch.metadata.get("dp_noise_applied", False))
        print(f"   Applied DP noise to {dp_count} activations")
    
    dp_recorder.cleanup()


def main():
    """Demonstrate advanced recording features."""
    # Setup
    setup_logging("INFO")
    print("=== Advanced Model Forensics Demo ===\n")
    
    # Create advanced configuration
    config = create_advanced_config()
    
    # Create model
    model = GPT2Like()
    model.eval()
    
    # Setup recorder with advanced features
    recorder = RecorderHooks(config)
    recorder.install_hooks(model)
    
    # Create exporter
    exporter = BundleExporter(config)
    
    # Create test scenarios
    scenarios = create_test_scenarios()
    
    # Demonstrate each advanced feature
    demonstrate_benign_donor_system(recorder, model, scenarios)
    demonstrate_compression_methods(recorder, model, scenarios)
    demonstrate_async_export(recorder, model, scenarios, exporter)
    demonstrate_differential_privacy(config, model, scenarios)
    
    # Final statistics
    print("\n=== Final Statistics ===")
    print(f"Benign donor pool: {recorder.get_benign_donor_stats()}")
    print(f"Async writer: {recorder.get_async_writer_stats()}")
    
    # Show overall compression performance
    overall_stats = recorder.sketcher.get_compression_stats()
    if overall_stats:
        total_original = sum(s["original_size"] for s in overall_stats.values())
        total_compressed = sum(s["compressed_size"] for s in overall_stats.values())
        overall_ratio = total_compressed / total_original if total_original > 0 else 0
        print(f"Overall compression ratio: {overall_ratio:.3f}")
        print(f"Total space saved: {(total_original - total_compressed) / (1024*1024):.2f} MB")
    
    # Cleanup
    recorder.cleanup()
    
    print(f"\n=== Advanced Demo Complete ===")
    print(f"Enhanced bundles saved to: {config.bundle_dir}")
    print("\nAdvanced features demonstrated:")
    print("✓ Random projection compression")
    print("✓ Benign donor pool for activation patching")
    print("✓ Async bundle export with threading")
    print("✓ Differential privacy noise (optional)")
    print("✓ Context manager for clean recording")
    print("✓ Multiple compression method comparison")


if __name__ == "__main__":
    main()