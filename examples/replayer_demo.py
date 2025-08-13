"""Demonstrate replayer functionality for model forensics."""

import torch
import torch.nn as nn
from pathlib import Path

from recorder import RecorderHooks, BundleExporter, BenignDonorPool
from replayer import Replayer, InterventionEngine, ActivationReconstructor
from core.config import RCAConfig, RecorderConfig
from core.types import Intervention, InterventionType
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


def setup_test_bundles():
    """Create test bundles for replay demonstration."""
    print("=== Setting Up Test Bundles ===")
    
    # Setup configuration
    config = RCAConfig()
    config.recorder.compression_method = "topk"
    config.recorder.compression_ratio = 0.2
    config.recorder.enable_benign_donors = True
    config.bundle_dir = Path("./bundles")
    config.bundle_dir.mkdir(exist_ok=True)
    
    # Create model
    model = GPT2Like()
    model.eval()
    
    # Setup recorder
    recorder = RecorderHooks(config)
    recorder.install_hooks(model)
    exporter = BundleExporter(config)
    
    # Create test scenarios
    benign_tokens = torch.tensor([[1, 15, 23, 42, 89, 156, 234, 456]])
    injection_tokens = torch.tensor([[999, 888, 777, 666, 555, 444, 333, 222]])
    
    bundles = {}
    
    # Record benign case
    print("1. Recording benign case...")
    inputs = {"input_ids": benign_tokens}
    
    recorder.start_recording(inputs, metadata={"scenario": "benign"})
    
    with torch.no_grad():
        benign_output = model(inputs["input_ids"])
    
    recorder.set_outputs({"logits": benign_output})
    benign_trace = recorder.stop_recording()
    
    benign_bundle = exporter.create_bundle(benign_trace, "benign_demo")
    bundles["benign"] = benign_bundle
    print(f"   Created benign bundle: {benign_bundle.bundle_id}")
    
    # Record injection case
    print("2. Recording injection case...")
    inputs = {"input_ids": injection_tokens}
    
    recorder.start_recording(inputs, metadata={"scenario": "injection"})
    
    with torch.no_grad():
        injection_output = model(inputs["input_ids"])
    
    recorder.set_outputs({"logits": injection_output})
    injection_trace = recorder.stop_recording()
    
    injection_bundle = exporter.create_bundle(injection_trace, "injection_demo")
    bundles["injection"] = injection_bundle
    print(f"   Created injection bundle: {injection_bundle.bundle_id}")
    
    # Setup benign donor pool
    benign_pool = BenignDonorPool(config.recorder)
    for layer_name, activation in recorder.activations.items():
        if activation.requires_grad:
            activation = activation.detach()
        if activation.is_cuda:
            activation = activation.cpu()
        benign_pool.add_benign_activation(layer_name, activation.numpy())
    
    recorder.cleanup()
    
    return bundles, benign_pool, model, config


def demonstrate_basic_replay(replayer: Replayer, bundle, model):
    """Demonstrate basic bundle loading and replay."""
    print("\n=== Basic Replay Demo ===")
    
    # Create replay session
    session = replayer.create_session(bundle, model)
    print(f"1. Created session: {session.session_id}")
    print(f"   Reconstructed {len(session.reconstructed_activations)} activations")
    
    # Show reconstruction quality
    reconstructor = ActivationReconstructor()
    print("2. Reconstruction quality estimates:")
    for layer_name, sketch in bundle.trace_data.activations.items():
        quality = reconstructor.estimate_reconstruction_quality(sketch)
        print(f"   {layer_name}: fidelity={quality['fidelity_score']:.3f}, "
              f"confidence={quality['confidence']:.3f}")
    
    # Replay original execution
    print("3. Replaying original execution...")
    replay_result = replayer.replay_original(session, validate=True)
    print(f"   Replay validation: {replay_result['validation']['status']}")
    
    return session


def demonstrate_interventions(replayer: Replayer, session, benign_pool):
    """Demonstrate various intervention types."""
    print("\n=== Intervention Demo ===")
    
    # Get some layer names to work with
    available_layers = list(session.reconstructed_activations.keys())
    target_layer = available_layers[0] if available_layers else None
    
    if not target_layer:
        print("   No layers available for intervention")
        return []
    
    print(f"   Target layer: {target_layer}")
    
    intervention_results = []
    
    # 1. Zero intervention
    print("1. Applying zero intervention...")
    zero_intervention = Intervention(
        type=InterventionType.ZERO,
        layer_name=target_layer,
        metadata={"description": "Zero out all activations"}
    )
    
    result = replayer.apply_intervention(session, zero_intervention)
    intervention_results.append(result)
    print(f"   Result: flip_success={result.flip_success}, confidence={result.confidence:.3f}")
    
    # 2. Mean intervention
    print("2. Applying mean intervention...")
    mean_intervention = Intervention(
        type=InterventionType.MEAN,
        layer_name=target_layer,
        metadata={"description": "Replace with mean activation"}
    )
    
    result = replayer.apply_intervention(session, mean_intervention)
    intervention_results.append(result)
    print(f"   Result: flip_success={result.flip_success}, confidence={result.confidence:.3f}")
    
    # 3. Benign donor patch (if available)
    if benign_pool and target_layer in session.reconstructed_activations:
        print("3. Applying benign donor patch...")
        try:
            result = replayer.patch_with_benign_donor(session, target_layer, benign_pool)
            intervention_results.append(result)
            print(f"   Result: flip_success={result.flip_success}, confidence={result.confidence:.3f}")
        except ValueError as e:
            print(f"   Patch failed: {e}")
    
    # 4. Scaling intervention
    print("4. Applying scaling intervention...")
    scale_intervention = Intervention(
        type=InterventionType.SCALE,
        layer_name=target_layer,
        scale_factor=0.1,
        metadata={"description": "Scale activations by 0.1"}
    )
    
    result = replayer.apply_intervention(session, scale_intervention)
    intervention_results.append(result)
    print(f"   Result: flip_success={result.flip_success}, confidence={result.confidence:.3f}")
    
    return intervention_results


def demonstrate_analysis(session, intervention_results):
    """Demonstrate analysis of intervention results."""
    print("\n=== Analysis Demo ===")
    
    if not intervention_results:
        print("   No intervention results to analyze")
        return
    
    # Summarize intervention effectiveness
    print("1. Intervention effectiveness summary:")
    flip_successes = sum(1 for r in intervention_results if r.flip_success)
    avg_confidence = sum(r.confidence for r in intervention_results) / len(intervention_results)
    
    print(f"   Successful flips: {flip_successes}/{len(intervention_results)} "
          f"({flip_successes/len(intervention_results)*100:.1f}%)")
    print(f"   Average confidence: {avg_confidence:.3f}")
    
    # Show best intervention
    best_result = max(intervention_results, key=lambda r: r.confidence)
    print(f"   Best intervention: {best_result.intervention.type.value} "
          f"(confidence: {best_result.confidence:.3f})")
    
    # Show side effects
    print("2. Side effects analysis:")
    for i, result in enumerate(intervention_results):
        intervention_type = result.intervention.type.value
        side_effects = result.side_effects
        print(f"   {intervention_type}: {side_effects}")
    
    # Activation magnitude changes
    print("3. Activation magnitude changes:")
    for result in intervention_results:
        orig_norm = torch.norm(result.original_activation).item()
        mod_norm = torch.norm(result.modified_activation).item()
        change_ratio = (mod_norm - orig_norm) / orig_norm if orig_norm > 0 else 0
        print(f"   {result.intervention.type.value}: {change_ratio:.3f} relative change")


def demonstrate_session_export(replayer: Replayer, session):
    """Demonstrate exporting session results."""
    print("\n=== Export Demo ===")
    
    output_file = replayer.export_session_results(session)
    print(f"1. Exported session results to: {output_file}")
    
    # Show file contents
    import json
    with open(output_file, 'r') as f:
        results = json.load(f)
    
    print("2. Session summary:")
    print(f"   Session ID: {results['session_id']}")
    print(f"   Bundle ID: {results['bundle_id']}")
    print(f"   Number of interventions: {len(results['interventions'])}")
    print(f"   Successful interventions: {sum(1 for i in results['interventions'] if i['flip_success'])}")


def main():
    """Main demonstration of replayer functionality."""
    setup_logging("INFO")
    print("=== Model Forensics Replayer Demo ===\n")
    
    # Setup test data
    bundles, benign_pool, model, config = setup_test_bundles()
    
    # Initialize replayer
    replayer = Replayer(config)
    
    # Use injection bundle for demonstration
    injection_bundle = bundles["injection"]
    
    # Demonstrate basic replay
    session = demonstrate_basic_replay(replayer, injection_bundle, model)
    
    # Demonstrate interventions
    intervention_results = demonstrate_interventions(replayer, session, benign_pool)
    
    # Analyze results
    demonstrate_analysis(session, intervention_results)
    
    # Export session
    demonstrate_session_export(replayer, session)
    
    # Cleanup
    print("\n=== Cleanup ===")
    cleaned_sessions = replayer.cleanup_all_sessions()
    print(f"Cleaned up {cleaned_sessions} sessions")
    
    print(f"\n=== Replayer Demo Complete ===")
    print("Demonstrated features:")
    print("✓ Bundle loading and activation reconstruction")
    print("✓ Original execution replay and validation")
    print("✓ Multiple intervention types (zero, mean, patch, scale)")
    print("✓ Benign donor patching")
    print("✓ Intervention effectiveness analysis")
    print("✓ Session result export")
    print("✓ Reconstruction quality estimation")


if __name__ == "__main__":
    main()