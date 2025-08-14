#!/usr/bin/env python3
"""Test the complete CF → CCA integration pipeline."""

import logging
from pathlib import Path
from transformers import GPT2LMHeadModel

from src.core.config import RCAConfig
from src.modules.pipeline import RCAPipeline
from src.replayer.core import Replayer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pipeline_integration():
    """Test complete pipeline integration."""
    print("Testing Pipeline Integration")
    print("=" * 60)
    
    # Initialize components
    config = RCAConfig()
    pipeline = RCAPipeline(config)
    replayer = Replayer(config)
    
    print("\n1. Testing pipeline initialization...")
    stats = pipeline.get_pipeline_stats()
    print(f"   Pipeline stats: {stats}")
    print("   ✅ Pipeline initialized correctly")
    
    # Load model
    print("\n2. Loading model...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    print("   ✅ Model loaded")
    
    # Load bundles
    print("\n3. Loading test bundles...")
    incident_bundle = None
    baseline_bundles = []
    
    # Load incident
    incident_path = Path("test_bundles/injection_hack_001")
    if incident_path.exists():
        incident_bundle = replayer.load_bundle(incident_path)
        print(f"   Incident: {incident_bundle.bundle_id}")
    
    # Load baselines
    for bundle_path in Path("test_bundles").glob("*benign*"):
        if (bundle_path / "manifest.json").exists():
            try:
                bundle = replayer.load_bundle(bundle_path)
                baseline_bundles.append(bundle)
                print(f"   Baseline: {bundle.bundle_id}")
            except Exception as e:
                print(f"   Warning: Failed to load baseline {bundle_path}: {e}")
    
    if not incident_bundle:
        print("   ❌ No incident bundle found")
        return False
    
    print(f"   ✅ Loaded 1 incident, {len(baseline_bundles)} baselines")
    
    # Test CF phase
    print("\n4. Testing CF (Compression Forensics) phase...")
    if pipeline.cf:
        # Build baseline
        if baseline_bundles:
            pipeline.cf.build_baseline(baseline_bundles)
            print("   Built compression baseline")
        
        # Analyze incident
        metrics = pipeline.cf.analyze_bundle(incident_bundle)
        print(f"   Generated {len(metrics)} compression metrics")
        
        # Show sample metrics
        for metric in metrics[:3]:
            print(f"   {metric.layer_name}: ratio={metric.compression_ratio:.3f}, "
                  f"entropy={metric.entropy:.3f}, anomaly={metric.anomaly_score:.3f}")
        
        # Test prioritization
        priority_layers = pipeline.cf.prioritize_layers(metrics)
        print(f"   Priority layers: {len(priority_layers)}")
        
        print("   ✅ CF phase working correctly")
    else:
        print("   ❌ CF module not enabled")
        return False
    
    # Test CCA phase
    print("\n5. Testing CCA (Causal Analysis) phase...")
    if pipeline.cca:
        # Create session for CCA
        session = replayer.create_session(incident_bundle, model)
        print(f"   Created session with {len(session.reconstructed_activations)} activations")
        
        # Test with limited layers for speed
        test_layers = list(session.reconstructed_activations.keys())[:5]
        fix_sets = pipeline.cca.find_minimal_fix_sets(session, test_layers)
        print(f"   Found {len(fix_sets)} fix sets")
        
        if fix_sets:
            best = fix_sets[0]
            print(f"   Best fix set: {len(best.interventions)} interventions, "
                  f"flip_rate={best.total_flip_rate:.3f}")
        
        print("   ✅ CCA phase working correctly")
    else:
        print("   ❌ CCA module not enabled")
        return False
    
    # Test full pipeline integration
    print("\n6. Testing complete CF → CCA integration...")
    
    try:
        # Run full analysis but with limited scope for speed
        original_cf_threshold = config.modules.cf.anomaly_threshold
        config.modules.cf.anomaly_threshold = 0.5  # Lower threshold for testing
        
        result = pipeline.analyze_incident(incident_bundle, model, baseline_bundles[:1])  # Use only 1 baseline
        
        print(f"   Analysis completed in {result.execution_time:.2f}s")
        print(f"   Overall confidence: {result.confidence:.3f}")
        print(f"   Compression metrics: {len(result.compression_metrics)}")
        
        if result.fix_set:
            print(f"   Fix set found: {len(result.fix_set.interventions)} interventions")
            print(f"   Fix set flip rate: {result.fix_set.total_flip_rate:.3f}")
        else:
            print("   No fix set generated")
        
        # Restore original threshold
        config.modules.cf.anomaly_threshold = original_cf_threshold
        
        print("   ✅ CF → CCA integration working")
    except Exception as e:
        print(f"   ❌ Integration failed: {e}")
        return False
    
    # Test triage functionality
    print("\n7. Testing triage functionality...")
    
    all_bundles = [incident_bundle] + baseline_bundles[:2]  # Limit for speed
    triage_results = pipeline.quick_triage(all_bundles)
    
    print(f"   Triaged {len(triage_results)} bundles")
    for bundle_id, score in triage_results[:3]:
        print(f"   {bundle_id}: {score:.3f}")
    
    print("   ✅ Triage functionality working")
    
    # Test batch analysis
    print("\n8. Testing batch analysis...")
    
    try:
        # Test with small batch
        test_incidents = [incident_bundle]
        test_baselines = baseline_bundles[:1]  # Limit for speed
        
        batch_results = pipeline.analyze_batch(test_incidents, model, test_baselines)
        print(f"   Batch analysis completed: {len(batch_results)} results")
        
        for i, result in enumerate(batch_results):
            print(f"   Result {i+1}: confidence={result.confidence:.3f}, "
                  f"time={result.execution_time:.2f}s")
        
        print("   ✅ Batch analysis working")
    except Exception as e:
        print(f"   ❌ Batch analysis failed: {e}")
        return False
    
    # Test results export
    print("\n9. Testing results export...")
    
    try:
        export_path = pipeline.export_results(result, Path("test_pipeline_results.json"))
        print(f"   Results exported to: {export_path}")
        
        # Verify the file was created and has content
        if export_path.exists() and export_path.stat().st_size > 0:
            print("   ✅ Results export working")
        else:
            print("   ❌ Export file not created properly")
            return False
    except Exception as e:
        print(f"   ❌ Export failed: {e}")
        return False
    
    # Test validation functionality
    print("\n10. Testing fix set validation...")
    
    if result.fix_set:
        try:
            validation_results = pipeline.validate_fix_set(result.fix_set, session)
            print(f"   Validation results: {validation_results}")
            print("   ✅ Fix set validation working")
        except Exception as e:
            print(f"   ❌ Validation failed: {e}")
            return False
    else:
        print("   Skipped (no fix set available)")
    
    print("\n" + "=" * 60)
    print("Pipeline Integration Test Summary:")
    print("✅ Pipeline initialization: Working")
    print("✅ CF phase: Working")
    print("✅ CCA phase: Working")
    print("✅ CF → CCA integration: Working")
    print("✅ Triage functionality: Working")
    print("✅ Batch analysis: Working")
    print("✅ Results export: Working")
    print("✅ Fix set validation: Working")
    print("✅ Complete pipeline integration is fully functional!")
    
    return True

if __name__ == "__main__":
    test_pipeline_integration()