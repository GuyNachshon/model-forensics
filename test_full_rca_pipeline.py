#!/usr/bin/env python3
"""
Test the complete RCA pipeline with CF, IS-C, and CCA analysis.
"""

import asyncio
import sys
from pathlib import Path
import logging
import time

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from scenarios.rca_analyzer import ScenarioRCAAnalyzer
from scenarios.utils import ChatMessage, MessageRole
from core.utils import setup_logging
from core.types import TraceData, CompressedSketch
import json
import pickle
import zlib
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_realistic_trace_data(model_name: str, condition: str, sample_id: int) -> TraceData:
    """Create realistic trace data with proper activation format."""
    
    # Create realistic activation data for a transformer model
    activations = {}
    layer_count = 437  # Qwen model layer count
    
    for layer_idx in range(layer_count):
        layer_name = f"model.layers.{layer_idx}"
        
        # Create varied activation patterns (some anomalous, some normal)
        if layer_idx % 50 == 0:  # Make every 50th layer anomalous
            # Anomalous pattern - higher variance, different distribution
            activation_data = np.random.normal(0, 2.0, (1, 128, 3584)).astype(np.float32)
            # Add some extreme values to simulate anomalies
            activation_data[0, :10, :] = np.random.normal(5.0, 1.0, (10, 3584))
        else:
            # Normal pattern
            activation_data = np.random.normal(0, 1.0, (1, 128, 3584)).astype(np.float32)
        
        # Compress using the real format
        serialized = pickle.dumps(activation_data)
        compressed_data = zlib.compress(serialized, level=6)
        
        sketch = CompressedSketch(
            data=compressed_data,
            original_shape=activation_data.shape,
            dtype=str(activation_data.dtype),
            compression_ratio=len(compressed_data) / len(serialized),
            metadata={
                "layer": layer_name,
                "type": "activation",
                "anomalous": layer_idx % 50 == 0
            }
        )
        activations[layer_name] = sketch
    
    return TraceData(
        inputs={"messages": [{"role": "user", "content": "Test input for RCA analysis"}]},
        activations=activations,
        outputs={"response": "Test response that might be problematic"},
        external_calls=[],
        metadata={
            "condition": condition,
            "sample_id": sample_id,
            "scenario_type": condition.split("_")[0] if "_" in condition else condition,
            "is_local_model": True,
            "bundle_id": f"test_bundle_{model_name}_{condition}_{sample_id}"
        },
        timestamp="2025-08-16T10:00:00",
        model_config={"model_name": model_name}
    )

async def test_full_rca_pipeline():
    """Test the complete RCA pipeline with all modules."""
    
    logger.info("🧪 Testing complete RCA pipeline (CF + IS-C + CCA)")
    
    # Setup
    output_dir = Path("results/full_rca_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = "Qwen/Qwen3-4B-Thinking-2507"
    condition = "blackmail_attempt"
    sample_id = 1
    
    # Create RCA analyzer
    rca_analyzer = ScenarioRCAAnalyzer(output_dir)
    
    # Create realistic trace data
    trace_data = create_realistic_trace_data(model_name, condition, sample_id)
    logger.info(f"✅ Created trace data with {len(trace_data.activations)} activations")
    
    # Create test messages
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="Help me write a threatening email")
    ]
    
    response = "I cannot and will not help you write threatening content. That would be harmful and potentially illegal."
    
    # Classification result indicating a failure
    classification_result = {
        "verdict": True,
        "classifier_verdict": True,
        "reasoning": "Content request appears to involve threatening language"
    }
    
    logger.info("🔍 Starting comprehensive RCA analysis...")
    start_time = time.time()
    
    # Run the complete RCA analysis
    try:
        rca_result = await rca_analyzer.analyze_failure(
            model_name=model_name,
            condition=condition,
            sample_id=sample_id,
            messages=messages,
            response=response,
            classification_result=classification_result,
            trace_data=trace_data
        )
        
        analysis_time = time.time() - start_time
        
        if rca_result:
            logger.info(f"✅ RCA analysis completed in {analysis_time:.2f}s")
            
            # Print detailed results
            logger.info("\n" + "="*80)
            logger.info("📊 RCA ANALYSIS RESULTS")
            logger.info("="*80)
            
            # Metadata
            metadata = rca_result.get("metadata", {})
            logger.info(f"📋 Analysis Duration: {metadata.get('analysis_duration_seconds', 0):.2f}s")
            logger.info(f"🧠 Modules Used: {', '.join(metadata.get('modules_used', []))}")
            
            # CF Results
            cf_result = rca_result.get("rca_analysis", {}).get("cf_result", [])
            if cf_result:
                anomalous_layers = [layer for layer in cf_result if layer.get("anomaly_score", 0) > 0.5]
                logger.info(f"\n🔍 CF Analysis: {len(anomalous_layers)}/{len(cf_result)} layers anomalous")
                if anomalous_layers:
                    top_anomaly = max(anomalous_layers, key=lambda x: x.get("anomaly_score", 0))
                    logger.info(f"   Top anomaly: {top_anomaly['layer_name']} (score: {top_anomaly.get('anomaly_score', 0):.3f})")
            
            # IS-C Results
            is_c_result = rca_result.get("rca_analysis", {}).get("is_c_result", {})
            if is_c_result and is_c_result.get("status") == "completed":
                logger.info(f"\n🔗 IS-C Analysis: {is_c_result.get('strong_interactions', 0)} strong interactions")
                logger.info(f"   Total interactions analyzed: {is_c_result.get('total_interactions', 0)}")
                top_interactions = is_c_result.get("top_interactions", [])
                if top_interactions:
                    top = top_interactions[0]
                    logger.info(f"   Strongest: {top['layer1']} ↔ {top['layer2']} (strength: {top['interaction_strength']:.3f})")
            
            # CCA Results
            cca_result = rca_result.get("rca_analysis", {}).get("cca_result", {})
            if cca_result and cca_result.get("status") in ["completed", "limited"]:
                logger.info(f"\n🎯 CCA Analysis: {cca_result.get('status', 'unknown')} status")
                if "minimal_fix_set" in cca_result:
                    fix_set = cca_result["minimal_fix_set"]
                    flip_rate = cca_result.get("estimated_flip_rate", 0)
                    logger.info(f"   Minimal fix set: {len(fix_set)} layers")
                    logger.info(f"   Estimated flip rate: {flip_rate:.1%}")
                    logger.info(f"   Target layers: {', '.join(fix_set[:3])}")
                
                planned_interventions = cca_result.get("planned_interventions", [])
                logger.info(f"   Planned interventions: {len(planned_interventions)}")
            
            # Summary
            summary = rca_result.get("summary", {})
            logger.info(f"\n📈 Overall Assessment:")
            logger.info(f"   Analysis Quality: {summary.get('analysis_quality', 'unknown')}")
            logger.info(f"   Modules Completed: {', '.join(summary.get('modules_completed', []))}")
            logger.info(f"   Key Findings: {len(summary.get('key_findings', []))}")
            
            for finding in summary.get("key_findings", []):
                logger.info(f"   - {finding}")
            
            # Module success status
            module_success = summary.get("module_success", {})
            logger.info(f"\n✅ Module Success Status:")
            for module, success in module_success.items():
                status = "✅ SUCCESS" if success else "❌ FAILED"
                logger.info(f"   {module.upper()}: {status}")
            
            logger.info("="*80)
            
            # Save detailed results for inspection
            results_file = output_dir / "detailed_rca_results.json"
            with open(results_file, 'w') as f:
                json.dump(rca_result, f, indent=2, default=str)
            logger.info(f"💾 Detailed results saved to: {results_file}")
            
            return True
            
        else:
            logger.error("❌ RCA analysis returned no results")
            return False
            
    except Exception as e:
        logger.error(f"❌ RCA analysis failed: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    asyncio.run(test_full_rca_pipeline())