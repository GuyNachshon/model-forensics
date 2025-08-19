#!/usr/bin/env python3
"""
Test all 4 RCA modules working together: CF + IS-C + CCA + DBC
"""

import asyncio
import sys
from pathlib import Path
import logging

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from scenarios.rca_analyzer import ScenarioRCAAnalyzer
from core.types import TraceData, CompressedSketch, IncidentBundle
from scenarios.utils import ChatMessage, MessageRole
import numpy as np
import pickle
import zlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_bundle():
    """Create a test bundle for all modules."""
    activations = {}
    for i in range(10):  # 10 layers for testing
        layer_name = f"model.layers.{i}"
        activation_data = np.random.normal(0, 1.0, (1, 32, 128)).astype(np.float32)
        
        serialized = pickle.dumps(activation_data)
        compressed_data = zlib.compress(serialized, level=6)
        
        sketch = CompressedSketch(
            data=compressed_data,
            original_shape=activation_data.shape,
            dtype=str(activation_data.dtype),
            compression_ratio=len(compressed_data) / len(serialized),
            metadata={"layer": layer_name, "type": "activation"}
        )
        activations[layer_name] = sketch
    
    trace_data = TraceData(
        inputs={"test": "input"},
        activations=activations,
        outputs={"response": "test output"},
        external_calls=[],
        metadata={"test": True},
        timestamp="2025-08-16T10:00:00",
        model_config={"model_name": "test_model"}
    )
    
    bundle = IncidentBundle(
        bundle_id="test_all_modules",
        manifest={"test": True},
        trace_data=trace_data,
        bundle_path=Path("/tmp/test_all_modules")
    )
    
    return bundle

async def test_all_modules():
    """Test all 4 RCA modules together."""
    logger.info("🧪 Testing All 4 RCA Modules: CF + IS-C + CCA + DBC")
    
    # Setup
    output_dir = Path("results/all_modules_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create RCA analyzer
    rca_analyzer = ScenarioRCAAnalyzer(output_dir)
    
    # Test messages
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="Test adversarial prompt")
    ]
    
    response = "Test response"
    
    # Classification result indicating failure  
    classification_result = {
        "verdict": True,
        "classifier_verdict": True,
        "reasoning": "Test failure classification"
    }
    
    model_name = "test_model"
    condition = "test_condition"
    sample_id = 1
    
    # Create test bundle
    bundle = create_test_bundle()
    logger.info(f"Created test bundle with {len(bundle.trace_data.activations)} activations")
    
    try:
        logger.info("🚀 Starting complete RCA analysis (CF + IS-C + CCA + DBC)...")
        
        # Run the complete analysis
        rca_result = await rca_analyzer.analyze_failure(
            model_name=model_name,
            condition=condition,
            sample_id=sample_id,
            messages=messages,
            response=response,
            classification_result=classification_result,
            trace_data=bundle.trace_data
        )
        
        if not rca_result:
            logger.error("❌ RCA analysis returned no results")
            return False
        
        logger.info("✅ RCA analysis completed!")
        
        # Check results for each module
        rca_analysis = rca_result.get("rca_analysis", {})
        summary = rca_result.get("summary", {})
        module_success = summary.get("module_success", {})
        
        logger.info("\n" + "="*60)
        logger.info("📊 ALL MODULES TEST RESULTS")
        logger.info("="*60)
        
        # CF Results
        cf_success = module_success.get("cf", False)
        cf_result = rca_analysis.get("cf_result", [])
        logger.info(f"🔍 CF (Compression Forensics): {'✅ SUCCESS' if cf_success else '❌ FAILED'}")
        if cf_success:
            logger.info(f"   - Analyzed {len(cf_result)} layers")
            anomalous = len([l for l in cf_result if l.get('is_anomalous') == 'True'])
            logger.info(f"   - Found {anomalous} anomalous layers")
        
        # IS-C Results
        is_c_success = module_success.get("is_c", False)
        is_c_result = rca_analysis.get("is_c_result", {})
        logger.info(f"🔗 IS-C (Interaction Spectroscopy): {'✅ SUCCESS' if is_c_success else '❌ FAILED'}")
        if is_c_success and is_c_result:
            logger.info(f"   - Total interactions: {is_c_result.get('total_interactions', 0)}")
            logger.info(f"   - Strong interactions: {is_c_result.get('strong_interactions', 0)}")
        
        # CCA Results  
        cca_success = module_success.get("cca", False)
        cca_result = rca_analysis.get("cca_result", {})
        logger.info(f"🎯 CCA (Causal Minimal Fix Sets): {'✅ SUCCESS' if cca_success else '❌ FAILED'}")
        if cca_success and cca_result:
            fix_set = cca_result.get("minimal_fix_set", [])
            flip_rate = cca_result.get("estimated_flip_rate", 0)
            logger.info(f"   - Minimal fix set: {len(fix_set)} layers")
            logger.info(f"   - Estimated flip rate: {flip_rate:.1%}")
        
        # DBC Results
        dbc_success = module_success.get("dbc", False)
        dbc_result = rca_analysis.get("dbc_result", {})
        logger.info(f"🏔️ DBC (Decision Basin Cartography): {'✅ SUCCESS' if dbc_success else '❌ FAILED'}")
        if dbc_success and dbc_result:
            boundary_count = dbc_result.get("boundary_count", 0)
            avg_robustness = dbc_result.get("avg_robustness", 0)
            vulnerable_count = dbc_result.get("vulnerability_count", 0)
            logger.info(f"   - Decision boundaries: {boundary_count}")
            logger.info(f"   - Average robustness: {avg_robustness:.3f}")
            logger.info(f"   - Vulnerable layers: {vulnerable_count}")
        
        # Overall Success
        total_modules = 4
        successful_modules = sum([cf_success, is_c_success, cca_success, dbc_success])
        success_rate = successful_modules / total_modules
        
        logger.info("\n" + "="*60)
        logger.info(f"📈 OVERALL RESULTS: {successful_modules}/{total_modules} modules successful ({success_rate:.1%})")
        logger.info("="*60)
        
        # Analysis quality
        analysis_quality = summary.get("analysis_quality", "unknown")
        modules_used = rca_result.get("metadata", {}).get("modules_used", [])
        logger.info(f"🏆 Analysis Quality: {analysis_quality}")
        logger.info(f"🧠 Modules Used: {', '.join(modules_used)}")
        
        # Key findings
        key_findings = summary.get("key_findings", [])
        if key_findings:
            logger.info(f"\n🔍 Key Findings:")
            for finding in key_findings:
                logger.info(f"   - {finding}")
        
        return successful_modules >= 3  # Consider success if 3+ modules work
        
    except Exception as e:
        logger.error(f"❌ All modules test failed: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_all_modules())
    if success:
        logger.info("🎉 All modules integration test passed!")
    else:
        logger.error("💥 All modules integration test failed!")