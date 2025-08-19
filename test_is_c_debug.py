#!/usr/bin/env python3
"""
Debug IS-C integration issues.
"""

import asyncio
import sys
from pathlib import Path
import logging

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from scenarios.rca_analyzer import ScenarioRCAAnalyzer
from core.types import TraceData, CompressedSketch, IncidentBundle
import numpy as np
import pickle
import zlib

# Setup logging with debug level to see warnings
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_simple_bundle():
    """Create a simple test bundle for IS-C debugging."""
    # Create minimal activations
    activations = {}
    for i in range(5):  # Just 5 layers for debugging
        layer_name = f"model.layers.{i}"
        activation_data = np.random.normal(0, 1.0, (1, 10, 100)).astype(np.float32)
        
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
        bundle_id="test_bundle_is_c_debug",
        manifest={"test": True},
        trace_data=trace_data,
        bundle_path=Path("/tmp/test_bundle")
    )
    
    return bundle

async def test_is_c_directly():
    """Test IS-C module directly to see what's failing."""
    logger.info("🔍 Testing IS-C integration directly")
    
    # Setup
    output_dir = Path("results/is_c_debug_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create RCA analyzer
    rca_analyzer = ScenarioRCAAnalyzer(output_dir)
    
    # Initialize components
    rca_analyzer._init_rca_components()
    
    # Create test bundle
    bundle = create_simple_bundle()
    logger.info(f"Created test bundle with {len(bundle.trace_data.activations)} activations")
    
    # Create mock CF result
    cf_result = []
    for i in range(5):
        cf_result.append({
            "layer_name": f"model.layers.{i}",
            "compression_ratio": 1.0 + i * 0.1,
            "anomaly_score": 0.1 + i * 0.05,
            "is_anomalous": "False"
        })
    
    logger.info(f"Created CF result with {len(cf_result)} layers")
    
    # Test IS-C directly
    try:
        logger.info("🧪 Running IS-C analysis...")
        is_c_result = await rca_analyzer._run_is_c_analysis(bundle, cf_result)
        
        if is_c_result:
            logger.info("✅ IS-C analysis completed!")
            logger.info(f"Status: {is_c_result.get('status', 'unknown')}")
            logger.info(f"Total interactions: {is_c_result.get('total_interactions', 0)}")
            logger.info(f"Strong interactions: {is_c_result.get('strong_interactions', 0)}")
            logger.info(f"Layer count: {is_c_result.get('layer_count', 0)}")
            logger.info(f"Reconstructed layers: {is_c_result.get('reconstructed_layers', 0)}")
            
            if 'error' in is_c_result:
                logger.error(f"IS-C error: {is_c_result['error']}")
                return False
            
            return True
        else:
            logger.error("IS-C returned None")
            return False
            
    except Exception as e:
        logger.error(f"IS-C test failed: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_is_c_directly())
    if success:
        logger.info("🎉 IS-C debug test passed!")
    else:
        logger.error("💥 IS-C debug test failed!")