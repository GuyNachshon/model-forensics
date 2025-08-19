#!/usr/bin/env python3
"""
Test CF baseline integration with real baseline data.
"""

import asyncio
import sys
from pathlib import Path
import logging
import json

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from scenarios.baseline_collector import BaselineCollector
from scenarios.rca_analyzer import ScenarioRCAAnalyzer
from core.utils import setup_logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_cf_baseline_integration():
    """Test that CF can build and use baseline data correctly."""
    
    logger.info("🧪 Testing CF Baseline Integration")
    
    # Setup - use the experiment directory that has baseline data
    output_dir = Path("results/experiment_20250816-182514")
    
    # Check if baseline data exists
    baseline_collector = BaselineCollector(output_dir)
    model_name = "Qwen/Qwen3-4B-Thinking-2507"
    
    logger.info(f"📊 Checking baseline availability for {model_name}...")
    has_baseline = baseline_collector.has_baseline(model_name, min_samples=9)  # Use 9 since we have 9 complete samples
    logger.info(f"Baseline available: {'✅ Yes' if has_baseline else '❌ No'}")
    
    if not has_baseline:
        logger.error("❌ No baseline data found - cannot test CF integration")
        return False
    
    # Load baseline info
    baseline_info = baseline_collector.load_baseline_info(model_name)
    if baseline_info:
        logger.info(f"📈 Baseline Info:")
        logger.info(f"  - Total samples: {baseline_info['total_samples']}")
        logger.info(f"  - Avg activations: {baseline_info['collection_stats']['avg_activations']:.1f}")
        logger.info(f"  - Categories: {baseline_info['collection_stats']['categories'][:3]}...")
    
    # Create RCA analyzer
    rca_analyzer = ScenarioRCAAnalyzer(output_dir)
    
    # Initialize RCA components
    rca_analyzer._init_rca_components()
    logger.info("✅ RCA components initialized")
    
    # Test baseline building for CF
    logger.info("🔍 Testing CF baseline building...")
    try:
        # This should use the baseline data we collected
        cf_baseline = await rca_analyzer._build_cf_baseline(model_name)
        
        if cf_baseline is not None:
            logger.info(f"✅ CF baseline built successfully!")
            logger.info(f"  - Baseline type: {type(cf_baseline)}")
            if hasattr(cf_baseline, '__len__'):
                logger.info(f"  - Baseline size: {len(cf_baseline)}")
            return True
        else:
            logger.error("❌ CF baseline building returned None")
            return False
            
    except Exception as e:
        logger.error(f"❌ CF baseline building failed: {e}")
        import traceback
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_cf_baseline_integration())
    if success:
        logger.info("🎉 CF baseline integration test passed!")
    else:
        logger.error("💥 CF baseline integration test failed!")