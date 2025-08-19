#!/usr/bin/env python3
"""
Test the baseline collection system.
"""

import asyncio
import sys
from pathlib import Path
import logging

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from scenarios.baseline_collector import BaselineCollector
from core.utils import setup_logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_baseline_collection():
    """Test baseline collection for a model."""
    
    logger.info("🧪 Testing Baseline Collection System")
    
    # Setup
    output_dir = Path("results/baseline_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    collector = BaselineCollector(output_dir)
    
    # Test model
    model_name = "Qwen/Qwen3-4B-Thinking-2507"
    
    # Check existing baseline
    logger.info(f"📊 Checking existing baseline for {model_name}...")
    has_baseline = collector.has_baseline(model_name)
    logger.info(f"Existing baseline: {'✅ Found' if has_baseline else '❌ Not found'}")
    
    # List available baselines
    baselines = collector.list_available_baselines()
    logger.info(f"📋 Available baselines: {len(baselines)}")
    for baseline in baselines:
        logger.info(f"  - {baseline['model_name']}: {baseline['sample_count']} samples")
    
    # Collect baseline if needed
    if not has_baseline:
        logger.info(f"🔄 Collecting baseline for {model_name}...")
        success = await collector.collect_baseline(model_name, target_samples=10)  # Small test
        
        if success:
            logger.info("✅ Baseline collection successful!")
            
            # Verify baseline info
            baseline_info = collector.load_baseline_info(model_name)
            if baseline_info:
                logger.info(f"📊 Baseline Info:")
                logger.info(f"  - Total samples: {baseline_info['total_samples']}")
                logger.info(f"  - Success rate: {baseline_info['collection_stats']['success_rate']:.1%}")
                logger.info(f"  - Categories: {baseline_info['collection_stats']['categories']}")
                logger.info(f"  - Avg activations: {baseline_info['collection_stats']['avg_activations']:.1f}")
        else:
            logger.error("❌ Baseline collection failed!")
            return False
    else:
        logger.info("✅ Using existing baseline")
    
    # Test baseline loading
    bundle_ids = collector.get_baseline_bundle_ids(model_name)
    logger.info(f"📦 Baseline bundle IDs: {len(bundle_ids)}")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_baseline_collection())
    if success:
        logger.info("🎉 Baseline collection test completed successfully!")
    else:
        logger.error("💥 Baseline collection test failed!")