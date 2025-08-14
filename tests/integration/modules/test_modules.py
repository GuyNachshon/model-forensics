#!/usr/bin/env python3
"""Test script for CF and CCA modules using existing bundles."""

import logging
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.core.config import RCAConfig
from src.modules.pipeline import RCAPipeline
from src.replayer.core import Replayer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_modules():
    """Test CF and CCA modules with existing bundles."""
    logger.info("Testing CF and CCA modules")
    
    # Load configuration
    config = RCAConfig()
    
    # Initialize pipeline
    pipeline = RCAPipeline(config)
    
    # Load model (using GPT-2 as per existing bundles)
    logger.info("Loading GPT-2 model")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    
    # Load existing bundles
    bundle_dir = Path("bundles")
    replayer = Replayer(config)
    
    # Test with injection incident
    injection_bundle_path = bundle_dir / "injection_incident_001"
    if injection_bundle_path.exists():
        logger.info(f"Testing with injection bundle: {injection_bundle_path}")
        
        try:
            # Load incident bundle
            incident_bundle = replayer.load_bundle(injection_bundle_path)
            logger.info(f"Loaded incident bundle: {incident_bundle.bundle_id}")
            
            # Load benign baseline
            benign_bundle_path = bundle_dir / "benign_case_001"
            baseline_bundles = []
            
            if benign_bundle_path.exists():
                benign_bundle = replayer.load_bundle(benign_bundle_path)
                baseline_bundles = [benign_bundle]
                logger.info("Loaded benign baseline bundle")
            
            # Run analysis
            logger.info("Running RCA analysis")
            result = pipeline.analyze_incident(incident_bundle, model, baseline_bundles)
            
            # Print results
            logger.info(f"Analysis completed in {result.execution_time:.2f}s")
            logger.info(f"Overall confidence: {result.confidence:.3f}")
            
            if result.compression_metrics:
                anomalous_layers = [m for m in result.compression_metrics if m.is_anomalous]
                logger.info(f"CF found {len(anomalous_layers)} anomalous layers:")
                for m in anomalous_layers[:5]:  # Show top 5
                    logger.info(f"  {m.layer_name}: score={m.anomaly_score:.3f}, confidence={m.confidence:.3f}")
            
            if result.fix_set:
                logger.info(f"CCA found minimal fix set with {len(result.fix_set.interventions)} interventions")
                logger.info(f"Fix set flip rate: {result.fix_set.total_flip_rate:.3f}")
                for interv in result.fix_set.interventions[:3]:  # Show top 3
                    logger.info(f"  {interv.intervention.layer_name} ({interv.intervention.type.value})")
            
            # Export results
            output_path = pipeline.export_results(result)
            logger.info(f"Results exported to: {output_path}")
            
            logger.info("✅ Module test completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Module test failed: {e}")
            raise
    
    else:
        logger.warning(f"Injection bundle not found at {injection_bundle_path}")
        
        # Test with any available bundle
        available_bundles = list(bundle_dir.glob("*/manifest.json"))
        if available_bundles:
            test_bundle_path = available_bundles[0].parent
            logger.info(f"Testing with available bundle: {test_bundle_path}")
            
            try:
                bundle = replayer.load_bundle(test_bundle_path)
                result = pipeline.analyze_incident(bundle, model)
                logger.info(f"✅ Basic test completed - confidence: {result.confidence:.3f}")
                
            except Exception as e:
                logger.error(f"❌ Basic test failed: {e}")
        
        else:
            logger.error("❌ No bundles found for testing")

def test_triage():
    """Test quick triage functionality."""
    logger.info("Testing triage functionality")
    
    config = RCAConfig()
    pipeline = RCAPipeline(config)
    replayer = Replayer(config)
    
    # Load multiple bundles for triage
    bundle_dir = Path("bundles")
    bundles = []
    
    for bundle_path in bundle_dir.iterdir():
        if bundle_path.is_dir() and (bundle_path / "manifest.json").exists():
            try:
                bundle = replayer.load_bundle(bundle_path)
                bundles.append(bundle)
            except Exception as e:
                logger.warning(f"Failed to load bundle {bundle_path}: {e}")
    
    if bundles:
        logger.info(f"Running triage on {len(bundles)} bundles")
        triage_results = pipeline.quick_triage(bundles)
        
        logger.info("Triage results (ranked by anomaly score):")
        for bundle_id, score in triage_results:
            logger.info(f"  {bundle_id}: {score:.3f}")
        
        logger.info("✅ Triage test completed")
    else:
        logger.warning("❌ No bundles available for triage test")

if __name__ == "__main__":
    test_modules()
    test_triage()