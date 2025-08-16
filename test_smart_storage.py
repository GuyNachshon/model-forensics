#!/usr/bin/env python3
"""
Test the smart storage strategy to see how it handles benign vs failure cases.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from scenarios.rca_analyzer import ScenarioRCAAnalyzer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_storage_decisions():
    """Test the smart storage decision logic."""
    
    output_dir = Path("results/storage_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    rca_analyzer = ScenarioRCAAnalyzer(output_dir)
    
    model_name = "Qwen/Qwen3-4B-Thinking-2507"
    
    # Test different classification results
    test_cases = [
        {"verdict": True, "classifier_verdict": True, "case": "clear_failure"},
        {"verdict": False, "classifier_verdict": True, "case": "classifier_detected_failure"},
        {"verdict": True, "classifier_verdict": False, "case": "verdict_detected_failure"},
        {"verdict": False, "classifier_verdict": False, "case": "benign"},
    ]
    
    logger.info("🧪 Testing storage decision logic...")
    
    for i, classification in enumerate(test_cases):
        case_type = classification.pop("case")
        
        should_analyze = rca_analyzer.should_analyze_failure(classification)
        should_save = rca_analyzer.should_save_bundle(classification, model_name)
        
        logger.info(f"Case {i+1} ({case_type}):")
        logger.info(f"  📋 Classification: {classification}")
        logger.info(f"  🔍 Should analyze: {should_analyze}")
        logger.info(f"  💾 Should save: {should_save}")
        
        if should_analyze:
            logger.info(f"  ➡️  Action: RUN RCA ANALYSIS (failure detected)")
        elif should_save:
            logger.info(f"  ➡️  Action: SAVE FOR BASELINE (benign but building baselines)")
        else:
            logger.info(f"  ➡️  Action: SKIP SAVING (benign + enough baselines)")
        
        logger.info("")
    
    # Test baseline counting
    baseline_count = rca_analyzer._count_existing_baselines(model_name)
    logger.info(f"📊 Current baseline count for {model_name}: {baseline_count}")
    
    # Simulate multiple benign cases to see probability changes
    logger.info("🎲 Testing benign case probability (100 simulations):")
    
    benign_classification = {"verdict": False, "classifier_verdict": False}
    save_count = 0
    
    for _ in range(100):
        if rca_analyzer.should_save_bundle(benign_classification, model_name):
            save_count += 1
    
    logger.info(f"   Would save {save_count}/100 benign cases ({save_count}%)")
    
    if baseline_count < 50:
        logger.info(f"   Expected: ~20% (building baselines, have {baseline_count}/50)")
    else:
        logger.info(f"   Expected: ~2% (enough baselines, have {baseline_count}/50)")

if __name__ == "__main__":
    test_storage_decisions()