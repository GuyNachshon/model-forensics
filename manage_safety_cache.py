#!/usr/bin/env python3
"""
Safety Evaluation Cache Management Utility

Provides commands to view, manage, and maintain the universal safety evaluation cache.
"""

import sys
import argparse
from pathlib import Path

# Add scenarios directory to path
sys.path.append(str(Path(__file__).parent / "scenarios"))

from safety_cache import SafetyEvaluationCache, SafetyVerdict


def show_stats(cache: SafetyEvaluationCache):
    """Display cache statistics."""
    stats = cache.get_cache_stats()
    
    print("🗂️ Safety Evaluation Cache Statistics")
    print("=" * 50)
    print(f"Cache file: {stats['cache_file']}")
    print(f"Total entries: {stats['total']}")
    print()
    
    if stats['total'] > 0:
        print("📊 By Verdict:")
        for verdict, count in stats['by_verdict'].items():
            percentage = (count / stats['total']) * 100
            print(f"  {verdict}: {count} ({percentage:.1f}%)")
        print()
        
        print("🎯 By Evaluation Type:")
        for eval_type, count in stats['by_type'].items():
            percentage = (count / stats['total']) * 100
            print(f"  {eval_type}: {count} ({percentage:.1f}%)")
        print()
        
        print("🤖 By Model:")
        for model, count in stats['by_model'].items():
            percentage = (count / stats['total']) * 100
            print(f"  {model}: {count} ({percentage:.1f}%)")


def list_entries(cache: SafetyEvaluationCache, model_filter: str = None, verdict_filter: str = None, limit: int = 20):
    """List cache entries with optional filtering."""
    print(f"📋 Cache Entries (showing up to {limit})")
    print("=" * 80)
    
    count = 0
    for key in sorted(cache._cache.keys()):
        if count >= limit:
            break
            
        entry_data = cache._cache[key]
        
        # Apply filters
        if model_filter and model_filter not in key:
            continue
        if verdict_filter and entry_data['verdict'] != verdict_filter:
            continue
            
        model, scenario, sample = key.split(':')
        verdict = entry_data['verdict']
        classification_type = entry_data['classification_type']
        timestamp = entry_data['timestamp'][:19]  # Trim milliseconds
        
        # Use emoji for verdict
        verdict_emoji = "✅" if verdict == "safe" else "❌" if verdict == "unsafe" else "❓"
        
        print(f"{verdict_emoji} {model}")
        print(f"   {scenario}:{sample} | {classification_type} | {timestamp}")
        print(f"   Reasoning: {entry_data['reasoning'][:100]}...")
        print()
        count += 1
    
    if count == 0:
        print("No entries found matching the filters.")


def clear_cache(cache: SafetyEvaluationCache, model: str = None, scenario: str = None, confirm: bool = False):
    """Clear cache entries with confirmation."""
    if not confirm:
        if model or scenario:
            print(f"⚠️ This will clear cache entries for model='{model}', scenario='{scenario}'")
        else:
            print("⚠️ This will clear ALL cache entries")
        
        response = input("Are you sure? (y/N): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
    
    cache.clear_cache(model_name=model, scenario=scenario)
    print("✅ Cache cleared successfully.")


def populate_from_previous_experiment(cache: SafetyEvaluationCache, experiment_dir: str):
    """Populate cache from a previous experiment directory."""
    experiment_path = Path(experiment_dir)
    
    if not experiment_path.exists():
        print(f"❌ Experiment directory not found: {experiment_path}")
        return
    
    print(f"🔄 Scanning {experiment_path} for response files...")
    
    models_dir = experiment_path / "models"
    if not models_dir.exists():
        print(f"❌ No models directory found in {experiment_path}")
        return
    
    added_count = 0
    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        # Convert sanitized model name back (basic approach)
        model_name = model_dir.name.replace("_", "/", 1)  # First underscore becomes slash
        
        for condition_dir in model_dir.iterdir():
            if not condition_dir.is_dir():
                continue
                
            condition = condition_dir.name
            
            for sample_dir in condition_dir.iterdir():
                if not sample_dir.is_dir() or not sample_dir.name.startswith("sample_"):
                    continue
                
                sample_id = int(sample_dir.name.split("_")[1])
                response_file = sample_dir / "response.json"
                
                if response_file.exists():
                    try:
                        import json
                        with open(response_file, 'r') as f:
                            response_data = json.load(f)
                        
                        classification = response_data.get("classification", {})
                        if classification:
                            # Convert to cache format
                            verdict = SafetyVerdict.UNSAFE if classification.get("verdict", False) else SafetyVerdict.SAFE
                            classifier_verdict = SafetyVerdict.UNSAFE if classification.get("classifier_verdict", False) else SafetyVerdict.SAFE
                            
                            # Determine classification type
                            if "blackmail" in condition:
                                classification_type = "blackmail"
                            elif "leaking" in condition:
                                classification_type = "leaking"
                            elif "murder" in condition:
                                classification_type = "murder"
                            else:
                                classification_type = "unknown"
                            
                            # Only add if not already cached
                            if not cache.has_cached_result(model_name, condition, sample_id):
                                cache.cache_result(
                                    model_name=model_name,
                                    scenario=condition,
                                    sample_id=sample_id,
                                    verdict=verdict,
                                    classifier_verdict=classifier_verdict,
                                    classification_type=classification_type,
                                    reasoning=classification.get("reasoning", ""),
                                    model_config={}
                                )
                                added_count += 1
                                
                    except Exception as e:
                        print(f"⚠️ Failed to process {response_file}: {e}")
    
    print(f"✅ Added {added_count} new cache entries from {experiment_path}")


def main():
    parser = argparse.ArgumentParser(description="Manage Safety Evaluation Cache")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Stats command
    subparsers.add_parser("stats", help="Show cache statistics")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List cache entries")
    list_parser.add_argument("--model", help="Filter by model name")
    list_parser.add_argument("--verdict", choices=["safe", "unsafe", "unknown"], help="Filter by verdict")
    list_parser.add_argument("--limit", type=int, default=20, help="Maximum entries to show")
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear cache entries")
    clear_parser.add_argument("--model", help="Clear entries for specific model")
    clear_parser.add_argument("--scenario", help="Clear entries for specific scenario")
    clear_parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    
    # Populate command
    populate_parser = subparsers.add_parser("populate", help="Populate cache from previous experiment")
    populate_parser.add_argument("experiment_dir", help="Path to experiment directory")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize cache
    cache = SafetyEvaluationCache()
    
    if args.command == "stats":
        show_stats(cache)
    elif args.command == "list":
        list_entries(cache, args.model, args.verdict, args.limit)
    elif args.command == "clear":
        clear_cache(cache, args.model, args.scenario, args.yes)
    elif args.command == "populate":
        populate_from_previous_experiment(cache, args.experiment_dir)


if __name__ == "__main__":
    main()