"""
Universal Safety Evaluation Cache System

Provides persistent caching for model safety evaluation results across all
evaluation types (alignment, jailbreak, toxicity, etc.) to avoid redundant
expensive processing.
"""

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class SafetyVerdict(Enum):
    """Safety evaluation verdict."""
    SAFE = "safe"
    UNSAFE = "unsafe" 
    UNKNOWN = "unknown"


@dataclass
class CacheEntry:
    """Safety evaluation cache entry."""
    verdict: SafetyVerdict
    classifier_verdict: SafetyVerdict
    classification_type: str  # blackmail, murder, leak, jailbreak, etc.
    reasoning: str
    timestamp: str
    model_hash: str
    confidence: Optional[float] = None


class SafetyEvaluationCache:
    """
    Persistent cache for safety evaluation results.
    
    Stores verdict results to avoid re-running expensive model inference
    and classification for samples we've already evaluated.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize cache with optional custom directory."""
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Default to user cache directory
            self.cache_dir = Path.home() / ".cache" / "model-forensics"
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "safety_evaluation_cache.json"
        
        # Load existing cache
        self._cache = self._load_cache()
        logger.info(f"Initialized SafetyEvaluationCache with {len(self._cache)} entries at {self.cache_file}")
    
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load cache from disk."""
        if not self.cache_file.exists():
            return {}
            
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            logger.info(f"Loaded {len(cache_data)} cache entries from {self.cache_file}")
            return cache_data
        except Exception as e:
            logger.warning(f"Failed to load cache from {self.cache_file}: {e}")
            return {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved cache with {len(self._cache)} entries to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache to {self.cache_file}: {e}")
    
    def _get_cache_key(self, model_name: str, scenario: str, sample_id: int) -> str:
        """Generate cache key for a sample."""
        return f"{model_name}:{scenario}:{sample_id}"
    
    def _get_model_hash(self, model_name: str, config: Dict[str, Any] = None) -> str:
        """Generate hash of model configuration for cache invalidation."""
        # Simple hash of model name + relevant config
        hash_input = model_name
        if config:
            # Add relevant config params that affect evaluation
            relevant_keys = ['temperature', 'max_tokens', 'top_p', 'top_k']
            config_str = ''.join(f"{k}:{config.get(k, '')}" for k in relevant_keys)
            hash_input += config_str
            
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def has_cached_result(self, model_name: str, scenario: str, sample_id: int) -> bool:
        """Check if sample has cached evaluation result."""
        cache_key = self._get_cache_key(model_name, scenario, sample_id)
        return cache_key in self._cache
    
    def get_cached_result(self, model_name: str, scenario: str, sample_id: int) -> Optional[CacheEntry]:
        """Get cached evaluation result."""
        cache_key = self._get_cache_key(model_name, scenario, sample_id)
        
        if cache_key not in self._cache:
            return None
            
        try:
            cache_data = self._cache[cache_key]
            entry = CacheEntry(
                verdict=SafetyVerdict(cache_data["verdict"]),
                classifier_verdict=SafetyVerdict(cache_data["classifier_verdict"]),
                classification_type=cache_data["classification_type"],
                reasoning=cache_data["reasoning"],
                timestamp=cache_data["timestamp"],
                model_hash=cache_data["model_hash"],
                confidence=cache_data.get("confidence")
            )
            return entry
        except Exception as e:
            logger.warning(f"Failed to parse cached result for {cache_key}: {e}")
            # Remove corrupted entry
            del self._cache[cache_key]
            return None
    
    def cache_result(self, model_name: str, scenario: str, sample_id: int, 
                    verdict: SafetyVerdict, classifier_verdict: SafetyVerdict,
                    classification_type: str, reasoning: str = "",
                    model_config: Dict[str, Any] = None, confidence: float = None):
        """Cache evaluation result."""
        cache_key = self._get_cache_key(model_name, scenario, sample_id)
        
        entry = CacheEntry(
            verdict=verdict,
            classifier_verdict=classifier_verdict,
            classification_type=classification_type,
            reasoning=reasoning,
            timestamp=datetime.now().isoformat(),
            model_hash=self._get_model_hash(model_name, model_config),
            confidence=confidence
        )
        
        # Store as dict for JSON serialization
        self._cache[cache_key] = asdict(entry)
        # Convert enums to strings
        self._cache[cache_key]["verdict"] = entry.verdict.value
        self._cache[cache_key]["classifier_verdict"] = entry.classifier_verdict.value
        
        self._save_cache()
        
        logger.info(f"📝 CACHED_RESULT | {cache_key} | verdict:{verdict.value} classifier:{classifier_verdict.value} type:{classification_type}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._cache:
            return {"total": 0, "by_verdict": {}, "by_type": {}, "by_model": {}}
        
        by_verdict = {}
        by_type = {}
        by_model = {}
        
        for key, entry in self._cache.items():
            model = key.split(":")[0]
            verdict = entry["verdict"]
            classification_type = entry["classification_type"]
            
            by_verdict[verdict] = by_verdict.get(verdict, 0) + 1
            by_type[classification_type] = by_type.get(classification_type, 0) + 1
            by_model[model] = by_model.get(model, 0) + 1
        
        return {
            "total": len(self._cache),
            "by_verdict": by_verdict,
            "by_type": by_type,
            "by_model": by_model,
            "cache_file": str(self.cache_file)
        }
    
    def clear_cache(self, model_name: str = None, scenario: str = None):
        """Clear cache entries. If model_name/scenario specified, clear only matching entries."""
        if model_name is None and scenario is None:
            # Clear all
            cleared = len(self._cache)
            self._cache.clear()
            logger.info(f"🗑️ CLEARED_ALL_CACHE | {cleared} entries removed")
        else:
            # Clear matching entries
            keys_to_remove = []
            for key in self._cache:
                parts = key.split(":")
                key_model = parts[0]
                key_scenario = parts[1] if len(parts) > 1 else ""
                
                should_remove = True
                if model_name and key_model != model_name:
                    should_remove = False
                if scenario and key_scenario != scenario:
                    should_remove = False
                    
                if should_remove:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._cache[key]
            
            logger.info(f"🗑️ CLEARED_CACHE | {len(keys_to_remove)} entries removed (model:{model_name}, scenario:{scenario})")
        
        self._save_cache()
    
    def should_run_full_evaluation(self, model_name: str, scenario: str, sample_id: int) -> Tuple[bool, Optional[str]]:
        """
        Determine if full evaluation should be run based on cache.
        
        Returns:
            (should_run_full, reason)
            - should_run_full: True if full evaluation needed
            - reason: Human-readable reason for the decision
        """
        cached_entry = self.get_cached_result(model_name, scenario, sample_id)
        
        if cached_entry is None:
            return True, "no_cache_entry"
        
        if cached_entry.verdict == SafetyVerdict.SAFE:
            return False, "cached_safe"
        elif cached_entry.verdict == SafetyVerdict.UNSAFE:
            return False, "cached_unsafe_run_rca_only"
        else:  # UNKNOWN
            return True, "cached_unknown_rerun"
    
    def should_run_rca_analysis(self, model_name: str, scenario: str, sample_id: int) -> bool:
        """Determine if RCA analysis should be run based on cache."""
        cached_entry = self.get_cached_result(model_name, scenario, sample_id)
        
        if cached_entry is None:
            return True  # Will be determined after classification
            
        # Only run RCA on unsafe samples
        return cached_entry.verdict == SafetyVerdict.UNSAFE