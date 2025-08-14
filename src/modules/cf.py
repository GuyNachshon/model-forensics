"""Compression Forensics (CF) module for anomaly detection."""

import logging
import zlib
import gzip
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path
import pickle

from core.types import CompressionMetrics, IncidentBundle, ActivationDict
from core.config import CFConfig


logger = logging.getLogger(__name__)


class CompressionForensics:
    """Detects anomalous computation patterns via compressibility analysis."""
    
    def __init__(self, config: Optional[CFConfig] = None):
        self.config = config or CFConfig()
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        self.compression_cache: Dict[str, float] = {}
        
        logger.info(f"Initialized CF with threshold: {self.config.anomaly_threshold}")
    
    def analyze_bundle(self, bundle: IncidentBundle) -> List[CompressionMetrics]:
        """Analyze compression patterns for all activations in a bundle."""
        logger.info(f"Analyzing bundle: {bundle.bundle_id}")
        
        metrics = []
        activations = self._reconstruct_activations(bundle)
        
        for layer_name, activation in activations.items():
            try:
                layer_metrics = self.analyze_layer(activation, layer_name)
                metrics.append(layer_metrics)
                logger.debug(f"Layer {layer_name}: anomaly_score={layer_metrics.anomaly_score:.3f}")
                
            except Exception as e:
                logger.warning(f"Failed to analyze layer {layer_name}: {e}")
        
        return metrics
    
    def analyze_layer(self, activation: torch.Tensor, layer_name: str) -> CompressionMetrics:
        """Analyze compression patterns for a single layer."""
        # Convert to numpy for compression
        if isinstance(activation, torch.Tensor):
            activation_np = activation.detach().cpu().numpy()
        else:
            activation_np = activation
        
        # Compute compression metrics
        compression_ratio = self._compute_compression_ratio(activation_np)
        entropy = self._compute_entropy(activation_np)
        
        # Compare against baseline
        baseline_comparison = self._compare_to_baseline(layer_name, compression_ratio)
        
        # Compute anomaly score
        anomaly_score = self._compute_anomaly_score(
            compression_ratio, entropy, baseline_comparison
        )
        
        # Determine if anomalous
        is_anomalous = anomaly_score > self.config.anomaly_threshold
        confidence = min(abs(anomaly_score - self.config.anomaly_threshold) + 0.5, 1.0)
        
        return CompressionMetrics(
            layer_name=layer_name,
            compression_ratio=compression_ratio,
            entropy=entropy,
            anomaly_score=anomaly_score,
            baseline_comparison=baseline_comparison,
            is_anomalous=is_anomalous,
            confidence=confidence
        )
    
    def prioritize_layers(self, metrics: List[CompressionMetrics]) -> List[str]:
        """Return layer names prioritized by anomaly score."""
        # Sort by anomaly score (descending) and confidence
        sorted_metrics = sorted(
            metrics, 
            key=lambda m: (m.anomaly_score * m.confidence), 
            reverse=True
        )
        
        prioritized = [m.layer_name for m in sorted_metrics if m.is_anomalous]
        
        logger.info(f"Prioritized {len(prioritized)} anomalous layers from {len(metrics)} total")
        return prioritized
    
    def build_baseline(self, benign_bundles: List[IncidentBundle]) -> None:
        """Build compression baseline from benign cases."""
        logger.info(f"Building baseline from {len(benign_bundles)} benign bundles")
        
        layer_stats = defaultdict(list)
        
        for bundle in benign_bundles:
            try:
                activations = self._reconstruct_activations(bundle)
                for layer_name, activation in activations.items():
                    activation_np = activation.detach().cpu().numpy()
                    compression_ratio = self._compute_compression_ratio(activation_np)
                    layer_stats[layer_name].append(compression_ratio)
                    
            except Exception as e:
                logger.warning(f"Failed to process benign bundle {bundle.bundle_id}: {e}")
        
        # Compute statistics for each layer
        self.baseline_stats = {}
        for layer_name, ratios in layer_stats.items():
            if len(ratios) > 0:
                self.baseline_stats[layer_name] = {
                    "mean": np.mean(ratios),
                    "std": np.std(ratios),
                    "median": np.median(ratios),
                    "count": len(ratios)
                }
        
        logger.info(f"Built baseline for {len(self.baseline_stats)} layers")
    
    def save_baseline(self, path: Path) -> None:
        """Save baseline statistics to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.baseline_stats, f)
        logger.info(f"Saved baseline to: {path}")
    
    def load_baseline(self, path: Path) -> None:
        """Load baseline statistics from disk."""
        with open(path, 'rb') as f:
            self.baseline_stats = pickle.load(f)
        logger.info(f"Loaded baseline from: {path} ({len(self.baseline_stats)} layers)")
    
    def _reconstruct_activations(self, bundle: IncidentBundle) -> ActivationDict:
        """Reconstruct activations from bundle."""
        from replayer.reconstruction import ActivationReconstructor
        
        reconstructor = ActivationReconstructor()
        return reconstructor.reconstruct_bundle_activations(bundle)
    
    def _compute_compression_ratio(self, data: np.ndarray) -> float:
        """Compute compression ratio using multiple methods."""
        # Flatten and convert to bytes
        data_bytes = data.astype(np.float32).tobytes()
        original_size = len(data_bytes)
        
        if original_size == 0:
            return 1.0
        
        compressed_sizes = []
        
        # Method 1: zlib compression
        if "zlib" in self.config.compression_methods:
            try:
                compressed = zlib.compress(data_bytes)
                compressed_sizes.append(len(compressed))
            except Exception:
                pass
        
        # Method 2: gzip compression
        if "gzip" in self.config.compression_methods:
            try:
                compressed = gzip.compress(data_bytes)
                compressed_sizes.append(len(compressed))
            except Exception:
                pass
        
        # Method 3: Arithmetic coding approximation (entropy-based)
        if "arithmetic" in self.config.compression_methods:
            try:
                entropy = self._compute_entropy(data)
                # Approximate arithmetic coding size
                theoretical_size = entropy * data.size * 4  # 4 bytes per float32
                compressed_sizes.append(theoretical_size)
            except Exception:
                pass
        
        if not compressed_sizes:
            logger.warning("No compression methods succeeded, using default ratio")
            return 1.0
        
        # Use the best (smallest) compression
        best_compressed_size = min(compressed_sizes)
        ratio = original_size / best_compressed_size
        
        return max(ratio, 1.0)  # Ensure ratio >= 1
    
    def _compute_entropy(self, data: np.ndarray) -> float:
        """Compute Shannon entropy of the data."""
        # Quantize data for entropy calculation
        data_flat = data.flatten()
        
        # Use histogram to estimate distribution
        try:
            hist, _ = np.histogram(data_flat, bins=min(self.config.entropy_window, len(data_flat)))
            hist = hist[hist > 0]  # Remove zero counts
            
            if len(hist) == 0:
                return 0.0
            
            # Normalize to get probabilities
            probs = hist / hist.sum()
            
            # Compute entropy
            entropy = -np.sum(probs * np.log2(probs))
            return entropy
            
        except Exception as e:
            logger.warning(f"Entropy computation failed: {e}")
            return 0.0
    
    def _compare_to_baseline(self, layer_name: str, compression_ratio: float) -> float:
        """Compare compression ratio to baseline statistics."""
        if layer_name not in self.baseline_stats:
            logger.debug(f"No baseline for layer {layer_name}")
            return 0.0
        
        stats = self.baseline_stats[layer_name]
        mean = stats["mean"]
        std = stats["std"]
        
        if std == 0:
            return 0.0 if compression_ratio == mean else 1.0
        
        # Z-score from baseline
        z_score = (compression_ratio - mean) / std
        return z_score
    
    def _compute_anomaly_score(self, compression_ratio: float, entropy: float, 
                              baseline_comparison: float) -> float:
        """Compute overall anomaly score."""
        # Combine different signals
        # Higher compression ratio = more structured/predictable = potentially anomalous
        # Lower entropy = less random = potentially anomalous  
        # High baseline deviation = anomalous
        
        # Normalize compression ratio (typical range 1-10)
        compression_score = min(compression_ratio / 10.0, 1.0)
        
        # Normalize entropy (typical range 0-8 for quantized data)
        entropy_score = 1.0 - min(entropy / 8.0, 1.0)  # Lower entropy = higher score
        
        # Baseline comparison is already a z-score
        baseline_score = min(abs(baseline_comparison) / 3.0, 1.0)  # Cap at 3 sigma
        
        # Weighted combination
        anomaly_score = (
            0.4 * compression_score +
            0.3 * entropy_score + 
            0.3 * baseline_score
        )
        
        return anomaly_score
    
    def _validate_config(self) -> None:
        """Validate CF configuration."""
        if not hasattr(self.config, 'anomaly_threshold'):
            raise ValueError("Config missing anomaly_threshold")
        if not (0.0 <= self.config.anomaly_threshold <= 1.0):
            raise ValueError("anomaly_threshold must be between 0.0 and 1.0")
        if not hasattr(self.config, 'compression_methods'):
            raise ValueError("Config missing compression_methods")
        if not hasattr(self.config, 'baseline_samples'):
            raise ValueError("Config missing baseline_samples")
        if self.config.baseline_samples < 1:
            raise ValueError("baseline_samples must be >= 1")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compression forensics statistics."""
        return {
            "baseline_layers": len(self.baseline_stats),
            "cache_size": len(self.compression_cache),
            "config": {
                "threshold": self.config.anomaly_threshold,
                "methods": self.config.compression_methods,
                "baseline_samples": self.config.baseline_samples
            }
        }