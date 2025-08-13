import logging
from typing import Dict, Optional, List, Any
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

from core.types import IncidentBundle, CompressedSketch, ActivationDict
from core.config import RCAConfig
from recorder.sketcher import KVSketcher

logger = logging.getLogger(__name__)


class ActivationReconstructor:
    """Reconstructs full activations from compressed sketches."""
    
    def __init__(self, config: Optional[RCAConfig] = None):
        self.config = config or RCAConfig()
        self.sketcher = KVSketcher(self.config.recorder)
        
    def reconstruct_bundle_activations(self, bundle: IncidentBundle) -> ActivationDict:
        """Reconstruct all activations from a bundle."""
        logger.info(f"Reconstructing activations from bundle: {bundle.bundle_id}")
        
        reconstructed = {}
        stats = {"total": 0, "successful": 0, "failed": 0}
        
        for layer_name, sketch in bundle.trace_data.activations.items():
            try:
                activation = self.reconstruct_activation(sketch)
                reconstructed[layer_name] = activation
                stats["successful"] += 1
                logger.debug(f"Reconstructed {layer_name}: {activation.shape}")
                
            except Exception as e:
                logger.warning(f"Failed to reconstruct {layer_name}: {e}")
                stats["failed"] += 1
            
            stats["total"] += 1
        
        success_rate = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
        logger.info(f"Reconstruction complete: {stats['successful']}/{stats['total']} layers "
                   f"({success_rate:.1%} success rate)")
        
        return reconstructed
    
    def reconstruct_activation(self, sketch: CompressedSketch) -> torch.Tensor:
        """Reconstruct single activation from compressed sketch."""
        try:
            # Use the sketcher's decompression logic
            tensor = self.sketcher.decompress_sketch(sketch)
            
            # Validate reconstruction
            if tensor.shape != sketch.original_shape:
                logger.warning(f"Shape mismatch: expected {sketch.original_shape}, got {tensor.shape}")
                tensor = tensor.reshape(sketch.original_shape)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Reconstruction failed: {e}")
            raise
    
    def partial_reconstruct(self, sketch: CompressedSketch, indices: List[int]) -> torch.Tensor:
        """Reconstruct only specific indices from a compressed sketch."""
        # For sparse reconstructions, we still need the full tensor first
        full_tensor = self.reconstruct_activation(sketch)
        
        # Handle different indexing schemes
        if len(indices) == 0:
            return torch.empty(0, dtype=full_tensor.dtype)
        
        # Flatten and index
        flat_tensor = full_tensor.flatten()
        if max(indices) >= len(flat_tensor):
            raise ValueError(f"Index {max(indices)} out of bounds for tensor of size {len(flat_tensor)}")
        
        return flat_tensor[indices].reshape(-1)
    
    def estimate_reconstruction_quality(self, sketch: CompressedSketch) -> Dict[str, float]:
        """Estimate quality metrics for reconstruction."""
        compression_method = sketch.metadata.get("compression_method", "unknown")
        compression_ratio = sketch.compression_ratio
        
        # Quality estimates based on compression method and ratio
        quality_metrics = {
            "compression_ratio": compression_ratio,
            "information_loss": self._estimate_information_loss(compression_method, compression_ratio),
            "fidelity_score": self._estimate_fidelity(compression_method, compression_ratio),
            "confidence": self._estimate_confidence(sketch.metadata)
        }
        
        return quality_metrics
    
    def _estimate_information_loss(self, method: str, ratio: float) -> float:
        """Estimate information loss based on compression method and ratio."""
        if method == "topk":
            # Top-k preserves most important values
            return min(0.9, 1.0 - ratio)
        elif method == "random_projection":
            # Random projections preserve structure better
            return min(0.7, 1.0 - ratio * 0.8)
        elif method == "hybrid":
            # Hybrid combines benefits
            return min(0.8, 1.0 - ratio * 0.9)
        else:
            # Conservative estimate for unknown methods
            return min(0.95, 1.0 - ratio * 0.5)
    
    def _estimate_fidelity(self, method: str, ratio: float) -> float:
        """Estimate reconstruction fidelity score."""
        base_fidelity = {
            "topk": 0.9,
            "random_projection": 0.8,
            "hybrid": 0.85,
            "none": 1.0
        }.get(method, 0.7)
        
        # Adjust for compression ratio
        ratio_penalty = (1.0 - ratio) * 0.3
        return max(0.1, base_fidelity - ratio_penalty)
    
    def _estimate_confidence(self, metadata: Dict[str, Any]) -> float:
        """Estimate confidence in reconstruction quality."""
        confidence = 0.8  # Base confidence
        
        # Adjust based on metadata
        if metadata.get("dp_noise_applied", False):
            confidence *= 0.9  # DP adds uncertainty
        
        if "original_size_bytes" in metadata and "compressed_size_bytes" in metadata:
            actual_ratio = metadata["compressed_size_bytes"] / metadata["original_size_bytes"]
            if actual_ratio < 0.1:  # Very high compression
                confidence *= 0.8
            elif actual_ratio > 0.5:  # Low compression
                confidence *= 1.1
        
        return min(1.0, confidence)
    
    def validate_reconstruction(self, original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
        """Validate reconstruction quality against original (if available)."""
        if original.shape != reconstructed.shape:
            logger.warning(f"Shape mismatch in validation: {original.shape} vs {reconstructed.shape}")
            return {"error": 1.0}
        
        # Calculate various similarity metrics
        with torch.no_grad():
            mse = torch.nn.functional.mse_loss(original, reconstructed).item()
            mae = torch.nn.functional.l1_loss(original, reconstructed).item()
            
            # Cosine similarity
            orig_flat = original.flatten()
            recon_flat = reconstructed.flatten()
            cosine_sim = torch.nn.functional.cosine_similarity(
                orig_flat.unsqueeze(0), 
                recon_flat.unsqueeze(0)
            ).item()
            
            # Relative error
            rel_error = (torch.norm(original - reconstructed) / torch.norm(original)).item()
        
        return {
            "mse": mse,
            "mae": mae,
            "cosine_similarity": cosine_sim,
            "relative_error": rel_error,
            "fidelity_score": max(0.0, cosine_sim)
        }
    
    def get_reconstruction_stats(self) -> Dict[str, Any]:
        """Get reconstruction statistics."""
        return {
            "sketcher_stats": self.sketcher.get_compression_stats(),
            "config": {
                "compression_method": self.config.recorder.compression_method,
                "compression_ratio": self.config.recorder.compression_ratio
            }
        }