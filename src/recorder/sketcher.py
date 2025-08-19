import logging
import zlib
import pickle
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import hashlib
from pathlib import Path
import threading
import queue
import asyncio

from core.types import CompressedSketch
from core.config import RecorderConfig


logger = logging.getLogger(__name__)


class KVSketcher:
    """Compresses and samples activation data for efficient storage."""
    
    def __init__(self, config: RecorderConfig):
        self.config = config
        self.compression_stats: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"Initialized KVSketcher with compression ratio: {config.compression_ratio}")
    
    def compress_activation(self, activation: torch.Tensor, layer_name: str) -> CompressedSketch:
        """Compress activation tensor using configured compression method."""
        original_size = activation.numel() * activation.element_size()
        
        # Convert to numpy for compression
        if activation.requires_grad:
            activation = activation.detach()
        if activation.is_cuda:
            activation = activation.cpu()
        
        activation_np = activation.numpy()
        
        # Apply differential privacy noise if enabled
        if self.config.enable_dp_noise:
            activation_np = self._add_dp_noise(activation_np, layer_name)
        
        # Choose compression method
        if self.config.compression_method == "topk":
            compressed_data = self._topk_compression(activation_np)
            compression_method = "topk"
        elif self.config.compression_method == "random_projection":
            compressed_data = self._random_projection_compression(activation_np, layer_name)
            compression_method = "random_projection"
        elif self.config.compression_method == "hybrid":
            compressed_data = self._hybrid_compression(activation_np, layer_name)
            compression_method = "hybrid"
        else:
            compressed_data = activation_np
            compression_method = "none"
        
        # Serialize and compress with validation
        try:
            serialized = pickle.dumps(compressed_data)
            compressed = zlib.compress(serialized, level=6)
            
            # Validate compression by attempting decompression
            test_decompressed = zlib.decompress(compressed)
            test_data = pickle.loads(test_decompressed)
            
        except Exception as e:
            logger.error(f"Compression validation failed for {layer_name}: {e}")
            # Fallback to uncompressed storage
            logger.warning(f"Using uncompressed storage for {layer_name} due to compression failure")
            serialized = pickle.dumps(compressed_data)
            compressed = serialized  # Store without compression
        
        # Calculate compression stats
        compressed_size = len(compressed)
        actual_ratio = compressed_size / original_size
        
        # Store stats for analysis
        self.compression_stats[layer_name] = {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": actual_ratio,
            "target_ratio": self.config.compression_ratio,
            "compression_method": compression_method
        }
        
        sketch = CompressedSketch(
            data=compressed,
            original_shape=tuple(activation.shape),
            dtype=str(activation.dtype),
            compression_ratio=actual_ratio,
            metadata={
                "layer_name": layer_name,
                "compression_method": compression_method,
                "dp_noise_applied": self.config.enable_dp_noise,
                "original_size_bytes": original_size,
                "compressed_size_bytes": compressed_size
            }
        )
        
        logger.debug(f"Compressed {layer_name} using {compression_method}: {original_size} -> {compressed_size} bytes "
                    f"(ratio: {actual_ratio:.3f})")
        
        return sketch
    
    def decompress_sketch(self, sketch: CompressedSketch) -> torch.Tensor:
        """Decompress sketch back to tensor."""
        try:
            # Decompress and deserialize
            decompressed = zlib.decompress(sketch.data)
            activation_data = pickle.loads(decompressed)
            
            # Handle different data types
            if isinstance(activation_data, dict):
                # This is sampled data
                tensor = self._restore_shape(activation_data, sketch.original_shape)
            else:
                # This is direct numpy array
                tensor = torch.from_numpy(activation_data)
                
                # Restore original shape if needed
                if tensor.shape != sketch.original_shape:
                    tensor = tensor.reshape(sketch.original_shape)
            
            logger.debug(f"Decompressed sketch to shape: {tensor.shape}")
            return tensor
            
        except Exception as e:
            logger.error(f"Failed to decompress sketch: {e}")
            
            # Try to provide a fallback tensor instead of crashing
            fallback_shape = sketch.original_shape if hasattr(sketch, 'original_shape') else (1,)
            logger.warning(f"Using zero fallback tensor with shape {fallback_shape} due to decompression failure")
            return torch.zeros(fallback_shape, dtype=torch.float32)
    
    def _sample_activation(self, activation: np.ndarray, ratio: float) -> np.ndarray:
        """Sample activation tensor to reduce size."""
        if ratio >= 1.0:
            return activation
        
        total_elements = activation.size
        target_elements = int(total_elements * ratio)
        
        if target_elements == 0:
            logger.warning("Compression ratio too aggressive - keeping at least 1 element")
            target_elements = 1
        
        # Flatten for sampling
        flat = activation.flatten()
        
        # Use top-k sampling based on magnitude for better preservation
        indices = np.argpartition(np.abs(flat), -target_elements)[-target_elements:]
        indices = np.sort(indices)  # Keep order for better compression
        
        sampled_values = flat[indices]
        
        # Store indices and values for reconstruction
        sampled_data = {
            "values": sampled_values,
            "indices": indices,
            "original_shape": activation.shape,
            "sampling_ratio": ratio
        }
        
        return sampled_data
    
    def _restore_shape(self, tensor: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Restore tensor to target shape after sampling."""
        if isinstance(tensor, dict):
            # Handle sampled data
            sampled_data = tensor
            values = torch.from_numpy(sampled_data["values"])
            indices = sampled_data["indices"]
            original_shape = sampled_data["original_shape"]
            
            # Create sparse representation
            full_tensor = torch.zeros(np.prod(original_shape))
            full_tensor[indices] = values
            
            # Reshape to original shape
            return full_tensor.reshape(target_shape)
        else:
            # Simple reshape
            return tensor.reshape(target_shape)
    
    def get_compression_stats(self) -> Dict[str, Dict[str, float]]:
        """Get compression statistics for all layers."""
        return self.compression_stats.copy()
    
    def estimate_bundle_size(self, activations: Dict[str, CompressedSketch]) -> float:
        """Estimate total bundle size in MB."""
        total_size = 0
        for sketch in activations.values():
            total_size += len(sketch.data)
            # Add metadata overhead
            total_size += len(pickle.dumps(sketch.metadata))
        
        # Convert to MB
        return total_size / (1024 * 1024)
    
    def should_apply_compression(self, layer_name: str, activation_size: int) -> bool:
        """Determine if compression should be applied based on config and size."""
        # Always compress if bundle size limit would be exceeded
        if hasattr(self, '_current_bundle_size'):
            estimated_new_size = self._current_bundle_size + (activation_size / (1024 * 1024))
            if estimated_new_size > self.config.max_bundle_size_mb:
                return True
        
        # Apply compression based on configuration
        return self.config.compression_ratio < 1.0
    
    def _topk_compression(self, activation: np.ndarray) -> Dict[str, Any]:
        """Apply top-k compression (existing method)."""
        return self._sample_activation(activation, self.config.compression_ratio)
    
    def _random_projection_compression(self, activation: np.ndarray, layer_name: str) -> Dict[str, Any]:
        """Apply random projection compression."""
        original_shape = activation.shape
        flattened = activation.flatten()
        
        # Generate or load projection matrix
        projection_matrix = self._get_projection_matrix(len(flattened), layer_name)
        
        # Apply projection
        projected = projection_matrix @ flattened
        
        return {
            "type": "random_projection",
            "projected_values": projected,
            "original_shape": original_shape,
            "projection_dim": self.config.random_projection_dim,
            "projection_matrix_hash": self._get_matrix_hash(layer_name)
        }
    
    def _hybrid_compression(self, activation: np.ndarray, layer_name: str) -> Dict[str, Any]:
        """Apply hybrid compression (top-k + random projection)."""
        # First apply top-k to get most important elements
        topk_data = self._topk_compression(activation)
        
        # Then apply random projection to the remainder
        flattened = activation.flatten()
        important_indices = topk_data["indices"]
        
        # Create mask for non-important elements
        mask = np.ones(len(flattened), dtype=bool)
        mask[important_indices] = False
        remainder = flattened[mask]
        
        if len(remainder) > self.config.random_projection_dim:
            projection_matrix = self._get_projection_matrix(len(remainder), f"{layer_name}_remainder")
            projected_remainder = projection_matrix @ remainder
        else:
            projected_remainder = remainder
        
        return {
            "type": "hybrid",
            "topk_data": topk_data,
            "projected_remainder": projected_remainder,
            "remainder_indices": np.where(mask)[0],
            "original_shape": activation.shape
        }
    
    def _add_dp_noise(self, activation: np.ndarray, layer_name: str) -> np.ndarray:
        """Add differential privacy noise to activations."""
        if not self.config.enable_dp_noise:
            return activation
        
        # Calculate noise scale based on DP parameters
        noise_scale = (self.config.dp_sensitivity * np.sqrt(2 * np.log(1.25 / self.config.dp_delta))) / self.config.dp_epsilon
        
        # Generate Laplacian noise
        noise = np.random.laplace(0, noise_scale, activation.shape)
        
        # Add noise
        noisy_activation = activation + noise
        
        logger.debug(f"Added DP noise to {layer_name}: scale={noise_scale:.4f}, epsilon={self.config.dp_epsilon}")
        
        return noisy_activation
    
    def _get_projection_matrix(self, input_dim: int, layer_name: str) -> np.ndarray:
        """Get or generate random projection matrix for a layer."""
        # Create a deterministic seed based on layer name
        seed = int(hashlib.md5(layer_name.encode()).hexdigest()[:8], 16)
        np.random.seed(seed)
        
        target_dim = min(self.config.random_projection_dim, input_dim)
        
        # Generate Gaussian random matrix (Johnson-Lindenstrauss)
        matrix = np.random.normal(0, 1/np.sqrt(target_dim), (target_dim, input_dim))
        
        return matrix
    
    def _get_matrix_hash(self, layer_name: str) -> str:
        """Get hash for projection matrix identification."""
        return hashlib.md5(f"{layer_name}_{self.config.random_projection_dim}".encode()).hexdigest()[:16]

    def compress_external_call(self, call_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress external call data if it's large."""
        # Serialize to check size
        serialized = pickle.dumps(call_data)
        size_mb = len(serialized) / (1024 * 1024)
        
        # Compress if larger than 1MB
        if size_mb > 1.0:
            compressed = zlib.compress(serialized)
            return {
                "compressed": True,
                "data": compressed,
                "original_size": len(serialized),
                "compressed_size": len(compressed)
            }
        
        return call_data
    
    def decompress_external_call(self, call_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress external call data if it was compressed."""
        if isinstance(call_data, dict) and call_data.get("compressed"):
            decompressed = zlib.decompress(call_data["data"])
            return pickle.loads(decompressed)
        
        return call_data


class AdaptiveKVSketcher(KVSketcher):
    """Adaptive sketcher that adjusts compression based on layer importance."""
    
    def __init__(self, config: RecorderConfig):
        super().__init__(config)
        self.layer_importance: Dict[str, float] = {}
        self.importance_threshold = 0.5
    
    def set_layer_importance(self, importance_scores: Dict[str, float]) -> None:
        """Set importance scores for layers (from prior analysis)."""
        self.layer_importance = importance_scores.copy()
        logger.info(f"Updated importance scores for {len(importance_scores)} layers")
    
    def compress_activation(self, activation: torch.Tensor, layer_name: str) -> CompressedSketch:
        """Compress with adaptive ratio based on layer importance."""
        # Adjust compression ratio based on importance
        base_ratio = self.config.compression_ratio
        importance = self.layer_importance.get(layer_name, 0.5)
        
        # Less compression for more important layers
        if importance > self.importance_threshold:
            adjusted_ratio = min(1.0, base_ratio * (1.0 + importance))
        else:
            adjusted_ratio = base_ratio * importance
        
        # Temporarily adjust config
        original_ratio = self.config.compression_ratio
        self.config.compression_ratio = adjusted_ratio
        
        try:
            sketch = super().compress_activation(activation, layer_name)
            # Add importance to metadata
            sketch.metadata["layer_importance"] = importance
            sketch.metadata["adjusted_ratio"] = adjusted_ratio
            return sketch
        finally:
            # Restore original ratio
            self.config.compression_ratio = original_ratio
    


class BenignDonorPool:
    """Manages a pool of benign activations for patching."""
    
    def __init__(self, config: RecorderConfig):
        self.config = config
        self.donor_pool: Dict[str, List[np.ndarray]] = {}
        self.max_donors_per_layer = config.benign_donor_pool_size
        
    def add_benign_activation(self, layer_name: str, activation: np.ndarray) -> None:
        """Add benign activation to donor pool."""
        if not self.config.enable_benign_donors:
            return
            
        if layer_name not in self.donor_pool:
            self.donor_pool[layer_name] = []
        
        # Add to pool (FIFO if full)
        if len(self.donor_pool[layer_name]) >= self.max_donors_per_layer:
            self.donor_pool[layer_name].pop(0)
        
        self.donor_pool[layer_name].append(activation.copy())
        logger.debug(f"Added benign donor for {layer_name}, pool size: {len(self.donor_pool[layer_name])}")
    
    def get_similar_donor(self, layer_name: str, target_activation: np.ndarray) -> Optional[np.ndarray]:
        """Find most similar benign donor for patching."""
        if layer_name not in self.donor_pool or not self.donor_pool[layer_name]:
            return None
        
        best_donor = None
        best_similarity = -1
        
        for donor in self.donor_pool[layer_name]:
            if donor.shape != target_activation.shape:
                continue
                
            # Calculate cosine similarity
            similarity = self._cosine_similarity(target_activation, donor)
            
            if similarity > best_similarity and similarity >= self.config.benign_donor_similarity_threshold:
                best_similarity = similarity
                best_donor = donor
        
        if best_donor is not None:
            logger.debug(f"Found benign donor for {layer_name} with similarity: {best_similarity:.3f}")
        
        return best_donor
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two arrays."""
        a_flat = a.flatten()
        b_flat = b.flatten()
        
        dot_product = np.dot(a_flat, b_flat)
        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def get_pool_stats(self) -> Dict[str, int]:
        """Get statistics about the donor pool."""
        return {layer: len(donors) for layer, donors in self.donor_pool.items()}


class AsyncBundleWriter:
    """Async writer for non-blocking bundle export."""
    
    def __init__(self, config: RecorderConfig):
        self.config = config
        self.write_queue = queue.Queue(maxsize=config.export_queue_size)
        self.worker_thread = None
        self.shutdown_event = threading.Event()
        self.stats = {"bundles_queued": 0, "bundles_written": 0, "errors": 0}
        
    def start(self) -> None:
        """Start the async writer thread."""
        if self.worker_thread is not None:
            return
            
        self.worker_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Started async bundle writer")
    
    def queue_bundle_export(self, trace_data, bundle_id: str, exporter) -> None:
        """Queue a bundle for async export."""
        try:
            self.write_queue.put((trace_data, bundle_id, exporter), timeout=1.0)
            self.stats["bundles_queued"] += 1
        except queue.Full:
            logger.warning("Bundle export queue is full, dropping bundle")
    
    def _writer_loop(self) -> None:
        """Main writer loop (runs in background thread)."""
        batch = []
        
        while not self.shutdown_event.is_set():
            try:
                # Try to get items for batch
                try:
                    item = self.write_queue.get(timeout=0.5)
                    batch.append(item)
                except queue.Empty:
                    if batch:
                        self._process_batch(batch)
                        batch = []
                    continue
                
                # Process batch when full or on timeout
                if len(batch) >= self.config.export_batch_size:
                    self._process_batch(batch)
                    batch = []
                    
            except Exception as e:
                logger.error(f"Error in async writer loop: {e}")
                self.stats["errors"] += 1
        
        # Process remaining items
        if batch:
            self._process_batch(batch)
    
    def _process_batch(self, batch: List) -> None:
        """Process a batch of bundle exports."""
        for trace_data, bundle_id, exporter in batch:
            try:
                exporter.create_bundle(trace_data, bundle_id)
                self.stats["bundles_written"] += 1
                logger.debug(f"Async exported bundle: {bundle_id}")
            except Exception as e:
                logger.error(f"Failed to export bundle {bundle_id}: {e}")
                self.stats["errors"] += 1
    
    def shutdown(self, timeout: float = 5.0) -> None:
        """Shutdown the async writer."""
        if self.worker_thread is None:
            return
            
        self.shutdown_event.set()
        self.worker_thread.join(timeout)
        logger.info(f"Async writer shutdown. Stats: {self.stats}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get writer statistics."""
        return self.stats.copy()