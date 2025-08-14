"""Unit tests for the Compression Forensics (CF) module."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

from src.modules.cf import CompressionForensics
from src.core.types import IncidentBundle, CompressionMetrics
from src.core.config import CFConfig


class TestCompressionForensics:
    """Unit tests for CF module."""
    
    def test_cf_initialization(self):
        """Test CF initialization."""
        config = CFConfig()
        cf = CompressionForensics(config)
        
        assert cf.config == config
        assert isinstance(cf.baseline_stats, dict)
        assert isinstance(cf.compression_cache, dict)
    
    def test_cf_initialization_validation(self):
        """Test CF initialization with invalid config."""
        # Test with invalid threshold
        config = CFConfig()
        config.anomaly_threshold = 1.5  # Invalid: > 1.0
        
        with pytest.raises(ValueError, match="anomaly_threshold must be between 0.0 and 1.0"):
            CompressionForensics(config)
        
        # Test with negative threshold
        config.anomaly_threshold = -0.1
        with pytest.raises(ValueError, match="anomaly_threshold must be between 0.0 and 1.0"):
            CompressionForensics(config)
    
    def test_compute_compression_ratio(self):
        """Test compression ratio computation."""
        config = CFConfig()
        cf = CompressionForensics(config)
        
        # Test with random data (should have lower compression ratio)
        random_data = np.random.randn(100, 50).astype(np.float32)
        random_ratio = cf._compute_compression_ratio(random_data)
        
        # Test with structured data (should have higher compression ratio)
        structured_data = np.ones((100, 50), dtype=np.float32) * 0.5
        structured_ratio = cf._compute_compression_ratio(structured_data)
        
        assert structured_ratio > random_ratio
        assert 0 < random_ratio < 10  # Reasonable range
        assert 0 < structured_ratio < 100  # Reasonable range
    
    def test_compute_entropy(self):
        """Test entropy computation."""
        config = CFConfig()
        cf = CompressionForensics(config)
        
        # Test with uniform data (should have low entropy)
        uniform_data = np.ones((100, 50), dtype=np.float32)
        uniform_entropy = cf._compute_entropy(uniform_data)
        
        # Test with random data (should have higher entropy)
        random_data = np.random.randn(100, 50).astype(np.float32)
        random_entropy = cf._compute_entropy(random_data)
        
        assert random_entropy > uniform_entropy
        assert uniform_entropy >= 0
        assert random_entropy > 0
    
    def test_analyze_layer(self):
        """Test single layer analysis."""
        config = CFConfig()
        cf = CompressionForensics(config)
        
        # Test with torch tensor
        activation = torch.randn(10, 768)
        metrics = cf.analyze_layer(activation, "test_layer")
        
        assert isinstance(metrics, CompressionMetrics)
        assert metrics.layer_name == "test_layer"
        assert metrics.compression_ratio > 0
        assert metrics.entropy > 0
        assert 0 <= metrics.anomaly_score <= 1
        assert 0 <= metrics.confidence <= 1
        
        # Test with numpy array
        activation_np = np.random.randn(10, 768).astype(np.float32)
        metrics_np = cf.analyze_layer(activation_np, "test_layer_np")
        
        assert isinstance(metrics_np, CompressionMetrics)
        assert metrics_np.layer_name == "test_layer_np"
    
    def test_analyze_layer_validation(self):
        """Test analyze_layer input validation."""
        config = CFConfig()
        cf = CompressionForensics(config)
        
        # Test None activation
        with pytest.raises(ValueError, match="Activation cannot be None"):
            cf.analyze_layer(None, "test_layer")
        
        # Test empty layer name
        with pytest.raises(ValueError, match="Layer name cannot be empty"):
            cf.analyze_layer(torch.randn(10, 768), "")
        
        # Test empty tensor
        with pytest.raises(ValueError, match="Activation tensor is empty"):
            cf.analyze_layer(torch.empty(0), "test_layer")
        
        # Test unsupported type
        with pytest.raises(ValueError, match="Unsupported activation type"):
            cf.analyze_layer([1, 2, 3], "test_layer")
    
    def test_analyze_layer_with_nan_values(self):
        """Test analyze_layer with NaN and infinite values."""
        config = CFConfig()
        cf = CompressionForensics(config)
        
        # Create tensor with NaN and infinite values
        activation = torch.randn(10, 768)
        activation[0, 0] = float('nan')
        activation[1, 1] = float('inf')
        activation[2, 2] = float('-inf')
        
        # Should not raise exception and handle gracefully
        metrics = cf.analyze_layer(activation, "test_layer_nan")
        
        assert isinstance(metrics, CompressionMetrics)
        assert np.isfinite(metrics.compression_ratio)
        assert np.isfinite(metrics.entropy)
        assert np.isfinite(metrics.anomaly_score)
    
    def test_analyze_bundle_validation(self):
        """Test analyze_bundle input validation."""
        config = CFConfig()
        cf = CompressionForensics(config)
        
        # Test None bundle
        with pytest.raises(ValueError, match="Bundle cannot be None"):
            cf.analyze_bundle(None)
        
        # Test bundle without bundle_id
        mock_bundle = Mock()
        delattr(mock_bundle, 'bundle_id') if hasattr(mock_bundle, 'bundle_id') else None
        with pytest.raises(ValueError, match="Invalid bundle: missing bundle_id"):
            cf.analyze_bundle(mock_bundle)
    
    @patch.object(CompressionForensics, '_reconstruct_activations')
    def test_analyze_bundle_success(self, mock_reconstruct, cf_module):
        """Test successful bundle analysis."""
        # Mock activations
        mock_activations = {
            "layer1": torch.randn(10, 768),
            "layer2": torch.randn(10, 768),
            "layer3": torch.randn(10, 768)
        }
        mock_reconstruct.return_value = mock_activations
        
        # Mock bundle
        mock_bundle = Mock()
        mock_bundle.bundle_id = "test_bundle_001"
        
        # Analyze bundle
        metrics = cf_module.analyze_bundle(mock_bundle)
        
        assert len(metrics) == 3
        assert all(isinstance(m, CompressionMetrics) for m in metrics)
        assert {m.layer_name for m in metrics} == {"layer1", "layer2", "layer3"}
    
    @patch.object(CompressionForensics, '_reconstruct_activations')
    def test_analyze_bundle_with_errors(self, mock_reconstruct, cf_module):
        """Test bundle analysis with some layer errors."""
        # Mock activations where one is None
        mock_activations = {
            "layer1": torch.randn(10, 768),
            "layer2": None,  # This should be skipped
            "layer3": torch.randn(10, 768)
        }
        mock_reconstruct.return_value = mock_activations
        
        # Mock bundle
        mock_bundle = Mock()
        mock_bundle.bundle_id = "test_bundle_001"
        
        # Analyze bundle
        metrics = cf_module.analyze_bundle(mock_bundle)
        
        # Should have 2 metrics (layer2 skipped due to None)
        assert len(metrics) == 2
        assert {m.layer_name for m in metrics} == {"layer1", "layer3"}
    
    def test_prioritize_layers(self, cf_module):
        """Test layer prioritization."""
        # Create test metrics
        metrics = [
            CompressionMetrics(
                layer_name="layer1",
                compression_ratio=1.2,
                entropy=3.5,
                anomaly_score=0.8,
                baseline_comparison=0.3,
                is_anomalous=True,
                confidence=0.9
            ),
            CompressionMetrics(
                layer_name="layer2",
                compression_ratio=1.1,
                entropy=3.2,
                anomaly_score=0.4,
                baseline_comparison=0.1,
                is_anomalous=False,
                confidence=0.6
            ),
            CompressionMetrics(
                layer_name="layer3",
                compression_ratio=1.5,
                entropy=4.0,
                anomaly_score=0.9,
                baseline_comparison=0.5,
                is_anomalous=True,
                confidence=0.95
            )
        ]
        
        priority_layers = cf_module.prioritize_layers(metrics)
        
        # Should be sorted by anomaly_score * confidence
        assert priority_layers[0] == "layer3"  # 0.9 * 0.95 = 0.855
        assert priority_layers[1] == "layer1"  # 0.8 * 0.9 = 0.72
        assert priority_layers[2] == "layer2"  # 0.4 * 0.6 = 0.24
    
    def test_build_baseline(self, cf_module):
        """Test baseline building."""
        # Mock bundles
        mock_bundles = []
        for i in range(3):
            bundle = Mock()
            bundle.bundle_id = f"baseline_{i}"
            mock_bundles.append(bundle)
        
        # Mock the _reconstruct_activations method
        def mock_reconstruct(bundle):
            return {
                "layer1": torch.randn(10, 768),
                "layer2": torch.randn(10, 768)
            }
        
        with patch.object(cf_module, '_reconstruct_activations', side_effect=mock_reconstruct):
            cf_module.build_baseline(mock_bundles)
        
        # Check that baseline stats were built
        assert len(cf_module.baseline_stats) == 2
        assert "layer1" in cf_module.baseline_stats
        assert "layer2" in cf_module.baseline_stats
        
        for layer_stats in cf_module.baseline_stats.values():
            assert "mean" in layer_stats
            assert "std" in layer_stats
            assert isinstance(layer_stats["mean"], float)
            assert isinstance(layer_stats["std"], float)
    
    def test_compare_to_baseline(self, cf_module):
        """Test baseline comparison."""
        # Set up baseline
        cf_module.baseline_stats = {
            "layer1": {"mean": 1.0, "std": 0.2}
        }
        
        # Test comparison
        comparison = cf_module._compare_to_baseline("layer1", 1.5)  # 2.5 std above mean
        assert comparison > 0
        
        comparison = cf_module._compare_to_baseline("layer1", 0.8)  # 1 std below mean  
        assert comparison < 0
        
        comparison = cf_module._compare_to_baseline("layer1", 1.0)  # At mean
        assert comparison == 0
        
        # Test with unknown layer
        comparison = cf_module._compare_to_baseline("unknown_layer", 1.5)
        assert comparison == 0
    
    def test_compute_anomaly_score(self, cf_module):
        """Test anomaly score computation."""
        # Test different combinations
        score1 = cf_module._compute_anomaly_score(2.0, 5.0, 1.0)  # High compression, high entropy, positive baseline
        score2 = cf_module._compute_anomaly_score(1.0, 3.0, 0.0)  # Low compression, low entropy, neutral baseline
        score3 = cf_module._compute_anomaly_score(1.5, 4.0, -0.5)  # Medium compression, medium entropy, negative baseline
        
        # Scores should be in [0, 1] range
        assert 0 <= score1 <= 1
        assert 0 <= score2 <= 1
        assert 0 <= score3 <= 1
        
        # Higher anomaly indicators should generally produce higher scores
        assert score1 > score2
    
    def test_get_stats(self, cf_module):
        """Test statistics reporting."""
        stats = cf_module.get_stats()
        
        assert "baseline_layers" in stats
        assert "cache_size" in stats
        assert "config" in stats
        
        assert isinstance(stats["baseline_layers"], int)
        assert isinstance(stats["cache_size"], int)
        assert isinstance(stats["config"], dict)
        
        config_stats = stats["config"]
        assert "threshold" in config_stats
        assert "methods" in config_stats
        assert "baseline_samples" in config_stats


@pytest.mark.integration
class TestCompressionForensicsIntegration:
    """Integration tests for CF module."""
    
    def test_cf_with_real_data(self, sample_activation):
        """Test CF with real activation data."""
        config = CFConfig()
        cf = CompressionForensics(config)
        
        metrics = cf.analyze_layer(sample_activation, "test_layer")
        
        assert isinstance(metrics, CompressionMetrics)
        assert metrics.compression_ratio > 0
        assert metrics.entropy > 0
        assert 0 <= metrics.anomaly_score <= 1