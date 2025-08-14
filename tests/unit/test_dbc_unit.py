"""Unit tests for Decision Basin Cartography (DBC) module."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.modules.dbc import DecisionBasinCartography
from src.core.config import DBCConfig
from src.core.types import (
    ReplaySession, IncidentBundle, DecisionBoundary, RobustnessMetrics,
    DecisionBasinMap, TraceData
)


class TestDecisionBasinCartography:
    """Test suite for DBC module."""
    
    @pytest.fixture
    def dbc_config(self):
        """Create DBC configuration for testing."""
        return DBCConfig(
            enabled=True,
            perturbation_magnitude=0.1,
            perturbation_steps=5,  # Reduced for faster testing
            basin_resolution=10,
            robustness_samples=20,  # Reduced for faster testing
            boundary_detection_method="gradient"
        )
    
    @pytest.fixture
    def dbc(self, dbc_config):
        """Create DBC instance for testing."""
        return DecisionBasinCartography(dbc_config)
    
    @pytest.fixture
    def mock_session(self):
        """Create mock replay session."""
        session = Mock(spec=ReplaySession)
        session.session_id = "test_session_123"
        session.reconstructed_activations = {
            "layer_1": torch.randn(1, 10, 768),
            "layer_2": torch.randn(1, 10, 768),
            "layer_3": torch.randn(1, 10, 768)
        }
        
        # Mock bundle
        bundle = Mock(spec=IncidentBundle)
        bundle.bundle_id = "test_bundle_123"
        bundle.bundle_path = Path("/tmp/test_bundle")
        
        # Mock trace data
        trace_data = Mock(spec=TraceData)
        trace_data.inputs = {"text": "test input"}
        trace_data.outputs = {"logits": torch.randn(1, 1000)}
        
        bundle.trace_data = trace_data
        session.bundle = bundle
        
        return session
    
    @pytest.fixture
    def failure_activations(self):
        """Create mock failure activations."""
        return {
            "layer_1": torch.randn(1, 10, 768),
            "layer_2": torch.randn(1, 10, 768),
            "layer_3": torch.randn(1, 10, 768)
        }
    
    def test_initialization(self, dbc_config):
        """Test DBC initialization."""
        dbc = DecisionBasinCartography(dbc_config)
        
        assert dbc.config == dbc_config
        assert dbc.baseline_robustness is None
        assert isinstance(dbc.boundary_cache, dict)
        assert len(dbc.boundary_cache) == 0
    
    def test_initialization_default_config(self):
        """Test DBC initialization with default config."""
        dbc = DecisionBasinCartography()
        
        assert isinstance(dbc.config, DBCConfig)
        assert dbc.config.enabled is True  # Should be True for our implementation
        assert dbc.config.boundary_detection_method in ["gradient", "random", "adversarial"]
    
    def test_config_validation_invalid_method(self):
        """Test config validation with invalid boundary detection method."""
        config = DBCConfig(boundary_detection_method="invalid_method")
        
        with pytest.raises(ValueError, match="Invalid boundary_detection_method"):
            DecisionBasinCartography(config)
    
    def test_config_validation_invalid_magnitude(self):
        """Test config validation with invalid perturbation magnitude."""
        config = DBCConfig(perturbation_magnitude=1.5)  # > 1.0
        
        with pytest.raises(ValueError, match="perturbation_magnitude must be between 0 and 1"):
            DecisionBasinCartography(config)
    
    def test_map_decision_basin_empty_activations(self, dbc, mock_session):
        """Test decision basin mapping with empty activations."""
        with pytest.raises(ValueError, match="Failure activations cannot be empty"):
            dbc.map_decision_basin(mock_session, {})
    
    @patch.object(DecisionBasinCartography, '_analyze_layer_robustness')
    @patch.object(DecisionBasinCartography, '_detect_decision_boundaries')
    @patch.object(DecisionBasinCartography, '_sample_perturbations')
    def test_map_decision_basin_success(self, mock_sample, mock_boundaries, mock_robustness,
                                      dbc, mock_session, failure_activations):
        """Test successful decision basin mapping."""
        # Setup mocks
        mock_robustness.return_value = RobustnessMetrics(
            layer_name="layer_1",
            perturbation_tolerance=0.05,
            gradient_magnitude=1.0,
            lipschitz_constant=2.0,
            basin_volume=0.1,
            boundary_distance=0.2,
            confidence=0.8
        )
        
        mock_boundaries.return_value = [
            DecisionBoundary(
                layer_name="layer_1",
                boundary_points=[np.random.randn(10)],
                normal_vector=np.random.randn(10),
                distance_from_failure=0.15,
                confidence=0.9,
                boundary_type="gradient"
            )
        ]
        
        mock_sample.return_value = [np.random.randn(1, 10, 768) for _ in range(5)]
        
        # Test basin mapping
        basin_map = dbc.map_decision_basin(mock_session, failure_activations)
        
        # Verify result
        assert isinstance(basin_map, DecisionBasinMap)
        assert len(basin_map.failure_point) == len(failure_activations)
        assert len(basin_map.robustness_metrics) <= len(failure_activations)
        assert len(basin_map.decision_boundaries) >= 0
        assert isinstance(basin_map.basin_volume_estimate, float)
        assert isinstance(basin_map.reversible_region_size, float)
        assert "perturbation_magnitude" in basin_map.analysis_metadata
        
        # Verify calls
        assert mock_robustness.call_count <= len(failure_activations)
        assert mock_boundaries.call_count <= len(failure_activations)
        assert mock_sample.call_count <= len(failure_activations)
    
    @patch.object(DecisionBasinCartography, '_analyze_layer_robustness')
    def test_map_decision_basin_with_target_layers(self, mock_robustness, dbc, 
                                                 mock_session, failure_activations):
        """Test basin mapping with specific target layers."""
        mock_robustness.return_value = RobustnessMetrics(
            layer_name="layer_1",
            perturbation_tolerance=0.05,
            gradient_magnitude=1.0,
            lipschitz_constant=2.0,
            basin_volume=0.1,
            boundary_distance=0.2,
            confidence=0.8
        )
        
        target_layers = ["layer_1", "layer_2"]
        basin_map = dbc.map_decision_basin(mock_session, failure_activations, target_layers)
        
        # Should only analyze target layers
        assert mock_robustness.call_count == len(target_layers)
    
    def test_analyze_layer_robustness(self, dbc, mock_session):
        """Test layer robustness analysis."""
        activation = torch.randn(1, 10, 768)
        
        with patch.object(dbc, '_estimate_perturbation_tolerance', return_value=0.05), \
             patch.object(dbc, '_estimate_lipschitz_constant', return_value=1.5), \
             patch.object(dbc, '_find_boundary_distance', return_value=0.2):
            
            metrics = dbc._analyze_layer_robustness(mock_session, "test_layer", activation)
            
            assert isinstance(metrics, RobustnessMetrics)
            assert metrics.layer_name == "test_layer"
            assert isinstance(metrics.perturbation_tolerance, float)
            assert isinstance(metrics.gradient_magnitude, float)
            assert isinstance(metrics.lipschitz_constant, float)
            assert isinstance(metrics.basin_volume, float)
            assert isinstance(metrics.boundary_distance, float)
            assert 0.0 <= metrics.confidence <= 1.0
    
    @patch.object(DecisionBasinCartography, '_detect_gradient_boundaries')
    def test_detect_decision_boundaries_gradient(self, mock_gradient, dbc, mock_session):
        """Test gradient-based boundary detection."""
        activation = torch.randn(1, 10, 768)
        mock_boundary = DecisionBoundary(
            layer_name="test_layer",
            boundary_points=[np.random.randn(10)],
            normal_vector=np.random.randn(10),
            distance_from_failure=0.1,
            confidence=0.8,
            boundary_type="gradient"
        )
        mock_gradient.return_value = [mock_boundary]
        
        boundaries = dbc._detect_decision_boundaries(mock_session, "test_layer", activation)
        
        assert len(boundaries) == 1
        assert boundaries[0] == mock_boundary
        mock_gradient.assert_called_once()
    
    def test_perturbation_tolerance_estimation(self, dbc, mock_session):
        """Test perturbation tolerance estimation."""
        activation = torch.randn(1, 10, 768)
        
        with patch.object(dbc, '_test_perturbation_flip') as mock_test:
            # Mock binary search behavior
            mock_test.side_effect = lambda s, l, a, m: m > 0.05
            
            tolerance = dbc._estimate_perturbation_tolerance(mock_session, "test_layer", activation)
            
            assert isinstance(tolerance, float)
            assert 0.0 <= tolerance <= 1.0
    
    def test_boundary_distance_calculation(self, dbc, mock_session):
        """Test boundary distance calculation."""
        activation = torch.randn(1, 10, 768)
        
        with patch.object(dbc, '_binary_search_boundary') as mock_search:
            # Mock finding boundaries at different distances
            mock_search.side_effect = [
                np.random.randn(*activation.shape) + 0.1,  # Close boundary
                np.random.randn(*activation.shape) + 0.3,  # Farther boundary
                None,  # No boundary found
            ]
            
            distance = dbc._find_boundary_distance(mock_session, "test_layer", activation)
            
            assert isinstance(distance, float)
            assert distance > 0.0
    
    def test_build_baseline_empty_sessions(self, dbc):
        """Test baseline building with empty sessions."""
        with pytest.raises(ValueError, match="Baseline sessions cannot be empty"):
            dbc.build_baseline([])
    
    @patch.object(DecisionBasinCartography, '_analyze_layer_robustness')
    def test_build_baseline_success(self, mock_robustness, dbc):
        """Test successful baseline building."""
        # Create mock sessions
        mock_sessions = []
        for i in range(3):
            session = Mock(spec=ReplaySession)
            session.session_id = f"baseline_session_{i}"
            session.reconstructed_activations = {
                "layer_1": torch.randn(1, 10, 768),
                "layer_2": torch.randn(1, 10, 768)
            }
            mock_sessions.append(session)
        
        # Mock robustness analysis
        mock_robustness.return_value = RobustnessMetrics(
            layer_name="layer_1",
            perturbation_tolerance=0.08,
            gradient_magnitude=1.2,
            lipschitz_constant=1.8,
            basin_volume=0.15,
            boundary_distance=0.25,
            confidence=0.7
        )
        
        dbc.build_baseline(mock_sessions)
        
        # Verify baseline was built
        assert dbc.baseline_robustness is not None
        assert len(dbc.baseline_robustness) >= 1
        assert "layer_1" in dbc.baseline_robustness or "layer_2" in dbc.baseline_robustness
    
    def test_compare_to_baseline_no_baseline(self, dbc):
        """Test baseline comparison without baseline."""
        mock_basin_map = Mock(spec=DecisionBasinMap)
        mock_basin_map.robustness_metrics = []
        mock_basin_map.critical_layers = []
        
        result = dbc.compare_to_baseline(mock_basin_map)
        
        assert "robustness_changes" in result
        assert "critical_layer_changes" in result
        assert result["robustness_changes"] == {}
        assert result["critical_layer_changes"] == []
    
    def test_find_intervention_boundaries(self, dbc):
        """Test finding intervention-relevant boundaries."""
        # Create mock basin map with boundaries
        mock_boundaries = [
            DecisionBoundary(
                layer_name="layer_1",
                boundary_points=[np.random.randn(10)],
                normal_vector=np.random.randn(10),
                distance_from_failure=0.1,
                confidence=0.9,
                boundary_type="gradient"
            ),
            DecisionBoundary(
                layer_name="layer_2",
                boundary_points=[np.random.randn(10)],
                normal_vector=np.random.randn(10),
                distance_from_failure=0.2,
                confidence=0.7,
                boundary_type="gradient"
            )
        ]
        
        mock_basin_map = Mock(spec=DecisionBasinMap)
        mock_basin_map.decision_boundaries = mock_boundaries
        
        intervention_candidates = ["layer_1", "layer_3"]  # layer_3 has no boundaries
        
        relevant_boundaries = dbc.find_intervention_boundaries(mock_basin_map, intervention_candidates)
        
        assert len(relevant_boundaries) == 1  # Only layer_1 has boundaries
        assert "layer_1" in relevant_boundaries
        assert "layer_3" not in relevant_boundaries
    
    def test_get_stats_no_baseline(self, dbc):
        """Test getting stats without baseline."""
        stats = dbc.get_stats()
        
        assert "baseline_robustness" in stats
        assert "boundary_cache_size" in stats
        assert "config" in stats
        assert stats["baseline_robustness"] == {}
        assert stats["boundary_cache_size"] == 0
        assert "boundary_detection_method" in stats["config"]
    
    def test_get_stats_with_baseline(self, dbc):
        """Test getting stats with baseline."""
        # Add mock baseline
        dbc.baseline_robustness = {
            "layer_1": RobustnessMetrics(
                layer_name="layer_1",
                perturbation_tolerance=0.1,
                gradient_magnitude=1.0,
                lipschitz_constant=2.0,
                basin_volume=0.2,
                boundary_distance=0.3,
                confidence=0.8
            )
        }
        
        # Add mock boundary cache
        dbc.boundary_cache["layer_1"] = [Mock(spec=DecisionBoundary)]
        
        stats = dbc.get_stats()
        
        assert stats["baseline_robustness"]["layers"] == 1
        assert "avg_perturbation_tolerance" in stats["baseline_robustness"]
        assert "avg_basin_volume" in stats["baseline_robustness"]
        assert stats["boundary_cache_size"] == 1
    
    def test_basin_volume_estimation(self, dbc):
        """Test basin volume estimation."""
        metrics = [
            RobustnessMetrics(
                layer_name="layer_1",
                perturbation_tolerance=0.1,
                gradient_magnitude=1.0,
                lipschitz_constant=2.0,
                basin_volume=0.1,
                boundary_distance=0.2,
                confidence=0.8
            ),
            RobustnessMetrics(
                layer_name="layer_2",
                perturbation_tolerance=0.2,
                gradient_magnitude=1.5,
                lipschitz_constant=1.5,
                basin_volume=0.15,
                boundary_distance=0.25,
                confidence=0.7
            )
        ]
        
        volume = dbc._estimate_basin_volume(metrics)
        
        assert isinstance(volume, float)
        assert volume > 0.0
    
    def test_reversible_region_estimation(self, dbc):
        """Test reversible region size estimation."""
        metrics = [
            RobustnessMetrics(
                layer_name="layer_1",
                perturbation_tolerance=0.1,
                gradient_magnitude=1.0,
                lipschitz_constant=2.0,
                basin_volume=0.1,
                boundary_distance=0.2,
                confidence=0.8
            )
        ]
        
        region_size = dbc._estimate_reversible_region_size(metrics)
        
        assert isinstance(region_size, float)
        assert region_size >= 0.0
        # Should be related to perturbation tolerance and reversibility threshold
        expected = metrics[0].perturbation_tolerance * dbc.config.reversibility_threshold
        assert abs(region_size - expected) < 1e-6