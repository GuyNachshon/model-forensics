"""Unit tests for the RCA pipeline module."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.modules.pipeline import RCAPipeline
from src.core.types import IncidentBundle, CausalResult, CompressionMetrics, FixSet
from src.core.config import RCAConfig


class TestRCAPipeline:
    """Unit tests for RCA pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        config = RCAConfig()
        pipeline = RCAPipeline(config)
        
        assert pipeline.config == config
        assert pipeline.cf is not None
        assert pipeline.cca is not None
        assert pipeline.replayer is not None
    
    def test_pipeline_initialization_error_handling(self):
        """Test pipeline initialization with invalid config."""
        with pytest.raises(Exception):
            RCAPipeline(None)
    
    def test_analyze_incident_validation(self, pipeline):
        """Test input validation for analyze_incident."""
        # Test None incident bundle
        with pytest.raises(ValueError, match="Incident bundle cannot be None"):
            pipeline.analyze_incident(None, Mock())
        
        # Test None model
        with pytest.raises(ValueError, match="Model cannot be None"):
            pipeline.analyze_incident(Mock(), None)
    
    @patch('src.modules.pipeline.Replayer')
    def test_analyze_incident_success(self, mock_replayer_class, pipeline):
        """Test successful incident analysis."""
        # Setup mocks
        mock_bundle = Mock(spec=IncidentBundle)
        mock_bundle.bundle_id = "test_bundle_001"
        mock_model = Mock()
        
        mock_session = Mock()
        mock_session.reconstructed_activations = {"layer1": torch.randn(1, 10, 768)}
        
        mock_replayer = Mock()
        mock_replayer.create_session.return_value = mock_session
        pipeline.replayer = mock_replayer
        
        # Mock CF results
        mock_cf_metrics = [
            CompressionMetrics(
                layer_name="layer1",
                compression_ratio=1.5,
                entropy=4.2,
                anomaly_score=0.8,
                baseline_comparison=0.3,
                is_anomalous=True,
                confidence=0.9
            )
        ]
        
        pipeline.cf.analyze_bundle = Mock(return_value=mock_cf_metrics)
        pipeline.cf.prioritize_layers = Mock(return_value=["layer1"])
        pipeline.cf.build_baseline = Mock()
        
        # Mock CCA results
        mock_fix_set = FixSet(
            interventions=[Mock()],
            sufficiency_score=0.9,
            necessity_scores=[0.8],
            minimality_rank=1,
            total_flip_rate=0.85,
            avg_side_effects=0.1
        )
        
        pipeline.cca.find_minimal_fix_sets = Mock(return_value=[mock_fix_set])
        
        # Run analysis
        result = pipeline.analyze_incident(mock_bundle, mock_model)
        
        # Verify result
        assert isinstance(result, CausalResult)
        assert result.fix_set == mock_fix_set
        assert len(result.compression_metrics) == 1
        assert result.confidence > 0
        assert result.execution_time > 0
    
    @patch('src.modules.pipeline.Replayer')
    def test_analyze_incident_cf_failure(self, mock_replayer_class, pipeline):
        """Test incident analysis when CF fails."""
        # Setup mocks
        mock_bundle = Mock(spec=IncidentBundle)
        mock_bundle.bundle_id = "test_bundle_001"
        mock_model = Mock()
        
        mock_session = Mock()
        mock_session.reconstructed_activations = {"layer1": torch.randn(1, 10, 768)}
        
        mock_replayer = Mock()
        mock_replayer.create_session.return_value = mock_session
        pipeline.replayer = mock_replayer
        
        # Make CF fail
        pipeline.cf.analyze_bundle = Mock(side_effect=Exception("CF failed"))
        
        # Mock CCA to work normally
        mock_fix_set = FixSet(
            interventions=[Mock()],
            sufficiency_score=0.7,
            necessity_scores=[0.6],
            minimality_rank=1,
            total_flip_rate=0.7,
            avg_side_effects=0.2
        )
        
        pipeline.cca.find_minimal_fix_sets = Mock(return_value=[mock_fix_set])
        
        # Run analysis - should not fail even if CF fails
        result = pipeline.analyze_incident(mock_bundle, mock_model)
        
        # Verify result
        assert isinstance(result, CausalResult)
        assert result.fix_set == mock_fix_set
        assert len(result.compression_metrics) == 0  # CF failed
        assert result.confidence > 0
    
    @patch('src.modules.pipeline.Replayer')
    def test_analyze_incident_cca_failure(self, mock_replayer_class, pipeline):
        """Test incident analysis when CCA fails."""
        # Setup mocks
        mock_bundle = Mock(spec=IncidentBundle)
        mock_bundle.bundle_id = "test_bundle_001"
        mock_model = Mock()
        
        mock_session = Mock()
        mock_session.reconstructed_activations = {"layer1": torch.randn(1, 10, 768)}
        
        mock_replayer = Mock()
        mock_replayer.create_session.return_value = mock_session
        pipeline.replayer = mock_replayer
        
        # Mock CF to work normally
        mock_cf_metrics = [
            CompressionMetrics(
                layer_name="layer1",
                compression_ratio=1.2,
                entropy=3.8,
                anomaly_score=0.6,
                baseline_comparison=0.2,
                is_anomalous=True,
                confidence=0.8
            )
        ]
        
        pipeline.cf.analyze_bundle = Mock(return_value=mock_cf_metrics)
        pipeline.cf.prioritize_layers = Mock(return_value=["layer1"])
        
        # Make CCA fail
        pipeline.cca.find_minimal_fix_sets = Mock(side_effect=Exception("CCA failed"))
        
        # Run analysis - should not fail even if CCA fails
        result = pipeline.analyze_incident(mock_bundle, mock_model)
        
        # Verify result
        assert isinstance(result, CausalResult)
        assert result.fix_set is None  # CCA failed
        assert len(result.compression_metrics) == 1  # CF succeeded
        assert result.confidence >= 0
    
    def test_select_best_fix_set(self, pipeline):
        """Test fix set selection logic."""
        # Create test fix sets
        fix_set1 = FixSet(
            interventions=[Mock()],
            sufficiency_score=0.7,
            necessity_scores=[0.6],
            minimality_rank=2,
            total_flip_rate=0.7,
            avg_side_effects=0.3
        )
        
        fix_set2 = FixSet(
            interventions=[Mock()],
            sufficiency_score=0.9,
            necessity_scores=[0.8],
            minimality_rank=1,
            total_flip_rate=0.9,
            avg_side_effects=0.1
        )
        
        fix_set3 = FixSet(
            interventions=[Mock(), Mock()],
            sufficiency_score=0.8,
            necessity_scores=[0.7, 0.5],
            minimality_rank=3,
            total_flip_rate=0.8,
            avg_side_effects=0.2
        )
        
        # Test selection
        best = pipeline._select_best_fix_set([fix_set1, fix_set2, fix_set3])
        
        # Should select fix_set2 (highest flip rate and lowest minimality rank)
        assert best == fix_set2
        
        # Test with empty list
        assert pipeline._select_best_fix_set([]) is None
        
        # Test with single fix set
        assert pipeline._select_best_fix_set([fix_set1]) == fix_set1
    
    def test_compute_overall_confidence(self, pipeline):
        """Test confidence computation."""
        # Test with both CF and CCA results
        cf_metrics = [
            CompressionMetrics(
                layer_name="layer1",
                compression_ratio=1.5,
                entropy=4.2,
                anomaly_score=0.8,
                baseline_comparison=0.3,
                is_anomalous=True,
                confidence=0.9
            )
        ]
        
        fix_set = FixSet(
            interventions=[Mock()],
            sufficiency_score=0.9,
            necessity_scores=[0.8],
            minimality_rank=1,
            total_flip_rate=0.85,
            avg_side_effects=0.1
        )
        
        confidence = pipeline._compute_overall_confidence(cf_metrics, [fix_set])
        assert 0 <= confidence <= 1
        
        # Test with no results
        confidence = pipeline._compute_overall_confidence([], [])
        assert confidence == 0.0
        
        # Test with only CF results
        confidence = pipeline._compute_overall_confidence(cf_metrics, [])
        assert 0 <= confidence <= 1
        
        # Test with only CCA results
        confidence = pipeline._compute_overall_confidence([], [fix_set])
        assert 0 <= confidence <= 1
    
    def test_get_pipeline_stats(self, pipeline):
        """Test pipeline statistics."""
        stats = pipeline.get_pipeline_stats()
        
        assert "cf_enabled" in stats
        assert "cca_enabled" in stats
        assert "analyzed_incidents" in stats
        assert "total_execution_time" in stats
        
        assert stats["cf_enabled"] is True
        assert stats["cca_enabled"] is True
        assert isinstance(stats["analyzed_incidents"], int)
        assert isinstance(stats["total_execution_time"], (int, float))


@pytest.mark.integration 
class TestRCAPipelineIntegration:
    """Integration tests for the RCA pipeline with real components."""
    
    def test_pipeline_with_real_components(self, test_bundle_path):
        """Test pipeline with real CF and CCA modules."""
        if not test_bundle_path.exists():
            pytest.skip("Test bundles not available")
        
        config = RCAConfig()
        pipeline = RCAPipeline(config)
        
        # Verify components are properly initialized
        assert hasattr(pipeline.cf, 'analyze_bundle')
        assert hasattr(pipeline.cca, 'find_minimal_fix_sets')
        assert hasattr(pipeline.replayer, 'load_bundle')
    
    def test_error_recovery(self):
        """Test pipeline error recovery mechanisms."""
        config = RCAConfig()
        pipeline = RCAPipeline(config)
        
        # Test with completely invalid inputs
        result = pipeline.analyze_incident(Mock(bundle_id="invalid"), Mock())
        
        # Should return a valid result even with errors
        assert isinstance(result, CausalResult)
        assert result.confidence >= 0