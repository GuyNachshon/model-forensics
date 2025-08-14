"""Unit tests for the Causal Minimal Fix Sets (CCA) module."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from typing import List

from src.modules.cca import CausalCCA
from src.core.types import (
    ReplaySession, FixSet, InterventionResult, Intervention, 
    InterventionType, IncidentBundle
)
from src.core.config import CCAConfig


class TestCausalCCA:
    """Unit tests for CCA module."""
    
    def test_cca_initialization(self):
        """Test CCA initialization."""
        config = CCAConfig()
        cca = CausalCCA(config)
        
        assert cca.config == config
        assert hasattr(cca, 'intervention_engine')
        assert isinstance(cca.tested_combinations, set)
        assert isinstance(cca.successful_interventions, list)
    
    def test_cca_initialization_validation(self):
        """Test CCA initialization with invalid config."""
        # Test with invalid search strategy
        config = CCAConfig()
        config.search_strategy = "invalid_strategy"
        
        with pytest.raises(ValueError, match="Invalid search strategy"):
            CausalCCA(config)
        
        # Test with invalid max_fix_set_size
        config = CCAConfig()
        config.max_fix_set_size = 0
        
        with pytest.raises(ValueError, match="Invalid max_fix_set_size"):
            CausalCCA(config)
    
    def test_find_minimal_fix_sets_validation(self, cca_module):
        """Test input validation for find_minimal_fix_sets."""
        # Test None session
        with pytest.raises(ValueError, match="Session cannot be None"):
            cca_module.find_minimal_fix_sets(None)
        
        # Test invalid session (missing session_id)
        mock_session = Mock()
        delattr(mock_session, 'session_id') if hasattr(mock_session, 'session_id') else None
        with pytest.raises(ValueError, match="Invalid session: missing session_id"):
            cca_module.find_minimal_fix_sets(mock_session)
        
        # Test invalid session (missing reconstructed_activations)
        mock_session = Mock()
        mock_session.session_id = "test_session"
        delattr(mock_session, 'reconstructed_activations') if hasattr(mock_session, 'reconstructed_activations') else None
        with pytest.raises(ValueError, match="Invalid session: missing or empty reconstructed_activations"):
            cca_module.find_minimal_fix_sets(mock_session)
        
        # Test empty reconstructed_activations
        mock_session = Mock()
        mock_session.session_id = "test_session"
        mock_session.reconstructed_activations = {}
        result = cca_module.find_minimal_fix_sets(mock_session)
        assert result == []
    
    @patch('src.modules.cca.InterventionEngine')
    def test_create_intervention(self, mock_engine_class, cca_module):
        """Test intervention creation."""
        # Mock activations
        activations = {
            "layer1": torch.randn(10, 768),
            "layer2": torch.randn(10, 512)
        }
        
        # Test ZERO intervention
        intervention = cca_module._create_intervention(
            InterventionType.ZERO, "layer1", activations
        )
        
        assert isinstance(intervention, Intervention)
        assert intervention.type == InterventionType.ZERO
        assert intervention.layer_name == "layer1"
        assert intervention.value == 0.0
        
        # Test MEAN intervention
        intervention = cca_module._create_intervention(
            InterventionType.MEAN, "layer2", activations
        )
        
        assert isinstance(intervention, Intervention)
        assert intervention.type == InterventionType.MEAN
        assert intervention.layer_name == "layer2"
        assert isinstance(intervention.value, float)
        
        # Test PATCH intervention
        intervention = cca_module._create_intervention(
            InterventionType.PATCH, "layer1", activations
        )
        
        assert isinstance(intervention, Intervention)
        assert intervention.type == InterventionType.PATCH
        assert intervention.layer_name == "layer1"
        assert intervention.donor_activation is not None
        
        # Test with invalid layer
        with pytest.raises(ValueError, match="Layer .* not found in activations"):
            cca_module._create_intervention(
                InterventionType.ZERO, "invalid_layer", activations
            )
    
    def test_get_best_intervention_for_layer(self, cca_module):
        """Test getting best intervention for a layer."""
        # Create mock results
        result1 = Mock(spec=InterventionResult)
        result1.intervention.layer_name = "layer1"
        result1.flip_success = False
        result1.confidence = 0.6
        
        result2 = Mock(spec=InterventionResult)
        result2.intervention.layer_name = "layer1"
        result2.flip_success = True
        result2.confidence = 0.8
        
        result3 = Mock(spec=InterventionResult)
        result3.intervention.layer_name = "layer2"
        result3.flip_success = True
        result3.confidence = 0.9
        
        results = [result1, result2, result3]
        
        # Test getting best for layer1
        best = cca_module._get_best_intervention_for_layer("layer1", results)
        assert best == result2  # Higher flip_success and confidence
        
        # Test getting best for layer2
        best = cca_module._get_best_intervention_for_layer("layer2", results)
        assert best == result3
        
        # Test with non-existent layer
        with pytest.raises(ValueError, match="No results found for layer"):
            cca_module._get_best_intervention_for_layer("invalid_layer", results)
    
    def test_evaluate_combination(self, cca_module):
        """Test combination evaluation."""
        # Create mock intervention results
        result1 = Mock(spec=InterventionResult)
        result1.flip_success = True
        result1.confidence = 0.8
        
        result2 = Mock(spec=InterventionResult)
        result2.flip_success = False
        result2.confidence = 0.6
        
        result3 = Mock(spec=InterventionResult)
        result3.flip_success = True
        result3.confidence = 0.9
        
        # Test with successful combination
        mock_session = Mock(spec=ReplaySession)
        combination = [result1, result3]
        
        flip_rate = cca_module._evaluate_combination(mock_session, combination)
        assert 0 <= flip_rate <= 1
        assert flip_rate > 0  # Should have positive rate with successful interventions
        
        # Test with empty combination
        flip_rate = cca_module._evaluate_combination(mock_session, [])
        assert flip_rate == 0.0
        
        # Test with failed interventions
        combination = [result2]
        flip_rate = cca_module._evaluate_combination(mock_session, combination)
        assert flip_rate == 0.0  # No successful interventions
    
    def test_compute_necessity_scores(self, cca_module):
        """Test necessity score computation."""
        # Create mock intervention results
        result1 = Mock(spec=InterventionResult)
        result1.flip_success = True
        result1.confidence = 0.8
        
        result2 = Mock(spec=InterventionResult)
        result2.flip_success = True
        result2.confidence = 0.6
        
        mock_session = Mock(spec=ReplaySession)
        combination = [result1, result2]
        
        necessity_scores = cca_module._compute_necessity_scores(mock_session, combination)
        
        assert len(necessity_scores) == 2
        assert all(0 <= score <= 1 for score in necessity_scores)
        
        # Test with single intervention (should be fully necessary)
        single_combination = [result1]
        necessity_scores = cca_module._compute_necessity_scores(mock_session, single_combination)
        
        assert len(necessity_scores) == 1
        assert necessity_scores[0] == 1.0
    
    def test_rank_fix_sets(self, cca_module):
        """Test fix set ranking."""
        # Create test fix sets
        fix_set1 = FixSet(
            interventions=[Mock()],
            sufficiency_score=0.7,
            necessity_scores=[0.8],
            minimality_rank=0,  # Will be overwritten
            total_flip_rate=0.7,
            avg_side_effects=0.3
        )
        
        fix_set2 = FixSet(
            interventions=[Mock()],
            sufficiency_score=0.9,
            necessity_scores=[0.9],
            minimality_rank=0,  # Will be overwritten
            total_flip_rate=0.9,
            avg_side_effects=0.1
        )
        
        fix_set3 = FixSet(
            interventions=[Mock(), Mock()],
            sufficiency_score=0.8,
            necessity_scores=[0.7, 0.6],
            minimality_rank=0,  # Will be overwritten
            total_flip_rate=0.8,
            avg_side_effects=0.2
        )
        
        fix_sets = [fix_set1, fix_set2, fix_set3]
        ranked = cca_module._rank_fix_sets(fix_sets)
        
        # Should be ranked by flip rate (desc), then minimality (asc), then side effects (asc)
        assert ranked[0] == fix_set2  # Highest flip rate, lowest side effects
        assert ranked[1] == fix_set3  # Second highest flip rate
        assert ranked[2] == fix_set1  # Lowest flip rate
        
        # Check that minimality ranks were assigned
        assert ranked[0].minimality_rank == 1
        assert ranked[1].minimality_rank == 2
        assert ranked[2].minimality_rank == 3
        
        # Test with empty list
        assert cca_module._rank_fix_sets([]) == []
    
    @patch.object(CausalCCA, '_test_individual_interventions')
    @patch.object(CausalCCA, '_greedy_search')
    def test_find_minimal_fix_sets_greedy(self, mock_greedy, mock_individual, cca_module):
        """Test fix set finding with greedy strategy."""
        # Setup
        cca_module.config.search_strategy = "greedy"
        
        mock_session = Mock(spec=ReplaySession)
        mock_session.session_id = "test_session"
        mock_session.reconstructed_activations = {"layer1": torch.randn(10, 768)}
        
        mock_individual_results = [Mock()]
        mock_individual.return_value = mock_individual_results
        
        mock_fix_sets = [Mock(spec=FixSet)]
        mock_greedy.return_value = mock_fix_sets
        
        # Test
        result = cca_module.find_minimal_fix_sets(mock_session)
        
        # Verify
        mock_individual.assert_called_once()
        mock_greedy.assert_called_once()
        assert result == mock_fix_sets
    
    @patch.object(CausalCCA, '_test_individual_interventions')
    @patch.object(CausalCCA, '_beam_search')
    def test_find_minimal_fix_sets_beam(self, mock_beam, mock_individual, cca_module):
        """Test fix set finding with beam strategy."""
        # Setup
        cca_module.config.search_strategy = "beam"
        
        mock_session = Mock(spec=ReplaySession)
        mock_session.session_id = "test_session"
        mock_session.reconstructed_activations = {"layer1": torch.randn(10, 768)}
        
        mock_individual_results = [Mock()]
        mock_individual.return_value = mock_individual_results
        
        mock_fix_sets = [Mock(spec=FixSet)]
        mock_beam.return_value = mock_fix_sets
        
        # Test
        result = cca_module.find_minimal_fix_sets(mock_session)
        
        # Verify
        mock_individual.assert_called_once()
        mock_beam.assert_called_once()
        assert result == mock_fix_sets
    
    def test_find_minimal_fix_sets_invalid_strategy(self, cca_module):
        """Test fix set finding with invalid strategy."""
        # Setup invalid strategy
        cca_module.config.search_strategy = "invalid"
        
        mock_session = Mock(spec=ReplaySession)
        mock_session.session_id = "test_session"
        mock_session.reconstructed_activations = {"layer1": torch.randn(10, 768)}
        
        # Should raise error
        with pytest.raises(ValueError, match="Unknown search strategy"):
            cca_module.find_minimal_fix_sets(mock_session)
    
    def test_get_stats(self, cca_module):
        """Test statistics reporting."""
        # Add some test data
        cca_module.tested_combinations.add(("layer1",))
        cca_module.tested_combinations.add(("layer1", "layer2"))
        cca_module.successful_interventions = [Mock(), Mock()]
        
        stats = cca_module.get_stats()
        
        assert "tested_combinations" in stats
        assert "successful_interventions" in stats
        assert "config" in stats
        
        assert stats["tested_combinations"] == 2
        assert stats["successful_interventions"] == 2
        
        config_stats = stats["config"]
        assert "max_fix_set_size" in config_stats
        assert "search_strategy" in config_stats
        assert "early_stop_threshold" in config_stats
        assert "intervention_types" in config_stats


@pytest.mark.integration
class TestCausalCCAIntegration:
    """Integration tests for CCA module."""
    
    def test_cca_with_mock_session(self):
        """Test CCA with a properly mocked session."""
        config = CCAConfig()
        config.max_fix_set_size = 2  # Limit for testing
        cca = CausalCCA(config)
        
        # Create realistic mock session
        mock_session = Mock(spec=ReplaySession)
        mock_session.session_id = "integration_test_session"
        mock_session.reconstructed_activations = {
            "layer1": torch.randn(1, 10, 768),
            "layer2": torch.randn(1, 10, 768)
        }
        mock_session.model = Mock()
        mock_session.bundle = Mock()
        mock_session.bundle.trace_data = Mock()
        mock_session.bundle.trace_data.inputs = ["test input"]
        
        # Mock intervention engine to always succeed
        with patch.object(cca, 'intervention_engine') as mock_engine:
            mock_result = Mock(spec=InterventionResult)
            mock_result.flip_success = True
            mock_result.confidence = 0.8
            mock_result.side_effects = {}
            mock_result.intervention = Mock()
            mock_result.intervention.layer_name = "layer1"
            
            mock_engine.apply_intervention.return_value = mock_result
            
            # Should not raise exceptions
            fix_sets = cca.find_minimal_fix_sets(mock_session, ["layer1"])
            
            assert isinstance(fix_sets, list)
            # Should have at least one fix set from successful interventions
            assert len(fix_sets) > 0
    
    def test_cca_error_recovery(self):
        """Test CCA error recovery mechanisms."""
        config = CCAConfig()
        cca = CausalCCA(config)
        
        # Test with invalid priority layers
        mock_session = Mock(spec=ReplaySession)
        mock_session.session_id = "error_test_session"
        mock_session.reconstructed_activations = {"layer1": torch.randn(1, 10, 768)}
        
        # Should handle invalid layers gracefully
        fix_sets = cca.find_minimal_fix_sets(mock_session, ["invalid_layer", "layer1"])
        
        assert isinstance(fix_sets, list)
        # Should still work with valid layers