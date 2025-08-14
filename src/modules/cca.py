"""Causal Minimal Fix Sets (CCA) module for finding minimal interventions."""

import logging
import random
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
import torch
import torch.nn as nn
from itertools import combinations
from collections import defaultdict

from core.types import (
    FixSet, InterventionResult, Intervention, InterventionType, 
    IncidentBundle, ReplaySession, ActivationDict
)
from core.config import CCAConfig
from replayer.interventions import InterventionEngine


logger = logging.getLogger(__name__)


class CausalCCA:
    """Finds minimal sets of interventions that correct model failures."""
    
    def __init__(self, config: Optional[CCAConfig] = None):
        try:
            self.config = config or CCAConfig()
            self.intervention_engine = InterventionEngine()
            
            # Validate configuration
            self._validate_config()
            
            # Search state
            self.tested_combinations: Set[Tuple[str, ...]] = set()
            self.successful_interventions: List[InterventionResult] = []
            
            logger.info(f"Initialized CCA with strategy: {self.config.search_strategy}")
        except Exception as e:
            logger.error(f"Failed to initialize CCA: {e}")
            raise
    
    def find_minimal_fix_sets(self, session: ReplaySession, 
                             priority_layers: Optional[List[str]] = None) -> List[FixSet]:
        """Find minimal sets of interventions that fix the failure."""
        if not session:
            raise ValueError("Session cannot be None")
        if not hasattr(session, 'session_id'):
            raise ValueError("Invalid session: missing session_id")
        if not hasattr(session, 'reconstructed_activations') or not session.reconstructed_activations:
            raise ValueError("Invalid session: missing or empty reconstructed_activations")
            
        logger.info(f"Finding minimal fix sets for session: {session.session_id}")
        
        try:
            if priority_layers is None:
                priority_layers = list(session.reconstructed_activations.keys())
            
            if not priority_layers:
                logger.warning("No priority layers available for analysis")
                return []
            
            # Validate priority layers exist in session
            invalid_layers = [layer for layer in priority_layers 
                            if layer not in session.reconstructed_activations]
            if invalid_layers:
                logger.warning(f"Invalid layers removed from priority list: {invalid_layers}")
                priority_layers = [layer for layer in priority_layers 
                                 if layer in session.reconstructed_activations]
            
            if not priority_layers:
                logger.warning("No valid priority layers remaining after validation")
                return []
            
            # Reset search state
            self.tested_combinations.clear()
            self.successful_interventions.clear()
            
            # Test individual interventions first
            individual_results = self._test_individual_interventions(session, priority_layers)
            
            # Find combinations based on strategy
            if self.config.search_strategy == "greedy":
                fix_sets = self._greedy_search(session, priority_layers, individual_results)
            elif self.config.search_strategy == "beam":
                fix_sets = self._beam_search(session, priority_layers, individual_results)
            elif self.config.search_strategy == "random":
                fix_sets = self._random_search(session, priority_layers, individual_results)
            else:
                raise ValueError(f"Unknown search strategy: {self.config.search_strategy}")
            
            # Sort by minimality and effectiveness
            fix_sets = self._rank_fix_sets(fix_sets)
            
            logger.info(f"Found {len(fix_sets)} minimal fix sets")
            return fix_sets
            
        except Exception as e:
            logger.error(f"Error finding minimal fix sets: {e}")
            raise
    
    def _test_individual_interventions(self, session: ReplaySession, 
                                     layers: List[str]) -> List[InterventionResult]:
        """Test individual interventions on each layer."""
        logger.info(f"Testing individual interventions on {len(layers)} layers")
        
        individual_results = []
        
        for layer_name in layers:
            for intervention_type_str in self.config.intervention_types:
                try:
                    intervention_type = InterventionType(intervention_type_str)
                    intervention = self._create_intervention(
                        intervention_type, layer_name, session.reconstructed_activations
                    )
                    
                    result = self.intervention_engine.apply_intervention(
                        session.model,
                        session.reconstructed_activations,
                        intervention,
                        session.bundle.trace_data.inputs
                    )
                    
                    individual_results.append(result)
                    
                    if result.flip_success:
                        self.successful_interventions.append(result)
                        logger.debug(f"Successful intervention: {layer_name} ({intervention_type_str})")
                    
                except Exception as e:
                    logger.warning(f"Failed intervention on {layer_name}: {e}")
        
        logger.info(f"Found {len(self.successful_interventions)} successful individual interventions")
        return individual_results
    
    def _greedy_search(self, session: ReplaySession, layers: List[str], 
                      individual_results: List[InterventionResult]) -> List[FixSet]:
        """Greedy search for minimal fix sets."""
        logger.debug("Starting greedy search")
        
        fix_sets = []
        
        # Start with successful individual interventions
        for result in self.successful_interventions:
            fix_set = FixSet(
                interventions=[result],
                sufficiency_score=1.0 if result.flip_success else 0.0,
                necessity_scores=[1.0],
                minimality_rank=1,
                total_flip_rate=1.0 if result.flip_success else 0.0,
                avg_side_effects=np.mean(list(result.side_effects.values())) if result.side_effects else 0.0
            )
            fix_sets.append(fix_set)
        
        # Early stop if we have good individual fixes
        if len(fix_sets) > 0 and max(fs.total_flip_rate for fs in fix_sets) >= self.config.early_stop_threshold:
            logger.info("Early stopping: found sufficient individual fixes")
            return fix_sets
        
        # If no individual interventions succeeded, create simplified fix sets
        if not self.successful_interventions:
            logger.warning("No successful individual interventions found")
            # Create a default fix set from the best available results
            if individual_results:
                best_results = sorted(individual_results, key=lambda r: r.confidence, reverse=True)[:3]
                fix_set = FixSet(
                    interventions=best_results,
                    sufficiency_score=0.5,  # Default moderate score
                    necessity_scores=[0.5] * len(best_results),
                    minimality_rank=len(best_results),
                    total_flip_rate=0.0,
                    avg_side_effects=np.mean([np.mean(list(r.side_effects.values())) 
                                            for r in best_results if r.side_effects])
                )
                fix_sets.append(fix_set)
        
        # Try combinations greedily - but only if we have successful interventions to combine
        if self.successful_interventions:
            remaining_layers = [layer for layer in layers 
                               if not any(result.intervention.layer_name == layer and result.flip_success 
                                        for result in individual_results)]
            
            current_combination = []
            current_flip_rate = 0.0
            
            while len(current_combination) < self.config.max_fix_set_size and current_flip_rate < self.config.early_stop_threshold:
                best_addition = None
                best_improvement = 0.0
                
                for layer_name in remaining_layers:
                    if layer_name in [interv.intervention.layer_name for interv in current_combination]:
                        continue
                    
                    # Test adding this layer - with safe fallback
                    try:
                        best_result = self._get_best_intervention_for_layer(layer_name, individual_results)
                        test_combination = current_combination + [best_result]
                        
                        if len(test_combination) <= self.config.max_fix_set_size:
                            flip_rate = self._evaluate_combination(session, test_combination)
                            improvement = flip_rate - current_flip_rate
                            
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_addition = best_result
                    except ValueError:
                        # Skip layers without valid results
                        continue
                
                if best_addition is None or best_improvement <= 0:
                    break
                
                current_combination.append(best_addition)
                current_flip_rate += best_improvement
                remaining_layers.remove(best_addition.intervention.layer_name)
            
            if len(current_combination) > 1:
                fix_set = FixSet(
                    interventions=current_combination,
                    sufficiency_score=current_flip_rate,
                    necessity_scores=self._compute_necessity_scores(session, current_combination),
                    minimality_rank=len(current_combination),
                    total_flip_rate=current_flip_rate,
                    avg_side_effects=np.mean([np.mean(list(r.side_effects.values())) 
                                            for r in current_combination if r.side_effects])
                )
                fix_sets.append(fix_set)
        
        return fix_sets
    
    def _beam_search(self, session: ReplaySession, layers: List[str],
                    individual_results: List[InterventionResult]) -> List[FixSet]:
        """Beam search for minimal fix sets."""
        logger.debug("Starting beam search")
        
        beam_width = min(5, len(self.successful_interventions))
        if beam_width == 0:
            return []
        
        # Initialize beam with best individual interventions
        beam = sorted(self.successful_interventions, 
                     key=lambda r: r.confidence * (1.0 if r.flip_success else 0.5),
                     reverse=True)[:beam_width]
        
        fix_sets = []
        
        # Add individual fixes
        for result in beam:
            fix_set = FixSet(
                interventions=[result],
                sufficiency_score=1.0 if result.flip_success else 0.0,
                necessity_scores=[1.0],
                minimality_rank=1,
                total_flip_rate=1.0 if result.flip_success else 0.0,
                avg_side_effects=np.mean(list(result.side_effects.values())) if result.side_effects else 0.0
            )
            fix_sets.append(fix_set)
        
        # Expand beam with combinations
        for size in range(2, min(self.config.max_fix_set_size + 1, len(layers) + 1)):
            new_beam = []
            
            for base_result in beam:
                for layer_name in layers:
                    if layer_name == base_result.intervention.layer_name:
                        continue
                    
                    for intervention_type_str in self.config.intervention_types:
                        try:
                            intervention_type = InterventionType(intervention_type_str)
                            new_intervention = self._create_intervention(
                                intervention_type, layer_name, session.reconstructed_activations
                            )
                            
                            # Actually apply the intervention
                            new_result = self.intervention_engine.apply_intervention(
                                session.model,
                                session.reconstructed_activations,
                                new_intervention,
                                session.bundle.trace_data.inputs
                            )
                            
                            combination = [base_result, new_result]
                            flip_rate = self._evaluate_combination(session, combination)
                            if flip_rate > 0:
                                new_beam.append((combination, flip_rate))
                                
                        except Exception as e:
                            logger.debug(f"Beam search intervention failed: {e}")
            
            # Keep best candidates
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = [combination for combination, _ in new_beam[:beam_width]]
            
            # Add successful combinations to fix sets
            for combination, flip_rate in new_beam:
                if flip_rate >= self.config.early_stop_threshold:
                    fix_set = FixSet(
                        interventions=combination,
                        sufficiency_score=flip_rate,
                        necessity_scores=self._compute_necessity_scores(session, combination),
                        minimality_rank=len(combination),
                        total_flip_rate=flip_rate,
                        avg_side_effects=np.mean([np.mean(list(r.side_effects.values())) 
                                                for r in combination if r.side_effects])
                    )
                    fix_sets.append(fix_set)
        
        return fix_sets
    
    def _random_search(self, session: ReplaySession, layers: List[str],
                      individual_results: List[InterventionResult]) -> List[FixSet]:
        """Random search for minimal fix sets."""
        logger.debug("Starting random search")
        
        fix_sets = []
        max_attempts = min(100, len(layers) * len(self.config.intervention_types))
        
        # Add successful individual interventions
        for result in self.successful_interventions:
            fix_set = FixSet(
                interventions=[result],
                sufficiency_score=1.0 if result.flip_success else 0.0,
                necessity_scores=[1.0],
                minimality_rank=1,
                total_flip_rate=1.0 if result.flip_success else 0.0,
                avg_side_effects=np.mean(list(result.side_effects.values())) if result.side_effects else 0.0
            )
            fix_sets.append(fix_set)
        
        # Random combination search
        for attempt in range(max_attempts):
            try:
                # Random combination size
                size = random.randint(2, min(self.config.max_fix_set_size, len(layers)))
                selected_layers = random.sample(layers, size)
                
                combination = []
                for layer_name in selected_layers:
                    intervention_type_str = random.choice(self.config.intervention_types)
                    intervention_type = InterventionType(intervention_type_str)
                    
                    intervention = self._create_intervention(
                        intervention_type, layer_name, session.reconstructed_activations
                    )
                    
                    # Actually apply the intervention
                    result = self.intervention_engine.apply_intervention(
                        session.model,
                        session.reconstructed_activations,
                        intervention,
                        session.bundle.trace_data.inputs
                    )
                    combination.append(result)
                
                flip_rate = self._evaluate_combination(session, combination)
                
                if flip_rate > 0.5:  # Threshold for random search
                    fix_set = FixSet(
                        interventions=combination,
                        sufficiency_score=flip_rate,
                        necessity_scores=self._compute_necessity_scores(session, combination),
                        minimality_rank=len(combination),
                        total_flip_rate=flip_rate,
                        avg_side_effects=0.0  # Simplified for random search
                    )
                    fix_sets.append(fix_set)
                
            except Exception as e:
                logger.debug(f"Random search attempt {attempt} failed: {e}")
        
        return fix_sets
    
    def _create_intervention(self, intervention_type: InterventionType, layer_name: str,
                           activations: ActivationDict) -> Intervention:
        """Create an intervention for a specific layer."""
        if layer_name not in activations:
            raise ValueError(f"Layer {layer_name} not found in activations")
        
        activation = activations[layer_name]
        
        if intervention_type == InterventionType.ZERO:
            return Intervention(
                type=intervention_type,
                layer_name=layer_name,
                value=0.0
            )
        elif intervention_type == InterventionType.MEAN:
            mean_value = torch.mean(activation).item()
            return Intervention(
                type=intervention_type,
                layer_name=layer_name,
                value=mean_value
            )
        elif intervention_type == InterventionType.PATCH:
            # For now, use mean as patch value - in practice would use benign donor
            mean_activation = torch.mean(activation, dim=0, keepdim=True).expand_as(activation)
            return Intervention(
                type=intervention_type,
                layer_name=layer_name,
                donor_activation=mean_activation
            )
        else:
            # Default to zero intervention
            return Intervention(
                type=InterventionType.ZERO,
                layer_name=layer_name,
                value=0.0
            )
    
    def _get_best_intervention_for_layer(self, layer_name: str, 
                                       results: List[InterventionResult]) -> InterventionResult:
        """Get the best intervention result for a specific layer."""
        layer_results = [r for r in results if r.intervention.layer_name == layer_name]
        if not layer_results:
            raise ValueError(f"No results found for layer {layer_name}")
        
        # Sort by flip success and confidence
        layer_results.sort(key=lambda r: (r.flip_success, r.confidence), reverse=True)
        return layer_results[0]
    
    def _evaluate_combination(self, session: ReplaySession, 
                            combination: List[InterventionResult]) -> float:
        """Evaluate the effectiveness of a combination of interventions."""
        # For now, use a simplified heuristic based on individual results
        # In practice, would need to actually test the combination
        
        if not combination:
            return 0.0
        
        # Average flip success weighted by confidence
        total_weight = 0.0
        weighted_success = 0.0
        
        for result in combination:
            weight = result.confidence
            success = 1.0 if result.flip_success else 0.0
            
            weighted_success += weight * success
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_success / total_weight
    
    def _compute_necessity_scores(self, session: ReplaySession,
                                combination: List[InterventionResult]) -> List[float]:
        """Compute necessity scores for each intervention in the combination."""
        necessity_scores = []
        
        for i, target_result in enumerate(combination):
            # Test combination without this intervention
            reduced_combination = combination[:i] + combination[i+1:]
            
            if not reduced_combination:
                # Single intervention is fully necessary
                necessity_scores.append(1.0)
            else:
                full_rate = self._evaluate_combination(session, combination)
                reduced_rate = self._evaluate_combination(session, reduced_combination)
                
                # Necessity = how much performance drops without this intervention
                necessity = max(0.0, full_rate - reduced_rate) / max(full_rate, 1e-6)
                necessity_scores.append(necessity)
        
        return necessity_scores
    
    def _rank_fix_sets(self, fix_sets: List[FixSet]) -> List[FixSet]:
        """Rank fix sets by quality metrics."""
        if not fix_sets:
            return fix_sets
        
        # Sort by: flip rate (desc), minimality (asc), side effects (asc)
        def ranking_key(fix_set: FixSet) -> Tuple[float, int, float]:
            return (
                -fix_set.total_flip_rate,  # Higher flip rate is better
                fix_set.minimality_rank,   # Smaller sets are better
                fix_set.avg_side_effects   # Lower side effects are better
            )
        
        ranked = sorted(fix_sets, key=ranking_key)
        
        # Assign minimality ranks
        for i, fix_set in enumerate(ranked):
            fix_set.minimality_rank = i + 1
        
        return ranked
    
    def _validate_config(self) -> None:
        """Validate CCA configuration."""
        if not hasattr(self.config, 'search_strategy'):
            raise ValueError("Config missing search_strategy")
        if self.config.search_strategy not in ["greedy", "beam", "random"]:
            raise ValueError(f"Invalid search strategy: {self.config.search_strategy}")
        if not hasattr(self.config, 'max_fix_set_size') or self.config.max_fix_set_size < 1:
            raise ValueError("Invalid max_fix_set_size: must be >= 1")
        if not hasattr(self.config, 'intervention_types') or not self.config.intervention_types:
            raise ValueError("Config missing intervention_types")
        if not hasattr(self.config, 'early_stop_threshold'):
            raise ValueError("Config missing early_stop_threshold")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get CCA statistics."""
        return {
            "tested_combinations": len(self.tested_combinations),
            "successful_interventions": len(self.successful_interventions),
            "config": {
                "max_fix_set_size": self.config.max_fix_set_size,
                "search_strategy": self.config.search_strategy,
                "early_stop_threshold": self.config.early_stop_threshold,
                "intervention_types": self.config.intervention_types
            }
        }