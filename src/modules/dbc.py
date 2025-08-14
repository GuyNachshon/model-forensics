"""Decision Basin Cartography (DBC) module for mapping decision landscapes."""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import torch.nn.functional as F

from core.types import (
    IncidentBundle, ActivationDict, DecisionBoundary, RobustnessMetrics,
    DecisionBasinMap, ReplaySession
)
from core.config import DBCConfig

logger = logging.getLogger(__name__)


class DecisionBasinCartography:
    """Maps decision landscapes around failure points to understand robustness boundaries."""
    
    def __init__(self, config: Optional[DBCConfig] = None):
        self.config = config or DBCConfig()
        self.baseline_robustness: Optional[Dict[str, RobustnessMetrics]] = None
        self.boundary_cache: Dict[str, List[DecisionBoundary]] = {}
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"Initialized DBC with method: {self.config.boundary_detection_method}")
    
    def map_decision_basin(self, session: ReplaySession, 
                          failure_activations: ActivationDict,
                          target_layers: Optional[List[str]] = None) -> DecisionBasinMap:
        """Map the decision basin around a failure point."""
        if not failure_activations:
            raise ValueError("Failure activations cannot be empty")
            
        logger.info(f"Mapping decision basin for session: {session.session_id}")
        
        try:
            # Select layers to analyze
            if target_layers is None:
                target_layers = list(failure_activations.keys())
            
            # Convert failure point to numpy for analysis
            failure_point = {
                layer: tensor.detach().cpu().numpy() 
                for layer, tensor in failure_activations.items()
                if layer in target_layers
            }
            
            # Analyze robustness for each layer
            robustness_metrics = []
            decision_boundaries = []
            perturbation_samples = {}
            
            for layer_name in target_layers:
                try:
                    logger.debug(f"Analyzing layer: {layer_name}")
                    
                    # Get robustness metrics for this layer
                    layer_robustness = self._analyze_layer_robustness(
                        session, layer_name, failure_activations[layer_name]
                    )
                    robustness_metrics.append(layer_robustness)
                    
                    # Detect decision boundaries
                    layer_boundaries = self._detect_decision_boundaries(
                        session, layer_name, failure_activations[layer_name]
                    )
                    decision_boundaries.extend(layer_boundaries)
                    
                    # Sample perturbations around failure point
                    layer_samples = self._sample_perturbations(
                        failure_activations[layer_name], layer_name
                    )
                    perturbation_samples[layer_name] = layer_samples
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze layer {layer_name}: {e}")
                    continue
            
            # Estimate overall basin volume
            basin_volume = self._estimate_basin_volume(robustness_metrics)
            
            # Identify critical layers (those with low robustness)
            critical_layers = [
                m.layer_name for m in robustness_metrics 
                if m.perturbation_tolerance < self.config.perturbation_magnitude
            ]
            
            # Estimate reversible region size
            reversible_region = self._estimate_reversible_region_size(robustness_metrics)
            
            # Create basin map
            basin_map = DecisionBasinMap(
                failure_point=failure_point,
                decision_boundaries=decision_boundaries,
                robustness_metrics=robustness_metrics,
                perturbation_samples=perturbation_samples,
                basin_volume_estimate=basin_volume,
                critical_layers=critical_layers,
                reversible_region_size=reversible_region,
                analysis_metadata={
                    "perturbation_magnitude": self.config.perturbation_magnitude,
                    "boundary_detection_method": self.config.boundary_detection_method,
                    "samples_per_layer": self.config.robustness_samples,
                    "layers_analyzed": len(target_layers)
                }
            )
            
            logger.info(f"Mapped basin: {len(decision_boundaries)} boundaries, "
                       f"{len(critical_layers)} critical layers, "
                       f"volume estimate: {basin_volume:.4f}")
            
            return basin_map
            
        except Exception as e:
            logger.error(f"Error mapping decision basin: {e}")
            raise
    
    def compare_to_baseline(self, basin_map: DecisionBasinMap) -> Dict[str, Any]:
        """Compare basin map to baseline robustness patterns."""
        if not self.baseline_robustness:
            logger.warning("No baseline robustness available for comparison")
            return {"robustness_changes": {}, "critical_layer_changes": []}
        
        try:
            robustness_changes = {}
            critical_layer_changes = []
            
            # Compare robustness metrics to baseline
            for metrics in basin_map.robustness_metrics:
                layer_name = metrics.layer_name
                baseline_metrics = self.baseline_robustness.get(layer_name)
                
                if baseline_metrics:
                    tolerance_change = metrics.perturbation_tolerance - baseline_metrics.perturbation_tolerance
                    volume_change = metrics.basin_volume - baseline_metrics.basin_volume
                    
                    robustness_changes[layer_name] = {
                        "tolerance_change": tolerance_change,
                        "volume_change": volume_change,
                        "gradient_magnitude_change": metrics.gradient_magnitude - baseline_metrics.gradient_magnitude,
                        "became_critical": (layer_name in basin_map.critical_layers and 
                                          layer_name not in self._get_baseline_critical_layers())
                    }
                    
                    if robustness_changes[layer_name]["became_critical"]:
                        critical_layer_changes.append(layer_name)
            
            return {
                "robustness_changes": robustness_changes,
                "critical_layer_changes": critical_layer_changes,
                "basin_volume_change": basin_map.basin_volume_estimate - self._get_baseline_basin_volume(),
                "num_new_critical_layers": len(critical_layer_changes)
            }
            
        except Exception as e:
            logger.error(f"Error comparing to baseline: {e}")
            return {"error": str(e)}
    
    def build_baseline(self, baseline_sessions: List[ReplaySession]) -> None:
        """Build baseline robustness patterns from benign sessions."""
        if not baseline_sessions:
            raise ValueError("Baseline sessions cannot be empty")
            
        logger.info(f"Building robustness baseline from {len(baseline_sessions)} sessions")
        
        try:
            all_robustness = defaultdict(list)
            
            # Analyze each baseline session
            for session in baseline_sessions:
                try:
                    # Get activations for this session
                    activations = session.reconstructed_activations
                    if not activations:
                        logger.warning(f"No activations in session {session.session_id}")
                        continue
                    
                    # Analyze robustness for each layer
                    for layer_name, activation in activations.items():
                        try:
                            robustness = self._analyze_layer_robustness(session, layer_name, activation)
                            all_robustness[layer_name].append(robustness)
                        except Exception as e:
                            logger.debug(f"Failed to analyze baseline layer {layer_name}: {e}")
                            continue
                            
                except Exception as e:
                    logger.warning(f"Failed to analyze baseline session {session.session_id}: {e}")
                    continue
            
            if not all_robustness:
                raise ValueError("No valid baseline robustness data found")
            
            # Aggregate baseline robustness
            self.baseline_robustness = self._aggregate_baseline_robustness(all_robustness)
            
            logger.info(f"Built baseline with {len(self.baseline_robustness)} layer robustness profiles")
            
        except Exception as e:
            logger.error(f"Error building baseline: {e}")
            raise
    
    def find_intervention_boundaries(self, basin_map: DecisionBasinMap, 
                                   intervention_candidates: List[str]) -> Dict[str, DecisionBoundary]:
        """Find decision boundaries most relevant for interventions."""
        relevant_boundaries = {}
        
        for candidate_layer in intervention_candidates:
            # Find boundaries involving this layer
            layer_boundaries = [
                boundary for boundary in basin_map.decision_boundaries
                if boundary.layer_name == candidate_layer
            ]
            
            if layer_boundaries:
                # Select boundary with highest confidence and shortest distance
                best_boundary = min(
                    layer_boundaries,
                    key=lambda b: (b.distance_from_failure * (2.0 - b.confidence))
                )
                relevant_boundaries[candidate_layer] = best_boundary
        
        return relevant_boundaries
    
    def _analyze_layer_robustness(self, session: ReplaySession, 
                                layer_name: str, activation: torch.Tensor) -> RobustnessMetrics:
        """Analyze robustness properties of a specific layer."""
        try:
            # Convert to numpy for analysis
            activation_np = activation.detach().cpu().numpy()
            
            # Compute gradient magnitude (simplified - use activation magnitude as proxy)
            gradient_magnitude = float(np.linalg.norm(activation_np))
            
            # Estimate perturbation tolerance using binary search
            perturbation_tolerance = self._estimate_perturbation_tolerance(
                session, layer_name, activation
            )
            
            # Estimate local Lipschitz constant
            lipschitz_constant = self._estimate_lipschitz_constant(
                session, layer_name, activation
            )
            
            # Estimate basin volume (simplified - based on activation magnitude and tolerance)
            basin_volume = float(perturbation_tolerance ** len(activation_np.shape))
            
            # Find boundary distance using perturbation sampling
            boundary_distance = self._find_boundary_distance(
                session, layer_name, activation
            )
            
            # Compute confidence based on sampling consistency
            confidence = min(1.0, 1.0 - (boundary_distance / self.config.max_boundary_distance))
            
            return RobustnessMetrics(
                layer_name=layer_name,
                perturbation_tolerance=perturbation_tolerance,
                gradient_magnitude=gradient_magnitude,
                lipschitz_constant=lipschitz_constant,
                basin_volume=basin_volume,
                boundary_distance=boundary_distance,
                confidence=confidence
            )
            
        except Exception as e:
            logger.debug(f"Error analyzing robustness for {layer_name}: {e}")
            # Return default metrics on error
            return RobustnessMetrics(
                layer_name=layer_name,
                perturbation_tolerance=0.0,
                gradient_magnitude=0.0,
                lipschitz_constant=1.0,
                basin_volume=0.0,
                boundary_distance=self.config.max_boundary_distance,
                confidence=0.0
            )
    
    def _detect_decision_boundaries(self, session: ReplaySession, 
                                  layer_name: str, activation: torch.Tensor) -> List[DecisionBoundary]:
        """Detect decision boundaries around the failure point."""
        try:
            boundaries = []
            
            if self.config.boundary_detection_method == "gradient":
                boundaries = self._detect_gradient_boundaries(session, layer_name, activation)
            elif self.config.boundary_detection_method == "random":
                boundaries = self._detect_random_boundaries(session, layer_name, activation)
            elif self.config.boundary_detection_method == "adversarial":
                boundaries = self._detect_adversarial_boundaries(session, layer_name, activation)
            
            return boundaries
            
        except Exception as e:
            logger.debug(f"Error detecting boundaries for {layer_name}: {e}")
            return []
    
    def _detect_gradient_boundaries(self, session: ReplaySession, 
                                  layer_name: str, activation: torch.Tensor) -> List[DecisionBoundary]:
        """Detect boundaries using gradient-based methods."""
        boundaries = []
        
        try:
            activation_np = activation.detach().cpu().numpy()
            
            # Sample points around the activation
            boundary_points = []
            normal_vectors = []
            
            for _ in range(self.config.perturbation_steps):
                # Generate random perturbation direction
                perturbation_direction = np.random.randn(*activation_np.shape)
                perturbation_direction /= np.linalg.norm(perturbation_direction)
                
                # Binary search for boundary along this direction
                boundary_point = self._binary_search_boundary(
                    session, layer_name, activation_np, perturbation_direction
                )
                
                if boundary_point is not None:
                    boundary_points.append(boundary_point)
                    normal_vectors.append(perturbation_direction)
            
            if boundary_points:
                # Create boundary from collected points
                boundary = DecisionBoundary(
                    layer_name=layer_name,
                    boundary_points=boundary_points,
                    normal_vector=np.mean(normal_vectors, axis=0) if normal_vectors else np.zeros_like(activation_np),
                    distance_from_failure=float(np.mean([
                        np.linalg.norm(point - activation_np) for point in boundary_points
                    ])),
                    confidence=min(1.0, len(boundary_points) / self.config.perturbation_steps),
                    boundary_type="gradient"
                )
                boundaries.append(boundary)
            
        except Exception as e:
            logger.debug(f"Error in gradient boundary detection: {e}")
        
        return boundaries
    
    def _detect_random_boundaries(self, session: ReplaySession, 
                                layer_name: str, activation: torch.Tensor) -> List[DecisionBoundary]:
        """Detect boundaries using random sampling."""
        boundaries = []
        
        try:
            activation_np = activation.detach().cpu().numpy()
            boundary_points = []
            
            # Random sampling around the failure point
            for _ in range(self.config.robustness_samples):
                # Generate random perturbation
                perturbation = np.random.normal(
                    0, self.config.perturbation_magnitude, activation_np.shape
                )
                perturbed_point = activation_np + perturbation
                
                # Test if this point crosses a boundary
                if self._test_decision_boundary_crossing(session, layer_name, 
                                                       torch.from_numpy(perturbed_point.astype(np.float32))):
                    boundary_points.append(perturbed_point)
            
            if boundary_points:
                # Estimate normal vector from boundary points
                if len(boundary_points) >= 2:
                    # Use PCA to find normal direction
                    boundary_matrix = np.array(boundary_points) - activation_np
                    if boundary_matrix.shape[0] > 1:
                        U, s, Vt = np.linalg.svd(boundary_matrix, full_matrices=False)
                        normal_vector = Vt[0]  # First principal component
                    else:
                        normal_vector = boundary_matrix[0] / np.linalg.norm(boundary_matrix[0])
                else:
                    normal_vector = np.zeros_like(activation_np)
                
                boundary = DecisionBoundary(
                    layer_name=layer_name,
                    boundary_points=boundary_points,
                    normal_vector=normal_vector,
                    distance_from_failure=float(np.mean([
                        np.linalg.norm(point - activation_np) for point in boundary_points
                    ])),
                    confidence=min(1.0, len(boundary_points) / (self.config.robustness_samples * 0.1)),
                    boundary_type="random"
                )
                boundaries.append(boundary)
        
        except Exception as e:
            logger.debug(f"Error in random boundary detection: {e}")
        
        return boundaries
    
    def _detect_adversarial_boundaries(self, session: ReplaySession, 
                                     layer_name: str, activation: torch.Tensor) -> List[DecisionBoundary]:
        """Detect boundaries using adversarial perturbations."""
        # Simplified adversarial boundary detection
        # In practice, this would use more sophisticated adversarial methods
        return self._detect_gradient_boundaries(session, layer_name, activation)
    
    def _sample_perturbations(self, activation: torch.Tensor, layer_name: str) -> List[np.ndarray]:
        """Sample perturbations around the activation."""
        samples = []
        activation_np = activation.detach().cpu().numpy()
        
        for _ in range(min(self.config.robustness_samples, 50)):  # Limit samples for memory
            perturbation = np.random.normal(
                0, self.config.perturbation_magnitude, activation_np.shape
            )
            perturbed = activation_np + perturbation
            samples.append(perturbed)
        
        return samples
    
    def _estimate_perturbation_tolerance(self, session: ReplaySession, 
                                       layer_name: str, activation: torch.Tensor) -> float:
        """Estimate how much the activation can be perturbed before decision flip."""
        try:
            # Binary search for maximum perturbation magnitude
            min_magnitude = 0.0
            max_magnitude = 1.0
            
            for _ in range(10):  # Binary search iterations
                test_magnitude = (min_magnitude + max_magnitude) / 2.0
                
                # Test if perturbation of this magnitude causes flip
                if self._test_perturbation_flip(session, layer_name, activation, test_magnitude):
                    max_magnitude = test_magnitude
                else:
                    min_magnitude = test_magnitude
                
                if max_magnitude - min_magnitude < 0.01:
                    break
            
            return float((min_magnitude + max_magnitude) / 2.0)
            
        except Exception:
            return 0.0
    
    def _test_perturbation_flip(self, session: ReplaySession, 
                              layer_name: str, activation: torch.Tensor, magnitude: float) -> bool:
        """Test if perturbation of given magnitude causes decision flip."""
        try:
            # Generate random perturbation
            perturbation = torch.randn_like(activation) * magnitude
            perturbed_activation = activation + perturbation
            
            # Test decision boundary crossing (simplified)
            return self._test_decision_boundary_crossing(session, layer_name, perturbed_activation)
            
        except Exception:
            return True  # Assume flip on error
    
    def _test_decision_boundary_crossing(self, session: ReplaySession, 
                                       layer_name: str, activation: torch.Tensor) -> bool:
        """Test if activation crosses a decision boundary."""
        try:
            # Simplified boundary test - check if activation magnitude changes significantly
            original_magnitude = torch.norm(session.reconstructed_activations[layer_name])
            new_magnitude = torch.norm(activation)
            
            # Consider it a boundary crossing if magnitude change > threshold
            magnitude_change = abs(new_magnitude - original_magnitude) / (original_magnitude + 1e-8)
            return magnitude_change > self.config.perturbation_magnitude * 2.0
            
        except Exception:
            return False
    
    def _binary_search_boundary(self, session: ReplaySession, layer_name: str,
                              activation: np.ndarray, direction: np.ndarray) -> Optional[np.ndarray]:
        """Binary search for boundary point along a direction."""
        try:
            min_distance = 0.0
            max_distance = self.config.max_boundary_distance
            
            for _ in range(10):  # Binary search iterations
                test_distance = (min_distance + max_distance) / 2.0
                test_point = activation + direction * test_distance
                
                if self._test_decision_boundary_crossing(
                    session, layer_name, torch.from_numpy(test_point.astype(np.float32))
                ):
                    max_distance = test_distance
                else:
                    min_distance = test_distance
                
                if max_distance - min_distance < 0.01:
                    break
            
            boundary_distance = (min_distance + max_distance) / 2.0
            return activation + direction * boundary_distance
            
        except Exception:
            return None
    
    def _estimate_lipschitz_constant(self, session: ReplaySession, 
                                   layer_name: str, activation: torch.Tensor) -> float:
        """Estimate local Lipschitz constant."""
        try:
            # Sample nearby points and compute Lipschitz estimate
            activation_np = activation.detach().cpu().numpy()
            lipschitz_estimates = []
            
            for _ in range(10):
                # Sample two nearby points
                perturbation1 = np.random.normal(0, 0.01, activation_np.shape)
                perturbation2 = np.random.normal(0, 0.01, activation_np.shape)
                
                point1 = activation_np + perturbation1
                point2 = activation_np + perturbation2
                
                # Estimate function values (using activation magnitude as proxy)
                f1 = np.linalg.norm(point1)
                f2 = np.linalg.norm(point2)
                
                input_distance = np.linalg.norm(perturbation1 - perturbation2)
                output_distance = abs(f1 - f2)
                
                if input_distance > 1e-8:
                    lipschitz_estimates.append(output_distance / input_distance)
            
            return float(np.mean(lipschitz_estimates)) if lipschitz_estimates else 1.0
            
        except Exception:
            return 1.0
    
    def _find_boundary_distance(self, session: ReplaySession, 
                              layer_name: str, activation: torch.Tensor) -> float:
        """Find distance to nearest decision boundary."""
        try:
            min_distance = self.config.max_boundary_distance
            activation_np = activation.detach().cpu().numpy()
            
            # Sample in multiple directions to find closest boundary
            for _ in range(20):
                direction = np.random.randn(*activation_np.shape)
                direction /= np.linalg.norm(direction)
                
                # Binary search for boundary in this direction
                boundary_point = self._binary_search_boundary(session, layer_name, activation_np, direction)
                
                if boundary_point is not None:
                    distance = np.linalg.norm(boundary_point - activation_np)
                    min_distance = min(min_distance, distance)
            
            return float(min_distance)
            
        except Exception:
            return self.config.max_boundary_distance
    
    def _estimate_basin_volume(self, robustness_metrics: List[RobustnessMetrics]) -> float:
        """Estimate total volume of decision basin."""
        if not robustness_metrics:
            return 0.0
        
        try:
            # Geometric mean of individual basin volumes
            volumes = [max(m.basin_volume, 1e-10) for m in robustness_metrics]
            return float(np.exp(np.mean(np.log(volumes))))
        except Exception:
            return 0.0
    
    def _estimate_reversible_region_size(self, robustness_metrics: List[RobustnessMetrics]) -> float:
        """Estimate size of region where perturbations are reversible."""
        if not robustness_metrics:
            return 0.0
        
        try:
            # Based on perturbation tolerance and reversibility threshold
            reversible_tolerances = [
                m.perturbation_tolerance * self.config.reversibility_threshold
                for m in robustness_metrics
            ]
            return float(np.mean(reversible_tolerances))
        except Exception:
            return 0.0
    
    def _aggregate_baseline_robustness(self, all_robustness: Dict[str, List[RobustnessMetrics]]) -> Dict[str, RobustnessMetrics]:
        """Aggregate robustness metrics across baseline sessions."""
        baseline_robustness = {}
        
        for layer_name, metrics_list in all_robustness.items():
            if not metrics_list:
                continue
            
            # Average metrics across baseline sessions
            avg_tolerance = np.mean([m.perturbation_tolerance for m in metrics_list])
            avg_gradient = np.mean([m.gradient_magnitude for m in metrics_list])
            avg_lipschitz = np.mean([m.lipschitz_constant for m in metrics_list])
            avg_volume = np.mean([m.basin_volume for m in metrics_list])
            avg_distance = np.mean([m.boundary_distance for m in metrics_list])
            avg_confidence = np.mean([m.confidence for m in metrics_list])
            
            baseline_robustness[layer_name] = RobustnessMetrics(
                layer_name=layer_name,
                perturbation_tolerance=avg_tolerance,
                gradient_magnitude=avg_gradient,
                lipschitz_constant=avg_lipschitz,
                basin_volume=avg_volume,
                boundary_distance=avg_distance,
                confidence=avg_confidence
            )
        
        return baseline_robustness
    
    def _get_baseline_critical_layers(self) -> List[str]:
        """Get critical layers from baseline."""
        if not self.baseline_robustness:
            return []
        
        return [
            layer for layer, metrics in self.baseline_robustness.items()
            if metrics.perturbation_tolerance < self.config.perturbation_magnitude
        ]
    
    def _get_baseline_basin_volume(self) -> float:
        """Get baseline basin volume estimate."""
        if not self.baseline_robustness:
            return 0.0
        
        volumes = [metrics.basin_volume for metrics in self.baseline_robustness.values()]
        return np.mean(volumes) if volumes else 0.0
    
    def _validate_config(self) -> None:
        """Validate DBC configuration."""
        if not hasattr(self.config, 'boundary_detection_method'):
            raise ValueError("Config missing boundary_detection_method")
        if self.config.boundary_detection_method not in ["gradient", "random", "adversarial"]:
            raise ValueError("Invalid boundary_detection_method")
        if not (0.0 < self.config.perturbation_magnitude <= 1.0):
            raise ValueError("perturbation_magnitude must be between 0 and 1")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get DBC module statistics."""
        baseline_stats = {}
        if self.baseline_robustness:
            baseline_stats = {
                "layers": len(self.baseline_robustness),
                "avg_perturbation_tolerance": np.mean([
                    m.perturbation_tolerance for m in self.baseline_robustness.values()
                ]),
                "avg_basin_volume": self._get_baseline_basin_volume()
            }
        
        return {
            "baseline_robustness": baseline_stats,
            "boundary_cache_size": sum(len(boundaries) for boundaries in self.boundary_cache.values()),
            "config": {
                "boundary_detection_method": self.config.boundary_detection_method,
                "perturbation_magnitude": self.config.perturbation_magnitude,
                "robustness_samples": self.config.robustness_samples
            }
        }