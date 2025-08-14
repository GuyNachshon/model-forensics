"""Interaction Spectroscopy (IS-C) module for analyzing component interactions."""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

from core.types import (
    IncidentBundle, ActivationDict, InteractionMetrics, 
    PropagationPath, InteractionGraph
)
from core.config import ISCConfig

logger = logging.getLogger(__name__)



class InteractionSpectroscopy:
    """Analyzes interaction patterns between model components."""
    
    def __init__(self, config: Optional[ISCConfig] = None):
        self.config = config or ISCConfig()
        self.baseline_interactions: Optional[InteractionGraph] = None
        self.interaction_cache: Dict[str, InteractionMetrics] = {}
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"Initialized IS-C with method: {self.config.analysis_method}")
    
    def analyze_interactions(self, bundle: IncidentBundle, 
                           activations: Optional[ActivationDict] = None) -> InteractionGraph:
        """Analyze interaction patterns in an incident bundle."""
        if not bundle:
            raise ValueError("Bundle cannot be None")
            
        logger.info(f"Analyzing interactions for bundle: {bundle.bundle_id}")
        
        try:
            # Reconstruct activations if not provided
            if activations is None:
                activations = self._reconstruct_activations(bundle)
            
            if not activations:
                logger.warning("No activations found for interaction analysis")
                return InteractionGraph(
                    nodes=[], edges={}, propagation_paths=[],
                    centrality_scores={}, community_structure={}
                )
            
            # Compute pairwise interactions
            interactions = self._compute_pairwise_interactions(activations)
            
            # Build interaction graph
            graph = self._build_interaction_graph(interactions, activations)
            
            # Analyze propagation paths
            graph.propagation_paths = self._analyze_propagation_paths(graph, activations)
            
            # Compute centrality scores
            graph.centrality_scores = self._compute_centrality_scores(graph)
            
            # Detect community structure
            graph.community_structure = self._detect_communities(graph)
            
            logger.info(f"Found {len(graph.edges)} interactions across {len(graph.nodes)} layers")
            return graph
            
        except Exception as e:
            logger.error(f"Error analyzing interactions: {e}")
            raise
    
    def compare_to_baseline(self, incident_graph: InteractionGraph) -> Dict[str, Any]:
        """Compare incident interactions to baseline patterns."""
        if not self.baseline_interactions:
            logger.warning("No baseline interactions available for comparison")
            return {"anomalous_interactions": [], "interaction_shifts": {}}
        
        try:
            anomalous_interactions = []
            interaction_shifts = {}
            
            # Find anomalous interactions
            for edge, metrics in incident_graph.edges.items():
                baseline_metrics = self.baseline_interactions.edges.get(edge)
                
                if baseline_metrics:
                    # Compute deviation from baseline
                    strength_diff = abs(metrics.interaction_strength - 
                                      baseline_metrics.interaction_strength)
                    
                    if strength_diff > self.config.anomaly_threshold:
                        anomalous_interactions.append({
                            "edge": edge,
                            "incident_strength": metrics.interaction_strength,
                            "baseline_strength": baseline_metrics.interaction_strength,
                            "deviation": strength_diff
                        })
                    
                    interaction_shifts[f"{edge[0]}->{edge[1]}"] = {
                        "strength_change": metrics.interaction_strength - baseline_metrics.interaction_strength,
                        "correlation_change": metrics.correlation - baseline_metrics.correlation,
                        "type_change": metrics.interaction_type != baseline_metrics.interaction_type
                    }
                else:
                    # New interaction not in baseline
                    if metrics.interaction_strength > self.config.novel_interaction_threshold:
                        anomalous_interactions.append({
                            "edge": edge,
                            "incident_strength": metrics.interaction_strength,
                            "baseline_strength": 0.0,
                            "deviation": metrics.interaction_strength,
                            "is_novel": True
                        })
            
            return {
                "anomalous_interactions": anomalous_interactions,
                "interaction_shifts": interaction_shifts,
                "num_anomalies": len(anomalous_interactions),
                "baseline_coverage": len(set(incident_graph.edges.keys()) & 
                                       set(self.baseline_interactions.edges.keys())) / 
                                     max(len(incident_graph.edges), 1)
            }
            
        except Exception as e:
            logger.error(f"Error comparing to baseline: {e}")
            return {"error": str(e)}
    
    def build_baseline(self, baseline_bundles: List[IncidentBundle]) -> None:
        """Build baseline interaction patterns from benign bundles."""
        if not baseline_bundles:
            raise ValueError("Baseline bundles cannot be empty")
            
        logger.info(f"Building interaction baseline from {len(baseline_bundles)} bundles")
        
        try:
            all_interactions = []
            
            # Analyze each baseline bundle
            for bundle in baseline_bundles:
                try:
                    activations = self._reconstruct_activations(bundle)
                    if activations:
                        interactions = self._compute_pairwise_interactions(activations)
                        all_interactions.append(interactions)
                except Exception as e:
                    logger.warning(f"Failed to analyze baseline bundle {bundle.bundle_id}: {e}")
                    continue
            
            if not all_interactions:
                raise ValueError("No valid baseline interactions found")
            
            # Aggregate baseline interactions
            self.baseline_interactions = self._aggregate_baseline_interactions(all_interactions)
            
            logger.info(f"Built baseline with {len(self.baseline_interactions.edges)} interactions")
            
        except Exception as e:
            logger.error(f"Error building baseline: {e}")
            raise
    
    def get_critical_pathways(self, graph: InteractionGraph, 
                            anomalous_layers: Optional[List[str]] = None) -> List[PropagationPath]:
        """Identify critical pathways involving anomalous layers."""
        if not anomalous_layers:
            # Use all nodes if no specific layers provided
            anomalous_layers = graph.nodes
        
        critical_paths = []
        
        for path in graph.propagation_paths:
            # Check if path involves anomalous layers
            path_involves_anomalous = any(layer in anomalous_layers for layer in path.layers)
            
            if path.is_critical_path and path_involves_anomalous:
                critical_paths.append(path)
        
        # Sort by total strength
        critical_paths.sort(key=lambda p: p.total_strength, reverse=True)
        
        return critical_paths[:self.config.max_critical_paths]
    
    def _compute_pairwise_interactions(self, activations: ActivationDict) -> Dict[Tuple[str, str], InteractionMetrics]:
        """Compute interactions between all pairs of layers."""
        interactions = {}
        layer_names = list(activations.keys())
        
        for i, source_layer in enumerate(layer_names):
            for target_layer in layer_names[i+1:]:  # Avoid duplicate pairs
                try:
                    source_activation = activations[source_layer]
                    target_activation = activations[target_layer]
                    
                    # Compute interaction metrics
                    metrics = self._compute_interaction_metrics(
                        source_activation, target_activation, 
                        source_layer, target_layer
                    )
                    
                    interactions[(source_layer, target_layer)] = metrics
                    
                except Exception as e:
                    logger.debug(f"Failed to compute interaction {source_layer}->{target_layer}: {e}")
        
        return interactions
    
    def _compute_interaction_metrics(self, source: torch.Tensor, target: torch.Tensor,
                                   source_name: str, target_name: str) -> InteractionMetrics:
        """Compute interaction metrics between two activations."""
        # Flatten activations for correlation analysis
        source_flat = source.detach().cpu().numpy().flatten()
        target_flat = target.detach().cpu().numpy().flatten()
        
        # Ensure same length by taking minimum
        min_len = min(len(source_flat), len(target_flat))
        source_flat = source_flat[:min_len]
        target_flat = target_flat[:min_len]
        
        # Correlation
        correlation = np.corrcoef(source_flat, target_flat)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # Mutual information (simplified)
        mutual_info = self._compute_mutual_information(source_flat, target_flat)
        
        # Causal strength (using Granger-like causality)
        causal_strength = self._compute_causal_strength(source_flat, target_flat)
        
        # Interaction type
        interaction_type = self._classify_interaction_type(correlation, mutual_info)
        
        # Temporal lag (simplified - assume 0 for now)
        temporal_lag = 0
        
        # Confidence based on signal strength
        confidence = min(1.0, abs(correlation) + mutual_info * 0.5 + causal_strength * 0.3)
        
        return InteractionMetrics(
            source_layer=source_name,
            target_layer=target_name,
            correlation=correlation,
            mutual_information=mutual_info,
            causal_strength=causal_strength,
            interaction_type=interaction_type,
            temporal_lag=temporal_lag,
            confidence=confidence
        )
    
    def _compute_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information between two signals."""
        try:
            # Discretize signals for MI computation
            x_discrete = np.digitize(x, np.percentile(x, [25, 50, 75]))
            y_discrete = np.digitize(y, np.percentile(y, [25, 50, 75]))
            
            # Compute joint and marginal histograms
            joint_hist, _, _ = np.histogram2d(x_discrete, y_discrete, bins=4)
            x_hist, _ = np.histogram(x_discrete, bins=4)
            y_hist, _ = np.histogram(y_discrete, bins=4)
            
            # Normalize to probabilities
            joint_prob = joint_hist / np.sum(joint_hist)
            x_prob = x_hist / np.sum(x_hist)
            y_prob = y_hist / np.sum(y_hist)
            
            # Compute mutual information
            mi = 0.0
            for i in range(len(x_prob)):
                for j in range(len(y_prob)):
                    if joint_prob[i, j] > 0 and x_prob[i] > 0 and y_prob[j] > 0:
                        mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (x_prob[i] * y_prob[j]))
            
            return max(0.0, mi)  # Ensure non-negative
            
        except Exception:
            return 0.0
    
    def _compute_causal_strength(self, source: np.ndarray, target: np.ndarray) -> float:
        """Compute causal strength using simplified Granger causality."""
        try:
            # Simple lag-based causality approximation
            if len(source) < 10:
                return 0.0
            
            # Compute lagged correlation
            lag_corr = np.corrcoef(source[:-1], target[1:])[0, 1]
            reverse_corr = np.corrcoef(target[:-1], source[1:])[0, 1]
            
            if np.isnan(lag_corr):
                lag_corr = 0.0
            if np.isnan(reverse_corr):
                reverse_corr = 0.0
            
            # Causal strength is forward correlation minus reverse
            return max(0.0, abs(lag_corr) - abs(reverse_corr))
            
        except Exception:
            return 0.0
    
    def _classify_interaction_type(self, correlation: float, mutual_info: float) -> str:
        """Classify the type of interaction."""
        if abs(correlation) < 0.1 and mutual_info < 0.1:
            return "neutral"
        elif correlation > 0.3:
            return "excitatory"
        elif correlation < -0.3:
            return "inhibitory"
        else:
            return "complex"
    
    def _build_interaction_graph(self, interactions: Dict[Tuple[str, str], InteractionMetrics],
                                activations: ActivationDict) -> InteractionGraph:
        """Build interaction graph from computed interactions."""
        # Filter significant interactions
        significant_interactions = {
            edge: metrics for edge, metrics in interactions.items()
            if metrics.interaction_strength >= self.config.min_interaction_strength
        }
        
        # Get all nodes
        nodes = list(activations.keys())
        
        return InteractionGraph(
            nodes=nodes,
            edges=significant_interactions,
            propagation_paths=[],  # Will be filled later
            centrality_scores={},  # Will be filled later
            community_structure={}  # Will be filled later
        )
    
    def _analyze_propagation_paths(self, graph: InteractionGraph, 
                                 activations: ActivationDict) -> List[PropagationPath]:
        """Analyze how anomalies propagate through the network."""
        paths = []
        
        # Simple path finding - look for chains of strong interactions
        visited = set()
        
        for start_node in graph.nodes:
            if start_node in visited:
                continue
                
            path = self._find_propagation_path(graph, start_node, visited)
            if len(path.layers) >= self.config.min_path_length:
                paths.append(path)
        
        return paths
    
    def _find_propagation_path(self, graph: InteractionGraph, start_node: str,
                             visited: set) -> PropagationPath:
        """Find a propagation path starting from a node."""
        path_layers = [start_node]
        path_strengths = []
        current_node = start_node
        visited.add(start_node)
        
        # Follow strongest outgoing interactions
        while len(path_layers) < self.config.max_path_length:
            best_next = None
            best_strength = 0.0
            
            for (source, target), metrics in graph.edges.items():
                if source == current_node and target not in visited:
                    if metrics.interaction_strength > best_strength:
                        best_next = target
                        best_strength = metrics.interaction_strength
                elif target == current_node and source not in visited:
                    # Reverse direction
                    if metrics.interaction_strength > best_strength:
                        best_next = source
                        best_strength = metrics.interaction_strength
            
            if best_next is None or best_strength < self.config.min_interaction_strength:
                break
            
            path_layers.append(best_next)
            path_strengths.append(best_strength)
            visited.add(best_next)
            current_node = best_next
        
        # Identify critical nodes (bottlenecks)
        critical_nodes = []
        for layer in path_layers:
            # Count connections for this layer
            connections = sum(1 for (s, t) in graph.edges.keys() 
                            if s == layer or t == layer)
            if connections <= 2:  # Bottleneck
                critical_nodes.append(layer)
        
        total_strength = np.mean(path_strengths) if path_strengths else 0.0
        
        return PropagationPath(
            layers=path_layers,
            propagation_strengths=path_strengths,
            total_strength=total_strength,
            path_length=len(path_layers),
            critical_nodes=critical_nodes
        )
    
    def _compute_centrality_scores(self, graph: InteractionGraph) -> Dict[str, float]:
        """Compute centrality scores for each node."""
        centrality = defaultdict(float)
        
        # Simple degree centrality
        for node in graph.nodes:
            degree = sum(1 for (s, t) in graph.edges.keys() if s == node or t == node)
            centrality[node] = degree
        
        # Normalize
        max_centrality = max(centrality.values()) if centrality else 1.0
        return {node: score / max_centrality for node, score in centrality.items()}
    
    def _detect_communities(self, graph: InteractionGraph) -> Dict[str, List[str]]:
        """Detect community structure in the interaction graph."""
        # Simple community detection based on interaction strength
        communities = defaultdict(list)
        community_id = 0
        assigned = set()
        
        for node in graph.nodes:
            if node in assigned:
                continue
                
            # Find strongly connected nodes
            community = [node]
            assigned.add(node)
            
            for (source, target), metrics in graph.edges.items():
                if metrics.interaction_strength > 0.7:  # Strong interaction
                    if source == node and target not in assigned:
                        community.append(target)
                        assigned.add(target)
                    elif target == node and source not in assigned:
                        community.append(source)
                        assigned.add(source)
            
            communities[f"community_{community_id}"] = community
            community_id += 1
        
        return dict(communities)
    
    def _aggregate_baseline_interactions(self, all_interactions: List[Dict]) -> InteractionGraph:
        """Aggregate interactions across baseline bundles."""
        # Collect all edges across baselines
        edge_metrics = defaultdict(list)
        all_nodes = set()
        
        for interactions in all_interactions:
            for edge, metrics in interactions.items():
                edge_metrics[edge].append(metrics)
                all_nodes.add(edge[0])
                all_nodes.add(edge[1])
        
        # Average metrics across baselines
        baseline_edges = {}
        for edge, metrics_list in edge_metrics.items():
            avg_correlation = np.mean([m.correlation for m in metrics_list])
            avg_mi = np.mean([m.mutual_information for m in metrics_list])
            avg_causal = np.mean([m.causal_strength for m in metrics_list])
            avg_confidence = np.mean([m.confidence for m in metrics_list])
            
            # Most common interaction type
            types = [m.interaction_type for m in metrics_list]
            most_common_type = max(set(types), key=types.count)
            
            baseline_edges[edge] = InteractionMetrics(
                source_layer=edge[0],
                target_layer=edge[1],
                correlation=avg_correlation,
                mutual_information=avg_mi,
                causal_strength=avg_causal,
                interaction_type=most_common_type,
                temporal_lag=0,
                confidence=avg_confidence
            )
        
        return InteractionGraph(
            nodes=list(all_nodes),
            edges=baseline_edges,
            propagation_paths=[],
            centrality_scores={},
            community_structure={}
        )
    
    def _reconstruct_activations(self, bundle: IncidentBundle) -> ActivationDict:
        """Reconstruct activations from bundle."""
        try:
            from replayer.reconstruction import ActivationReconstructor
            reconstructor = ActivationReconstructor()
            return reconstructor.reconstruct_bundle_activations(bundle)
        except Exception as e:
            logger.warning(f"Failed to reconstruct activations: {e}")
            return {}
    
    def _validate_config(self) -> None:
        """Validate IS-C configuration."""
        if not hasattr(self.config, 'analysis_method'):
            raise ValueError("Config missing analysis_method")
        if not hasattr(self.config, 'min_interaction_strength'):
            raise ValueError("Config missing min_interaction_strength")
        if not (0.0 <= self.config.min_interaction_strength <= 1.0):
            raise ValueError("min_interaction_strength must be between 0 and 1")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get IS-C module statistics."""
        baseline_stats = {}
        if self.baseline_interactions:
            baseline_stats = {
                "nodes": len(self.baseline_interactions.nodes),
                "edges": len(self.baseline_interactions.edges),
                "communities": len(self.baseline_interactions.community_structure)
            }
        
        return {
            "baseline_interactions": baseline_stats,
            "cache_size": len(self.interaction_cache),
            "config": {
                "analysis_method": self.config.analysis_method,
                "min_interaction_strength": self.config.min_interaction_strength,
                "anomaly_threshold": self.config.anomaly_threshold
            }
        }