"""Core type definitions for model forensics."""

from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from enum import Enum


class InterventionType(Enum):
    """Types of interventions that can be applied."""
    ZERO = "zero"  # Zero out activations
    MEAN = "mean"  # Replace with mean activation
    PATCH = "patch"  # Replace with benign donor
    NOISE = "noise"  # Add noise
    SCALE = "scale"  # Scale activations
    CLAMP = "clamp"  # Clamp to range


@dataclass
class CompressedSketch:
    """Compressed representation of activation data."""
    data: bytes
    original_shape: Tuple[int, ...]
    dtype: str
    compression_ratio: float
    metadata: Dict[str, Any]


@dataclass
class TraceData:
    """Complete execution trace for a model run."""
    inputs: Dict[str, Any]
    activations: Dict[str, CompressedSketch]
    outputs: Dict[str, Any]
    external_calls: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: str
    model_config: Dict[str, Any]


@dataclass
class IncidentBundle:
    """Self-contained package for incident analysis."""
    bundle_id: str
    manifest: Dict[str, Any]
    trace_data: TraceData
    bundle_path: Path
    
    @property
    def size_mb(self) -> float:
        """Bundle size in megabytes."""
        if self.bundle_path.exists():
            return sum(f.stat().st_size for f in self.bundle_path.rglob('*')) / (1024 * 1024)
        return 0.0


@dataclass
class Intervention:
    """Configuration for a single intervention."""
    type: InterventionType
    layer_name: str
    indices: Optional[List[int]] = None  # Specific indices to intervene on
    value: Optional[Any] = None  # Replacement value
    donor_activation: Optional[torch.Tensor] = None  # For patching
    scale_factor: Optional[float] = None  # For scaling
    clamp_range: Optional[Tuple[float, float]] = None  # For clamping
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class InterventionResult:
    """Result of applying an intervention."""
    intervention: Intervention
    original_activation: torch.Tensor
    modified_activation: torch.Tensor
    original_output: Any
    modified_output: Any
    flip_success: bool
    confidence: float
    side_effects: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class ReplaySession:
    """Session for replaying and analyzing bundles."""
    bundle: IncidentBundle
    model: torch.nn.Module
    reconstructed_activations: Dict[str, torch.Tensor]
    interventions: List[InterventionResult]
    session_id: str
    created_at: str
    metadata: Dict[str, Any]


@dataclass
class FixSet:
    """Minimal set of interventions that fix a failure."""
    interventions: List[InterventionResult]
    sufficiency_score: float
    necessity_scores: List[float]
    minimality_rank: int
    total_flip_rate: float
    avg_side_effects: float


@dataclass
class CompressionMetrics:
    """Compression analysis results for anomaly detection."""
    layer_name: str
    compression_ratio: float
    entropy: float
    anomaly_score: float
    baseline_comparison: float
    is_anomalous: bool
    confidence: float


@dataclass
class InteractionMetrics:
    """Metrics for interaction between two components."""
    source_layer: str
    target_layer: str
    correlation: float
    mutual_information: float
    causal_strength: float
    interaction_type: str  # "excitatory", "inhibitory", "neutral", "complex"
    temporal_lag: int
    confidence: float
    
    @property
    def interaction_strength(self) -> float:
        """Combined interaction strength score."""
        import numpy as np
        return np.sqrt(
            self.correlation**2 * 0.3 + 
            self.mutual_information * 0.4 + 
            self.causal_strength * 0.3
        )


@dataclass
class PropagationPath:
    """A path through which anomalies propagate."""
    layers: List[str]
    propagation_strengths: List[float]
    total_strength: float
    path_length: int
    critical_nodes: List[str]
    
    @property
    def is_critical_path(self) -> bool:
        """Whether this is a critical propagation path."""
        return self.total_strength > 0.7 and len(self.critical_nodes) > 0


@dataclass
class InteractionGraph:
    """Graph representation of component interactions."""
    nodes: List[str]
    edges: Dict[Tuple[str, str], InteractionMetrics]
    propagation_paths: List[PropagationPath]
    centrality_scores: Dict[str, float]
    community_structure: Dict[str, List[str]]
    
    def get_critical_interactions(self, threshold: float = 0.7) -> List[InteractionMetrics]:
        """Get interactions above strength threshold."""
        return [metrics for metrics in self.edges.values() 
                if metrics.interaction_strength >= threshold]
    
    def get_propagation_sources(self) -> List[str]:
        """Get layers that are sources of propagation."""
        sources = []
        for path in self.propagation_paths:
            if path.is_critical_path and path.layers:
                sources.append(path.layers[0])
        return list(set(sources))


@dataclass
class CausalResult:
    """Result of causal analysis."""
    fix_set: FixSet
    compression_metrics: List[CompressionMetrics]
    interaction_graph: Optional[InteractionGraph]  # Now properly typed
    decision_basin_map: Optional[Dict[str, Any]]
    provenance_info: Optional[Dict[str, Any]]
    execution_time: float
    confidence: float


# Type aliases for commonly used types
ActivationDict = Dict[str, torch.Tensor]
LayerName = str
InterventionConfig = Dict[str, Any]
ModelOutput = Union[torch.Tensor, Dict[str, torch.Tensor]]
InterventionHook = Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]