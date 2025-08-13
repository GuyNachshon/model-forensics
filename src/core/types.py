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
class CausalResult:
    """Result of causal analysis."""
    fix_set: FixSet
    compression_metrics: List[CompressionMetrics]
    interaction_graph: Optional[Dict[str, Any]]
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