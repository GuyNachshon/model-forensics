# Core Types API Reference

## Overview

The Model Forensics framework uses a comprehensive type system to ensure type safety and clear interfaces. All core types are defined in `src/core/types.py` and used throughout the framework.

## Bundle Types

### `IncidentBundle`

Represents a complete incident recording with all necessary data for analysis.

**Fields:**
- `bundle_id` (str): Unique identifier for the bundle
- `trace_data` (TraceData): Captured execution traces
- `metadata` (BundleMetadata): Bundle metadata and configuration
- `created_at` (datetime): Bundle creation timestamp
- `format_version` (str): Bundle format version

**Example:**
```python
from src.replayer.core import Replayer

replayer = Replayer(config)
bundle = replayer.load_bundle("incident_001")

print(f"Bundle ID: {bundle.bundle_id}")
print(f"Created: {bundle.created_at}")
print(f"Format: {bundle.format_version}")
print(f"Inputs: {len(bundle.trace_data.inputs)} sequences")
```

### `TraceData`

Contains the actual captured traces from model execution.

**Fields:**
- `inputs` (List[str]): Input sequences that caused the failure
- `outputs` (List[str]): Model outputs that exhibited the failure
- `activations` (Dict[str, Any]): Layer activations (compressed)
- `external_calls` (List[ExternalCall]): External API/DB calls made
- `state_diffs` (List[StateDiff]): State changes during execution
- `performance_metrics` (Dict[str, float]): Performance measurements

**Example:**
```python
trace = bundle.trace_data
print(f"Captured {len(trace.activations)} layer activations")
print(f"Input: {trace.inputs[0]}")
print(f"Output: {trace.outputs[0]}")
```

### `BundleMetadata`

Metadata about the bundle and capture conditions.

**Fields:**
- `model_config` (Dict[str, Any]): Model configuration
- `recorder_config` (Dict[str, Any]): Recorder settings
- `environment` (Dict[str, str]): Environment information
- `checksums` (Dict[str, str]): File integrity checksums
- `compression_info` (Dict[str, Any]): Compression metadata

## Analysis Result Types

### `CompressionMetrics`

Results from CF analysis of a single layer.

**Fields:**
- `layer_name` (str): Name of the analyzed layer
- `compression_ratio` (float): Compression ratio (higher = more compressible)
- `entropy` (float): Shannon entropy of activations
- `anomaly_score` (float): Computed anomaly score [0, 1]
- `baseline_comparison` (float): Deviation from baseline (signed)
- `is_anomalous` (bool): Whether layer exceeds anomaly threshold
- `confidence` (float): Confidence in the analysis [0, 1]

**Example:**
```python
# From CF analysis
metrics = cf.analyze_bundle(bundle)
for metric in metrics:
    if metric.is_anomalous:
        print(f"Anomalous layer: {metric.layer_name}")
        print(f"  Anomaly score: {metric.anomaly_score:.3f}")
        print(f"  Confidence: {metric.confidence:.3f}")
```

### `CausalResult`

Complete results from RCA pipeline analysis.

**Fields:**
- `fix_set` (FixSet, optional): Best fix set found
- `compression_metrics` (List[CompressionMetrics]): CF analysis results
- `interaction_graph` (Any, optional): Interaction analysis (future)
- `decision_basin_map` (Any, optional): Decision boundaries (future)
- `provenance_info` (Any, optional): Provenance tracking (future)
- `execution_time` (float): Analysis duration in seconds
- `confidence` (float): Overall confidence [0, 1]

**Example:**
```python
result = pipeline.analyze_incident(incident, model)

print(f"Analysis Results:")
print(f"  Time: {result.execution_time:.2f}s")
print(f"  Confidence: {result.confidence:.3f}")
print(f"  Fix found: {result.fix_set is not None}")
```

## Intervention Types

### `Intervention`

Specification for a model intervention.

**Fields:**
- `type` (InterventionType): Type of intervention
- `layer_name` (str): Target layer name
- `value` (float, optional): Value for ZERO/MEAN interventions
- `donor_activation` (torch.Tensor, optional): Activation for PATCH interventions
- `indices` (List[int], optional): Specific indices to intervene on
- `metadata` (Dict[str, Any]): Additional intervention metadata

**Example:**
```python
from src.core.types import Intervention, InterventionType

# Zero intervention
zero_interv = Intervention(
    type=InterventionType.ZERO,
    layer_name="transformer.h.5.mlp",
    value=0.0
)

# Patch intervention
patch_interv = Intervention(
    type=InterventionType.PATCH,
    layer_name="transformer.h.3.attn",
    donor_activation=benign_activation
)
```

### `InterventionType`

Enumeration of intervention types.

**Values:**
- `ZERO`: Set activations to zero
- `MEAN`: Set activations to layer mean
- `PATCH`: Replace with benign activations
- `NOISE`: Add controlled noise
- `CLAMP`: Clamp values to range
- `ABLATE`: Remove specific components

**Example:**
```python
from src.core.types import InterventionType

# Check intervention type
if intervention.type == InterventionType.ZERO:
    print("Zero intervention")
elif intervention.type == InterventionType.PATCH:
    print("Patch intervention")
```

### `InterventionResult`

Result of applying a single intervention.

**Fields:**
- `intervention` (Intervention): The applied intervention
- `original_activation` (torch.Tensor): Original layer activation
- `modified_activation` (torch.Tensor): Modified activation after intervention
- `original_output` (Any): Original model output
- `modified_output` (Any): Modified model output
- `flip_success` (bool): Whether intervention changed the failure outcome
- `confidence` (float): Confidence in the result [0, 1]
- `side_effects` (Dict[str, float]): Measured side effects
- `metadata` (Dict[str, Any]): Additional result metadata

**Example:**
```python
# Apply intervention and check result
result = intervention_engine.apply_intervention(
    model, activations, intervention, inputs
)

print(f"Intervention on {result.intervention.layer_name}")
print(f"Success: {result.flip_success}")
print(f"Confidence: {result.confidence:.3f}")

if result.side_effects:
    print(f"Side effects: {result.side_effects}")
```

### `FixSet`

A minimal set of interventions that corrects the failure.

**Fields:**
- `interventions` (List[InterventionResult]): List of intervention results
- `sufficiency_score` (float): How well the set fixes the problem [0, 1]
- `necessity_scores` (List[float]): Necessity of each intervention [0, 1]
- `minimality_rank` (int): Rank by size (1 = most minimal)
- `total_flip_rate` (float): Overall success rate [0, 1]
- `avg_side_effects` (float): Average side effect severity [0, 1]

**Example:**
```python
fix_set = result.fix_set
if fix_set:
    print(f"Fix Set Analysis:")
    print(f"  Interventions: {len(fix_set.interventions)}")
    print(f"  Success Rate: {fix_set.total_flip_rate:.1%}")
    print(f"  Minimality Rank: {fix_set.minimality_rank}")
    
    # Individual interventions
    for i, interv_result in enumerate(fix_set.interventions):
        necessity = fix_set.necessity_scores[i]
        print(f"  {i+1}. {interv_result.intervention.layer_name}")
        print(f"      Necessity: {necessity:.3f}")
```

## Session Types

### `ReplaySession`

Context for replaying and analyzing an incident.

**Fields:**
- `session_id` (str): Unique session identifier
- `bundle` (IncidentBundle): The incident bundle being analyzed
- `model` (Any): The model instance for replay
- `reconstructed_activations` (ActivationDict): Reconstructed layer activations
- `baseline_activations` (ActivationDict, optional): Baseline activations
- `created_at` (datetime): Session creation timestamp
- `metadata` (Dict[str, Any]): Session metadata

**Example:**
```python
session = replayer.create_session(bundle, model)
print(f"Session {session.session_id}")
print(f"Bundle: {session.bundle.bundle_id}")
print(f"Activations: {len(session.reconstructed_activations)} layers")
```

## Utility Types

### `ActivationDict`

Type alias for activation dictionaries.

```python
ActivationDict = Dict[str, torch.Tensor]
```

**Usage:**
```python
# Function signature
def analyze_activations(activations: ActivationDict) -> List[str]:
    return list(activations.keys())

# Dictionary of layer name -> activation tensor
activations: ActivationDict = {
    "transformer.h.0.attn": torch.randn(1, 10, 768),
    "transformer.h.0.mlp": torch.randn(1, 10, 768),
    # ...
}
```

### `ExternalCall`

Record of external API or database calls.

**Fields:**
- `timestamp` (datetime): When the call was made
- `call_type` (str): Type of call (api, db, file, etc.)
- `target` (str): Target system or endpoint
- `request` (Dict[str, Any]): Request data
- `response` (Dict[str, Any]): Response data
- `duration_ms` (float): Call duration in milliseconds
- `success` (bool): Whether call succeeded

### `StateDiff`

Record of state changes during execution.

**Fields:**
- `timestamp` (datetime): When the change occurred
- `component` (str): Component that changed
- `old_state` (Dict[str, Any]): Previous state
- `new_state` (Dict[str, Any]): New state
- `change_type` (str): Type of change (create, update, delete)

## Type Validation

All types include validation methods:

```python
# Example validation
def validate_intervention(intervention: Intervention) -> bool:
    if not intervention.layer_name:
        raise ValueError("Layer name cannot be empty")
    
    if intervention.type == InterventionType.ZERO and intervention.value is None:
        raise ValueError("Zero intervention requires value")
        
    return True
```

## Serialization

Types support JSON serialization for persistence:

```python
import json
from src.core.types import InterventionResult

# Serialize
result_dict = result.to_dict()
json_str = json.dumps(result_dict, indent=2)

# Deserialize
loaded_dict = json.loads(json_str)
result = InterventionResult.from_dict(loaded_dict)
```

## Type Safety

The framework uses Python type hints throughout:

```python
from typing import List, Optional, Dict, Any
from src.core.types import IncidentBundle, CausalResult

def analyze_incidents(
    bundles: List[IncidentBundle],
    model: Any,
    baseline: Optional[List[IncidentBundle]] = None
) -> List[CausalResult]:
    """Type-safe incident analysis."""
    # Implementation with full type checking
    pass
```

## Creating Custom Types

Extend base types for custom use cases:

```python
from dataclasses import dataclass
from src.core.types import InterventionResult

@dataclass
class EnhancedInterventionResult(InterventionResult):
    """Extended result with additional metrics."""
    
    semantic_similarity: float
    perplexity_change: float
    attention_shift: Dict[str, float]
    
    def compute_composite_score(self) -> float:
        """Compute composite effectiveness score."""
        return (
            self.confidence * 0.4 +
            self.semantic_similarity * 0.3 +
            (1 - abs(self.perplexity_change)) * 0.3
        )
```

## Common Patterns

### Result Checking

```python
# Safe result access
result = pipeline.analyze_incident(incident, model)

if result.fix_set and result.confidence > 0.7:
    print("High-confidence fix found")
    for interv_result in result.fix_set.interventions:
        if interv_result.flip_success:
            print(f"  Effective: {interv_result.intervention.layer_name}")
```

### Batch Processing

```python
# Process multiple results
results: List[CausalResult] = pipeline.analyze_batch(incidents, model)

successful_fixes = [r for r in results if r.fix_set]
high_confidence = [r for r in results if r.confidence > 0.8]

print(f"Found {len(successful_fixes)} fixes")
print(f"High confidence: {len(high_confidence)} results")
```

### Error Handling

```python
from src.core.types import IncidentBundle

def safe_bundle_load(path: str) -> Optional[IncidentBundle]:
    try:
        bundle = replayer.load_bundle(path)
        # Validate bundle
        if not bundle.bundle_id:
            raise ValueError("Invalid bundle: missing ID")
        return bundle
    except Exception as e:
        print(f"Failed to load bundle {path}: {e}")
        return None
```