# CF Module API Reference

## Overview

The Compression Forensics (CF) module detects anomalous computation patterns by analyzing the compressibility of activation traces. It serves as the first phase of the RCA pipeline, identifying layers with suspicious behavior that warrant deeper causal analysis.

## Class: `CompressionForensics`

### Constructor

```python
CompressionForensics(config: Optional[CFConfig] = None)
```

**Parameters:**
- `config` (CFConfig, optional): Configuration object. Defaults to `CFConfig()`.

**Raises:**
- `ValueError`: If configuration is invalid (e.g., threshold out of range)

**Example:**
```python
from src.modules.cf import CompressionForensics
from src.core.config import CFConfig

config = CFConfig()
config.anomaly_threshold = 0.7
cf = CompressionForensics(config)
```

### Methods

#### `analyze_bundle(bundle: IncidentBundle) -> List[CompressionMetrics]`

Analyzes compression patterns for all activations in an incident bundle.

**Parameters:**
- `bundle` (IncidentBundle): The incident bundle to analyze

**Returns:**
- `List[CompressionMetrics]`: List of compression metrics for each layer

**Raises:**
- `ValueError`: If bundle is None or invalid
- `Exception`: If reconstruction fails

**Example:**
```python
# Load bundle and analyze
bundle = replayer.load_bundle("incident_001")
metrics = cf.analyze_bundle(bundle)

for metric in metrics:
    print(f"{metric.layer_name}: anomaly={metric.anomaly_score:.3f}")
```

#### `analyze_layer(activation: torch.Tensor, layer_name: str) -> CompressionMetrics`

Analyzes compression patterns for a single layer activation.

**Parameters:**
- `activation` (torch.Tensor): The activation tensor to analyze
- `layer_name` (str): Name of the layer

**Returns:**
- `CompressionMetrics`: Compression analysis results

**Raises:**
- `ValueError`: If activation is None, empty, or layer_name is invalid
- `Exception`: If analysis fails (returns default metrics)

**Example:**
```python
import torch

activation = torch.randn(10, 768)  # Sample activation
metrics = cf.analyze_layer(activation, "transformer.h.0")

print(f"Compression ratio: {metrics.compression_ratio:.3f}")
print(f"Entropy: {metrics.entropy:.3f}")
print(f"Is anomalous: {metrics.is_anomalous}")
```

#### `build_baseline(bundles: List[IncidentBundle]) -> None`

Builds compression baseline statistics from benign incident bundles.

**Parameters:**
- `bundles` (List[IncidentBundle]): List of benign bundles for baseline

**Example:**
```python
# Load benign cases
benign_bundles = [
    replayer.load_bundle("benign_001"),
    replayer.load_bundle("benign_002"),
    replayer.load_bundle("benign_003")
]

# Build baseline
cf.build_baseline(benign_bundles)
print("Baseline built from", len(benign_bundles), "cases")
```

#### `prioritize_layers(metrics: List[CompressionMetrics]) -> List[str]`

Returns layer names prioritized by anomaly score and confidence.

**Parameters:**
- `metrics` (List[CompressionMetrics]): Compression metrics from analysis

**Returns:**
- `List[str]`: Layer names sorted by priority (most anomalous first)

**Example:**
```python
metrics = cf.analyze_bundle(incident_bundle)
priority_layers = cf.prioritize_layers(metrics)

print(f"Top priority layers: {priority_layers[:3]}")
```

#### `get_stats() -> Dict[str, Any]`

Returns current CF module statistics.

**Returns:**
- `Dict[str, Any]`: Dictionary containing:
  - `baseline_layers`: Number of layers in baseline
  - `cache_size`: Size of compression cache
  - `config`: Configuration parameters

**Example:**
```python
stats = cf.get_stats()
print(f"Baseline has {stats['baseline_layers']} layers")
print(f"Cache contains {stats['cache_size']} entries")
```

## Data Types

### `CompressionMetrics`

Results from compression analysis of a single layer.

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
metric = cf.analyze_layer(activation, "layer_1")
print(f"""
Layer: {metric.layer_name}
Compression: {metric.compression_ratio:.3f}
Entropy: {metric.entropy:.3f}
Anomaly Score: {metric.anomaly_score:.3f}
Anomalous: {metric.is_anomalous}
Confidence: {metric.confidence:.3f}
""")
```

### `CFConfig`

Configuration for CF module.

**Fields:**
- `anomaly_threshold` (float): Threshold for anomaly detection [0, 1]
- `compression_methods` (List[str]): Methods to use ["gzip", "zlib"]
- `baseline_samples` (int): Number of baseline samples to use

**Example:**
```python
from src.core.config import CFConfig

config = CFConfig()
config.anomaly_threshold = 0.8  # Higher threshold
config.compression_methods = ["gzip", "zlib"]
config.baseline_samples = 15

cf = CompressionForensics(config)
```

## Internal Methods

The following methods are internal implementation details:

- `_compute_compression_ratio(data: np.ndarray) -> float`
- `_compute_entropy(data: np.ndarray) -> float`
- `_compare_to_baseline(layer_name: str, ratio: float) -> float`
- `_compute_anomaly_score(ratio: float, entropy: float, baseline: float) -> float`
- `_reconstruct_activations(bundle: IncidentBundle) -> ActivationDict`
- `_validate_config() -> None`

## Error Handling

The CF module includes comprehensive error handling:

1. **Input Validation**: All public methods validate inputs and raise `ValueError` for invalid parameters
2. **Configuration Validation**: Constructor validates config on initialization
3. **Graceful Degradation**: Analysis failures return default metrics rather than crashing
4. **Non-finite Values**: NaN and infinite values are automatically cleaned

## Performance Notes

- **Caching**: Compression ratios are cached to avoid recomputation
- **Baseline**: Building baseline is expensive but only done once
- **Memory**: Large activations are processed in chunks
- **Parallel**: Multiple compression methods can be computed in parallel

## Integration Example

```python
from src.core.config import RCAConfig
from src.modules.cf import CompressionForensics
from src.replayer.core import Replayer

# Setup
config = RCAConfig()
cf = CompressionForensics(config.modules.cf)
replayer = Replayer(config)

# Load data
incident = replayer.load_bundle("injection_hack_001")
baselines = [replayer.load_bundle(f"benign_{i:03d}") for i in range(5)]

# Analysis workflow
cf.build_baseline(baselines)
metrics = cf.analyze_bundle(incident)
priority_layers = cf.prioritize_layers(metrics)

# Results
anomalous_layers = [m for m in metrics if m.is_anomalous]
print(f"Found {len(anomalous_layers)} anomalous layers")
print(f"Priority layers: {priority_layers[:3]}")
```