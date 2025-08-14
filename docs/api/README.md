# Model Forensics API Documentation

## Overview

The Model Forensics framework provides a comprehensive API for Root Cause Analysis (RCA) of foundation model failures. The API is organized into several key modules that work together to analyze incidents, detect anomalies, and find minimal intervention sets.

## Core Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CF Module     │    │   CCA Module    │    │   Pipeline      │
│ (Compression    │───▶│  (Causal       │───▶│   (Integration) │
│  Forensics)     │    │   Analysis)     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Replayer      │    │   Recorder      │    │   CLI Tools     │
│  (Reconstruction│    │  (Capture)      │    │  (User Interface│
│   & Replay)     │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Module Overview

### 1. **CF (Compression Forensics)** - [`src/modules/cf.py`](../../src/modules/cf.py)
Detects anomalous computation patterns via compressibility analysis.

### 2. **CCA (Causal Minimal Fix Sets)** - [`src/modules/cca.py`](../../src/modules/cca.py)
Finds minimal sets of interventions that correct model failures.

### 3. **Pipeline** - [`src/modules/pipeline.py`](../../src/modules/pipeline.py)
Orchestrates CF → CCA analysis workflow with batch processing and triage.

### 4. **Replayer** - [`src/replayer/core.py`](../../src/replayer/core.py)
Reconstructs and replays incidents with intervention capabilities.

### 5. **Recorder** - [`src/recorder/`](../../src/recorder/)
Captures model execution traces during incidents.

### 6. **CLI** - [`src/cli.py`](../../src/cli.py)
Command-line interface for analysis and batch operations.

## Quick Start

```python
from src.core.config import RCAConfig
from src.modules.pipeline import RCAPipeline
from src.replayer.core import Replayer
from transformers import GPT2LMHeadModel

# Initialize components
config = RCAConfig()
pipeline = RCAPipeline(config)
replayer = Replayer(config)
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load incident bundle
incident_bundle = replayer.load_bundle("path/to/incident")

# Analyze incident
result = pipeline.analyze_incident(incident_bundle, model)

# Access results
print(f"Confidence: {result.confidence:.3f}")
if result.fix_set:
    print(f"Fix set with {len(result.fix_set.interventions)} interventions")
```

## API Reference

- [CF Module API](cf_module.md)
- [CCA Module API](cca_module.md)
- [Pipeline API](pipeline.md)
- [Replayer API](replayer.md)
- [Core Types](types.md)
- [Configuration](configuration.md)
- [CLI Commands](cli.md)

## Configuration

The framework uses YAML-based configuration with modular settings for each component:

```yaml
modules:
  cf:
    anomaly_threshold: 0.7
    compression_methods: ["gzip", "zlib"]
    baseline_samples: 10
  cca:
    max_fix_set_size: 5
    search_strategy: "greedy"
    early_stop_threshold: 0.8
    intervention_types: ["zero", "mean", "patch"]
```

## Error Handling

All API methods include comprehensive error handling:
- **Input validation** with clear error messages
- **Configuration validation** on initialization
- **Graceful degradation** when components fail
- **Detailed logging** for debugging

## Performance Considerations

- **Progressive narrowing**: CF → CCA workflow optimizes computational efficiency
- **Batch processing**: Analyze multiple incidents with shared baselines
- **Early stopping**: Configurable thresholds prevent exhaustive search
- **Caching**: Compression ratios and other computations are cached

## Examples

See the [`examples/`](../../examples/) directory for complete usage examples:
- [Basic Recording](../../examples/basic_recording.py)
- [Advanced Analysis](../../examples/advanced_recording.py)
- [Real-world Forensics](../../examples/real_world_forensics.py)