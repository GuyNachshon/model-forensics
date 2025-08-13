# Model Forensics

**Root Cause Analysis (RCA) framework for foundation model failures**

A comprehensive toolkit for post-hoc causal analysis of specific incidents in frozen models without requiring retraining. Built for researchers and practitioners who need to understand *why* a specific model failure occurred.

## Features

- **Model-Agnostic Recording**: Drop-in SDK that works with any PyTorch model
- **Deterministic Replay**: Exactly reproduce incidents for analysis
- **Compression Forensics (CF)**: Detect anomalous computation patterns
- **Causal Minimal Fix Sets (CCA)**: Find smallest interventions that correct failures
- **Progressive Analysis**: CF → CCA pipeline optimizes search efficiency
- **Comprehensive Evaluation**: Statistical validation and side-effect measurement

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/guynachshon/model-forensics.git
cd model-forensics

# Install in development mode
pip install -e ".[dev]"
```

### Basic Usage

```python
from model_forensics import RecorderHooks, SandboxReplayer, CompressionForensics, CausalCCA

# 1. Record a failure incident
recorder = RecorderHooks(config="configs/default.yaml")
model = load_your_model()
recorder.install_hooks(model)

# Run the failing case
inputs = {"input_ids": failing_prompt_tokens}
outputs = model(**inputs)
bundle = recorder.create_bundle("incident_20240812")

# 2. Analyze the failure
replayer = SandboxReplayer()
session = replayer.load_bundle(bundle)

# Run Compression Forensics to find anomalies
cf = CompressionForensics()
anomalies = cf.analyze(session)

# Run Causal analysis to find minimal fix
cca = CausalCCA() 
fix_set = cca.search_fix_set(session, guided_by=anomalies)

print(f"Found fix set with {len(fix_set.interventions)} interventions")
print(f"Flip rate: {fix_set.total_flip_rate:.2%}")
```

## Architecture

The system consists of three main components:

### 1. Recorder SDK
- **Hooks**: Capture activations during model execution
- **KV Sketcher**: Compress and sample critical activation data
- **Bundle Exporter**: Create self-contained incident packages

### 2. Sandbox Replayer  
- **Deterministic Execution**: Replay incidents with bit-identical results
- **Intervention Engine**: Apply patches to activations mid-replay
- **Mock Environment**: Isolated execution with mocked external dependencies

### 3. RCA Modules
- **CF (Compression Forensics)**: Anomaly detection via compressibility analysis
- **CCA (Causal Minimal Fix Sets)**: Find smallest intervention sets
- **IS-C (Interaction Spectroscopy)**: Multi-feature causal interactions [Future]
- **DBC (Decision Basin Cartography)**: Decision boundary mapping [Future]
- **Provenance**: Training-time attribution [Future]

## Test Cases

The framework includes several test scenarios of increasing complexity:

### Starter Cases (MVP)
1. **GPT-2 Prompt Injection**: Model ignores instructions and outputs "HACKED"
2. **BERT Sentiment Bypass**: Adversarial text flips sentiment classification
3. **GPT-2 Toxic Generation**: Subtle prompts trigger harmful content

### Advanced Cases (Future)
4. **Multi-turn Deception**: Model lies across conversation turns
5. **Tool Misuse**: Code generation creates malicious functions
6. **Replit Agent Recreation**: Full multi-step failure with DB operations

## Configuration

The system is highly configurable via YAML files:

```yaml
recorder:
  enabled: true
  granularity: "layer"
  compression_ratio: 0.1
  max_bundle_size_mb: 500

modules:
  cf:
    enabled: true
    anomaly_threshold: 2.0
    baseline_samples: 1000
  
  cca:
    enabled: true
    max_fix_set_size: 50
    search_strategy: "greedy"
    early_stop_threshold: 0.8
```

## Development

### Setting up Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run type checking
mypy src/

# Format code
black src/ tests/
isort src/ tests/
```

### Project Structure

```
model-forensics/
├── src/
│   ├── core/           # Shared infrastructure
│   ├── recorder/       # Recording SDK
│   ├── replayer/       # Sandbox replayer
│   ├── modules/        # RCA analysis modules
│   └── scenarios/      # Test failure cases
├── tests/             # Test suite
├── configs/           # Configuration files
├── examples/          # Usage examples
└── docs/             # Documentation
```

## Research Background

This framework implements the research proposal described in [RESEARCH.md](RESEARCH.md), focusing on:

- **Causal Analysis**: Moving beyond correlation to causal sufficiency and necessity
- **Minimality**: Finding the smallest intervention sets that fix failures
- **Anomaly Detection**: Using compression to detect irregular computation patterns
- **Statistical Validation**: Rigorous testing of causal claims
- **Scalability**: Progressive narrowing to handle large models efficiently

## Contributing

We welcome contributions! Please see our [contribution guidelines](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{model_forensics,
  title={Model Forensics: Root Cause Analysis for Foundation Model Failures},
  author={Guy Nachshon},
  year={2024},
  url={https://github.com/guynachshon/model-forensics}
}
```