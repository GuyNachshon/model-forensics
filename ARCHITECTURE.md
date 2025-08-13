# Model Forensics Architecture

## System Overview

The Model Forensics framework consists of three main components:
1. **Recorder**: Captures model execution traces and context
2. **Replayer**: Deterministically reproduces incidents with intervention capabilities  
3. **RCA Engine**: Analyzes traces to identify root causes

```
┌─────────────────────────────────────────────────────────────┐
│                    Model Forensics System                   │
├─────────────────┬─────────────────┬─────────────────────────┤
│    Recorder     │    Replayer     │      RCA Engine         │
│                 │                 │                         │
│ ┌─────────────┐ │ ┌─────────────┐ │ ┌─────────┬───────────┐ │
│ │   Hooks     │ │ │  Sandbox    │ │ │   CF    │    CCA    │ │
│ │             │ │ │             │ │ │         │           │ │
│ │ - Forward   │ │ │ - Mocked    │ │ │ Anomaly │ Minimal   │ │
│ │ - Backward  │ │ │   APIs      │ │ │ Detect  │ Fix Set   │ │
│ │ - External  │ │ │ - State     │ │ │         │           │ │
│ │   Calls     │ │ │   Replay    │ │ └─────────┴───────────┘ │
│ └─────────────┘ │ └─────────────┘ │                         │
│                 │                 │                         │
│ ┌─────────────┐ │ ┌─────────────┐ │ ┌─────────┬───────────┐ │
│ │KV Sketcher  │ │ │ Patch API   │ │ │  IS-C   │    DBC    │ │
│ │             │ │ │             │ │ │         │           │ │
│ │ - Compress  │ │ │ - Activation│ │ │ Feature │ Decision  │ │
│ │   States    │ │ │   Override  │ │ │ Interact│ Basin     │ │
│ │ - Sample    │ │ │ - State     │ │ │         │ Mapping   │ │
│ │   Critical  │ │ │   Modify    │ │ │         │           │ │
│ └─────────────┘ │ └─────────────┘ │ └─────────┴───────────┘ │
│                 │                 │                         │
│ ┌─────────────┐ │                 │ ┌─────────────────────┐ │
│ │  Exporter   │ │                 │ │    Certificate      │ │
│ │             │ │                 │ │     Generator       │ │
│ │ - Bundle    │ │                 │ │                     │ │
│ │   Creation  │ │                 │ │ - Causal Chain      │ │
│ │ - Metadata  │ │                 │ │ - Interventions     │ │
│ │   Track     │ │                 │ │ - Confidence        │ │
│ └─────────────┘ │                 │ └─────────────────────┘ │
└─────────────────┴─────────────────┴─────────────────────────┘
```

## Core Components

### 1. Recorder SDK

**Purpose**: Drop-in instrumentation for any model/framework

#### Key Interfaces:
```python
class RecorderHooks:
    def on_forward(self, module, input, output)
    def on_backward(self, module, grad_input, grad_output) 
    def on_external_call(self, call_type, args, response)

class KVSketcher:
    def compress_activations(self, tensor: Tensor) -> CompressedSketch
    def sample_critical(self, activations: Dict) -> Dict

class BundleExporter:
    def create_bundle(self, trace_data: TraceData) -> IncidentBundle
```

#### Recording Flow:
1. **Hook Installation**: Automatic or manual hook registration
2. **Selective Capture**: Configure which layers/operations to record
3. **Compression**: Real-time activation compression and sampling
4. **Context Tracking**: External API calls, step IDs, policy context
5. **Bundle Export**: Self-contained incident packages

#### Bundle Structure:
```
incident_bundle_20240812_143022/
├── manifest.json          # Bundle metadata and checksums
├── inputs.jsonl          # Input sequence and prompts  
├── activations/          # Compressed activation traces
│   ├── layer_0.npz
│   ├── layer_12.npz
│   └── attention_heads.npz
├── external_calls.jsonl  # API calls, DB queries, tool usage
├── state_diffs.json      # State changes and side effects
├── model_config.json     # Model architecture and parameters
└── metrics.json          # Performance and metadata
```

### 2. Sandbox Replayer

**Purpose**: Deterministic incident reproduction with intervention capabilities

#### Key Interfaces:
```python  
class SandboxReplayer:
    def load_bundle(self, bundle_path: str) -> ReplaySession
    def replay(self, session: ReplaySession, interventions: Dict = None)
    def patch_activations(self, layer_name: str, new_values: Tensor)

class MockEngine:
    def mock_api_call(self, call_signature: str, response: Any)
    def mock_database(self, schema: Dict, responses: Dict)
    def mock_filesystem(self, file_tree: Dict)
```

#### Replay Flow:
1. **Bundle Loading**: Parse incident bundle and prepare environment
2. **Environment Setup**: Install mocks for external dependencies  
3. **State Restoration**: Restore exact model state and context
4. **Deterministic Execution**: Replay computation with identical results
5. **Intervention Testing**: Apply patches and measure effects

#### Intervention Types:
- **Activation Zeroing**: Zero out specific neurons/attention heads
- **Mean Replacement**: Replace activations with layer-wise means
- **Benign Patching**: Replace with activations from successful cases
- **Scaling**: Apply multiplicative scaling to activation magnitudes

### 3. RCA Engine Modules

#### 3.1 Compression Forensics (CF)
**Purpose**: Detect anomalous computation patterns via compressibility analysis

```python
class CompressionForensics:
    def analyze_layer(self, activations: Tensor) -> CompressionMetrics
    def detect_anomalies(self, baseline: Dict, incident: Dict) -> List[str]
    def prioritize_layers(self, anomaly_scores: Dict) -> List[str]
```

**Algorithm**:
1. Compute per-layer compression ratios (zlib, arithmetic coding)
2. Compare against benign baseline distributions
3. Flag layers with significant compression anomalies
4. Output prioritized list for targeted CCA analysis

#### 3.2 Causal Minimal Fix Set (CCA) 
**Purpose**: Find smallest intervention set that corrects the failure

```python
class CausalCCA:
    def search_fix_set(self, replay_session: ReplaySession) -> FixSet
    def test_sufficiency(self, fix_set: FixSet) -> CausalResult
    def measure_side_effects(self, fix_set: FixSet, benign_cases: List) -> float
```

**Algorithm**:
1. **Greedy Search**: Start with highest-impact activations (guided by CF)
2. **Sufficiency Testing**: Apply interventions via replayer, measure flip rate
3. **Necessity Testing**: Remove interventions one by one, verify still effective
4. **Side Effect Analysis**: Test fix set on benign cases, measure degradation

#### 3.3 Interaction Spectroscopy (IS-C)
**Purpose**: Detect multi-feature causal interactions

```python  
class InteractionSpectroscopy:
    def compute_shapley_interactions(self, features: List[str]) -> InteractionGraph
    def test_joint_necessity(self, feature_pairs: List[Tuple]) -> Dict
    def visualize_causal_graph(self, interactions: InteractionGraph) -> Plot
```

#### 3.4 Decision Basin Cartography (DBC)
**Purpose**: Map when model commits to faulty outputs

```python
class DecisionBasinCartography:
    def map_decision_boundary(self, replay_session: ReplaySession) -> BasinMap
    def find_commitment_point(self, basin_map: BasinMap) -> LayerIndex
    def test_reversibility(self, layer_idx: int, interventions: Dict) -> bool
```

## Data Flow Architecture

### Recording Phase:
```
Model Execution → Hooks → KV Sketcher → Compression → Bundle Export
     ↓              ↓           ↓            ↓            ↓
  Forward Pass → Activations → Samples → .npz files → manifest.json
  External API → Context → Metadata → .jsonl → Bundle/
```

### Analysis Phase:
```
Bundle → Replayer → RCA Modules → Certificate
   ↓        ↓           ↓            ↓
Load → Replay → CF/CCA/IS-C/DBC → Report
```

### Intervention Testing:
```
Original Execution → Record → Bundle
        ↓                        ↓
Failed Output              Replay + Patch → Success?
                                ↓              ↓
                          Intervention → Measure Effect
```

## Scalability & Performance

### Memory Management:
- **Streaming Compression**: Compress activations during forward pass
- **Selective Recording**: Configure recording granularity per use case
- **Chunk Processing**: Process large models in activation chunks
- **Memory Mapping**: Use mmap for large bundle files

### Compute Optimization:
- **Parallel Search**: Distribute CCA search across multiple processes  
- **GPU Acceleration**: Keep replayer computations on GPU when possible
- **Caching**: Cache compression baselines and intervention results
- **Early Stopping**: Halt search when confidence thresholds met

### Storage:
- **Bundle Compression**: Aggressive compression for inactive bundles
- **Deduplicate**: Share common activations across similar incidents
- **Retention Policy**: Automatic cleanup of old bundles
- **Distributed Storage**: Support for remote bundle storage

## Extension Points

### Custom Modules:
```python
class RCAModule(ABC):
    @abstractmethod
    def analyze(self, bundle: IncidentBundle) -> ModuleResult
    
    @abstractmethod  
    def integrate_with_pipeline(self, prior_results: Dict) -> None
```

### Custom Interventions:
```python
class InterventionStrategy(ABC):
    @abstractmethod
    def apply(self, activations: Tensor, target_layer: str) -> Tensor
    
    @abstractmethod
    def validate(self, original: Tensor, modified: Tensor) -> bool
```

### Framework Adapters:
```python
class ModelAdapter(ABC):
    @abstractmethod
    def install_hooks(self, model: Any) -> None
    
    @abstractmethod
    def extract_activations(self, module_name: str) -> Tensor
```

## Security & Privacy

### Bundle Security:
- **Encryption**: Encrypt sensitive bundles at rest
- **Access Control**: Role-based bundle access
- **Sanitization**: Remove PII from recorded traces
- **Audit Logging**: Track bundle access and modifications

### Replay Isolation:
- **Sandboxed Execution**: Isolated replay environment
- **Network Isolation**: Block external network access during replay
- **Resource Limits**: CPU/memory limits for replay processes
- **Clean Teardown**: Guaranteed cleanup of replay artifacts

## Configuration Management

### Recording Configuration:
```yaml
recorder:
  enabled: true
  granularity: layer  # layer, attention_head, neuron
  compression_ratio: 0.1
  external_calls: true
  max_bundle_size_mb: 500

models:
  gpt2:
    layers_to_record: [0, 6, 11]  
    attention_heads: all
  bert:
    layers_to_record: all
    pooler: false
```

### Analysis Configuration:  
```yaml
rca:
  modules_enabled: [cf, cca]
  timeout_minutes: 120
  
cf:
  anomaly_threshold: 2.0
  baseline_samples: 1000
  
cca:
  max_fix_set_size: 50
  search_strategy: greedy
  side_effect_threshold: 0.05
```