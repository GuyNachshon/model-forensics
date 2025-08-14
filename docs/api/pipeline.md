# Pipeline API Reference

## Overview

The RCA Pipeline orchestrates the complete CF → CCA analysis workflow, providing high-level interfaces for incident analysis, batch processing, and results management. It serves as the main entry point for the Model Forensics framework.

## Class: `RCAPipeline`

### Constructor

```python
RCAPipeline(config: RCAConfig)
```

**Parameters:**
- `config` (RCAConfig): Complete configuration for all modules

**Raises:**
- `ValueError`: If configuration is invalid

**Example:**
```python
from src.modules.pipeline import RCAPipeline
from src.core.config import RCAConfig

config = RCAConfig.from_file("configs/default.yaml")
pipeline = RCAPipeline(config)
```

### Primary Methods

#### `analyze_incident(incident_bundle: IncidentBundle, model, baseline_bundles: Optional[List[IncidentBundle]] = None) -> CausalResult`

Performs complete RCA analysis on a single incident.

**Parameters:**
- `incident_bundle` (IncidentBundle): The incident to analyze
- `model`: The model to replay (torch.nn.Module or similar)
- `baseline_bundles` (List[IncidentBundle], optional): Benign cases for baseline

**Returns:**
- `CausalResult`: Complete analysis results

**Raises:**
- `ValueError`: If incident_bundle or model is None
- `Exception`: Analysis failures are handled gracefully

**Example:**
```python
from transformers import GPT2LMHeadModel

# Setup
model = GPT2LMHeadModel.from_pretrained('gpt2')
incident = replayer.load_bundle("injection_hack_001")
baselines = [replayer.load_bundle(f"benign_{i}") for i in range(3)]

# Analyze
result = pipeline.analyze_incident(incident, model, baselines)

# Results
print(f"Analysis completed in {result.execution_time:.1f}s")
print(f"Confidence: {result.confidence:.3f}")
print(f"CF metrics: {len(result.compression_metrics)} layers")

if result.fix_set:
    print(f"Fix set: {len(result.fix_set.interventions)} interventions")
    print(f"Success rate: {result.fix_set.total_flip_rate:.1%}")
```

#### `analyze_batch(incident_bundles: List[IncidentBundle], model, baseline_bundles: Optional[List[IncidentBundle]] = None) -> List[CausalResult]`

Analyzes multiple incidents efficiently with shared baselines.

**Parameters:**
- `incident_bundles` (List[IncidentBundle]): List of incidents to analyze
- `model`: The model to replay
- `baseline_bundles` (List[IncidentBundle], optional): Shared benign baselines

**Returns:**
- `List[CausalResult]`: Results for each incident (same order)

**Example:**
```python
# Load multiple incidents
incidents = [
    replayer.load_bundle("injection_001"),
    replayer.load_bundle("injection_002"),
    replayer.load_bundle("toxic_001")
]

# Batch analysis
results = pipeline.analyze_batch(incidents, model, baselines)

# Summary
successful_fixes = sum(1 for r in results if r.fix_set)
avg_confidence = sum(r.confidence for r in results) / len(results)

print(f"Analyzed {len(incidents)} incidents")
print(f"Found fixes for {successful_fixes} incidents")
print(f"Average confidence: {avg_confidence:.3f}")
```

#### `quick_triage(bundles: List[IncidentBundle]) -> List[Tuple[str, float]]`

Quickly ranks incidents by severity for prioritization.

**Parameters:**
- `bundles` (List[IncidentBundle]): Bundles to triage

**Returns:**
- `List[Tuple[str, float]]`: (bundle_id, severity_score) sorted by severity

**Example:**
```python
# Load suspicious cases
candidates = [
    replayer.load_bundle(f"suspicious_{i:03d}") 
    for i in range(20)
]

# Triage
triage_results = pipeline.quick_triage(candidates)

# Prioritize analysis
high_priority = [bid for bid, score in triage_results[:5]]
print(f"High priority incidents: {high_priority}")
```

### Analysis Methods

#### `export_results(result: CausalResult, output_path: Path) -> Path`

Exports analysis results to JSON file.

**Parameters:**
- `result` (CausalResult): Results to export
- `output_path` (Path): Output file path

**Returns:**
- `Path`: Path to created file

**Example:**
```python
from pathlib import Path

# Analyze and export
result = pipeline.analyze_incident(incident, model)
export_path = pipeline.export_results(
    result, 
    Path("results/analysis_001.json")
)

print(f"Results saved to {export_path}")
```

#### `validate_fix_set(fix_set: FixSet, session: ReplaySession) -> Dict[str, Any]`

Validates a fix set by re-testing its interventions.

**Parameters:**
- `fix_set` (FixSet): Fix set to validate
- `session` (ReplaySession): Session for testing

**Returns:**
- `Dict[str, Any]`: Validation results

**Example:**
```python
# Validate best fix set
if result.fix_set:
    session = replayer.create_session(incident, model)
    validation = pipeline.validate_fix_set(result.fix_set, session)
    
    print(f"Validation flip rate: {validation['flip_rate']:.3f}")
    print(f"Consistency: {validation['consistency']:.3f}")
```

### Utility Methods

#### `get_pipeline_stats() -> Dict[str, Any]`

Returns comprehensive pipeline statistics.

**Returns:**
- `Dict[str, Any]`: Statistics including:
  - `cf_enabled`: Whether CF module is active
  - `cca_enabled`: Whether CCA module is active
  - `analyzed_incidents`: Number of incidents analyzed
  - `total_execution_time`: Total time spent on analysis

**Example:**
```python
stats = pipeline.get_pipeline_stats()
print(f"Pipeline enabled: CF={stats['cf_enabled']}, CCA={stats['cca_enabled']}")
print(f"Analyzed {stats['analyzed_incidents']} incidents")
print(f"Total runtime: {stats['total_execution_time']:.1f}s")
```

## Data Types

### `CausalResult`

Complete results from RCA analysis.

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
print(f"  Execution time: {result.execution_time:.2f}s")
print(f"  Confidence: {result.confidence:.3f}")
print(f"  CF layers analyzed: {len(result.compression_metrics)}")

if result.fix_set:
    print(f"  Fix set size: {len(result.fix_set.interventions)}")
    print(f"  Fix success rate: {result.fix_set.total_flip_rate:.1%}")
else:
    print(f"  No fix set found")
```

## Configuration

The pipeline uses the complete RCA configuration:

```yaml
# Example pipeline configuration
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

replayer:
  max_activation_size: "1GB"
  timeout_seconds: 300

recorder:
  granularity: "layer"
  compress_activations: true
```

## Workflow

The pipeline implements the complete CF → CCA workflow:

```
1. Input Validation
   ├── Validate incident bundle
   ├── Validate model
   └── Validate baseline bundles

2. CF Phase (Compression Forensics)
   ├── Build baseline (if baselines provided)
   ├── Analyze incident bundle
   ├── Compute compression metrics
   └── Prioritize anomalous layers

3. CCA Phase (Causal Analysis)
   ├── Create replay session
   ├── Focus search on CF priority layers
   ├── Find minimal fix sets
   └── Rank by effectiveness & minimality

4. Results Integration
   ├── Select best fix set
   ├── Compute overall confidence
   ├── Package results
   └── Log performance metrics
```

## Error Handling

The pipeline includes comprehensive error handling:

1. **Graceful Degradation**: CF failures don't prevent CCA analysis
2. **Partial Results**: Returns partial results when components fail
3. **Input Validation**: All inputs validated before processing
4. **Resource Management**: Timeouts and memory limits prevent hanging
5. **Detailed Logging**: Comprehensive logs for debugging

## Performance Optimization

### Progressive Narrowing

The pipeline uses CF results to optimize CCA search:

```python
# CF identifies suspicious layers
metrics = cf.analyze_bundle(incident)
priority_layers = cf.prioritize_layers(metrics)

# CCA focuses on priority layers only
fix_sets = cca.find_minimal_fix_sets(session, priority_layers)
```

### Batch Processing

Batch analysis shares baselines and models:

```python
# Shared baseline built once
if baseline_bundles and self.cf:
    self.cf.build_baseline(baseline_bundles)

# Each incident analyzed without rebuilding baseline
for incident in incidents:
    result = self.analyze_incident(incident, model, baseline_bundles=None)
```

## Integration Examples

### Basic Analysis

```python
from src.core.config import RCAConfig
from src.modules.pipeline import RCAPipeline
from src.replayer.core import Replayer
from transformers import GPT2LMHeadModel

# Initialize
config = RCAConfig()
pipeline = RCAPipeline(config)
replayer = Replayer(config)
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Single incident analysis
incident = replayer.load_bundle("prompt_injection_001")
result = pipeline.analyze_incident(incident, model)

if result.fix_set:
    print(f"Found fix with {len(result.fix_set.interventions)} interventions")
else:
    print("No fix found")
```

### Advanced Batch Analysis

```python
# Load all incidents
incident_dir = Path("incident_bundles")
incidents = []
baselines = []

for bundle_path in incident_dir.glob("*"):
    bundle = replayer.load_bundle(bundle_path)
    if "benign" in bundle.bundle_id:
        baselines.append(bundle)
    else:
        incidents.append(bundle)

print(f"Loaded {len(incidents)} incidents, {len(baselines)} baselines")

# Batch analysis with triage
print("Performing triage...")
triage_results = pipeline.quick_triage(incidents)
priority_incidents = [
    next(b for b in incidents if b.bundle_id == bid)
    for bid, score in triage_results[:10]  # Top 10
]

print("Analyzing priority incidents...")
results = pipeline.analyze_batch(priority_incidents, model, baselines)

# Summary
fixes_found = sum(1 for r in results if r.fix_set)
avg_confidence = sum(r.confidence for r in results) / len(results)

print(f"Analysis complete:")
print(f"  {fixes_found}/{len(results)} incidents had fixable causes")
print(f"  Average confidence: {avg_confidence:.3f}")
```

### Custom Workflow

```python
class CustomAnalyzer:
    def __init__(self, config):
        self.pipeline = RCAPipeline(config)
        self.replayer = Replayer(config)
        
    def analyze_with_validation(self, incident_path, model):
        # Load and analyze
        incident = self.replayer.load_bundle(incident_path)
        result = self.pipeline.analyze_incident(incident, model)
        
        # Validate fix set if found
        if result.fix_set:
            session = self.replayer.create_session(incident, model)
            validation = self.pipeline.validate_fix_set(result.fix_set, session)
            
            if validation['flip_rate'] < 0.7:
                print("Warning: Fix set validation failed")
        
        # Export results
        output_path = Path(f"results/{incident.bundle_id}_analysis.json")
        self.pipeline.export_results(result, output_path)
        
        return result, output_path

# Usage
analyzer = CustomAnalyzer(config)
result, path = analyzer.analyze_with_validation("incident_001", model)
```