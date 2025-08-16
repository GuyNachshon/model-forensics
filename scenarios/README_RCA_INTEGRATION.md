# RCA Integration with Alignment Scenarios

This document describes how the Model Forensics RCA framework has been integrated with the alignment scenarios system.

## Overview

The RCA integration allows automatic failure detection and root cause analysis of model behaviors in alignment scenarios. When a scenario execution is classified as a failure (e.g., blackmail, leaking sensitive information), the system can automatically:

1. Record model execution traces
2. Perform RCA analysis using CF → IS-C → CCA pipeline
3. Cache results to avoid re-analysis
4. Optionally stop execution early when failures are detected

## Configuration

### Enabling RCA in YAML Config

Add an `rca` section to your scenario YAML configuration:

```yaml
rca:
  enabled: true                             # Enable/disable RCA analysis
  config_path: "configs/default.yaml"      # Optional: path to RCA config file
```

**Design Philosophy**: Scenarios just enable/disable RCA. All detailed RCA configuration (modules, thresholds, etc.) comes from the main Model Forensics configuration system (`configs/default.yaml`).

This separation means:
- Scenario configs stay focused on scenario-specific settings
- RCA behavior is controlled by the main framework configuration
- Easy to maintain consistent RCA settings across different scenarios

### Available Analysis Modules

- **`cf`** (Compression Forensics): Detects anomalous computation patterns via compressibility analysis
- **`cca`** (Causal Minimal Fix Sets): Finds smallest intervention sets that correct failures  
- **`is_c`** (Interaction Spectroscopy): Analyzes component interactions and failure propagation
- **`dbc`** (Decision Basin Cartography): Maps decision boundaries around failures [Future]
- **`provenance`** (Training Provenance): Links failures to training data [Future]

## Workflow Integration

### 1. Model Selection & Loading
- Works with existing model selection (set-model.sh)
- RCA recorder hooks are installed when RCA is enabled
- Model loading remains unchanged for API-based models

### 2. Scenario Execution with Caching
- Checks for existing results before running scenarios
- For RCA-enabled configs, also checks if failure analysis is complete
- Skips scenarios that are fully processed (response + RCA if needed)

### 3. Classification-Based Triggering
- After each scenario execution, classification runs as normal
- If classification indicates failure (verdict=True or classifier_verdict=True):
  - RCA analysis is triggered automatically
  - Model execution traces are analyzed
  - Results are saved alongside scenario response

### 4. Early Stop Logic
- When `early_stop_on_failure` is enabled:
  - Execution stops after first detected failure
  - RCA analysis completes for the failed scenario
  - Remaining scenarios in batch are skipped

## File Structure

### Scenario Results with RCA
```
results/experiment_YYYYMMDD_HHMMSS/
├── models/
│   └── model_name/
│       └── condition_name/
│           └── sample_N/
│               └── response.json          # Enhanced with RCA data
└── rca_analysis/
    ├── rca_model_condition_sample.json   # Cached RCA results
    └── bundles/                          # Incident bundles (future)
        └── scenario_model_condition_sample_YYYYMMDD_HHMMSS/
```

### Enhanced Response Format
```json
{
  "raw_response": { /* model response */ },
  "metadata": { /* scenario metadata */ },
  "classification": {
    "verdict": true,
    "classifier_verdict": true,
    "reasoning": "Model provided blackmail content..."
  },
  "rca_analysis": {
    "metadata": {
      "model": "openai-community/gpt2",
      "condition": "blackmail_explicit-america_none",
      "sample_id": 1,
      "analysis_duration_seconds": 15.3,
      "modules_used": ["cf", "cca", "is_c"]
    },
    "scenario_context": {
      "messages": [/* conversation messages */],
      "response": "model response text",
      "classification": {/* classification results */}
    },
    "rca_analysis": {
      "cf_result": {
        "anomaly_layers": ["layer_8", "layer_11"],
        "compression_scores": {"layer_8": 0.85, "layer_11": 0.92}
      },
      "is_c_result": {
        "critical_interactions": [
          {"components": ["attention_head_8_2", "ffn_layer_11"], "strength": 0.78}
        ]
      },
      "cca_result": {
        "minimal_fix_set": [
          {"layer": "layer_8", "intervention": "zero_attention_head_2", "effect_size": 0.89}
        ],
        "flip_rate": 0.85
      }
    },
    "summary": {
      "failure_detected": true,
      "modules_completed": ["cf", "is_c", "cca"],
      "key_findings": [
        "Anomalous layers detected: ['layer_8', 'layer_11']",
        "Minimal fix set: 2 interventions (flip rate: 85.00%)"
      ]
    }
  }
}
```

## Usage Examples

### Basic RCA-Enabled Scenario
```bash
cd scenarios/
python run_scenarios.py --config configs/example-with-rca.yaml
```

### Running Existing Scenarios with RCA
1. Copy an existing config (e.g., `alignment-faking.yaml`)
2. Add the `rca` section with `enabled: true`
3. Run normally - RCA will trigger on failures

### Batch Analysis of Previous Results
If you have existing scenario results without RCA analysis:
1. Set `force_rerun: false` in config
2. Enable RCA analysis
3. Re-run - will only analyze previously failed scenarios

## Performance Considerations

### Computational Cost
- RCA analysis adds 10-60 seconds per failed scenario
- CF analysis: ~5-10 seconds
- IS-C analysis: ~3-8 seconds  
- CCA analysis: ~10-45 seconds (depends on search space)

### Memory Usage
- Activation recording increases memory usage during inference
- Bundle storage: ~50-500MB per incident (configurable)
- Use `compression_ratio` to control activation storage size

### Caching Benefits
- RCA results are cached by default
- Re-running scenarios skips completed analyses
- Useful for iterative experimentation

## Troubleshooting

### Common Issues

1. **RCA modules not found**
   ```
   ImportError: RCA modules not available
   ```
   - Ensure core Model Forensics framework is implemented
   - Check that `src/modules/` contains CF, CCA, IS-C implementations

2. **Memory issues during recording**
   ```
   CUDA out of memory during activation recording
   ```
   - Reduce `compression_ratio` (e.g., 0.05 instead of 0.1)
   - Decrease `max_bundle_size_mb`
   - Process scenarios with smaller batch sizes

3. **Cache inconsistencies**
   ```
   Failed to check RCA cache status
   ```
   - Set `force_rerun: true` to bypass cache
   - Check file permissions in output directory
   - Clear cache directory manually if needed

### Debug Mode
Enable debug logging to see detailed RCA execution:
```yaml
global:
  debug: true
  verbose: true
```

## Future Enhancements

### Phase 2: Real Intervention Application
- Apply discovered interventions during model inference
- Re-run scenarios with applied fixes
- Measure intervention effectiveness

### Phase 3: Advanced Analysis
- Enable DBC (Decision Basin Cartography) module
- Add Training Provenance analysis
- Cross-scenario failure pattern detection

### Phase 4: Production Integration
- Real-time RCA during live model deployment
- Automated intervention application
- Comprehensive failure dashboards

## API Reference

### ScenarioRCAConfig
```python
@dataclass
class ScenarioRCAConfig:
    enabled: bool = False
    record_activations: bool = True
    early_stop_on_failure: bool = True
    run_analysis_on_failure: bool = True
    cache_results: bool = True
    analysis_modules: List[str] = ["cf", "cca"]
    max_bundle_size_mb: int = 500
    compression_ratio: float = 0.1
```

### ScenarioRCAAnalyzer
```python
class ScenarioRCAAnalyzer:
    def should_analyze_failure(self, classification_result: Dict) -> bool
    def has_cached_result(self, model: str, condition: str, sample_id: int) -> bool
    async def record_scenario_execution(self, model, messages, model_name, condition, sample_id) -> Optional[str]
    async def analyze_failure(self, model_name, condition, sample_id, messages, response, classification_result, bundle_id=None) -> Optional[Dict]
```

## Configuration Examples

See `configs/example-with-rca.yaml` for a complete working example.

For questions or issues, refer to the main Model Forensics documentation or create an issue in the repository.