# CCA Module API Reference

## Overview

The Causal Minimal Fix Sets (CCA) module finds minimal sets of interventions that correct model failures. It serves as the second phase of the RCA pipeline, building on anomalous layers identified by CF to discover targeted fixes.

## Class: `CausalCCA`

### Constructor

```python
CausalCCA(config: Optional[CCAConfig] = None)
```

**Parameters:**
- `config` (CCAConfig, optional): Configuration object. Defaults to `CCAConfig()`.

**Raises:**
- `ValueError`: If configuration is invalid (e.g., invalid search strategy)

**Example:**
```python
from src.modules.cca import CausalCCA
from src.core.config import CCAConfig

config = CCAConfig()
config.search_strategy = "greedy"
config.max_fix_set_size = 3
cca = CausalCCA(config)
```

### Methods

#### `find_minimal_fix_sets(session: ReplaySession, priority_layers: Optional[List[str]] = None) -> List[FixSet]`

Finds minimal sets of interventions that fix the failure.

**Parameters:**
- `session` (ReplaySession): Replay session with model and reconstructed activations
- `priority_layers` (List[str], optional): Layers to focus search on. Defaults to all layers.

**Returns:**
- `List[FixSet]`: List of fix sets ranked by effectiveness and minimality

**Raises:**
- `ValueError`: If session is None or invalid
- `Exception`: If search fails

**Example:**
```python
# Create session and find fix sets
session = replayer.create_session(incident_bundle, model)
priority_layers = ["transformer.h.0", "transformer.h.1"]

fix_sets = cca.find_minimal_fix_sets(session, priority_layers)

for i, fix_set in enumerate(fix_sets[:3]):
    print(f"Fix Set {i+1}:")
    print(f"  Interventions: {len(fix_set.interventions)}")
    print(f"  Flip Rate: {fix_set.total_flip_rate:.3f}")
    print(f"  Minimality: {fix_set.minimality_rank}")
```

#### `get_stats() -> Dict[str, Any]`

Returns current CCA module statistics.

**Returns:**
- `Dict[str, Any]`: Dictionary containing:
  - `tested_combinations`: Number of combinations tested
  - `successful_interventions`: Number of successful individual interventions
  - `config`: Configuration parameters

**Example:**
```python
stats = cca.get_stats()
print(f"Tested {stats['tested_combinations']} combinations")
print(f"Found {stats['successful_interventions']} successful interventions")
```

## Data Types

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
fix_set = fix_sets[0]  # Best fix set
print(f"Fix set has {len(fix_set.interventions)} interventions")
print(f"Success rate: {fix_set.total_flip_rate:.1%}")
print(f"Side effects: {fix_set.avg_side_effects:.3f}")

# Access individual interventions
for result in fix_set.interventions:
    intervention = result.intervention
    print(f"  - {intervention.layer_name}: {intervention.type.value}")
```

### `InterventionResult`

Result of applying a single intervention.

**Fields:**
- `intervention` (Intervention): The intervention that was applied
- `original_activation` (torch.Tensor): Original layer activation
- `modified_activation` (torch.Tensor): Modified activation after intervention
- `original_output` (Any): Original model output
- `modified_output` (Any): Modified model output
- `flip_success` (bool): Whether intervention changed the failure outcome
- `confidence` (float): Confidence in the result [0, 1]
- `side_effects` (Dict[str, float]): Measured side effects
- `metadata` (Dict[str, Any]): Additional metadata

### `Intervention`

Specification for a model intervention.

**Fields:**
- `type` (InterventionType): Type of intervention (ZERO, MEAN, PATCH)
- `layer_name` (str): Target layer name
- `value` (float, optional): Value for ZERO/MEAN interventions
- `donor_activation` (torch.Tensor, optional): Activation for PATCH interventions

**Example:**
```python
from src.core.types import InterventionType

# Different intervention types
zero_intervention = Intervention(
    type=InterventionType.ZERO,
    layer_name="transformer.h.5.mlp",
    value=0.0
)

mean_intervention = Intervention(
    type=InterventionType.MEAN,
    layer_name="transformer.h.3.attn",
    value=0.125  # Computed mean
)

patch_intervention = Intervention(
    type=InterventionType.PATCH,
    layer_name="transformer.h.1",
    donor_activation=benign_activation
)
```

### `CCAConfig`

Configuration for CCA module.

**Fields:**
- `max_fix_set_size` (int): Maximum interventions per fix set
- `search_strategy` (str): Search strategy ("greedy", "beam", "random")
- `early_stop_threshold` (float): Stop when flip rate exceeds this [0, 1]
- `intervention_types` (List[str]): Types to try ["zero", "mean", "patch"]
- `beam_width` (int): Beam width for beam search
- `random_samples` (int): Number of samples for random search

**Example:**
```python
from src.core.config import CCAConfig

config = CCAConfig()
config.max_fix_set_size = 4
config.search_strategy = "beam"
config.early_stop_threshold = 0.85
config.intervention_types = ["zero", "mean", "patch"]

cca = CausalCCA(config)
```

## Search Strategies

### Greedy Search

Builds fix sets by greedily adding the best intervention at each step.

**Characteristics:**
- **Speed**: Fastest search method
- **Quality**: Good for simple failures
- **Use case**: When you need quick results

### Beam Search

Maintains multiple candidate fix sets and expands the best ones.

**Characteristics:**
- **Speed**: Moderate (configurable beam width)
- **Quality**: Better exploration than greedy
- **Use case**: Balanced speed/quality trade-off

### Random Search

Randomly samples intervention combinations within constraints.

**Characteristics:**
- **Speed**: Configurable (number of samples)
- **Quality**: Can find unexpected solutions
- **Use case**: When other methods fail or for exploration

## Internal Methods

The following methods are internal implementation details:

- `_test_individual_interventions(session, layers) -> List[InterventionResult]`
- `_greedy_search(session, layers, results) -> List[FixSet]`
- `_beam_search(session, layers, results) -> List[FixSet]`
- `_random_search(session, layers, results) -> List[FixSet]`
- `_create_intervention(type, layer_name, activations) -> Intervention`
- `_evaluate_combination(session, results) -> float`
- `_compute_necessity_scores(session, results) -> List[float]`
- `_rank_fix_sets(fix_sets) -> List[FixSet]`
- `_validate_config() -> None`

## Error Handling

The CCA module includes comprehensive error handling:

1. **Session Validation**: Validates replay session has required fields
2. **Layer Validation**: Checks that priority layers exist in activations
3. **Configuration Validation**: Validates search strategy and parameters
4. **Intervention Failures**: Gracefully handles intervention application failures
5. **Search Robustness**: Continues search even if individual interventions fail

## Performance Notes

- **Early Stopping**: Search stops when flip rate threshold is reached
- **Layer Filtering**: Invalid layers are filtered before search
- **Combination Caching**: Tested combinations are cached to avoid duplicates
- **Parallel Search**: Beam search can explore multiple paths simultaneously

## Integration Example

```python
from src.core.config import RCAConfig
from src.modules.cca import CausalCCA
from src.replayer.core import Replayer
from transformers import GPT2LMHeadModel

# Setup
config = RCAConfig()
cca = CausalCCA(config.modules.cca)
replayer = Replayer(config)
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load incident and create session
incident = replayer.load_bundle("prompt_injection_001")
session = replayer.create_session(incident, model)

# Use CF results to focus CCA search
priority_layers = ["transformer.h.3.attn", "transformer.h.5.mlp"]
fix_sets = cca.find_minimal_fix_sets(session, priority_layers)

# Analyze results
if fix_sets:
    best_fix = fix_sets[0]
    print(f"Best fix set:")
    print(f"  {len(best_fix.interventions)} interventions")
    print(f"  {best_fix.total_flip_rate:.1%} success rate")
    print(f"  Rank {best_fix.minimality_rank} minimality")
    
    # Show interventions
    for result in best_fix.interventions:
        intervention = result.intervention
        print(f"  - {intervention.layer_name}: {intervention.type.value}")
else:
    print("No effective fix sets found")
```

## Advanced Usage

### Custom Intervention Types

```python
# Create custom interventions
custom_intervention = Intervention(
    type=InterventionType.PATCH,
    layer_name="target_layer",
    donor_activation=custom_benign_activation
)

# Apply manually through intervention engine
result = cca.intervention_engine.apply_intervention(
    model, activations, custom_intervention, inputs
)
```

### Batch Analysis

```python
# Analyze multiple incidents
incidents = [
    replayer.load_bundle(f"incident_{i:03d}") 
    for i in range(10)
]

all_fix_sets = []
for incident in incidents:
    session = replayer.create_session(incident, model)
    fix_sets = cca.find_minimal_fix_sets(session)
    all_fix_sets.append(fix_sets)

# Analyze patterns across incidents
common_layers = set()
for fix_sets in all_fix_sets:
    if fix_sets:
        layers = {r.intervention.layer_name 
                 for r in fix_sets[0].interventions}
        common_layers.update(layers)
        
print(f"Layers involved in fixes: {sorted(common_layers)}")
```