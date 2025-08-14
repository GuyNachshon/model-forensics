# CLI API Reference

## Overview

The Model Forensics CLI provides a complete command-line interface for RCA analysis, batch processing, and incident management. It's built using Click and provides both interactive and scriptable interfaces.

## Installation & Setup

```bash
# Install in development mode
pip install -e ".[dev]"

# Or using uv
uv sync --all-extras

# Verify installation
model-forensics --help
```

## Global Options

Available for all commands:

```bash
--config PATH        Configuration file path (default: configs/default.yaml)
--verbose, -v        Enable verbose output
--quiet, -q          Suppress non-essential output
--log-level LEVEL    Set logging level (DEBUG, INFO, WARNING, ERROR)
--help               Show help message
```

## Commands

### `analyze`

Analyze a single incident bundle.

```bash
model-forensics analyze [OPTIONS] INCIDENT_PATH MODEL_TYPE
```

**Arguments:**
- `INCIDENT_PATH`: Path to incident bundle directory
- `MODEL_TYPE`: Model type (gpt2, bert, etc.)

**Options:**
- `--baseline DIR`: Directory containing baseline bundles
- `--output FILE`: Output file for results (JSON format)
- `--model-name TEXT`: Specific model name/path
- `--max-fix-size INT`: Maximum interventions per fix set (default: 5)
- `--search-strategy CHOICE`: Search strategy (greedy/beam/random)
- `--threshold FLOAT`: Anomaly detection threshold [0-1]

**Examples:**

```bash
# Basic analysis
model-forensics analyze incident_001 gpt2

# With baselines and custom output
model-forensics analyze incident_001 gpt2 \
  --baseline baseline_bundles/ \
  --output results/incident_001.json

# Custom search parameters
model-forensics analyze injection_hack_001 gpt2 \
  --search-strategy beam \
  --max-fix-size 3 \
  --threshold 0.8

# Verbose analysis with specific model
model-forensics analyze incident_001 gpt2 \
  --model-name gpt2-medium \
  --verbose \
  --output detailed_results.json
```

**Output Format:**
```json
{
  "bundle_id": "incident_001",
  "execution_time": 45.2,
  "confidence": 0.847,
  "fix_set": {
    "interventions": [
      {
        "layer_name": "transformer.h.3.attn",
        "type": "zero",
        "flip_success": true,
        "confidence": 0.92
      }
    ],
    "total_flip_rate": 0.89,
    "minimality_rank": 1
  },
  "compression_metrics": [
    {
      "layer_name": "transformer.h.3.attn",
      "anomaly_score": 0.91,
      "is_anomalous": true
    }
  ]
}
```

### `batch`

Analyze multiple incidents with shared resources.

```bash
model-forensics batch [OPTIONS] INCIDENT_DIR MODEL_TYPE
```

**Arguments:**
- `INCIDENT_DIR`: Directory containing incident bundles
- `MODEL_TYPE`: Model type for all incidents

**Options:**
- `--baseline DIR`: Baseline bundles directory
- `--output DIR`: Output directory for results
- `--parallel INT`: Number of parallel workers (default: 1)
- `--pattern TEXT`: Glob pattern for incident bundles (default: "*")
- `--limit INT`: Maximum number of incidents to process
- `--summary`: Generate summary report

**Examples:**

```bash
# Analyze all incidents in directory
model-forensics batch incident_bundles/ gpt2 \
  --output batch_results/

# Parallel processing with baselines
model-forensics batch incident_bundles/ gpt2 \
  --baseline baseline_bundles/ \
  --parallel 4 \
  --summary

# Process specific pattern with limit
model-forensics batch incident_bundles/ gpt2 \
  --pattern "*injection*" \
  --limit 10 \
  --output injection_results/

# Generate comprehensive report
model-forensics batch incident_bundles/ gpt2 \
  --baseline baseline_bundles/ \
  --parallel 2 \
  --summary \
  --verbose \
  --output comprehensive_analysis/
```

**Batch Summary:**
```
Batch Analysis Summary
======================
Total incidents: 25
Successful analyses: 23
Fixes found: 18
Average confidence: 0.742

Top anomalous layers:
1. transformer.h.3.attn (12 incidents)
2. transformer.h.5.mlp (8 incidents)
3. transformer.h.1.attn (6 incidents)

Fix set statistics:
- Average size: 2.3 interventions
- Success rate: 87.4%
- Most common intervention: zero (67%)
```

### `triage`

Quickly rank incidents by severity for prioritization.

```bash
model-forensics triage [OPTIONS] INCIDENT_DIR
```

**Arguments:**
- `INCIDENT_DIR`: Directory containing incident bundles

**Options:**
- `--output FILE`: Output file for triage results
- `--top INT`: Show only top N incidents (default: 10)
- `--format CHOICE`: Output format (table/json/csv)
- `--sort-by CHOICE`: Sort criteria (severity/confidence/size)

**Examples:**

```bash
# Quick triage of incidents
model-forensics triage incident_bundles/

# Export top 20 to CSV
model-forensics triage incident_bundles/ \
  --top 20 \
  --format csv \
  --output high_priority.csv

# JSON output for further processing
model-forensics triage incident_bundles/ \
  --format json \
  --output triage_results.json
```

**Triage Output:**
```
Incident Triage Results
=======================
Rank  Bundle ID              Severity  Size    Priority
1     injection_hack_003     0.94      2.1MB   HIGH
2     toxic_generation_001   0.89      1.8MB   HIGH  
3     prompt_bypass_007      0.85      2.3MB   HIGH
4     jailbreak_attempt_002  0.82      1.9MB   MEDIUM
5     benign_false_pos_001   0.31      1.2MB   LOW
```

### `validate`

Validate bundle integrity and format.

```bash
model-forensics validate [OPTIONS] BUNDLE_PATH
```

**Arguments:**
- `BUNDLE_PATH`: Path to bundle or directory of bundles

**Options:**
- `--fix`: Attempt to fix minor issues
- `--verbose`: Show detailed validation info
- `--format-version TEXT`: Expected format version

**Examples:**

```bash
# Validate single bundle
model-forensics validate incident_001/

# Validate and fix issues
model-forensics validate incident_001/ --fix

# Validate all bundles in directory
model-forensics validate incident_bundles/ --verbose
```

### `export`

Export results in various formats for external tools.

```bash
model-forensics export [OPTIONS] RESULTS_FILE
```

**Arguments:**
- `RESULTS_FILE`: Analysis results file (JSON)

**Options:**
- `--format CHOICE`: Export format (csv/html/pdf/xlsx)
- `--output FILE`: Output file path
- `--template TEXT`: Custom template file
- `--include CHOICE`: Data to include (all/fixes/metrics/summary)

**Examples:**

```bash
# Export to HTML report
model-forensics export results.json \
  --format html \
  --output report.html

# CSV export with only fix sets
model-forensics export results.json \
  --format csv \
  --include fixes \
  --output fixes.csv

# Custom template export
model-forensics export results.json \
  --format html \
  --template custom_template.html \
  --output custom_report.html
```

### `config`

Manage configuration files.

```bash
model-forensics config [OPTIONS] COMMAND
```

**Subcommands:**
- `show`: Display current configuration
- `validate`: Validate configuration file
- `create`: Create new configuration template
- `merge`: Merge multiple configuration files

**Examples:**

```bash
# Show current config
model-forensics config show

# Validate custom config
model-forensics config validate custom.yaml

# Create new config template
model-forensics config create --output my_config.yaml

# Merge configurations
model-forensics config merge base.yaml custom.yaml \
  --output merged.yaml
```

### `info`

Display system and framework information.

```bash
model-forensics info [OPTIONS]
```

**Options:**
- `--verbose`: Show detailed system info
- `--check-deps`: Check dependency versions
- `--gpu-info`: Show GPU information

**Examples:**

```bash
# Basic info
model-forensics info

# Detailed system information
model-forensics info --verbose --gpu-info
```

**Info Output:**
```
Model Forensics v0.1.0
======================
Python: 3.10.12
PyTorch: 2.0.1
Transformers: 4.30.2
Platform: Linux-5.15.0-1083-gcp
GPU: NVIDIA A100-SXM4-40GB (Available)

Configuration: configs/default.yaml
Modules: CF ✓, CCA ✓, Pipeline ✓
```

## Configuration

The CLI uses the same YAML configuration as the Python API:

```yaml
# ~/.model-forensics/config.yaml
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

cli:
  default_output_format: "json"
  show_progress_bars: true
  parallel_workers: 2
  log_level: "INFO"
```

## Scripting Examples

### Automated Analysis Pipeline

```bash
#!/bin/bash
# analyze_incidents.sh

INCIDENT_DIR="$1"
MODEL_TYPE="$2"
OUTPUT_DIR="results/$(date +%Y%m%d_%H%M%S)"

echo "Starting analysis pipeline..."

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Triage incidents
echo "Triaging incidents..."
model-forensics triage "$INCIDENT_DIR" \
  --format json \
  --output "$OUTPUT_DIR/triage.json"

# Step 2: Analyze high-priority incidents
echo "Analyzing high-priority incidents..."
model-forensics batch "$INCIDENT_DIR" "$MODEL_TYPE" \
  --pattern "*high*" \
  --baseline "baseline_bundles/" \
  --parallel 4 \
  --output "$OUTPUT_DIR/analysis/" \
  --summary

# Step 3: Generate reports
echo "Generating reports..."
for result_file in "$OUTPUT_DIR"/analysis/*.json; do
  model-forensics export "$result_file" \
    --format html \
    --output "${result_file%.json}.html"
done

echo "Analysis complete. Results in $OUTPUT_DIR"
```

### Monitoring Script

```bash
#!/bin/bash
# monitor_incidents.sh

WATCH_DIR="incident_queue/"
PROCESSED_DIR="processed/"
RESULTS_DIR="results/"

while inotifywait -e create "$WATCH_DIR"; do
  for incident in "$WATCH_DIR"/*; do
    if [[ -d "$incident" ]]; then
      echo "Processing new incident: $(basename "$incident")"
      
      # Analyze incident
      model-forensics analyze "$incident" gpt2 \
        --baseline "baseline_bundles/" \
        --output "$RESULTS_DIR/$(basename "$incident").json" \
        --verbose
      
      # Move to processed
      mv "$incident" "$PROCESSED_DIR/"
      
      echo "Completed: $(basename "$incident")"
    fi
  done
done
```

### Batch Processing with Error Handling

```bash
#!/bin/bash
# robust_batch_analysis.sh

set -euo pipefail  # Exit on error

INCIDENT_DIR="$1"
MODEL_TYPE="${2:-gpt2}"
MAX_RETRIES=3

# Create timestamped output directory
OUTPUT_DIR="results/batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Log file
LOGFILE="$OUTPUT_DIR/analysis.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

log "Starting batch analysis of $INCIDENT_DIR"

# Validate bundles first
log "Validating bundles..."
if ! model-forensics validate "$INCIDENT_DIR" --verbose >> "$LOGFILE" 2>&1; then
  log "WARNING: Some bundles failed validation"
fi

# Run analysis with retry logic
for attempt in $(seq 1 $MAX_RETRIES); do
  log "Analysis attempt $attempt"
  
  if model-forensics batch "$INCIDENT_DIR" "$MODEL_TYPE" \
     --baseline "baseline_bundles/" \
     --parallel 2 \
     --output "$OUTPUT_DIR" \
     --summary \
     --verbose >> "$LOGFILE" 2>&1; then
    
    log "Analysis completed successfully"
    break
  else
    log "Analysis attempt $attempt failed"
    if [[ $attempt -eq $MAX_RETRIES ]]; then
      log "ERROR: All attempts failed"
      exit 1
    fi
    sleep 30  # Wait before retry
  fi
done

# Generate summary report
log "Generating summary report..."
{
  echo "# Batch Analysis Report"
  echo "Generated: $(date)"
  echo
  cat "$OUTPUT_DIR"/summary.txt
} > "$OUTPUT_DIR/report.md"

log "Analysis complete. Results in $OUTPUT_DIR"
```

## Environment Variables

The CLI respects these environment variables:

```bash
export MODEL_FORENSICS_CONFIG="path/to/config.yaml"
export MODEL_FORENSICS_LOG_LEVEL="DEBUG"
export MODEL_FORENSICS_CACHE_DIR="~/.cache/model-forensics"
export MODEL_FORENSICS_PARALLEL_WORKERS="4"
export CUDA_VISIBLE_DEVICES="0,1"  # For GPU selection
```

## Exit Codes

- `0`: Success
- `1`: General error
- `2`: Invalid arguments
- `3`: Configuration error
- `4`: Bundle validation failed
- `5`: Analysis failed
- `6`: Export failed
- `130`: Interrupted (Ctrl+C)

## Integration with Other Tools

### Using with Docker

```dockerfile
FROM python:3.10

COPY . /app
WORKDIR /app

RUN pip install -e ".[dev]"

ENTRYPOINT ["model-forensics"]
```

```bash
# Build and run
docker build -t model-forensics .
docker run -v $(pwd)/data:/data model-forensics analyze /data/incident_001 gpt2
```

### Using with CI/CD

```yaml
# .github/workflows/forensics.yml
name: Model Forensics Analysis
on: [push, pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -e ".[dev]"
      
      - name: Validate bundles
        run: model-forensics validate test_bundles/
      
      - name: Run analysis
        run: |
          model-forensics batch test_bundles/ gpt2 \
            --output results/ \
            --summary
      
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: analysis-results
          path: results/
```