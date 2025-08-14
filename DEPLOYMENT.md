# Model Forensics Production Deployment Guide

## Overview

Model Forensics is a complete Root Cause Analysis (RCA) framework for foundation model failures. This guide covers production deployment, configuration, and operational best practices.

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository_url>
cd model-forensics

# Install with all dependencies
pip install -e ".[dev]"

# Or using uv (recommended)
uv sync
```

### Basic Usage

```bash
# Analyze a single incident
model-forensics analyze incident_bundle/ --baseline benign_bundle/ --output results.json

# Triage multiple incidents
model-forensics triage incidents_directory/ --sort-by anomaly

# Batch analysis
model-forensics batch incidents_directory/ --output batch_results/

# System information
model-forensics info
```

## Deployment Architectures

### 1. Local Development

```yaml
Environment: Local machine
Use Cases: Research, development, small-scale analysis
Requirements: 8GB RAM, GPU optional
Setup: Direct Python installation
```

### 2. Production Server

```yaml
Environment: Dedicated server/VM
Use Cases: Production incident analysis, batch processing
Requirements: 32GB+ RAM, GPU recommended, SSD storage
Setup: Docker container or systemd service
```

### 3. Cloud Deployment

```yaml
Environment: AWS/GCP/Azure
Use Cases: Scalable incident processing, integration with ML pipelines
Requirements: Auto-scaling, managed storage, monitoring
Setup: Kubernetes, container orchestration
```

## Configuration

### Core Configuration (`config.yaml`)

```yaml
# Model Forensics Configuration
bundle_dir: "./bundles"
output_dir: "./analysis_results"
device: "auto"  # auto, cpu, cuda
timeout_minutes: 120

# CF Module - Compression Forensics
modules:
  cf:
    enabled: true
    anomaly_threshold: 2.0
    compression_methods: ["zlib", "gzip", "arithmetic"]
    entropy_window: 256
    baseline_samples: 50

  # CCA Module - Causal Analysis
  cca:
    enabled: true
    search_strategy: "greedy"  # greedy, beam, random
    max_fix_set_size: 5
    early_stop_threshold: 0.8
    intervention_types: ["zero", "mean", "patch"]

  # Advanced modules (disabled by default)
  isc:
    enabled: false
  dbc:
    enabled: false
  provenance:
    enabled: false

# Logging
log_level: "INFO"
log_file: "model_forensics.log"
```

### Environment Variables

```bash
# Required
export MODEL_FORENSICS_CONFIG="/path/to/config.yaml"

# Optional
export MODEL_FORENSICS_DEVICE="cuda"
export MODEL_FORENSICS_LOG_LEVEL="DEBUG"
export MODEL_FORENSICS_OUTPUT_DIR="/data/analysis_results"
export MODEL_FORENSICS_BUNDLE_DIR="/data/bundles"
```

## Production Setup

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

# Install dependencies
RUN pip install -e ".[dev]"

# Create data directories
RUN mkdir -p /data/bundles /data/results

# Set environment
ENV MODEL_FORENSICS_BUNDLE_DIR="/data/bundles"
ENV MODEL_FORENSICS_OUTPUT_DIR="/data/results"

EXPOSE 8000
CMD ["model-forensics", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t model-forensics .
docker run -v /host/data:/data -p 8000:8000 model-forensics
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-forensics
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-forensics
  template:
    metadata:
      labels:
        app: model-forensics
    spec:
      containers:
      - name: model-forensics
        image: model-forensics:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_FORENSICS_DEVICE
          value: "cuda"
        volumeMounts:
        - name: data-volume
          mountPath: /data
        resources:
          requests:
            memory: "16Gi"
            nvidia.com/gpu: 1
          limits:
            memory: "32Gi"
            nvidia.com/gpu: 1
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: model-forensics-data
```

### Systemd Service

```ini
# /etc/systemd/system/model-forensics.service
[Unit]
Description=Model Forensics RCA Service
After=network.target

[Service]
Type=simple
User=modelforensics
WorkingDirectory=/opt/model-forensics
Environment=MODEL_FORENSICS_CONFIG=/etc/model-forensics/config.yaml
ExecStart=/opt/model-forensics/.venv/bin/model-forensics serve
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Operational Procedures

### Incident Response Workflow

1. **Incident Detection**
   ```bash
   # Automated monitoring detects model failure
   # Creates incident bundle automatically
   ```

2. **Triage**
   ```bash
   # Quick assessment of severity
   model-forensics triage /incidents/$(date +%Y%m%d)/ --sort-by anomaly
   ```

3. **Analysis**
   ```bash
   # Full RCA for high-priority incidents
   model-forensics analyze /incidents/high_priority/incident_001/ \
     --baseline /baselines/benign_cases/ \
     --output /analysis/incident_001_rca.json
   ```

4. **Intervention**
   ```bash
   # Apply recommended fixes
   model-forensics validate-fix /analysis/incident_001_rca.json \
     --model /models/production_model
   ```

### Batch Processing

```bash
# Daily batch analysis
#!/bin/bash
DATE=$(date +%Y%m%d)
model-forensics batch /incidents/$DATE/ \
  --baseline /baselines/daily/ \
  --output /analysis/batch_$DATE/ \
  --config /etc/model-forensics/batch_config.yaml

# Archive results
tar -czf /archive/analysis_$DATE.tar.gz /analysis/batch_$DATE/
```

### Monitoring

```bash
# Health check
model-forensics info --format json | jq '.health'

# Performance metrics
model-forensics stats --since "24h" --format json

# Log monitoring
tail -f /var/log/model-forensics/analysis.log | grep "ERROR\|WARNING"
```

## Performance Optimization

### Hardware Requirements

| Deployment | CPU | RAM | GPU | Storage |
|------------|-----|-----|-----|---------|
| Development | 4 cores | 8GB | Optional | 100GB |
| Production | 16 cores | 32GB | V100/A100 | 1TB SSD |
| Enterprise | 32 cores | 128GB | Multiple GPU | 10TB NVMe |

### Configuration Tuning

```yaml
# High-performance configuration
bundle_dir: "/fast_storage/bundles"
output_dir: "/fast_storage/results"
device: "cuda"

modules:
  cf:
    compression_methods: ["zlib"]  # Fastest method only
    baseline_samples: 20  # Reduce for speed
  
  cca:
    search_strategy: "greedy"  # Fastest search
    max_fix_set_size: 3  # Limit search space
    early_stop_threshold: 0.6  # Earlier stopping
```

### Caching Strategy

```python
# Enable caching for repeated analysis
import os
os.environ['MODEL_FORENSICS_CACHE_ENABLED'] = "true"
os.environ['MODEL_FORENSICS_CACHE_SIZE'] = "10GB"
```

## Integration

### MLOps Integration

```python
# Integration with MLflow
import mlflow
from model_forensics import RCAPipeline

def analyze_model_failure(model_run_id, incident_data):
    # Load model from MLflow
    model = mlflow.pytorch.load_model(f"runs:/{model_run_id}/model")
    
    # Run analysis
    pipeline = RCAPipeline()
    result = pipeline.analyze_incident(incident_data, model)
    
    # Log results to MLflow
    with mlflow.start_run(run_id=model_run_id):
        mlflow.log_metrics({
            "rca_confidence": result.confidence,
            "anomalous_layers": len([m for m in result.compression_metrics if m.is_anomalous])
        })
        mlflow.log_artifact(result.export_path)
```

### API Integration

```python
# REST API endpoint
from fastapi import FastAPI
from model_forensics import RCAPipeline

app = FastAPI()
pipeline = RCAPipeline()

@app.post("/analyze")
async def analyze_incident(incident_bundle: str, baseline_bundle: str = None):
    result = pipeline.analyze_incident(incident_bundle, baseline_bundle)
    return {
        "confidence": result.confidence,
        "execution_time": result.execution_time,
        "anomalous_layers": len([m for m in result.compression_metrics if m.is_anomalous]),
        "fix_set_size": len(result.fix_set.interventions) if result.fix_set else 0
    }
```

### Alerting Integration

```yaml
# Prometheus metrics
model_forensics_analysis_duration_seconds: Histogram
model_forensics_anomaly_score: Gauge
model_forensics_fix_rate: Gauge
model_forensics_bundle_size_bytes: Gauge

# Alerting rules
groups:
- name: model_forensics
  rules:
  - alert: HighAnomalyScore
    expr: model_forensics_anomaly_score > 0.8
    for: 5m
    annotations:
      summary: "High anomaly score detected: {{ $value }}"
  
  - alert: LowFixRate
    expr: model_forensics_fix_rate < 0.5
    for: 10m
    annotations:
      summary: "Low intervention success rate: {{ $value }}"
```

## Security

### Access Control

```yaml
# RBAC configuration
roles:
  analyst:
    permissions: [read_bundles, run_analysis, view_results]
  admin:
    permissions: [all]
  viewer:
    permissions: [view_results]

authentication:
  type: "oauth2"
  provider: "okta"
  scopes: ["model_forensics:analyze"]
```

### Data Protection

```bash
# Encrypt bundles at rest
export MODEL_FORENSICS_ENCRYPTION_KEY="base64_encoded_key"

# Network security
ufw allow from 10.0.0.0/8 to any port 8000
ufw enable

# Audit logging
export MODEL_FORENSICS_AUDIT_LOG="/var/log/audit/model_forensics.log"
```

## Troubleshooting

### Common Issues

1. **Memory Issues**
   ```bash
   # Reduce bundle size or enable streaming
   export MODEL_FORENSICS_STREAMING_ENABLED=true
   export MODEL_FORENSICS_MAX_MEMORY=16GB
   ```

2. **GPU Issues**
   ```bash
   # Force CPU fallback
   export MODEL_FORENSICS_DEVICE=cpu
   
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Performance Issues**
   ```bash
   # Enable debug logging
   export MODEL_FORENSICS_LOG_LEVEL=DEBUG
   
   # Profile analysis
   model-forensics analyze --profile incident_bundle/
   ```

### Log Analysis

```bash
# Error patterns
grep "ERROR" /var/log/model-forensics.log | tail -20

# Performance analysis
grep "execution_time" /var/log/model-forensics.log | awk '{print $5}' | sort -n

# Memory usage patterns
grep "memory_usage" /var/log/model-forensics.log | tail -10
```

## Maintenance

### Regular Tasks

```bash
# Daily: Clean old analysis results
find /data/results -name "*.json" -mtime +30 -delete

# Weekly: Update baselines
model-forensics build-baseline /data/benign_cases_$(date +%Y%m%d)/

# Monthly: Optimize storage
model-forensics compress-bundles /data/bundles/ --older-than 90d
```

### Updates

```bash
# Update framework
pip install --upgrade model-forensics

# Migrate configuration
model-forensics migrate-config --from v1.0 --to v1.1

# Test after update
model-forensics test --config /etc/model-forensics/config.yaml
```

## Support

- **Documentation**: https://model-forensics.readthedocs.io
- **Issues**: https://github.com/anthropics/model-forensics/issues  
- **Community**: https://discord.gg/model-forensics
- **Enterprise Support**: support@anthropic.com

## License

Model Forensics is released under the Apache 2.0 License. See LICENSE file for details.