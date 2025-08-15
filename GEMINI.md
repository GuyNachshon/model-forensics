# Project: Model Forensics

## Project Overview

This project, "Model Forensics," is a Python-based toolkit for conducting Root Cause Analysis (RCA) on foundation model failures. It provides a comprehensive framework for recording, replaying, and analyzing model incidents to understand the underlying causes of failures. The core technologies used are Python, PyTorch, and the Hugging Face Transformers library.

The architecture is modular, consisting of three main components:

*   **Recorder:** An SDK for instrumenting PyTorch models to capture execution traces, including activations and external calls.
*   **Replayer:** A sandboxed environment for deterministically replaying incidents, allowing for controlled interventions and analysis.
*   **RCA Engine:** A collection of analysis modules, including Compression Forensics (CF) for anomaly detection and Causal Minimal Fix Sets (CCA) for identifying the smallest interventions to correct failures.

The project is designed for researchers and practitioners who need to diagnose and understand model failures in a post-hoc manner, without requiring model retraining.

## Building and Running

The project uses `setuptools` for packaging and `pytest` for testing. The following commands are essential for development and execution:

**Installation:**

To install the project and its dependencies in development mode, run:

```bash
pip install -e ".[dev]"
```

**Running Tests:**

The test suite can be executed using `pytest`:

```bash
pytest
```

**Running the CLI:**

The project includes a command-line interface for analyzing incident bundles. The main entry point is `model-forensics`.

*   **Analyze an incident:**

    ```bash
    model-forensics analyze <path_to_incident_bundle> --model <model_name> --baseline <path_to_baseline_bundle>
    ```

*   **Batch analysis:**

    ```bash
    model-forensics batch <path_to_bundle_directory> --model <model_name>
    ```

*   **Triage bundles:**

    ```bash
    model-forensics triage <path_to_bundle_directory>
    ```

## Development Conventions

*   **Code Style:** The project uses `black` for code formatting and `isort` for import sorting.
*   **Type Checking:** `mypy` is used for static type checking.
*   **Linting:** `flake8` is used for linting.
*   **Pre-commit Hooks:** The project is set up to use pre-commit hooks to enforce code style and quality.
*   **Configuration:** The application is configured via YAML files, with a default configuration provided in `configs/default.yaml`.
*   **Testing:** Tests are located in the `tests/` directory and are written using `pytest`. The configuration in `pyproject.toml` specifies test paths and options.
