import glob
import importlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union, Set
from pathlib import Path
from enum import Enum

import requests
import yaml


@dataclass
class ExperimentConfig:
    """Configuration for experiment execution."""
    prompts_dir: Path
    experiment_id: str
    model: str
    samples_per_condition: int
    output_dir: Path
    concurrency_limits: Dict[str, int]
    provider_concurrency_limits: Dict[str, int] = None
    classification_enabled: bool = True
    resume_from: Optional[Path] = None
    verbose: bool = False
    rate_limits: Optional[Dict[str, Any]] = None
    debug: bool = False
    temperature: float = 1.0  # Temperature for model inference
    force_rerun: bool = False  # Ignore existing outputs and rerun all tasks


@dataclass
class SampleProgress:
    """Progress tracking for individual samples."""
    condition: str
    model: str
    sample_id: int
    status: str  # "pending", "in_progress", "completed", "failed"
    timestamp: Optional[str] = None
    error: Optional[str] = None


class ExperimentState(Enum):
    """Enumeration for experiment states."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    PARTIAL = "partial"


@dataclass
class ExperimentStatus:
    """Global experiment state for checkpointing."""
    experiment_id: str
    start_time: str
    config: ExperimentConfig
    total_samples: int
    completed_samples: int
    failed_samples: int
    status: ExperimentState  # "running", "completed", "failed", "paused"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    role: MessageRole
    content: str


@dataclass
class LLMResponse:
    model_id: str
    completion: str
    stop_reason: Optional[str] = None
    cost: Optional[float] = None
    duration: Optional[float] = None
    api_duration: Optional[float] = None
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize model name for use in directory paths by replacing slashes with underscores.

    Args:
        model_name: Original model name (e.g., "deepseek-ai/DeepSeek-R1")

    Returns:
        Sanitized model name (e.g., "deepseek-ai_DeepSeek-R1")
    """
    return model_name.replace("/", "_").replace(":", "_")


def get_max_tokens(model_name):
    hf_url = f"https://huggingface.co/{model_name}/raw/main/config.json"
    try:
        response = requests.get(hf_url)
        response.raise_for_status()
        config = response.json()
        typical_fields = ["max_position_embeddings", "n_positions", "seq_len", "seq_length", "n_ctx", "sliding_window"]
        context_windows = [config.get(field) for field in typical_fields if field in config]
        if context_windows:
            return context_windows[-1]
        return 2048
    except requests.RequestException as e:
        print(f"Error fetching model config for {model_name}: {e}")
        return 2048


def load_hf_model(model_id):
    hf_url = f"https://huggingface.co/api/models/{model_id}"
    response = requests.get(hf_url)
    response.raise_for_status()
    model_info = response.json()
    transformers_info = model_info.get("transformersInfo", {})
    storage_used = model_info.get("usedStorage", 0)

    # if storage size is above 50GB, warn but continue automatically
    if storage_used > 50 * 1024 * 1024 * 1024:
        print(f"Warning: Model {model_id} is large ({storage_used / (1024 * 1024 * 1024):.2f} GB). Continuing automatically...")
        # Continue automatically in non-interactive mode

    # Import transformers components properly
    import transformers
    
    model_module = transformers_info.get("auto_model", "AutoModel")
    tokenizer_module = transformers_info.get("processor", "AutoTokenizer")
    
    # Get the actual classes from transformers
    model_class = getattr(transformers, model_module)
    tokenizer_class = getattr(transformers, tokenizer_module)
    
    model = model_class.from_pretrained(model_id)
    tokenizer = tokenizer_class.from_pretrained(model_id)

    return model, tokenizer


def update_used_model(model_id):
    files = glob.glob("configs/*.yaml")
    for file in files:
        with open(file, 'r') as f:
            content = yaml.safe_load(f.read())

        content["global"]["models"] = [model_id]

        with open(file, 'w') as f:
            yaml.safe_dump(content, f, default_flow_style=False)
