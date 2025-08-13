"""Utility functions for model forensics."""

import logging
import random
import os
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import torch


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def ensure_reproducibility(seed: int = 42) -> None:
    """Ensure reproducible results across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(device_spec: str = "auto") -> torch.device:
    """Get the appropriate device for computation."""
    if device_spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_spec)


def format_bytes(bytes_count: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.1f} PB"


def safe_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Safely convert tensor to numpy array."""
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()


def create_directory_structure(base_path: Path, structure: Dict[str, Any]) -> None:
    """Create directory structure from nested dictionary."""
    for name, content in structure.items():
        path = base_path / name
        if isinstance(content, dict):
            path.mkdir(parents=True, exist_ok=True)
            create_directory_structure(path, content)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            if content is not None:
                path.write_text(str(content))


def validate_model_architecture(model: torch.nn.Module) -> Dict[str, Any]:
    """Extract and validate model architecture information."""
    info = {
        "model_type": type(model).__name__,
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "layers": [],
        "device": next(model.parameters()).device.type,
    }
    
    # Extract layer information
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            layer_info = {
                "name": name,
                "type": type(module).__name__,
                "params": sum(p.numel() for p in module.parameters()),
            }
            
            # Add layer-specific information
            if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                layer_info.update({
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                })
            elif hasattr(module, 'num_embeddings') and hasattr(module, 'embedding_dim'):
                layer_info.update({
                    "num_embeddings": module.num_embeddings,
                    "embedding_dim": module.embedding_dim,
                })
            
            info["layers"].append(layer_info)
    
    return info


class Timer:
    """Simple context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.start_time:
            self.start_time.record()
        else:
            import time
            self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(self.start_time, torch.cuda.Event):
            self.end_time = torch.cuda.Event(enable_timing=True)
            self.end_time.record()
            torch.cuda.synchronize()
            elapsed = self.start_time.elapsed_time(self.end_time) / 1000.0  # Convert to seconds
        else:
            import time
            elapsed = time.time() - self.start_time
        
        logging.info(f"{self.name} completed in {elapsed:.3f} seconds")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time is None:
            return 0.0
        
        if isinstance(self.start_time, torch.cuda.Event):
            return self.start_time.elapsed_time(self.end_time) / 1000.0
        else:
            return self.end_time - self.start_time