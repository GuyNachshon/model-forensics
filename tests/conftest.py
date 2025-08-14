"""Pytest configuration and shared fixtures."""

import pytest
import torch
import logging
from pathlib import Path
from transformers import GPT2LMHeadModel

from src.core.config import RCAConfig
from src.modules.cf import CompressionForensics
from src.modules.cca import CausalCCA
from src.modules.pipeline import RCAPipeline
from src.replayer.core import Replayer


# Configure logging for tests
logging.basicConfig(level=logging.WARNING)  # Reduce verbosity during tests


@pytest.fixture
def config():
    """Provide a default RCA configuration."""
    return RCAConfig()


@pytest.fixture
def cf_module(config):
    """Provide a CF module instance."""
    return CompressionForensics(config.modules.cf)


@pytest.fixture
def cca_module(config):
    """Provide a CCA module instance.""" 
    return CausalCCA(config.modules.cca)


@pytest.fixture
def pipeline(config):
    """Provide a RCA pipeline instance."""
    return RCAPipeline(config)


@pytest.fixture
def replayer(config):
    """Provide a replayer instance."""
    return Replayer(config)


@pytest.fixture
def small_model():
    """Provide a small GPT-2 model for testing."""
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    return model


@pytest.fixture
def test_bundle_path():
    """Provide path to test bundles directory."""
    return Path("test_bundles")


@pytest.fixture
def sample_activation():
    """Provide a sample activation tensor for testing."""
    return torch.randn(1, 10, 768)  # Batch, seq, hidden


@pytest.fixture
def test_data_dir(tmp_path):
    """Provide a temporary directory for test data."""
    return tmp_path / "test_data"


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests") 
    config.addinivalue_line("markers", "functional: Functional tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "gpu: GPU required tests")
    config.addinivalue_line("markers", "online: Internet required tests")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark tests based on directory structure
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "functional" in str(item.fspath):
            item.add_marker(pytest.mark.functional)
        
        # Mark slow tests (can be customized based on test names)
        if "pipeline" in str(item.fspath).lower() or "comprehensive" in str(item.fspath).lower():
            item.add_marker(pytest.mark.slow)