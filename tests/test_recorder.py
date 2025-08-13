"""Tests for the recording SDK."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil

from recorder import RecorderHooks, KVSketcher, BundleExporter
from recorder.hooks import ContextRecorder
from core.config import RCAConfig


class SimpleModel(nn.Module):
    """Simple test model."""
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


@pytest.fixture
def simple_model():
    """Create simple test model."""
    return SimpleModel()


@pytest.fixture
def test_config():
    """Create test configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = RCAConfig()
        config.bundle_dir = Path(temp_dir)
        yield config


@pytest.fixture
def recorder_hooks(test_config):
    """Create recorder hooks."""
    return RecorderHooks(test_config)


class TestRecorderHooks:
    """Test the RecorderHooks class."""
    
    def test_initialization(self, test_config):
        """Test recorder initialization."""
        hooks = RecorderHooks(test_config)
        assert not hooks.is_recording
        assert len(hooks.hooks) == 0
        assert hooks.config.granularity == "layer"
    
    def test_hook_installation(self, recorder_hooks, simple_model):
        """Test installing hooks on model."""
        recorder_hooks.install_hooks(simple_model)
        
        # Should install hooks on leaf modules only
        expected_hooks = ["linear1_forward", "linear2_forward", "relu_forward"]
        assert len(recorder_hooks.hooks) == len(expected_hooks)
        
        for hook_name in expected_hooks:
            assert hook_name in recorder_hooks.hooks
    
    def test_recording_lifecycle(self, recorder_hooks, simple_model):
        """Test complete recording lifecycle."""
        recorder_hooks.install_hooks(simple_model)
        
        # Start recording
        inputs = {"input": torch.randn(1, 10)}
        recorder_hooks.start_recording(inputs)
        assert recorder_hooks.is_recording
        
        # Run model
        with torch.no_grad():
            output = simple_model(inputs["input"])
        
        # Set outputs and stop recording
        recorder_hooks.set_outputs({"output": output})
        trace_data = recorder_hooks.stop_recording()
        
        assert not recorder_hooks.is_recording
        assert trace_data is not None
        assert len(trace_data.activations) > 0
        assert trace_data.inputs == inputs
        assert "output" in trace_data.outputs
    
    def test_external_call_recording(self, recorder_hooks):
        """Test recording external calls."""
        inputs = {"test": "input"}
        recorder_hooks.start_recording(inputs)
        
        # Record external call
        recorder_hooks.record_external_call(
            call_type="api_call",
            args={"endpoint": "/test", "data": {"key": "value"}},
            response={"status": "success"},
            metadata={"duration_ms": 150}
        )
        
        trace_data = recorder_hooks.stop_recording()
        
        assert len(trace_data.external_calls) == 1
        call = trace_data.external_calls[0]
        assert call["call_type"] == "api_call"
        assert call["args"]["endpoint"] == "/test"
        assert call["response"]["status"] == "success"
    
    def test_cleanup(self, recorder_hooks, simple_model):
        """Test proper cleanup of hooks."""
        recorder_hooks.install_hooks(simple_model)
        assert len(recorder_hooks.hooks) > 0
        
        recorder_hooks.cleanup()
        assert len(recorder_hooks.hooks) == 0
        assert not recorder_hooks.is_recording


class TestKVSketcher:
    """Test the KVSketcher class."""
    
    def test_compression_decompression(self, test_config):
        """Test activation compression and decompression."""
        sketcher = KVSketcher(test_config.recorder)
        
        # Create test activation
        activation = torch.randn(2, 10, 5)
        layer_name = "test_layer"
        
        # Compress
        sketch = sketcher.compress_activation(activation, layer_name)
        
        assert sketch.original_shape == activation.shape
        assert sketch.dtype == str(activation.dtype)
        assert sketch.compression_ratio > 0
        assert isinstance(sketch.data, bytes)
        
        # Decompress
        restored = sketcher.decompress_sketch(sketch)
        
        # Should restore to same shape
        assert restored.shape == activation.shape
    
    def test_compression_stats(self, test_config):
        """Test compression statistics tracking."""
        sketcher = KVSketcher(test_config.recorder)
        
        activation = torch.randn(5, 20)
        sketcher.compress_activation(activation, "layer1")
        
        stats = sketcher.get_compression_stats()
        assert "layer1" in stats
        assert "original_size" in stats["layer1"]
        assert "compressed_size" in stats["layer1"]
        assert "compression_ratio" in stats["layer1"]
    
    def test_bundle_size_estimation(self, test_config):
        """Test bundle size estimation."""
        sketcher = KVSketcher(test_config.recorder)
        
        activations = {}
        for i in range(3):
            activation = torch.randn(10, 10)
            sketch = sketcher.compress_activation(activation, f"layer_{i}")
            activations[f"layer_{i}"] = sketch
        
        size_mb = sketcher.estimate_bundle_size(activations)
        assert size_mb > 0
        assert isinstance(size_mb, float)


class TestBundleExporter:
    """Test the BundleExporter class."""
    
    def test_bundle_creation_and_loading(self, test_config, simple_model):
        """Test creating and loading bundles."""
        # Create trace data
        hooks = RecorderHooks(test_config)
        hooks.install_hooks(simple_model)
        
        inputs = {"input": torch.randn(1, 10)}
        hooks.start_recording(inputs)
        
        with torch.no_grad():
            output = simple_model(inputs["input"])
        
        hooks.set_outputs({"output": output})
        trace_data = hooks.stop_recording()
        
        # Create bundle
        exporter = BundleExporter(test_config)
        bundle = exporter.create_bundle(trace_data)
        
        assert bundle.bundle_path.exists()
        assert (bundle.bundle_path / "manifest.json").exists()
        assert (bundle.bundle_path / "inputs.json").exists()
        assert (bundle.bundle_path / "outputs.json").exists()
        assert (bundle.bundle_path / "activations").exists()
        
        # Load bundle
        loaded_bundle = exporter.load_bundle(bundle.bundle_path)
        
        assert loaded_bundle.bundle_id == bundle.bundle_id
        assert len(loaded_bundle.trace_data.activations) == len(trace_data.activations)
        assert loaded_bundle.trace_data.inputs.keys() == trace_data.inputs.keys()
    
    def test_manifest_creation(self, test_config):
        """Test manifest creation."""
        from core.types import TraceData
        
        trace_data = TraceData(
            inputs={"test": "input"},
            activations={},
            outputs={"test": "output"},
            external_calls=[],
            metadata={"test": "metadata"},
            timestamp="2024-01-01T00:00:00",
            model_config={"test": "config"}
        )
        
        exporter = BundleExporter(test_config)
        manifest = exporter._create_manifest(trace_data, "test_bundle")
        
        assert manifest["bundle_id"] == "test_bundle"
        assert "created_at" in manifest
        assert "components" in manifest
        assert "stats" in manifest
        assert manifest["stats"]["num_activations"] == 0
        assert manifest["stats"]["num_external_calls"] == 0


@pytest.mark.integration
class TestRecorderIntegration:
    """Integration tests for the complete recording workflow."""
    
    def test_complete_workflow(self, test_config):
        """Test complete workflow from recording to bundle creation."""
        model = SimpleModel()
        
        # Setup recorder
        hooks = RecorderHooks(test_config)
        hooks.install_hooks(model)
        
        # Record incident
        inputs = {"input": torch.randn(2, 10)}
        
        recorder = ContextRecorder(hooks, inputs)
        with recorder:
            # Simulate external call
            hooks.record_external_call(
                call_type="database_query",
                args={"query": "SELECT * FROM users"},
                response={"rows": 42}
            )
            
            # Run model
            with torch.no_grad():
                output = model(inputs["input"])
            
            hooks.set_outputs({"prediction": output})
        
        trace_data = recorder.get_trace()
        assert trace_data is not None
        
        # Create bundle
        exporter = BundleExporter(test_config)
        bundle = exporter.create_bundle(trace_data, "test_incident")
        
        # Verify bundle
        assert bundle.bundle_id == "test_incident"
        assert bundle.size_mb > 0
        assert len(bundle.trace_data.activations) > 0
        assert len(bundle.trace_data.external_calls) == 1
        
        # Test loading
        loaded_bundle = exporter.load_bundle(bundle.bundle_path)
        assert loaded_bundle.bundle_id == bundle.bundle_id


if __name__ == "__main__":
    pytest.main([__file__])