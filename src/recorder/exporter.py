import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np

from core.types import TraceData, IncidentBundle, CompressedSketch
from core.config import RCAConfig


logger = logging.getLogger(__name__)


class BundleExporter:
    """Creates self-contained incident bundles for analysis."""
    
    def __init__(self, config: Optional[RCAConfig] = None):
        self.config = config or RCAConfig()
        
    def create_bundle(self, trace_data: TraceData, bundle_id: Optional[str] = None) -> IncidentBundle:
        """Create incident bundle from trace data."""
        if bundle_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bundle_id = f"incident_{timestamp}"
        
        bundle_path = self.config.bundle_dir / bundle_id
        bundle_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating bundle: {bundle_id}")
        
        # Create manifest
        manifest = self._create_manifest(trace_data, bundle_id)
        
        # Export components
        self._export_inputs(trace_data, bundle_path)
        self._export_activations(trace_data, bundle_path)
        self._export_external_calls(trace_data, bundle_path)
        self._export_outputs(trace_data, bundle_path)
        self._export_model_config(trace_data, bundle_path)
        self._export_metadata(trace_data, bundle_path)
        
        # Save manifest with checksums
        manifest["checksums"] = self._calculate_checksums(bundle_path)
        self._save_manifest(manifest, bundle_path)
        
        bundle = IncidentBundle(
            bundle_id=bundle_id,
            manifest=manifest,
            trace_data=trace_data,
            bundle_path=bundle_path
        )
        
        logger.info(f"Created bundle {bundle_id} ({bundle.size_mb:.1f} MB)")
        return bundle
    
    def _create_manifest(self, trace_data: TraceData, bundle_id: str) -> Dict[str, Any]:
        """Create bundle manifest with metadata."""
        return {
            "bundle_id": bundle_id,
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "trace_timestamp": trace_data.timestamp,
            "components": {
                "inputs": "inputs.json",
                "activations": "activations/",
                "external_calls": "external_calls.jsonl",
                "outputs": "outputs.json",
                "model_config": "model_config.json",
                "metadata": "metadata.json"
            },
            "stats": {
                "num_activations": len(trace_data.activations),
                "num_external_calls": len(trace_data.external_calls),
                "input_keys": list(trace_data.inputs.keys()),
                "output_keys": list(trace_data.outputs.keys())
            }
        }
    
    def _export_inputs(self, trace_data: TraceData, bundle_path: Path) -> None:
        """Export input data to JSON."""
        inputs_file = bundle_path / "inputs.json"
        
        # Convert tensors to serializable format
        serializable_inputs = self._make_serializable(trace_data.inputs)
        
        with open(inputs_file, 'w') as f:
            json.dump(serializable_inputs, f, indent=2)
        
        logger.debug(f"Exported inputs to {inputs_file}")
    
    def _export_activations(self, trace_data: TraceData, bundle_path: Path) -> None:
        """Export compressed activations."""
        activations_dir = bundle_path / "activations"
        activations_dir.mkdir(exist_ok=True)
        
        for layer_name, sketch in trace_data.activations.items():
            # Create safe filename
            safe_name = self._sanitize_filename(layer_name)
            activation_file = activations_dir / f"{safe_name}.npz"
            
            # Save compressed sketch
            np.savez_compressed(
                activation_file,
                data=sketch.data,
                original_shape=sketch.original_shape,
                dtype=sketch.dtype,
                compression_ratio=sketch.compression_ratio,
                metadata=json.dumps(sketch.metadata)
            )
        
        logger.debug(f"Exported {len(trace_data.activations)} activations")
    
    def _export_external_calls(self, trace_data: TraceData, bundle_path: Path) -> None:
        """Export external calls as JSONL."""
        calls_file = bundle_path / "external_calls.jsonl"
        
        with open(calls_file, 'w') as f:
            for call in trace_data.external_calls:
                serializable_call = self._make_serializable(call)
                f.write(json.dumps(serializable_call) + '\n')
        
        logger.debug(f"Exported {len(trace_data.external_calls)} external calls")
    
    def _export_outputs(self, trace_data: TraceData, bundle_path: Path) -> None:
        """Export output data to JSON."""
        outputs_file = bundle_path / "outputs.json"
        
        serializable_outputs = self._make_serializable(trace_data.outputs)
        
        with open(outputs_file, 'w') as f:
            json.dump(serializable_outputs, f, indent=2)
        
        logger.debug(f"Exported outputs to {outputs_file}")
    
    def _export_model_config(self, trace_data: TraceData, bundle_path: Path) -> None:
        """Export model configuration."""
        config_file = bundle_path / "model_config.json"
        
        with open(config_file, 'w') as f:
            json.dump(trace_data.model_config, f, indent=2)
        
        logger.debug(f"Exported model config to {config_file}")
    
    def _export_metadata(self, trace_data: TraceData, bundle_path: Path) -> None:
        """Export trace metadata."""
        metadata_file = bundle_path / "metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(trace_data.metadata, f, indent=2)
        
        logger.debug(f"Exported metadata to {metadata_file}")
    
    def _save_manifest(self, manifest: Dict[str, Any], bundle_path: Path) -> None:
        """Save manifest with checksums."""
        manifest_file = bundle_path / "manifest.json"
        
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def _calculate_checksums(self, bundle_path: Path) -> Dict[str, str]:
        """Calculate checksums for all files in bundle."""
        checksums = {}
        
        for file_path in bundle_path.rglob('*'):
            if file_path.is_file() and file_path.name != "manifest.json":
                rel_path = file_path.relative_to(bundle_path)
                with open(file_path, 'rb') as f:
                    content = f.read()
                    checksums[str(rel_path)] = hashlib.sha256(content).hexdigest()
        
        return checksums
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'numpy'):  # PyTorch tensor
            return {
                "__tensor__": True,
                "data": obj.detach().cpu().numpy().tolist(),
                "shape": list(obj.shape),
                "dtype": str(obj.dtype)
            }
        elif isinstance(obj, np.ndarray):
            return {
                "__array__": True,
                "data": obj.tolist(),
                "shape": list(obj.shape),
                "dtype": str(obj.dtype)
            }
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    def _sanitize_filename(self, name: str) -> str:
        """Create safe filename from layer name."""
        # Replace problematic characters
        safe_name = name.replace('/', '_').replace('\\', '_').replace('.', '_')
        safe_name = safe_name.replace(':', '_').replace('*', '_').replace('?', '_')
        safe_name = safe_name.replace('<', '_').replace('>', '_').replace('|', '_')
        
        # Limit length
        if len(safe_name) > 100:
            safe_name = safe_name[:100]
        
        return safe_name
    
    def load_bundle(self, bundle_path: Path) -> IncidentBundle:
        """Load existing bundle from disk."""
        bundle_path = Path(bundle_path)
        
        if not bundle_path.exists():
            raise FileNotFoundError(f"Bundle not found: {bundle_path}")
        
        # Load manifest
        manifest_file = bundle_path / "manifest.json"
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        
        # Verify checksums
        if not self._verify_checksums(manifest, bundle_path):
            logger.warning(f"Checksum verification failed for bundle: {bundle_path}")
        
        # Load trace data
        trace_data = self._load_trace_data(manifest, bundle_path)
        
        bundle = IncidentBundle(
            bundle_id=manifest["bundle_id"],
            manifest=manifest,
            trace_data=trace_data,
            bundle_path=bundle_path
        )
        
        logger.info(f"Loaded bundle {bundle.bundle_id} ({bundle.size_mb:.1f} MB)")
        return bundle
    
    def _verify_checksums(self, manifest: Dict[str, Any], bundle_path: Path) -> bool:
        """Verify bundle integrity using checksums."""
        stored_checksums = manifest.get("checksums", {})
        current_checksums = self._calculate_checksums(bundle_path)
        
        for file_path, stored_checksum in stored_checksums.items():
            current_checksum = current_checksums.get(file_path)
            if current_checksum != stored_checksum:
                logger.error(f"Checksum mismatch for {file_path}")
                return False
        
        return True
    
    def _load_trace_data(self, manifest: Dict[str, Any], bundle_path: Path) -> TraceData:
        """Load trace data from bundle components."""
        components = manifest["components"]
        
        # Load inputs
        with open(bundle_path / components["inputs"], 'r') as f:
            inputs = json.load(f)
        
        # Load outputs  
        with open(bundle_path / components["outputs"], 'r') as f:
            outputs = json.load(f)
        
        # Load model config
        with open(bundle_path / components["model_config"], 'r') as f:
            model_config = json.load(f)
        
        # Load metadata
        with open(bundle_path / components["metadata"], 'r') as f:
            metadata = json.load(f)
        
        # Load external calls
        external_calls = []
        calls_file = bundle_path / components["external_calls"]
        if calls_file.exists():
            with open(calls_file, 'r') as f:
                for line in f:
                    if line.strip():
                        external_calls.append(json.loads(line))
        
        # Load activations
        activations = {}
        activations_dir = bundle_path / components["activations"]
        if activations_dir.exists():
            for activation_file in activations_dir.glob("*.npz"):
                data = np.load(activation_file, allow_pickle=True)
                
                sketch = CompressedSketch(
                    data=data["data"].item(),
                    original_shape=tuple(data["original_shape"]),
                    dtype=str(data["dtype"]),
                    compression_ratio=float(data["compression_ratio"]),
                    metadata=json.loads(str(data["metadata"]))
                )
                
                # Extract original layer name from filename
                layer_name = sketch.metadata.get("layer_name", activation_file.stem)
                activations[layer_name] = sketch
        
        return TraceData(
            inputs=inputs,
            activations=activations,
            outputs=outputs,
            external_calls=external_calls,
            metadata=metadata,
            timestamp=manifest["trace_timestamp"],
            model_config=model_config
        )