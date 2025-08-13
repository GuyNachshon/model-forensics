"""Core replayer functionality for model forensics."""

import logging
from typing import Dict, Optional, List, Any, Union
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
import uuid

from core.types import (
    IncidentBundle, TraceData, ReplaySession, ActivationDict, 
    InterventionResult, Intervention, InterventionType
)
from core.config import RCAConfig
from recorder.exporter import BundleExporter
from recorder.sketcher import BenignDonorPool
from replayer.reconstruction import ActivationReconstructor
from replayer.interventions import InterventionEngine

logger = logging.getLogger(__name__)


class Replayer:
    """Main replayer class for bundle analysis and replay."""
    
    def __init__(self, config: Optional[RCAConfig] = None):
        self.config = config or RCAConfig()
        self.reconstructor = ActivationReconstructor(self.config)
        self.intervention_engine = InterventionEngine(self.config)
        self.bundle_exporter = BundleExporter(self.config)
        
        # Active sessions
        self.sessions: Dict[str, ReplaySession] = {}
        
        logger.info("Initialized Replayer")
    
    def load_bundle(self, bundle_path: Union[str, Path]) -> IncidentBundle:
        """Load a bundle from disk."""
        bundle_path = Path(bundle_path)
        logger.info(f"Loading bundle from: {bundle_path}")
        
        try:
            bundle = self.bundle_exporter.load_bundle(bundle_path)
            logger.info(f"Successfully loaded bundle: {bundle.bundle_id}")
            return bundle
            
        except Exception as e:
            logger.error(f"Failed to load bundle: {e}")
            raise
    
    def create_session(self, bundle: IncidentBundle, model: nn.Module, 
                      session_id: Optional[str] = None) -> ReplaySession:
        """Create a new replay session."""
        if session_id is None:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Creating replay session: {session_id}")
        
        # Reconstruct activations
        reconstructed_activations = self.reconstructor.reconstruct_bundle_activations(bundle)
        
        # Create session
        session = ReplaySession(
            bundle=bundle,
            model=model,
            reconstructed_activations=reconstructed_activations,
            interventions=[],
            session_id=session_id,
            created_at=datetime.now().isoformat(),
            metadata={
                "model_type": type(model).__name__,
                "num_reconstructed": len(reconstructed_activations),
                "bundle_id": bundle.bundle_id
            }
        )
        
        # Store session
        self.sessions[session_id] = session
        
        logger.info(f"Created session {session_id} with {len(reconstructed_activations)} reconstructed activations")
        return session
    
    def get_session(self, session_id: str) -> Optional[ReplaySession]:
        """Get an existing session."""
        return self.sessions.get(session_id)
    
    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self.sessions.keys())
    
    def replay_original(self, session: ReplaySession, validate: bool = True) -> Dict[str, Any]:
        """Replay the original execution from the session."""
        logger.info(f"Replaying original execution for session: {session.session_id}")
        
        # Get original inputs
        original_inputs = session.bundle.trace_data.inputs
        
        # Run model with original inputs
        with torch.no_grad():
            # Convert inputs if needed
            model_inputs = self._prepare_model_inputs(original_inputs)
            replay_output = session.model(**model_inputs)
        
        # Compare with stored outputs if validation enabled
        validation_results = {}
        if validate:
            original_outputs = session.bundle.trace_data.outputs
            validation_results = self._validate_replay_outputs(original_outputs, replay_output)
        
        return {
            "replay_output": replay_output,
            "validation": validation_results,
            "session_id": session.session_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def apply_intervention(self, session: ReplaySession, intervention: Intervention) -> InterventionResult:
        """Apply a single intervention and record the result."""
        logger.info(f"Applying intervention: {intervention.type.value} on {intervention.layer_name}")
        
        try:
            result = self.intervention_engine.apply_intervention(
                session.model, 
                session.reconstructed_activations,
                intervention,
                session.bundle.trace_data.inputs
            )
            
            # Add to session history
            session.interventions.append(result)
            
            logger.info(f"Intervention applied: flip_success={result.flip_success}, "
                       f"confidence={result.confidence:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to apply intervention: {e}")
            raise
    
    def apply_interventions(self, session: ReplaySession, interventions: List[Intervention]) -> List[InterventionResult]:
        """Apply multiple interventions in sequence."""
        logger.info(f"Applying {len(interventions)} interventions")
        
        results = []
        for i, intervention in enumerate(interventions):
            try:
                result = self.apply_intervention(session, intervention)
                results.append(result)
                logger.debug(f"Intervention {i+1}/{len(interventions)} completed")
                
            except Exception as e:
                logger.error(f"Intervention {i+1} failed: {e}")
                # Continue with remaining interventions
                
        return results
    
    def patch_with_benign_donor(self, session: ReplaySession, layer_name: str, 
                               donor_pool: BenignDonorPool) -> InterventionResult:
        """Patch a layer with a similar benign activation."""
        logger.info(f"Patching {layer_name} with benign donor")
        
        if layer_name not in session.reconstructed_activations:
            raise ValueError(f"Layer {layer_name} not found in reconstructed activations")
        
        target_activation = session.reconstructed_activations[layer_name]
        
        # Find similar benign donor
        donor = donor_pool.get_similar_donor(layer_name, target_activation.numpy())
        if donor is None:
            raise ValueError(f"No suitable benign donor found for {layer_name}")
        
        # Create patch intervention
        intervention = Intervention(
            type=InterventionType.PATCH,
            layer_name=layer_name,
            donor_activation=torch.from_numpy(donor),
            metadata={"patch_source": "benign_donor_pool"}
        )
        
        return self.apply_intervention(session, intervention)
    
    def compare_activations(self, session: ReplaySession, layer_name: str, 
                          baseline_activation: torch.Tensor) -> Dict[str, float]:
        """Compare reconstructed activation with a baseline."""
        if layer_name not in session.reconstructed_activations:
            raise ValueError(f"Layer {layer_name} not found in session")
        
        current = session.reconstructed_activations[layer_name]
        return self.reconstructor.validate_reconstruction(baseline_activation, current)
    
    def export_session_results(self, session: ReplaySession, output_dir: Optional[Path] = None) -> Path:
        """Export session results to disk."""
        if output_dir is None:
            output_dir = self.config.output_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        session_file = output_dir / f"session_{session.session_id}_results.json"
        
        # Prepare serializable results
        results = {
            "session_id": session.session_id,
            "bundle_id": session.bundle.bundle_id,
            "created_at": session.created_at,
            "metadata": session.metadata,
            "interventions": [
                {
                    "type": result.intervention.type.value,
                    "layer_name": result.intervention.layer_name,
                    "flip_success": result.flip_success,
                    "confidence": result.confidence,
                    "side_effects": result.side_effects,
                    "metadata": result.metadata
                }
                for result in session.interventions
            ],
            "reconstruction_stats": self.reconstructor.get_reconstruction_stats()
        }
        
        import json
        with open(session_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Exported session results to: {session_file}")
        return session_file
    
    def _prepare_model_inputs(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert stored inputs to model inputs."""
        model_inputs = {}
        
        for key, value in inputs.items():
            if isinstance(value, dict) and value.get("__tensor__"):
                # Reconstruct tensor from serialized format
                tensor_data = torch.tensor(value["data"])
                tensor_data = tensor_data.reshape(value["shape"])
                model_inputs[key] = tensor_data
            elif isinstance(value, (list, tuple)):
                model_inputs[key] = torch.tensor(value)
            else:
                model_inputs[key] = value
        
        return model_inputs
    
    def _validate_replay_outputs(self, original: Dict[str, Any], replayed: Any) -> Dict[str, Any]:
        """Validate replayed outputs against original."""
        validation = {"status": "unknown", "details": {}}
        
        try:
            if isinstance(replayed, torch.Tensor):
                # Single tensor output
                if "logits" in original:
                    original_tensor = torch.tensor(original["logits"]["data"]).reshape(original["logits"]["shape"])
                    mse = torch.nn.functional.mse_loss(original_tensor, replayed).item()
                    validation = {
                        "status": "success",
                        "mse": mse,
                        "shapes_match": original_tensor.shape == replayed.shape
                    }
            
        except Exception as e:
            validation = {
                "status": "error", 
                "error": str(e)
            }
        
        return validation
    
    def cleanup_session(self, session_id: str) -> bool:
        """Clean up a session and free resources."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleaned up session: {session_id}")
            return True
        return False
    
    def cleanup_all_sessions(self) -> int:
        """Clean up all sessions."""
        count = len(self.sessions)
        self.sessions.clear()
        logger.info(f"Cleaned up {count} sessions")
        return count