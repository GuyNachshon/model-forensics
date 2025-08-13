"""Intervention engine for applying modifications during replay."""

import logging
from typing import Dict, Optional, List, Any, Callable
import torch
import torch.nn as nn
import numpy as np
from contextlib import contextmanager

from core.types import (
    Intervention, InterventionResult, InterventionType, 
    ActivationDict, InterventionHook
)
from core.config import RCAConfig

logger = logging.getLogger(__name__)


class InterventionEngine:
    """Engine for applying interventions during model replay."""
    
    def __init__(self, config: Optional[RCAConfig] = None):
        self.config = config or RCAConfig()
        self.active_hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self.intervention_data: Dict[str, Any] = {}
        
    def apply_intervention(self, model: nn.Module, activations: ActivationDict, 
                          intervention: Intervention, inputs: Dict[str, Any]) -> InterventionResult:
        """Apply a single intervention and return the result."""
        logger.debug(f"Applying {intervention.type.value} intervention on {intervention.layer_name}")
        
        # Get original activation
        if intervention.layer_name not in activations:
            raise ValueError(f"Layer {intervention.layer_name} not found in activations")
        
        original_activation = activations[intervention.layer_name].clone()
        
        # Get original model output (without intervention)
        with torch.no_grad():
            model_inputs = self._prepare_inputs(inputs)
            original_output = model(**model_inputs)
        
        # Apply intervention and get modified output
        modified_activation = self._apply_intervention_to_activation(original_activation, intervention)
        modified_output = self._run_with_intervention(model, inputs, intervention, modified_activation)
        
        # Evaluate intervention success
        flip_success = self._evaluate_flip_success(original_output, modified_output)
        confidence = self._calculate_confidence(original_activation, modified_activation, intervention)
        side_effects = self._calculate_side_effects(original_output, modified_output)
        
        result = InterventionResult(
            intervention=intervention,
            original_activation=original_activation,
            modified_activation=modified_activation,
            original_output=original_output,
            modified_output=modified_output,
            flip_success=flip_success,
            confidence=confidence,
            side_effects=side_effects,
            metadata={
                "intervention_timestamp": torch.cuda.Event().record() if torch.cuda.is_available() else None,
                "activation_norm_change": torch.norm(modified_activation - original_activation).item()
            }
        )
        
        return result
    
    def _apply_intervention_to_activation(self, activation: torch.Tensor, 
                                        intervention: Intervention) -> torch.Tensor:
        """Apply intervention to a single activation tensor."""
        modified = activation.clone()
        
        if intervention.type == InterventionType.ZERO:
            if intervention.indices is not None:
                flat_modified = modified.flatten()
                flat_modified[intervention.indices] = 0.0
                modified = flat_modified.reshape(activation.shape)
            else:
                modified = torch.zeros_like(activation)
                
        elif intervention.type == InterventionType.MEAN:
            if intervention.value is not None:
                mean_value = intervention.value
            else:
                mean_value = activation.mean()
            
            if intervention.indices is not None:
                flat_modified = modified.flatten()
                flat_modified[intervention.indices] = mean_value
                modified = flat_modified.reshape(activation.shape)
            else:
                modified = torch.full_like(activation, mean_value)
                
        elif intervention.type == InterventionType.PATCH:
            if intervention.donor_activation is not None:
                donor = intervention.donor_activation
                if donor.shape != activation.shape:
                    donor = donor.reshape(activation.shape)
                modified = donor.clone()
            else:
                raise ValueError("Patch intervention requires donor_activation")
                
        elif intervention.type == InterventionType.NOISE:
            noise_std = intervention.value or 0.1
            noise = torch.randn_like(activation) * noise_std
            modified = activation + noise
            
        elif intervention.type == InterventionType.SCALE:
            scale_factor = intervention.scale_factor or 0.5
            if intervention.indices is not None:
                flat_modified = modified.flatten()
                flat_modified[intervention.indices] *= scale_factor
                modified = flat_modified.reshape(activation.shape)
            else:
                modified = activation * scale_factor
                
        elif intervention.type == InterventionType.CLAMP:
            if intervention.clamp_range is not None:
                min_val, max_val = intervention.clamp_range
                modified = torch.clamp(activation, min_val, max_val)
            else:
                # Default clamp to [-1, 1]
                modified = torch.clamp(activation, -1.0, 1.0)
                
        else:
            raise ValueError(f"Unknown intervention type: {intervention.type}")
        
        return modified
    
    def _run_with_intervention(self, model: nn.Module, inputs: Dict[str, Any], 
                             intervention: Intervention, modified_activation: torch.Tensor) -> Any:
        """Run model with intervention applied via hooks."""
        # Store intervention data for hook access
        self.intervention_data = {
            "layer_name": intervention.layer_name,
            "modified_activation": modified_activation
        }
        
        # Install intervention hook
        hook_handle = self._install_intervention_hook(model, intervention.layer_name)
        
        try:
            with torch.no_grad():
                model_inputs = self._prepare_inputs(inputs)
                output = model(**model_inputs)
            return output
            
        finally:
            # Clean up hook
            if hook_handle:
                hook_handle.remove()
            self.intervention_data = {}
    
    def _install_intervention_hook(self, model: nn.Module, layer_name: str) -> Optional[torch.utils.hooks.RemovableHandle]:
        """Install a hook to intercept and modify activations."""
        target_module = None
        
        # Find the target module
        for name, module in model.named_modules():
            if name == layer_name:
                target_module = module
                break
        
        if target_module is None:
            logger.warning(f"Could not find module: {layer_name}")
            return None
        
        def intervention_hook(module, input, output):
            # Replace output with modified activation
            modified = self.intervention_data.get("modified_activation")
            if modified is not None:
                # Ensure shapes match
                if isinstance(output, torch.Tensor):
                    if output.shape == modified.shape:
                        return modified
                    else:
                        logger.warning(f"Shape mismatch in intervention: {output.shape} vs {modified.shape}")
                        return modified.reshape(output.shape)
                elif isinstance(output, (tuple, list)):
                    # Handle multiple outputs - replace first tensor
                    if len(output) > 0 and isinstance(output[0], torch.Tensor):
                        output_list = list(output)
                        output_list[0] = modified.reshape(output[0].shape)
                        return type(output)(output_list)
            
            return output
        
        return target_module.register_forward_hook(intervention_hook)
    
    def _evaluate_flip_success(self, original_output: Any, modified_output: Any) -> bool:
        """Evaluate if intervention successfully flipped the output behavior."""
        try:
            if isinstance(original_output, torch.Tensor) and isinstance(modified_output, torch.Tensor):
                # For classification: check if predicted class changed
                if original_output.ndim >= 2:  # Batch of predictions
                    orig_pred = torch.argmax(original_output, dim=-1)
                    mod_pred = torch.argmax(modified_output, dim=-1)
                    return not torch.equal(orig_pred, mod_pred)
                else:
                    # Single prediction
                    return torch.argmax(original_output) != torch.argmax(modified_output)
            
            # For other output types, use simple inequality
            return not torch.equal(original_output, modified_output) if isinstance(original_output, torch.Tensor) else False
            
        except Exception as e:
            logger.warning(f"Could not evaluate flip success: {e}")
            return False
    
    def _calculate_confidence(self, original: torch.Tensor, modified: torch.Tensor, 
                            intervention: Intervention) -> float:
        """Calculate confidence score for the intervention."""
        try:
            # Base confidence depends on intervention type
            base_confidence = {
                InterventionType.ZERO: 0.9,
                InterventionType.MEAN: 0.8,
                InterventionType.PATCH: 0.85,
                InterventionType.NOISE: 0.6,
                InterventionType.SCALE: 0.7,
                InterventionType.CLAMP: 0.75
            }.get(intervention.type, 0.5)
            
            # Adjust based on magnitude of change
            change_magnitude = torch.norm(modified - original) / torch.norm(original)
            
            # Moderate changes are more confident
            if 0.1 < change_magnitude < 0.5:
                confidence = base_confidence * 1.1
            elif change_magnitude > 0.8:
                confidence = base_confidence * 0.8
            else:
                confidence = base_confidence
            
            return min(1.0, max(0.0, confidence))
            
        except Exception:
            return 0.5
    
    def _calculate_side_effects(self, original_output: Any, modified_output: Any) -> Dict[str, float]:
        """Calculate side effect metrics."""
        side_effects = {}
        
        try:
            if isinstance(original_output, torch.Tensor) and isinstance(modified_output, torch.Tensor):
                # Output distribution changes
                orig_flat = original_output.flatten()
                mod_flat = modified_output.flatten()
                
                # L2 change
                side_effects["l2_change"] = torch.norm(mod_flat - orig_flat).item()
                
                # Distribution shift (if probabilistic)
                if torch.allclose(orig_flat.sum(), torch.tensor(1.0), atol=0.1):
                    # Treat as probability distribution
                    kl_div = torch.nn.functional.kl_div(
                        torch.log(mod_flat + 1e-8), 
                        orig_flat + 1e-8, 
                        reduction='sum'
                    ).item()
                    side_effects["kl_divergence"] = kl_div
                
                # Confidence change (for classification)
                if original_output.ndim >= 2:
                    orig_conf = torch.max(torch.softmax(original_output, dim=-1), dim=-1)[0].mean().item()
                    mod_conf = torch.max(torch.softmax(modified_output, dim=-1), dim=-1)[0].mean().item()
                    side_effects["confidence_change"] = abs(orig_conf - mod_conf)
            
        except Exception as e:
            logger.warning(f"Could not calculate side effects: {e}")
            side_effects["error"] = 1.0
        
        return side_effects
    
    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare inputs for model execution."""
        model_inputs = {}
        
        for key, value in inputs.items():
            if isinstance(value, dict) and value.get("__tensor__"):
                # Reconstruct tensor from serialized format
                tensor_data = torch.tensor(value["data"])
                tensor_data = tensor_data.reshape(value["shape"])
                model_inputs[key] = tensor_data
            elif isinstance(value, (list, tuple)):
                model_inputs[key] = torch.tensor(value)
            elif isinstance(value, torch.Tensor):
                model_inputs[key] = value
            else:
                model_inputs[key] = value
        
        return model_inputs
    
    @contextmanager
    def batch_interventions(self, model: nn.Module, interventions: List[Intervention]):
        """Context manager for applying multiple interventions simultaneously."""
        hooks = []
        
        try:
            for intervention in interventions:
                hook = self._install_intervention_hook(model, intervention.layer_name)
                if hook:
                    hooks.append(hook)
            
            yield
            
        finally:
            for hook in hooks:
                hook.remove()
    
    def cleanup(self):
        """Clean up any remaining hooks."""
        for hook in self.active_hooks.values():
            hook.remove()
        self.active_hooks.clear()
        self.intervention_data = {}