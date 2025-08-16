import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
import torch
import torch.nn as nn
from datetime import datetime
import weakref

from core.types import TraceData, ActivationDict
from core.config import RecorderConfig, RCAConfig
from recorder.sketcher import KVSketcher, BenignDonorPool, AsyncBundleWriter


logger = logging.getLogger(__name__)


class RecorderHooks:
    """Drop-in recording SDK that hooks into model execution."""
    
    def __init__(self, config: Optional[RCAConfig] = None):
        """Initialize recorder with configuration."""
        self.config = config.recorder if config else RecorderConfig()
        self.sketcher = KVSketcher(self.config)
        
        # Initialize new components
        self.benign_donor_pool = BenignDonorPool(self.config) if self.config.enable_benign_donors else None
        self.async_writer = AsyncBundleWriter(self.config) if self.config.async_export else None
        
        # State tracking
        self.is_recording = False
        self.current_trace: Optional[TraceData] = None
        self.activations: Dict[str, torch.Tensor] = {}
        self.external_calls: List[Dict[str, Any]] = []
        
        # Hook management
        self.hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self.model_ref: Optional[weakref.ref] = None
        
        # Start async writer if enabled
        if self.async_writer:
            self.async_writer.start()
        
        logger.info(f"Initialized RecorderHooks with granularity: {self.config.granularity}")
    
    def install_hooks(self, model: nn.Module) -> None:
        """Install recording hooks on model."""
        if not self.config.enabled:
            logger.warning("Recording is disabled in config")
            return
            
        self.model_ref = weakref.ref(model)
        self._remove_existing_hooks()
        
        # Install hooks based on granularity
        if self.config.granularity == "layer":
            self._install_layer_hooks(model)
        elif self.config.granularity == "attention_head":
            self._install_attention_hooks(model)
        elif self.config.granularity == "neuron":
            self._install_neuron_hooks(model)
        else:
            raise ValueError(f"Unknown granularity: {self.config.granularity}")
        
        logger.info(f"Installed {len(self.hooks)} hooks on model")
    
    def _install_layer_hooks(self, model: nn.Module) -> None:
        """Install hooks at layer granularity."""
        for name, module in model.named_modules():
            # Skip container modules
            if len(list(module.children())) > 0:
                continue
                
            # Apply layer filtering if specified
            if self.config.layers_to_record and name not in self.config.layers_to_record:
                continue
            
            # Install forward hook
            hook = module.register_forward_hook(
                self._create_forward_hook(name)
            )
            self.hooks[f"{name}_forward"] = hook
            
            logger.debug(f"Installed forward hook on {name}")
    
    def _install_attention_hooks(self, model: nn.Module) -> None:
        """Install hooks on attention heads (transformer-specific)."""
        for name, module in model.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                if hasattr(module, 'num_heads'):
                    hook = module.register_forward_hook(
                        self._create_attention_hook(name)
                    )
                    self.hooks[f"{name}_attention"] = hook
                    logger.debug(f"Installed attention hook on {name}")
    
    def _install_neuron_hooks(self, model: nn.Module) -> None:
        """Install hooks at neuron granularity (expensive - use sparingly)."""
        logger.warning("Neuron-level recording is computationally expensive")
        # For now, fall back to layer-level
        self._install_layer_hooks(model)
    
    def _create_forward_hook(self, layer_name: str) -> Callable:
        """Create forward hook function for a specific layer."""
        def hook_fn(module: nn.Module, input: Tuple[torch.Tensor, ...], output: Any) -> None:
            if not self.is_recording:
                return
                
            # Sample based on sample_rate
            if torch.rand(1).item() > self.config.sample_rate:
                return
            
            try:
                # Handle different output types with SAFER memory operations
                if isinstance(output, torch.Tensor):
                    # CRITICAL: Clone first to avoid memory alignment issues during generation
                    # Don't move to CPU immediately - keep on same device as model
                    activation = output.detach().clone()
                    # Convert from BFloat16 to Float32 for better compatibility
                    if activation.dtype == torch.bfloat16:
                        activation = activation.to(torch.float32)
                    # Only move to CPU if it's safe (not during generation loop)
                    if not torch.is_grad_enabled():  # Only move to CPU during inference
                        activation = activation.cpu()
                    self.activations[layer_name] = activation
                    logger.debug(f"Recorded activation for {layer_name}: {activation.shape}")
                    
                elif isinstance(output, (tuple, list)):
                    # Handle multiple tensor outputs with same safety
                    for i, out in enumerate(output):
                        if isinstance(out, torch.Tensor):
                            activation = out.detach().clone()
                            if activation.dtype == torch.bfloat16:
                                activation = activation.to(torch.float32)
                            if not torch.is_grad_enabled():
                                activation = activation.cpu()
                            self.activations[f"{layer_name}_output_{i}"] = activation
                            
                elif hasattr(output, 'logits'):
                    # Handle transformer model outputs (like GPT-2 CausalLMOutput)
                    if isinstance(output.logits, torch.Tensor):
                        activation = output.logits.detach().clone()
                        if activation.dtype == torch.bfloat16:
                            activation = activation.to(torch.float32)
                        if not torch.is_grad_enabled():
                            activation = activation.cpu()
                        self.activations[f"{layer_name}_logits"] = activation
                        logger.debug(f"Recorded logits for {layer_name}: {activation.shape}")
                        
                elif hasattr(output, 'last_hidden_state'):
                    # Handle transformer hidden state outputs
                    if isinstance(output.last_hidden_state, torch.Tensor):
                        activation = output.last_hidden_state.detach().clone()
                        if activation.dtype == torch.bfloat16:
                            activation = activation.to(torch.float32)
                        if not torch.is_grad_enabled():
                            activation = activation.cpu()
                        self.activations[f"{layer_name}_hidden"] = activation
                        logger.debug(f"Recorded hidden state for {layer_name}: {activation.shape}")
                        
                else:
                    logger.debug(f"Skipping unsupported output type for {layer_name}: {type(output)}")
                    
            except Exception as e:
                logger.warning(f"Failed to record activation for {layer_name}: {e}")
        
        return hook_fn
    
    def _create_attention_hook(self, layer_name: str) -> Callable:
        """Create attention-specific hook."""
        def hook_fn(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if not self.is_recording:
                return
                
            # Store attention weights if available
            if hasattr(module, 'attention_weights'):
                weights = module.attention_weights.detach().cpu()
                self.activations[f"{layer_name}_attn_weights"] = weights
        
        return hook_fn
    
    def start_recording(self, inputs: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Start recording a new trace."""
        if self.is_recording:
            logger.warning("Already recording - stopping previous trace")
            self.stop_recording()
        
        self.is_recording = True
        self.activations.clear()
        self.external_calls.clear()
        
        # Initialize trace data
        self.current_trace = TraceData(
            inputs=inputs,
            activations={},  # Will be populated during recording
            outputs={},
            external_calls=[],
            metadata=metadata or {},
            timestamp=datetime.now().isoformat(),
            model_config=self._get_model_config()
        )
        
        logger.info("Started recording trace")
    
    def stop_recording(self) -> Optional[TraceData]:
        """Stop recording and return trace data."""
        if not self.is_recording:
            logger.warning("Not currently recording")
            return None
        
        self.is_recording = False
        
        if self.current_trace is None:
            logger.error("No current trace to stop")
            return None
        
        # Compress activations using sketcher
        compressed_activations = {}
        for layer_name, activation in self.activations.items():
            sketch = self.sketcher.compress_activation(activation, layer_name)
            compressed_activations[layer_name] = sketch
        
        # Finalize trace data
        self.current_trace.activations = compressed_activations
        self.current_trace.external_calls = self.external_calls.copy()
        
        logger.info(f"Stopped recording - captured {len(compressed_activations)} activations")
        return self.current_trace
    
    def record_external_call(self, call_type: str, args: Dict[str, Any], response: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record external API/DB call during execution."""
        if not self.is_recording:
            return
        
        call_record = {
            "timestamp": datetime.now().isoformat(),
            "call_type": call_type,
            "args": args,
            "response": response,
            "metadata": metadata or {}
        }
        
        self.external_calls.append(call_record)
        logger.debug(f"Recorded external call: {call_type}")
    
    def set_outputs(self, outputs: Dict[str, Any]) -> None:
        """Set the final model outputs for the current trace."""
        if self.current_trace is not None:
            self.current_trace.outputs = outputs
    
    def add_benign_activations(self, activations: Dict[str, torch.Tensor]) -> None:
        """Add benign activations to the donor pool."""
        if not self.benign_donor_pool:
            return
            
        for layer_name, activation in activations.items():
            if activation.requires_grad:
                activation = activation.detach()
            if activation.is_cuda:
                activation = activation.cpu()
            
            activation_np = activation.numpy()
            self.benign_donor_pool.add_benign_activation(layer_name, activation_np)
    
    def export_bundle_async(self, trace_data: TraceData, bundle_id: str, exporter) -> None:
        """Export bundle asynchronously if async writer is enabled."""
        if self.async_writer:
            self.async_writer.queue_bundle_export(trace_data, bundle_id, exporter)
        else:
            # Fallback to synchronous export
            exporter.create_bundle(trace_data, bundle_id)
    
    def get_benign_donor_stats(self) -> Dict[str, int]:
        """Get statistics about the benign donor pool."""
        if self.benign_donor_pool:
            return self.benign_donor_pool.get_pool_stats()
        return {}
    
    def get_async_writer_stats(self) -> Dict[str, int]:
        """Get statistics about the async writer."""
        if self.async_writer:
            return self.async_writer.get_stats()
        return {}
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Extract model configuration information."""
        if self.model_ref is None or self.model_ref() is None:
            return {}
        
        model = self.model_ref()
        try:
            from core.utils import validate_model_architecture
            return validate_model_architecture(model)
        except Exception as e:
            logger.warning(f"Failed to extract model config: {e}")
            return {"error": str(e)}
    
    def _remove_existing_hooks(self) -> None:
        """Remove all existing hooks."""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()
        logger.debug("Removed all existing hooks")
    
    def cleanup(self) -> None:
        """Clean up hooks and resources."""
        self._remove_existing_hooks()
        self.activations.clear()
        self.external_calls.clear()
        self.current_trace = None
        self.is_recording = False
        
        # Shutdown async writer
        if self.async_writer:
            self.async_writer.shutdown()
            
        logger.info("Cleaned up RecorderHooks")
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during cleanup


class ContextRecorder:
    """Context manager for easy recording."""
    
    def __init__(self, hooks: RecorderHooks, inputs: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        self.hooks = hooks
        self.inputs = inputs
        self.metadata = metadata
        self.trace_data: Optional[TraceData] = None
    
    def __enter__(self) -> "ContextRecorder":
        self.hooks.start_recording(self.inputs, self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.trace_data = self.hooks.stop_recording()
    
    def get_trace(self) -> Optional[TraceData]:
        """Get the recorded trace data."""
        return self.trace_data