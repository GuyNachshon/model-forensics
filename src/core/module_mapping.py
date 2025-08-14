"""Module name mapping utilities for different model architectures."""

import logging
from typing import Dict, List, Optional, Any
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModuleNameMapper:
    """Maps bundle layer names to actual PyTorch module names."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.model_type = type(model).__name__
        self.module_dict = dict(model.named_modules())
        self._build_mappings()
    
    def _build_mappings(self) -> None:
        """Build bidirectional mappings between bundle names and module names."""
        self.bundle_to_module: Dict[str, str] = {}
        self.module_to_bundle: Dict[str, str] = {}
        
        if "GPT2" in self.model_type:
            self._build_gpt2_mappings()
        elif "BERT" in self.model_type or "Bert" in self.model_type:
            self._build_bert_mappings()
        else:
            logger.warning(f"Unknown model type: {self.model_type}, using generic mapping")
            self._build_generic_mappings()
    
    def _build_gpt2_mappings(self) -> None:
        """Build mappings for GPT-2 style models."""
        logger.debug("Building GPT-2 mappings")
        
        # Common patterns observed in bundles vs actual GPT-2 structure
        mappings = {
            # Embedding layers
            "embedding": "transformer.wte",
            
            # Layer normalization
            "ln_final": "transformer.ln_f",
            
            # Output layer
            "lm_head": "lm_head",
        }
        
        # Pattern-based mappings for transformer layers
        for name in self.module_dict.keys():
            if name.startswith("transformer.h."):
                # Extract layer number and component
                parts = name.split(".")
                if len(parts) >= 3:
                    layer_num = parts[2]  # transformer.h.{layer_num}...
                    
                    # Map different components
                    if name.endswith(".ln_1"):
                        bundle_name = f"layers.{layer_num}.norm1"
                        mappings[bundle_name] = name
                    elif name.endswith(".ln_2"):
                        bundle_name = f"layers.{layer_num}.norm2"
                        mappings[bundle_name] = name
                    elif name.endswith(".mlp.c_fc"):
                        bundle_name = f"layers.{layer_num}.linear1"
                        mappings[bundle_name] = name
                    elif name.endswith(".mlp.c_proj"):
                        bundle_name = f"layers.{layer_num}.linear2"
                        mappings[bundle_name] = name
                    elif name.endswith(".mlp.dropout"):
                        bundle_name = f"layers.{layer_num}.dropout"
                        mappings[bundle_name] = name
                    elif name.endswith(".attn.attn_dropout"):
                        bundle_name = f"layers.{layer_num}.dropout1"
                        mappings[bundle_name] = name
                    elif name.endswith(".attn.resid_dropout"):
                        bundle_name = f"layers.{layer_num}.dropout2"
                        mappings[bundle_name] = name
        
        self.bundle_to_module.update(mappings)
        self.module_to_bundle.update({v: k for k, v in mappings.items()})
        
        logger.info(f"Built {len(mappings)} GPT-2 module mappings")
    
    def _build_bert_mappings(self) -> None:
        """Build mappings for BERT style models."""
        logger.debug("Building BERT mappings")
        
        # Common BERT patterns
        mappings = {
            # Embeddings
            "embeddings.word_embeddings": "bert.embeddings.word_embeddings",
            "embeddings.position_embeddings": "bert.embeddings.position_embeddings", 
            "embeddings.token_type_embeddings": "bert.embeddings.token_type_embeddings",
            "embeddings.LayerNorm": "bert.embeddings.LayerNorm",
            "embeddings.dropout": "bert.embeddings.dropout",
            
            # Pooler
            "pooler.dense": "bert.pooler.dense",
            "pooler.activation": "bert.pooler.activation",
            
            # Classifier (for sequence classification)
            "classifier": "classifier",
            "dropout": "dropout",
        }
        
        # Pattern-based mappings for encoder layers
        for name in self.module_dict.keys():
            if "bert.encoder.layer" in name:
                # Extract layer number and component
                parts = name.split(".")
                if len(parts) >= 4 and parts[2] == "layer":
                    layer_num = parts[3]
                    
                    if name.endswith(".attention.self.query"):
                        bundle_name = f"encoder.layer.{layer_num}.attention.self.query"
                        mappings[bundle_name] = name
                    elif name.endswith(".attention.self.key"):
                        bundle_name = f"encoder.layer.{layer_num}.attention.self.key"
                        mappings[bundle_name] = name
                    elif name.endswith(".attention.self.value"):
                        bundle_name = f"encoder.layer.{layer_num}.attention.self.value"
                        mappings[bundle_name] = name
                    elif name.endswith(".attention.output.dense"):
                        bundle_name = f"encoder.layer.{layer_num}.attention.output.dense"
                        mappings[bundle_name] = name
                    elif name.endswith(".attention.output.LayerNorm"):
                        bundle_name = f"encoder.layer.{layer_num}.attention.output.LayerNorm"
                        mappings[bundle_name] = name
                    elif name.endswith(".attention.output.dropout"):
                        bundle_name = f"encoder.layer.{layer_num}.attention.output.dropout"
                        mappings[bundle_name] = name
                    elif name.endswith(".intermediate.dense"):
                        bundle_name = f"encoder.layer.{layer_num}.intermediate.dense"
                        mappings[bundle_name] = name
                    elif name.endswith(".output.dense"):
                        bundle_name = f"encoder.layer.{layer_num}.output.dense"
                        mappings[bundle_name] = name
                    elif name.endswith(".output.LayerNorm"):
                        bundle_name = f"encoder.layer.{layer_num}.output.LayerNorm"
                        mappings[bundle_name] = name
                    elif name.endswith(".output.dropout"):
                        bundle_name = f"encoder.layer.{layer_num}.output.dropout"
                        mappings[bundle_name] = name
        
        self.bundle_to_module.update(mappings)
        self.module_to_bundle.update({v: k for k, v in mappings.items()})
        
        logger.info(f"Built {len(mappings)} BERT module mappings")
    
    def _build_generic_mappings(self) -> None:
        """Build generic mappings using direct name matching."""
        logger.debug("Building generic mappings")
        
        # For unknown models, try exact name matching
        for module_name in self.module_dict.keys():
            # Direct mapping
            self.bundle_to_module[module_name] = module_name
            self.module_to_bundle[module_name] = module_name
    
    def bundle_to_module_name(self, bundle_name: str) -> Optional[str]:
        """Convert bundle layer name to PyTorch module name."""
        # Check if bundle name is already a correct PyTorch module name
        if bundle_name in self.module_dict:
            return bundle_name
        
        # Otherwise use mapping
        return self.bundle_to_module.get(bundle_name)
    
    def module_to_bundle_name(self, module_name: str) -> Optional[str]:
        """Convert PyTorch module name to bundle layer name."""
        return self.module_to_bundle.get(module_name)
    
    def get_module_by_bundle_name(self, bundle_name: str) -> Optional[nn.Module]:
        """Get PyTorch module by bundle name."""
        # Check if bundle name is already a correct PyTorch module name
        if bundle_name in self.module_dict:
            return self.module_dict[bundle_name]
            
        # Otherwise use mapping
        module_name = self.bundle_to_module_name(bundle_name)
        if module_name and module_name in self.module_dict:
            return self.module_dict[module_name]
        return None
    
    def validate_mappings(self, bundle_layer_names: List[str]) -> Dict[str, Any]:
        """Validate mappings against actual bundle layer names."""
        validation = {
            "total_layers": len(bundle_layer_names),
            "mapped_layers": 0,
            "unmapped_layers": [],
            "mapping_coverage": 0.0
        }
        
        for bundle_name in bundle_layer_names:
            if bundle_name in self.bundle_to_module:
                validation["mapped_layers"] += 1
            else:
                validation["unmapped_layers"].append(bundle_name)
        
        validation["mapping_coverage"] = validation["mapped_layers"] / max(validation["total_layers"], 1)
        
        logger.info(f"Mapping validation: {validation['mapped_layers']}/{validation['total_layers']} layers mapped "
                   f"({validation['mapping_coverage']:.1%} coverage)")
        
        return validation
    
    def get_mapping_stats(self) -> Dict[str, int]:
        """Get mapping statistics."""
        return {
            "bundle_to_module_mappings": len(self.bundle_to_module),
            "module_to_bundle_mappings": len(self.module_to_bundle),
            "total_model_modules": len(self.module_dict),
            "model_type": self.model_type
        }