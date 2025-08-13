"""Configuration management for model forensics."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class RecorderConfig:
    """Configuration for the recording SDK."""
    enabled: bool = True
    granularity: str = "layer"  # layer, attention_head, neuron
    compression_ratio: float = 0.1
    external_calls: bool = True
    max_bundle_size_mb: int = 500
    layers_to_record: Optional[List[str]] = None
    sample_rate: float = 1.0
    
    # Advanced compression options
    compression_method: str = "topk"  # topk, random_projection, hybrid
    random_projection_dim: int = 512  # Target dimension for random projections
    topk_preserve_ratio: float = 0.1  # For top-k sampling
    
    # Async writing
    async_export: bool = True
    export_queue_size: int = 100
    export_batch_size: int = 10
    
    # Benign donor system
    enable_benign_donors: bool = True
    benign_donor_pool_size: int = 50
    benign_donor_similarity_threshold: float = 0.8
    
    # Privacy options
    enable_dp_noise: bool = False
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_sensitivity: float = 1.0


@dataclass
class CFConfig:
    """Configuration for Compression Forensics module."""
    enabled: bool = True
    anomaly_threshold: float = 2.0
    baseline_samples: int = 1000
    compression_methods: List[str] = field(default_factory=lambda: ["zlib", "arithmetic"])
    entropy_window: int = 100


@dataclass
class CCAConfig:
    """Configuration for Causal Minimal Fix Set module."""
    enabled: bool = True
    max_fix_set_size: int = 50
    search_strategy: str = "greedy"  # greedy, beam, random
    side_effect_threshold: float = 0.05
    early_stop_threshold: float = 0.8
    intervention_types: List[str] = field(default_factory=lambda: ["zero", "mean", "patch"])


@dataclass
class ISCConfig:
    """Configuration for Interaction Spectroscopy module."""
    enabled: bool = False  # Disabled by default for MVP
    max_interaction_order: int = 3
    sampling_ratio: float = 0.1
    significance_threshold: float = 0.05


@dataclass
class DBCConfig:
    """Configuration for Decision Basin Cartography module."""
    enabled: bool = False  # Disabled by default for MVP
    perturbation_magnitude: float = 0.1
    layer_granularity: int = 1
    reversibility_threshold: float = 0.9


@dataclass
class ProvenanceConfig:
    """Configuration for Training-Time Provenance module."""
    enabled: bool = False  # Disabled by default for MVP
    influence_samples: int = 1000
    synthetic_probing: bool = True
    max_provenance_depth: int = 5


@dataclass
class ModuleConfig:
    """Configuration for all RCA modules."""
    cf: CFConfig = field(default_factory=CFConfig)
    cca: CCAConfig = field(default_factory=CCAConfig)
    isc: ISCConfig = field(default_factory=ISCConfig)
    dbc: DBCConfig = field(default_factory=DBCConfig)
    provenance: ProvenanceConfig = field(default_factory=ProvenanceConfig)


@dataclass
class RCAConfig:
    """Main configuration for the RCA system."""
    recorder: RecorderConfig = field(default_factory=RecorderConfig)
    modules: ModuleConfig = field(default_factory=ModuleConfig)
    timeout_minutes: int = 120
    output_dir: Path = field(default_factory=lambda: Path("./results"))
    bundle_dir: Path = field(default_factory=lambda: Path("./bundles"))
    log_level: str = "INFO"
    random_seed: int = 42
    device: str = "auto"  # auto, cpu, cuda
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "RCAConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RCAConfig":
        """Create configuration from dictionary."""
        # Extract nested configs
        recorder_config = RecorderConfig(**config_dict.get("recorder", {}))
        
        modules_dict = config_dict.get("modules", {})
        modules_config = ModuleConfig(
            cf=CFConfig(**modules_dict.get("cf", {})),
            cca=CCAConfig(**modules_dict.get("cca", {})),
            isc=ISCConfig(**modules_dict.get("isc", {})),
            dbc=DBCConfig(**modules_dict.get("dbc", {})),
            provenance=ProvenanceConfig(**modules_dict.get("provenance", {}))
        )
        
        # Create main config excluding nested ones
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ["recorder", "modules"]}
        
        return cls(
            recorder=recorder_config,
            modules=modules_config,
            **main_config
        )
    
    def to_yaml(self, config_path: Path) -> None:
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "recorder": {
                "enabled": self.recorder.enabled,
                "granularity": self.recorder.granularity,
                "compression_ratio": self.recorder.compression_ratio,
                "external_calls": self.recorder.external_calls,
                "max_bundle_size_mb": self.recorder.max_bundle_size_mb,
                "layers_to_record": self.recorder.layers_to_record,
                "sample_rate": self.recorder.sample_rate,
            },
            "modules": {
                "cf": {
                    "enabled": self.modules.cf.enabled,
                    "anomaly_threshold": self.modules.cf.anomaly_threshold,
                    "baseline_samples": self.modules.cf.baseline_samples,
                    "compression_methods": self.modules.cf.compression_methods,
                    "entropy_window": self.modules.cf.entropy_window,
                },
                "cca": {
                    "enabled": self.modules.cca.enabled,
                    "max_fix_set_size": self.modules.cca.max_fix_set_size,
                    "search_strategy": self.modules.cca.search_strategy,
                    "side_effect_threshold": self.modules.cca.side_effect_threshold,
                    "early_stop_threshold": self.modules.cca.early_stop_threshold,
                    "intervention_types": self.modules.cca.intervention_types,
                },
                "isc": {
                    "enabled": self.modules.isc.enabled,
                    "max_interaction_order": self.modules.isc.max_interaction_order,
                    "sampling_ratio": self.modules.isc.sampling_ratio,
                    "significance_threshold": self.modules.isc.significance_threshold,
                },
                "dbc": {
                    "enabled": self.modules.dbc.enabled,
                    "perturbation_magnitude": self.modules.dbc.perturbation_magnitude,
                    "layer_granularity": self.modules.dbc.layer_granularity,
                    "reversibility_threshold": self.modules.dbc.reversibility_threshold,
                },
                "provenance": {
                    "enabled": self.modules.provenance.enabled,
                    "influence_samples": self.modules.provenance.influence_samples,
                    "synthetic_probing": self.modules.provenance.synthetic_probing,
                    "max_provenance_depth": self.modules.provenance.max_provenance_depth,
                },
            },
            "timeout_minutes": self.timeout_minutes,
            "output_dir": str(self.output_dir),
            "bundle_dir": str(self.bundle_dir),
            "log_level": self.log_level,
            "random_seed": self.random_seed,
            "device": self.device,
        }