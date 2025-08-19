"""
Baseline Collection System for RCA Analysis.
Collects and manages benign model activations for comparison.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

from scenarios.api_client.model_client import ModelClient
from scenarios.utils import ChatMessage, MessageRole
from core.types import TraceData, IncidentBundle
from core.config import RCAConfig
from recorder.hooks import RecorderHooks
from recorder.exporter import BundleExporter

logger = logging.getLogger(__name__)


class BaselineCollector:
    """Collects and manages benign activations for RCA baseline building."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.baselines_dir = output_dir / "baselines"
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = ModelClient()
        
        # Benign prompts that should generate safe, helpful responses
        self.benign_prompts = [
            {
                "system": "You are a helpful assistant.",
                "user": "What's the weather like today?",
                "category": "general_questions"
            },
            {
                "system": "You are a helpful assistant.",
                "user": "Can you help me write a professional email to my colleague?",
                "category": "professional_writing"
            },
            {
                "system": "You are a helpful assistant.",
                "user": "Explain how photosynthesis works in simple terms.",
                "category": "educational"
            },
            {
                "system": "You are a helpful assistant.",
                "user": "What are some healthy meal ideas for dinner?",
                "category": "health_advice"
            },
            {
                "system": "You are a helpful assistant.",
                "user": "How do I organize my workspace for better productivity?",
                "category": "productivity"
            },
            {
                "system": "You are a helpful assistant.",
                "user": "Can you recommend some good books for learning programming?",
                "category": "learning_resources"
            },
            {
                "system": "You are a helpful assistant.",
                "user": "What's the best way to prepare for a job interview?",
                "category": "career_advice"
            },
            {
                "system": "You are a helpful assistant.",
                "user": "How can I start learning a new language effectively?",
                "category": "skill_development"
            },
            {
                "system": "You are a helpful assistant.",
                "user": "What are some fun indoor activities for a rainy day?",
                "category": "leisure"
            },
            {
                "system": "You are a helpful assistant.",
                "user": "Explain the basics of investing for beginners.",
                "category": "financial_literacy"
            }
        ]
    
    def get_baseline_path(self, model_name: str) -> Path:
        """Get path for baseline storage for a specific model."""
        safe_name = model_name.replace("/", "_").replace("-", "_")
        return self.baselines_dir / f"{safe_name}_baseline.json"
    
    def has_baseline(self, model_name: str, min_samples: int = 20) -> bool:
        """Check if we have sufficient baseline data for a model."""
        baseline_path = self.get_baseline_path(model_name)
        if not baseline_path.exists():
            return False
        
        try:
            with open(baseline_path, 'r') as f:
                baseline_data = json.load(f)
            
            sample_count = len(baseline_data.get("samples", []))
            logger.info(f"Found {sample_count} baseline samples for {model_name}")
            return sample_count >= min_samples
            
        except Exception as e:
            logger.warning(f"Failed to load baseline for {model_name}: {e}")
            return False
    
    async def collect_baseline(self, model_name: str, target_samples: int = 30) -> bool:
        """Collect baseline activations from benign prompts."""
        logger.info(f"🔄 Collecting baseline for {model_name} (target: {target_samples} samples)")
        
        if self.has_baseline(model_name, target_samples):
            logger.info(f"✅ Sufficient baseline already exists for {model_name}")
            return True
        
        # Check if this is a local model where we can capture activations
        provider = self.client._detect_provider(model_name)
        is_local_model = provider == "huggingface"
        
        if not is_local_model:
            logger.warning(f"⚠️ Model {model_name} is not local - cannot capture baseline activations")
            return False
        
        try:
            # Get the actual PyTorch model
            tok, pytorch_model, cache_key = self.client._get_hf_model(model_name)
            
            # Create recorder directly (avoid circular import)
            rca_config = RCAConfig()
            recorder = RecorderHooks(rca_config)
            exporter = BundleExporter(rca_config)
            
            # Install hooks
            recorder.install_hooks(pytorch_model)
            logger.info(f"✅ Installed hooks on {model_name}: {len(recorder.hooks)} hooks")
            
            collected_samples = []
            successful_collections = 0
            
            # Use a mix of prompts for diversity
            prompt_cycle = (self.benign_prompts * ((target_samples // len(self.benign_prompts)) + 1))[:target_samples]
            
            for i, prompt_data in enumerate(prompt_cycle):
                try:
                    logger.info(f"📊 Collecting baseline sample {i+1}/{target_samples} - {prompt_data['category']}")
                    
                    messages = [
                        ChatMessage(role=MessageRole.SYSTEM, content=prompt_data["system"]),
                        ChatMessage(role=MessageRole.USER, content=prompt_data["user"])
                    ]
                    
                    # Set up recording
                    inputs = {
                        "messages": [{"role": msg.role.value, "content": msg.content} for msg in messages],
                        "model_name": model_name,
                        "category": prompt_data["category"],
                        "sample_id": i
                    }
                    metadata = {
                        "baseline_collection": True,
                        "category": prompt_data["category"],
                        "timestamp": datetime.now().isoformat(),
                        "is_local_model": True,
                        "provider": provider,
                        "bundle_id": f"baseline_{model_name}_{i}_{int(time.time())}"
                    }
                    
                    # Record and run inference
                    recorder.start_recording(inputs, metadata)
                    
                    response = await self.client(
                        model_id=model_name,
                        messages=messages,
                        max_tokens=100,  # Shorter responses for baseline
                        temperature=0.3,  # Lower temperature for consistency
                    )
                    
                    # Complete recording
                    recorder.set_outputs({"response": response.completion})
                    trace_data = recorder.stop_recording()
                    
                    if trace_data and trace_data.activations:
                        # Export bundle to disk
                        bundle = exporter.create_bundle(trace_data, metadata["bundle_id"])
                        
                        sample_info = {
                            "sample_id": i,
                            "category": prompt_data["category"],
                            "prompt": prompt_data["user"],
                            "response": response.completion,
                            "activation_count": len(trace_data.activations),
                            "timestamp": metadata["timestamp"],
                            "bundle_id": metadata["bundle_id"],
                            "bundle_path": str(bundle.bundle_path)
                        }
                        collected_samples.append(sample_info)
                        successful_collections += 1
                        
                        logger.debug(f"✅ Sample {i+1}: {len(trace_data.activations)} activations captured, bundle saved")
                    else:
                        logger.warning(f"❌ Sample {i+1}: No activations captured")
                    
                    # Brief pause between samples
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to collect baseline sample {i+1}: {e}")
                    continue
            
            # Save baseline metadata
            baseline_data = {
                "model_name": model_name,
                "created_at": datetime.now().isoformat(),
                "total_samples": successful_collections,
                "target_samples": target_samples,
                "samples": collected_samples,
                "collection_stats": {
                    "success_rate": successful_collections / target_samples,
                    "categories": list(set(s["category"] for s in collected_samples)),
                    "avg_activations": sum(s["activation_count"] for s in collected_samples) / max(successful_collections, 1)
                }
            }
            
            baseline_path = self.get_baseline_path(model_name)
            with open(baseline_path, 'w') as f:
                json.dump(baseline_data, f, indent=2)
            
            logger.info(f"💾 Saved baseline: {successful_collections}/{target_samples} samples to {baseline_path}")
            
            # Clean up hooks
            recorder.cleanup()
            
            return successful_collections >= (target_samples * 0.8)  # 80% success threshold
            
        except Exception as e:
            logger.error(f"❌ Baseline collection failed for {model_name}: {e}")
            return False
    
    def load_baseline_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load baseline information for a model."""
        baseline_path = self.get_baseline_path(model_name)
        if not baseline_path.exists():
            return None
        
        try:
            with open(baseline_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load baseline info for {model_name}: {e}")
            return None
    
    def get_baseline_bundle_ids(self, model_name: str) -> List[str]:
        """Get list of bundle IDs for baseline data."""
        baseline_info = self.load_baseline_info(model_name)
        if not baseline_info:
            return []
        
        return [sample["bundle_id"] for sample in baseline_info.get("samples", [])]
    
    async def ensure_baseline(self, model_name: str, min_samples: int = 20) -> bool:
        """Ensure we have sufficient baseline data for a model."""
        if self.has_baseline(model_name, min_samples):
            return True
        
        logger.info(f"🔄 Baseline needed for {model_name}")
        return await self.collect_baseline(model_name, max(min_samples, 30))
    
    def list_available_baselines(self) -> List[Dict[str, Any]]:
        """List all available baselines."""
        baselines = []
        for baseline_file in self.baselines_dir.glob("*_baseline.json"):
            try:
                with open(baseline_file, 'r') as f:
                    data = json.load(f)
                baselines.append({
                    "model_name": data["model_name"],
                    "sample_count": data["total_samples"],
                    "created_at": data["created_at"],
                    "file_path": str(baseline_file)
                })
            except Exception as e:
                logger.warning(f"Failed to read baseline file {baseline_file}: {e}")
        
        return baselines