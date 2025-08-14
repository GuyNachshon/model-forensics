"""Integration pipeline for CF → CCA workflow."""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import time
from datetime import datetime

from core.types import (
    IncidentBundle, CausalResult, FixSet, CompressionMetrics,
    ReplaySession
)
from core.config import RCAConfig
from replayer.core import Replayer
from modules.cf import CompressionForensics
from modules.cca import CausalCCA


logger = logging.getLogger(__name__)


class RCAPipeline:
    """Main pipeline orchestrating CF → CCA workflow for root cause analysis."""
    
    def __init__(self, config: Optional[RCAConfig] = None):
        self.config = config or RCAConfig()
        
        # Initialize modules based on config
        self.cf = CompressionForensics(self.config.modules.cf) if self.config.modules.cf.enabled else None
        self.cca = CausalCCA(self.config.modules.cca) if self.config.modules.cca.enabled else None
        self.replayer = Replayer(self.config)
        
        logger.info(f"Initialized RCA Pipeline - CF: {'✓' if self.cf else '✗'}, CCA: {'✓' if self.cca else '✗'}")
    
    def analyze_incident(self, incident_bundle: IncidentBundle, 
                        model, baseline_bundles: Optional[List[IncidentBundle]] = None) -> CausalResult:
        """Full pipeline analysis of an incident bundle."""
        logger.info(f"Starting RCA analysis for incident: {incident_bundle.bundle_id}")
        start_time = time.time()
        
        # Create replay session
        session = self.replayer.create_session(incident_bundle, model)
        
        # Phase 1: Compression Forensics (CF) - Anomaly Detection
        compression_metrics = []
        priority_layers = None
        
        if self.cf:
            logger.info("Phase 1: Running Compression Forensics")
            
            # Build baseline if provided
            if baseline_bundles:
                self.cf.build_baseline(baseline_bundles)
            
            # Analyze incident bundle
            compression_metrics = self.cf.analyze_bundle(incident_bundle)
            priority_layers = self.cf.prioritize_layers(compression_metrics)
            
            logger.info(f"CF identified {len(priority_layers)} priority layers from {len(compression_metrics)} total")
        
        # Phase 2: Causal Minimal Fix Sets (CCA) - Intervention Search
        fix_sets = []
        
        if self.cca:
            logger.info("Phase 2: Running Causal CCA")
            
            # Focus search on CF-flagged layers if available
            search_layers = priority_layers if priority_layers else None
            fix_sets = self.cca.find_minimal_fix_sets(session, search_layers)
            
            logger.info(f"CCA found {len(fix_sets)} minimal fix sets")
        
        # Select best fix set
        best_fix_set = self._select_best_fix_set(fix_sets) if fix_sets else None
        
        # Calculate execution time and confidence
        execution_time = time.time() - start_time
        confidence = self._compute_overall_confidence(compression_metrics, fix_sets)
        
        # Create result
        result = CausalResult(
            fix_set=best_fix_set,
            compression_metrics=compression_metrics,
            interaction_graph=None,  # Not implemented in MVP
            decision_basin_map=None,  # Not implemented in MVP
            provenance_info=None,     # Not implemented in MVP
            execution_time=execution_time,
            confidence=confidence
        )
        
        logger.info(f"RCA analysis completed in {execution_time:.2f}s with confidence {confidence:.3f}")
        return result
    
    def analyze_batch(self, incident_bundles: List[IncidentBundle], 
                     model, baseline_bundles: Optional[List[IncidentBundle]] = None) -> List[CausalResult]:
        """Analyze multiple incidents in batch."""
        logger.info(f"Starting batch RCA analysis for {len(incident_bundles)} incidents")
        
        results = []
        
        # Build shared baseline once if provided
        if baseline_bundles and self.cf:
            logger.info("Building shared baseline from benign cases")
            self.cf.build_baseline(baseline_bundles)
        
        for i, bundle in enumerate(incident_bundles):
            try:
                logger.info(f"Processing incident {i+1}/{len(incident_bundles)}: {bundle.bundle_id}")
                result = self.analyze_incident(bundle, model, baseline_bundles=None)  # Don't rebuild baseline
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to analyze incident {bundle.bundle_id}: {e}")
                # Create error result
                error_result = CausalResult(
                    fix_set=None,
                    compression_metrics=[],
                    interaction_graph=None,
                    decision_basin_map=None,
                    provenance_info=None,
                    execution_time=0.0,
                    confidence=0.0
                )
                results.append(error_result)
        
        logger.info(f"Batch analysis completed: {len(results)} results")
        return results
    
    def quick_triage(self, incident_bundles: List[IncidentBundle]) -> List[Tuple[str, float]]:
        """Quick triage of incidents by anomaly score for prioritization."""
        if not self.cf:
            logger.warning("CF module not enabled - cannot perform triage")
            return [(bundle.bundle_id, 0.0) for bundle in incident_bundles]
        
        logger.info(f"Triaging {len(incident_bundles)} incidents")
        
        triage_results = []
        
        for bundle in incident_bundles:
            try:
                metrics = self.cf.analyze_bundle(bundle)
                # Overall anomaly score = max anomaly score across layers
                max_anomaly = max((m.anomaly_score for m in metrics), default=0.0)
                triage_results.append((bundle.bundle_id, max_anomaly))
                
            except Exception as e:
                logger.warning(f"Triage failed for {bundle.bundle_id}: {e}")
                triage_results.append((bundle.bundle_id, 0.0))
        
        # Sort by anomaly score (descending)
        triage_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("Triage completed - incidents ranked by anomaly score")
        return triage_results
    
    def validate_fix_set(self, fix_set: FixSet, session: ReplaySession) -> Dict[str, float]:
        """Validate a fix set by applying interventions."""
        logger.info(f"Validating fix set with {len(fix_set.interventions)} interventions")
        
        try:
            # Apply all interventions in the fix set
            validation_results = self.replayer.apply_interventions(session, 
                [result.intervention for result in fix_set.interventions])
            
            # Compute validation metrics
            flip_rate = sum(1 for r in validation_results if r.flip_success) / len(validation_results)
            avg_confidence = sum(r.confidence for r in validation_results) / len(validation_results)
            avg_side_effects = sum(sum(r.side_effects.values()) for r in validation_results) / len(validation_results)
            
            return {
                "flip_rate": flip_rate,
                "confidence": avg_confidence,
                "side_effects": avg_side_effects,
                "interventions_applied": len(validation_results)
            }
            
        except Exception as e:
            logger.error(f"Fix set validation failed: {e}")
            return {"error": str(e)}
    
    def export_results(self, result: CausalResult, output_path: Optional[Path] = None) -> Path:
        """Export analysis results to disk."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.config.output_dir / f"rca_analysis_{timestamp}.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare serializable results
        export_data = {
            "analysis_timestamp": datetime.now().isoformat(),
            "execution_time": result.execution_time,
            "confidence": result.confidence,
            "compression_metrics": [
                {
                    "layer_name": m.layer_name,
                    "compression_ratio": m.compression_ratio,
                    "entropy": m.entropy,
                    "anomaly_score": m.anomaly_score,
                    "baseline_comparison": m.baseline_comparison,
                    "is_anomalous": bool(m.is_anomalous),
                    "confidence": m.confidence
                }
                for m in result.compression_metrics
            ],
            "fix_set": None
        }
        
        if result.fix_set:
            export_data["fix_set"] = {
                "num_interventions": len(result.fix_set.interventions),
                "sufficiency_score": result.fix_set.sufficiency_score,
                "necessity_scores": result.fix_set.necessity_scores,
                "minimality_rank": result.fix_set.minimality_rank,
                "total_flip_rate": result.fix_set.total_flip_rate,
                "avg_side_effects": result.fix_set.avg_side_effects,
                "interventions": [
                    {
                        "layer_name": interv.intervention.layer_name,
                        "type": interv.intervention.type.value,
                        "flip_success": bool(interv.flip_success),
                        "confidence": float(interv.confidence),
                        "side_effects": {k: float(v) for k, v in interv.side_effects.items()}
                    }
                    for interv in result.fix_set.interventions
                ]
            }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported RCA results to: {output_path}")
        return output_path
    
    def _select_best_fix_set(self, fix_sets: List[FixSet]) -> Optional[FixSet]:
        """Select the best fix set based on quality metrics."""
        if not fix_sets:
            return None
        
        # Score each fix set
        def score_fix_set(fix_set: FixSet) -> float:
            # Weighted combination of metrics
            flip_rate_score = fix_set.total_flip_rate * 0.4
            minimality_score = (1.0 / max(fix_set.minimality_rank, 1)) * 0.3
            side_effects_score = (1.0 - min(fix_set.avg_side_effects, 1.0)) * 0.2
            sufficiency_score = fix_set.sufficiency_score * 0.1
            
            return flip_rate_score + minimality_score + side_effects_score + sufficiency_score
        
        best_fix_set = max(fix_sets, key=score_fix_set)
        logger.debug(f"Selected best fix set with {len(best_fix_set.interventions)} interventions")
        
        return best_fix_set
    
    def _compute_overall_confidence(self, compression_metrics: List[CompressionMetrics],
                                  fix_sets: List[FixSet]) -> float:
        """Compute overall confidence in the analysis."""
        if not compression_metrics and not fix_sets:
            return 0.0
        
        cf_confidence = 0.0
        if compression_metrics:
            # Average confidence of anomalous layers
            anomalous_metrics = [m for m in compression_metrics if m.is_anomalous]
            if anomalous_metrics:
                cf_confidence = sum(m.confidence for m in anomalous_metrics) / len(anomalous_metrics)
        
        cca_confidence = 0.0
        if fix_sets:
            # Confidence of best fix set
            best_fix_set = self._select_best_fix_set(fix_sets)
            if best_fix_set:
                cca_confidence = best_fix_set.total_flip_rate
        
        # Combined confidence
        if compression_metrics and fix_sets:
            return (cf_confidence + cca_confidence) / 2.0
        elif compression_metrics:
            return cf_confidence
        elif fix_sets:
            return cca_confidence
        else:
            return 0.0
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics from all pipeline components."""
        stats = {
            "pipeline_config": {
                "cf_enabled": self.cf is not None,
                "cca_enabled": self.cca is not None,
                "timeout_minutes": self.config.timeout_minutes
            }
        }
        
        if self.cf:
            stats["cf_stats"] = self.cf.get_stats()
        
        if self.cca:
            stats["cca_stats"] = self.cca.get_stats()
        
        return stats