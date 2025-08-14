#!/usr/bin/env python3
"""Command-line interface for Model Forensics."""

import logging
import click
import json
from pathlib import Path
from typing import Optional
from transformers import GPT2LMHeadModel, BertForSequenceClassification, AutoModel, AutoTokenizer

from core.config import RCAConfig
from modules.pipeline import RCAPipeline
from replayer.core import Replayer


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', type=click.Path(exists=True), help='Path to configuration YAML file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Model Forensics - Root Cause Analysis for foundation model failures."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if config:
        ctx.obj = RCAConfig.from_yaml(Path(config))
    else:
        ctx.obj = RCAConfig()
    
    logger.info(f"Model Forensics CLI initialized with {ctx.obj.log_level} logging")


@cli.command()
@click.argument('incident_bundle', type=click.Path(exists=True))
@click.option('--model', '-m', default='gpt2', help='Model name or path (default: gpt2)')
@click.option('--baseline', '-b', type=click.Path(exists=True), multiple=True, 
              help='Path to benign baseline bundle(s)')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--triage-only', is_flag=True, help='Only perform quick triage analysis')
@click.pass_obj
def analyze(config: RCAConfig, incident_bundle, model, baseline, output, triage_only):
    """Analyze an incident bundle for root causes."""
    logger.info(f"Analyzing incident: {incident_bundle}")
    
    try:
        # Load model
        logger.info(f"Loading model: {model}")
        if model == 'gpt2':
            model_obj = GPT2LMHeadModel.from_pretrained('gpt2')
        elif model == 'bert-sentiment':
            model_obj = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        else:
            model_obj = AutoModel.from_pretrained(model)
        model_obj.eval()
        
        # Initialize pipeline
        pipeline = RCAPipeline(config)
        replayer = Replayer(config)
        
        # Load incident bundle
        incident = replayer.load_bundle(Path(incident_bundle))
        
        # Load baseline bundles
        baseline_bundles = []
        for baseline_path in baseline:
            try:
                baseline_bundle = replayer.load_bundle(Path(baseline_path))
                baseline_bundles.append(baseline_bundle)
                logger.info(f"Loaded baseline: {baseline_path}")
            except Exception as e:
                logger.warning(f"Failed to load baseline {baseline_path}: {e}")
        
        if triage_only:
            # Quick triage analysis
            all_bundles = [incident] + baseline_bundles
            triage_results = pipeline.quick_triage(all_bundles)
            
            click.echo("\n=== TRIAGE RESULTS ===")
            for bundle_id, score in triage_results:
                status = "🚨 HIGH" if score > 0.5 else "✅ NORMAL"
                click.echo(f"{bundle_id}: {score:.3f} ({status})")
            
        else:
            # Full RCA analysis
            result = pipeline.analyze_incident(incident, model_obj, baseline_bundles)
            
            # Display results
            click.echo(f"\n=== ANALYSIS COMPLETE ===")
            click.echo(f"Execution time: {result.execution_time:.2f}s")
            click.echo(f"Overall confidence: {result.confidence:.3f}")
            
            if result.compression_metrics:
                anomalous = [m for m in result.compression_metrics if m.is_anomalous]
                click.echo(f"\n🔍 CF Analysis: {len(anomalous)} anomalous layers detected")
                for metric in anomalous[:5]:  # Show top 5
                    click.echo(f"  • {metric.layer_name}: score={metric.anomaly_score:.3f}")
            
            if result.fix_set:
                click.echo(f"\n🔧 CCA Analysis: Minimal fix set found")
                click.echo(f"  • {len(result.fix_set.interventions)} interventions")
                click.echo(f"  • Flip rate: {result.fix_set.total_flip_rate:.3f}")
                click.echo(f"  • Avg side effects: {result.fix_set.avg_side_effects:.3f}")
                
                click.echo("\n  Interventions:")
                for interv in result.fix_set.interventions[:5]:  # Show top 5
                    success = "✅" if interv.flip_success else "❌"
                    click.echo(f"    {success} {interv.intervention.layer_name} ({interv.intervention.type.value})")
            
            # Export results
            if output:
                output_path = Path(output)
            else:
                output_path = Path(f"rca_analysis_{incident.bundle_id}.json")
            
            exported_path = pipeline.export_results(result, output_path)
            click.echo(f"\n📄 Results exported to: {exported_path}")
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.argument('bundle_dir', type=click.Path(exists=True))
@click.option('--model', '-m', default='gpt2', help='Model name or path')
@click.option('--output', '-o', type=click.Path(), help='Output directory for batch results')
@click.pass_obj
def batch(config: RCAConfig, bundle_dir, model, output):
    """Analyze multiple incident bundles in batch."""
    bundle_path = Path(bundle_dir)
    logger.info(f"Batch analysis of bundles in: {bundle_path}")
    
    try:
        # Load model
        logger.info(f"Loading model: {model}")
        if model == 'gpt2':
            model_obj = GPT2LMHeadModel.from_pretrained('gpt2')
        elif model == 'bert-sentiment':
            model_obj = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        else:
            model_obj = AutoModel.from_pretrained(model)
        model_obj.eval()
        
        # Initialize pipeline
        pipeline = RCAPipeline(config)
        replayer = Replayer(config)
        
        # Find all bundle directories
        bundle_dirs = [d for d in bundle_path.iterdir() 
                      if d.is_dir() and (d / "manifest.json").exists()]
        
        if not bundle_dirs:
            raise click.ClickException("No valid bundles found in directory")
        
        logger.info(f"Found {len(bundle_dirs)} bundles to analyze")
        
        # Load bundles
        bundles = []
        for bundle_dir in bundle_dirs:
            try:
                bundle = replayer.load_bundle(bundle_dir)
                bundles.append(bundle)
            except Exception as e:
                logger.warning(f"Failed to load bundle {bundle_dir.name}: {e}")
        
        # Separate benign and incident bundles (heuristic: bundles with "benign" in name)
        benign_bundles = [b for b in bundles if "benign" in b.bundle_id.lower()]
        incident_bundles = [b for b in bundles if "benign" not in b.bundle_id.lower()]
        
        logger.info(f"Classified {len(benign_bundles)} benign, {len(incident_bundles)} incident bundles")
        
        # Run batch analysis
        results = pipeline.analyze_batch(incident_bundles, model_obj, benign_bundles)
        
        # Display summary
        click.echo(f"\n=== BATCH ANALYSIS COMPLETE ===")
        click.echo(f"Analyzed: {len(results)} incidents")
        
        successful = sum(1 for r in results if r.confidence > 0.1)
        click.echo(f"Successful analyses: {successful}/{len(results)}")
        
        # Show top incidents by confidence
        sorted_results = sorted(zip(incident_bundles, results), 
                              key=lambda x: x[1].confidence, reverse=True)
        
        click.echo(f"\nTop incidents by confidence:")
        for bundle, result in sorted_results[:5]:
            click.echo(f"  • {bundle.bundle_id}: {result.confidence:.3f}")
        
        # Export batch results
        if output:
            output_dir = Path(output)
        else:
            output_dir = Path("batch_results")
        
        output_dir.mkdir(exist_ok=True)
        
        for bundle, result in zip(incident_bundles, results):
            result_file = output_dir / f"{bundle.bundle_id}_analysis.json"
            pipeline.export_results(result, result_file)
        
        click.echo(f"\n📁 Batch results exported to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.argument('bundle_dirs', type=click.Path(exists=True), nargs=-1)
@click.option('--sort-by', type=click.Choice(['anomaly', 'size', 'date']), 
              default='anomaly', help='Sort criteria')
@click.pass_obj
def triage(config: RCAConfig, bundle_dirs, sort_by):
    """Quick triage of multiple incident bundles."""
    if not bundle_dirs:
        raise click.ClickException("No bundle directories specified")
    
    logger.info(f"Triaging bundles from {len(bundle_dirs)} directories")
    
    try:
        # Initialize pipeline
        pipeline = RCAPipeline(config)
        replayer = Replayer(config)
        
        # Load all bundles
        all_bundles = []
        for bundle_dir_str in bundle_dirs:
            bundle_dir = Path(bundle_dir_str)
            if bundle_dir.is_file() and bundle_dir.name == "manifest.json":
                bundle_dir = bundle_dir.parent
            
            if (bundle_dir / "manifest.json").exists():
                try:
                    bundle = replayer.load_bundle(bundle_dir)
                    all_bundles.append(bundle)
                except Exception as e:
                    logger.warning(f"Failed to load bundle {bundle_dir.name}: {e}")
            else:
                # Directory with multiple bundles
                for sub_dir in bundle_dir.iterdir():
                    if sub_dir.is_dir() and (sub_dir / "manifest.json").exists():
                        try:
                            bundle = replayer.load_bundle(sub_dir)
                            all_bundles.append(bundle)
                        except Exception as e:
                            logger.warning(f"Failed to load bundle {sub_dir.name}: {e}")
        
        if not all_bundles:
            raise click.ClickException("No valid bundles found")
        
        logger.info(f"Loaded {len(all_bundles)} bundles for triage")
        
        # Run triage
        triage_results = pipeline.quick_triage(all_bundles)
        
        # Sort results
        if sort_by == 'anomaly':
            # Already sorted by anomaly score
            pass
        elif sort_by == 'size':
            triage_results = sorted(triage_results, 
                                  key=lambda x: next(b.size_mb for b in all_bundles if b.bundle_id == x[0]), 
                                  reverse=True)
        elif sort_by == 'date':
            triage_results = sorted(triage_results, 
                                  key=lambda x: x[0])  # Sort by bundle ID (which often contains dates)
        
        # Display results
        click.echo(f"\n=== TRIAGE RESULTS (sorted by {sort_by}) ===")
        click.echo(f"{'Bundle ID':<30} {'Anomaly Score':<15} {'Priority':<10}")
        click.echo("-" * 55)
        
        for bundle_id, score in triage_results:
            if score > 0.7:
                priority = "🚨 URGENT"
            elif score > 0.5:
                priority = "⚠️  HIGH"
            elif score > 0.3:
                priority = "📋 MEDIUM"
            else:
                priority = "✅ LOW"
            
            click.echo(f"{bundle_id:<30} {score:<15.3f} {priority}")
        
        # Summary
        urgent = sum(1 for _, score in triage_results if score > 0.7)
        high = sum(1 for _, score in triage_results if 0.5 < score <= 0.7)
        medium = sum(1 for _, score in triage_results if 0.3 < score <= 0.5)
        low = sum(1 for _, score in triage_results if score <= 0.3)
        
        click.echo(f"\n📊 Summary: {urgent} urgent, {high} high, {medium} medium, {low} low priority")
        
    except Exception as e:
        logger.error(f"Triage failed: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.option('--output', '-o', type=click.Path(), default='config.yaml',
              help='Output configuration file path')
@click.pass_obj
def init_config(config: RCAConfig, output):
    """Initialize a configuration file with default settings."""
    output_path = Path(output)
    
    if output_path.exists():
        if not click.confirm(f"Config file {output_path} exists. Overwrite?"):
            return
    
    try:
        config.to_yaml(output_path)
        click.echo(f"✅ Configuration initialized: {output_path}")
        click.echo("\nEdit the configuration file to customize:")
        click.echo("  • CF anomaly thresholds")
        click.echo("  • CCA search strategies") 
        click.echo("  • Recording granularity")
        click.echo("  • Output directories")
        
    except Exception as e:
        logger.error(f"Failed to create config: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.pass_obj
def info(config: RCAConfig):
    """Display system information and configuration."""
    click.echo("=== Model Forensics System Info ===")
    click.echo(f"Version: 0.1.0")
    click.echo(f"Config loaded: {type(config).__name__}")
    
    click.echo(f"\nModules:")
    click.echo(f"  CF (Compression Forensics): {'✅ Enabled' if config.modules.cf.enabled else '❌ Disabled'}")
    click.echo(f"  CCA (Causal Analysis): {'✅ Enabled' if config.modules.cca.enabled else '❌ Disabled'}")
    click.echo(f"  ISC: {'✅ Enabled' if config.modules.isc.enabled else '❌ Disabled'}")
    click.echo(f"  DBC: {'✅ Enabled' if config.modules.dbc.enabled else '❌ Disabled'}")
    click.echo(f"  Provenance: {'✅ Enabled' if config.modules.provenance.enabled else '❌ Disabled'}")
    
    click.echo(f"\nDirectories:")
    click.echo(f"  Output: {config.output_dir}")
    click.echo(f"  Bundles: {config.bundle_dir}")
    
    click.echo(f"\nSettings:")
    click.echo(f"  CF anomaly threshold: {config.modules.cf.anomaly_threshold}")
    click.echo(f"  CCA search strategy: {config.modules.cca.search_strategy}")
    click.echo(f"  Timeout: {config.timeout_minutes} minutes")
    click.echo(f"  Device: {config.device}")


if __name__ == '__main__':
    cli()