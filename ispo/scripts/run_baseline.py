#!/usr/bin/env python3
"""
Script to run baseline ISP inference.
"""

from pathlib import Path
import os

# Load environment variables from .env file for wandb API key
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try loading from current directory
        load_dotenv()
except ImportError:
    # python-dotenv not installed, continue without it
    pass

from ispo.core.baseline import GeneformerISPOptimizer
import torch
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    logger.warning("PyYAML not installed. Install with: pip install pyyaml")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not HAS_YAML:
        raise ImportError("PyYAML is required to load config files. Install with: pip install pyyaml")
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from: {config_path}")
    return config


def merge_config_with_args(config: dict, args) -> dict:
    """Merge configuration file with command-line arguments (args take precedence)."""
    # Start with config defaults
    merged = {
        'data_path': config.get('dataset', {}).get('data_path', 'data/SrivatsanTrapnell2020_sciplex2.h5ad'),
        'num_cells': config.get('dataset', {}).get('num_cells', 1000),
        'model': config.get('model', {}).get('name', 'gf-6L-10M-i2048'),
        'device': config.get('model', {}).get('device', None),
        'batch_size': config.get('inference', {}).get('batch_size', 1),
        'num_gpus': config.get('model', {}).get('num_gpus', None),  # None = use all available
        'output_dir': config.get('output', {}).get('output_dir', 'results/baseline'),
        'wandb_project': config.get('wandb', {}).get('project', 'ispo-baseline'),
        'wandb_run_name': config.get('wandb', {}).get('run_name', None),
        'use_wandb': config.get('wandb', {}).get('enabled', False),
    }
    
    # Override with command-line arguments if provided
    if args.model is not None:
        merged['model'] = args.model
    if args.num_perturbations is not None:
        merged['num_cells'] = args.num_perturbations
    if args.device is not None:
        merged['device'] = args.device
    if args.output_dir is not None:
        merged['output_dir'] = args.output_dir
    if args.wandb_project is not None:
        merged['wandb_project'] = args.wandb_project
    if args.wandb_run_name is not None:
        merged['wandb_run_name'] = args.wandb_run_name
    if args.use_wandb:
        merged['use_wandb'] = True
    if args.no_wandb:
        merged['use_wandb'] = False
    if args.num_gpus is not None:
        merged['num_gpus'] = args.num_gpus
    
    return merged


def main():
    """Main function to run baseline ISP inference."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run baseline ISP inference with optional wandb tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config
  python -m ispo.scripts.run_baseline --config config/baseline_gf_v1_6L.yaml

  # Override config with command-line arguments
  python -m ispo.scripts.run_baseline --config config/baseline_gf_v1_6L.yaml --model gf-12L-38M-i4096

  # Use command-line arguments only
  python -m ispo.scripts.run_baseline --model gf-6L-10M-i2048 --num_perturbations 1000 --use_wandb
        """
    )
    parser.add_argument('--config', type=str, default=None, help='Path to YAML configuration file')
    parser.add_argument('--model', type=str, default=None, help='Geneformer model name (overrides config)')
    parser.add_argument('--num_perturbations', type=int, default=None, help='Number of cells to process (overrides config)')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu), auto-detected if not specified (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (overrides config)')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases tracking (overrides config)')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases tracking (overrides config)')
    parser.add_argument('--wandb_project', type=str, default=None, help='wandb project name (overrides config)')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name (overrides config, optional)')
    parser.add_argument('--num_gpus', type=int, default=None, help='Number of GPUs to use for multi-GPU inference (None = use all available, overrides config)')
    
    args = parser.parse_args()

    # Load configuration
    config = {}
    config_path = args.config
    
    # If no config specified, try default location
    if config_path is None:
        default_config = Path(__file__).parent.parent.parent / 'config' / 'baseline_gf_v1_6L.yaml'
        if default_config.exists():
            config_path = str(default_config)
            logger.info(f"Using default config: {config_path}")
    
    # Load config if path provided
    if config_path is not None:
        try:
            config = load_config(config_path)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.info("Continuing with defaults and command-line arguments")
            config = {}
    
    # Merge config with command-line arguments
    settings = merge_config_with_args(config, args)

    # Extract settings
    MODEL_NAME = settings['model']
    NUM_CELLS = settings['num_cells']
    DEVICE = settings['device']
    if DEVICE == "auto" or DEVICE is None:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = settings['batch_size']
    NUM_GPUS = settings.get('num_gpus', None)
    USE_WANDB = settings['use_wandb']
    OUTPUT_DIR = settings['output_dir']
    WANDB_PROJECT = settings['wandb_project']
    WANDB_RUN_NAME = settings['wandb_run_name']
    
    # Set random seed from config if available
    random_seed = config.get('dataset', {}).get('random_seed', 42)
    np.random.seed(random_seed)

    logger.info("=" * 80)
    logger.info("In-Silico Perturbation Optimization Challenge - Baseline")
    logger.info("=" * 80)
    if config_path:
        logger.info(f"Config file: {config_path}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Number of cells: {NUM_CELLS}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    if NUM_GPUS is not None:
        logger.info(f"Number of GPUs: {NUM_GPUS}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Wandb tracking: {USE_WANDB}")
    if USE_WANDB:
        logger.info(f"Wandb project: {WANDB_PROJECT}")
        logger.info(f"Wandb run name: {WANDB_RUN_NAME or 'auto-generated'}")
    logger.info("=" * 80)

    # Initialize optimizer
    wandb_config = {
        'dataset': 'sciplex2',
        'num_cells': NUM_CELLS,
        'model_name': MODEL_NAME,
    }
    
    # Add config file path to wandb config if used
    if config_path:
        wandb_config['config_file'] = config_path

    optimizer = GeneformerISPOptimizer(
        model_name=MODEL_NAME,
        device=DEVICE,
        use_wandb=USE_WANDB,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=WANDB_RUN_NAME or f"baseline_{MODEL_NAME}_{NUM_CELLS}cells",
        wandb_config=wandb_config,
        num_gpus=NUM_GPUS
    )

    # Load model
    logger.info("Loading Geneformer model...")
    optimizer.load_model()

    # Load perturbation data
    data_path = config.get('dataset', {}).get('data_path', 'data/SrivatsanTrapnell2020_sciplex2.h5ad')
    logger.info("Loading sciplex2 data...")
    perturbation_data = optimizer.load_perturbation_data(
        data_path=data_path,
        num_cells=NUM_CELLS,
        dataset="sciplex2"
    )

    # Run baseline inference
    logger.info("Starting baseline inference...")
    results = optimizer.run_baseline_inference(
        perturbation_data,
        batch_size=BATCH_SIZE,
        output_dir=OUTPUT_DIR
    )

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "baseline_summary.csv")
    optimizer.save_results_summary([results], summary_path)

    # Log summary
    logger.info("=" * 80)
    logger.info("Baseline Evaluation Summary")
    logger.info("=" * 80)
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Number of cells processed: {len(perturbation_data)}")
    logger.info(f"Embeddings shape: {results['embeddings'].shape}")
    
    if results.get('performance_metrics'):
        perf = results['performance_metrics']
        logger.info(f"Total time: {perf.get('total_time_seconds', 'N/A'):.2f} seconds")
        logger.info(f"Throughput: {len(perturbation_data) / perf.get('total_time_seconds', 1):.2f} cells/second")
        if 'avg_gpu_memory_mb' in perf:
            logger.info(f"GPU memory: {perf['avg_gpu_memory_mb'] / 1024:.2f} GB (avg)")
    
    if results.get('evaluation_metrics'):
        eval_metrics = results['evaluation_metrics']
        logger.info(f"Zero-shot accuracy: {eval_metrics.get('zero_shot_accuracy', 'N/A'):.4f}")
        logger.info(f"Zero-shot F1 score: {eval_metrics.get('zero_shot_f1_score', 'N/A'):.4f}")
    
    if results.get('geometry_metrics'):
        geom = results['geometry_metrics']
        if geom.get('silhouette_score') is not None:
            logger.info(f"Silhouette score: {geom['silhouette_score']:.4f}")
        if geom.get('separation_ratio') is not None:
            logger.info(f"Separation ratio: {geom['separation_ratio']:.4f}")

    logger.info(f"\nResults saved to: {OUTPUT_DIR}")
    logger.info("=" * 80)

    # Finish wandb run
    if USE_WANDB and optimizer.wandb_run is not None:
        optimizer.wandb_run.finish()
        logger.info("Wandb run completed")

    logger.info("Baseline run completed successfully!")


if __name__ == "__main__":
    main()



