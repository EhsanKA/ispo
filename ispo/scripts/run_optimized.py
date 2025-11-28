#!/usr/bin/env python3
"""
Script to run optimized ISP inference with various optimization methods.
"""

from pathlib import Path

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

from ispo.core.optimized import OptimizedGeneformerISPOptimizer
import torch
import logging
import os
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
        'num_cells': config.get('dataset', {}).get('num_cells', 200),
        'model': config.get('model', {}).get('name', 'gf-6L-10M-i2048'),
        'device': config.get('model', {}).get('device', None),
        'num_gpus': config.get('model', {}).get('num_gpus', None),  # None = use all available
        'method': config.get('optimization', {}).get('method', None),
        'batch_size': config.get('optimization', {}).get('batch_size', config.get('inference', {}).get('batch_size', 32)),
        'precision': config.get('optimization', {}).get('precision', 'fp16'),
        'output_dir': config.get('output', {}).get('output_dir', 'results/optimized'),
        'wandb_project': config.get('wandb', {}).get('project', 'ispo-optimized'),
        'wandb_run_name': config.get('wandb', {}).get('run_name', None),
        'use_wandb': config.get('wandb', {}).get('enabled', False),
        'baseline_embeddings_path': config.get('baseline', {}).get('embeddings_path', None) if config.get('baseline', {}).get('enabled', False) else None,
    }
    
    # Override with command-line arguments if provided
    if args.model is not None:
        merged['model'] = args.model
    if args.num_perturbations is not None:
        merged['num_cells'] = args.num_perturbations
    if args.device is not None:
        merged['device'] = args.device
    if args.method is not None:
        merged['method'] = args.method
    if args.batch_size is not None:
        merged['batch_size'] = args.batch_size
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
    if args.baseline_embeddings_path is not None:
        merged['baseline_embeddings_path'] = args.baseline_embeddings_path
    if args.num_gpus is not None:
        merged['num_gpus'] = args.num_gpus
    
    return merged


def main():
    """Main function to run optimized ISP inference."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run optimized ISP inference with various optimization methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config file
  python -m ispo.scripts.run_optimized --config config/v1/batching/batching_bs8.yaml

  # Override config with command-line arguments
  python -m ispo.scripts.run_optimized --config config/v1/batching/batching_bs8.yaml --batch_size 16

  # Use command-line arguments only
  python -m ispo.scripts.run_optimized --method batching --batch_size 8 --use_wandb
        """
    )
    parser.add_argument('--config', type=str, default=None, help='Path to YAML configuration file')
    parser.add_argument('--model', type=str, default=None, help='Geneformer model name (overrides config)')
    parser.add_argument('--num_perturbations', type=int, default=None, help='Number of perturbations to process (overrides config)')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu), auto-detected if not specified (overrides config)')
    parser.add_argument('--method', type=str, default=None,
                       choices=['batching', 'mixed_precision', 'quantization', 'onnx', 'tensorrt'],
                       help='Optimization method to use (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for optimization (overrides config)')
    parser.add_argument('--precision', type=str, default=None, choices=['fp16', 'bf16'], help='Precision for mixed precision (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (overrides config)')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases tracking (overrides config)')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases tracking (overrides config)')
    parser.add_argument('--wandb_project', type=str, default=None, help='wandb project name (overrides config)')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name (overrides config, optional)')
    parser.add_argument('--baseline_embeddings_path', type=str, default=None, help='Path to baseline embeddings for comparison (overrides config)')
    parser.add_argument('--num_gpus', type=int, default=None, help='Number of GPUs to use for multi-GPU inference (None = use all available, overrides config)')
    
    args = parser.parse_args()

    # Load configuration
    config = {}
    config_path = args.config
    
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
    NUM_PERTURBATIONS = settings['num_cells']
    DEVICE = settings['device']
    if DEVICE == "auto" or DEVICE is None:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_GPUS = settings.get('num_gpus', None)
    METHOD = settings['method']
    BATCH_SIZE = settings['batch_size']
    PRECISION = settings.get('precision', 'fp16')
    OUTPUT_DIR = settings['output_dir']
    USE_WANDB = settings['use_wandb']
    WANDB_PROJECT = settings['wandb_project']
    WANDB_RUN_NAME = settings['wandb_run_name']
    BASELINE_EMBEDDINGS_PATH = settings.get('baseline_embeddings_path', None)
    
    # Validate required settings
    if METHOD is None:
        raise ValueError("Method must be specified either in config file or via --method argument")
    
    # Set random seed from config if available
    random_seed = config.get('dataset', {}).get('random_seed', 42)
    np.random.seed(random_seed)

    logger.info("=" * 80)
    logger.info("Optimized In-Silico Perturbation Challenge")
    logger.info("=" * 80)
    if config_path:
        logger.info(f"Config file: {config_path}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Device: {DEVICE}")
    if NUM_GPUS is not None:
        logger.info(f"Number of GPUs: {NUM_GPUS}")
    logger.info(f"Method: {METHOD}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Number of perturbations: {NUM_PERTURBATIONS}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Wandb tracking: {USE_WANDB}")
    if USE_WANDB:
        logger.info(f"Wandb project: {WANDB_PROJECT}")
        logger.info(f"Wandb run name: {WANDB_RUN_NAME or 'auto-generated'}")
    if BASELINE_EMBEDDINGS_PATH:
        logger.info(f"Baseline embeddings path: {BASELINE_EMBEDDINGS_PATH}")
    logger.info("=" * 80)

    # Initialize optimizer
    optimizer = OptimizedGeneformerISPOptimizer(
        model_name=MODEL_NAME, 
        device=DEVICE,
        use_wandb=USE_WANDB,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=WANDB_RUN_NAME,
        baseline_embeddings_path=BASELINE_EMBEDDINGS_PATH,
        num_gpus=NUM_GPUS
    )
    optimizer.load_model()

    # Get data path from config or use default
    DATA_PATH = settings.get('data_path', 'data/SrivatsanTrapnell2020_sciplex2.h5ad')
    
    perturbation_data = optimizer.load_perturbation_data(
        data_path=DATA_PATH,
        num_cells=NUM_PERTURBATIONS
    )

    results_list = []

    if METHOD == 'batching':
        logger.info("Running batching optimization...")
        results = optimizer.run_batching_optimized_inference(
            perturbation_data,
            batch_size=BATCH_SIZE,
            output_dir=OUTPUT_DIR
        )
        results_list.append(results)

    elif METHOD == 'mixed_precision':
        logger.info("Running mixed precision optimization...")
        results = optimizer.run_mixed_precision_inference(
            perturbation_data,
            batch_size=BATCH_SIZE,
            precision=PRECISION,
            output_dir=OUTPUT_DIR
        )
        results_list.append(results)

    elif METHOD == 'quantization':
        try:
            logger.info("Running quantization optimization...")
            results = optimizer.run_quantized_inference(
                perturbation_data,
                batch_size=BATCH_SIZE,
                output_dir=OUTPUT_DIR
            )
            results_list.append(results)
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")

    elif METHOD == 'onnx':
        try:
            logger.info("Running ONNX Runtime optimization...")
            # First export to ONNX
            onnx_path = "model.onnx"
            if not os.path.exists(onnx_path):
                optimizer.export_to_onnx(onnx_path)
            results = optimizer.run_onnx_inference(
                perturbation_data,
                onnx_path,
                batch_size=BATCH_SIZE,
                output_dir=OUTPUT_DIR
            )
            results_list.append(results)
        except Exception as e:
            logger.warning(f"ONNX Runtime failed: {e}")

    elif METHOD == 'tensorrt':
        try:
            logger.info("Running TensorRT optimization...")
            # First export to ONNX if not exists
            onnx_path = "model.onnx"
            if not os.path.exists(onnx_path):
                optimizer.export_to_onnx(onnx_path)
            # Convert ONNX to TensorRT
            tensorrt_path = optimizer.convert_to_tensorrt(
                onnx_path,
                "model.trt",
                precision=PRECISION
            )
            results = optimizer.run_tensorrt_inference(
                perturbation_data,
                tensorrt_path,
                batch_size=BATCH_SIZE,
                output_dir=OUTPUT_DIR
            )
            results_list.append(results)
        except Exception as e:
            logger.warning(f"TensorRT failed: {e}")

    # Save comprehensive comparison
    if results_list:
        optimizer.save_results_summary(results_list, "results/optimized_summary.csv")

    # Finish wandb run
    if USE_WANDB and optimizer.wandb_run is not None:
        optimizer.wandb_run.finish()
        logger.info("Wandb run completed")

    logger.info("Optimized run completed successfully!")


if __name__ == "__main__":
    main()



