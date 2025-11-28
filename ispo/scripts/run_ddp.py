#!/usr/bin/env python3
"""
Script to run ISPO inference with Distributed Data Parallel (DDP) support.

This script should be launched using torchrun for proper DDP initialization.

Example usage:
    # Run on 4 GPUs with DDP
    torchrun --nproc_per_node=4 -m ispo.scripts.run_ddp \
        --method batching \
        --batch_size 256 \
        --num_perturbations 1000 \
        --model_name gf-6L-10M-i2048

    # Run on 2 GPUs with mixed precision
    torchrun --nproc_per_node=2 -m ispo.scripts.run_ddp \
        --method mixed_precision \
        --batch_size 128 \
        --num_perturbations 1000
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ispo.core.optimized import OptimizedGeneformerISPOptimizer
from ispo.core.distributed import cleanup_ddp, is_main_process

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function for DDP inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ISPO inference with DDP')
    parser.add_argument('--method', type=str, required=True,
                       choices=['batching', 'mixed_precision', 'quantization'],
                       help='Optimization method to use')
    parser.add_argument('--model_name', type=str, default='gf-6L-10M-i2048',
                       help='Geneformer model name')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size per GPU')
    parser.add_argument('--num_perturbations', type=int, default=1000,
                       help='Number of perturbations to process')
    parser.add_argument('--precision', type=str, default='fp16',
                       choices=['fp16', 'bf16'],
                       help='Precision for mixed precision (if applicable)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (auto-generated if not specified)')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Enable Weights & Biases tracking')
    parser.add_argument('--wandb_project', type=str, default='ispo-ddp',
                       help='wandb project name')
    parser.add_argument('--data_path', type=str, default='data/SrivatsanTrapnell2020_sciplex2.h5ad',
                       help='Path to perturbation data')
    
    args = parser.parse_args()
    
    # Check if DDP is properly initialized
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        logger.error("DDP not initialized. Please use torchrun to launch this script.")
        logger.error("Example: torchrun --nproc_per_node=4 -m ispo.scripts.run_ddp --method batching")
        sys.exit(1)
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    
    if is_main_process():
        logger.info(f"DDP Inference: {world_size} GPUs")
        logger.info(f"Method: {args.method}")
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Batch size per GPU: {args.batch_size}")
        logger.info(f"Total batch size: {args.batch_size * world_size}")
    
    try:
        # Initialize optimizer with DDP enabled
        optimizer = OptimizedGeneformerISPOptimizer(
            model_name=args.model_name,
            device="cuda",
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            use_ddp=True  # Enable DDP
        )
        
        # Load model
        optimizer.load_model()
        
        # Load data
        if is_main_process():
            logger.info("Loading perturbation data...")
        data = optimizer.load_perturbation_data(
            data_path=args.data_path,
            num_cells=args.num_perturbations
        )
        
        # Set output directory
        if args.output_dir is None:
            output_dir = f"results/ddp_{args.method}_bs{args.batch_size}_n{world_size}gpu"
        else:
            output_dir = args.output_dir
        
        # Run inference based on method
        if args.method == 'batching':
            results = optimizer.run_batching_optimized_inference(
                data,
                batch_size=args.batch_size,
                output_dir=output_dir
            )
        elif args.method == 'mixed_precision':
            results = optimizer.run_mixed_precision_inference(
                data,
                batch_size=args.batch_size,
                precision=args.precision,
                output_dir=output_dir
            )
        elif args.method == 'quantization':
            results = optimizer.run_quantized_inference(
                data,
                batch_size=args.batch_size,
                output_dir=output_dir
            )
        
        if is_main_process():
            logger.info(f"Inference completed! Results saved to {output_dir}")
            logger.info(f"Total time: {results['performance_metrics']['total_time_seconds']:.2f}s")
            logger.info(f"Throughput: {args.num_perturbations / results['performance_metrics']['total_time_seconds']:.2f} samples/s")
    
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        raise
    finally:
        # Cleanup DDP
        cleanup_ddp()


if __name__ == "__main__":
    main()


