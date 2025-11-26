#!/usr/bin/env python3
"""
Simple Evaluation Script

This script loads the model, perturbation data, and generates embeddings for 100 examples.
It can be run with baseline or optimized settings.

Usage:
    python simple_evaluation.py --method baseline
    python simple_evaluation.py --method batching --batch_size 32
    python simple_evaluation.py --method mixed_precision --batch_size 32
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import anndata
import urllib.request
from helical.models.geneformer import Geneformer, GeneformerConfig
from embedding_evaluator import EmbeddingEvaluator
import warnings
warnings.filterwarnings("ignore")


def load_perturbation_data(num_samples: int = 100, data_file: str = "SrivatsanTrapnell2020_sciplex2.h5ad"):
    """
    Load and subset perturbation data.
    
    Args:
        num_samples: Number of samples to use
        data_file: Path to data file
        
    Returns:
        AnnData object with subset of data
    """
    # Download if needed
    if not os.path.exists(data_file):
        print("Downloading SciPlex2 data (this may take a minute)...")
        url = "https://zenodo.org/record/10044268/files/SrivatsanTrapnell2020_sciplex2.h5ad?download=1"
        urllib.request.urlretrieve(url, data_file)
        print("✅ Data downloaded!")
    
    # Load data
    print(f"Loading data from {data_file}...")
    adata = anndata.read_h5ad(data_file)
    print(f"Loaded full dataset: {adata.shape}")
    
    # Subset to requested number
    if num_samples < len(adata):
        perturbation_types = adata.obs['perturbation'].unique()
        samples_per_type = num_samples // len(perturbation_types)
        remaining = num_samples % len(perturbation_types)
        
        selected_indices = []
        for i, pert_type in enumerate(perturbation_types):
            type_indices = adata.obs[adata.obs['perturbation'] == pert_type].index
            n_samples = samples_per_type + (1 if i < remaining else 0)
            if len(type_indices) >= n_samples:
                selected = np.random.choice(type_indices, n_samples, replace=False)
            else:
                selected = type_indices
            selected_indices.extend(selected.tolist())
        
        adata = adata[selected_indices]
    
    print(f"Using subset: {adata.shape}")
    return adata


def get_embeddings_baseline(model, adata, batch_size: int = 10):
    """Get embeddings using baseline method."""
    print(f"Running baseline inference (batch_size={batch_size})...")
    dataset = model.process_data(adata)
    
    start_time = time.time()
    embeddings = model.get_embeddings(dataset)
    inference_time = time.time() - start_time
    
    return embeddings, inference_time


def get_embeddings_batching(model, adata, batch_size: int = 32):
    """Get embeddings using batching optimization."""
    print(f"Running batching optimization (batch_size={batch_size})...")
    dataset = model.process_data(adata)
    
    start_time = time.time()
    embeddings = model.get_embeddings(dataset)
    inference_time = time.time() - start_time
    
    return embeddings, inference_time


def get_embeddings_mixed_precision(model, adata, batch_size: int = 32):
    """Get embeddings using mixed precision (FP16)."""
    print(f"Running mixed precision inference (batch_size={batch_size})...")
    dataset = model.process_data(adata)
    
    start_time = time.time()
    with torch.cuda.amp.autocast():
        embeddings = model.get_embeddings(dataset)
    inference_time = time.time() - start_time
    
    return embeddings, inference_time


def main():
    parser = argparse.ArgumentParser(description='Simple evaluation script for ISP embeddings')
    parser.add_argument('--method', type=str, default='baseline',
                       choices=['baseline', 'batching', 'mixed_precision'],
                       help='Inference method to use')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Batch size (default: 10 for baseline, 32 for optimizations)')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of perturbation samples to use')
    parser.add_argument('--model_name', type=str, default='gf-6L-10M-i2048',
                       help='Geneformer model name')
    parser.add_argument('--output_dir', type=str, default='results/simple_evaluation',
                       help='Output directory for embeddings')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu), auto-detect if not specified')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("="*60)
    print("SIMPLE EVALUATION SCRIPT")
    print("="*60)
    print(f"Method: {args.method}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print("="*60)
    
    # Load model
    print("\n1. Loading model...")
    config = GeneformerConfig(
        model_name=args.model_name,
        device=device,
        batch_size=args.batch_size
    )
    model = Geneformer(config)
    print("✅ Model loaded")
    
    # Load data
    print("\n2. Loading perturbation data...")
    adata = load_perturbation_data(num_samples=args.num_samples)
    print("✅ Data loaded")
    
    # Get embeddings
    print("\n3. Generating embeddings...")
    if args.method == 'baseline':
        embeddings, inference_time = get_embeddings_baseline(model, adata, args.batch_size)
    elif args.method == 'batching':
        embeddings, inference_time = get_embeddings_batching(model, adata, args.batch_size)
    elif args.method == 'mixed_precision':
        embeddings, inference_time = get_embeddings_mixed_precision(model, adata, args.batch_size)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    print(f"✅ Embeddings generated!")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Time: {inference_time:.2f} seconds")
    print(f"   Throughput: {len(embeddings) / inference_time:.1f} samples/second")
    
    # Save embeddings
    print("\n4. Saving embeddings...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_file = os.path.join(args.output_dir, f'embeddings_{args.method}.npz')
    np.savez_compressed(output_file, embeddings=embeddings)
    
    # Save metadata
    metadata = {
        'method': args.method,
        'model_name': args.model_name,
        'device': device,
        'batch_size': args.batch_size,
        'num_samples': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'inference_time': inference_time,
        'throughput': len(embeddings) / inference_time
    }
    
    metadata_file = os.path.join(args.output_dir, f'metadata_{args.method}.json')
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Embeddings saved to {output_file}")
    print(f"✅ Metadata saved to {metadata_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Method: {args.method}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Inference time: {inference_time:.2f}s")
    print(f"Throughput: {len(embeddings) / inference_time:.1f} samples/s")
    print(f"Output directory: {args.output_dir}")
    print("="*60)
    
    print("\n💡 Tip: To evaluate consistency, compare embeddings from different methods")
    print("   using the embedding_evaluator module or 04_evaluate_embeddings.ipynb")


if __name__ == "__main__":
    main()




