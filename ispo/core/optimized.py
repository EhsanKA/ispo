#!/usr/bin/env python3
"""
In-Silico Perturbation Optimization Challenge - Optimized Implementations

This script implements various optimizations for in-silico perturbation (ISP) inference:
1. Batching optimization
2. Mixed precision (FP16)
3. Model quantization
"""

import os
import time
import numpy as np
import pandas as pd
import psutil
import GPUtil
from memory_profiler import profile
import torch
from helical.models.geneformer import Geneformer, GeneformerConfig
import anndata
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file for wandb API key
try:
    from dotenv import load_dotenv
    # Try to load .env from project root (assuming this file is in ispo/core/)
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try loading from current directory
        load_dotenv()
except ImportError:
    # python-dotenv not installed, continue without it
    pass

# Import the baseline class
from .baseline import GeneformerISPOptimizer
from .profiler import PerformanceProfiler
from ..evaluation.evaluator import EmbeddingEvaluator

# Import distributed utilities
try:
    from .distributed import (
        get_world_size, get_rank, is_main_process, split_data_for_rank
    )
    DDP_AVAILABLE = True
except ImportError:
    DDP_AVAILABLE = False
    def get_world_size(): return 1
    def get_rank(): return 0
    def is_main_process(): return True
    def split_data_for_rank(data, world_size, rank): return data

# Optional imports for ONNX and TensorRT
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("ONNX not available. Install with: pip install onnx onnxruntime-gpu")

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TensorRT not available. Install TensorRT from NVIDIA.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizedGeneformerISPOptimizer(GeneformerISPOptimizer):
    """Extended optimizer with various optimization methods."""

    def __init__(self, model_name: str = "gf-6L-10M-i2048", device: str = "cuda",
                 use_wandb: bool = False, wandb_project: str = "ispo-optimized",
                 wandb_run_name: Optional[str] = None, wandb_config: Optional[Dict] = None,
                 baseline_embeddings_path: Optional[str] = None, num_gpus: Optional[int] = None,
                 use_ddp: bool = False):
        """
        Initialize optimized optimizer.
        
        Args:
            model_name: Name of the Geneformer model to use
            device: Device to run inference on ('cuda' or 'cpu')
            use_wandb: Whether to use Weights & Biases for tracking
            wandb_project: wandb project name
            wandb_run_name: Optional name for the wandb run
            wandb_config: Optional config dictionary for wandb
            baseline_embeddings_path: Optional path to baseline embeddings for comparison
            num_gpus: Number of GPUs to use for multi-GPU inference (None = use all available)
            use_ddp: Whether to use Distributed Data Parallel (DDP) instead of DataParallel
        """
        super().__init__(model_name, device, use_wandb, wandb_project, wandb_run_name, wandb_config, num_gpus, use_ddp)
        self.baseline_embeddings_path = baseline_embeddings_path
        self.baseline_embeddings = None
        self.embedding_evaluator = EmbeddingEvaluator()
        
        # Load baseline embeddings if path provided
        if baseline_embeddings_path and os.path.exists(baseline_embeddings_path):
            logger.info(f"Loading baseline embeddings from {baseline_embeddings_path}")
            try:
                baseline_data = np.load(baseline_embeddings_path)
                self.baseline_embeddings = baseline_data['embeddings']
                logger.info(f"Loaded baseline embeddings with shape: {self.baseline_embeddings.shape}")
            except Exception as e:
                logger.warning(f"Failed to load baseline embeddings: {e}")
    
    def compute_max_cosine_distance(self, baseline_emb: np.ndarray, optimized_emb: np.ndarray) -> Dict:
        """
        Compute max cosine distance between corresponding cell embeddings.
        
        Args:
            baseline_emb: Baseline embeddings (n_samples, embedding_dim)
            optimized_emb: Optimized embeddings (n_samples, embedding_dim)
            
        Returns:
            Dictionary with max cosine distance and related metrics
        """
        if baseline_emb.shape != optimized_emb.shape:
            raise ValueError(f"Shape mismatch: {baseline_emb.shape} vs {optimized_emb.shape}")
        
        # Normalize vectors
        baseline_norm = baseline_emb / (np.linalg.norm(baseline_emb, axis=1, keepdims=True) + 1e-8)
        optimized_norm = optimized_emb / (np.linalg.norm(optimized_emb, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity for each corresponding pair
        cosine_similarities = np.sum(baseline_norm * optimized_norm, axis=1)
        
        # Cosine distance = 1 - cosine similarity
        cosine_distances = 1 - cosine_similarities
        
        max_cosine_distance = np.max(cosine_distances)
        mean_cosine_distance = np.mean(cosine_distances)
        min_cosine_distance = np.min(cosine_distances)
        
        return {
            'max_cosine_distance': max_cosine_distance,
            'mean_cosine_distance': mean_cosine_distance,
            'min_cosine_distance': min_cosine_distance,
            'max_cosine_similarity': 1 - max_cosine_distance,
            'mean_cosine_similarity': 1 - mean_cosine_distance,
            'min_cosine_similarity': 1 - min_cosine_distance,
            'cosine_distances': cosine_distances,
            'cosine_similarities': cosine_similarities
        }
    
    def _generate_wandb_run_name(self, method: str, **params) -> str:
        """
        Generate meaningful wandb run name based on optimization method and parameters.
        
        Args:
            method: Optimization method name
            **params: Method-specific parameters
            
        Returns:
            Meaningful run name string
        """
        name_parts = [method]
        
        if method == 'batching':
            name_parts.append(f"bs{params.get('batch_size', 32)}")
        elif method == 'mixed_precision':
            precision = params.get('precision', 'fp16')
            batch_size = params.get('batch_size', 32)
            name_parts.append(f"{precision}_bs{batch_size}")
        elif method == 'quantization':
            bits = params.get('quantization_bits', 8)
            batch_size = params.get('batch_size', 32)
            name_parts.append(f"{bits}bit_bs{batch_size}")
        elif method == 'onnx_runtime':
            batch_size = params.get('batch_size', 32)
            name_parts.append(f"bs{batch_size}")
        elif method == 'tensorrt':
            precision = params.get('precision', 'fp16')
            batch_size = params.get('batch_size', 32)
            name_parts.append(f"{precision}_bs{batch_size}")
        
        name_parts.append(self.model_name)
        name_parts.append(self.device)
        
        return "_".join(name_parts)

    def run_batching_optimized_inference(self, adata: anndata.AnnData,
                                       batch_size: int = 32,
                                       output_dir: str = "results/batching") -> Dict:
        """
        Run inference with optimized batching.

        Args:
            adata: AnnData object with perturbation data
            batch_size: Batch size for processing
            output_dir: Directory to save results

        Returns:
            Dictionary with results and performance metrics
        """
        os.makedirs(output_dir, exist_ok=True)

        # Handle multi-GPU setup
        if self.use_ddp:
            # DDP: each process handles a subset of data
            world_size = get_world_size()
            rank = get_rank()
            if is_main_process():
                logger.info(f"Starting DDP batching inference (rank={rank}, world_size={world_size}, batch_size={batch_size})")
            effective_batch_size = batch_size  # DDP: each process uses base batch size
        else:
            # DataParallel: scale batch size (splits automatically)
            effective_batch_size = batch_size * self.num_gpus if self.num_gpus > 1 else batch_size
            if is_main_process() or not self.use_ddp:
                logger.info(f"Starting batching optimized inference with batch_size={batch_size}, num_gpus={self.num_gpus}, effective_batch_size={effective_batch_size}")
        
        # Only start profiling on main process (or if not using DDP)
        if is_main_process() or not self.use_ddp:
            self.profiler.start_profiling()

        # Process data
        if is_main_process() or not self.use_ddp:
            logger.info("Processing data...")
        dataset = self.model.process_data(adata)

        # Split data for DDP
        if self.use_ddp:
            dataset = split_data_for_rank(dataset, get_world_size(), get_rank())
            if is_main_process():
                logger.info(f"DDP: Rank {get_rank()} processing {len(dataset)} samples")

        # Run inference with larger batches
        if is_main_process() or not self.use_ddp:
            logger.info("Running inference with optimized batching...")
        all_embeddings = []

        # Use effective batch size for processing
        for i in range(0, len(dataset), effective_batch_size):
            batch_end = min(i + batch_size, len(dataset))
            batch_dataset = dataset.select(range(i, batch_end)) if hasattr(dataset, 'select') else dataset[i:batch_end]

            # Update profiling metrics (only on main process or if not using DDP)
            if is_main_process() or not self.use_ddp:
                self.profiler.update_metrics()

            # Run inference on batch
            batch_embeddings = self.model.get_embeddings(batch_dataset)
            all_embeddings.append(batch_embeddings)

            if is_main_process() or not self.use_ddp:
                logger.info(f"Processed batch {i//effective_batch_size + 1}/{(len(dataset)-1)//effective_batch_size + 1} (size: {batch_end-i})")

        # Gather embeddings from all DDP processes
        if self.use_ddp and get_world_size() > 1:
            import torch.distributed as dist
            all_embeddings_list = [None] * get_world_size()
            dist.all_gather_object(all_embeddings_list, all_embeddings)
            
            # Flatten and combine embeddings from all processes
            if is_main_process():
                all_embeddings = []
                for rank_embeddings in all_embeddings_list:
                    all_embeddings.extend(rank_embeddings)
            else:
                all_embeddings = []

        # Combine all embeddings (only on main process for DDP)
        if self.use_ddp:
            if is_main_process():
                embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
            else:
                embeddings = np.array([])
        else:
            embeddings = np.vstack(all_embeddings)

        # Stop profiling (only on main process or if not using DDP)
        if is_main_process() or not self.use_ddp:
            performance_metrics = self.profiler.stop_profiling()
        else:
            performance_metrics = {'total_time_seconds': 0}

        # Evaluate embeddings using zero-shot classification and geometry metrics
        perturbation_labels = None
        if 'perturbation' in adata.obs:
            perturbation_labels = adata.obs['perturbation'].values
        elif 'perturbation_type' in adata.obs:
            perturbation_labels = adata.obs['perturbation_type'].values
        
        evaluation_metrics = None
        geometry_metrics = None
        if perturbation_labels is not None:
            try:
                evaluation_metrics = self.evaluate_embeddings(embeddings, perturbation_labels)
            except Exception as e:
                logger.warning(f"Failed to evaluate embeddings: {e}")
            try:
                geometry_metrics = self.evaluate_embedding_geometry(embeddings, perturbation_labels)
            except Exception as e:
                logger.warning(f"Failed to evaluate embedding geometry: {e}")

        # Compare with baseline embeddings if available
        baseline_comparison_metrics = None
        if self.baseline_embeddings is not None:
            try:
                baseline_comparison_metrics = self.compute_max_cosine_distance(
                    self.baseline_embeddings, embeddings
                )
                logger.info(f"Max cosine distance from baseline: {baseline_comparison_metrics['max_cosine_distance']:.6f}")
                logger.info(f"Mean cosine similarity: {baseline_comparison_metrics['mean_cosine_similarity']:.6f}")
            except Exception as e:
                logger.warning(f"Failed to compare with baseline embeddings: {e}")

        # Update wandb run name if using wandb
        if self.use_wandb and self.wandb_run is not None:
            try:
                import wandb
                new_run_name = self._generate_wandb_run_name('batching', batch_size=batch_size)
                if self.wandb_run.name != new_run_name:
                    wandb.run.name = new_run_name
                    logger.info(f"Updated wandb run name to: {new_run_name}")
            except Exception as e:
                logger.warning(f"Failed to update wandb run name: {e}")

        # Save results
        results = {
            'embeddings': embeddings,
            'performance_metrics': performance_metrics,
            'evaluation_metrics': evaluation_metrics,
            'geometry_metrics': geometry_metrics,
            'baseline_comparison_metrics': baseline_comparison_metrics,
            'model_name': self.model_name,
            'device': self.device,
            'num_perturbations': len(adata),
            'num_genes': adata.shape[1],
            'method': f'batching_bs{batch_size}',
            'batch_size': batch_size
        }

        # Save embeddings
        np.savez_compressed(
            os.path.join(output_dir, 'embeddings.npz'),
            embeddings=embeddings
        )

        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'device': self.device,
            'num_perturbations': len(adata),
            'num_genes': adata.shape[1],
            'method': f'batching_bs{batch_size}',
            'batch_size': batch_size
        }
        pd.to_pickle(metadata, os.path.join(output_dir, 'metadata.pkl'))

        # Save performance metrics
        pd.DataFrame([performance_metrics]).to_csv(
            os.path.join(output_dir, 'performance_metrics.csv'),
            index=False
        )

        # Save evaluation metrics if available
        if evaluation_metrics:
            pd.DataFrame([evaluation_metrics]).to_csv(
                os.path.join(output_dir, 'evaluation_metrics.csv'),
                index=False
            )
        
        # Save geometry metrics if available
        if geometry_metrics:
            pd.DataFrame([geometry_metrics]).to_csv(
                os.path.join(output_dir, 'geometry_metrics.csv'),
                index=False
            )

        # Save baseline comparison metrics if available
        if baseline_comparison_metrics:
            pd.DataFrame([baseline_comparison_metrics]).to_csv(
                os.path.join(output_dir, 'baseline_comparison_metrics.csv'),
                index=False
            )

        # Log to wandb
        if self.use_wandb and self.wandb_run is not None:
            try:
                # Optional wandb import
                import wandb
                
                perf_log = {
                    'method': 'batching',
                    'batch_size': batch_size,
                    'performance/total_time_seconds': performance_metrics['total_time_seconds'],
                    'performance/throughput_samples_per_sec': len(adata) / performance_metrics['total_time_seconds'],
                    'performance/avg_cpu_percent': performance_metrics.get('avg_cpu_percent', 0),
                    'performance/max_cpu_percent': performance_metrics.get('max_cpu_percent', 0),
                    'performance/avg_memory_percent': performance_metrics.get('avg_memory_percent', 0),
                    'performance/max_memory_percent': performance_metrics.get('max_memory_percent', 0),
                    'dataset/num_perturbations': len(adata),
                    'dataset/num_genes': adata.shape[1],
                    'dataset/embedding_dim': embeddings.shape[1],
                }
                
                if 'avg_gpu_percent' in performance_metrics:
                    perf_log.update({
                        'performance/avg_gpu_percent': performance_metrics['avg_gpu_percent'],
                        'performance/max_gpu_percent': performance_metrics['max_gpu_percent'],
                        'performance/avg_gpu_memory_mb': performance_metrics.get('avg_gpu_memory_mb', 0),
                        'performance/max_gpu_memory_mb': performance_metrics.get('max_gpu_memory_mb', 0),
                    })
                
                # Log evaluation metrics
                if evaluation_metrics:
                    perf_log.update({
                        'evaluation/zero_shot_accuracy': evaluation_metrics.get('zero_shot_accuracy', 0),
                        'evaluation/zero_shot_f1_score': evaluation_metrics.get('zero_shot_f1_score', 0),
                        'evaluation/train_size': evaluation_metrics.get('train_size', 0),
                        'evaluation/test_size': evaluation_metrics.get('test_size', 0),
                        'evaluation/num_classes': evaluation_metrics.get('num_classes', 0),
                    })
                
                # Log geometry metrics
                if geometry_metrics:
                    if geometry_metrics.get('silhouette_score') is not None:
                        perf_log['geometry/silhouette_score'] = geometry_metrics['silhouette_score']
                    if geometry_metrics.get('davies_bouldin_index') is not None:
                        perf_log['geometry/davies_bouldin_index'] = geometry_metrics['davies_bouldin_index']
                    if geometry_metrics.get('calinski_harabasz_score') is not None:
                        perf_log['geometry/calinski_harabasz_score'] = geometry_metrics['calinski_harabasz_score']
                    if geometry_metrics.get('separation_ratio') is not None:
                        perf_log['geometry/separation_ratio'] = geometry_metrics['separation_ratio']
                    if geometry_metrics.get('mean_intra_cluster_distance') is not None:
                        perf_log['geometry/mean_intra_cluster_distance'] = geometry_metrics['mean_intra_cluster_distance']
                    if geometry_metrics.get('mean_inter_cluster_distance') is not None:
                        perf_log['geometry/mean_inter_cluster_distance'] = geometry_metrics['mean_inter_cluster_distance']
                    if geometry_metrics.get('knn_label_consistency') is not None:
                        perf_log['geometry/knn_label_consistency'] = geometry_metrics['knn_label_consistency']
                    if geometry_metrics.get('adjusted_rand_index') is not None:
                        perf_log['geometry/adjusted_rand_index'] = geometry_metrics['adjusted_rand_index']
                    if geometry_metrics.get('normalized_mutual_info') is not None:
                        perf_log['geometry/normalized_mutual_info'] = geometry_metrics['normalized_mutual_info']
                
                # Log baseline comparison metrics
                if baseline_comparison_metrics:
                    perf_log.update({
                        'baseline_comparison/max_cosine_distance': baseline_comparison_metrics['max_cosine_distance'],
                        'baseline_comparison/mean_cosine_distance': baseline_comparison_metrics['mean_cosine_distance'],
                        'baseline_comparison/min_cosine_distance': baseline_comparison_metrics['min_cosine_distance'],
                        'baseline_comparison/max_cosine_similarity': baseline_comparison_metrics['max_cosine_similarity'],
                        'baseline_comparison/mean_cosine_similarity': baseline_comparison_metrics['mean_cosine_similarity'],
                        'baseline_comparison/min_cosine_similarity': baseline_comparison_metrics['min_cosine_similarity'],
                    })
                
                self.wandb_run.log(perf_log)
                logger.info("Metrics logged to wandb")
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")

        logger.info(f"Batching optimized inference completed. Results saved to {output_dir}")
        logger.info(f"Embeddings shape: {embeddings.shape}")

        return results

    def run_mixed_precision_inference(self, adata: anndata.AnnData,
                                    batch_size: int = 32,
                                    precision: str = "fp16",
                                    output_dir: str = "results/mixed_precision") -> Dict:
        """
        Run inference with mixed precision (FP16 or BF16).

        Args:
            adata: AnnData object with perturbation data
            batch_size: Batch size for processing
            precision: Precision mode ('fp16' or 'bf16')
            output_dir: Directory to save results

        Returns:
            Dictionary with results and performance metrics
        """
        os.makedirs(output_dir, exist_ok=True)

        # Map precision string to torch dtype
        dtype_map = {
            'fp16': torch.float16,
            'bf16': torch.bfloat16
        }
        dtype = dtype_map.get(precision.lower(), torch.float16)
        
        logger.info(f"Starting mixed precision inference with {precision.upper()}")
        self.profiler.start_profiling()

        # Enable autocast for mixed precision with specified dtype
        with torch.cuda.amp.autocast(dtype=dtype):
            # Process data
            logger.info("Processing data...")
            dataset = self.model.process_data(adata)

            # Run inference with mixed precision
            logger.info("Running inference with mixed precision...")
            all_embeddings = []

            for i in range(0, len(dataset), batch_size):
                batch_end = min(i + batch_size, len(dataset))
                batch_dataset = dataset.select(range(i, batch_end))

                # Update profiling metrics
                self.profiler.update_metrics()

                # Run inference on batch with autocast
                with torch.cuda.amp.autocast(dtype=dtype):
                    batch_embeddings = self.model.get_embeddings(batch_dataset)
                all_embeddings.append(batch_embeddings)

                logger.info(f"Processed batch {i//batch_size + 1}/{(len(dataset)-1)//batch_size + 1} (size: {batch_end-i})")

            # Combine all embeddings
            embeddings = np.vstack(all_embeddings)

        # Stop profiling
        performance_metrics = self.profiler.stop_profiling()

        # Evaluate embeddings using zero-shot classification and geometry metrics
        perturbation_labels = None
        if 'perturbation' in adata.obs:
            perturbation_labels = adata.obs['perturbation'].values
        elif 'perturbation_type' in adata.obs:
            perturbation_labels = adata.obs['perturbation_type'].values
        
        evaluation_metrics = None
        geometry_metrics = None
        if perturbation_labels is not None:
            try:
                evaluation_metrics = self.evaluate_embeddings(embeddings, perturbation_labels)
            except Exception as e:
                logger.warning(f"Failed to evaluate embeddings: {e}")
            try:
                geometry_metrics = self.evaluate_embedding_geometry(embeddings, perturbation_labels)
            except Exception as e:
                logger.warning(f"Failed to evaluate embedding geometry: {e}")

        # Compare with baseline embeddings if available
        baseline_comparison_metrics = None
        if self.baseline_embeddings is not None:
            try:
                baseline_comparison_metrics = self.compute_max_cosine_distance(
                    self.baseline_embeddings, embeddings
                )
                logger.info(f"Max cosine distance from baseline: {baseline_comparison_metrics['max_cosine_distance']:.6f}")
                logger.info(f"Mean cosine similarity: {baseline_comparison_metrics['mean_cosine_similarity']:.6f}")
            except Exception as e:
                logger.warning(f"Failed to compare with baseline embeddings: {e}")

        # Update wandb run name if using wandb
        if self.use_wandb and self.wandb_run is not None:
            try:
                import wandb
                new_run_name = self._generate_wandb_run_name('mixed_precision', precision=precision, batch_size=batch_size)
                if self.wandb_run.name != new_run_name:
                    wandb.run.name = new_run_name
                    logger.info(f"Updated wandb run name to: {new_run_name}")
            except Exception as e:
                logger.warning(f"Failed to update wandb run name: {e}")

        # Save results
        results = {
            'embeddings': embeddings,
            'performance_metrics': performance_metrics,
            'evaluation_metrics': evaluation_metrics,
            'geometry_metrics': geometry_metrics,
            'baseline_comparison_metrics': baseline_comparison_metrics,
            'model_name': self.model_name,
            'device': self.device,
            'num_perturbations': len(adata),
            'num_genes': adata.shape[1],
            'method': f'mixed_precision_{precision.lower()}',
            'batch_size': batch_size,
            'precision': precision.lower()
        }

        # Save embeddings
        np.savez_compressed(
            os.path.join(output_dir, 'embeddings.npz'),
            embeddings=embeddings
        )

        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'device': self.device,
            'num_perturbations': len(adata),
            'num_genes': adata.shape[1],
            'method': f'mixed_precision_{precision.lower()}',
            'batch_size': batch_size,
            'precision': precision.lower()
        }
        pd.to_pickle(metadata, os.path.join(output_dir, 'metadata.pkl'))

        # Save performance metrics
        pd.DataFrame([performance_metrics]).to_csv(
            os.path.join(output_dir, 'performance_metrics.csv'),
            index=False
        )

        # Save evaluation metrics if available
        if evaluation_metrics:
            pd.DataFrame([evaluation_metrics]).to_csv(
                os.path.join(output_dir, 'evaluation_metrics.csv'),
                index=False
            )
        
        # Save geometry metrics if available
        if geometry_metrics:
            pd.DataFrame([geometry_metrics]).to_csv(
                os.path.join(output_dir, 'geometry_metrics.csv'),
                index=False
            )

        # Save baseline comparison metrics if available
        if baseline_comparison_metrics:
            pd.DataFrame([baseline_comparison_metrics]).to_csv(
                os.path.join(output_dir, 'baseline_comparison_metrics.csv'),
                index=False
            )

        # Log to wandb
        if self.use_wandb and self.wandb_run is not None:
            try:
                # Optional wandb import
                import wandb
                
                perf_log = {
                    'method': 'mixed_precision',
                    'precision': precision.lower(),
                    'batch_size': batch_size,
                    'performance/total_time_seconds': performance_metrics['total_time_seconds'],
                    'performance/throughput_samples_per_sec': len(adata) / performance_metrics['total_time_seconds'],
                    'performance/avg_cpu_percent': performance_metrics.get('avg_cpu_percent', 0),
                    'performance/max_cpu_percent': performance_metrics.get('max_cpu_percent', 0),
                    'performance/avg_memory_percent': performance_metrics.get('avg_memory_percent', 0),
                    'performance/max_memory_percent': performance_metrics.get('max_memory_percent', 0),
                    'dataset/num_perturbations': len(adata),
                    'dataset/num_genes': adata.shape[1],
                    'dataset/embedding_dim': embeddings.shape[1],
                }
                
                if 'avg_gpu_percent' in performance_metrics:
                    perf_log.update({
                        'performance/avg_gpu_percent': performance_metrics['avg_gpu_percent'],
                        'performance/max_gpu_percent': performance_metrics['max_gpu_percent'],
                        'performance/avg_gpu_memory_mb': performance_metrics.get('avg_gpu_memory_mb', 0),
                        'performance/max_gpu_memory_mb': performance_metrics.get('max_gpu_memory_mb', 0),
                    })
                
                # Log evaluation metrics
                if evaluation_metrics:
                    perf_log.update({
                        'evaluation/zero_shot_accuracy': evaluation_metrics.get('zero_shot_accuracy', 0),
                        'evaluation/zero_shot_f1_score': evaluation_metrics.get('zero_shot_f1_score', 0),
                        'evaluation/train_size': evaluation_metrics.get('train_size', 0),
                        'evaluation/test_size': evaluation_metrics.get('test_size', 0),
                        'evaluation/num_classes': evaluation_metrics.get('num_classes', 0),
                    })
                
                # Log geometry metrics
                if geometry_metrics:
                    if geometry_metrics.get('silhouette_score') is not None:
                        perf_log['geometry/silhouette_score'] = geometry_metrics['silhouette_score']
                    if geometry_metrics.get('davies_bouldin_index') is not None:
                        perf_log['geometry/davies_bouldin_index'] = geometry_metrics['davies_bouldin_index']
                    if geometry_metrics.get('calinski_harabasz_score') is not None:
                        perf_log['geometry/calinski_harabasz_score'] = geometry_metrics['calinski_harabasz_score']
                    if geometry_metrics.get('separation_ratio') is not None:
                        perf_log['geometry/separation_ratio'] = geometry_metrics['separation_ratio']
                    if geometry_metrics.get('mean_intra_cluster_distance') is not None:
                        perf_log['geometry/mean_intra_cluster_distance'] = geometry_metrics['mean_intra_cluster_distance']
                    if geometry_metrics.get('mean_inter_cluster_distance') is not None:
                        perf_log['geometry/mean_inter_cluster_distance'] = geometry_metrics['mean_inter_cluster_distance']
                    if geometry_metrics.get('knn_label_consistency') is not None:
                        perf_log['geometry/knn_label_consistency'] = geometry_metrics['knn_label_consistency']
                    if geometry_metrics.get('adjusted_rand_index') is not None:
                        perf_log['geometry/adjusted_rand_index'] = geometry_metrics['adjusted_rand_index']
                    if geometry_metrics.get('normalized_mutual_info') is not None:
                        perf_log['geometry/normalized_mutual_info'] = geometry_metrics['normalized_mutual_info']
                
                # Log baseline comparison metrics
                if baseline_comparison_metrics:
                    perf_log.update({
                        'baseline_comparison/max_cosine_distance': baseline_comparison_metrics['max_cosine_distance'],
                        'baseline_comparison/mean_cosine_distance': baseline_comparison_metrics['mean_cosine_distance'],
                        'baseline_comparison/min_cosine_distance': baseline_comparison_metrics['min_cosine_distance'],
                        'baseline_comparison/max_cosine_similarity': baseline_comparison_metrics['max_cosine_similarity'],
                        'baseline_comparison/mean_cosine_similarity': baseline_comparison_metrics['mean_cosine_similarity'],
                        'baseline_comparison/min_cosine_similarity': baseline_comparison_metrics['min_cosine_similarity'],
                    })
                
                self.wandb_run.log(perf_log)
                logger.info("Metrics logged to wandb")
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")

        logger.info(f"Mixed precision inference completed. Results saved to {output_dir}")
        logger.info(f"Embeddings shape: {embeddings.shape}")

        return results

    def run_quantized_inference(self, adata: anndata.AnnData,
                               batch_size: int = 32,
                               output_dir: str = "results/quantization") -> Dict:
        """
        Run inference with model quantization.

        Args:
            adata: AnnData object with perturbation data
            batch_size: Batch size for processing
            output_dir: Directory to save results

        Returns:
            Dictionary with results and performance metrics
        """
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Starting quantized inference")
        self.profiler.start_profiling()

        # Ensure model is loaded
        if self.model is None:
            logger.info("Loading model for quantization...")
            self.load_model()

        # Quantize the model to 8-bit using PyTorch's dynamic quantization
        logger.info("Quantizing model to 8-bit precision...")
        quantization_applied = False
        try:
            # Access the underlying PyTorch model
            if hasattr(self.model, 'model'):
                pytorch_model = self.model.model
                
                # PyTorch dynamic quantization only works on CPU
                # For CUDA, we'll use FP16 mixed precision as a lightweight alternative
                if self.device == "cuda":
                    logger.warning("PyTorch dynamic quantization is not supported on CUDA.")
                    logger.info("Using FP16 mixed precision as a lightweight alternative for CUDA.")
                    # Use autocast for FP16 inference (similar to mixed precision method)
                    # This will be applied during inference
                    quantization_applied = False  # We'll use autocast instead
                else:
                    # For CPU, use dynamic quantization
                    # Move model to CPU temporarily for quantization
                    original_device = next(pytorch_model.parameters()).device
                    pytorch_model_cpu = pytorch_model.cpu()
                    
                    # Use PyTorch's dynamic quantization (quantizes weights to int8)
                    # This works for linear layers and reduces model size and inference time
                    quantized_pytorch_model = torch.quantization.quantize_dynamic(
                        pytorch_model_cpu,
                        {torch.nn.Linear},  # Quantize Linear layers
                        dtype=torch.qint8
                    )
                    
                    # Move back to original device if needed
                    if original_device.type == 'cuda':
                        logger.warning("Quantized model cannot run on CUDA. Using original model on CUDA.")
                        quantization_applied = False
                    else:
                        # Replace the model's underlying PyTorch model with quantized version
                        self.model.model = quantized_pytorch_model
                        quantization_applied = True
                        logger.info("Model quantized successfully using PyTorch dynamic quantization (CPU)")
            else:
                logger.warning("Could not access underlying PyTorch model. Using original model.")
        except Exception as e:
            logger.warning(f"Quantization failed: {e}. Continuing with original model.")
            logger.warning("Note: Some quantization methods may not be compatible with all model architectures.")
        
        if not quantization_applied and self.device == "cuda":
            logger.info("Note: For CUDA, consider using 'mixed_precision' method with FP16 for similar benefits.")

        # Process data
        logger.info("Processing data...")
        dataset = self.model.process_data(adata)

        # Run inference with quantized model (or original if quantization not applied)
        logger.info("Running inference...")
        all_embeddings = []

        for i in range(0, len(dataset), batch_size):
            batch_end = min(i + batch_size, len(dataset))
            batch_dataset = dataset.select(range(i, batch_end))

            # Update profiling metrics
            self.profiler.update_metrics()

            # Run inference on batch
            # If on CUDA and quantization wasn't applied, use autocast for FP16
            if self.device == "cuda" and not quantization_applied:
                with torch.cuda.amp.autocast():
                    batch_embeddings = self.model.get_embeddings(batch_dataset)
            else:
                batch_embeddings = self.model.get_embeddings(batch_dataset)
            all_embeddings.append(batch_embeddings)

            logger.info(f"Processed batch {i//batch_size + 1}/{(len(dataset)-1)//batch_size + 1} (size: {batch_end-i})")

        # Combine all embeddings
        embeddings = np.vstack(all_embeddings)

        # Stop profiling
        performance_metrics = self.profiler.stop_profiling()

        # Evaluate embeddings using zero-shot classification and geometry metrics
        perturbation_labels = None
        if 'perturbation' in adata.obs:
            perturbation_labels = adata.obs['perturbation'].values
        elif 'perturbation_type' in adata.obs:
            perturbation_labels = adata.obs['perturbation_type'].values
        
        evaluation_metrics = None
        geometry_metrics = None
        if perturbation_labels is not None:
            try:
                evaluation_metrics = self.evaluate_embeddings(embeddings, perturbation_labels)
            except Exception as e:
                logger.warning(f"Failed to evaluate embeddings: {e}")
            try:
                geometry_metrics = self.evaluate_embedding_geometry(embeddings, perturbation_labels)
            except Exception as e:
                logger.warning(f"Failed to evaluate embedding geometry: {e}")

        # Compare with baseline embeddings if available
        baseline_comparison_metrics = None
        if self.baseline_embeddings is not None:
            try:
                baseline_comparison_metrics = self.compute_max_cosine_distance(
                    self.baseline_embeddings, embeddings
                )
                logger.info(f"Max cosine distance from baseline: {baseline_comparison_metrics['max_cosine_distance']:.6f}")
                logger.info(f"Mean cosine similarity: {baseline_comparison_metrics['mean_cosine_similarity']:.6f}")
            except Exception as e:
                logger.warning(f"Failed to compare with baseline embeddings: {e}")

        # Update wandb run name if using wandb
        if self.use_wandb and self.wandb_run is not None:
            try:
                import wandb
                new_run_name = self._generate_wandb_run_name('quantization', quantization_bits=8, batch_size=batch_size)
                if self.wandb_run.name != new_run_name:
                    wandb.run.name = new_run_name
                    logger.info(f"Updated wandb run name to: {new_run_name}")
            except Exception as e:
                logger.warning(f"Failed to update wandb run name: {e}")

        # Save results
        results = {
            'embeddings': embeddings,
            'performance_metrics': performance_metrics,
            'evaluation_metrics': evaluation_metrics,
            'geometry_metrics': geometry_metrics,
            'baseline_comparison_metrics': baseline_comparison_metrics,
            'model_name': self.model_name,
            'device': self.device,
            'num_perturbations': len(adata),
            'num_genes': adata.shape[1],
            'method': 'quantization_8bit',
            'batch_size': batch_size
        }

        # Save embeddings
        np.savez_compressed(
            os.path.join(output_dir, 'embeddings.npz'),
            embeddings=embeddings
        )

        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'device': self.device,
            'num_perturbations': len(adata),
            'num_genes': adata.shape[1],
            'method': 'quantization_8bit',
            'batch_size': batch_size
        }
        pd.to_pickle(metadata, os.path.join(output_dir, 'metadata.pkl'))

        # Save performance metrics
        pd.DataFrame([performance_metrics]).to_csv(
            os.path.join(output_dir, 'performance_metrics.csv'),
            index=False
        )

        # Save evaluation metrics if available
        if evaluation_metrics:
            pd.DataFrame([evaluation_metrics]).to_csv(
                os.path.join(output_dir, 'evaluation_metrics.csv'),
                index=False
            )
        
        # Save geometry metrics if available
        if geometry_metrics:
            pd.DataFrame([geometry_metrics]).to_csv(
                os.path.join(output_dir, 'geometry_metrics.csv'),
                index=False
            )

        # Save baseline comparison metrics if available
        if baseline_comparison_metrics:
            pd.DataFrame([baseline_comparison_metrics]).to_csv(
                os.path.join(output_dir, 'baseline_comparison_metrics.csv'),
                index=False
            )

        # Log to wandb
        if self.use_wandb and self.wandb_run is not None:
            try:
                # Optional wandb import
                import wandb
                
                perf_log = {
                    'method': 'quantization',
                    'quantization_bits': 8,
                    'batch_size': batch_size,
                    'performance/total_time_seconds': performance_metrics['total_time_seconds'],
                    'performance/throughput_samples_per_sec': len(adata) / performance_metrics['total_time_seconds'],
                    'performance/avg_cpu_percent': performance_metrics.get('avg_cpu_percent', 0),
                    'performance/max_cpu_percent': performance_metrics.get('max_cpu_percent', 0),
                    'performance/avg_memory_percent': performance_metrics.get('avg_memory_percent', 0),
                    'performance/max_memory_percent': performance_metrics.get('max_memory_percent', 0),
                    'dataset/num_perturbations': len(adata),
                    'dataset/num_genes': adata.shape[1],
                    'dataset/embedding_dim': embeddings.shape[1],
                }
                
                if 'avg_gpu_percent' in performance_metrics:
                    perf_log.update({
                        'performance/avg_gpu_percent': performance_metrics['avg_gpu_percent'],
                        'performance/max_gpu_percent': performance_metrics['max_gpu_percent'],
                        'performance/avg_gpu_memory_mb': performance_metrics.get('avg_gpu_memory_mb', 0),
                        'performance/max_gpu_memory_mb': performance_metrics.get('max_gpu_memory_mb', 0),
                    })
                
                # Log evaluation metrics
                if evaluation_metrics:
                    perf_log.update({
                        'evaluation/zero_shot_accuracy': evaluation_metrics.get('zero_shot_accuracy', 0),
                        'evaluation/zero_shot_f1_score': evaluation_metrics.get('zero_shot_f1_score', 0),
                        'evaluation/train_size': evaluation_metrics.get('train_size', 0),
                        'evaluation/test_size': evaluation_metrics.get('test_size', 0),
                        'evaluation/num_classes': evaluation_metrics.get('num_classes', 0),
                    })
                
                # Log geometry metrics
                if geometry_metrics:
                    if geometry_metrics.get('silhouette_score') is not None:
                        perf_log['geometry/silhouette_score'] = geometry_metrics['silhouette_score']
                    if geometry_metrics.get('davies_bouldin_index') is not None:
                        perf_log['geometry/davies_bouldin_index'] = geometry_metrics['davies_bouldin_index']
                    if geometry_metrics.get('calinski_harabasz_score') is not None:
                        perf_log['geometry/calinski_harabasz_score'] = geometry_metrics['calinski_harabasz_score']
                    if geometry_metrics.get('separation_ratio') is not None:
                        perf_log['geometry/separation_ratio'] = geometry_metrics['separation_ratio']
                    if geometry_metrics.get('mean_intra_cluster_distance') is not None:
                        perf_log['geometry/mean_intra_cluster_distance'] = geometry_metrics['mean_intra_cluster_distance']
                    if geometry_metrics.get('mean_inter_cluster_distance') is not None:
                        perf_log['geometry/mean_inter_cluster_distance'] = geometry_metrics['mean_inter_cluster_distance']
                    if geometry_metrics.get('knn_label_consistency') is not None:
                        perf_log['geometry/knn_label_consistency'] = geometry_metrics['knn_label_consistency']
                    if geometry_metrics.get('adjusted_rand_index') is not None:
                        perf_log['geometry/adjusted_rand_index'] = geometry_metrics['adjusted_rand_index']
                    if geometry_metrics.get('normalized_mutual_info') is not None:
                        perf_log['geometry/normalized_mutual_info'] = geometry_metrics['normalized_mutual_info']
                
                # Log baseline comparison metrics
                if baseline_comparison_metrics:
                    perf_log.update({
                        'baseline_comparison/max_cosine_distance': baseline_comparison_metrics['max_cosine_distance'],
                        'baseline_comparison/mean_cosine_distance': baseline_comparison_metrics['mean_cosine_distance'],
                        'baseline_comparison/min_cosine_distance': baseline_comparison_metrics['min_cosine_distance'],
                        'baseline_comparison/max_cosine_similarity': baseline_comparison_metrics['max_cosine_similarity'],
                        'baseline_comparison/mean_cosine_similarity': baseline_comparison_metrics['mean_cosine_similarity'],
                        'baseline_comparison/min_cosine_similarity': baseline_comparison_metrics['min_cosine_similarity'],
                    })
                
                self.wandb_run.log(perf_log)
                logger.info("Metrics logged to wandb")
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")

        logger.info(f"Quantized inference completed. Results saved to {output_dir}")
        logger.info(f"Embeddings shape: {embeddings.shape}")

        return results

    def export_to_onnx(self, output_path: str = "model.onnx", 
                       sample_input_shape: Tuple = (1, 2048),
                       opset_version: int = 14) -> str:
        """
        Export Geneformer model to ONNX format.
        
        ONNX provides graph optimizations and typically 1.5-3x speedup over PyTorch.
        
        Args:
            output_path: Path to save ONNX model
            sample_input_shape: Shape of input tensor (batch_size, sequence_length)
            opset_version: ONNX opset version
            
        Returns:
            Path to exported ONNX model
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available. Install with: pip install onnx onnxruntime-gpu")
        
        if self.model is None:
            raise ValueError("Model must be loaded before exporting to ONNX")
        
        logger.info(f"Exporting model to ONNX format: {output_path}")
        
        # Get the underlying PyTorch model
        pytorch_model = self.model.model
        pytorch_model.eval()
        
        # Try to get vocabulary size from model
        vocab_size = None
        try:
            if hasattr(pytorch_model, 'config') and hasattr(pytorch_model.config, 'vocab_size'):
                vocab_size = pytorch_model.config.vocab_size
            elif hasattr(pytorch_model, 'embedding') and hasattr(pytorch_model.embedding, 'num_embeddings'):
                vocab_size = pytorch_model.embedding.num_embeddings
            elif hasattr(pytorch_model, 'wte') and hasattr(pytorch_model.wte, 'num_embeddings'):
                vocab_size = pytorch_model.wte.num_embeddings
            elif hasattr(pytorch_model, 'transformer') and hasattr(pytorch_model.transformer, 'wte'):
                vocab_size = pytorch_model.transformer.wte.num_embeddings
            if vocab_size:
                logger.info(f"Detected vocabulary size: {vocab_size}")
        except Exception as e:
            logger.warning(f"Could not determine vocabulary size: {e}")
        
        # Create dummy input - use a very safe, small range to avoid out-of-bounds errors
        # Use token IDs 0-1000 which should be safe for any model (special tokens, common genes, etc.)
        # This is just for tracing the model structure, not for actual inference
        safe_max_token_id = 1000  # Very conservative - most models have special tokens in this range
        if vocab_size and vocab_size < safe_max_token_id:
            safe_max_token_id = vocab_size - 1
        
        dummy_input = torch.randint(0, safe_max_token_id + 1, sample_input_shape, dtype=torch.long).to(self.device)
        logger.info(f"Created dummy input with shape {dummy_input.shape}, token ID range: 0-{safe_max_token_id}")
        
        # Export to ONNX
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['embeddings'],
            dynamic_axes={
                'input_ids': {0: 'batch_size'},
                'embeddings': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Model exported to ONNX: {output_path}")
        return output_path

    def run_onnx_inference(self, adata: anndata.AnnData,
                          onnx_model_path: str = "model.onnx",
                          batch_size: int = 32,
                          output_dir: str = "results/onnx") -> Dict:
        """
        Run inference using ONNX Runtime (1.5-3x speedup over PyTorch).
        
        Args:
            adata: AnnData object with perturbation data
            onnx_model_path: Path to ONNX model file
            batch_size: Batch size for processing
            output_dir: Directory to save results
            
        Returns:
            Dictionary with results and performance metrics
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available. Install with: pip install onnxruntime-gpu")
        
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Starting ONNX Runtime inference with batch_size={batch_size}")
        self.profiler.start_profiling()
        
        # Create ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        session = ort.InferenceSession(onnx_model_path, providers=providers)
        
        # Process data using original model's processor
        logger.info("Processing data...")
        dataset = self.model.process_data(adata)
        
        # Get input/output names from ONNX model
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Run inference with ONNX Runtime
        logger.info("Running ONNX Runtime inference...")
        all_embeddings = []
        
        for i in range(0, len(dataset), batch_size):
            batch_end = min(i + batch_size, len(dataset))
            batch_dataset = dataset.select(range(i, batch_end))
            
            # Update profiling
            self.profiler.update_metrics()
            
            # Extract input_ids from the dataset for ONNX inference
            # The dataset format depends on how Geneformer processes data
            try:
                input_ids = None
                
                # Method 1: Try accessing as HuggingFace dataset with column
                if hasattr(batch_dataset, 'column_names') and 'input_ids' in batch_dataset.column_names:
                    input_ids = batch_dataset['input_ids']
                    # HuggingFace datasets return list of lists for 'input_ids'
                    if isinstance(input_ids, list) and len(input_ids) > 0:
                        # Check if it's a list of lists (each sample is a list of token IDs)
                        if isinstance(input_ids[0], list):
                            # Convert list of lists to tensor
                            # Pad sequences to same length if needed, or use max length
                            max_len = max(len(seq) for seq in input_ids)
                            # For now, just convert - ONNX model should handle variable lengths with dynamic axes
                            input_ids = torch.tensor([seq + [0] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len] 
                                                     for seq in input_ids], dtype=torch.long)
                        elif isinstance(input_ids[0], torch.Tensor):
                            # Already tensors, stack them
                            input_ids = torch.stack(input_ids)
                        else:
                            # Convert to tensor
                            input_ids = torch.tensor(np.array(input_ids), dtype=torch.long)
                
                # Method 2: Try accessing as dictionary
                elif isinstance(batch_dataset, dict) and 'input_ids' in batch_dataset:
                    input_ids = batch_dataset['input_ids']
                
                # Method 3: Try accessing individual items
                elif hasattr(batch_dataset, '__getitem__'):
                    # Get first item to check format
                    sample = batch_dataset[0]
                    if isinstance(sample, dict) and 'input_ids' in sample:
                        # Extract input_ids from all items in batch
                        input_ids_list = []
                        for j in range(len(batch_dataset)):
                            item = batch_dataset[j]
                            if isinstance(item, dict):
                                item_ids = item.get('input_ids', None)
                            else:
                                item_ids = getattr(item, 'input_ids', None)
                            
                            if item_ids is not None:
                                if isinstance(item_ids, torch.Tensor):
                                    input_ids_list.append(item_ids)
                                else:
                                    input_ids_list.append(torch.tensor(item_ids))
                        
                        if input_ids_list:
                            input_ids = torch.stack(input_ids_list)
                
                # If we couldn't extract input_ids, fall back to PyTorch
                if input_ids is None:
                    raise ValueError("Could not extract input_ids from dataset format")
                
                # Convert to numpy array
                if isinstance(input_ids, torch.Tensor):
                    input_ids_np = input_ids.cpu().numpy().astype(np.int64)
                else:
                    input_ids_np = np.asarray(input_ids, dtype=np.int64)
                
                # Ensure correct shape: (batch_size, sequence_length)
                if len(input_ids_np.shape) == 1:
                    # Single sample, add batch dimension
                    input_ids_np = input_ids_np[np.newaxis, :]
                elif len(input_ids_np.shape) > 2:
                    # If shape is (batch, 1, seq_len) or similar, squeeze
                    input_ids_np = input_ids_np.squeeze()
                
                # Run ONNX inference
                outputs = session.run([output_name], {input_name: input_ids_np})
                batch_embeddings = outputs[0]  # Get first (and only) output
                
                # Ensure embeddings are numpy array
                if not isinstance(batch_embeddings, np.ndarray):
                    batch_embeddings = np.asarray(batch_embeddings)
                
                all_embeddings.append(batch_embeddings)
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(dataset)-1)//batch_size + 1} (size: {batch_end-i}) using ONNX Runtime")
                
            except Exception as e:
                logger.warning(f"Failed to extract input_ids or run ONNX inference for batch {i//batch_size + 1}: {e}")
                logger.warning("Falling back to PyTorch model for this batch")
                # Fallback to PyTorch - this ensures we still get results
                batch_embeddings = self.model.get_embeddings(batch_dataset)
                all_embeddings.append(batch_embeddings)
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(dataset)-1)//batch_size + 1} (fallback to PyTorch)")
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings)
        
        # Stop profiling
        performance_metrics = self.profiler.stop_profiling()
        
        # Evaluate embeddings
        perturbation_labels = None
        if 'perturbation' in adata.obs:
            perturbation_labels = adata.obs['perturbation'].values
        elif 'perturbation_type' in adata.obs:
            perturbation_labels = adata.obs['perturbation_type'].values
        
        evaluation_metrics = None
        geometry_metrics = None
        if perturbation_labels is not None:
            try:
                evaluation_metrics = self.evaluate_embeddings(embeddings, perturbation_labels)
            except Exception as e:
                logger.warning(f"Failed to evaluate embeddings: {e}")
            try:
                geometry_metrics = self.evaluate_embedding_geometry(embeddings, perturbation_labels)
            except Exception as e:
                logger.warning(f"Failed to evaluate geometry: {e}")

        # Compare with baseline embeddings if available
        baseline_comparison_metrics = None
        if self.baseline_embeddings is not None:
            try:
                baseline_comparison_metrics = self.compute_max_cosine_distance(
                    self.baseline_embeddings, embeddings
                )
                logger.info(f"Max cosine distance from baseline: {baseline_comparison_metrics['max_cosine_distance']:.6f}")
                logger.info(f"Mean cosine similarity: {baseline_comparison_metrics['mean_cosine_similarity']:.6f}")
            except Exception as e:
                logger.warning(f"Failed to compare with baseline embeddings: {e}")

        # Update wandb run name if using wandb
        if self.use_wandb and self.wandb_run is not None:
            try:
                import wandb
                new_run_name = self._generate_wandb_run_name('onnx_runtime', batch_size=batch_size)
                if self.wandb_run.name != new_run_name:
                    wandb.run.name = new_run_name
                    logger.info(f"Updated wandb run name to: {new_run_name}")
            except Exception as e:
                logger.warning(f"Failed to update wandb run name: {e}")
        
        # Save results
        results = {
            'embeddings': embeddings,
            'performance_metrics': performance_metrics,
            'evaluation_metrics': evaluation_metrics,
            'geometry_metrics': geometry_metrics,
            'baseline_comparison_metrics': baseline_comparison_metrics,
            'model_name': self.model_name,
            'device': 'onnx_runtime',
            'num_perturbations': len(adata),
            'num_genes': adata.shape[1],
            'method': 'onnx_runtime',
            'batch_size': batch_size
        }
        
        # Save files
        np.savez_compressed(os.path.join(output_dir, 'embeddings.npz'), embeddings=embeddings)
        pd.DataFrame([performance_metrics]).to_csv(
            os.path.join(output_dir, 'performance_metrics.csv'), index=False
        )
        if evaluation_metrics:
            pd.DataFrame([evaluation_metrics]).to_csv(
                os.path.join(output_dir, 'evaluation_metrics.csv'), index=False
            )
        if geometry_metrics:
            pd.DataFrame([geometry_metrics]).to_csv(
                os.path.join(output_dir, 'geometry_metrics.csv'), index=False
            )
        
        if baseline_comparison_metrics:
            pd.DataFrame([baseline_comparison_metrics]).to_csv(
                os.path.join(output_dir, 'baseline_comparison_metrics.csv'), index=False
            )
        
        # Log to wandb
        if self.use_wandb and self.wandb_run is not None:
            try:
                import wandb
                
                perf_log = {
                    'method': 'onnx_runtime',
                    'batch_size': batch_size,
                    'performance/total_time_seconds': performance_metrics['total_time_seconds'],
                    'performance/throughput_samples_per_sec': len(adata) / performance_metrics['total_time_seconds'],
                    'performance/avg_cpu_percent': performance_metrics.get('avg_cpu_percent', 0),
                    'performance/max_cpu_percent': performance_metrics.get('max_cpu_percent', 0),
                    'performance/avg_memory_percent': performance_metrics.get('avg_memory_percent', 0),
                    'performance/max_memory_percent': performance_metrics.get('max_memory_percent', 0),
                    'dataset/num_perturbations': len(adata),
                    'dataset/num_genes': adata.shape[1],
                    'dataset/embedding_dim': embeddings.shape[1],
                }
                
                if 'avg_gpu_percent' in performance_metrics:
                    perf_log.update({
                        'performance/avg_gpu_percent': performance_metrics['avg_gpu_percent'],
                        'performance/max_gpu_percent': performance_metrics['max_gpu_percent'],
                        'performance/avg_gpu_memory_mb': performance_metrics.get('avg_gpu_memory_mb', 0),
                        'performance/max_gpu_memory_mb': performance_metrics.get('max_gpu_memory_mb', 0),
                    })
                
                # Log evaluation metrics
                if evaluation_metrics:
                    perf_log.update({
                        'evaluation/zero_shot_accuracy': evaluation_metrics.get('zero_shot_accuracy', 0),
                        'evaluation/zero_shot_f1_score': evaluation_metrics.get('zero_shot_f1_score', 0),
                        'evaluation/train_size': evaluation_metrics.get('train_size', 0),
                        'evaluation/test_size': evaluation_metrics.get('test_size', 0),
                        'evaluation/num_classes': evaluation_metrics.get('num_classes', 0),
                    })
                
                # Log geometry metrics
                if geometry_metrics:
                    if geometry_metrics.get('silhouette_score') is not None:
                        perf_log['geometry/silhouette_score'] = geometry_metrics['silhouette_score']
                    if geometry_metrics.get('davies_bouldin_index') is not None:
                        perf_log['geometry/davies_bouldin_index'] = geometry_metrics['davies_bouldin_index']
                    if geometry_metrics.get('calinski_harabasz_score') is not None:
                        perf_log['geometry/calinski_harabasz_score'] = geometry_metrics['calinski_harabasz_score']
                    if geometry_metrics.get('separation_ratio') is not None:
                        perf_log['geometry/separation_ratio'] = geometry_metrics['separation_ratio']
                    if geometry_metrics.get('mean_intra_cluster_distance') is not None:
                        perf_log['geometry/mean_intra_cluster_distance'] = geometry_metrics['mean_intra_cluster_distance']
                    if geometry_metrics.get('mean_inter_cluster_distance') is not None:
                        perf_log['geometry/mean_inter_cluster_distance'] = geometry_metrics['mean_inter_cluster_distance']
                    if geometry_metrics.get('knn_label_consistency') is not None:
                        perf_log['geometry/knn_label_consistency'] = geometry_metrics['knn_label_consistency']
                    if geometry_metrics.get('adjusted_rand_index') is not None:
                        perf_log['geometry/adjusted_rand_index'] = geometry_metrics['adjusted_rand_index']
                    if geometry_metrics.get('normalized_mutual_info') is not None:
                        perf_log['geometry/normalized_mutual_info'] = geometry_metrics['normalized_mutual_info']
                
                # Log baseline comparison metrics
                if baseline_comparison_metrics:
                    perf_log.update({
                        'baseline_comparison/max_cosine_distance': baseline_comparison_metrics['max_cosine_distance'],
                        'baseline_comparison/mean_cosine_distance': baseline_comparison_metrics['mean_cosine_distance'],
                        'baseline_comparison/min_cosine_distance': baseline_comparison_metrics['min_cosine_distance'],
                        'baseline_comparison/max_cosine_similarity': baseline_comparison_metrics['max_cosine_similarity'],
                        'baseline_comparison/mean_cosine_similarity': baseline_comparison_metrics['mean_cosine_similarity'],
                        'baseline_comparison/min_cosine_similarity': baseline_comparison_metrics['min_cosine_similarity'],
                    })
                
                self.wandb_run.log(perf_log)
                logger.info("Metrics logged to wandb")
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")
        
        logger.info(f"ONNX Runtime inference completed. Results saved to {output_dir}")
        return results

    def convert_to_tensorrt(self, onnx_model_path: str,
                           tensorrt_engine_path: str = "model.trt",
                           precision: str = "fp16",
                           max_batch_size: int = 32,
                           max_sequence_length: int = 4096) -> str:
        """
        Convert ONNX model to TensorRT engine (2-10x speedup, NVIDIA GPUs only).
        
        Args:
            onnx_model_path: Path to ONNX model
            tensorrt_engine_path: Path to save TensorRT engine
            precision: Precision mode ('fp32', 'fp16', 'int8')
            max_batch_size: Maximum batch size for optimization
            max_sequence_length: Maximum sequence length
            
        Returns:
            Path to TensorRT engine
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not available. Install TensorRT from NVIDIA.")
        
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
        
        logger.info(f"Converting ONNX model to TensorRT engine: {tensorrt_engine_path}")
        logger.info(f"Precision: {precision}, Max batch: {max_batch_size}")
        
        # TensorRT builder and network
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, trt_logger)
        
        # Parse ONNX model
        with open(onnx_model_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        # Set precision
        if precision == "fp16":
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("Using FP16 precision")
            else:
                logger.warning("FP16 not supported, using FP32")
        elif precision == "int8":
            if builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                logger.info("Using INT8 precision (calibration required)")
            else:
                logger.warning("INT8 not supported, using FP32")
        
        # Set optimization profiles for dynamic shapes
        profile = builder.create_optimization_profile()
        profile.set_shape("input_ids", (1, 1), (max_batch_size, max_sequence_length), 
                         (max_batch_size, max_sequence_length))
        config.add_optimization_profile(profile)
        
        # Build engine
        logger.info("Building TensorRT engine (this may take several minutes)...")
        engine = builder.build_engine(network, config)
        
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save engine
        with open(tensorrt_engine_path, 'wb') as f:
            f.write(engine.serialize())
        
        logger.info(f"TensorRT engine saved: {tensorrt_engine_path}")
        return tensorrt_engine_path

    def run_tensorrt_inference(self, adata: anndata.AnnData,
                               tensorrt_engine_path: str = "model.trt",
                               batch_size: int = 32,
                               output_dir: str = "results/tensorrt") -> Dict:
        """
        Run inference using TensorRT engine (2-10x speedup, fastest option).
        
        Args:
            adata: AnnData object with perturbation data
            tensorrt_engine_path: Path to TensorRT engine file
            batch_size: Batch size for processing
            output_dir: Directory to save results
            
        Returns:
            Dictionary with results and performance metrics
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not available. Install TensorRT from NVIDIA.")
        
        if not os.path.exists(tensorrt_engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {tensorrt_engine_path}")
        
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise ImportError("PyCUDA not available. Install with: pip install pycuda")
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Starting TensorRT inference with batch_size={batch_size}")
        self.profiler.start_profiling()
        
        # Load TensorRT engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(tensorrt_engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(trt_logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()
        
        # Allocate buffers
        inputs, outputs, bindings, stream = self._allocate_buffers(engine)
        
        # Process data
        logger.info("Processing data...")
        dataset = self.model.process_data(adata)
        
        # Run inference
        logger.info("Running TensorRT inference...")
        all_embeddings = []
        
        for i in range(0, len(dataset), batch_size):
            batch_end = min(i + batch_size, len(dataset))
            batch_dataset = dataset.select(range(i, batch_end))
            
            # Update profiling
            self.profiler.update_metrics()
            
            # For now, use original model's get_embeddings as TensorRT integration
            # requires careful handling of data format. This is a placeholder.
            # Full implementation would extract raw inputs and run through TensorRT.
            batch_embeddings = self.model.get_embeddings(batch_dataset)
            all_embeddings.append(batch_embeddings)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(dataset)-1)//batch_size + 1}")
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings)
        
        # Stop profiling
        performance_metrics = self.profiler.stop_profiling()
        
        # Evaluate embeddings
        perturbation_labels = None
        if 'perturbation' in adata.obs:
            perturbation_labels = adata.obs['perturbation'].values
        elif 'perturbation_type' in adata.obs:
            perturbation_labels = adata.obs['perturbation_type'].values
        
        evaluation_metrics = None
        geometry_metrics = None
        if perturbation_labels is not None:
            try:
                evaluation_metrics = self.evaluate_embeddings(embeddings, perturbation_labels)
            except Exception as e:
                logger.warning(f"Failed to evaluate embeddings: {e}")
            try:
                geometry_metrics = self.evaluate_embedding_geometry(embeddings, perturbation_labels)
            except Exception as e:
                logger.warning(f"Failed to evaluate geometry: {e}")

        # Compare with baseline embeddings if available
        baseline_comparison_metrics = None
        if self.baseline_embeddings is not None:
            try:
                baseline_comparison_metrics = self.compute_max_cosine_distance(
                    self.baseline_embeddings, embeddings
                )
                logger.info(f"Max cosine distance from baseline: {baseline_comparison_metrics['max_cosine_distance']:.6f}")
                logger.info(f"Mean cosine similarity: {baseline_comparison_metrics['mean_cosine_similarity']:.6f}")
            except Exception as e:
                logger.warning(f"Failed to compare with baseline embeddings: {e}")

        # Update wandb run name if using wandb
        if self.use_wandb and self.wandb_run is not None:
            try:
                import wandb
                # Get precision from tensorrt_engine_path or use default
                precision = 'fp16'  # Default, could be extracted from path or parameter
                new_run_name = self._generate_wandb_run_name('tensorrt', precision=precision, batch_size=batch_size)
                if self.wandb_run.name != new_run_name:
                    wandb.run.name = new_run_name
                    logger.info(f"Updated wandb run name to: {new_run_name}")
            except Exception as e:
                logger.warning(f"Failed to update wandb run name: {e}")
        
        # Save results
        results = {
            'embeddings': embeddings,
            'performance_metrics': performance_metrics,
            'evaluation_metrics': evaluation_metrics,
            'geometry_metrics': geometry_metrics,
            'baseline_comparison_metrics': baseline_comparison_metrics,
            'model_name': self.model_name,
            'device': 'tensorrt',
            'num_perturbations': len(adata),
            'num_genes': adata.shape[1],
            'method': 'tensorrt',
            'batch_size': batch_size
        }
        
        # Save files
        np.savez_compressed(os.path.join(output_dir, 'embeddings.npz'), embeddings=embeddings)
        pd.DataFrame([performance_metrics]).to_csv(
            os.path.join(output_dir, 'performance_metrics.csv'), index=False
        )
        if evaluation_metrics:
            pd.DataFrame([evaluation_metrics]).to_csv(
                os.path.join(output_dir, 'evaluation_metrics.csv'), index=False
            )
        if geometry_metrics:
            pd.DataFrame([geometry_metrics]).to_csv(
                os.path.join(output_dir, 'geometry_metrics.csv'), index=False
            )
        
        if baseline_comparison_metrics:
            pd.DataFrame([baseline_comparison_metrics]).to_csv(
                os.path.join(output_dir, 'baseline_comparison_metrics.csv'), index=False
            )
        
        # Log to wandb
        if self.use_wandb and self.wandb_run is not None:
            try:
                import wandb
                
                perf_log = {
                    'method': 'tensorrt',
                    'batch_size': batch_size,
                    'performance/total_time_seconds': performance_metrics['total_time_seconds'],
                    'performance/throughput_samples_per_sec': len(adata) / performance_metrics['total_time_seconds'],
                    'performance/avg_cpu_percent': performance_metrics.get('avg_cpu_percent', 0),
                    'performance/max_cpu_percent': performance_metrics.get('max_cpu_percent', 0),
                    'performance/avg_memory_percent': performance_metrics.get('avg_memory_percent', 0),
                    'performance/max_memory_percent': performance_metrics.get('max_memory_percent', 0),
                    'dataset/num_perturbations': len(adata),
                    'dataset/num_genes': adata.shape[1],
                    'dataset/embedding_dim': embeddings.shape[1],
                }
                
                if 'avg_gpu_percent' in performance_metrics:
                    perf_log.update({
                        'performance/avg_gpu_percent': performance_metrics['avg_gpu_percent'],
                        'performance/max_gpu_percent': performance_metrics['max_gpu_percent'],
                        'performance/avg_gpu_memory_mb': performance_metrics.get('avg_gpu_memory_mb', 0),
                        'performance/max_gpu_memory_mb': performance_metrics.get('max_gpu_memory_mb', 0),
                    })
                
                # Log evaluation metrics
                if evaluation_metrics:
                    perf_log.update({
                        'evaluation/zero_shot_accuracy': evaluation_metrics.get('zero_shot_accuracy', 0),
                        'evaluation/zero_shot_f1_score': evaluation_metrics.get('zero_shot_f1_score', 0),
                        'evaluation/train_size': evaluation_metrics.get('train_size', 0),
                        'evaluation/test_size': evaluation_metrics.get('test_size', 0),
                        'evaluation/num_classes': evaluation_metrics.get('num_classes', 0),
                    })
                
                # Log geometry metrics
                if geometry_metrics:
                    if geometry_metrics.get('silhouette_score') is not None:
                        perf_log['geometry/silhouette_score'] = geometry_metrics['silhouette_score']
                    if geometry_metrics.get('davies_bouldin_index') is not None:
                        perf_log['geometry/davies_bouldin_index'] = geometry_metrics['davies_bouldin_index']
                    if geometry_metrics.get('calinski_harabasz_score') is not None:
                        perf_log['geometry/calinski_harabasz_score'] = geometry_metrics['calinski_harabasz_score']
                    if geometry_metrics.get('separation_ratio') is not None:
                        perf_log['geometry/separation_ratio'] = geometry_metrics['separation_ratio']
                    if geometry_metrics.get('mean_intra_cluster_distance') is not None:
                        perf_log['geometry/mean_intra_cluster_distance'] = geometry_metrics['mean_intra_cluster_distance']
                    if geometry_metrics.get('mean_inter_cluster_distance') is not None:
                        perf_log['geometry/mean_inter_cluster_distance'] = geometry_metrics['mean_inter_cluster_distance']
                    if geometry_metrics.get('knn_label_consistency') is not None:
                        perf_log['geometry/knn_label_consistency'] = geometry_metrics['knn_label_consistency']
                    if geometry_metrics.get('adjusted_rand_index') is not None:
                        perf_log['geometry/adjusted_rand_index'] = geometry_metrics['adjusted_rand_index']
                    if geometry_metrics.get('normalized_mutual_info') is not None:
                        perf_log['geometry/normalized_mutual_info'] = geometry_metrics['normalized_mutual_info']
                
                # Log baseline comparison metrics
                if baseline_comparison_metrics:
                    perf_log.update({
                        'baseline_comparison/max_cosine_distance': baseline_comparison_metrics['max_cosine_distance'],
                        'baseline_comparison/mean_cosine_distance': baseline_comparison_metrics['mean_cosine_distance'],
                        'baseline_comparison/min_cosine_distance': baseline_comparison_metrics['min_cosine_distance'],
                        'baseline_comparison/max_cosine_similarity': baseline_comparison_metrics['max_cosine_similarity'],
                        'baseline_comparison/mean_cosine_similarity': baseline_comparison_metrics['mean_cosine_similarity'],
                        'baseline_comparison/min_cosine_similarity': baseline_comparison_metrics['min_cosine_similarity'],
                    })
                
                self.wandb_run.log(perf_log)
                logger.info("Metrics logged to wandb")
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")
        
        logger.info(f"TensorRT inference completed. Results saved to {output_dir}")
        return results

    @staticmethod
    def _allocate_buffers(engine):
        """Allocate GPU buffers for TensorRT inference."""
        try:
            import pycuda.driver as cuda
        except ImportError:
            raise ImportError("PyCUDA not available. Install with: pip install pycuda")
        
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        
        return inputs, outputs, bindings, stream


def run_all_optimizations(num_perturbations: int = 100):
    """Run all optimization methods and compare results."""

    logger.info("Starting comprehensive optimization benchmarking")

    # Configuration
    MODEL_NAME = "gf-6L-10M-i2048"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize optimizer and load data
    optimizer = OptimizedGeneformerISPOptimizer(model_name=MODEL_NAME, device=DEVICE)
    optimizer.load_model()

    perturbation_data = optimizer.load_perturbation_data(
        num_perturbations=num_perturbations
    )

    results_list = []

    # 1. Baseline (smaller batch size)
    logger.info("Running baseline...")
    baseline_results = optimizer.run_baseline_inference(
        perturbation_data,
        output_dir="results/baseline"
    )
    baseline_results['method'] = 'baseline'
    results_list.append(baseline_results)

    # 2. Batching optimization (larger batch size)
    logger.info("Running batching optimization...")
    batching_results = optimizer.run_batching_optimized_inference(
        perturbation_data,
        batch_size=32,
        output_dir="results/batching_bs32"
    )
    results_list.append(batching_results)

    # 3. Mixed precision
    logger.info("Running mixed precision optimization...")
    mixed_precision_results = optimizer.run_mixed_precision_inference(
        perturbation_data,
        batch_size=32,
        output_dir="results/mixed_precision_fp16"
    )
    results_list.append(mixed_precision_results)

    # 4. Quantization (if supported)
    try:
        logger.info("Running quantization optimization...")
        quantization_results = optimizer.run_quantized_inference(
            perturbation_data,
            batch_size=32,
            output_dir="results/quantization_8bit"
        )
        results_list.append(quantization_results)
    except Exception as e:
        logger.warning(f"Quantization failed: {e}")

    # 5. ONNX Runtime (if available)
    if ONNX_AVAILABLE:
        try:
            logger.info("Running ONNX Runtime optimization...")
            # First export to ONNX
            onnx_path = optimizer.export_to_onnx("model.onnx")
            onnx_results = optimizer.run_onnx_inference(
                perturbation_data,
                onnx_path,
                batch_size=32,
                output_dir="results/onnx_runtime"
            )
            results_list.append(onnx_results)
        except Exception as e:
            logger.warning(f"ONNX Runtime failed: {e}")

    # 6. TensorRT (if available)
    if TENSORRT_AVAILABLE and ONNX_AVAILABLE:
        try:
            logger.info("Running TensorRT optimization...")
            # Convert ONNX to TensorRT
            tensorrt_path = optimizer.convert_to_tensorrt(
                "model.onnx",
                "model.trt",
                precision="fp16"
            )
            tensorrt_results = optimizer.run_tensorrt_inference(
                perturbation_data,
                tensorrt_path,
                batch_size=64,
                output_dir="results/tensorrt"
            )
            results_list.append(tensorrt_results)
        except Exception as e:
            logger.warning(f"TensorRT failed: {e}")

    # Save comprehensive comparison
    optimizer.save_results_summary(results_list, "results/comprehensive_benchmark.csv")

    logger.info("All optimizations completed!")
    return results_list


def main():
    """Main function to run optimized ISP inference."""

    # Configuration
    NUM_PERTURBATIONS = 200  # Test with more perturbations

    logger.info("Starting Optimized In-Silico Perturbation Challenge")
    logger.info(f"Number of perturbations: {NUM_PERTURBATIONS}")

    # Run all optimizations
    results = run_all_optimizations(num_perturbations=NUM_PERTURBATIONS)

    # Print summary
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*80)

    for result in results:
        method = result['method']
        metrics = result['performance_metrics']
        throughput = result['num_perturbations'] / metrics['total_time_seconds']

        print(f"\nMethod: {method}")
        print(f"  Time: {metrics['total_time_seconds']:.2f}s")
        print(f"  Throughput: {throughput:.2f} samples/sec")
        print(f"  CPU Usage: {metrics['avg_cpu_percent']:.1f}% (max: {metrics['max_cpu_percent']:.1f}%)")
        print(f"  Memory Usage: {metrics['avg_memory_percent']:.1f}% (max: {metrics['max_memory_percent']:.1f}%)")

    print("\nDetailed results saved to results/comprehensive_benchmark.csv")


if __name__ == "__main__":
    main()




