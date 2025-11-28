#!/usr/bin/env python3
"""
In-Silico Perturbation Optimization Challenge - Baseline Implementation

This script implements the baseline for in-silico perturbation (ISP) inference using Geneformer.
It measures performance metrics and saves outputs for comparison with optimized versions.
"""

import os
import time
import numpy as np
import pandas as pd
import psutil
import GPUtil
from memory_profiler import profile
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from helical.models.geneformer import Geneformer, GeneformerConfig
import anndata
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Import distributed utilities
try:
    from .distributed import (
        setup_ddp, cleanup_ddp, is_ddp_available, get_world_size, get_rank,
        is_main_process, wrap_model_with_ddp, split_data_for_rank
    )
    DDP_AVAILABLE = True
except ImportError:
    DDP_AVAILABLE = False
    logger.warning("DDP utilities not available")

# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Install with: pip install wandb")

# Load environment variables from .env file for wandb API key
try:
    from dotenv import load_dotenv
    from pathlib import Path
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import RidgeClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from .profiler import PerformanceProfiler


class GeneformerISPOptimizer:
    """Main class for Geneformer In-Silico Perturbation optimization."""

    def __init__(self, model_name: str = "gf-6L-10M-i2048", device: str = "cuda",
                 use_wandb: bool = False, wandb_project: str = "ispo-baseline",
                 wandb_run_name: Optional[str] = None, wandb_config: Optional[Dict] = None,
                 num_gpus: Optional[int] = None, use_ddp: bool = False):
        """
        Initialize the Geneformer ISP optimizer.

        Args:
            model_name: Name of the Geneformer model to use
            device: Device to run inference on ('cuda' or 'cpu')
            use_wandb: Whether to use Weights & Biases for tracking
            wandb_project: wandb project name
            wandb_run_name: Optional name for the wandb run
            wandb_config: Optional config dictionary for wandb
            num_gpus: Number of GPUs to use for multi-GPU inference (None = use all available, 1 = single GPU)
            use_ddp: Whether to use Distributed Data Parallel (DDP) instead of DataParallel
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_run = None
        self.use_ddp = use_ddp and DDP_AVAILABLE
        
        # Multi-GPU support
        if device == "cuda" and torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            
            # DDP mode: use environment variables set by torchrun
            if self.use_ddp:
                if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                    self.rank = int(os.environ['RANK'])
                    self.world_size = int(os.environ['WORLD_SIZE'])
                    self.local_rank = int(os.environ.get('LOCAL_RANK', self.rank))
                    self.num_gpus = self.world_size
                    
                    # Setup DDP
                    setup_ddp(self.rank, self.world_size)
                    torch.cuda.set_device(self.local_rank)
                    self.device = f"cuda:{self.local_rank}"
                    
                    logger.info(f"DDP mode: rank={self.rank}, world_size={self.world_size}, local_rank={self.local_rank}")
                else:
                    logger.warning("DDP requested but RANK/WORLD_SIZE not set. Falling back to DataParallel.")
                    self.use_ddp = False
                    self.rank = 0
                    self.world_size = 1
                    self.local_rank = 0
                    if num_gpus is None:
                        self.num_gpus = available_gpus
                    else:
                        self.num_gpus = min(num_gpus, available_gpus)
            else:
                # DataParallel mode
                self.rank = 0
                self.world_size = 1
                self.local_rank = 0
                if num_gpus is None:
                    # Use all available GPUs by default
                    self.num_gpus = available_gpus
                else:
                    # Use specified number, but cap at available
                    self.num_gpus = min(num_gpus, available_gpus)
            
            if self.num_gpus > 1 and not self.use_ddp:
                logger.info(f"Multi-GPU mode (DataParallel): Using {self.num_gpus} GPUs")
                for i in range(self.num_gpus):
                    logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            elif self.num_gpus == 1:
                self.num_gpus = 1
        else:
            self.num_gpus = 1
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.use_ddp = False
        
        # Initialize wandb if requested
        if self.use_wandb:
            config = wandb_config or {}
            config.update({
                'model_name': model_name,
                'device': device,
            })
            self.wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name or f"{model_name}_{device}",
                config=config,
                tags=["baseline", "geneformer", "isp"]
            )
            logger.info(f"Initialized wandb run: {self.wandb_run.name}")
        
        self.profiler = PerformanceProfiler(use_wandb=self.use_wandb, wandb_run=self.wandb_run)

        # Check device availability
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"

        logger.info(f"Using device: {self.device}")

    def load_model(self):
        """Load the Geneformer model."""
        if is_main_process() or not self.use_ddp:
            logger.info(f"Loading Geneformer model: {self.model_name}")
        
        config = GeneformerConfig(
            model_name=self.model_name,
            device=self.device,
            batch_size=1  # Start with batch size 1 for baseline
        )
        self.model = Geneformer(config)
        
        # Wrap model with DDP or DataParallel for multi-GPU support
        if self.use_ddp and self.num_gpus > 1:
            # DDP mode
            if hasattr(self.model, 'model'):
                device = torch.device(self.device)
                self.model.model = wrap_model_with_ddp(
                    self.model.model,
                    device=device,
                    find_unused_parameters=False
                )
                if is_main_process():
                    logger.info(f"Model wrapped with DDP on {self.num_gpus} GPUs")
            else:
                logger.warning("Model does not have 'model' attribute. DDP may not work correctly.")
        elif self.num_gpus > 1 and torch.cuda.device_count() > 1 and not self.use_ddp:
            # DataParallel mode (fallback)
            if is_main_process() or not self.use_ddp:
                logger.info(f"Wrapping model with DataParallel for {self.num_gpus} GPUs")
            # Access the underlying PyTorch model
            if hasattr(self.model, 'model'):
                self.model.model = DataParallel(
                    self.model.model,
                    device_ids=list(range(self.num_gpus))
                )
                if is_main_process() or not self.use_ddp:
                    logger.info(f"Model wrapped with DataParallel on GPUs: {list(range(self.num_gpus))}")
            else:
                logger.warning("Model does not have 'model' attribute. Multi-GPU may not work correctly.")
        
        if is_main_process() or not self.use_ddp:
            logger.info("Model loaded successfully")

    def load_perturbation_data(self, data_path: str = "SrivatsanTrapnell2020_sciplex2.h5ad",
                              num_cells: int = 1000, dataset: str = "sciplex2") -> anndata.AnnData:
        """
        Load real perturbation data from sciplex2 dataset.

        Args:
            data_path: Path to the sciplex2 .h5ad file
            num_cells: Number of cells to select (default: 1000)
            dataset: Dataset name ('sciplex2')

        Returns:
            AnnData object with perturbation data
        """
        logger.info(f"Loading perturbation data from {data_path}")

        if not os.path.exists(data_path):
            logger.info("Downloading sciplex2 data...")
            url = "https://zenodo.org/record/10044268/files/SrivatsanTrapnell2020_sciplex2.h5ad?download=1"
            import urllib.request
            try:
                urllib.request.urlretrieve(url, data_path)
                logger.info(f"Downloaded sciplex2 data to {data_path}")
            except Exception as e:
                logger.error(f"Failed to download sciplex2: {e}")
                logger.error(f"Please download manually from: {url}")
                raise

        # Load the data
        adata = anndata.read_h5ad(data_path)
        logger.info(f"Loaded data with shape: {adata.shape}")

        # Subset to requested number of cells
        if num_cells < len(adata):
            # Randomly sample cells
            np.random.seed(42)  # For reproducibility
            selected_indices = np.random.choice(adata.obs.index, num_cells, replace=False)
            adata = adata[selected_indices]
            logger.info(f"Selected {num_cells} cells from dataset")

        logger.info(f"Using {len(adata)} cells from sciplex2 dataset")
        if 'perturbation' in adata.obs:
            perturbation_types = adata.obs['perturbation'].value_counts().to_dict()
            logger.info(f"Perturbation types: {perturbation_types}")
        elif 'perturbation_type' in adata.obs:
            perturbation_types = adata.obs['perturbation_type'].value_counts().to_dict()
            logger.info(f"Perturbation types: {perturbation_types}")
        elif 'treatment' in adata.obs:
            treatment_types = adata.obs['treatment'].value_counts().to_dict()
            logger.info(f"Treatment types: {treatment_types}")

        return adata

    def evaluate_embeddings(self, embeddings: np.ndarray, labels: np.ndarray,
                           test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
        """
        Evaluate embedding quality using zero-shot classification with RidgeClassifier.
        This follows the evaluation approach from Helical's Geneformer scaling evaluation.

        Args:
            embeddings: Cell embeddings array (n_samples, n_features)
            labels: Perturbation labels array (n_samples,)
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with accuracy and F1 score metrics
        """
        logger.info("Evaluating embeddings using zero-shot classification...")
        
        # Split data into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels if len(np.unique(labels)) > 1 else None
        )
        
        logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
        logger.info(f"Number of unique labels: {len(np.unique(labels))}")
        
        # Train RidgeClassifier (no hyperparameter tuning, as in Helical evaluation)
        clf = RidgeClassifier()
        clf.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = clf.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'zero_shot_accuracy': accuracy,
            'zero_shot_f1_score': f1,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'num_classes': len(np.unique(labels))
        }
        
        logger.info(f"Zero-shot Accuracy: {accuracy:.4f}")
        logger.info(f"Zero-shot F1 Score (weighted): {f1:.4f}")
        
        return metrics

    def evaluate_embedding_geometry(self, embeddings: np.ndarray, labels: np.ndarray,
                                    k_neighbors: int = 15) -> Dict[str, float]:
        """
        Evaluate embedding geometry and clustering quality based on perturbation labels.
        These metrics assess how well-separated different perturbations are in embedding space.
        
        Good embeddings should:
        - Have high separation between different perturbations (high inter-cluster distance)
        - Have tight clustering within the same perturbation (low intra-cluster distance)
        - Show clear separation from control
        
        Args:
            embeddings: Cell embeddings array (n_samples, n_features)
            labels: Perturbation labels array (n_samples,)
            k_neighbors: Number of neighbors for k-NN metrics
            
        Returns:
            Dictionary with geometry and clustering metrics
        """
        logger.info("Evaluating embedding geometry and clustering quality...")
        
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        if n_classes < 2:
            logger.warning("Need at least 2 classes for geometry evaluation. Skipping.")
            return {}
        
        metrics = {}
        
        # 1. Silhouette Score - measures how well-separated clusters are
        # Range: -1 to 1, higher is better
        # Positive values indicate good separation
        try:
            silhouette = silhouette_score(embeddings, labels)
            metrics['silhouette_score'] = silhouette
            logger.info(f"Silhouette Score: {silhouette:.4f} (higher is better, range: -1 to 1)")
        except Exception as e:
            logger.warning(f"Failed to compute silhouette score: {e}")
            metrics['silhouette_score'] = None
        
        # 2. Davies-Bouldin Index - measures average similarity ratio of clusters
        # Lower is better (0 is best)
        try:
            db_index = davies_bouldin_score(embeddings, labels)
            metrics['davies_bouldin_index'] = db_index
            logger.info(f"Davies-Bouldin Index: {db_index:.4f} (lower is better)")
        except Exception as e:
            logger.warning(f"Failed to compute Davies-Bouldin index: {e}")
            metrics['davies_bouldin_index'] = None
        
        # 3. Calinski-Harabasz Score (Variance Ratio Criterion)
        # Higher is better - measures ratio of between-cluster to within-cluster variance
        try:
            ch_score = calinski_harabasz_score(embeddings, labels)
            metrics['calinski_harabasz_score'] = ch_score
            logger.info(f"Calinski-Harabasz Score: {ch_score:.4f} (higher is better)")
        except Exception as e:
            logger.warning(f"Failed to compute Calinski-Harabasz score: {e}")
            metrics['calinski_harabasz_score'] = None
        
        # 4. Inter-cluster vs Intra-cluster distances
        # Good embeddings should have high inter-cluster and low intra-cluster distances
        try:
            inter_cluster_dists = []
            intra_cluster_dists = []
            
            for label in unique_labels:
                label_mask = labels == label
                label_embeddings = embeddings[label_mask]
                
                # Intra-cluster distances (within same perturbation)
                if len(label_embeddings) > 1:
                    intra_dists = euclidean_distances(label_embeddings)
                    # Get upper triangle (avoid duplicates and diagonal)
                    mask = np.triu(np.ones_like(intra_dists, dtype=bool), k=1)
                    intra_cluster_dists.extend(intra_dists[mask].tolist())
                
                # Inter-cluster distances (to other perturbations)
                other_mask = labels != label
                if np.any(other_mask):
                    other_embeddings = embeddings[other_mask]
                    # Compute distances from this cluster to others
                    inter_dists = euclidean_distances(label_embeddings, other_embeddings)
                    inter_cluster_dists.extend(inter_dists.flatten().tolist())
            
            mean_intra = np.mean(intra_cluster_dists) if intra_cluster_dists else 0
            mean_inter = np.mean(inter_cluster_dists) if inter_cluster_dists else 0
            separation_ratio = mean_inter / (mean_intra + 1e-8) if mean_intra > 0 else 0
            
            metrics['mean_intra_cluster_distance'] = mean_intra
            metrics['mean_inter_cluster_distance'] = mean_inter
            metrics['separation_ratio'] = separation_ratio  # Higher is better
            
            logger.info(f"Mean intra-cluster distance: {mean_intra:.4f}")
            logger.info(f"Mean inter-cluster distance: {mean_inter:.4f}")
            logger.info(f"Separation ratio (inter/intra): {separation_ratio:.4f} (higher is better)")
        except Exception as e:
            logger.warning(f"Failed to compute cluster distance metrics: {e}")
            metrics['mean_intra_cluster_distance'] = None
            metrics['mean_inter_cluster_distance'] = None
            metrics['separation_ratio'] = None
        
        # 5. k-Nearest Neighbor Label Consistency
        # For each cell, check how many of its k nearest neighbors share the same label
        # Higher values indicate better local clustering
        try:
            nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean')
            nn.fit(embeddings)
            distances, indices = nn.kneighbors(embeddings)
            
            # Remove self (first neighbor is always self)
            indices = indices[:, 1:]
            
            k_nn_consistency = []
            for i, label in enumerate(labels):
                neighbor_labels = labels[indices[i]]
                same_label_count = np.sum(neighbor_labels == label)
                consistency = same_label_count / k_neighbors
                k_nn_consistency.append(consistency)
            
            mean_knn_consistency = np.mean(k_nn_consistency)
            metrics['knn_label_consistency'] = mean_knn_consistency
            metrics['knn_k'] = k_neighbors
            logger.info(f"k-NN label consistency (k={k_neighbors}): {mean_knn_consistency:.4f} (higher is better)")
        except Exception as e:
            logger.warning(f"Failed to compute k-NN consistency: {e}")
            metrics['knn_label_consistency'] = None
        
        # 6. Clustering quality using KMeans (unsupervised) vs ground truth
        # This measures if unsupervised clustering recovers the perturbation structure
        try:
            kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
            predicted_clusters = kmeans.fit_predict(embeddings)
            
            # Adjusted Rand Index - measures agreement between clustering and ground truth
            ari = adjusted_rand_score(labels, predicted_clusters)
            metrics['adjusted_rand_index'] = ari
            logger.info(f"Adjusted Rand Index (KMeans vs ground truth): {ari:.4f} (higher is better, range: -1 to 1)")
            
            # Normalized Mutual Information
            nmi = normalized_mutual_info_score(labels, predicted_clusters)
            metrics['normalized_mutual_info'] = nmi
            logger.info(f"Normalized Mutual Information: {nmi:.4f} (higher is better, range: 0 to 1)")
        except Exception as e:
            logger.warning(f"Failed to compute clustering metrics: {e}")
            metrics['adjusted_rand_index'] = None
            metrics['normalized_mutual_info'] = None
        
        # 7. Control separation - how well is control separated from perturbations?
        if 'control' in unique_labels:
            try:
                control_mask = labels == 'control'
                perturbation_mask = labels != 'control'
                
                if np.any(control_mask) and np.any(perturbation_mask):
                    control_embeddings = embeddings[control_mask]
                    perturbation_embeddings = embeddings[perturbation_mask]
                    
                    # Distance from control to perturbations
                    control_to_pert_dist = euclidean_distances(control_embeddings, perturbation_embeddings)
                    mean_control_separation = np.mean(control_to_pert_dist)
                    
                    # Within-control distance
                    if len(control_embeddings) > 1:
                        control_intra_dist = euclidean_distances(control_embeddings)
                        mask = np.triu(np.ones_like(control_intra_dist, dtype=bool), k=1)
                        mean_control_intra = np.mean(control_intra_dist[mask])
                    else:
                        mean_control_intra = 0
                    
                    metrics['mean_control_to_perturbation_distance'] = mean_control_separation
                    metrics['mean_control_intra_distance'] = mean_control_intra
                    metrics['control_separation_ratio'] = mean_control_separation / (mean_control_intra + 1e-8) if mean_control_intra > 0 else mean_control_separation
                    
                    logger.info(f"Control separation - mean distance to perturbations: {mean_control_separation:.4f}")
                    logger.info(f"Control separation ratio: {metrics['control_separation_ratio']:.4f}")
            except Exception as e:
                logger.warning(f"Failed to compute control separation metrics: {e}")
        
        metrics['num_classes'] = n_classes
        metrics['num_samples'] = len(embeddings)
        
        return metrics

    def run_baseline_inference(self, adata: anndata.AnnData,
                             batch_size: int = 1,
                             output_dir: str = "results/baseline") -> Dict:
        """
        Run baseline inference on perturbation data with performance profiling.

        Args:
            adata: AnnData object with perturbation data
            batch_size: Batch size for processing (default: 1)
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
                logger.info(f"Starting DDP baseline inference (rank={rank}, world_size={world_size}, batch_size={batch_size})")
        else:
            # DataParallel: scale batch size (splits automatically)
            effective_batch_size = batch_size * self.num_gpus if self.num_gpus > 1 else batch_size
            if is_main_process() or not self.use_ddp:
                logger.info(f"Starting baseline inference with profiling (batch_size={batch_size}, num_gpus={self.num_gpus}, effective_batch_size={effective_batch_size})")
        
        # Start profiling on all ranks for DDP synchronization, but only log on main process
        if self.use_ddp:
            self.profiler.start_profiling()
            if is_main_process():
                logger.info("Started performance profiling")
        elif is_main_process() or not self.use_ddp:
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

        # Run inference with periodic profiling updates
        if is_main_process() or not self.use_ddp:
            logger.info("Running inference...")
        all_embeddings = []

        # Determine batch size for iteration
        if self.use_ddp:
            effective_batch_size = batch_size  # DDP: each process uses base batch size
        else:
            effective_batch_size = batch_size * self.num_gpus if self.num_gpus > 1 else batch_size

        # Use effective batch size for processing
        for i in range(0, len(dataset), effective_batch_size):
            batch_end = min(i + effective_batch_size, len(dataset))
            batch_dataset = dataset.select(range(i, batch_end)) if hasattr(dataset, 'select') else dataset[i:batch_end]

            # Update profiling metrics (only on main process or if not using DDP)
            if is_main_process() or not self.use_ddp:
                self.profiler.update_metrics()

            # Run inference on batch
            batch_embeddings = self.model.get_embeddings(batch_dataset)
            all_embeddings.append(batch_embeddings)

            if is_main_process() or not self.use_ddp:
                logger.info(f"Processed batch {i//effective_batch_size + 1}/{(len(dataset)-1)//effective_batch_size + 1}")

        # Gather embeddings from all DDP processes
        if self.use_ddp and get_world_size() > 1:
            # Collect embeddings from all ranks
            import torch.distributed as dist
            all_embeddings_list = [None] * get_world_size()
            dist.all_gather_object(all_embeddings_list, all_embeddings)
            
            # Flatten and combine embeddings from all processes
            if is_main_process():
                all_embeddings = []
                for rank_embeddings in all_embeddings_list:
                    all_embeddings.extend(rank_embeddings)
            else:
                # Non-main processes don't need full embeddings
                all_embeddings = []

        # Combine all embeddings (only on main process for DDP)
        if self.use_ddp:
            if is_main_process():
                embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
            else:
                embeddings = np.array([])  # Non-main processes don't need embeddings
        else:
            embeddings = np.vstack(all_embeddings)

        # Stop profiling on all ranks for DDP synchronization (barrier requires all ranks)
        # Only collect detailed metrics on main process to avoid unnecessary work
        if self.use_ddp:
            performance_metrics = self.profiler.stop_profiling(collect_metrics=is_main_process())
        elif is_main_process() or not self.use_ddp:
            performance_metrics = self.profiler.stop_profiling()
        else:
            performance_metrics = {'total_time_seconds': 0}  # Placeholder for non-main processes

        # Evaluate embeddings using zero-shot classification
        # Get perturbation labels from adata
        if 'perturbation' in adata.obs:
            perturbation_labels = adata.obs['perturbation'].values
        elif 'perturbation_type' in adata.obs:
            perturbation_labels = adata.obs['perturbation_type'].values
        else:
            logger.warning("No perturbation labels found in adata.obs. Skipping embedding evaluation.")
            perturbation_labels = None
        
        evaluation_metrics = None
        geometry_metrics = None
        if perturbation_labels is not None:
            try:
                # Classification-based evaluation (zero-shot)
                evaluation_metrics = self.evaluate_embeddings(embeddings, perturbation_labels)
            except Exception as e:
                logger.warning(f"Failed to evaluate embeddings: {e}. Continuing without evaluation metrics.")
            
            try:
                # Geometry-based evaluation (clustering and separation)
                geometry_metrics = self.evaluate_embedding_geometry(embeddings, perturbation_labels)
            except Exception as e:
                logger.warning(f"Failed to evaluate embedding geometry: {e}. Continuing without geometry metrics.")

        # Save results
        results = {
            'embeddings': embeddings,
            'performance_metrics': performance_metrics,
            'evaluation_metrics': evaluation_metrics,
            'geometry_metrics': geometry_metrics,
            'model_name': self.model_name,
            'device': self.device,
            'num_perturbations': len(adata),
            'num_genes': adata.shape[1]
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
            'perturbation_types': adata.obs['perturbation'].tolist() if 'perturbation' in adata.obs else (
                adata.obs['perturbation_type'].tolist() if 'perturbation_type' in adata.obs else None
            )
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

        # Log to wandb
        if self.use_wandb and self.wandb_run is not None:
            # Log performance metrics
            perf_log = {
                'performance/total_time_seconds': performance_metrics['total_time_seconds'],
                'performance/throughput_samples_per_sec': len(adata) / performance_metrics['total_time_seconds'],
                'performance/avg_cpu_percent': performance_metrics['avg_cpu_percent'],
                'performance/max_cpu_percent': performance_metrics['max_cpu_percent'],
                'performance/avg_memory_percent': performance_metrics['avg_memory_percent'],
                'performance/max_memory_percent': performance_metrics['max_memory_percent'],
            }
            
            if 'avg_gpu_percent' in performance_metrics:
                perf_log.update({
                    'performance/avg_gpu_percent': performance_metrics['avg_gpu_percent'],
                    'performance/max_gpu_percent': performance_metrics['max_gpu_percent'],
                    'performance/avg_gpu_memory_mb': performance_metrics['avg_gpu_memory_mb'],
                    'performance/max_gpu_memory_mb': performance_metrics['max_gpu_memory_mb'],
                    'performance/avg_gpu_memory_gb': performance_metrics['avg_gpu_memory_mb'] / 1024,
                })
            
            # Log evaluation metrics
            if evaluation_metrics:
                eval_log = {
                    'evaluation/zero_shot_accuracy': evaluation_metrics['zero_shot_accuracy'],
                    'evaluation/zero_shot_f1_score': evaluation_metrics['zero_shot_f1_score'],
                    'evaluation/train_size': evaluation_metrics['train_size'],
                    'evaluation/test_size': evaluation_metrics['test_size'],
                    'evaluation/num_classes': evaluation_metrics['num_classes'],
                }
                perf_log.update(eval_log)
            
            # Log geometry metrics
            if geometry_metrics:
                geom_log = {}
                if geometry_metrics.get('silhouette_score') is not None:
                    geom_log['geometry/silhouette_score'] = geometry_metrics['silhouette_score']
                if geometry_metrics.get('davies_bouldin_index') is not None:
                    geom_log['geometry/davies_bouldin_index'] = geometry_metrics['davies_bouldin_index']
                if geometry_metrics.get('calinski_harabasz_score') is not None:
                    geom_log['geometry/calinski_harabasz_score'] = geometry_metrics['calinski_harabasz_score']
                if geometry_metrics.get('separation_ratio') is not None:
                    geom_log['geometry/separation_ratio'] = geometry_metrics['separation_ratio']
                if geometry_metrics.get('mean_intra_cluster_distance') is not None:
                    geom_log['geometry/mean_intra_cluster_distance'] = geometry_metrics['mean_intra_cluster_distance']
                if geometry_metrics.get('mean_inter_cluster_distance') is not None:
                    geom_log['geometry/mean_inter_cluster_distance'] = geometry_metrics['mean_inter_cluster_distance']
                if geometry_metrics.get('knn_label_consistency') is not None:
                    geom_log['geometry/knn_label_consistency'] = geometry_metrics['knn_label_consistency']
                if geometry_metrics.get('adjusted_rand_index') is not None:
                    geom_log['geometry/adjusted_rand_index'] = geometry_metrics['adjusted_rand_index']
                if geometry_metrics.get('normalized_mutual_info') is not None:
                    geom_log['geometry/normalized_mutual_info'] = geometry_metrics['normalized_mutual_info']
                if geometry_metrics.get('control_separation_ratio') is not None:
                    geom_log['geometry/control_separation_ratio'] = geometry_metrics['control_separation_ratio']
                
                perf_log.update(geom_log)
            
            # Log dataset info
            perf_log.update({
                'dataset/num_perturbations': len(adata),
                'dataset/num_genes': adata.shape[1],
                'dataset/embedding_dim': embeddings.shape[1],
            })
            
            # Log all metrics at once
            self.wandb_run.log(perf_log)
            
            # Log embeddings as artifact (for later analysis)
            try:
                # Create a summary of embeddings (first 1000 samples for visualization)
                n_samples_to_log = min(1000, len(embeddings))
                embedding_sample = embeddings[:n_samples_to_log]
                
                # Log as wandb table for visualization
                if perturbation_labels is not None:
                    labels_sample = perturbation_labels[:n_samples_to_log]
                    # Create table with embeddings and labels
                    embedding_table_data = []
                    for i in range(n_samples_to_log):
                        row = {'sample_id': i, 'perturbation': labels_sample[i]}
                        # Add first 10 dimensions for visualization
                        for dim in range(min(10, embeddings.shape[1])):
                            row[f'embedding_dim_{dim}'] = embedding_sample[i, dim]
                        embedding_table_data.append(row)
                    
                    embedding_table = wandb.Table(dataframe=pd.DataFrame(embedding_table_data))
                    self.wandb_run.log({'embeddings/sample_table': embedding_table})
                
                # Log full embeddings as artifact
                embeddings_file = os.path.join(output_dir, 'embeddings.npz')
                if os.path.exists(embeddings_file):
                    artifact = wandb.Artifact(f'embeddings_{self.model_name}', type='embeddings')
                    artifact.add_file(embeddings_file)
                    self.wandb_run.log_artifact(artifact)
            except Exception as e:
                logger.warning(f"Failed to log embeddings to wandb: {e}")
            
            # Log output directory as artifact
            try:
                artifact = wandb.Artifact(f'results_{self.model_name}', type='results')
                artifact.add_dir(output_dir)
                self.wandb_run.log_artifact(artifact)
            except Exception as e:
                logger.warning(f"Failed to log results directory to wandb: {e}")
            
            logger.info("Metrics logged to wandb")

        logger.info(f"Baseline inference completed. Results saved to {output_dir}")
        logger.info(f"Embeddings shape: {embeddings.shape}")
        if evaluation_metrics:
            logger.info(f"Zero-shot Accuracy: {evaluation_metrics['zero_shot_accuracy']:.4f}")
            logger.info(f"Zero-shot F1 Score: {evaluation_metrics['zero_shot_f1_score']:.4f}")
        if geometry_metrics:
            if geometry_metrics.get('silhouette_score') is not None:
                logger.info(f"Silhouette Score: {geometry_metrics['silhouette_score']:.4f}")
            if geometry_metrics.get('separation_ratio') is not None:
                logger.info(f"Separation Ratio: {geometry_metrics['separation_ratio']:.4f}")

        return results

    @staticmethod
    def save_results_summary(results_list: List[Dict], output_file: str):
        """Save a summary of multiple runs for comparison."""
        summary_data = []
        for result in results_list:
            metrics = result['performance_metrics']
            eval_metrics = result.get('evaluation_metrics', {})
            summary_data.append({
                'method': result.get('method', 'baseline'),
                'model_name': result['model_name'],
                'device': result['device'],
                'num_perturbations': result['num_perturbations'],
                'total_time_seconds': metrics['total_time_seconds'],
                'avg_cpu_percent': metrics['avg_cpu_percent'],
                'max_cpu_percent': metrics['max_cpu_percent'],
                'avg_memory_percent': metrics['avg_memory_percent'],
                'max_memory_percent': metrics['max_memory_percent'],
                'avg_gpu_percent': metrics.get('avg_gpu_percent', None),
                'max_gpu_percent': metrics.get('max_gpu_percent', None),
                'avg_gpu_memory_mb': metrics.get('avg_gpu_memory_mb', None),
                'max_gpu_memory_mb': metrics.get('max_gpu_memory_mb', None),
                'throughput_samples_per_sec': result['num_perturbations'] / metrics['total_time_seconds'],
                'zero_shot_accuracy': eval_metrics.get('zero_shot_accuracy', None),
                'zero_shot_f1_score': eval_metrics.get('zero_shot_f1_score', None),
                'num_classes': eval_metrics.get('num_classes', None),
                'silhouette_score': result.get('geometry_metrics', {}).get('silhouette_score', None),
                'separation_ratio': result.get('geometry_metrics', {}).get('separation_ratio', None),
                'knn_label_consistency': result.get('geometry_metrics', {}).get('knn_label_consistency', None),
                'adjusted_rand_index': result.get('geometry_metrics', {}).get('adjusted_rand_index', None)
            })

        df = pd.DataFrame(summary_data)
        df.to_csv(output_file, index=False)
        logger.info(f"Results summary saved to {output_file}")


def main():
    """Main function to run baseline ISP inference."""
    import argparse

    parser = argparse.ArgumentParser(description='Run baseline ISP inference with optional wandb tracking')
    parser.add_argument('--model', type=str, default="gf-6L-10M-i2048", help='Geneformer model name')
    parser.add_argument('--num_perturbations', type=int, default=100, help='Number of perturbations to process')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu), auto-detected if not specified')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases tracking')
    parser.add_argument('--wandb_project', type=str, default='ispo-baseline', help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name (optional)')
    
    args = parser.parse_args()

    # Configuration
    MODEL_NAME = args.model
    NUM_PERTURBATIONS = args.num_perturbations
    DEVICE = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    USE_WANDB = args.use_wandb

    logger.info("Starting In-Silico Perturbation Optimization Challenge - Baseline")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Number of perturbations: {NUM_PERTURBATIONS}")
    logger.info(f"Wandb tracking: {USE_WANDB}")

    # Initialize optimizer
    optimizer = GeneformerISPOptimizer(
        model_name=MODEL_NAME,
        device=DEVICE,
        use_wandb=USE_WANDB,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name
    )

    # Load model
    optimizer.load_model()

    # Load perturbation data
    perturbation_data = optimizer.load_perturbation_data(
        num_perturbations=NUM_PERTURBATIONS
    )

    # Run baseline inference
    results = optimizer.run_baseline_inference(
        perturbation_data,
        output_dir="results/baseline"
    )

    # Save summary
    optimizer.save_results_summary([results], "results/baseline_summary.csv")

    # Finish wandb run
    if USE_WANDB and optimizer.wandb_run is not None:
        optimizer.wandb_run.finish()
        logger.info("Wandb run completed")

    logger.info("Baseline run completed successfully!")


if __name__ == "__main__":
    main()
