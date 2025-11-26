#!/usr/bin/env python3
"""
Embedding Evaluation Module

This module provides geometry-focused metrics for evaluating embedding consistency
between baseline and optimized inference methods.

Key insight: Geometry matters more than absolute values. We evaluate:
1. Cosine similarity (angle preservation)
2. Distance matrix correlation (relative geometry preservation)
3. Nearest neighbor preservation (ranking consistency)
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")


class EmbeddingEvaluator:
    """Evaluates embedding consistency using geometry-focused metrics."""
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def cosine_similarity_elementwise(self, emb1: np.ndarray, emb2: np.ndarray) -> Dict:
        """
        Calculate element-wise cosine similarity between embeddings.
        
        Args:
            emb1: Baseline embeddings (n_samples, embedding_dim)
            emb2: Optimized embeddings (n_samples, embedding_dim)
            
        Returns:
            Dictionary with cosine similarity metrics
        """
        if emb1.shape != emb2.shape:
            raise ValueError(f"Shape mismatch: {emb1.shape} vs {emb2.shape}")
        
        # Normalize vectors
        emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
        
        # Element-wise cosine similarity
        cosine_sims = np.sum(emb1_norm * emb2_norm, axis=1)
        
        return {
            'cosine_similarities': cosine_sims,
            'mean_cosine_similarity': np.mean(cosine_sims),
            'std_cosine_similarity': np.std(cosine_sims),
            'min_cosine_similarity': np.min(cosine_sims),
            'max_cosine_similarity': np.max(cosine_sims),
            'median_cosine_similarity': np.median(cosine_sims)
        }
    
    def euclidean_distance_metrics(self, emb1: np.ndarray, emb2: np.ndarray) -> Dict:
        """
        Calculate Euclidean distance metrics between embeddings.
        
        Args:
            emb1: Baseline embeddings
            emb2: Optimized embeddings
            
        Returns:
            Dictionary with Euclidean distance metrics
        """
        if emb1.shape != emb2.shape:
            raise ValueError(f"Shape mismatch: {emb1.shape} vs {emb2.shape}")
        
        # Element-wise Euclidean distances
        euclidean_dists = np.linalg.norm(emb1 - emb2, axis=1)
        
        # Relative change (normalized by baseline magnitude)
        baseline_magnitudes = np.linalg.norm(emb1, axis=1)
        relative_changes = euclidean_dists / (baseline_magnitudes + 1e-8)
        
        return {
            'euclidean_distances': euclidean_dists,
            'mean_euclidean_distance': np.mean(euclidean_dists),
            'mean_relative_change': np.mean(relative_changes),
            'max_relative_change': np.max(relative_changes)
        }
    
    def distance_matrix_correlation(self, emb1: np.ndarray, emb2: np.ndarray) -> Dict:
        """
        Compare distance matrices to evaluate relative geometry preservation.
        
        This is the most important metric - it checks if relative distances
        between samples are preserved.
        
        Args:
            emb1: Baseline embeddings
            emb2: Optimized embeddings
            
        Returns:
            Dictionary with distance matrix correlation metrics
        """
        # Compute pairwise distance matrices
        dist_matrix_1 = euclidean_distances(emb1)
        dist_matrix_2 = euclidean_distances(emb2)
        
        # Flatten matrices (excluding diagonal)
        n = len(emb1)
        mask = ~np.eye(n, dtype=bool)
        dists_1 = dist_matrix_1[mask]
        dists_2 = dist_matrix_2[mask]
        
        # Pearson correlation
        pearson_corr = np.corrcoef(dists_1, dists_2)[0, 1]
        
        # Spearman correlation (rank-based)
        spearman_corr, _ = spearmanr(dists_1, dists_2)
        
        return {
            'distance_matrix_pearson': pearson_corr,
            'distance_matrix_spearman': spearman_corr,
            'baseline_distance_mean': np.mean(dists_1),
            'optimized_distance_mean': np.mean(dists_2),
            'distance_ratio': np.mean(dists_2) / (np.mean(dists_1) + 1e-8)
        }
    
    def nearest_neighbor_preservation(self, emb1: np.ndarray, emb2: np.ndarray, 
                                     k: int = 5) -> Dict:
        """
        Evaluate nearest neighbor preservation.
        
        Checks if the k nearest neighbors of each sample remain the same
        between baseline and optimized embeddings.
        
        Args:
            emb1: Baseline embeddings
            emb2: Optimized embeddings
            k: Number of nearest neighbors to check
            
        Returns:
            Dictionary with NN preservation metrics
        """
        n = len(emb1)
        
        # Compute distance matrices
        dist_matrix_1 = euclidean_distances(emb1)
        dist_matrix_2 = euclidean_distances(emb2)
        
        # Set diagonal to large value (exclude self)
        np.fill_diagonal(dist_matrix_1, np.inf)
        np.fill_diagonal(dist_matrix_2, np.inf)
        
        # Get k nearest neighbors for each sample
        nn_preserved = []
        nn_rank_correlations = []
        
        for i in range(n):
            # Get k nearest in baseline
            nn_baseline = np.argsort(dist_matrix_1[i])[:k]
            
            # Get k nearest in optimized
            nn_optimized = np.argsort(dist_matrix_2[i])[:k]
            
            # Count how many are preserved
            preserved = len(set(nn_baseline) & set(nn_optimized))
            nn_preserved.append(preserved / k)
            
            # Rank correlation of all neighbors
            all_indices = list(set(nn_baseline) | set(nn_optimized))
            if len(all_indices) > 1:
                ranks_baseline = [np.where(np.argsort(dist_matrix_1[i]) == idx)[0][0] 
                                for idx in all_indices]
                ranks_optimized = [np.where(np.argsort(dist_matrix_2[i]) == idx)[0][0] 
                                 for idx in all_indices]
                corr, _ = spearmanr(ranks_baseline, ranks_optimized)
                if not np.isnan(corr):
                    nn_rank_correlations.append(corr)
        
        return {
            'mean_nn_preservation': np.mean(nn_preserved),
            'std_nn_preservation': np.std(nn_preserved),
            'min_nn_preservation': np.min(nn_preserved),
            'mean_rank_correlation': np.mean(nn_rank_correlations) if nn_rank_correlations else 0.0
        }
    
    def evaluate_embeddings(self, baseline_emb: np.ndarray, 
                          optimized_emb: np.ndarray,
                          k_neighbors: int = 5) -> Dict:
        """
        Comprehensive evaluation of embedding consistency.
        
        Args:
            baseline_emb: Baseline embeddings
            optimized_emb: Optimized embeddings
            k_neighbors: Number of neighbors for NN preservation
            
        Returns:
            Dictionary with all evaluation metrics
        """
        results = {}
        
        # 1. Cosine similarity (primary metric)
        cosine_results = self.cosine_similarity_elementwise(baseline_emb, optimized_emb)
        results.update(cosine_results)
        
        # 2. Euclidean distance metrics
        euclidean_results = self.euclidean_distance_metrics(baseline_emb, optimized_emb)
        results.update(euclidean_results)
        
        # 3. Distance matrix correlation (most important for geometry)
        dist_matrix_results = self.distance_matrix_correlation(baseline_emb, optimized_emb)
        results.update(dist_matrix_results)
        
        # 4. Nearest neighbor preservation
        nn_results = self.nearest_neighbor_preservation(baseline_emb, optimized_emb, k_neighbors)
        results.update(nn_results)
        
        # 5. Overall acceptability
        results['acceptable'] = (
            results['mean_cosine_similarity'] > 0.99 and
            results['distance_matrix_pearson'] > 0.95 and
            results['mean_nn_preservation'] > 0.8
        )
        
        return results
    
    def print_evaluation_report(self, evaluation_results: Dict, method_name: str = "Optimized"):
        """
        Print a formatted evaluation report.
        
        Args:
            evaluation_results: Results from evaluate_embeddings()
            method_name: Name of the optimization method
        """
        print(f"\n{'='*60}")
        print(f"EMBEDDING EVALUATION REPORT: {method_name}")
        print(f"{'='*60}\n")
        
        print("1. COSINE SIMILARITY (Primary Metric)")
        print(f"   Mean:     {evaluation_results['mean_cosine_similarity']:.6f}")
        print(f"   Std:      {evaluation_results['std_cosine_similarity']:.6f}")
        print(f"   Min:      {evaluation_results['min_cosine_similarity']:.6f}")
        print(f"   Max:      {evaluation_results['max_cosine_similarity']:.6f}")
        print(f"   Median:   {evaluation_results['median_cosine_similarity']:.6f}")
        
        print("\n2. EUCLIDEAN DISTANCE")
        print(f"   Mean distance:        {evaluation_results['mean_euclidean_distance']:.6f}")
        print(f"   Mean relative change: {evaluation_results['mean_relative_change']:.6f}")
        print(f"   Max relative change:  {evaluation_results['max_relative_change']:.6f}")
        
        print("\n3. DISTANCE MATRIX CORRELATION (Geometry Preservation)")
        print(f"   Pearson correlation:  {evaluation_results['distance_matrix_pearson']:.6f}")
        print(f"   Spearman correlation: {evaluation_results['distance_matrix_spearman']:.6f}")
        print(f"   Distance ratio:      {evaluation_results['distance_ratio']:.6f}")
        
        print("\n4. NEAREST NEIGHBOR PRESERVATION")
        print(f"   Mean preservation:    {evaluation_results['mean_nn_preservation']:.6f}")
        print(f"   Rank correlation:     {evaluation_results['mean_rank_correlation']:.6f}")
        
        print(f"\n5. OVERALL ASSESSMENT")
        acceptable = evaluation_results['acceptable']
        status = "ACCEPTABLE" if acceptable else "❌ NOT ACCEPTABLE"
        print(f"   Status: {status}")
        
        if acceptable:
            print("\n   ✓ Embeddings preserve geometry")
            print("   ✓ Optimizations maintain scientific validity")
        else:
            print("\n   ⚠ Embeddings may have significant changes")
            print("   ⚠ Review optimization method")
        
        print(f"\n{'='*60}\n")


def evaluate_embeddings_simple(baseline_emb: np.ndarray, 
                               optimized_emb: np.ndarray) -> Dict:
    """
    Simple wrapper function for quick evaluation.
    
    Args:
        baseline_emb: Baseline embeddings
        optimized_emb: Optimized embeddings
        
    Returns:
        Dictionary with key metrics
    """
    evaluator = EmbeddingEvaluator()
    return evaluator.evaluate_embeddings(baseline_emb, optimized_emb)

