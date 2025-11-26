# Exploration Notebooks Guide

This directory contains step-by-step notebooks for exploring In-Silico Perturbation optimization.

## Notebook Sequence

### 1. `01_load_model_and_data.ipynb`
**Purpose**: Load model and data, explore structure

**What it does**:
- Loads Geneformer model (`gf-6L-10M-i2048`)
- Downloads and loads SciPlex2 perturbation data
- Explores data structure and perturbation types
- Visualizes perturbation distribution
- Creates subset of 100 samples for testing

**Output**: Model and data ready for inference

---

### 2. `02_get_embeddings_baseline.ipynb`
**Purpose**: Generate baseline embeddings

**What it does**:
- Processes data for Geneformer
- Generates embeddings using baseline settings (batch_size=10)
- Visualizes embeddings using PCA
- Saves baseline embeddings for comparison

**Output**: 
- `results/baseline_exploration/embeddings.npz`
- `results/baseline_exploration/metadata.json`

---

### 3. `03_optimized_inference.ipynb`
**Purpose**: Test optimization methods

**What it does**:
- Runs baseline inference (for comparison)
- Tests batching optimization (batch_size=32)
- Tests mixed precision (FP16) optimization
- Compares performance metrics
- Saves optimized embeddings

**Output**:
- `results/optimized_exploration/embeddings_baseline.npz`
- `results/optimized_exploration/embeddings_batching.npz`
- `results/optimized_exploration/embeddings_fp16.npz`

---

### 4. `04_evaluate_embeddings.ipynb`
**Purpose**: Evaluate embedding consistency and quality

**What it does**:
- Loads baseline and optimized embeddings
- Calculates geometry-focused metrics:
  - Cosine similarity (angle preservation)
  - Distance matrix correlation (relative geometry)
  - Nearest neighbor preservation (ranking consistency)
- Evaluates perturbation separation:
  - Classification metrics (zero-shot accuracy/F1)
  - Clustering quality (silhouette score, ARI, NMI)
  - Separation ratios (inter/intra-cluster distances)
  - k-NN label consistency
- Visualizes consistency metrics
- Generates evaluation report

**Output**:
- `results/evaluation_report/evaluation_summary.csv`
- `results/evaluation_report/geometry_metrics.csv`
- Visualization plots

---

## Quick Start

1. **Run notebooks in sequence**:
   ```bash
   jupyter notebook 01_load_model_and_data.ipynb
   # Then 02, 03, 04 in order
   ```

2. **Or use the simple evaluation script**:
   ```bash
   # Baseline
   python simple_evaluation.py --method baseline --num_samples 100
   
   # Batching optimization
   python simple_evaluation.py --method batching --batch_size 32 --num_samples 100
   
   # Mixed precision
   python simple_evaluation.py --method mixed_precision --batch_size 32 --num_samples 100
   ```

## Evaluation Metrics Explained

### Why Geometry Matters

Embeddings are used for:
- **Similarity search**: Find similar cells/perturbations
- **Clustering**: Group related samples
- **Downstream tasks**: Classification, regression

**Key insight**: What matters is **relative positions** in embedding space, not absolute values.

### Metrics Used

#### Consistency Metrics (Baseline vs Optimized)

1. **Cosine Similarity** (Primary)
   - Measures angle between vectors (0-1, where 1 = identical direction)
   - Invariant to magnitude
   - **Threshold**: >0.99 for acceptable consistency

2. **Distance Matrix Correlation** (Most Important)
   - Compares pairwise distance matrices
   - Checks if relative distances between samples are preserved
   - **Threshold**: >0.95 correlation

3. **Nearest Neighbor Preservation**
   - Checks if k nearest neighbors remain the same
   - Measures ranking consistency
   - **Threshold**: >0.8 preservation

#### Embedding Quality Metrics (Perturbation Separation)

4. **Classification Metrics** (Zero-Shot)
   - **Accuracy**: RidgeClassifier accuracy on perturbation prediction
   - **F1 Score**: Weighted F1 score for multi-class classification
   - **Good values**: >0.65 accuracy (baseline models achieve 0.66-0.73)

5. **Clustering Quality**
   - **Silhouette Score**: Cluster separation quality (-1 to 1, higher is better)
   - **Adjusted Rand Index (ARI)**: Agreement with ground truth labels
   - **Davies-Bouldin Index**: Cluster similarity (lower is better)
   - **Good values**: Silhouette > 0.3, ARI > 0.5

6. **Separation Metrics**
   - **Separation Ratio**: Inter-cluster / intra-cluster distance
   - **k-NN Label Consistency**: How many neighbors share same label
   - **Control Separation**: Distance from control to perturbations
   - **Good values**: Separation ratio > 2.0, k-NN consistency > 0.7

### Acceptability Criteria

**For Optimization Consistency** (baseline vs optimized):
- ✅ Cosine similarity > 0.99
- ✅ Distance correlation > 0.95
- ✅ NN preservation > 0.8

**For Embedding Quality** (perturbation separation):
- ✅ Zero-shot accuracy > 0.65
- ✅ Silhouette score > 0.3
- ✅ Separation ratio > 2.0

This ensures that optimizations maintain scientific validity while improving performance, and that embeddings capture meaningful perturbation-specific structure.

## Dependencies

```bash
pip install helical anndata numpy pandas matplotlib seaborn scikit-learn scipy
```

## Notes

- Notebooks are designed to run independently (each loads what it needs)
- Results are saved to `results/` directory
- Can be run on Google Colab (use smaller batch sizes for Colab GPUs)
- First run will download model weights and data (may take time)




