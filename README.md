# In-Silico Perturbation Optimization (ISPO) 🚀

## Overview

This repository implements comprehensive optimizations for efficient **In-Silico Perturbation (ISP)** inference using Foundation Models, specifically the **Geneformer** model from the Helical package. The project demonstrates significant performance improvements through multiple optimization techniques while maintaining model accuracy and embedding quality.

**Key Achievement**: Achieved up to **10.9x speedup** in inference throughput while preserving embedding quality and biological accuracy.

## 🎯 Challenge Objectives

- **Baseline Establishment**: Create a comprehensive profiling baseline for Geneformer ISP inference
- **Optimization Implementation**: Apply multiple optimization techniques (batching, mixed precision)
- **Performance Benchmarking**: Measure improvements in speed, memory usage, and scalability
- **Result Validation**: Ensure optimizations preserve model accuracy and output consistency

## 🧬 Technical Background

In-Silico Perturbations involve computational modeling of gene expression changes to predict cellular responses. This technique is crucial for:

- **Drug Discovery**: Rapid screening of perturbation effects on cellular states
- **Disease Modeling**: Large-scale genetic perturbation studies
- **Therapeutic Development**: Efficient target identification and validation
- **Biological Research**: Scalable hypothesis testing and mechanism understanding

As foundation models grow larger and perturbation sets increase in scale, inference time and computational cost become significant bottlenecks. This project addresses these challenges through systematic optimization.

## 🏗️ Project Structure

```
ispo/
├── ispo/                          # Main package
│   ├── core/                      # Core functionality
│   │   ├── baseline.py           # Baseline ISP optimizer
│   │   ├── optimized.py          # Optimized ISP methods
│   │   └── profiler.py           # Performance profiler
│   ├── evaluation/                # Evaluation modules
│   │   └── evaluator.py          # Embedding quality evaluator
│   ├── optimization/              # Optimization methods
│   │   └── bayesian.py           # Bayesian hyperparameter optimization
│   └── scripts/                   # Executable scripts
│       ├── run_baseline.py
│       ├── run_optimized.py
│       └── run_bayesian.py
├── config/                        # Configuration files
├── examples/                      # Usage examples
├── results/                       # Benchmark results
└── main.py                        # Main entry point
```

## 📊 Results Summary

### Performance Improvements

Based on comprehensive benchmarking with 1000 SciPlex2 perturbations:

| Method | Time (s) | Throughput (samples/s) | Speedup | Improvement |
|--------|----------|------------------------|---------|-------------|
| **Baseline** | 74.05 | 13.5 | 1.00x | - |
| **Batching (bs=256)** | 6.85 | 146.0 | **10.8x** | +981% |
| **Mixed Precision FP16** | 6.77 | 147.7 | **10.9x** | +994% |

### Key Performance Metrics

- **Dataset**: 1000 SciPlex2 perturbations (58K genes each)
- **Model**: Geneformer gf-6L-10M-i2048
- **Hardware**: CUDA GPU (80GB total memory)
- **Embedding Consistency**: >0.9999 cosine similarity (excellent preservation)

### Evaluation Results

#### Classification Metrics
- **Zero-Shot Accuracy**: 0.615 (61.5%)
- **Zero-Shot F1 Score**: 0.602
- **Classes**: 5 perturbation types (Dex, Nutlin, SAHA, BMS, Control)
- **Train/Test Split**: 800/200 samples

#### Geometry Metrics
- **Silhouette Score**: -0.039 (indicates mixed clustering structure)
- **Separation Ratio**: 1.07 (inter/intra-cluster distance ratio)
- **k-NN Label Consistency**: 0.445 (44.5% of neighbors share same label)
- **Adjusted Rand Index**: 0.067 (clustering agreement with ground truth)
- **Control Separation Ratio**: 1.33 (control vs perturbation separation)

### Resource Utilization

**Baseline Performance**:
- CPU Usage: 3.5% average, 10% peak
- Memory Usage: 3.7% average (3.5 GB)
- GPU Usage: 2.1% average, 6% peak
- GPU Memory: 844 MB average, 845 MB peak

**Optimized Performance (Batching bs=256)**:
- CPU Usage: 3.4% average, 3.8% peak
- Memory Usage: 3.6% average (3.4 GB)
- GPU Usage: 17.3% average, 23% peak (better utilization!)
- GPU Memory: 756 MB average, 847 MB peak

The optimizations significantly improve GPU utilization (from 2% to 17%), indicating better hardware efficiency.

## 🚀 Quick Start

### Prerequisites

```bash
# Create conda environment
conda env create -f environment.yml
conda activate helical-ispo

# Or install manually
pip install -r requirements.txt

# Optional: Install wandb for experiment tracking
pip install wandb
wandb login
```

### Basic Usage

```bash
# Run baseline inference
python -m ispo baseline --num_perturbations 1000

# Run optimized inference with batching
python -m ispo optimized --method batching --batch_size 256 --num_perturbations 1000

# Run optimized inference with mixed precision
python -m ispo optimized --method mixed_precision --num_perturbations 1000

# Run with Weights & Biases tracking
python -m ispo baseline --use_wandb --wandb_project ispo-baseline --num_perturbations 1000
```

### Programmatic Usage

```python
from ispo.core.baseline import GeneformerISPOptimizer
from ispo.core.optimized import OptimizedGeneformerISPOptimizer

# Initialize baseline optimizer
optimizer = GeneformerISPOptimizer(
    model_name="gf-6L-10M-i2048",
    device="cuda"
)

# Load model and data
optimizer.load_model()
data = optimizer.load_perturbation_data(num_perturbations=1000)

# Run baseline inference
results = optimizer.run_baseline_inference(
    data, 
    output_dir="results/baseline"
)

# Initialize optimized optimizer
optimized = OptimizedGeneformerISPOptimizer(
    model_name="gf-6L-10M-i2048",
    device="cuda"
)

# Run optimized inference with batching
optimized.load_model()
results = optimized.run_batching_inference(
    data,
    batch_size=256,
    output_dir="results/batching_bs256"
)
```

## 🔧 Optimization Techniques

### 1. Optimized Batching

**Strategy**: Increase batch size from default (10) to optimal size (256)

**Implementation**:
```python
# Process data in larger batches
for i in range(0, len(dataset), batch_size):
    batch = dataset.select(range(i, i + batch_size))
    embeddings = model.get_embeddings(batch)
```

**Benefits**:
- Better GPU utilization (2% → 17%)
- Reduced per-sample overhead
- Improved memory bandwidth efficiency

**Results**: 10.8x speedup (74.05s → 6.85s for 1000 samples)

### 2. Mixed Precision (FP16)

**Strategy**: Use automatic mixed precision with `torch.cuda.amp`

**Implementation**:
```python
with torch.cuda.amp.autocast():
    embeddings = model.get_embeddings(batch)
```

**Benefits**:
- Faster computation (FP16 operations)
- Reduced memory bandwidth requirements
- Maintains numerical stability through automatic casting

**Results**: 10.9x speedup (74.05s → 6.77s for 1000 samples)

## 📈 Evaluation Methodology

The project includes comprehensive evaluation to ensure optimization quality:

### Classification-Based Evaluation

- **Zero-Shot Accuracy**: Classification accuracy using RidgeClassifier (80/20 train/test split)
- **Zero-Shot F1 Score**: Weighted F1 score for multi-class perturbation prediction
- **Methodology**: Follows Helical's Geneformer scaling evaluation approach

### Geometry-Based Evaluation

These metrics assess how well-separated different perturbations are in embedding space:

- **Silhouette Score**: Measures cluster separation quality (-1 to 1, higher is better)
- **Separation Ratio**: Inter-cluster vs intra-cluster distance ratio (higher = better separation)
- **k-NN Label Consistency**: How many nearest neighbors share the same perturbation label
- **Adjusted Rand Index (ARI)**: Agreement between unsupervised clustering and ground truth
- **Davies-Bouldin Index**: Cluster similarity measure (lower is better)
- **Calinski-Harabasz Score**: Variance ratio criterion (higher is better)
- **Control Separation**: Specialized metrics for control vs perturbation separation

### Embedding Quality Validation

All optimization methods maintain embedding consistency:
- **Cosine Similarity**: >0.9999 between baseline and optimized embeddings
- **Classification Accuracy**: Preserved across all optimization methods
- **Geometry Metrics**: Consistent clustering structure maintained

## 📊 Weights & Biases Integration

The implementation includes optional Weights & Biases (wandb) integration for comprehensive experiment tracking.

### Tracked Metrics

**Real-time Metrics** (logged during inference):
- CPU utilization (%)
- GPU utilization (%)
- Memory usage (RAM and GPU)
- Memory available

**Performance Metrics** (logged after inference):
- Total runtime (seconds)
- Throughput (samples/second)
- Average and peak CPU/GPU usage
- Average and peak memory usage

**Evaluation Metrics**:
- Zero-shot accuracy and F1 score
- Train/test set sizes
- Number of classes

**Geometry Metrics**:
- Silhouette score
- Separation ratio
- k-NN label consistency
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Davies-Bouldin Index
- Calinski-Harabasz Score
- Control separation metrics

**Artifacts**:
- Full embeddings (as .npz file)
- Results directory (all CSV files and metadata)
- Embedding sample table (first 1000 samples for visualization)

### Usage

```bash
# Enable wandb tracking
python -m ispo baseline --use_wandb --wandb_project my-project --num_perturbations 1000

# Custom run name
python -m ispo baseline --use_wandb --wandb_project my-project --wandb_run_name experiment-001
```

## 🎯 Impact & Applications

### Use Cases Enabled

- **Drug Discovery**: Rapid screening of perturbation effects (10x faster enables 10x more experiments)
- **Disease Modeling**: Large-scale genetic perturbation studies
- **Therapeutic Development**: Efficient target identification
- **Biological Research**: Scalable hypothesis testing

### Performance Gains

- **Time Savings**: 10.9x faster inference enables processing 10x more perturbations in same time
- **Cost Reduction**: Proportional reduction in compute costs (90% reduction)
- **Scalability**: Enable processing of 10x-100x more perturbations
- **Research Acceleration**: Faster iteration cycles for experiments

### Production Deployment

```python
# Example: Scale to 10K perturbations
optimizer = OptimizedGeneformerISPOptimizer()
data = optimizer.load_perturbation_data(num_perturbations=10000)

# Use best performing method (mixed precision)
results = optimizer.run_mixed_precision_inference(
    data,
    batch_size=256,
    output_dir="production_results/"
)
```

## 🔬 Technical Details

### Data Processing Pipeline

1. **Data Loading**: SciPlex2 perturbation dataset
2. **Preprocessing**: Gene symbol to Ensembl ID mapping
3. **Tokenization**: Geneformer-specific tokenization
4. **Batch Processing**: Optimized batch inference
5. **Embedding Extraction**: Cell-level embeddings
6. **Result Validation**: 
   - Classification evaluation (zero-shot accuracy/F1)
   - Geometry evaluation (clustering, separation metrics)
   - Consistency checking

### Profiling Methodology

- **CPU/GPU Monitoring**: Real-time utilization tracking using `psutil` and `GPUtil`
- **Memory Profiling**: Peak and average usage measurement
- **Timing**: High-precision execution time measurement
- **Throughput Calculation**: Samples processed per second

### Model Information

- **Model**: Geneformer gf-6L-10M-i2048
  - 6 layers
  - 10 million parameters
  - 2048 input dimension
- **Dataset**: SciPlex2
  - 5 perturbation types: Dexamethasone, Nutlin, SAHA, BMS, Control
  - ~58,000 genes per sample
  - High-quality perturbation data

## 📈 Scaling Considerations

### Current Limitations

- **Memory Constraints**: GPU memory limits batch size
- **Model Size**: Larger Geneformer models may require different strategies
- **Quantization Support**: Current Helical version lacks quantization

### Future Optimizations

- **Multi-GPU Scaling**: Distributed inference across multiple GPUs
- **Model Parallelism**: Split large models across devices
- **Advanced Quantization**: Implement custom quantization pipelines
- **TensorRT Integration**: Further optimization for NVIDIA GPUs
- **CPU Optimization**: SIMD instructions and threading

## 🤝 Contributing

### Adding New Optimizations

1. Extend `OptimizedGeneformerISPOptimizer` class in `ispo/core/optimized.py`
2. Implement optimization in dedicated method
3. Add to benchmarking suite
4. Update documentation

### Testing New Models

```python
# Test with different Geneformer variants
models_to_test = [
    "gf-6L-10M-i2048",      # Current baseline
    "gf-12L-38M-i4096",     # Larger model
    "gf-12L-104M-i4096",    # Even larger
]
```

## 📚 References

- **Helical Package**: https://github.com/helicalAI/helical
- **Geneformer Paper**: Theodoris et al. (2023) Nature
- **Geneformer Scaling Evaluation**: Helical's Geneformer-Series-Comparison notebook
- **SciPlex2 Dataset**: Srivatsan et al. (2020)
- **In-Silico Perturbations**: https://www.nature.com/articles/s41586-023-06139-9

## 📄 License

This implementation is provided for educational and research purposes as part of the Helical In-Silico Perturbation Optimization Challenge.

## 🙏 Acknowledgments

- **Helical Team**: For the excellent Bio Foundation Model framework
- **Geneformer Authors**: For the groundbreaking work on gene expression modeling
- **SciPlex Consortium**: For providing high-quality perturbation data

---

**Challenge Completed**: ✅ Demonstrated **10.9x performance improvement** (994% throughput increase) while maintaining result accuracy and consistency. All optimization methods preserve embedding quality with >0.9999 cosine similarity to baseline.
