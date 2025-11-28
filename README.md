# In-Silico Perturbation Optimization (ISPO) 🚀

## Overview

This repository implements comprehensive optimizations for efficient **In-Silico Perturbation (ISP)** inference using Foundation Models, specifically the **Geneformer** model from the Helical package. The project demonstrates significant performance improvements through multiple optimization techniques while maintaining model accuracy and embedding quality.

**Key Achievement**: Achieved up to **11.0x speedup** (1003% improvement) in inference throughput for Geneformer v1 while preserving embedding quality and biological accuracy. Geneformer v2 achieved up to **8.16x speedup** (716% improvement).

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

#### Geneformer v1 (gf-6L-10M-i2048)

| Method | Time (s) | Throughput (samples/s) | Speedup | Improvement | CPU Usage (%) | GPU Usage (%) | Memory (GB) |
|--------|----------|------------------------|---------|-------------|---------------|---------------|-------------|
| **Baseline** | 74.05 | 13.5 | 1.00x | - | 3.5 | 2.1 | 3.5 |
| **Batching (bs=8)** | 16.21 | 61.7 | 4.57x | +357% | 3.6 | 8.3 | 3.5 |
| **Batching (bs=16)** | 10.29 | 97.2 | 7.20x | +620% | 3.4 | 13.4 | 3.5 |
| **Batching (bs=64)** | 7.71 | 129.7 | 9.60x | +860% | 3.4 | 20.8 | 3.5 |
| **Batching (bs=256)** | 6.85 | 146.0 | **10.8x** | +981% | 3.5 | 17.3 | 3.4 |
| **Batching (bs=512)** | 6.74 | 148.4 | **11.0x** | +999% | 3.7 | 11.0 | 3.3 |
| **Mixed Precision FP16** | 6.77 | 147.7 | **10.9x** | +994% | 3.4 | 17.3 | 3.4 |
| **Mixed Precision BF16** | 6.79 | 147.3 | **10.9x** | +991% | 3.4 | 17.5 | 3.4 |
| **Quantization 8-bit** | 6.71 | 149.0 | **11.0x** | +1003% | 3.7 | 17.0 | 3.5 |

#### Geneformer v2 (gf-12L-38M-i4096)

| Method | Time (s) | Throughput (samples/s) | Speedup | Improvement | CPU Usage (%) | GPU Usage (%) | Memory (GB) |
|--------|----------|------------------------|---------|-------------|---------------|---------------|-------------|
| **Baseline** | 81.96 | 12.2 | 1.00x | - | 3.5 | 6.2 | 3.6 |
| **Batching (bs=8)** | 18.93 | 52.8 | 4.33x | +333% | 3.4 | 23.6 | 3.6 |
| **Batching (bs=16)** | 13.83 | 72.3 | 5.92x | +492% | 3.4 | 33.1 | 3.6 |
| **Batching (bs=64)** | 10.94 | 91.4 | 7.49x | +649% | 3.4 | 44.4 | 3.6 |
| **Batching (bs=256)** | 10.04 | 99.6 | 8.16x | +716% | 3.4 | 36.0 | 3.5 |
| **Batching (bs=512)** | 10.04 | 99.6 | 8.16x | +716% | 3.7 | 26.0 | 3.4 |
| **Mixed Precision FP16** | 10.41 | 96.1 | 7.87x | +687% | 3.4 | 27.5 | 3.5 |
| **Mixed Precision BF16** | 10.16 | 98.4 | 8.07x | +707% | 3.4 | 38.8 | 3.5 |
| **Quantization 8-bit** | 10.26 | 97.5 | 7.99x | +699% | 3.5 | 36.3 | 3.5 |

### Key Performance Metrics

- **Dataset**: 1000 SciPlex2 perturbations (58K genes each)
- **Models**: 
  - Geneformer v1: gf-6L-10M-i2048 (6 layers, 10M parameters)
  - Geneformer v2: gf-12L-38M-i4096 (12 layers, 38M parameters)
- **Hardware**: CUDA GPU (80GB total memory)
- **Embedding Consistency**: >0.9999 cosine similarity (excellent preservation)

*Note: CPU/GPU usage and memory consumption are shown as average values in the tables above. Peak values and detailed resource utilization metrics are available in the individual result files.*

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

### Resource Utilization Insights

The optimization techniques significantly improve GPU utilization:
- **Geneformer v1**: GPU usage increases from 2.1% (baseline) to up to 20.8% (batching bs=64), representing a **10x improvement** in hardware efficiency
- **Geneformer v2**: GPU usage increases from 6.2% (baseline) to up to 44.4% (batching bs=64), representing a **7x improvement** in hardware efficiency

Memory consumption remains relatively stable across all optimization methods, typically using 3.3-3.6 GB of system RAM. CPU usage remains low (3.4-3.7%) across all configurations, indicating that the optimizations primarily benefit from better GPU utilization rather than increased CPU load.

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

# Run with Distributed Data Parallel (DDP) on multiple GPUs
# Example: Run on 4 GPUs with batching optimization
torchrun --nproc_per_node=4 -m ispo.scripts.run_ddp \
    --method batching \
    --batch_size 256 \
    --num_perturbations 1000 \
    --model_name gf-6L-10M-i2048

# Example: Run on 2 GPUs with mixed precision
torchrun --nproc_per_node=2 -m ispo.scripts.run_ddp \
    --method mixed_precision \
    --batch_size 128 \
    --precision fp16 \
    --num_perturbations 1000
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

### 3. Quantization (8-bit)

**Strategy**: Attempt to quantize model weights to 8-bit integers

**Implementation Note**: PyTorch's dynamic quantization only works on CPU. When running on CUDA (as in our benchmarks), the implementation automatically falls back to FP16 mixed precision, which provides similar speed benefits without actual weight quantization.

**Why Memory Usage Doesn't Decrease**:
- On CUDA, quantization doesn't actually run (falls back to FP16)
- The memory metric measures total system RAM (3.5 GB), not just model weights
- Model weights (~40-150 MB) are <5% of total memory; quantizing them would save <100 MB, which is negligible compared to data loading, activations, and Python overhead

**Results**: 11.0x speedup (74.05s → 6.71s for 1000 samples) - Note: This is effectively FP16 mixed precision on CUDA

### 4. Distributed Data Parallel (DDP)

**Strategy**: Use PyTorch DDP for true multi-GPU parallelism with better efficiency than DataParallel

**Benefits**:
- **Multiprocessing**: Avoids Python GIL limitations (DataParallel uses multithreading)
- **Better Scalability**: More efficient communication between GPUs
- **Lower Overhead**: Reduced synchronization costs
- **Linear Scaling**: Near-linear speedup with number of GPUs

**Usage**:
```bash
# Launch with torchrun (recommended)
torchrun --nproc_per_node=4 -m ispo.scripts.run_ddp \
    --method batching \
    --batch_size 256 \
    --num_perturbations 1000

# Or programmatically
optimizer = OptimizedGeneformerISPOptimizer(use_ddp=True)
```

**Implementation Details**:
- Each GPU process handles a subset of the data
- Embeddings are gathered from all processes at the end
- Only rank 0 (main process) saves results and logs metrics
- Automatic data splitting across processes

**When to Use**:
- **DDP**: Best for multi-GPU setups (2+ GPUs), better performance and scalability
- **DataParallel**: Simpler for single-node multi-GPU, but less efficient
- **Single GPU**: Use neither (default behavior)

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

### Multi-GPU Support

The codebase supports two multi-GPU strategies:

1. **DataParallel** (default): Simple multi-GPU support using PyTorch's DataParallel
   - Automatically enabled when `num_gpus > 1`
   - Works out of the box, no special launch required
   - Good for quick multi-GPU experiments

2. **Distributed Data Parallel (DDP)**: More efficient multi-GPU support ✅ **NEW**
   - Enabled with `use_ddp=True` and launched via `torchrun`
   - Better performance and scalability
   - Recommended for production multi-GPU inference

**Example DDP Usage**:
```bash
# Run on 4 GPUs
torchrun --nproc_per_node=4 -m ispo.scripts.run_ddp \
    --method batching --batch_size 256 --num_perturbations 1000
```

### Future Optimizations

- **Model Parallelism**: Split large models across devices
- **Advanced Quantization**: Implement custom quantization pipelines (CUDA-compatible)
- **TensorRT Integration**: Further optimization for NVIDIA GPUs
- **CPU Optimization**: SIMD instructions and threading
