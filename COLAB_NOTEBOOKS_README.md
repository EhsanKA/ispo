# Google Colab Notebooks Guide

This directory contains Jupyter notebooks designed to run on Google Colab. The notebooks import functions from the existing Python modules, keeping the code modular and maintainable.

## 📓 Available Notebooks

### 1. `00_colab_setup.ipynb` - Setup and Installation
**Purpose**: Set up the Colab environment

**What it does**:
- Installs all required dependencies (helical, wandb, etc.)
- Provides options to upload files or clone from GitHub
- Verifies installation and GPU availability
- Checks that all required Python modules can be imported

**Run this first** before running any other notebooks.

### 2. `01_colab_baseline_analysis.ipynb` - Main Analysis Notebook
**Purpose**: Run baseline ISP inference with full evaluation

**What it does**:
- Loads Geneformer model
- Downloads and processes SciPlex2 perturbation data
- Generates embeddings
- Evaluates embedding quality (classification + geometry metrics)
- Creates visualizations (PCA, UMAP, performance charts)
- Saves all results
- Optional wandb tracking

**Output**:
- `results/baseline/embeddings.npz`
- `results/baseline/evaluation_metrics.csv`
- `results/baseline/geometry_metrics.csv`
- `results/baseline/performance_metrics.csv`
- Visualizations displayed in notebook

### 3. `02_colab_comparison.ipynb` - Compare Multiple Runs
**Purpose**: Compare different model configurations

**What it does**:
- Runs multiple configurations (different models, parameters)
- Compares performance, accuracy, and geometry metrics
- Creates comparison visualizations
- Generates summary CSV

**Use this** to compare different Geneformer models or configurations.

## 🚀 Quick Start

### Step 1: Upload Files to Colab

**Option A: Upload directly**
1. Open `00_colab_setup.ipynb` in Colab
2. Upload these files using the file browser:
   - `baseline_isp.py`
   - `embedding_evaluator.py`
   - Any other required files

**Option B: Clone from GitHub**
1. If your repository is public, clone it in the setup notebook
2. Or use `git clone` in a code cell

### Step 2: Run Setup Notebook

1. Open `00_colab_setup.ipynb`
2. Run all cells
3. Verify that all imports work and GPU is available

### Step 3: Run Analysis

1. Open `01_colab_baseline_analysis.ipynb`
2. Adjust configuration in the second cell:
   ```python
   CONFIG = {
       'model_name': 'gf-6L-10M-i2048',  # Start with smaller model
       'num_perturbations': 100,  # Adjust for Colab memory
       'use_wandb': False,  # Set True for tracking
   }
   ```
3. Run all cells sequentially

### Step 4: View Results

- Metrics are displayed in the notebook
- Visualizations are shown inline
- Results are saved to `results/` directory
- Download results using the last cell

## ⚙️ Configuration

### Model Selection

For Colab, start with smaller models:
- `gf-6L-10M-i2048` - Smallest, fastest (recommended for Colab)
- `gf-12L-38M-i4096` - Medium (may work on Colab Pro)
- `gf-18L-316M-i4096` - Large (may require Colab Pro+ or local GPU)

### Memory Considerations

- **Free Colab**: Use `num_perturbations=50-100`
- **Colab Pro**: Can handle `num_perturbations=200-500`
- **Colab Pro+**: Can handle larger datasets

### Wandb Tracking

To enable wandb tracking:
1. Install wandb: `pip install wandb`
2. Login: `wandb login` (run in a cell)
3. Set `use_wandb=True` in configuration

## 📊 What Gets Tracked

### Performance Metrics
- Runtime (seconds)
- Throughput (samples/second)
- CPU/GPU utilization
- Memory usage (RAM and GPU)

### Evaluation Metrics
- Zero-shot accuracy
- Zero-shot F1 score
- Train/test set sizes

### Geometry Metrics
- Silhouette score
- Separation ratio
- k-NN label consistency
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Davies-Bouldin Index
- Calinski-Harabasz Score

## 🔧 Troubleshooting

### Import Errors
- Make sure you've uploaded `baseline_isp.py` and `embedding_evaluator.py`
- Run the setup notebook first
- Check that all dependencies are installed

### Out of Memory
- Reduce `num_perturbations` in configuration
- Use a smaller model (`gf-6L-10M-i2048`)
- Restart runtime and try again

### GPU Not Available
- Runtime → Change runtime type → Hardware accelerator → GPU
- Make sure you're using a GPU runtime (T4, V100, or A100)

### Slow Performance
- Use smaller batch sizes (already optimized in code)
- Reduce number of perturbations
- Use smaller model

## 📥 Downloading Results

Each notebook includes cells to download results:
- Individual results: Use the download cell in `01_colab_baseline_analysis.ipynb`
- Comparison results: Use the download cell in `02_colab_comparison.ipynb`

Results are zipped and downloaded automatically.

## 🔗 Integration with Local Code

The notebooks import directly from your Python files:
```python
from baseline_isp import GeneformerISPOptimizer
```

This means:
- ✅ Code stays modular
- ✅ Changes to Python files automatically reflected
- ✅ No code duplication
- ✅ Easy to maintain

## 📝 Notes

- **First Run**: Model weights and data will be downloaded (may take time)
- **Subsequent Runs**: Uses cached data (much faster)
- **Colab Sessions**: Results are lost when session ends (download important files!)
- **File Persistence**: Upload files each session or use Google Drive mounting

## 🎯 Next Steps

After running the baseline analysis:
1. Try different model sizes
2. Compare configurations using `02_colab_comparison.ipynb`
3. Enable wandb for better tracking
4. Experiment with different numbers of perturbations
5. Analyze the geometry metrics to understand embedding quality

---

**Happy Analyzing! 🧬**

