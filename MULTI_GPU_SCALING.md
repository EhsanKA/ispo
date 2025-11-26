# Multi-GPU Scaling Guide

## Current Implementation

The current codebase uses a **single GPU** approach:
- Device selection: `device="cuda"` uses the default GPU (GPU 0)
- Model loading: Model is loaded onto a single device
- Batch processing: Batches are processed sequentially on one GPU

## Multi-GPU Scaling Strategies

To scale to multiple GPUs, you can implement several strategies:

### 1. Data Parallelism (Recommended for Inference)

**Concept**: Split batches across multiple GPUs, each GPU processes a subset of the data.

**Implementation Approach**:

```python
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

class MultiGPUGeneformerISPOptimizer(GeneformerISPOptimizer):
    """Extended optimizer with multi-GPU support."""
    
    def __init__(self, model_name: str = "gf-12L-38M-i4096", 
                 num_gpus: int = None, **kwargs):
        """
        Initialize with multi-GPU support.
        
        Args:
            num_gpus: Number of GPUs to use (None = use all available)
        """
        super().__init__(model_name=model_name, device="cuda", **kwargs)
        self.num_gpus = num_gpus or torch.cuda.device_count()
        
    def load_model(self):
        """Load model and wrap with DataParallel for multi-GPU."""
        logger.info(f"Loading Geneformer model: {self.model_name}")
        config = GeneformerConfig(
            model_name=self.model_name,
            device="cuda",
            batch_size=1
        )
        self.model = Geneformer(config)
        
        # Wrap model with DataParallel if multiple GPUs available
        if self.num_gpus > 1 and torch.cuda.device_count() > 1:
            logger.info(f"Using {self.num_gpus} GPUs with DataParallel")
            # Access the underlying PyTorch model
            if hasattr(self.model, 'model'):
                self.model.model = nn.DataParallel(
                    self.model.model,
                    device_ids=list(range(self.num_gpus))
                )
        
        logger.info("Model loaded successfully")
    
    def run_multi_gpu_inference(self, adata, output_dir="results/multi_gpu"):
        """
        Run inference with data parallelism across multiple GPUs.
        
        Key changes:
        1. Larger effective batch size (batch_size * num_gpus)
        2. Automatic batch splitting across GPUs
        3. Results concatenated automatically
        """
        # Process data
        dataset = self.model.process_data(adata)
        
        # With DataParallel, you can use larger batch sizes
        # Effective batch size = batch_size * num_gpus
        batch_size = 32 * self.num_gpus  # Scale batch size with GPUs
        
        all_embeddings = []
        for i in range(0, len(dataset), batch_size):
            batch_end = min(i + batch_size, len(dataset))
            batch_dataset = dataset.select(range(i, batch_end))
            
            # DataParallel automatically splits this across GPUs
            batch_embeddings = self.model.get_embeddings(batch_dataset)
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        return embeddings
```

**Benefits**:
- Simple to implement (PyTorch DataParallel handles distribution)
- Near-linear speedup with number of GPUs
- No changes needed to model architecture
- Works well for inference workloads

**Limitations**:
- Requires batch size divisible by number of GPUs
- Communication overhead between GPUs
- All GPUs must fit the model in memory

### 2. Distributed Data Parallel (DDP) - Better Performance

**Concept**: More efficient than DataParallel, uses distributed training primitives.

**Implementation Approach**:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp(rank, world_size):
    """Initialize distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """Clean up distributed process group."""
    dist.destroy_process_group()

class DDPGeneformerISPOptimizer(GeneformerISPOptimizer):
    """Optimizer using DistributedDataParallel for better multi-GPU performance."""
    
    def load_model(self, rank=0, world_size=1):
        """Load model with DDP."""
        logger.info(f"Loading Geneformer model: {self.model_name}")
        config = GeneformerConfig(
            model_name=self.model_name,
            device=f"cuda:{rank}",
            batch_size=1
        )
        self.model = Geneformer(config)
        
        if world_size > 1:
            # Move model to specific GPU
            self.model.model = self.model.model.to(f"cuda:{rank}")
            # Wrap with DDP
            self.model.model = DDP(
                self.model.model,
                device_ids=[rank],
                output_device=rank
            )
            logger.info(f"Model on GPU {rank} with DDP")
```

**Benefits**:
- Better performance than DataParallel
- More efficient communication
- Scales better to many GPUs

**Limitations**:
- More complex setup (requires process spawning)
- Need to handle distributed data loading
- More code complexity

### 3. Model Parallelism (For Very Large Models)

**Concept**: Split the model itself across multiple GPUs.

**When to Use**:
- Model is too large for single GPU memory
- Each GPU holds part of the model
- Forward pass requires communication between GPUs

**Implementation Approach**:

```python
class ModelParallelGeneformerISPOptimizer(GeneformerISPOptimizer):
    """Split model across multiple GPUs."""
    
    def load_model(self, num_gpus=2):
        """Split model layers across GPUs."""
        config = GeneformerConfig(
            model_name=self.model_name,
            device="cuda:0",
            batch_size=1
        )
        self.model = Geneformer(config)
        
        # Manually split transformer layers across GPUs
        model = self.model.model
        num_layers = len(model.transformer.h)  # Example: transformer layers
        layers_per_gpu = num_layers // num_gpus
        
        for i, layer in enumerate(model.transformer.h):
            gpu_id = i // layers_per_gpu
            layer = layer.to(f"cuda:{gpu_id}")
        
        logger.info(f"Model split across {num_gpus} GPUs")
```

**Benefits**:
- Enables running models too large for single GPU
- Can combine with data parallelism

**Limitations**:
- Complex implementation
- Communication overhead between layers
- Sequential processing (each layer waits for previous)

### 4. Pipeline Parallelism

**Concept**: Process different batches on different GPUs in a pipeline.

**When to Use**:
- Very large models
- Want to maximize GPU utilization
- Can overlap computation and communication

## Recommended Approach for This Codebase

For **inference workloads** with Geneformer, **Data Parallelism** is recommended:

1. **Simple to implement**: Minimal code changes
2. **Good speedup**: Near-linear scaling for inference
3. **Compatible**: Works with existing Helical Geneformer interface
4. **Flexible**: Easy to switch between single and multi-GPU

### Implementation Steps

1. **Detect available GPUs**:
```python
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    use_multi_gpu = True
else:
    use_multi_gpu = False
```

2. **Wrap model with DataParallel**:
```python
if use_multi_gpu:
    model.model = nn.DataParallel(model.model)
```

3. **Scale batch size**:
```python
# Effective batch size = batch_size * num_gpus
batch_size = 32 * num_gpus
```

4. **Process batches** (no other changes needed):
```python
# DataParallel automatically handles distribution
embeddings = model.get_embeddings(batch_dataset)
```

## Expected Performance

- **2 GPUs**: ~1.8-1.9x speedup (due to communication overhead)
- **4 GPUs**: ~3.5-3.8x speedup
- **8 GPUs**: ~6.5-7.5x speedup

**Bottlenecks**:
- GPU-to-GPU communication
- Data loading and preprocessing
- Synchronization overhead

## Code Modifications Needed

### Minimal Changes (DataParallel)

1. Add GPU detection in `__init__`
2. Wrap model in `load_model()`
3. Scale batch size in `run_baseline_inference()`
4. Update profiling to track all GPUs

### Example Integration

```python
# In baseline_isp.py, modify GeneformerISPOptimizer.__init__:
def __init__(self, model_name: str = "gf-12L-38M-i4096", 
             device: str = "cuda",
             num_gpus: int = None,  # NEW
             **kwargs):
    # ... existing code ...
    self.num_gpus = num_gpus or (torch.cuda.device_count() if device == "cuda" else 1)
    
# In load_model():
if self.num_gpus > 1 and torch.cuda.device_count() > 1:
    if hasattr(self.model, 'model'):
        self.model.model = nn.DataParallel(
            self.model.model,
            device_ids=list(range(self.num_gpus))
        )
```

## Testing Multi-GPU Setup

```python
# Test script
import torch
print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Run with multi-GPU
optimizer = GeneformerISPOptimizer(
    model_name="gf-6L-10M-i2048",
    num_gpus=torch.cuda.device_count()
)
```

## Limitations and Considerations

1. **Memory**: Each GPU needs enough memory for the model
2. **Batch Size**: Must be divisible by number of GPUs
3. **Communication**: Overhead increases with number of GPUs
4. **Helical Compatibility**: May need to access underlying PyTorch model
5. **Colab**: Limited to single GPU (use local/cluster for multi-GPU)

## Future Enhancements

- Automatic GPU selection and load balancing
- Dynamic batch size adjustment based on GPU memory
- Support for heterogeneous GPU setups
- Integration with distributed inference frameworks (TensorRT, ONNX Runtime)
- Pipeline parallelism for very large models

