# ONNX and TensorRT Optimization Guide

## How ONNX/TensorRT Improve Performance

### ONNX (Open Neural Network Exchange)

**What it does:**
- Converts models to a standardized format
- Applies graph-level optimizations
- Enables optimized inference engines

**Performance Benefits:**
- **1.5-3x speedup** over standard PyTorch inference
- **Graph optimizations**: Operator fusion, constant folding, dead code elimination
- **Memory optimization**: Better memory layout and reuse
- **Cross-platform**: Works on CPU, GPU, and specialized hardware

**Key Optimizations:**
1. **Operator Fusion**: Combines multiple operations into single kernels
   - Example: Conv + BatchNorm + ReLU → Single fused operation
   - Reduces memory transfers and kernel launch overhead

2. **Constant Folding**: Pre-computes constant operations at export time
   - Reduces runtime computation

3. **Graph Simplification**: Removes unnecessary operations
   - Dead code elimination
   - Identity operation removal

4. **Layout Optimization**: Optimizes tensor memory layout for better cache usage

### TensorRT (NVIDIA's Inference Optimizer)

**What it does:**
- NVIDIA-specific deep learning inference optimizer
- Converts models to optimized TensorRT engines
- Hardware-specific kernel optimization

**Performance Benefits:**
- **2-10x speedup** over PyTorch (typically 3-5x for transformers)
- **Kernel fusion**: Combines operations into optimized CUDA kernels
- **Layer fusion**: Merges adjacent layers to reduce memory access
- **Precision optimization**: FP16 and INT8 quantization
- **Dynamic shape optimization**: Optimizes for different input sizes

**Key Optimizations:**

1. **Kernel Fusion**
   - Combines multiple operations into single CUDA kernels
   - Reduces GPU memory bandwidth usage
   - Example: Attention + LayerNorm → Single optimized kernel

2. **Layer Fusion**
   - Merges adjacent layers (e.g., Linear + Activation)
   - Reduces intermediate memory allocations
   - Fewer kernel launches

3. **Precision Optimization**
   - **FP16**: 2x faster, 2x less memory (minimal accuracy loss)
   - **INT8**: 4x faster, 4x less memory (requires calibration, some accuracy loss)
   - Automatic precision selection based on layer sensitivity

4. **Dynamic Shape Optimization**
   - Optimizes for different batch sizes and sequence lengths
   - Creates multiple optimized kernels for different shapes
   - Balances memory usage and performance

5. **Tensor Core Utilization**
   - Maximizes use of Tensor Cores (V100, A100, etc.)
   - Optimized matrix multiplication kernels

## Performance Comparison

### Typical Speedups (Inference)

| Method | Speedup | Use Case |
|--------|---------|----------|
| PyTorch (baseline) | 1.0x | Development, flexibility |
| ONNX Runtime | 1.5-3x | Cross-platform, easy deployment |
| TensorRT FP32 | 2-4x | NVIDIA GPUs, maximum accuracy |
| TensorRT FP16 | 3-6x | NVIDIA GPUs, good accuracy |
| TensorRT INT8 | 5-10x | NVIDIA GPUs, production (with calibration) |

### Memory Usage

| Method | Memory | Notes |
|--------|--------|-------|
| PyTorch | 100% | Baseline |
| ONNX Runtime | 80-90% | Optimized memory layout |
| TensorRT FP32 | 70-85% | Fused operations reduce intermediates |
| TensorRT FP16 | 35-50% | Half precision |
| TensorRT INT8 | 20-30% | Quarter precision |

## Implementation in Codebase

### 1. Export to ONNX

```python
from optimized_isp import OptimizedGeneformerISPOptimizer

optimizer = OptimizedGeneformerISPOptimizer(model_name="gf-6L-10M-i2048")
optimizer.load_model()

# Export to ONNX
onnx_path = optimizer.export_to_onnx(
    output_path="geneformer.onnx",
    sample_input_shape=(1, 2048),  # (batch_size, sequence_length)
    opset_version=13
)
```

**What happens:**
- Model is traced/exported to ONNX format
- Graph optimizations are applied
- Model saved as `.onnx` file

### 2. Run ONNX Inference

```python
# Run inference with ONNX Runtime
results = optimizer.run_onnx_inference(
    adata,
    onnx_model_path="geneformer.onnx",
    batch_size=32,
    output_dir="results/onnx"
)
```

**Benefits:**
- Faster inference (1.5-3x)
- Lower memory usage
- Cross-platform compatibility

### 3. Convert to TensorRT

```python
# Convert ONNX to TensorRT engine
tensorrt_path = optimizer.convert_to_tensorrt(
    onnx_model_path="geneformer.onnx",
    tensorrt_engine_path="geneformer.trt",
    precision="fp16",  # or "fp32", "int8"
    max_batch_size=64,
    max_sequence_length=4096
)
```

**What happens:**
- ONNX model is parsed
- TensorRT applies kernel fusion and layer fusion
- Precision optimization (FP16/INT8) if specified
- Engine is built and optimized for your specific GPU
- Saved as `.trt` file (GPU-specific)

**Note:** TensorRT engine building can take 5-30 minutes depending on model size.

### 4. Run TensorRT Inference

```python
# Run inference with TensorRT
results = optimizer.run_tensorrt_inference(
    adata,
    tensorrt_engine_path="geneformer.trt",
    batch_size=64,  # Can use larger batches
    output_dir="results/tensorrt"
)
```

**Benefits:**
- Fastest inference (2-10x speedup)
- Lowest latency
- Highest throughput
- Optimized for your specific GPU

## Complete Workflow

```python
from optimized_isp import OptimizedGeneformerISPOptimizer
import anndata

# 1. Initialize and load model
optimizer = OptimizedGeneformerISPOptimizer(model_name="gf-6L-10M-i2048")
optimizer.load_model()

# 2. Load data
adata = optimizer.load_perturbation_data(num_perturbations=200)

# 3. Export to ONNX (one-time, can reuse)
onnx_path = optimizer.export_to_onnx("geneformer.onnx")

# 4. Option A: Use ONNX Runtime
onnx_results = optimizer.run_onnx_inference(
    adata, onnx_path, batch_size=32
)

# 5. Option B: Convert to TensorRT (one-time, GPU-specific)
tensorrt_path = optimizer.convert_to_tensorrt(
    onnx_path, "geneformer.trt", precision="fp16"
)

# 6. Use TensorRT (fastest)
tensorrt_results = optimizer.run_tensorrt_inference(
    adata, tensorrt_path, batch_size=64
)

# 7. Compare results
print(f"ONNX speedup: {baseline_time / onnx_time:.2f}x")
print(f"TensorRT speedup: {baseline_time / tensorrt_time:.2f}x")
```

## Installation

### ONNX and ONNX Runtime

```bash
# CPU version
pip install onnx onnxruntime

# GPU version (recommended)
pip install onnx onnxruntime-gpu
```

### TensorRT

TensorRT installation is more complex:

1. **Download TensorRT** from NVIDIA Developer site
2. **Install Python package**:
   ```bash
   pip install nvidia-tensorrt
   ```
3. **Install PyCUDA** (required for TensorRT):
   ```bash
   pip install pycuda
   ```

**Note:** TensorRT version must match your CUDA version.

## When to Use Each

### Use ONNX Runtime when:
- ✅ You need cross-platform deployment (CPU, GPU, mobile)
- ✅ You want easy optimization without GPU-specific setup
- ✅ You're deploying to cloud services
- ✅ You need 1.5-3x speedup (good enough)

### Use TensorRT when:
- ✅ You're deploying on NVIDIA GPUs only
- ✅ You need maximum performance (2-10x speedup)
- ✅ You can invest time in engine building
- ✅ You're doing production inference at scale
- ✅ You want to use INT8 quantization for maximum speed

## Limitations and Considerations

### ONNX Limitations:
- Some PyTorch operations may not be supported
- Dynamic control flow can be challenging
- May require model modifications for export

### TensorRT Limitations:
- NVIDIA GPUs only
- Engine is GPU-specific (not portable)
- Building engine takes time (but only once)
- INT8 requires calibration dataset
- Some operations may not be supported

### For Geneformer Specifically:
- Transformer models work well with both ONNX and TensorRT
- Attention mechanisms are well-optimized
- Dynamic sequence lengths may require optimization profiles
- Batch processing benefits significantly from optimization

## Expected Performance Gains

For Geneformer inference on typical hardware:

| Configuration | Baseline | ONNX | TensorRT FP16 | TensorRT INT8 |
|---------------|----------|------|---------------|---------------|
| Throughput (samples/s) | 100 | 150-300 | 300-600 | 500-1000 |
| Latency (ms/batch) | 100 | 50-67 | 25-50 | 15-30 |
| Memory (GB) | 4.0 | 3.2-3.6 | 1.6-2.0 | 0.8-1.2 |

**Note:** Actual performance depends on:
- GPU model (Tensor Cores help significantly)
- Batch size (larger batches = better optimization)
- Model size (larger models benefit more)
- Sequence length (longer sequences = more optimization opportunity)

## Best Practices

1. **Profile First**: Always measure baseline performance
2. **Start with ONNX**: Easier to set up, good speedup
3. **Move to TensorRT**: For production, maximum performance
4. **Use FP16 First**: Good balance of speed and accuracy
5. **Consider INT8**: For production at scale (with calibration)
6. **Cache Engines**: Build TensorRT engines once, reuse them
7. **Batch Optimization**: Larger batches = better GPU utilization
8. **Monitor Accuracy**: Verify optimized models maintain accuracy

## Troubleshooting

### ONNX Export Fails
- Check for unsupported operations
- Try different opset versions
- Simplify model if needed

### TensorRT Build Fails
- Verify CUDA version compatibility
- Check GPU compute capability
- Reduce max batch size if memory issues

### Performance Not as Expected
- Ensure using GPU version of ONNX Runtime
- Check batch size (should be > 1)
- Verify TensorRT engine was built for your GPU
- Profile to identify bottlenecks

## References

- [ONNX Documentation](https://onnx.ai/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/best-practices/)

