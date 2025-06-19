# N2V2 CPU Call Optimization Analysis

## Function: `n2v2_denoise_torch`

**Location**: `openhcs/processing/backends/enhance/n2v2_processor_torch.py:434`

## Critical CPU Synchronization Issues Found

### 🚨 ISSUE 1: Random Patch Extraction (CRITICAL)

**Location**: Lines 380-382 in `extract_random_patches_2d()`
```python
z_idx = torch.randint(0, z, (1,), device=device).item()  # ❌ CPU SYNC!
y_start = torch.randint(0, y - patch_size + 1, (1,), device=device).item()  # ❌ CPU SYNC!
x_start = torch.randint(0, x - patch_size + 1, (1,), device=device).item()  # ❌ CPU SYNC!
```

**Impact Analysis**:
- Called in training loop: `num_batches * max_epochs` times
- Default: 100 batches × 10 epochs = 1,000 calls
- Each call has 3 CPU syncs = **3,000 CPU-GPU synchronizations**
- Each sync can take 10-100μs = **30-300ms total overhead**

**Root Cause**: Using `.item()` to extract scalar values forces GPU→CPU transfer

### 🚨 ISSUE 2: Nested Loop Masking (CRITICAL)

**Location**: Lines 406-429 in `apply_n2v2_masking()`
```python
for b in range(batch_size):  # ❌ CPU loop
    mask_indices = torch.where(mask[b])  # GPU operation
    for i, j in zip(mask_indices[0], mask_indices[1]):  # ❌ CPU iteration over GPU tensors
        # Neighborhood extraction and median calculation
        neighborhood = patches[b, y_min:y_max, x_min:x_max].flatten()
        masked_patches[b, i, j] = torch.median(neighborhood)  # ❌ Individual GPU calls
```

**Impact Analysis**:
- Called every training batch: `num_batches * max_epochs` times
- Processes each masked pixel individually
- With 5% masking on 64×64 patches: ~200 pixels per patch
- Batch size 4: 800 individual median calculations per call
- Total: 800 × 1,000 = **800,000 individual GPU operations**

**Root Cause**: CPU loops over GPU tensor indices, no vectorization

### 🔶 ISSUE 3: Loss Accumulation (MODERATE)

**Location**: Line 548
```python
epoch_loss += masked_loss.item()  # ❌ CPU SYNC per batch
```

**Impact Analysis**:
- Called every batch: `num_batches * max_epochs` times
- Default: 100 × 10 = **1,000 CPU syncs**
- Used only for logging, not algorithm-critical

### 🔶 ISSUE 4: Shape Queries (MODERATE)

**Location**: Lines 371, 562
```python
z, y, x = image.shape  # ❌ Potential CPU sync
```

**Impact Analysis**:
- PyTorch may sync for shape queries on some versions
- Called multiple times throughout function
- Low individual impact but cumulative effect

## Performance Impact Estimation

### Current Performance Profile
```
Total CPU-GPU Syncs: ~4,000+ per training run
Estimated Overhead: 40-400ms per training run
GPU Utilization: Severely degraded due to constant CPU waits
Memory Bandwidth: Wasted on small scalar transfers
```

### Optimization Potential
```
Vectorized Operations: 10-100x speedup for masking
Batch Random Generation: Eliminate 3,000 syncs
Accumulated Logging: Reduce to 1 sync per epoch
Total Speedup Estimate: 5-20x for training phase
```

## Optimization Strategies

### Strategy 1: Vectorized Random Patch Extraction
**Replace**: Individual `.item()` calls
**With**: Batch tensor operations
```python
# Generate all random indices at once
indices = torch.randint(0, max_val, (num_patches, 3), device=device)
z_indices = indices[:, 0] % z
y_indices = indices[:, 1] % (y - patch_size + 1)
x_indices = indices[:, 2] % (x - patch_size + 1)
```

### Strategy 2: GPU-Native Masking
**Replace**: Nested CPU loops
**With**: Vectorized GPU operations using unfold/conv operations
```python
# Use unfold to extract all neighborhoods at once
neighborhoods = F.unfold(patches, kernel_size=3, padding=1)
# Vectorized median calculation
medians = torch.median(neighborhoods, dim=1)[0]
```

### Strategy 3: Deferred Loss Logging
**Replace**: Per-batch `.item()` calls
**With**: Accumulated tensor logging
```python
# Accumulate losses as tensors
epoch_losses.append(masked_loss.detach())
# Single sync at epoch end
if verbose: logger.info(f"Loss: {torch.stack(epoch_losses).mean().item()}")
```

### Strategy 4: Cached Shape Information
**Replace**: Repeated shape queries
**With**: Cached values
```python
# Cache at function start
z_size, y_size, x_size = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
```

## Implementation Priority

1. **HIGH**: Vectorized random patch extraction (eliminates 3,000 syncs)
2. **HIGH**: GPU-native masking (eliminates nested loops)
3. **MEDIUM**: Deferred loss logging (eliminates 1,000 syncs)
4. **LOW**: Cached shape queries (minor optimization)

## Expected Results

- **Training Speed**: 5-20x faster due to eliminated CPU waits
- **GPU Utilization**: Near 100% during compute phases
- **Memory Efficiency**: Reduced CPU-GPU transfer overhead
- **Scalability**: Better performance with larger batch sizes
