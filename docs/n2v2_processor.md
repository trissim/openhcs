# N2V2 Denoising Processor

This document describes the N2V2-inspired denoising processor implementation in ezstitcher.

## Overview

The N2V2 processor is a 3D GPU-native, planner-compatible, self-supervised denoising step based on the N2V2 paper (Höck et al., 2022) using PyTorch. It improves upon the original Noise2Void algorithm with several key enhancements:

1. **BlurPool instead of MaxPool**: Replaces MaxPool layers with BlurPool to remove checkerboard artifacts
2. **No top-level skip connections**: Removes the top-level skip connection in the U-Net architecture
3. **Improved blind-spot sampling**: Uses a more effective blind-spot masking strategy
4. **Self-supervised learning**: Trains on the input image itself without requiring clean targets

The implementation is fully compatible with the OpenHCS doctrinal principles:
- **Clause 3 — Declarative Primacy**: All functions are pure and stateless
- **Clause 65 — Fail Loudly**: No silent fallbacks or inferred capabilities
- **Clause 88 — No Inferred Capabilities**: Explicit PyTorch dependency
- **Clause 273 — Memory Backend Restrictions**: GPU-only implementation

## Usage

### Basic Usage

```python
import torch
from ezstitcher.core.processing.backends.n2v2_processor_torch import n2v2_denoise_torch

# Load your 3D image as a torch.Tensor of shape (Z, Y, X)
image = torch.randn(32, 128, 128, dtype=torch.uint16, device="cuda")

# Apply N2V2 denoising
denoised = n2v2_denoise_torch(
    image,
    max_epochs=10,
    batch_size=4,
    patch_size=64,
    learning_rate=1e-4,
    verbose=True
)
```

### Using a Pre-trained Model

```python
# Denoise using a pre-trained model
denoised = n2v2_denoise_torch(
    image,
    model_path="path/to/model.pt",
    device="cuda",
    verbose=True
)
```

### Training and Saving a Model

```python
# Train and save a model
denoised = n2v2_denoise_torch(
    image,
    max_epochs=20,
    batch_size=4,
    patch_size=64,
    learning_rate=1e-4,
    save_model_path="path/to/save/model.pt",
    verbose=True
)
```

## Parameters

The `n2v2_denoise_torch` function accepts the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | torch.Tensor | (required) | Input 3D tensor of shape (Z, Y, X) |
| `model_path` | Optional[str] | None | Path to a pre-trained model |
| `random_seed` | int | 42 | Random seed for reproducibility |
| `device` | str | "cuda" | Device to run the model on ("cuda" or "cpu") |
| `blindspot_prob` | float | 0.05 | Probability of masking a pixel for blind-spot training |
| `max_epochs` | int | 10 | Maximum number of training epochs |
| `batch_size` | int | 4 | Batch size for training |
| `patch_size` | int | 64 | Size of the patches for training |
| `learning_rate` | float | 1e-4 | Learning rate for the optimizer |
| `save_model_path` | Optional[str] | None | Path to save the trained model |
| `verbose` | bool | False | Whether to print progress information |

## Architecture

The N2V2 processor uses a 3D U-Net architecture with the following modifications:

1. **BlurPool3d**: A custom layer that applies a Gaussian blur before downsampling to prevent checkerboard artifacts
2. **No top-level skip connection**: The final upsampling block does not use a skip connection
3. **Blind-spot training**: Pixels are randomly masked and the network is trained to predict them from their context

### BlurPool3d

The BlurPool3d layer replaces traditional MaxPool operations with a blur-then-subsample approach:

1. Applies a 3D Gaussian blur kernel to the input
2. Downsamples the blurred input using strided convolution
3. This prevents aliasing and reduces checkerboard artifacts in the output

### Blind-spot Training

The blind-spot training approach works as follows:

1. Randomly mask a small percentage of pixels in the input image
2. Train the network to predict the masked pixels from their context
3. Compute loss only on the masked pixels
4. This allows the network to learn to denoise without requiring clean target images

## Performance Considerations

- The processor is designed to run entirely on GPU for maximum performance
- For large images, the processor automatically processes the image in patches to avoid memory issues
- Training time depends on the image size, number of epochs, and GPU capabilities
- Inference is much faster than training, especially with a pre-trained model

## Example

See the example script at `examples/n2v2_denoising_example.py` for a complete demonstration of the N2V2 processor, including:

- Creating synthetic data with noise
- Applying N2V2 denoising
- Evaluating the results using PSNR and SSIM
- Visualizing and saving the results

## References

1. Höck, A., Hahn, J., Beier, T., Jug, F. (2022). "N2V2 - Fixing Noise2Void Checkerboard Artifacts with Modified Sampling Strategies and a Tweaked Network Architecture"
2. Krull, A., Buchholz, T. O., & Jug, F. (2019). "Noise2Void - Learning Denoising from Single Noisy Images"
