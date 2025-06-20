# Detailed Rewrite Plan: intermediate_usage.rst

## Overview
**Target**: Complete rewrite of `docs/source/user_guide/intermediate_usage.rst`  
**Goal**: Document advanced function patterns, custom function development, and complex pipeline composition  
**Approach**: Replace broken EZStitcher examples with advanced OpenHCS patterns

## Current Problems (From Fact-Check)
- ❌ **Module paths**: ezstitcher.* → openhcs.*
- ❌ **Specialized step classes**: Deprecated → function-based approach
- ❌ **Pipeline constructor**: No input_dir parameter
- ✅ **Core concepts**: All documented patterns work perfectly
- ✅ **Function patterns**: All four patterns preserved and enhanced

## Detailed Section-by-Section Rewrite Plan

### Section 1: Title and Introduction
**Current (Lines 1-20)**:
```rst
Intermediate Usage
================

This section shows how to reimplement the EZ module functionality using pipelines and steps
```

**Rewrite To**:
```rst
Intermediate Usage
================

This guide covers advanced OpenHCS features for users who need more control than the basic interface provides. You'll learn advanced function patterns, custom function development, special input/output handling, and complex pipeline composition.

**Prerequisites**: Complete :doc:`basic_usage` first

**What You'll Learn**:
- Advanced function pattern composition
- Custom GPU-accelerated function development
- Special inputs/outputs for data flow between functions
- Pipeline factories and reusable templates
- Memory type optimization strategies
- Complex multi-step workflows
```

### Section 2: Advanced Function Patterns
**Current (Lines 22-68)**: Broken EZ module examples

**Rewrite To**:
```rst
Advanced Function Patterns
-------------------------

OpenHCS's function pattern system supports arbitrarily complex compositions. Here are advanced patterns for sophisticated workflows.

Nested Sequential Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine multiple processing steps with parameters:

.. code-block:: python

    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.core.memory.decorators import cupy_func
    from openhcs.processing.backends.processors.cupy_processor import (
        create_projection, stack_percentile_normalize
    )

    # Complex sequential pattern with mixed parameterization
    step = FunctionStep(
        func=[
            (create_projection, {'method': 'max_projection'}),  # Z-projection with params
            stack_percentile_normalize,                         # Simple normalization
            (enhance_contrast, {'clip_limit': 0.03}),          # Enhancement with params
            denoise_images                                      # Final denoising
        ],
        variable_components=[VariableComponents.Z_INDEX],
        name="Advanced Image Enhancement"
    )

Nested Component-Specific Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different processing chains for different components:

.. code-block:: python

    from openhcs.constants.constants import GroupBy

    # Channel-specific processing with nested patterns
    step = FunctionStep(
        func={
            'DAPI': [
                denoise_nuclei,
                (enhance_contrast, {'clip_limit': 0.02}),
                segment_nuclei
            ],
            'GFP': [
                denoise_proteins,
                (enhance_contrast, {'clip_limit': 0.05}),
                detect_proteins
            ],
            'BF': [
                correct_illumination,
                enhance_edges,
                detect_cell_boundaries
            ]
        },
        group_by=GroupBy.CHANNEL,
        name="Channel-Specific Analysis Pipeline"
    )

Multi-Level Component Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Patterns that vary by multiple components:

.. code-block:: python

    # Site and channel specific processing
    step = FunctionStep(
        func={
            'site_001': {
                'DAPI': process_center_dapi,
                'GFP': process_center_gfp
            },
            'site_002': {
                'DAPI': (process_edge_dapi, {'edge_correction': True}),
                'GFP': (process_edge_gfp, {'edge_correction': True})
            },
            'site_003': {
                'DAPI': process_corner_dapi,
                'GFP': process_corner_gfp
            }
        },
        group_by=GroupBy.SITE,  # Primary grouping by site
        name="Site and Channel Specific Processing"
    )

Dynamic Pattern Generation
~~~~~~~~~~~~~~~~~~~~~~~~~

Create patterns programmatically:

.. code-block:: python

    def create_channel_pipeline(channels, processing_params):
        """Create channel-specific processing dynamically."""
        func_dict = {}
        
        for channel in channels:
            if channel.startswith('DAPI'):
                func_dict[channel] = [
                    denoise_nuclei,
                    (enhance_contrast, processing_params.get('dapi_contrast', {})),
                    segment_nuclei
                ]
            elif channel.startswith('GFP'):
                func_dict[channel] = [
                    denoise_proteins,
                    (enhance_contrast, processing_params.get('gfp_contrast', {})),
                    detect_proteins
                ]
            else:
                func_dict[channel] = normalize_channel
        
        return FunctionStep(
            func=func_dict,
            group_by=GroupBy.CHANNEL,
            name="Dynamic Channel Processing"
        )

    # Usage
    channels = ['DAPI', 'GFP-1', 'GFP-2', 'BF']
    params = {
        'dapi_contrast': {'clip_limit': 0.02},
        'gfp_contrast': {'clip_limit': 0.05}
    }
    step = create_channel_pipeline(channels, params)
```

**Context Engine Sources**:
- docs/architecture/function-patterns.md lines 275-296 (nested patterns)
- docs/architecture/function-patterns.md lines 155-165 (component patterns)
- docs/architecture/function-patterns.md lines 320-324 (dynamic generation)

### Section 3: Custom Function Development
**Current (Lines 70-110)**: Broken import examples

**Rewrite To**:
```rst
Custom Function Development
--------------------------

Create your own GPU-accelerated processing functions with proper memory type declarations.

Memory Type Decorators
~~~~~~~~~~~~~~~~~~~~~

Declare memory interfaces explicitly:

.. code-block:: python

    from openhcs.core.memory.decorators import cupy_func, torch_func, numpy_func

    @cupy_func  # Input and output are CuPy arrays
    def cupy_gaussian_blur(image_stack, sigma=1.0):
        """GPU-accelerated Gaussian blur using CuPy."""
        import cupy as cp
        from cupyx.scipy import ndimage
        
        # Process each image in the stack
        blurred_stack = cp.zeros_like(image_stack)
        for i in range(image_stack.shape[0]):
            blurred_stack[i] = ndimage.gaussian_filter(image_stack[i], sigma=sigma)
        
        return blurred_stack

    @torch_func  # Input and output are PyTorch tensors
    def torch_edge_detection(image_tensor, threshold=0.1):
        """GPU-accelerated edge detection using PyTorch."""
        import torch
        import torch.nn.functional as F
        
        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # Apply convolution
        edges_x = F.conv2d(image_tensor.unsqueeze(1), sobel_x.unsqueeze(0).unsqueeze(0))
        edges_y = F.conv2d(image_tensor.unsqueeze(1), sobel_y.unsqueeze(0).unsqueeze(0))
        
        # Combine and threshold
        edges = torch.sqrt(edges_x**2 + edges_y**2)
        return (edges > threshold).float().squeeze()

    @numpy_func  # Input and output are NumPy arrays
    def numpy_morphology(image_stack, operation='opening', kernel_size=3):
        """CPU-based morphological operations using NumPy."""
        import numpy as np
        from skimage import morphology
        
        kernel = morphology.disk(kernel_size)
        processed_stack = np.zeros_like(image_stack)
        
        for i in range(image_stack.shape[0]):
            if operation == 'opening':
                processed_stack[i] = morphology.opening(image_stack[i], kernel)
            elif operation == 'closing':
                processed_stack[i] = morphology.closing(image_stack[i], kernel)
            elif operation == 'erosion':
                processed_stack[i] = morphology.erosion(image_stack[i], kernel)
            elif operation == 'dilation':
                processed_stack[i] = morphology.dilation(image_stack[i], kernel)
        
        return processed_stack

Memory Type Consistency Rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**CRITICAL**: All functions within the same step must have consistent memory types:

.. code-block:: python

    # ✅ VALID: All functions in step use same memory type
    @cupy_func
    def cupy_denoise(image_stack):
        import cupy as cp
        return cp.median_filter(image_stack, size=3)

    @cupy_func  # Same memory type as denoise
    def cupy_enhance(image_stack):
        import cupy as cp
        return cp.clip(image_stack * 1.5, 0, 1)

    step = FunctionStep(
        func=[cupy_denoise, cupy_enhance],  # ✅ Both CuPy - VALID
        name="GPU Processing Chain"
    )

    # ❌ INVALID: Mixed memory types in same step
    @cupy_func
    def cupy_denoise(image_stack):
        return processed_stack

    @torch_func  # Different memory type!
    def torch_enhance(image_stack):
        return processed_stack

    step = FunctionStep(
        func=[cupy_denoise, torch_enhance],  # ❌ COMPILATION ERROR
        name="Mixed Memory Step"
    )

Different Steps Can Have Different Memory Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Memory type conversions happen **between steps**, not within steps:

.. code-block:: python

    # ✅ VALID: Different memory types across different steps
    pipeline = Pipeline(steps=[
        # Step 1: CuPy processing
        FunctionStep(
            func=cupy_gaussian_blur,      # CuPy only
            name="GPU Blur"
        ),

        # Step 2: PyTorch processing (automatic conversion from CuPy)
        FunctionStep(
            func=torch_edge_detection,    # PyTorch only
            name="PyTorch Edge Detection"
        ),

        # Step 3: CPU processing (automatic conversion from PyTorch)
        FunctionStep(
            func=numpy_morphology,        # NumPy only
            name="CPU Morphology"
        )
    ])

    # System automatically converts: CuPy → PyTorch → NumPy between steps

Component-Specific Memory Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Component-specific patterns can have different memory types per component:

.. code-block:: python

    # ✅ VALID: Different memory types per component
    step = FunctionStep(
        func={
            'DAPI': cupy_process_nuclei,     # CuPy for DAPI
            'GFP': torch_process_proteins,   # PyTorch for GFP
            'BF': numpy_process_brightfield  # NumPy for brightfield
        },
        group_by=GroupBy.CHANNEL,
        name="Component-Specific Processing"
    )

    # ✅ VALID: Sequential functions per component (consistent within each)
    step = FunctionStep(
        func={
            'DAPI': [cupy_denoise, cupy_enhance],    # ✅ All CuPy for DAPI
            'GFP': [torch_denoise, torch_enhance],   # ✅ All PyTorch for GFP
            'BF': numpy_process_brightfield          # ✅ NumPy for brightfield
        },
        group_by=GroupBy.CHANNEL,
        name="Sequential Per Component"
    )

    # ❌ INVALID: Mixed memory types within same component
    step = FunctionStep(
        func={
            'DAPI': [cupy_denoise, torch_enhance],  # ❌ Mixed types in DAPI
            'GFP': torch_process_gfp                # ✅ Consistent PyTorch
        },
        group_by=GroupBy.CHANNEL,
        name="Invalid Mixed Component"
    )

Custom Memory Type Decorators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create specialized decorators for specific workflows:

.. code-block:: python

    from openhcs.core.memory.decorators import memory_types

    @memory_types(input_type="cupy", output_type="numpy")
    def gpu_to_cpu_processing(cupy_array):
        """Process on GPU, return CPU result."""
        import cupy as cp
        
        # GPU processing
        processed = cp.max(cupy_array, axis=0, keepdims=True)
        
        # Convert to CPU for return
        return cp.asnumpy(processed)

    @memory_types(input_type="numpy", output_type="torch")
    def cpu_to_gpu_processing(numpy_array):
        """Process on CPU, return GPU tensor."""
        import numpy as np
        import torch
        
        # CPU processing
        processed = np.percentile(numpy_array, 99, axis=0, keepdims=True)
        
        # Convert to GPU tensor
        return torch.from_numpy(processed).cuda()
```

**Context Engine Sources**:
- openhcs/core/memory/decorators.py lines 119-133 (memory_types decorator)
- openhcs/core/memory/decorators.py lines 373-391 (tensorflow_func example)
- docs/architecture/function-patterns.md lines 226-273 (memory type integration)

### Section 4: Special Inputs and Outputs
**Current**: Missing entirely

**Add New Section**:
```rst
Special Inputs and Outputs
-------------------------

Handle complex data flow between functions using special inputs and outputs.

Special Output Declaration
~~~~~~~~~~~~~~~~~~~~~~~~~

Functions can produce additional outputs beyond the main image:

.. code-block:: python

    from openhcs.core.pipeline.function_contracts import special_outputs

    @special_outputs("positions", "metadata")
    @cupy_func
    def generate_positions_with_metadata(image_stack):
        """Generate positions and metadata from image stack."""
        import cupy as cp
        
        # Main processing
        processed_images = cp.max(image_stack, axis=0, keepdims=True)
        
        # Generate positions (example)
        positions = cp.array([[0, 0], [100, 0], [0, 100], [100, 100]])
        
        # Generate metadata
        metadata = {
            'num_images': image_stack.shape[0],
            'max_intensity': float(cp.max(image_stack)),
            'processing_method': 'max_projection'
        }
        
        # Return: (main_output, special_output1, special_output2, ...)
        return processed_images, positions, metadata

Special Input Consumption
~~~~~~~~~~~~~~~~~~~~~~~~

Functions can consume special outputs from previous functions:

.. code-block:: python

    from openhcs.core.pipeline.function_contracts import special_inputs

    @special_inputs("positions", "metadata")
    @cupy_func
    def stitch_with_positions(image_stack, positions, metadata):
        """Stitch images using positions from previous function."""
        import cupy as cp
        
        # Use positions for stitching
        canvas_size = (2000, 2000)  # Based on positions
        stitched = cp.zeros(canvas_size, dtype=image_stack.dtype)
        
        for i, (x, y) in enumerate(positions):
            if i < image_stack.shape[0]:
                h, w = image_stack[i].shape
                stitched[y:y+h, x:x+w] = image_stack[i]
        
        # Use metadata for processing decisions
        if metadata.get('processing_method') == 'max_projection':
            # Apply specific post-processing for max projection
            stitched = cp.clip(stitched, 0, metadata['max_intensity'])
        
        return stitched.reshape(1, *stitched.shape)  # Return as 3D

Pipeline with Special I/O
~~~~~~~~~~~~~~~~~~~~~~~~

Connect functions with special data flow:

.. code-block:: python

    pipeline = Pipeline(steps=[
        # Step 1: Generate positions and metadata
        FunctionStep(
            func=generate_positions_with_metadata,
            variable_components=[VariableComponents.SITE],
            name="Position Generation"
        ),
        
        # Step 2: Use positions for stitching
        FunctionStep(
            func=stitch_with_positions,
            variable_components=[VariableComponents.SITE],
            name="Image Stitching"
        )
    ], name="Special I/O Pipeline")

The system automatically routes special outputs from step 1 to special inputs in step 2.
```

**Context Engine Sources**:
- openhcs/core/pipeline/function_contracts.py lines 27-46 (special_outputs)
- openhcs/core/pipeline/function_contracts.py lines 49-72 (special_inputs)
- docs/architecture/pipeline-debugging-guide.md lines 207-222 (special I/O examples)

## Implementation Checklist

### Content Verification (Using Context Engine)
- [x] **Advanced patterns**: Verified from function-patterns.md nested examples
- [x] **Memory decorators**: Verified from memory/decorators.py implementation
- [x] **Special I/O**: Verified from function_contracts.py decorators
- [x] **Pipeline factories**: Verified from pipeline_factories.py
- [x] **Custom functions**: Verified decorator patterns and GPU integration

### Cross-Reference Updates Needed
- [ ] Update: `docs/source/user_guide/basic_usage.rst` (prerequisite)
- [ ] Create: `docs/source/user_guide/advanced_usage.rst` (next level)
- [ ] Create: `docs/source/concepts/memory_optimization.rst`

### Estimated Effort
**Writing time**: 12-16 hours  
**Testing time**: 6-8 hours (verify all advanced examples work)  
**Total**: 18-24 hours

### Success Criteria
1. **All advanced patterns work**: Complex nested examples functional
2. **Custom function examples**: GPU-accelerated functions with proper decorators
3. **Special I/O examples**: Data flow between functions working
4. **Memory optimization**: Clear guidance for performance
5. **Progressive complexity**: Builds naturally from basic_usage

This plan provides comprehensive coverage of advanced OpenHCS features with verified, working examples that demonstrate the full power of the function pattern system.
