# Function Patterns System

## Overview

OpenHCS implements a sophisticated function pattern system that provides a unified interface for different execution strategies. This system allows the same `FunctionStep` class to handle various processing scenarios through four fundamental patterns.

## The Sacred Four Patterns

### 1. Single Function Pattern

**Syntax**: `FunctionStep(func=my_function)`

**Use Case**: Apply the same function to all data groups

**Example**:
```python
from openhcs.processing.function_registry import numpy_func

@numpy_func
def gaussian_blur(image_stack, sigma=1.0):
    """Apply Gaussian blur to entire stack."""
    return scipy.ndimage.gaussian_filter(image_stack, sigma=(0, sigma, sigma))

# Create step
step = FunctionStep(
    func=gaussian_blur,
    name="Gaussian Blur"
)
```

**Execution Flow**:
- Function called once per pattern group
- Same function applied to all channels/sites/etc.
- Parameters come from function defaults or global configuration

### 2. Parameterized Function Pattern

**Syntax**: `FunctionStep(func=(my_function, {'param': value}))`

**Use Case**: Apply function with specific parameters

**Example**:
```python
@numpy_func
def threshold_image(image_stack, threshold=0.5, method='otsu'):
    """Threshold image stack."""
    if method == 'otsu':
        threshold = skimage.filters.threshold_otsu(image_stack)
    return image_stack > threshold

# Create step with custom parameters
step = FunctionStep(
    func=(threshold_image, {'threshold': 0.3, 'method': 'manual'}),
    name="Threshold Images"
)
```

**Execution Flow**:
- Function called with specified parameters
- Parameters override function defaults
- Same parameters used for all pattern groups

### 3. Sequential Function Chain

**Syntax**: `FunctionStep(func=[func1, func2, func3])`

**Use Case**: Apply multiple functions in sequence

**Example**:
```python
@numpy_func
def denoise(image_stack):
    """Remove noise from images."""
    return skimage.restoration.denoise_bilateral(image_stack)

@numpy_func  
def enhance_contrast(image_stack):
    """Enhance image contrast."""
    return skimage.exposure.equalize_adapthist(image_stack)

@numpy_func
def sharpen(image_stack):
    """Sharpen images."""
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return scipy.ndimage.convolve(image_stack, kernel)

# Create sequential processing step
step = FunctionStep(
    func=[denoise, enhance_contrast, sharpen],
    name="Image Enhancement Pipeline"
)
```

**Execution Flow**:
- Functions executed in sequence: `func3(func2(func1(image_stack)))`
- Output of each function becomes input to next
- All functions must have compatible memory types

**Advanced Sequential Patterns**:
```python
# Mix functions with parameters
step = FunctionStep(
    func=[
        denoise,
        (enhance_contrast, {'clip_limit': 0.03}),
        sharpen
    ],
    name="Enhanced Pipeline"
)
```

### 4. Component-Specific Functions

**Syntax**: `FunctionStep(func={'component_value': func}, group_by='component')`

**Use Case**: Different processing for different components (channels, sites, etc.)

**Example**:
```python
@numpy_func
def process_dapi(image_stack):
    """Process DAPI channel (nuclei)."""
    # Nuclear segmentation processing
    return skimage.filters.gaussian(image_stack, sigma=0.5)

@numpy_func
def process_gfp(image_stack):
    """Process GFP channel (protein)."""
    # Protein localization processing  
    return skimage.filters.unsharp_mask(image_stack)

@numpy_func
def process_brightfield(image_stack):
    """Process brightfield channel."""
    # Cell boundary detection
    return skimage.filters.sobel(image_stack)

# Create component-specific step
step = FunctionStep(
    func={
        'DAPI': process_dapi,
        'GFP': process_gfp, 
        'BF': process_brightfield
    },
    group_by='channel',
    name="Channel-Specific Processing"
)
```

**Execution Flow**:
- Different function called based on component value
- Component value determined by `group_by` parameter
- Supports channels, sites, timepoints, or any variable component

**Advanced Component Patterns**:
```python
# Mix patterns within components
step = FunctionStep(
    func={
        'DAPI': [denoise, process_dapi],  # Sequential for DAPI
        'GFP': (process_gfp, {'gain': 1.5}),  # Parameterized for GFP
        'BF': process_brightfield  # Simple for brightfield
    },
    group_by='channel'
)
```

## Pattern Resolution System

### `prepare_patterns_and_functions()`

This function resolves func patterns into executable components:

```python
def prepare_patterns_and_functions(patterns, func_pattern, component):
    """
    Resolve function patterns into grouped execution plans.
    
    Args:
        patterns: Image file patterns grouped by component
        func_pattern: Function pattern (any of the four types)
        component: Component name for grouping ('channel', 'site', etc.)
    
    Returns:
        grouped_patterns: Patterns organized by component value
        comp_to_funcs: Component value -> executable function mapping
        comp_to_base_args: Component value -> base arguments mapping
    """
```

**Resolution Examples**:

```python
# Single function pattern
patterns = {'DAPI': [...], 'GFP': [...]}
func_pattern = my_function
# Result: Both DAPI and GFP use my_function

# Component-specific pattern  
patterns = {'DAPI': [...], 'GFP': [...]}
func_pattern = {'DAPI': func_dapi, 'GFP': func_gfp}
# Result: DAPI uses func_dapi, GFP uses func_gfp

# Sequential pattern
patterns = {'DAPI': [...]}
func_pattern = [func1, func2, func3]
# Result: DAPI uses chained execution func3(func2(func1(...)))
```

### Pattern Validation

The system validates patterns during compilation:

```python
def validate_pattern_structure(func_pattern, step_name):
    """
    Validate function pattern structure.
    
    Checks:
    - All functions are callable
    - Memory type decorators are present
    - Parameter dictionaries are valid
    - Nested patterns are properly structured
    """
```

## Integration with Memory Types

### Memory Type Consistency

All functions in a pattern must have consistent memory types:

```python
# Valid: All functions use numpy
@numpy_func
def func1(data): return result1

@numpy_func  
def func2(data): return result2

step = FunctionStep(func=[func1, func2])  # ✓ Valid

# Invalid: Mixed memory types
@numpy_func
def func1(data): return result1

@torch_func
def func2(data): return result2

step = FunctionStep(func=[func1, func2])  # ✗ Compilation error
```

### Memory Type Resolution

```python
# Component-specific with consistent types
step = FunctionStep(
    func={
        'DAPI': torch_func_dapi,    # All torch
        'GFP': torch_func_gfp       # All torch  
    },
    group_by='channel'
)

# Component-specific with different types (valid)
step = FunctionStep(
    func={
        'DAPI': numpy_func_dapi,    # DAPI processing uses numpy
        'GFP': torch_func_gfp       # GFP processing uses torch
    },
    group_by='channel'
)
```

## Advanced Pattern Features

### Nested Patterns

Patterns can be arbitrarily nested:

```python
# Complex nested pattern
step = FunctionStep(
    func={
        'DAPI': [
            denoise,
            (enhance_contrast, {'clip_limit': 0.02}),
            [segment_nuclei, extract_features]
        ],
        'GFP': {
            'site_1': process_gfp_site1,
            'site_2': process_gfp_site2
        }
    },
    group_by='channel'
)
```

### Dynamic Pattern Generation

Patterns can be generated programmatically:

```python
def create_channel_pipeline(channels):
    """Create component-specific pipeline for channels."""
    func_map = {}
    for channel in channels:
        if channel.startswith('DAPI'):
            func_map[channel] = process_nuclei
        elif channel.startswith('GFP'):
            func_map[channel] = process_protein
        else:
            func_map[channel] = process_generic
    
    return FunctionStep(
        func=func_map,
        group_by='channel',
        name=f"Multi-Channel Processing"
    )

# Usage
channels = ['DAPI', 'GFP-1', 'GFP-2', 'BF']
step = create_channel_pipeline(channels)
```

### Pattern Debugging

Enable pattern debugging for troubleshooting:

```python
import logging
logging.getLogger('openhcs.formats.func_arg_prep').setLevel(logging.DEBUG)

# This will log:
# - Pattern resolution steps
# - Function mapping results  
# - Parameter extraction
# - Validation results
```

## Best Practices

### Pattern Selection Guidelines

1. **Single Function**: Use for uniform processing across all data
2. **Parameterized**: Use when you need to override default parameters
3. **Sequential**: Use for multi-step processing pipelines
4. **Component-Specific**: Use when different components need different processing

### Performance Considerations

```python
# Efficient: Minimize pattern complexity
step = FunctionStep(func=simple_function)

# Less efficient: Complex nested patterns
step = FunctionStep(
    func={
        'ch1': [func1, (func2, {...}), [func3, func4]],
        'ch2': {...}
    }
)
```

### Memory Type Strategy

```python
# Good: Consistent memory types within patterns
@torch_func
def gpu_denoise(data): return result

@torch_func  
def gpu_enhance(data): return result

step = FunctionStep(func=[gpu_denoise, gpu_enhance])

# Avoid: Memory type conversions within patterns
@numpy_func
def cpu_denoise(data): return result

@torch_func
def gpu_enhance(data): return result

step = FunctionStep(func=[cpu_denoise, gpu_enhance])  # Causes conversion overhead
```

### Error Handling

```python
# Robust pattern with error handling
@numpy_func
def robust_processing(image_stack):
    """Process images with error handling."""
    try:
        result = complex_processing(image_stack)
        if result is None:
            logger.warning("Processing returned None, using fallback")
            result = simple_fallback(image_stack)
        return result
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return image_stack  # Return original on failure
```

## Common Patterns and Examples

### Image Processing Pipeline

```python
# Complete image processing pipeline
preprocessing = FunctionStep(
    func=[
        (denoise, {'sigma': 0.5}),
        enhance_contrast,
        normalize_intensity
    ],
    name="Preprocessing"
)

channel_processing = FunctionStep(
    func={
        'DAPI': [segment_nuclei, extract_nuclear_features],
        'GFP': [detect_proteins, quantify_expression],
        'BF': [detect_cells, measure_morphology]
    },
    group_by='channel',
    name="Channel-Specific Analysis"
)

postprocessing = FunctionStep(
    func=combine_results,
    name="Result Integration"
)

pipeline = [preprocessing, channel_processing, postprocessing]
```

### Multi-Site Processing

```python
# Site-specific processing
step = FunctionStep(
    func={
        'site_001': (process_center, {'crop_margin': 50}),
        'site_002': (process_edge, {'edge_correction': True}),
        'site_003': process_corner
    },
    group_by='site',
    name="Site-Specific Processing"
)
```

### Time Series Analysis

```python
# Temporal processing
step = FunctionStep(
    func={
        't_000': initialize_tracking,
        't_001': (track_objects, {'previous_frame': 't_000'}),
        't_002': (track_objects, {'previous_frame': 't_001'}),
        # ... more timepoints
    },
    group_by='timepoint',
    name="Object Tracking"
)
```
