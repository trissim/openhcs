# Pyclesperanto Processor Extension Plan

## Overview
Extend the existing `pyclesperanto_processor.py` file with comprehensive 2D/3D wrapper functions for all pyclesperanto filters, following the established patterns and using existing decorators.

## Current File Analysis
- **Location**: `openhcs/processing/backends/processors/pyclesperanto_processor.py`
- **Current Functions**: 20 functions (normalization, projections, morphology, etc.)
- **Existing Patterns**: `@pyclesperanto_func` decorator, GPU memory management, validation
- **Architecture**: Clean, well-documented, follows OpenHCS patterns

## Extension Strategy

### **Approach: Systematic Function Categories**
Instead of 600+ individual files, add organized sections to the existing file:

```python
# ============================================================================
# ARITHMETIC OPERATIONS - 2D/3D Wrappers
# ============================================================================

@pyclesperanto_func
def add_images_2d(image1: "cle.Array", image2: "cle.Array") -> "cle.Array":
    """2D wrapper for pyclesperanto.add_images"""
    
@pyclesperanto_func  
def add_images_3d(image1: "cle.Array", image2: "cle.Array") -> "cle.Array":
    """3D wrapper for pyclesperanto.add_images"""

# ============================================================================
# MORPHOLOGICAL OPERATIONS - 2D/3D Wrappers  
# ============================================================================

@pyclesperanto_func
def binary_dilate_2d(image: "cle.Array", radius: int = 1) -> "cle.Array":
    """2D wrapper for pyclesperanto.binary_dilate"""

@pyclesperanto_func
def binary_dilate_3d(image: "cle.Array", radius: int = 1) -> "cle.Array":
    """3D wrapper for pyclesperanto.binary_dilate"""
```

### **Implementation Categories (Priority Order)**

#### **Phase 1: Core Operations (High Priority)**
1. **Arithmetic Operations** (~20 functions)
   - `add_images_2d/3d`, `subtract_images_2d/3d`, `multiply_images_2d/3d`
   - `add_scalar_2d/3d`, `multiply_scalar_2d/3d`, `divide_images_2d/3d`

2. **Binary Operations** (~15 functions)
   - `binary_and_2d/3d`, `binary_or_2d/3d`, `binary_not_2d/3d`
   - `binary_dilate_2d/3d`, `binary_erode_2d/3d`

3. **Comparison Operations** (~15 functions)
   - `greater_2d/3d`, `smaller_2d/3d`, `equal_2d/3d`
   - `greater_constant_2d/3d`, `smaller_constant_2d/3d`

#### **Phase 2: Filtering Operations (Medium Priority)**
1. **Basic Filters** (~20 functions)
   - `gaussian_blur_2d/3d`, `median_2d/3d`, `mean_filter_2d/3d`
   - `maximum_filter_2d/3d`, `minimum_filter_2d/3d`

2. **Morphological Filters** (~25 functions)
   - `dilate_box_2d/3d`, `erode_sphere_2d/3d`, `opening_2d/3d`
   - `closing_2d/3d`, `top_hat_2d/3d`, `bottom_hat_2d/3d`

3. **Edge Detection** (~10 functions)
   - `sobel_2d/3d`, `laplace_2d/3d`, `gradient_x_3d`, `gradient_y_3d`

#### **Phase 3: Advanced Operations (Lower Priority)**
1. **Label Operations** (~40 functions)
   - `connected_components_2d/3d`, `label_spots_2d/3d`
   - `dilate_labels_2d/3d`, `filter_labels_2d/3d`

2. **Statistical Operations** (~20 functions)
   - `histogram_2d/3d`, `statistics_2d/3d`, `variance_2d/3d`

3. **Transformation Operations** (~10 functions)
   - `rotate_2d/3d`, `scale_2d/3d`, `translate_2d/3d`

### **Naming Convention**
```python
# For dimension-agnostic functions:
def function_name_2d(args) -> "cle.Array":
    """2D wrapper for pyclesperanto.function_name - processes 2D images"""
    
def function_name_3d(args) -> "cle.Array": 
    """3D wrapper for pyclesperanto.function_name - processes 3D volumes"""

# For dimension-specific functions:
def projection_z_3d(args) -> "cle.Array":
    """3D-only wrapper for pyclesperanto.maximum_z_projection"""
```

### **Template Pattern**
```python
@pyclesperanto_func
def template_function_2d(
    image: "cle.Array",
    param1: float = 1.0,
    param2: int = 5
) -> "cle.Array":
    """
    2D wrapper for pyclesperanto.template_function.
    
    Processes 2D images with specified parameters.
    
    Args:
        image: 2D pyclesperanto Array of shape (Y, X)
        param1: Description of parameter 1
        param2: Description of parameter 2
        
    Returns:
        Processed 2D pyclesperanto Array of shape (Y, X)
    """
    _check_pyclesperanto_available()
    import pyclesperanto as cle
    
    # Validate 2D input
    if len(image.shape) != 2:
        raise ValueError(f"Expected 2D array, got {len(image.shape)}D array")
    
    # Call pyclesperanto function
    result = cle.template_function(image, param1=param1, param2=param2)
    
    return result

@pyclesperanto_func  
def template_function_3d(
    image: "cle.Array",
    param1: float = 1.0,
    param2: int = 5
) -> "cle.Array":
    """
    3D wrapper for pyclesperanto.template_function.
    
    Processes 3D volumes with specified parameters.
    
    Args:
        image: 3D pyclesperanto Array of shape (Z, Y, X)
        param1: Description of parameter 1  
        param2: Description of parameter 2
        
    Returns:
        Processed 3D pyclesperanto Array of shape (Z, Y, X)
    """
    _check_pyclesperanto_available()
    import pyclesperanto as cle
    
    # Validate 3D input
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D array, got {len(image.shape)}D array")
    
    # Call pyclesperanto function
    result = cle.template_function(image, param1=param1, param2=param2)
    
    return result
```

### **File Organization**
```python
# Current content (lines 1-767)
# ... existing functions ...

# ============================================================================
# EXTENDED PYCLESPERANTO WRAPPERS - 2D/3D FILTER SYSTEM
# ============================================================================

# Phase 1: Core Operations
# ============================================================================
# ARITHMETIC OPERATIONS
# ============================================================================
# ... arithmetic functions ...

# ============================================================================  
# BINARY OPERATIONS
# ============================================================================
# ... binary functions ...

# Phase 2: Filtering Operations  
# ============================================================================
# FILTERING OPERATIONS
# ============================================================================
# ... filter functions ...

# Phase 3: Advanced Operations
# ============================================================================
# LABEL OPERATIONS  
# ============================================================================
# ... label functions ...
```

### **Benefits of This Approach**
1. **Leverages existing infrastructure** - decorators, validation, patterns
2. **Maintains consistency** - follows established OpenHCS conventions
3. **Single file management** - easier to maintain than 600+ separate files
4. **Gradual implementation** - can add functions incrementally by priority
5. **Automatic GPU memory management** - uses existing `@pyclesperanto_func`
6. **Clear organization** - functions grouped by category with clear headers

### **Implementation Steps**
1. **Start with Phase 1** - Add ~50 core arithmetic/binary/comparison functions
2. **Test thoroughly** - Ensure all wrappers work correctly with existing patterns
3. **Add Phase 2** - Filtering and morphological operations (~55 functions)
4. **Add Phase 3** - Advanced label and statistical operations (~70 functions)
5. **Documentation** - Update function inventory with implementation status

### **Estimated Timeline**
- **Phase 1**: ~175 wrapper functions (arithmetic, binary, comparison)
- **Phase 2**: ~55 wrapper functions (filtering, morphology)  
- **Phase 3**: ~70 wrapper functions (labels, statistics, transforms)
- **Total**: ~300 wrapper functions in organized sections

This approach is much more practical and maintainable than creating 600+ separate files!
