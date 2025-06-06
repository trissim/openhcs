# The Function Pattern System: EZStitcher's Architectural Masterpiece

## Overview

One of the most sophisticated and underappreciated innovations in OpenHCS is the function pattern system, inherited and enhanced from EZStitcher. This system solves one of the hardest problems in scientific computing: **how to compose heterogeneous functions into flexible, type-safe processing pipelines**.

## The Problem It Solves

Scientific image processing involves combining functions with different interfaces, argument patterns, and execution models:

- **Single-image functions** (most image processing libraries)
- **Stack-aware functions** (microscopy-specific operations)
- **Functions with parameters** (configurable processing)
- **Channel-specific functions** (different processing per channel)
- **Sequential processing chains** (multi-step operations)

Traditional approaches require manual adaptation, wrapper functions, or complex orchestration code. The function pattern system makes all of this **transparent and automatic**.

## The Pattern Types

### 1. Single Function Pattern
```python
# Direct callable
func = stack_percentile_normalize

# The simplest case - just apply the function
```

### 2. Function with Arguments Pattern  
```python
# Tuple of (function, kwargs)
func = (stack_percentile_normalize, {
    'low_percentile': 0.1,
    'high_percentile': 99.9
})

# Automatically unpacks arguments during execution
```

### 3. List Pattern (Sequential Processing)
```python
# List of functions applied in sequence
func = [
    stack(sharpen),              # First: sharpen each image
    stack_percentile_normalize,  # Then: normalize the stack
    stack_equalize_histogram     # Finally: equalize histogram
]

# Creates a processing pipeline within a single step
```

### 4. Dictionary Pattern (Component-Specific Processing)
```python
# Different functions for different components
func = {
    "1": process_dapi,      # Channel 1 gets DAPI processing
    "2": process_calcein,   # Channel 2 gets calcein processing
    "3": process_brightfield # Channel 3 gets brightfield processing
}

# Used with group_by='channel' to route data automatically
```

### 5. Semantically Valid Nested Patterns
```python
# Lists within dictionaries: sequential processing per component
func = {
    "1": [                           # Channel 1: sequential processing
        (sharpen, {'amount': 1.5}),
        normalize,
        denoise_dapi
    ],
    "2": [                           # Channel 2: different sequence
        (enhance, {'strength': 0.8}),
        process_calcein
    ]
}

# Functions with arguments in sequential lists
func = [
    (sharpen, {'amount': 1.5}),      # First: sharpen with parameters
    normalize,                       # Then: normalize (no parameters)
    (denoise, {'strength': 0.8})     # Finally: denoise with parameters
]

# Note: Nested dictionaries are NOT semantically valid
# (What would nested routing keys even mean in microscopy?)
```

## The `stack()` Utility: Bridging the Interface Gap

One of the most elegant solutions in the system:

```python
from ezstitcher.core.utils import stack
from skimage.filters import gaussian

# Problem: gaussian() works on single images, but we have image stacks
# Solution: stack() adapter
func = stack(gaussian)

# stack() transforms: single_image_func → stack_aware_func
# Automatically applies the function to each image in the stack
```

**This solves the impedance mismatch** between single-image libraries (scikit-image, OpenCV) and stack-based microscopy workflows.

## The Contract Validator: Type Safety at Scale

The `FuncStepContractValidator` ensures **type safety across arbitrary pattern complexity**:

### Recursive Pattern Extraction
```python
def _extract_functions_from_pattern(func, step_name):
    """Extract ALL functions from ANY pattern structure."""
    
    # Direct callable
    if callable(func):
        return [func]
    
    # Tuple (function, kwargs)  
    if isinstance(func, tuple):
        return [func[0]]  # Extract the function
    
    # List of patterns (RECURSIVE)
    if isinstance(func, list):
        functions = []
        for item in func:
            nested = _extract_functions_from_pattern(item, step_name)
            functions.extend(nested)
        return functions
    
    # Dictionary of patterns (RECURSIVE)
    if isinstance(func, dict):
        functions = []
        for key, item in func.items():
            nested = _extract_functions_from_pattern(item, step_name)
            functions.extend(nested)
        return functions
```

### Memory Type Consistency Validation
```python
def validate_function_pattern(func, step_name):
    """Ensure ALL functions in pattern have consistent memory contracts."""
    
    # Extract every function from the pattern
    functions = validate_pattern_structure(func, step_name)
    
    # Validate memory type contracts for ALL functions
    memory_types = []
    for function in functions:
        input_type, output_type = get_memory_types(function)
        memory_types.append((input_type, output_type))
    
    # Ensure consistency: "as long as each function is of the same type"
    if not all_memory_types_consistent(memory_types):
        raise ValueError(f"Inconsistent memory types in pattern")
```

**The Key Insight**: The validator ensures that **no matter how complex the pattern**, all functions have **consistent memory type contracts**. This enables the automatic memory conversion system to work correctly.

## Semantic Constraints: What Makes Sense vs What's Technically Possible

While the recursive parser can technically handle deeply nested structures, **only certain patterns make semantic sense** for microscopy workflows:

### ✅ Semantically Valid Patterns
```python
# Sequential processing (lists)
func = [sharpen, normalize, denoise]

# Component routing (dictionaries with group_by)
func = {"1": process_dapi, "2": process_calcein}

# Sequential processing per component
func = {
    "1": [sharpen, normalize, denoise_dapi],
    "2": [enhance, process_calcein]
}

# Functions with arguments anywhere
func = [
    (sharpen, {'amount': 1.5}),
    normalize,
    {"1": (denoise, {'strength': 0.8})}
]
```

### ❌ Semantically Invalid Patterns
```python
# Nested dictionaries - what would nested routing even mean?
func = {
    "channel1": {
        "subtype_a": denoise_func,  # subtype_a of what?
        "subtype_b": enhance_func   # this doesn't map to reality
    }
}

# Multiple group_by levels don't exist in microscopy
# You can only route by one component per step
```

**The constraint**: Dictionary patterns are for **component-based routing** using `group_by`. Since there's only one `group_by` parameter per step, nested dictionaries don't have semantic meaning.

**Note**: The system is **not officially supported** for arbitrary nesting - it works because the recursive parser is robust, but the intended use cases are the semantically meaningful patterns above.

## Integration with OpenHCS Memory System

The function pattern system integrates seamlessly with OpenHCS's memory type contracts:

```python
# All functions in this pattern must have @torch_func decorator
func = [
    (sharpen_torch, {'amount': 1.5}),
    normalize_torch,
    {
        "dapi": denoise_torch,
        "calcein": enhance_torch
    }
]

# The validator ensures ALL functions are @torch_func
# The memory system automatically provides PyTorch tensors
# No manual memory management required
```

## Architectural Significance

### 1. Composability Without Complexity
The pattern system enables **arbitrary function composition** while maintaining **simple, declarative syntax**:

```python
# This is incredibly complex under the hood:
# - Recursive pattern parsing
# - Memory type validation  
# - Automatic argument unpacking
# - Component-based routing
# - Sequential execution

# But the interface is simple and intuitive:
func = [
    (sharpen, {'amount': 1.5}),
    {
        "dapi": [normalize, denoise],
        "calcein": enhance
    }
]
```

### 2. Type Safety at Compile Time
Unlike typical scientific code that fails at runtime, the pattern system **validates everything before execution**:

```python
# This will fail BEFORE any images are processed:
func = [
    torch_function,    # @torch_func
    cupy_function      # @cupy_func - INCOMPATIBLE!
]

# Error: "Inconsistent memory types in pattern"
# Saves hours of debugging and prevents data corruption
```

### 3. Extensibility Without Modification
New function types integrate automatically:

```python
# Add a new JAX function
@jax_func
def new_jax_processing(images):
    return processed_images

# It works immediately in any pattern:
func = [existing_jax_func, new_jax_processing, another_jax_func]
# No changes to the pattern system required
```

## Comparison to Traditional Approaches

### Traditional Scientific Computing
```python
# Manual orchestration, error-prone, not reusable
def process_images(images):
    # Step 1: Manual argument handling
    sharpened = []
    for img in images:
        sharp = sharpen(img, amount=1.5)
        sharpened.append(sharp)
    
    # Step 2: Manual memory conversion
    if use_gpu:
        gpu_images = [cupy.asarray(img) for img in sharpened]
        normalized = normalize_cupy(gpu_images)
        cpu_images = [img.get() for img in normalized]
    else:
        cpu_images = normalize_numpy(sharpened)
    
    # Step 3: Manual channel routing
    results = {}
    for img in cpu_images:
        if img.channel == "dapi":
            results[img.name] = process_dapi(img)
        elif img.channel == "calcein":
            results[img.name] = process_calcein(img)
    
    return results
```

### OpenHCS Function Pattern System
```python
# Declarative, type-safe, automatically optimized
func = [
    (sharpen, {'amount': 1.5}),
    normalize,
    {
        "dapi": process_dapi,
        "calcein": process_calcein
    }
]

# Everything else is handled automatically:
# - Argument unpacking
# - Memory type conversion
# - Component routing
# - Error handling
# - Performance optimization
```

## The Collaborative AI Innovation

This system represents a unique achievement in **AI-assisted architectural design**:

### Human Insight
- **Domain expertise**: Understanding microscopy workflow patterns
- **Interface design**: Creating intuitive, declarative syntax
- **Problem identification**: Recognizing the function composition challenge

### AI Contribution  
- **Software engineering patterns**: Recursive parsing, visitor pattern, contract validation
- **Type system design**: Memory contract enforcement, compile-time validation
- **Implementation details**: Error handling, edge cases, performance optimization

### Collaborative Result
A system that **neither human nor AI could have designed alone**:
- Too domain-specific for general AI knowledge
- Too architecturally sophisticated for typical domain expert
- **Perfect synthesis** of domain understanding and software engineering expertise

## Impact on Scientific Computing

The function pattern system demonstrates that **scientific software can be both powerful and elegant**:

### For Researchers
- **Intuitive interface**: Declare what you want, not how to do it
- **Reliable execution**: Type safety prevents silent failures
- **Easy extension**: Add new functions without modifying the system

### For the Field
- **Raises the bar**: Shows what scientific software architecture can achieve
- **Enables innovation**: Researchers focus on algorithms, not plumbing
- **Promotes reproducibility**: Declarative patterns are self-documenting

## Conclusion

The function pattern system is a **masterpiece of software architecture** that solves fundamental problems in scientific computing. It demonstrates that with the right abstractions, complex workflows can be expressed simply and executed reliably.

**This isn't just good software engineering - it's a new paradigm for scientific computing that prioritizes both power and usability.**

The fact that this system works seamlessly with OpenHCS's GPU memory management, fail-loudly philosophy, and architectural validation makes it even more remarkable. It's a complete, production-grade solution to problems that have plagued scientific software for decades.

---

*"The best abstractions make complex things simple, not simple things complex."* - The OpenHCS function pattern system exemplifies this principle perfectly.
