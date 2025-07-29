# OpenHCS Autoregistry Pickling Investigation

## Problem Summary

**Core Issue**: Auto-discovered functions from external libraries (pyclesperanto, scikit-image) cannot be pickled for subprocess execution in OpenHCS, breaking multiprocessing workflows.

**Root Cause**: Module replacement/monkey patching was removed to fix code generation issues, but this broke pickling compatibility.

## Background: OpenHCS Function System

### Memory Decorators
OpenHCS uses memory type decorators that add essential attributes and functionality to functions:

- **Attributes**: `input_memory_type`, `output_memory_type` 
- **Features**: 
  - Dtype preservation wrappers
  - Thread-local CUDA stream management (GPU backends)
  - OOM recovery
  - Additional parameters: `slice_by_slice`, `dtype_conversion`

### Autoregistration Process
1. **Phase 1**: Scan OpenHCS processing directory for native functions (already decorated)
2. **Phase 2**: Register external libraries by applying decorators at runtime

### Function Contract
- First input: 3D array
- First output: 3D array  
- Same dimensions input→output
- Perfect fit for image processing functions from external libraries

## The Historical Problem Chain

### Original Working System (Before July 2025)
```python
# 1. Auto-discover external functions
pyclesperanto.sobel  # Original function

# 2. Apply OpenHCS decorators
wrapped_sobel = pyclesperanto_decorator(pyclesperanto.sobel)

# 3. MONKEY PATCH the original module
pyclesperanto.sobel = wrapped_sobel  # Replace original with wrapped version

# 4. Functions were pickleable and worked everywhere
```

### The Code Generation Problem (July 2025)
When users clicked "Generate Code" in the TUI, it would output:
```python
# Generated code called original function
result = pyclesperanto.sobel(image)  # This was the wrapped version due to monkey patching
```

**Issue**: The generated code looked like it was calling the original library function, but actually called the OpenHCS-wrapped version. This was confusing and potentially problematic for code portability.

### The "Fix" That Broke Pickling (Commit 57532b6)
```python
# Module replacement was disabled:
# Skip module replacement to avoid interfering with internal library usage
# OpenHCS wrapped functions should only be used through the pipeline system
```

**Result**: 
- ✅ Code generation now calls actual original functions
- ❌ Auto-discovered functions can't be pickled for subprocess execution
- ❌ Functions only available through `get_function_by_name("sobel", "pyclesperanto")`

## Current State Analysis

### What Works
- **Manual functions**: Functions manually defined with decorators can be pickled
- **Native OpenHCS functions**: Work perfectly in all contexts
- **Registry lookup**: `get_function_by_name()` finds auto-discovered functions

### What's Broken
- **Subprocess execution**: Auto-discovered functions fail to pickle
- **Multiprocessing**: Worker processes can't receive auto-discovered functions
- **Pipeline compilation**: Some workflows break when using auto-discovered functions

### Technical Details
**Pickling Issue**: Auto-discovered functions are decorated at registry initialization time and capture thread-local state, making them unpickleable.

**Decoration Timing**:
- Manual functions: Decorated at import time → pickleable
- Auto-discovered functions: Decorated at registry runtime → unpickleable

## Key Files and Locations

### Core Registry System
- `openhcs/processing/func_registry.py` - Main registry logic
- `openhcs/core/memory/decorators.py` - Memory type decorators

### Backend Registries  
- `openhcs/processing/backends/analysis/pyclesperanto_registry.py`
- `openhcs/processing/backends/analysis/scikit_image_registry.py`
- `openhcs/processing/backends/analysis/cupy_registry.py`

### Critical Commits
- **57532b6** (July 2025): Removed module replacement to fix code generation
- **8062a7f** (Most recent): Added manual sobel as workaround, documented the issue

## Potential Solutions

### Option 1: Fix Code Generation (Recommended)
- Restore monkey patching for pickling compatibility
- Fix code generator to be aware of OpenHCS-wrapped functions
- Generate appropriate function calls: `get_function_by_name("sobel", "pyclesperanto")`

### Option 2: Fix Decoration Timing
- Move decoration to import time rather than registry initialization
- Use import hooks or other mechanisms to decorate at module import
- Maintain current no-monkey-patching approach

### Option 3: Hybrid Approach
- Keep monkey patching for internal use (pickling)
- Code generator explicitly calls original functions when generating portable code
- Best of both worlds but more complex

## Next Steps

1. **Understand requirements**: Clarify if code generation portability is critical
2. **Choose approach**: Decide between fixing code generation vs decoration timing
3. **Implement solution**: Restore pickling while maintaining code generation quality
4. **Test thoroughly**: Ensure both subprocess execution and code generation work

## Context for Future Investigation

The user (Tristan) is a cell biology PhD student who built OpenHCS for microscopy format compatibility. He requires:
- Extremely detailed implementation plans
- Thorough investigation before making changes  
- Solutions that maintain system reliability
- Clear explanations of technical decisions

The system processes large datasets (up to 100GB per plate) and requires robust multiprocessing capabilities, making the pickling issue critical for performance.
