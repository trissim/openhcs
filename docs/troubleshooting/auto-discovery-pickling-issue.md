# Auto-Discovery Function Pickling Issue

## Problem Summary

Auto-discovered functions from external libraries (pyclesperanto, scikit-image, CuCIM) cannot be pickled for subprocess execution, causing pipeline failures when using the TUI's "Code" button or subprocess execution. Manual functions work fine.

**Error**: `TypeError: cannot pickle '_thread._local' object`

## Current Status

- ✅ **Manual functions**: Fully functional and pickleable
- ❌ **Auto-discovered functions**: Functional but not pickleable
- ✅ **Workaround**: Manual sobel function added to cupy_processor.py
- ❌ **Root cause**: Still unresolved

## Investigation Timeline

### Historical Context (CRITICAL)
**Previous Implementation**: Commit `1a4eff8` (July 3, 2025) implemented the **exact same import hook system** we attempted to recreate:

```python
# From commit 1a4eff8 - WORKING import hook system
def _install_import_hook():
    def _decorating_import(name, *args, **kwargs):
        module = _original_import(name, *args, **kwargs)
        if name == 'pyclesperanto':
            _decorate_pyclesperanto_on_import(module)
        # ... etc
    __builtins__['__import__'] = _decorating_import

# Auto-decoration with minimal attributes (NOT full decorators)
def _decorate_pyclesperanto_on_import(cle_module):
    for func_name in common_functions:
        func.input_memory_type = "pyclesperanto"  # MINIMAL decoration
        func.output_memory_type = "pyclesperanto"  # NOT @pyclesperanto decorator
```

**Key Insight**: The original working system used **minimal attribute decoration**, not full OpenHCS decorators. This avoided thread-local capture while maintaining registry compatibility.

### Initial Hypothesis (INCORRECT)
- **Theory**: Auto-discovered functions have different decoration timing
- **Finding**: Both manual and auto-discovered functions use same decorators
- **Conclusion**: Decoration process itself is not the issue

### Module Reference Hypothesis (PARTIALLY CORRECT)
- **Theory**: `__module__` attribute differences cause pickling issues
- **Finding**: Auto-discovered functions reference external library modules
- **Test**: Changing `__module__` attribute didn't fix the issue
- **Conclusion**: Module reference is related but not the root cause

### Thread-Local Capture Hypothesis (CORRECT)
- **Theory**: Auto-discovered functions capture thread-locals during decoration
- **Finding**: Functions decorated at runtime (after TUI starts) capture active thread-locals
- **Key insight**: Manual functions decorated at module import time (clean environment)
- **Conclusion**: **TIMING OF DECORATION IS THE ROOT CAUSE**

## Root Cause Analysis

### The Core Issue

**Manual Functions**:
```python
# In cupy_processor.py - decorated at MODULE IMPORT TIME
@cupy_func  
def sobel(image, ...):
    return cucim_filters.sobel(image, ...)
```
- ✅ Decorated during module import (clean environment, no active threads)
- ✅ No thread-locals captured in function closure
- ✅ Pickleable

**Auto-Discovered Functions**:
```python
# In registry - decorated at REGISTRY RUNTIME
wrapper_func = cupy(original_func)  # Happens when TUI is running
```
- ❌ Decorated during registry scanning (contaminated environment, active threads)
- ❌ Thread-locals (`_thread_gpu_contexts`) captured in function closure
- ❌ Not pickleable

### Technical Details

1. **Thread-local object**: `_thread_gpu_contexts = threading.local()` in decorators.py line 216
2. **Capture mechanism**: When decorators create wrapper functions at runtime, they capture references to module globals
3. **Pickling failure**: Pickle tries to serialize function's module globals, encounters unpickleable thread-local object

### Evidence

**Test Results**:
```python
# Manual function (cupy_processor.sobel)
✅ Manual function pickles successfully

# Auto-discovered function (from registry)  
❌ Auto-discovered function failed: cannot pickle '_thread._local' object

# Raw external function (before decoration)
✅ Raw external function pickles successfully
```

**Key Finding**: The issue occurs specifically when external library functions are decorated **after** the TUI has started and thread-locals are active.

## Failed Solutions

### 1. Import Hook System (FAILED - BUT PREVIOUSLY WORKED!)
- **Approach**: Decorate functions at import time using import hooks
- **Implementation**: `_install_import_hook()` with `_decorating_import()`
- **Result**: Only decorated small subset of functions, lost comprehensive coverage
- **Issue**: We tried to use **full decorators** (@pyclesperanto) instead of **minimal attributes**

**CRITICAL DISCOVERY**: Commit `1a4eff8` had a **working import hook system** that used minimal decoration:
```python
# WORKING approach (commit 1a4eff8)
func.input_memory_type = "pyclesperanto"  # Just set attributes
func.output_memory_type = "pyclesperanto"  # No full decorator

# FAILED approach (our attempt)
decorated_func = pyclesperanto(func)  # Full decorator with thread-locals
```

The original system worked because it avoided full decoration and thread-local capture!

### 2. Clean Wrapper Creation (FAILED)
- **Approach**: Create functions with clean globals to avoid thread-local capture
- **Implementation**: `types.FunctionType()` with minimal globals
- **Result**: Functions lost all OpenHCS features (slice_by_slice, dtype_conversion)
- **Issue**: Workaround removed the benefits of decoration

### 3. Module Reference Changes (FAILED)
- **Approach**: Change `__module__` attribute to point to clean modules
- **Implementation**: `wrapper_func.__module__ = __name__`
- **Result**: Still failed to pickle
- **Issue**: Function's actual globals still contained thread-local references

## Current Workaround

### Manual Function Implementation
Added manual `sobel` function to `cupy_processor.py`:

```python
@cupy_func
def sobel(image, mask=None, *, axis=None, mode="reflect", cval=0.0):
    """Manual wrapper around CuCIM sobel - pickleable alternative."""
    return cucim_filters.sobel(image, mask=mask, axis=axis, mode=mode, cval=cval)
```

**Benefits**:
- ✅ Fully pickleable for subprocess execution
- ✅ All OpenHCS features (slice_by_slice, dtype_conversion, OOM recovery)
- ✅ Same signature as CuCIM version
- ✅ Works in TUI and generated scripts

**Limitations**:
- ❌ Only covers one function (sobel)
- ❌ Requires manual implementation for each needed function
- ❌ Doesn't solve the systematic auto-discovery issue

## Proposed Solutions

### Solution 1: Restore Working Import Hook System (RECOMMENDED)
**Concept**: Restore the working import hook system from commit `1a4eff8` with minimal decoration.

**Implementation**:
1. Restore `_install_import_hook()` with minimal attribute decoration (not full decorators)
2. Use `func.input_memory_type = "pyclesperanto"` instead of `@pyclesperanto(func)`
3. Expand to comprehensive function coverage (not just hardcoded lists)
4. Combine with existing registry system for full feature coverage

**Pros**:
- ✅ Known to work (was implemented before)
- ✅ Avoids thread-local capture completely
- ✅ Maintains pickleable functions
- ✅ Can be expanded to comprehensive coverage

**Cons**:
- ❌ Auto-discovered functions lose advanced OpenHCS features (slice_by_slice, dtype_conversion)
- ❌ Two-tier system (manual functions have more features than auto-discovered)

### Solution 2: Early Decoration System
**Concept**: Move auto-discovery decoration to happen at the same time as manual functions (module import time).

**Implementation**:
1. Create import-time decoration hooks that run before TUI starts
2. Modify existing registries to detect already-decorated functions
3. Preserve comprehensive function coverage

**Pros**: Maintains all existing functionality, fixes root cause
**Cons**: Complex implementation, requires careful timing coordination

### Solution 3: Lazy Decoration
**Concept**: Defer thread-local initialization until function execution, not decoration.

**Implementation**:
1. Modify decorators to avoid capturing thread-locals at decoration time
2. Initialize thread-locals only when functions are actually called
3. Use lazy initialization patterns in decorator implementation

**Pros**: Fixes root cause without changing registry system
**Cons**: Requires significant decorator refactoring

### Solution 4: Subprocess-Safe Serialization
**Concept**: Use function name references instead of function objects for subprocess communication.

**Implementation**:
1. Serialize pipeline data using function names/paths instead of objects
2. Reconstruct function objects in subprocess from registry
3. Avoid pickling decorated functions entirely

**Pros**: Avoids pickling issue completely
**Cons**: Requires changes to pipeline serialization system

## Registry Status

**Current Function Count**: 609 functions
- Manual functions: ~50 (pickleable)
- Auto-discovered functions: ~559 (not pickleable)

**Affected Libraries**:
- pyclesperanto: ~200+ functions
- scikit-image: ~300+ functions  
- CuCIM: ~50+ functions

## Testing Framework

### Pickle Test Script
Location: `test_pickle_investigation.py`

**Usage**:
```bash
python test_pickle_investigation.py
```

**Tests**:
1. Manual function pickling
2. Auto-discovered function pickling
3. Raw external function pickling
4. Function attribute comparison

### Registry Verification
```python
from openhcs.processing.func_registry import get_functions_by_memory_type
pycle_funcs = get_functions_by_memory_type('pyclesperanto')
print(f"Total pyclesperanto functions: {len(pycle_funcs)}")
```

## Next Steps

1. **Choose solution approach** (Early Decoration vs Lazy Decoration vs Serialization)
2. **Implement comprehensive fix** that maintains function coverage
3. **Test thoroughly** with subprocess execution and TUI code generation
4. **Add more manual functions** as interim workarounds for critical functions
5. **Document the final solution** for future reference

## Files Involved

**Core Files**:
- `openhcs/processing/func_registry.py` - Main registry system
- `openhcs/core/memory/decorators.py` - Decorator implementation with thread-locals
- `openhcs/processing/backends/processors/cupy_processor.py` - Manual functions

**Registry Files**:
- `openhcs/processing/backends/analysis/pyclesperanto_registry.py`
- `openhcs/processing/backends/analysis/scikit_image_registry.py`
- `openhcs/processing/backends/analysis/cupy_registry.py`

**Subprocess System**:
- `openhcs/textual_tui/widgets/plate_manager.py` - Where pickling fails
- `openhcs/textual_tui/subprocess_runner.py` - Subprocess execution

## Contact Context

This issue has been investigated extensively with focus on understanding the exact mechanism of thread-local capture during function decoration. The root cause is confirmed but a systematic solution is still needed.
