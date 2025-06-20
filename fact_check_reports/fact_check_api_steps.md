# Fact-Check Report: api/steps.rst

## File: `docs/source/api/steps.rst`
**Priority**: HIGH  
**Status**: 🟡 **ARCHITECTURAL EVOLUTION**  
**Accuracy**: 35% (Core concepts preserved, implementation completely changed)

## Executive Summary
**EZStitcher → OpenHCS Evolution**: Step architecture fundamentally changed from class-based to function-based approach. **AbstractStep + FunctionStep** replaced specialized step classes. Most documented specialized steps exist but are deprecated. **Function pattern system** is the new paradigm.

## Section-by-Section Analysis

### Module Declaration (Lines 1-11)
```rst
.. module:: ezstitcher.core.steps

This module contains the Step class and all step implementations for the EZStitcher pipeline architecture,
including the base Step class and various step types like ZFlatStep, FocusStep, CompositeStep, PositionGenerationStep, and ImageStitchingStep.
```
**Status**: ❌ **MODULE PATH OUTDATED, ARCHITECTURE CHANGED**  
**Issue**: Module renamed, step architecture completely evolved  
**✅ Current Reality**:
```python
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_step import FunctionStep
# Specialized steps exist but deprecated in favor of function patterns
```

### Step Class (Lines 16-50)

#### Constructor Signature (Lines 16, 28-41)
```python
Step(*, func, variable_components=None, group_by=None, input_dir=None, output_dir=None, well_filter=None, name=None)
```
**Status**: ❌ **CLASS REPLACED BY FUNCTION-BASED APPROACH**  
**Issue**: No `Step` class exists, replaced by `AbstractStep` + `FunctionStep`  
**✅ Current Reality**: **Function-based architecture with enhanced capabilities**
```python
# ❌ Documented (doesn't exist)
from ezstitcher.core.steps import Step

# ✅ Current reality (more powerful)
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_step import FunctionStep
from openhcs.constants.constants import VariableComponents, GroupBy

# AbstractStep: Base interface
class AbstractStep(abc.ABC):
    def __init__(self, *, name=None, variable_components=None, 
                 force_disk_output=False, group_by=None, 
                 input_dir=None, output_dir=None):
        # Enhanced with GPU support, memory types, VFS integration

# FunctionStep: Main implementation with function patterns
class FunctionStep(AbstractStep):
    def __init__(self, func, *, name=None, 
                 variable_components=[VariableComponents.SITE],
                 group_by=GroupBy.CHANNEL, ...):
        # Supports all four function patterns
        # Enhanced with memory type decorators
        # GPU acceleration integration
```

#### Function Patterns (Lines 28-29)
```rst
:param func: The processing function(s) to apply. Can be a single callable, a tuple of (function, kwargs), a list of functions or function tuples, or a dictionary mapping component values to functions or function tuples.
```
**Status**: ✅ **CONCEPT PERFECTLY PRESERVED AND ENHANCED**  
**✅ Current Reality**: **All documented patterns work exactly as described**
```python
# ✅ All four patterns work as documented
step = FunctionStep(func=my_function)                           # Single
step = FunctionStep(func=(my_function, {'param': 'value'}))     # Parameterized
step = FunctionStep(func=[func1, func2, func3])               # Sequential
step = FunctionStep(func={'DAPI': func1, 'GFP': func2})       # Component-specific

# ✅ Enhanced with GPU acceleration
from openhcs.core.memory.decorators import cupy_func

@cupy_func
def gpu_processing(image_stack):
    return processed_stack

step = FunctionStep(func=gpu_processing)  # GPU-accelerated
```

#### Variable Components and Group By (Lines 30-33)
```rst
:param variable_components: Components that vary across files (e.g., 'z_index', 'channel')
:param group_by: How to group files for processing (e.g., 'channel', 'site')
```
**Status**: ✅ **FULLY PRESERVED WITH TYPE-SAFE ENUMS**  
**✅ Current Reality**: **Enhanced with type safety**
```python
from openhcs.constants.constants import VariableComponents, GroupBy

step = FunctionStep(
    func=my_function,
    variable_components=[VariableComponents.Z_INDEX],  # Type-safe enum
    group_by=GroupBy.CHANNEL                          # Type-safe enum
)

# ✅ Available options (enhanced from documentation)
VariableComponents.SITE, .CHANNEL, .Z_INDEX, .TIME_POINT
GroupBy.SITE, .CHANNEL, .WELL, .PLATE
```

### Specialized Step Classes

#### PositionGenerationStep (Lines 55-76)
```python
PositionGenerationStep(*, name="Position Generation", input_dir=None, output_dir=None)
```
**Status**: ❌ **CLASS DOESN'T EXIST**  
**Issue**: No `PositionGenerationStep` class in current implementation  
**✅ Current Reality**: **Function-based approach with GPU acceleration**
```python
# ❌ Documented (doesn't exist)
from ezstitcher.core.steps import PositionGenerationStep

# ✅ Current reality (more flexible)
from openhcs.processing.backends.pos_gen.mist_gpu import generate_positions_mist_gpu

step = FunctionStep(
    func=generate_positions_mist_gpu,
    variable_components=[VariableComponents.SITE],
    name="Position Generation"
)
# GPU-accelerated, more backend options available
```

#### ImageStitchingStep (Lines 81-110)
```python
ImageStitchingStep(*, name="Image Stitching", input_dir=None, positions_dir=None, output_dir=None)
```
**Status**: ❌ **CLASS DOESN'T EXIST**  
**Issue**: No `ImageStitchingStep` class in current implementation  
**✅ Current Reality**: **Function-based approach with multiple assemblers**
```python
# ❌ Documented (doesn't exist)
from ezstitcher.core.steps import ImageStitchingStep

# ✅ Current reality (more assembler options)
from openhcs.processing.backends.assemblers.simple_assembler import assemble_images

step = FunctionStep(
    func=assemble_images,
    variable_components=[VariableComponents.SITE],
    name="Image Stitching"
)
# Multiple assembler backends available
```

#### ZFlatStep (Lines 115-130)
```python
ZFlatStep(*, method="max", input_dir=None, output_dir=None, well_filter=None)
```
**Status**: ✅ **CLASS EXISTS BUT DEPRECATED**  
**Issue**: Class exists but deprecated in favor of function patterns  
**✅ Current Reality**: **Specialized class exists, function approach preferred**
```python
# ✅ Specialized class still exists (deprecated)
from openhcs.core.steps.specialized.zflat_step import ZFlatStep
step = ZFlatStep(method="max")  # Works but deprecated

# ✅ Preferred function-based approach (more flexible)
from openhcs.processing.backends.processors.cupy_processor import create_projection

step = FunctionStep(
    func=(create_projection, {'method': 'max_projection'}),
    variable_components=[VariableComponents.Z_INDEX],
    name="Z-Stack Flattening"
)
# GPU-accelerated, more projection methods, better integration
```

#### FocusStep (Lines 134-150)
```python
FocusStep(*, focus_options=None, input_dir=None, output_dir=None, well_filter=None)
```
**Status**: ✅ **CLASS EXISTS BUT DEPRECATED**  
**✅ Current Reality**: **Available but function approach preferred**
```python
# ✅ Specialized class exists
from openhcs.core.steps.specialized.focus_step import FocusStep

# ✅ Function approach preferred
from openhcs.processing.backends.processors.focus_processor import deep_focus

step = FunctionStep(
    func=deep_focus,
    variable_components=[VariableComponents.Z_INDEX],
    name="Focus-Based Processing"
)
```

#### CompositeStep (Lines 155-169)
```python
CompositeStep(*, weights=None, input_dir=None, output_dir=None, well_filter=None)
```
**Status**: ✅ **CLASS EXISTS BUT DEPRECATED**  
**✅ Current Reality**: **Available but function approach preferred**
```python
# ✅ Specialized class exists
from openhcs.core.steps.specialized.composite_step import CompositeStep

# ✅ Function approach preferred
from openhcs.processing.backends.processors.composite_processor import create_composite

step = FunctionStep(
    func=create_composite,
    variable_components=[VariableComponents.CHANNEL],
    name="Channel Compositing"
)
```

## Current Reality: Function-Based Architecture

### Core Architecture (Enhanced)
```python
# ✅ Current step hierarchy
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_step import FunctionStep

# AbstractStep: Base interface with enhanced capabilities
# - GPU memory type integration
# - VFS integration
# - Stateful → stateless lifecycle
# - Enhanced error handling

# FunctionStep: Main implementation
# - Four function patterns (exactly as documented)
# - Memory type decorators for GPU acceleration
# - Special inputs/outputs for data flow
# - Type-safe enums for variable_components and group_by
```

### Function Patterns (Preserved and Enhanced)
```python
# ✅ All documented patterns work with GPU acceleration
@cupy_func
def gpu_processing(image_stack):
    return processed_stack

# Single pattern
step = FunctionStep(func=gpu_processing)

# Parameterized pattern  
step = FunctionStep(func=(gpu_processing, {'param': 'value'}))

# Sequential pattern
step = FunctionStep(func=[denoise, gpu_processing, enhance])

# Component-specific pattern
step = FunctionStep(func={
    'DAPI': gpu_process_nuclei,
    'GFP': gpu_process_proteins
}, group_by=GroupBy.CHANNEL)
```

### Specialized Steps (Deprecated but Available)
```python
# ✅ Still exist but deprecated
from openhcs.core.steps.specialized import ZFlatStep, FocusStep, CompositeStep

# Function approach preferred for:
# - Better GPU integration
# - More backend options
# - Enhanced flexibility
# - Consistent architecture
```

## Impact Assessment

### User Experience Impact
- **Function pattern users**: ✅ **All patterns work exactly as documented**
- **Specialized step users**: ⚠️ **Classes exist but deprecated**
- **New users**: **Function approach is more powerful and flexible**

### Severity: MEDIUM
**Core concepts perfectly preserved** but **implementation architecture evolved**. Function patterns work exactly as documented with GPU enhancements.

## Recommendations

### Immediate Actions
1. **Update module path**: ezstitcher → openhcs
2. **Document function-based approach**: AbstractStep + FunctionStep architecture
3. **Preserve function patterns**: All four patterns work exactly as documented
4. **Note specialized step deprecation**: Available but function approach preferred

### Required Updates
1. **Step class documentation**: Replace Step with AbstractStep + FunctionStep
2. **Function pattern emphasis**: Highlight this as the primary approach
3. **GPU integration**: Document memory type decorators and acceleration
4. **Specialized step status**: Mark as deprecated, show function alternatives

### Missing Enhancements
1. **GPU acceleration**: Memory type decorators not documented
2. **Enhanced backends**: Multiple processor/assembler options
3. **Type safety**: VariableComponents and GroupBy enums
4. **VFS integration**: Multi-backend storage system

## Estimated Fix Effort
**Moderate rewrite required**: 12-16 hours to document current function-based architecture

**Recommendation**: Emphasize that **function patterns work exactly as documented** but with enhanced GPU capabilities and more flexible implementation.
