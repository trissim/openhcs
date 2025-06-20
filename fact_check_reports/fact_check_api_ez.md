# Fact-Check Report: api/ez.rst

## File: `docs/source/api/ez.rst`
**Priority**: HIGH  
**Status**: 🔴 **COMPLETELY NON-FUNCTIONAL**  
**Accuracy**: 0% (No EZ module exists, all APIs fail)

## Executive Summary
**EZStitcher → OpenHCS Evolution**: The entire EZ module API was **intentionally replaced** by superior interfaces. No `ezstitcher.ez` module exists because **TUI provides revolutionary improvement** for non-programmers and **two-phase orchestrator** provides more robust programmatic interface. This represents **architectural advancement**, not regression.

## Section-by-Section Analysis

### Module Declaration (Lines 1-6)
```rst
EZ Module
=========

.. module:: ezstitcher.ez

This module provides a simplified interface for stitching microscopy images with minimal code.
```
**Status**: ❌ **MODULE DOESN'T EXIST**  
**Issue**: No `ezstitcher.ez` module in current codebase  
**✅ Current Reality**: **TUI provides superior simplified interface**
```bash
# Superior non-programmer interface
python -m openhcs.textual_tui
```

### EZStitcher Class (Lines 8-46)

#### Class Declaration (Lines 11-32)
```python
EZStitcher(input_path, output_path=None, normalize=True, flatten_z=None, z_method="max", channel_weights=None, well_filter=None)
```
**Status**: ❌ **CLASS DOESN'T EXIST**  
**Issue**: No `EZStitcher` class in current codebase  
**✅ Current Reality**: **TUI + PipelineOrchestrator provide more powerful alternatives**

**TUI Alternative** (Better for non-programmers):
```bash
python -m openhcs.textual_tui
# Visual interface with all documented parameters:
# - normalize (visual toggle)
# - flatten_z (visual Z-stack options)  
# - z_method (dropdown: max/mean/focus)
# - channel_weights (visual channel editor)
# - well_filter (visual well selector)
```

**Programmatic Alternative** (Revolutionary improvement):
```python
from pathlib import Path
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.pipeline import Pipeline
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.memory.decorators import cupy_func

@cupy_func
def gpu_processing(image_stack):
    import cupy as cp
    return cp.max(image_stack, axis=0, keepdims=True)

# Two-phase execution (more robust than simple stitch_plate function)
pipeline = Pipeline(steps=[
    FunctionStep(func=gpu_processing, name="GPU Processing")
], name="Enhanced Pipeline")

orchestrator = PipelineOrchestrator(plate_path=Path("input_path"))
orchestrator.initialize()

# Phase 1: Compile (early error detection)
compiled_contexts = orchestrator.compile_pipelines(pipeline)

# Phase 2: Execute (parallel processing)
results = orchestrator.execute_compiled_plate(pipeline, compiled_contexts)

# Revolutionary advantages over simple stitch_plate():
# - GPU acceleration with memory type system
# - Early error detection during compilation
# - Parallel safety with frozen contexts
# - VFS multi-backend storage
```

#### Class Methods (Lines 33-45)

##### set_options Method (Lines 33-39)
```python
set_options(**kwargs)
```
**Status**: ❌ **METHOD DOESN'T EXIST**  
**✅ Current Reality**: **TUI provides real-time parameter editing**
- Visual parameter adjustment
- Immediate feedback
- No coding required

##### stitch Method (Lines 40-45)
```python
stitch()
```
**Status**: ❌ **METHOD DOESN'T EXIST**  
**✅ Current Reality**: **TUI provides one-click execution**
- Visual "Run Pipeline" button
- Real-time progress monitoring
- GPU-accelerated processing

### stitch_plate Function (Lines 47-61)

#### Function Declaration (Lines 50-60)
```python
stitch_plate(input_path, output_path=None, **kwargs)
```
**Status**: ❌ **FUNCTION DOESN'T EXIST**  
**Issue**: No `stitch_plate` function in current codebase  
**✅ Current Reality**: **TUI provides superior one-click interface**

**Documented Workflow**:
```python
from ezstitcher import stitch_plate  # ❌ FAILS
stitch_plate("path/to/plate")        # ❌ FAILS
```

**✅ Actual Workflow** (Superior):
```bash
# One command, visual interface, more powerful
python -m openhcs.textual_tui
# Then: Add Plate → Select Path → Configure → Run
# Features not in documented API:
# - Visual pipeline building
# - Real-time parameter editing
# - GPU memory type selection
# - Multi-backend storage options
# - Live progress monitoring
```

## Current Reality: What Actually Works

### No EZ Module - TUI is Superior
```bash
# The actual "EZ" interface - better than documented API
python -m openhcs.textual_tui

# Features beyond documented EZ module:
# ✅ Visual plate selection
# ✅ Real-time parameter editing  
# ✅ GPU acceleration options
# ✅ Multiple storage backends
# ✅ Live progress monitoring
# ✅ Error handling with visual feedback
# ✅ No coding required
```

### Programmatic Alternative (More Powerful)
```python
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.pipeline import Pipeline
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.processors.cupy_processor import stack_percentile_normalize
from openhcs.processing.backends.pos_gen.mist_gpu import generate_positions_mist_gpu

# More flexible than any documented EZ API
pipeline = Pipeline(steps=[
    FunctionStep(func=stack_percentile_normalize, name="Normalize"),
    FunctionStep(func=generate_positions_mist_gpu, name="Generate Positions")
], name="Custom Pipeline")

orchestrator = PipelineOrchestrator(plate_path=Path("input_path"))
# Full control + GPU acceleration + memory type system
```

## Impact Assessment

### User Experience Impact
- **Beginners**: **TUI provides superior experience** (visual vs. coding)
- **Documentation followers**: **100% failure rate** on all code examples
- **Quick start**: **Impossible via documented API**, but TUI is better

### Severity: CRITICAL
**Every single API call fails**. However, **TUI provides superior functionality** than documented EZ module.

## Recommendations

### Immediate Actions
1. **Add critical warning**: "⚠️ EZ MODULE DOESN'T EXIST - USE TUI INSTEAD"
2. **Document TUI**: The actual simplified interface for non-programmers
3. **Update all examples**: Replace with TUI workflow

### Required Complete Rewrite
This documentation cannot be fixed - it needs **complete replacement**:

1. **Remove EZ module documentation**: Non-existent API
2. **Document TUI interface**: The actual simplified interface
3. **Show TUI workflow**: Visual pipeline building process
4. **Document programmatic alternative**: For users who need coding interface

### Missing Critical Content
1. **TUI documentation**: The actual simplified interface
2. **Visual workflow**: How to use the TUI step-by-step
3. **GPU capabilities**: Memory type system not mentioned
4. **Multi-backend support**: Storage options not documented

## Estimated Fix Effort
**Complete replacement required**: 15-20 hours to create TUI-based documentation

**Recommendation**: Replace entire file with TUI documentation rather than trying to fix non-existent API.
