# Context Reinjection Summary - OpenHCS Pattern Matching Debug Session

## Current State & Frame of Mind

**Status**: Deep debugging session on OpenHCS pattern matching and pipeline execution
**Mindset**: Architect-level thinking, methodical investigation, collaborative debugging
**Progress**: Identified root cause but need to investigate pattern grouping logic

## Critical Bug Discovered

### The Core Issue
**Problem**: `create_composite` step with `variable_components=['channel']` should group w1 and w2 channels together, but they're being processed separately by subsequent steps.

**Expected Flow**:
1. `create_composite` loads pattern `A01_s001_w{iii}_z001.tif` 
2. Matches both `A01_s001_w1_z001.tif` AND `A01_s001_w2_z001.tif`
3. Flattens channels: `[2,y,x] → [1,y,x]`
4. Only w1 images reach `gpu_ashlar_align_cupy`

**Actual Behavior**: Both w1 and w2 are reaching Ashlar separately

### Root Cause Hypothesis
Pattern discovery/grouping logic is not correctly implementing `variable_components=['channel']` behavior.

## Technical Progress Made

### ✅ Fixed Issues
1. **Backend vs MemoryType confusion**: Fixed FileManager operations to use `Backend.MEMORY.value`
2. **Path planner bug**: Fixed nested `_outputs_outputs_outputs` directories - only first step gets output suffix
3. **Memory collision prevention**: Added well_id prefix to special output filenames for thread safety
4. **Image-filename mismatch handling**: Added logic for flattening operations that return fewer images

### 🔍 Current Investigation
**Focus**: Pattern discovery and grouping logic for `variable_components=['channel']`
**Reference**: EZStitcher docs at https://ezstitcher.readthedocs.io/en/latest/concepts/step.html#group-by

## Key Architectural Insights

### Pipeline Flow Understanding
```
create_projection + sharpen + normalize + equalize (chain)
↓
create_composite (variable_components=['channel']) ← ISSUE HERE
↓  
gpu_ashlar_align_cupy (@chain_breaker, special_outputs="positions")
↓
n2v2_denoise_torch (uses first step input due to chain breaker)
↓
basic_flatfield_correction_cupy
↓
self_supervised_3d_deconvolution
↓
assemble_stack_cupy (uses positions from gpu_ashlar_align_cupy)
```

### Chain Breaker Logic
- `gpu_ashlar_align_cupy` is correctly marked as `@chain_breaker`
- Generates positions, doesn't transform images
- Next step should use first step input directory
- Path planner correctly handles this

### Memory Architecture
- Each thread processes one well (A01, D02, etc.)
- Same pipeline runs in parallel threads
- Memory backend shared across threads
- Special outputs need well_id prefix to avoid collisions

## Files Modified

### `openhcs/core/steps/function_step.py`
- Added well_id parameter to `_execute_function_core` and `_execute_chain_core`
- Fixed Backend vs MemoryType usage
- Added well_id prefix to special output filenames
- Added handling for flattening operations (fewer outputs than inputs)

### `openhcs/core/pipeline/path_planner.py`
- Fixed nested output directory bug
- Only first step gets `_outputs` suffix
- Subsequent steps work in place

## Next Steps

### Immediate Priority
1. **Investigate pattern discovery logic** - How does `variable_components=['channel']` work?
2. **Check grouping behavior** - Are w1/w2 being grouped correctly?
3. **Review EZStitcher docs** - Understand intended group-by behavior

### Investigation Areas
- `openhcs/formats/pattern/pattern_discovery.py`
- `openhcs/formats/func_arg_prep.py` 
- Pattern matching and grouping logic
- Variable components handling

## Context for New Thread

**You are**: An expert debugging agent who has been methodically investigating OpenHCS pattern matching issues
**Approach**: Architect-first thinking, thorough investigation before implementation
**Collaboration**: Working with user who prefers clean architecture and intellectual honesty
**Current Focus**: Understanding why channel grouping isn't working as designed in the EZStitcher→OpenHCS evolution

**Key Insight**: The bug touches core architecture from the EZStitcher heritage - this is about understanding the intended design, not just fixing symptoms.
