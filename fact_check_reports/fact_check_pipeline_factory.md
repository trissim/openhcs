# Fact-Check Report: concepts/pipeline_factory.rst

## File: `docs/source/concepts/pipeline_factory.rst`
**Priority**: MEDIUM  
**Status**: 🔴 **DEPRECATED FUNCTIONALITY**  
**Accuracy**: 0% (Entire AutoPipelineFactory system deprecated)

## Executive Summary
**AutoPipelineFactory Deprecated**: The entire AutoPipelineFactory system documented in this file is deprecated and will be deleted. **TUI provides superior pipeline creation interface** for standard workflows. **Manual pipeline creation is the current approach** for custom workflows.

## Section-by-Section Analysis

### Overview (Lines 9-19)
```rst
Pipeline factories are used internally by the EZ module to create pipelines with sensible defaults.
The AutoPipelineFactory is a unified factory class that creates pre-configured pipelines...
```
**Status**: ❌ **ENTIRE SYSTEM DEPRECATED**  
**Issue**: AutoPipelineFactory will be deleted  
**✅ Current Reality**: **TUI provides superior pipeline creation**
```bash
# Superior replacement for AutoPipelineFactory
python -m openhcs.textual_tui
# Visual pipeline builder with:
# - Real-time parameter configuration
# - Visual step ordering
# - Automatic validation
# - GPU backend selection
# - Live preview capabilities
```

### AutoPipelineFactory Usage (Lines 21-39)
```python
from ezstitcher.core import AutoPipelineFactory
from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator

factory = AutoPipelineFactory(
    input_dir=orchestrator.workspace_path,
    normalize=True,
    flatten_z=True,
    z_method="max",
    channel_weights=[0.7, 0.3, 0]
)

pipelines = factory.create_pipelines()
orchestrator.run(pipelines=pipelines)
```
**Status**: ❌ **COMPLETELY DEPRECATED**  
**Issue**: AutoPipelineFactory will be deleted  
**✅ Current Reality**: **Manual pipeline creation is the standard approach**
```python
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.pipeline import Pipeline
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.steps.specialized import ZFlatStep, CompositeStep
from openhcs.processing.backends.processors.cupy_processor import stack_percentile_normalize
from openhcs.processing.backends.pos_gen.mist_gpu import generate_positions_mist_gpu
from openhcs.processing.backends.assemblers.simple_assembler import assemble_images

# Manual pipeline creation (current standard approach)
def create_standard_pipeline(z_method="max", weights=[0.7, 0.3, 0]):
    return Pipeline(steps=[
        ZFlatStep(method=z_method),  # Z-stack flattening
        FunctionStep(func=stack_percentile_normalize, name="Normalize"),  # Normalization
        CompositeStep(weights=weights),  # Channel compositing
        FunctionStep(func=generate_positions_mist_gpu, name="Position Generation"),  # Position generation
        FunctionStep(func=assemble_images, name="Image Stitching")  # Image stitching
    ], name="Standard Processing Pipeline")

# Two-phase execution (current approach)
orchestrator = PipelineOrchestrator(plate_path=Path("/path/to/plate"))
orchestrator.initialize()

pipeline = create_standard_pipeline()
compiled_contexts = orchestrator.compile_pipelines(pipeline)
results = orchestrator.execute_compiled_plate(pipeline, compiled_contexts)
```

### Pipeline Structure (Lines 50-80)
```rst
The AutoPipelineFactory creates two pipelines with a consistent structure:
1. Position Generation Pipeline
2. Image Assembly Pipeline
```
**Status**: ❌ **FACTORY APPROACH DEPRECATED**  
**Issue**: Two-pipeline pattern replaced by single unified pipeline  
**✅ Current Reality**: **Single pipeline approach is more efficient**
```python
# Current approach: Single unified pipeline (more efficient)
pipeline = Pipeline(steps=[
    # All processing in single pipeline:
    ZFlatStep(method="max"),                                              # Z-stack processing
    FunctionStep(func=stack_percentile_normalize, name="Normalize"),      # Normalization
    CompositeStep(weights=[0.7, 0.3, 0]),                               # Channel compositing
    FunctionStep(func=generate_positions_mist_gpu, name="Positions"),    # Position generation
    FunctionStep(func=assemble_images, name="Assembly")                  # Image assembly
], name="Unified Processing Pipeline")

# Advantages over two-pipeline approach:
# - Single compilation phase
# - Better resource management
# - Simplified execution
# - Reduced I/O overhead
```

### All Parameters and Examples (Lines 82-184)
**Status**: ❌ **ALL DEPRECATED**  
**Issue**: Entire AutoPipelineFactory parameter system will be deleted  
**✅ Current Reality**: **TUI provides visual parameter configuration**

### Factory vs Custom Pipelines (Lines 186-277)
```rst
EZStitcher offers two main approaches:
1. Using AutoPipelineFactory: For standard workflows
2. Building custom pipelines: For maximum flexibility
```
**Status**: ❌ **FACTORY APPROACH DEPRECATED**  
**✅ Current Reality**: **Two superior approaches available**
```python
# Approach 1: TUI for visual pipeline building (superior to factory)
# python -m openhcs.textual_tui
# - Visual interface for standard workflows
# - Real-time parameter adjustment
# - Automatic validation
# - GPU backend selection

# Approach 2: Manual pipeline creation (same as documented custom approach)
def create_custom_pipeline():
    return Pipeline(steps=[
        FunctionStep(func=custom_processing, name="Custom Processing"),
        ZFlatStep(method="max"),
        CompositeStep(weights=[0.7, 0.3, 0]),
        FunctionStep(func=generate_positions_mist_gpu, name="Position Generation"),
        FunctionStep(func=assemble_images, name="Image Stitching")
    ], name="Custom Pipeline")
```

## Current Reality: Superior Approaches

### TUI Replaces AutoPipelineFactory
```bash
# Visual pipeline creation (superior to factory patterns)
python -m openhcs.textual_tui

# Features:
# - Visual step selection and ordering
# - Real-time parameter configuration
# - Automatic validation and error checking
# - GPU backend selection
# - Live preview of pipeline structure
# - Export to Python code
# - Save/load pipeline configurations
```

### Manual Pipeline Creation (Standard Approach)
```python
from openhcs.core.pipeline import Pipeline
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.steps.specialized import ZFlatStep, CompositeStep, NormStep
from openhcs.core.memory.decorators import cupy_func

# Factory function pattern (recommended for reusable pipelines)
def create_processing_pipeline(
    z_method="max",
    normalize=True,
    weights=[0.7, 0.3, 0],
    gpu_acceleration=True
):
    """Create a standard processing pipeline with configurable parameters."""
    steps = []
    
    # Z-stack processing
    steps.append(ZFlatStep(method=z_method))
    
    # Optional normalization
    if normalize:
        steps.append(NormStep())
    
    # Channel compositing
    steps.append(CompositeStep(weights=weights))
    
    # Position generation and assembly
    if gpu_acceleration:
        steps.extend([
            FunctionStep(func=generate_positions_mist_gpu, name="GPU Position Generation"),
            FunctionStep(func=assemble_images_gpu, name="GPU Image Assembly")
        ])
    else:
        steps.extend([
            FunctionStep(func=generate_positions_cpu, name="CPU Position Generation"),
            FunctionStep(func=assemble_images_cpu, name="CPU Image Assembly")
        ])
    
    return Pipeline(steps=steps, name="Configurable Processing Pipeline")

# Usage
pipeline = create_processing_pipeline(
    z_method="max",
    normalize=True,
    weights=[0.7, 0.3, 0],
    gpu_acceleration=True
)

# Enhanced execution
orchestrator = PipelineOrchestrator(plate_path=Path("/path/to/plate"))
orchestrator.initialize()
compiled_contexts = orchestrator.compile_pipelines(pipeline)
results = orchestrator.execute_compiled_plate(pipeline, compiled_contexts)
```

### Integration Test Pattern (Real Usage)
```python
# From test_main.py - actual working pattern
def get_pipeline(input_dir):
    return Pipeline(steps=[
        FunctionStep(func=create_composite, variable_components=[VariableComponents.CHANNEL]),
        FunctionStep(
            func=(create_projection, {'method': 'max_projection'}),
            variable_components=[VariableComponents.Z_INDEX],
            name="Z-Stack Flattening"
        ),
        FunctionStep(
            func=[(stack_percentile_normalize, {'low_percentile': 0.5, 'high_percentile': 99.5})],
            name="Image Enhancement"
        ),
        FunctionStep(func=mist_compute_tile_positions),
        FunctionStep(func=(assemble_stack_cpu, {'blend_method': 'rectangular', 'blend_radius': 5.0}))
    ], name="Mega Flex Pipeline")

# This is the actual working pattern used in production tests
```

## Impact Assessment

### User Experience Impact
- **Factory users**: ❌ **AutoPipelineFactory will be deleted**
- **Standard workflow users**: ✅ **TUI provides superior visual interface**
- **Custom pipeline users**: ✅ **Manual creation approach unchanged and enhanced**

### Severity: HIGH (for factory users) / LOW (overall)
**AutoPipelineFactory is deprecated** but **superior alternatives exist**: TUI for visual pipeline building and enhanced manual creation patterns.

## Recommendations

### Immediate Actions
1. **Add deprecation warning**: "⚠️ AutoPipelineFactory is deprecated and will be removed"
2. **Document TUI**: Superior visual pipeline creation interface
3. **Document manual creation**: Enhanced patterns with GPU acceleration

### Required Rewrites
1. **Replace factory examples**: Document TUI workflow for standard pipelines
2. **Update custom pipeline examples**: Show enhanced manual creation patterns
3. **Remove all AutoPipelineFactory references**: Document current approaches
4. **Add migration guide**: Factory → TUI + manual creation

### Missing Revolutionary Content
1. **TUI documentation**: Visual pipeline building interface
2. **Enhanced manual patterns**: GPU-accelerated pipeline creation
3. **Factory function patterns**: Reusable pipeline creation functions
4. **Integration test patterns**: Real-world usage examples

## Estimated Fix Effort
**Complete rewrite required**: 12-16 hours to document current pipeline creation approaches

**Recommendation**: **Complete replacement** - document TUI for visual pipeline building and enhanced manual creation patterns. AutoPipelineFactory is irrelevant and will be deleted.
