# Fact-Check Report: api/pipeline_factory.rst

## File: `docs/source/api/pipeline_factory.rst`
**Priority**: MEDIUM  
**Status**: 🔴 **DEPRECATED FUNCTIONALITY**  
**Accuracy**: 0% (Entire AutoPipelineFactory system deprecated)

## Executive Summary
**AutoPipelineFactory Deprecated**: The entire AutoPipelineFactory API documented in this file is deprecated and will be deleted. **TUI provides superior pipeline creation interface** for standard workflows. **Manual pipeline creation is the current approach** for custom workflows.

## Section-by-Section Analysis

### Module Documentation (Lines 4-7)
```rst
.. module:: ezstitcher.core

This module contains the AutoPipelineFactory class that creates pre-configured pipelines
for all common workflows, leveraging specialized steps to reduce boilerplate code.
```
**Status**: ❌ **ENTIRE MODULE DEPRECATED**  
**Issue**: AutoPipelineFactory will be deleted  
**✅ Current Reality**: **TUI provides superior pipeline creation**
```bash
# Superior replacement for AutoPipelineFactory API
python -m openhcs.textual_tui
# Visual pipeline builder with:
# - Real-time parameter configuration
# - Visual step ordering and selection
# - Automatic validation and error checking
# - GPU backend selection
# - Live preview capabilities
# - Export to Python code
```

### Cross-References (Lines 9-10)
```rst
For conceptual explanation, see :doc:`../concepts/pipeline_factory`.
For information about pipeline configuration, see :doc:`../concepts/pipeline`.
```
**Status**: ❌ **FIRST REFERENCE DEPRECATED**  
**Issue**: concepts/pipeline_factory.rst documents deprecated functionality  
**✅ Current Reality**: **Pipeline configuration reference remains valid**
- `:doc:`../concepts/pipeline_factory`` → **Deprecated, will be removed**
- `:doc:`../concepts/pipeline`` → **Valid, documents current Pipeline class**

### AutoPipelineFactory Class (Lines 12-21)
```rst
AutoPipelineFactory
-----------------

.. autoclass:: AutoPipelineFactory
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: create_pipelines
```
**Status**: ❌ **CLASS COMPLETELY DEPRECATED**  
**Issue**: AutoPipelineFactory class will be deleted  
**✅ Current Reality**: **Manual pipeline creation and TUI are the current approaches**

## Current Reality: Superior Approaches

### TUI Replaces AutoPipelineFactory API
```bash
# Visual pipeline creation (superior to programmatic factory)
python -m openhcs.textual_tui

# Features that replace AutoPipelineFactory:
# - Visual step selection and ordering
# - Real-time parameter configuration
# - Automatic validation and error checking
# - GPU backend selection
# - Live preview of pipeline structure
# - Export to Python code
# - Save/load pipeline configurations
# - No programming required for standard workflows
```

### Manual Pipeline Creation API (Current Standard)
```python
from openhcs.core.pipeline import Pipeline
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.steps.specialized import ZFlatStep, CompositeStep, NormStep
from openhcs.core.memory.decorators import cupy_func
from openhcs.processing.backends.processors.cupy_processor import stack_percentile_normalize
from openhcs.processing.backends.pos_gen.mist_gpu import generate_positions_mist_gpu
from openhcs.processing.backends.assemblers.simple_assembler import assemble_images

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
            FunctionStep(func=assemble_images, name="Image Assembly")
        ])
    else:
        steps.extend([
            FunctionStep(func=generate_positions_cpu, name="CPU Position Generation"),
            FunctionStep(func=assemble_images_cpu, name="CPU Image Assembly")
        ])
    
    return Pipeline(steps=steps, name="Configurable Processing Pipeline")

# Usage (replaces AutoPipelineFactory.create_pipelines())
pipeline = create_processing_pipeline(
    z_method="max",
    normalize=True,
    weights=[0.7, 0.3, 0],
    gpu_acceleration=True
)

# Enhanced execution
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
orchestrator = PipelineOrchestrator(plate_path=Path("/path/to/plate"))
orchestrator.initialize()
compiled_contexts = orchestrator.compile_pipelines(pipeline)
results = orchestrator.execute_compiled_plate(pipeline, compiled_contexts)
```

### Integration Test Pattern (Real Usage)
```python
# From test_main.py - actual working pipeline creation pattern
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
# More flexible than AutoPipelineFactory approach
```

## Impact Assessment

### User Experience Impact
- **AutoPipelineFactory users**: ❌ **API will be completely removed**
- **Standard workflow users**: ✅ **TUI provides superior visual interface**
- **Programmatic users**: ✅ **Manual creation patterns are more flexible**

### Severity: HIGH (for factory users) / LOW (overall)
**AutoPipelineFactory API is deprecated** but **superior alternatives exist**: TUI for visual pipeline building and enhanced manual creation patterns.

## Recommendations

### Immediate Actions
1. **Add deprecation warning**: "⚠️ AutoPipelineFactory is deprecated and will be removed"
2. **Document TUI**: Superior visual pipeline creation interface
3. **Document manual creation**: Enhanced patterns with GPU acceleration

### Required Rewrites
1. **Replace factory API**: Document TUI workflow for standard pipelines
2. **Update manual creation examples**: Show enhanced patterns with GPU acceleration
3. **Remove all AutoPipelineFactory references**: Document current approaches
4. **Add migration guide**: Factory API → TUI + manual creation

### Missing Revolutionary Content
1. **TUI API documentation**: Visual pipeline building interface
2. **Enhanced manual patterns**: GPU-accelerated pipeline creation
3. **Factory function patterns**: Reusable pipeline creation functions
4. **Integration test patterns**: Real-world usage examples

## Estimated Fix Effort
**Complete rewrite required**: 8-12 hours to document current pipeline creation approaches

**Recommendation**: **Complete replacement** - document TUI for visual pipeline building and enhanced manual creation patterns. AutoPipelineFactory API is irrelevant and will be deleted.

---

**Note**: This fact-check was completed as part of the systematic medium priority files review. The AutoPipelineFactory represents deprecated functionality that has been replaced by superior TUI and manual creation approaches.
