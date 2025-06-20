# Fact-Check Report: api/pipeline_orchestrator.rst

## File: `docs/source/api/pipeline_orchestrator.rst`
**Priority**: HIGH  
**Status**: 🟡 **ARCHITECTURAL EVOLUTION**  
**Accuracy**: 40% (Core class exists, API significantly evolved)

## Executive Summary
**EZStitcher → OpenHCS Evolution**: PipelineOrchestrator class exists with **revolutionary architectural improvements**. Constructor simplified, **two-phase execution** provides superior error handling and parallel safety. Core orchestration responsibilities preserved and enhanced with GPU-native processing and VFS integration.

## Section-by-Section Analysis

### Module Declaration (Lines 1-10)
```rst
PipelineOrchestrator
==================

.. module:: ezstitcher.core.pipeline_orchestrator
```
**Status**: ❌ **MODULE PATH OUTDATED**  
**Issue**: Module renamed ezstitcher → openhcs  
**✅ Current Reality**:
```python
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
```

### Class Declaration (Lines 11-30)

#### Constructor Signature (Lines 11)
```python
PipelineOrchestrator(plate_path=None, workspace_path=None, config=None, fs_manager=None, image_preprocessor=None, focus_analyzer=None)
```
**Status**: ❌ **SIGNATURE COMPLETELY CHANGED**  
**Issue**: All parameter names and types changed  
**✅ Current Reality**:
```python
PipelineOrchestrator(
    plate_path: Union[str, Path],           # Required, must be Path type
    workspace_path: Optional[Union[str, Path]] = None,  # ✅ Preserved
    *,
    global_config: Optional[GlobalPipelineConfig] = None,  # config → global_config
    storage_registry: Optional[Any] = None  # Replaces fs_manager
)
# fs_manager, image_preprocessor, focus_analyzer removed - handled internally
```

#### Parameter Documentation (Lines 19-30)
**Status**: ❌ **PARAMETERS CHANGED**  
**Issues**:
- `config` → `global_config` (different type)
- `fs_manager` → `storage_registry` (different concept)
- `image_preprocessor`, `focus_analyzer` → removed (handled internally)

**✅ Current Reality**: **Simplified constructor, more robust**
```python
from openhcs.core.config import get_default_global_config

global_config = get_default_global_config()
orchestrator = PipelineOrchestrator(
    plate_path=Path("/path/to/plate"),  # Required
    global_config=global_config         # Optional, auto-created if None
)
# Simpler interface, automatic component initialization
```

### Method Documentation (Lines 32-91)

#### run Method (Lines 32-41)
```python
run(plate_path=None, pipelines=None)
```
**Status**: ❌ **METHOD REPLACED BY TWO-PHASE SYSTEM**  
**Issue**: No single `run` method exists  
**✅ Current Reality**: **Revolutionary two-phase execution architecture**
```python
# Enhanced initialization
orchestrator = PipelineOrchestrator(plate_path=Path("/path/to/plate"))
orchestrator.initialize()

# Phase 1: Compile (early error detection + resource planning)
compiled_contexts = orchestrator.compile_pipelines(
    pipeline_definition=pipeline,  # Pipeline IS a List[AbstractStep]
    well_filter=["A01", "B01"],
    enable_visualizer_override=False
)

# Phase 2: Execute (stateless parallel processing)
results = orchestrator.execute_compiled_plate(
    pipeline_definition=pipeline,
    compiled_contexts=compiled_contexts,
    max_workers=4
)

# Revolutionary advantages over single run():
# - Early error detection during compilation
# - Frozen contexts enable safe parallel execution
# - GPU resource allocation optimized across wells
# - VFS-based data flow with automatic serialization
```

#### process_well Method (Lines 43-52)
```python
process_well(well, pipelines)
```
**Status**: ❌ **METHOD DOESN'T EXIST**  
**Issue**: No `process_well` method  
**✅ Current Reality**: **Handled by two-phase system**
```python
# Single well processing through compilation system
compiled_contexts = orchestrator.compile_pipelines(
    pipeline_definition=pipeline.steps,
    well_filter=["A01"]  # Single well
)
# More flexible than dedicated process_well method
```

#### detect_plate_structure Method (Lines 59-64)
```python
detect_plate_structure(plate_path)
```
**Status**: ✅ **CONCEPT PRESERVED**  
**✅ Current Reality**: **Integrated into initialization**
```python
orchestrator = PipelineOrchestrator(plate_path=Path("/path/to/plate"))
orchestrator.initialize()  # Automatically detects plate structure
# Same functionality, automatic integration
```

#### generate_positions Method (Lines 66-77)
```python
generate_positions(well, input_dir, positions_dir)
```
**Status**: ❌ **METHOD REPLACED BY PIPELINE SYSTEM**  
**Issue**: No direct method, handled by function-based pipelines  
**✅ Current Reality**: **More flexible pipeline-based approach**
```python
from openhcs.processing.backends.pos_gen.mist_gpu import generate_positions_mist_gpu

pos_step = FunctionStep(
    func=generate_positions_mist_gpu,
    variable_components=[VariableComponents.SITE],
    name="Position Generation"
)
# More flexible than direct method, GPU-accelerated
```

#### stitch_images Method (Lines 79-90)
```python
stitch_images(well, input_dir, output_dir, positions_path)
```
**Status**: ❌ **METHOD REPLACED BY PIPELINE SYSTEM**  
**Issue**: No direct method, handled by function-based pipelines  
**✅ Current Reality**: **More flexible pipeline-based approach**
```python
from openhcs.processing.backends.assemblers.simple_assembler import assemble_images

stitch_step = FunctionStep(
    func=assemble_images,
    variable_components=[VariableComponents.SITE],
    name="Image Stitching"
)
# More flexible than direct method, multiple assembler options
```

## Current Reality: Enhanced API

### Actual Constructor (Simplified)
```python
from pathlib import Path
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.config import get_default_global_config

# Simpler, more robust constructor
orchestrator = PipelineOrchestrator(
    plate_path=Path("/path/to/plate"),
    global_config=get_default_global_config()  # Optional
)
```

### Actual Methods (Enhanced)
```python
# Initialize (required)
orchestrator.initialize()

# Two-phase execution (more robust than run())
compiled_contexts = orchestrator.compile_pipelines(
    pipeline_definition=pipeline.steps,
    well_filter=["A01", "B01"]
)

results = orchestrator.execute_compiled_plate(
    pipeline_definition=pipeline.steps,
    compiled_contexts=compiled_contexts
)
```

### Properties (Preserved)
```python
workspace_path = orchestrator.workspace_path  # ✅ Works as documented
plate_path = orchestrator.plate_path          # ✅ Works as documented
```

## Impact Assessment

### User Experience Impact
- **API users**: **Constructor signature completely changed** - all examples fail
- **Method users**: **Execution model evolved** - run() method doesn't exist
- **Concept users**: **Core concepts preserved** - orchestrator still coordinates pipelines

### Severity: MEDIUM
**Core class exists and works**, but API has evolved significantly. **Two-phase execution is more robust** than documented single run() method.

## Recommendations

### Immediate Actions
1. **Update module path**: ezstitcher → openhcs
2. **Update constructor**: Document actual parameters and types
3. **Document two-phase execution**: More robust than single run() method

### Required Updates
1. **Constructor signature**: Update all parameters and types
2. **Method documentation**: Replace run() with compile_pipelines() + execute_compiled_plate()
3. **Remove deprecated methods**: process_well, generate_positions, stitch_images
4. **Add new methods**: initialize(), compile_pipelines(), execute_compiled_plate()

### Missing Enhancements
1. **Two-phase execution benefits**: Better error handling, parallel execution
2. **GPU scheduling**: Resource management capabilities
3. **Memory type system**: GPU-native processing integration
4. **VFS integration**: Enhanced data flow system

## Estimated Fix Effort
**Major API update required**: 12-16 hours to document current API accurately
