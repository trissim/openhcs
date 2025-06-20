# Fact-Check Report: api/pipeline.rst

## File: `docs/source/api/pipeline.rst`
**Priority**: HIGH  
**Status**: 🟡 **ARCHITECTURAL EVOLUTION**  
**Accuracy**: 45% (Core concepts preserved, API significantly evolved)

## Executive Summary
**EZStitcher → OpenHCS Evolution**: Pipeline class exists but with fundamentally different architecture. **Pipeline IS a List** (inherits from `list`), not just contains steps. Constructor signature evolved, `run()` method replaced by two-phase execution. ProcessingContext exists but with different interface.

## Section-by-Section Analysis

### Module Declaration (Lines 1-10)
```rst
.. module:: ezstitcher.core.pipeline

This module contains the Pipeline class and related components for the EZStitcher pipeline architecture.
```
**Status**: ❌ **MODULE PATH OUTDATED**  
**Issue**: Module renamed ezstitcher → openhcs  
**✅ Current Reality**:
```python
from openhcs.core.pipeline import Pipeline
```

### Pipeline Class Declaration (Lines 14-24)

#### Constructor Signature (Lines 14, 21-24)
```python
Pipeline(steps=None, name=None)
```
**Status**: ❌ **SIGNATURE INCOMPLETE**  
**Issue**: Missing parameters, different architecture  
**✅ Current Reality**: **Pipeline IS a List with enhanced metadata**
```python
class Pipeline(list):  # ✅ Inherits from list!
    def __init__(self, steps=None, *, name=None, metadata=None, description=None):
        super().__init__(steps or [])  # ✅ IS a list of steps
        self.name = name or f"Pipeline_{id(self)}"
        self.description = description
        self.metadata = metadata or {}
        # Automatic timestamp for debugging
        self.metadata.setdefault('created_at', datetime.now().isoformat())

# ✅ Backward compatibility preserved
pipeline.steps  # Returns self (Pipeline IS the list)
```

### Pipeline Methods

#### add_step Method (Lines 26-34)
```python
add_step(step) -> Pipeline
```
**Status**: ✅ **METHOD EXISTS AND WORKS**  
**✅ Current Reality**: **Exactly as documented**
```python
def add_step(self, step):
    """Add a step to the pipeline and return self for method chaining."""
    self.append(step)  # Uses list.append since Pipeline IS a list
    return self  # ✅ Method chaining works
```

#### run Method (Lines 36-60)
```python
run(input_dir=None, output_dir=None, well_filter=None, microscope_handler=None, orchestrator=None, positions_file=None)
```
**Status**: ❌ **METHOD REPLACED BY TWO-PHASE SYSTEM**  
**Issue**: No `run()` method exists on Pipeline  
**✅ Current Reality**: **Two-phase execution through PipelineOrchestrator**
```python
# ❌ Documented (doesn't exist)
pipeline.run(input_dir="...", output_dir="...")

# ✅ Current reality (more robust)
orchestrator = PipelineOrchestrator(plate_path=Path("/path/to/plate"))
orchestrator.initialize()

# Phase 1: Compile
compiled_contexts = orchestrator.compile_pipelines(
    pipeline_definition=pipeline,  # Pipeline IS a list of steps
    well_filter=["A01", "B01"]
)

# Phase 2: Execute
results = orchestrator.execute_compiled_plate(
    pipeline_definition=pipeline,
    compiled_contexts=compiled_contexts
)
```

#### Pipeline Attributes (Lines 62-72)
```python
input_dir: Path or None
output_dir: Path or None
```
**Status**: ❌ **ATTRIBUTES DON'T EXIST**  
**Issue**: Pipeline doesn't have input_dir/output_dir attributes  
**✅ Current Reality**: **Metadata-focused design**
```python
# ❌ Documented (don't exist)
pipeline.input_dir
pipeline.output_dir

# ✅ Current reality
pipeline.name          # Human-readable name
pipeline.description    # Optional description
pipeline.metadata       # Dictionary with timestamps, etc.
len(pipeline)          # Number of steps (Pipeline IS a list)
pipeline[0]            # First step (list indexing)
pipeline.append(step)  # Add step (list method)
```

### ProcessingContext Class (Lines 77-126)

#### Constructor Signature (Lines 77, 88-96)
```python
ProcessingContext(input_dir=None, output_dir=None, well_filter=None, config=None, **kwargs)
```
**Status**: ❌ **SIGNATURE COMPLETELY CHANGED**  
**Issue**: Constructor requires GlobalPipelineConfig, different parameters  
**✅ Current Reality**: **Enhanced with global configuration**
```python
class ProcessingContext:
    def __init__(
        self,
        global_config: GlobalPipelineConfig,  # Required parameter
        step_plans: Optional[Dict[str, Dict[str, Any]]] = None,
        well_id: Optional[str] = None,
        **kwargs  # filemanager, microscope_handler, etc.
    ):
        self.step_plans = step_plans or {}  # ✅ Core execution plans
        self.well_id = well_id              # ✅ Well identifier
        self.global_config = global_config  # ✅ System configuration
        self.filemanager = None             # ✅ VFS access
        # Additional attributes from kwargs
```

#### ProcessingContext Attributes (Lines 98-126)
**Status**: ❌ **ATTRIBUTES CHANGED**  
**Issues**:
- `input_dir`, `output_dir` → handled by step plans
- `well_filter` → handled by orchestrator
- `config` → `global_config` with different type
- `results` → handled by VFS system

**✅ Current Reality**: **VFS-centric design**
```python
# ✅ Actual ProcessingContext attributes
context.step_plans      # Dict[step_id, execution_plan] - core data
context.well_id         # Current well being processed
context.global_config   # GlobalPipelineConfig instance
context.filemanager     # VFS access for I/O operations
context._is_frozen      # Immutability flag after compilation

# ✅ Step plans contain I/O configuration
step_plan = context.step_plans[step_id]
step_plan['input_dir']   # Input directory for this step
step_plan['output_dir']  # Output directory for this step
step_plan['read_backend']   # VFS backend for reading
step_plan['write_backend']  # VFS backend for writing
```

## Current Reality: Enhanced Architecture

### Pipeline as List (Revolutionary Change)
```python
# ✅ Pipeline IS a list, not just contains steps
pipeline = Pipeline(steps=[step1, step2], name="My Pipeline")

# ✅ All list operations work
len(pipeline)           # Number of steps
pipeline[0]             # First step
pipeline.append(step)   # Add step
pipeline.extend(steps)  # Add multiple steps
for step in pipeline:   # Iterate steps
    print(step.name)

# ✅ Backward compatibility
pipeline.steps          # Returns self (Pipeline IS the list)
```

### Two-Phase Execution (Architectural Enhancement)
```python
# ✅ More robust than single run() method
orchestrator = PipelineOrchestrator(plate_path=Path("/path/to/plate"))

# Compilation: Create frozen execution contexts
compiled_contexts = orchestrator.compile_pipelines(pipeline)

# Execution: Stateless processing with parallel safety
results = orchestrator.execute_compiled_plate(pipeline, compiled_contexts)
```

### VFS-Centric ProcessingContext
```python
# ✅ Enhanced context with VFS integration
context = ProcessingContext(global_config=config, well_id="A01")
context.filemanager.load("path", "disk")    # VFS operations
context.filemanager.save(data, "path", "memory")  # Multi-backend storage
```

## Impact Assessment

### User Experience Impact
- **Pipeline users**: **Constructor works**, but `run()` method doesn't exist
- **Advanced users**: **Pipeline IS a list** - more powerful than documented
- **ProcessingContext users**: **Constructor signature completely changed**

### Severity: MEDIUM-HIGH
**Core concepts preserved** but **API significantly evolved**. Pipeline inheritance from `list` is a major architectural enhancement not documented.

## Recommendations

### Immediate Actions
1. **Update module path**: ezstitcher → openhcs
2. **Document Pipeline inheritance**: Pipeline IS a List[AbstractStep]
3. **Remove run() method**: Document two-phase execution instead
4. **Update ProcessingContext**: Document actual constructor and VFS integration

### Required Updates
1. **Pipeline class**: Document list inheritance and enhanced capabilities
2. **Constructor signatures**: Update all parameters and types
3. **Method documentation**: Replace run() with orchestrator patterns
4. **ProcessingContext**: Complete rewrite with VFS-centric design

### Missing Enhancements
1. **List inheritance benefits**: Direct indexing, iteration, standard operations
2. **Metadata system**: Rich debugging and UI support
3. **VFS integration**: Multi-backend storage system
4. **Two-phase execution**: Compilation and execution separation

## Estimated Fix Effort
**Major rewrite required**: 16-20 hours to document current architecture accurately

**Recommendation**: Focus on documenting the enhanced list-based Pipeline architecture and VFS-centric ProcessingContext rather than trying to preserve outdated API documentation.
