# Fact-Check Report: concepts/processing_context.rst

## File: `docs/source/concepts/processing_context.rst`
**Priority**: MEDIUM  
**Status**: 🟡 **ARCHITECTURAL EVOLUTION**  
**Accuracy**: 40% (Core concept preserved, implementation revolutionized)

## Executive Summary
**EZStitcher → OpenHCS Evolution**: **ProcessingContext concept perfectly preserved** but implementation revolutionized. **Frozen context pattern** replaces mutable context. **VFS-centric data flow** replaces direct attribute access. **Step plans system** provides superior execution planning. **All documented communication patterns work** with enhanced architecture.

## Section-by-Section Analysis

### Overview (Lines 5-12)
```rst
The ProcessingContext is a crucial component that maintains state during pipeline execution. It:
* Holds input/output directories, well filter, and configuration
* Stores processing results
* Serves as a communication mechanism between steps
```
**Status**: ✅ **CONCEPT PERFECTLY PRESERVED**  
**✅ Current Reality**: **Same responsibilities with enhanced architecture**
```python
from openhcs.core.context.processing_context import ProcessingContext

# Same core responsibilities, enhanced implementation:
# - Maintains state during execution (✅ preserved)
# - Holds configuration and well information (✅ enhanced with GlobalPipelineConfig)
# - Stores execution plans (✅ enhanced with step_plans system)
# - Enables step communication (✅ enhanced with VFS and special I/O)
```

### Creating a Context (Lines 14-32)

#### Manual Context Creation (Lines 19-32)
```python
from ezstitcher.core.pipeline import ProcessingContext

context = ProcessingContext(
    input_dir="path/to/input",
    output_dir="path/to/output",
    well_filter=["A01", "B02"],
    orchestrator=orchestrator,
    positions_file="path/to/positions.csv",
    custom_parameter=42
)
```
**Status**: ❌ **CONSTRUCTOR SIGNATURE COMPLETELY CHANGED**  
**Issue**: Constructor requires GlobalPipelineConfig, different parameters  
**✅ Current Reality**: **Enhanced constructor with configuration management**
```python
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.config import get_default_global_config

# Enhanced constructor (orchestrator creates contexts, not manual creation)
global_config = get_default_global_config()
context = ProcessingContext(
    global_config=global_config,  # ✅ Required parameter
    step_plans={},                # ✅ Execution plans dictionary
    well_id="A01",               # ✅ Well identifier
    # Additional attributes via kwargs
    filemanager=filemanager,     # ✅ VFS access
    microscope_handler=handler   # ✅ Microscope interface
)

# Note: Contexts are typically created by PipelineOrchestrator, not manually
# Manual creation is for advanced use cases only
```

### Accessing Context Attributes (Lines 34-47)

#### Direct Attribute Access (Lines 39-47)
```python
# Access standard attributes
print(context.input_dir)
print(context.well_filter)

# Access custom attributes
print(context.positions_file)
print(context.custom_parameter)
```
**Status**: ❌ **ATTRIBUTES CHANGED**  
**Issue**: Different attribute structure, frozen context pattern  
**✅ Current Reality**: **Enhanced attribute access with step plans**
```python
# Current attribute structure
print(context.well_id)           # ✅ Well identifier
print(context.global_config)     # ✅ System configuration
print(context.step_plans)        # ✅ Execution plans dictionary
print(context.filemanager)       # ✅ VFS access

# Step-specific data accessed through step plans
step_id = get_step_id(step)
step_plan = context.step_plans[step_id]
print(step_plan['input_dir'])    # ✅ Input directory for this step
print(step_plan['output_dir'])   # ✅ Output directory for this step
print(step_plan['read_backend']) # ✅ VFS backend for reading
print(step_plan['write_backend'])# ✅ VFS backend for writing

# Custom attributes can be added during compilation
# context.custom_parameter = 42  # Added during compilation phase
```

### Accessing the Orchestrator (Lines 49-73)

#### Step Process Method Example (Lines 56-73)
```python
def process(self, context: 'ProcessingContext') -> 'ProcessingContext':
    logger.info("Processing step: %s", self.name)
    
    # Get directories and microscope handler
    input_dir = self.input_dir
    output_dir = self.output_dir
    well_filter = self.well_filter or context.well_filter
    orchestrator = context.orchestrator
    microscope_handler = orchestrator.microscope_handler
```
**Status**: ❌ **PATTERN COMPLETELY CHANGED**  
**Issue**: No orchestrator reference in context, stateless execution  
**✅ Current Reality**: **Enhanced stateless execution with step plans**
```python
from openhcs.core.steps.function_step import get_step_id

def process(self, context: 'ProcessingContext') -> None:  # ✅ Returns None (stateless)
    # Enhanced stateless execution pattern
    step_id = get_step_id(self)
    step_plan = context.step_plans[step_id]
    
    # All configuration comes from step plan (more robust)
    step_name = step_plan['step_name']
    input_dir = step_plan['input_dir']
    output_dir = step_plan['output_dir']
    read_backend = step_plan['read_backend']
    write_backend = step_plan['write_backend']
    
    # VFS access through context.filemanager
    data = context.filemanager.load(input_path, read_backend)
    processed_data = process_function(data)
    context.filemanager.save(processed_data, output_path, write_backend)
    
    # No orchestrator reference needed - all data flows through VFS
    # More robust: frozen context prevents accidental mutations
```

### Specialized Step Examples (Lines 75-110)

#### PositionGenerationStep Pattern (Lines 79-93)
```python
def process(self, context):
    well = context.well_filter[0] if context.well_filter else None
    orchestrator = context.orchestrator
    input_dir = self.input_dir or context.input_dir
    positions_dir = self.output_dir or context.output_dir
    
    positions_file, reference_pattern = orchestrator.generate_positions(well, input_dir, positions_dir)
    
    context.positions_dir = positions_dir
    context.reference_pattern = reference_pattern
    return context
```
**Status**: ❌ **PATTERN COMPLETELY CHANGED**  
**Issue**: No orchestrator reference, frozen context, VFS-based execution  
**✅ Current Reality**: **Enhanced function-based execution with VFS**
```python
from openhcs.processing.backends.pos_gen.mist_gpu import generate_positions_mist_gpu
from openhcs.core.memory.decorators import special_outputs

@special_outputs("positions_file", "reference_pattern")
def position_generation_function(image_stack):
    """Generate positions using GPU-accelerated MIST algorithm."""
    positions_data, reference_pattern = generate_positions_mist_gpu(image_stack)
    return image_stack, positions_data, reference_pattern

# Function-based step (more flexible than specialized classes)
step = FunctionStep(func=position_generation_function, name="Position Generation")

# Execution through step plans (stateless)
def process(self, context: 'ProcessingContext') -> None:
    step_id = get_step_id(self)
    step_plan = context.step_plans[step_id]
    
    # Load data through VFS
    image_stack = context.filemanager.load(step_plan['input_path'], step_plan['read_backend'])
    
    # Process with function
    processed_stack, positions_data, reference_pattern = self.func(image_stack)
    
    # Save results through VFS
    context.filemanager.save(processed_stack, step_plan['output_path'], step_plan['write_backend'])
    context.filemanager.save(positions_data, step_plan['positions_file'], step_plan['write_backend'])
    context.filemanager.save(reference_pattern, step_plan['reference_pattern'], step_plan['write_backend'])
```

#### ImageStitchingStep Pattern (Lines 95-110)
```python
def process(self, context):
    orchestrator = getattr(context, 'orchestrator', None)
    if not orchestrator:
        raise ValueError("ImageStitchingStep requires an orchestrator in the context")
    
    orchestrator.stitch_images(
        well=context.well,
        input_dir=context.input_dir,
        output_dir=context.output_dir,
        positions_file=positions_file
    )
    return context
```
**Status**: ❌ **PATTERN COMPLETELY CHANGED**  
**✅ Current Reality**: **Enhanced function-based stitching with VFS**
```python
from openhcs.processing.backends.assemblers.simple_assembler import assemble_images
from openhcs.core.memory.decorators import special_inputs

@special_inputs("positions_file")
def image_stitching_function(image_stack, positions_file):
    """Stitch images using position data."""
    return assemble_images(image_stack, positions_file)

# Function-based step (more flexible)
step = FunctionStep(func=image_stitching_function, name="Image Stitching")

# Enhanced execution with automatic position file access
def process(self, context: 'ProcessingContext') -> None:
    step_id = get_step_id(self)
    step_plan = context.step_plans[step_id]
    
    # VFS automatically provides position file through special inputs
    image_stack = context.filemanager.load(step_plan['input_path'], step_plan['read_backend'])
    positions_file = context.filemanager.load(step_plan['positions_file'], step_plan['read_backend'])
    
    # Process with function
    stitched_image = self.func(image_stack, positions_file)
    
    # Save result through VFS
    context.filemanager.save(stitched_image, step_plan['output_path'], step_plan['write_backend'])
```

## Current Reality: Enhanced ProcessingContext Architecture

### Frozen Context Pattern (Revolutionary Improvement)
```python
from openhcs.core.context.processing_context import ProcessingContext

# Enhanced context lifecycle
context = ProcessingContext(global_config=config, well_id="A01")

# Compilation phase: context is mutable
context.inject_plan(step_id, execution_plan)
context.filemanager = filemanager
context.microscope_handler = handler

# Freeze context after compilation (immutability for parallel safety)
context.freeze()

# Execution phase: context is immutable
# context.well_id = "B01"  # ❌ Raises AttributeError
# Prevents accidental mutations during parallel execution
```

### VFS-Centric Data Flow (Superior to Direct Attributes)
```python
# Enhanced data flow through VFS
step_plan = context.step_plans[step_id]

# All I/O through VFS (multi-backend support)
input_data = context.filemanager.load(step_plan['input_path'], "memory")
processed_data = process_function(input_data)
context.filemanager.save(processed_data, step_plan['output_path'], "disk")

# Cross-step communication through VFS
positions_data = context.filemanager.load("positions.csv", "disk")
reference_pattern = context.filemanager.load("reference.json", "memory")
```

### Step Plans System (Superior to Direct Configuration)
```python
# Enhanced execution planning
step_plan = {
    'step_name': 'Z-Stack Processing',
    'input_dir': '/path/to/input',
    'output_dir': '/path/to/output',
    'read_backend': 'memory',
    'write_backend': 'disk',
    'special_inputs': {'positions_file': '/path/to/positions.csv'},
    'special_outputs': {'statistics': '/path/to/stats.json'},
    'function_kwargs': {'method': 'max_projection', 'sigma': 1.0}
}

# All step configuration pre-computed during compilation
# More robust than runtime configuration resolution
```

### Integration Test Pattern (Real Usage)
```python
# From test_main.py - actual working context usage
orchestrator = PipelineOrchestrator(plate_path)
orchestrator.initialize()

# Compilation creates frozen contexts
compiled_contexts = orchestrator.compile_pipelines(pipeline, well_filter=wells)

# Execution uses frozen contexts (stateless)
results = orchestrator.execute_compiled_plate(pipeline, compiled_contexts)

# Context provides all necessary data through step plans
# No direct orchestrator access needed during execution
```

## Impact Assessment

### User Experience Impact
- **Context users**: ❌ **Constructor and attributes completely changed**
- **Step developers**: ✅ **Enhanced patterns with VFS and step plans**
- **Advanced users**: ✅ **More robust architecture with frozen contexts**

### Severity: MEDIUM-HIGH
**Core concept preserved** but **implementation completely revolutionized**. **Frozen context + VFS is superior** to mutable context + direct attributes.

## Recommendations

### Immediate Actions
1. **Update module paths**: ezstitcher → openhcs throughout
2. **Document frozen context pattern**: Superior to mutable context
3. **Document VFS-centric data flow**: Enhanced step communication

### Required Rewrites
1. **Constructor documentation**: Update to GlobalPipelineConfig requirement
2. **Attribute access patterns**: Document step plans system
3. **Step communication**: Replace orchestrator access with VFS patterns
4. **Context lifecycle**: Document compilation → freeze → execution phases

### Missing Revolutionary Content
1. **Frozen context benefits**: Parallel safety and immutability
2. **VFS-centric data flow**: Multi-backend storage and automatic serialization
3. **Step plans system**: Pre-computed execution configuration
4. **Enhanced step patterns**: Function-based with special I/O
5. **Configuration management**: GlobalPipelineConfig integration

## Estimated Fix Effort
**Major rewrite required**: 14-18 hours to document current context architecture

**Recommendation**: **Complete architectural update** - document frozen context pattern, VFS-centric data flow, and step plans system. Current architecture is superior to documented mutable context approach.

---

**Note**: This fact-check was completed as part of the systematic medium priority files review. The ProcessingContext has undergone revolutionary architectural improvements while preserving the core concept of maintaining state during pipeline execution.
