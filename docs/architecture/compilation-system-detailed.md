# OpenHCS Pipeline Compilation System - Complete Architecture

**Status**: CANONICAL  
**Date**: 2025-01-19  
**Purpose**: Comprehensive documentation of the OpenHCS compilation system flow, function pattern storage, and metadata injection mechanisms.

## Overview

The OpenHCS compilation system transforms stateless pipeline definitions into executable contexts through a 5-phase process. This document traces the complete flow from function patterns to execution, solving the mystery of where and how function patterns (including metadata-injected patterns) are stored and retrieved.

## Compilation Flow Summary

```
Pipeline Definition → Context Creation → 5-Phase Compilation → Frozen Context → Execution
```

**Key Insight**: Function patterns are modified during compilation (metadata injection) and stored in `step_plans['func']` by the FuncStepContractValidator, then retrieved during execution.

## Phase-by-Phase Detailed Flow

### Entry Point: `PipelineOrchestrator.compile_pipelines()`

```python
for well_id in wells_to_process:
    context = self.create_context(well_id)

    # 5-Phase Compilation (actual implementation)
    PipelineCompiler.initialize_step_plans_for_context(context, pipeline_definition, metadata_writer=is_responsible, plate_path=self.plate_path)
    PipelineCompiler.declare_zarr_stores_for_context(context, pipeline_definition, self)
    PipelineCompiler.plan_materialization_flags_for_context(context, pipeline_definition, self)
    PipelineCompiler.validate_memory_contracts_for_context(context, pipeline_definition, self)
    PipelineCompiler.assign_gpu_resources_for_context(context)

    context.freeze()
    compiled_contexts[well_id] = context
```

### Phase 1: Step Plan Initialization

**File**: `openhcs/core/pipeline/compiler.py:54-85`

```python
def initialize_step_plans_for_context(context, steps_definition):
    # Pre-initialize basic step_plans
    for step in steps_definition:
        context.step_plans[step.step_id] = {
            "step_name": step.name,
            "step_type": step.__class__.__name__,
            "well_id": context.well_id,
        }
    
    # Call path planner - THIS IS WHERE METADATA INJECTION HAPPENS
    PipelinePathPlanner.prepare_pipeline_paths(context, steps_definition)
    
    # Post-path-planner processing (stores func_name but NOT func)
    for step in steps_definition:
        if isinstance(step, FunctionStep):
            current_plan = context.step_plans[step.step_id]
            if hasattr(step, 'func'):
                current_plan["func_name"] = getattr(step.func, '__name__', str(step.func))
                # NOTE: step.func is NOT stored here - happens in Phase 3
```

#### Critical Sub-Phase: Metadata Injection in Path Planner

**File**: `openhcs/core/pipeline/path_planner.py:396-410`

```python
# For functions with @special_inputs("grid_dimensions")
if metadata_injected_steps and isinstance(step, FunctionStep):
    original_func = step.func  # e.g., mist_compute_tile_positions
    modified_func = original_func
    
    # Inject metadata into function pattern
    for metadata_key, metadata_value in metadata_injected_steps.items():
        # metadata_key = "grid_dimensions"
        # metadata_value = (4, 6) from context.microscope_handler.get_grid_dimensions()
        modified_func = inject_metadata_into_pattern(modified_func, metadata_key, metadata_value)
    
    # Transform: func → (func, {"grid_dimensions": (4, 6)})
    step.func = modified_func  # MODIFIED PATTERN STORED IN STEP OBJECT
```

**Key Point**: After this phase, `step.func` contains the modified function pattern with injected metadata, but it's not yet in `step_plans`.

### Phase 2: Zarr Store Declaration

**File**: `openhcs/core/pipeline/compiler.py:204-224`

This phase declares zarr stores for steps that will use zarr backend, setting up zarr_config in step_plans.

### Phase 3: Materialization Flag Planning

**File**: `openhcs/core/pipeline/materialization_flag_planner.py:34-91`

This phase sets backend flags (`read_backend`, `write_backend`, etc.) in `step_plans`. It does NOT touch function patterns.

### Phase 4: Memory Contract Validation (THE CRITICAL PHASE)

**File**: `openhcs/core/pipeline/compiler.py:194-221`

```python
def validate_memory_contracts_for_context(context, steps_definition):
    # Validator processes steps and returns memory types + function patterns
    step_memory_types = FuncStepContractValidator.validate_pipeline(
        steps=steps_definition,
        pipeline_context=context
    )
    
    # Store memory types AND function patterns in step_plans
    for step_id, memory_types in step_memory_types.items():
        if step_id in context.step_plans:
            context.step_plans[step_id].update(memory_types)  # ← FUNCTION STORED HERE!
```

#### The Function Storage Mechanism

**File**: `openhcs/core/pipeline/funcstep_contract_validator.py:210-215`

```python
def validate_funcstep(step: FunctionStep) -> Dict[str, str]:
    func_pattern = step.func  # Gets the MODIFIED pattern from path planner
    
    # Validate memory types...
    input_type, output_type = validate_function_pattern(func_pattern, step_name)
    
    # Return memory types AND the function pattern
    return {
        'input_memory_type': input_type,
        'output_memory_type': output_type,
        'func': func_pattern  # ← THE MODIFIED FUNCTION PATTERN IS RETURNED!
    }
```

**Critical Understanding**: The validator returns the function pattern (potentially modified by the path planner) as part of the memory types dictionary. When the compiler calls `step_plans[step_id].update(memory_types)`, the `'func'` key gets stored in the step plan.

### Phase 5: GPU Resource Assignment

This phase only assigns GPU IDs and doesn't affect function patterns.

## Execution: Function Pattern Retrieval

**File**: `openhcs/core/steps/function_step.py:550-556`

```python
def process(self, context):
    step_plan = context.step_plans[step_id]
    
    # Get func from step plan (stored by FuncStepContractValidator during compilation)
    func_from_plan = step_plan.get('func')  # ← RETRIEVES MODIFIED PATTERN
    if func_from_plan is None:
        raise ValueError(f"Step plan missing 'func' for step: {step_plan.get('step_name', 'Unknown')}")
    
    # Process the function pattern
    grouped_patterns, comp_to_funcs, comp_to_base_args = prepare_patterns_and_functions(
        patterns_by_well[well_id], func_from_plan, component=group_by.value if group_by else None
    )
```

## Function Pattern Transformation Examples

### Example 1: Simple Function (No Special Inputs)

```python
# Original
step.func = create_composite

# After path planner: No change
step.func = create_composite

# Stored in step_plans['func']
step_plans[step_id]['func'] = create_composite
```

### Example 2: Function with Metadata Injection

```python
# Original
step.func = mist_compute_tile_positions  # Has @special_inputs("grid_dimensions")

# After path planner: Metadata injected
step.func = (mist_compute_tile_positions, {"grid_dimensions": (4, 6)})

# Stored in step_plans['func']
step_plans[step_id]['func'] = (mist_compute_tile_positions, {"grid_dimensions": (4, 6)})
```

### Example 3: Function with Parameters

```python
# Original
step.func = (create_projection, {'method': 'max_projection'})

# After path planner: No special inputs, no change
step.func = (create_projection, {'method': 'max_projection'})

# Stored in step_plans['func']
step_plans[step_id]['func'] = (create_projection, {'method': 'max_projection'})
```

## Implications for Lazy Loading (@lazy_args)

This architecture is perfectly designed for lazy loading:

1. **Metadata Resolution**: Path planner resolves metadata and could create lazy wrappers
2. **Pattern Storage**: Modified patterns (with lazy wrappers) stored in step_plans
3. **Execution Retrieval**: Execution gets lazy wrappers from step_plans
4. **Lazy Loading**: First access to lazy wrapper triggers actual loading

### Proposed Lazy Loading Flow

```python
# 1. Path planner creates lazy wrapper
lazy_model = LazyN2V2Model(context)
step.func = (n2v2_denoise_torch, {"n2v2_model": lazy_model})

# 2. Validator stores lazy wrapper in step_plans
step_plans[step_id]['func'] = (n2v2_denoise_torch, {"n2v2_model": lazy_model})

# 3. Execution gets lazy wrapper
func_from_plan = step_plan.get('func')  # Contains lazy wrapper

# 4. Function receives lazy wrapper
def n2v2_denoise_torch(image, n2v2_model, **kwargs):
    n2v2_model.eval()  # ← Triggers lazy loading here
```

## Key Architectural Insights

1. **Function patterns are mutable during compilation** - the path planner can modify them
2. **The FuncStepContractValidator is the storage mechanism** - it stores function patterns in step_plans
3. **Execution is completely stateless** - everything needed is in the frozen context
4. **Metadata injection happens early** - during path planning, before validation
5. **The system supports complex function patterns** - tuples, lists, with kwargs injection

## Common Misconceptions Clarified

❌ **Wrong**: "Functions are stored in the compiler's post-path-planner loop"  
✅ **Correct**: Functions are stored by the FuncStepContractValidator

❌ **Wrong**: "step.func is used directly during execution"  
✅ **Correct**: step_plans['func'] is used during execution

❌ **Wrong**: "Metadata injection happens during execution"  
✅ **Correct**: Metadata injection happens during compilation (path planning)

❌ **Wrong**: "The validator only validates, doesn't store anything"  
✅ **Correct**: The validator stores the validated function pattern in step_plans

## Files and Line Numbers Reference

- **Orchestrator entry**: `openhcs/core/orchestrator/orchestrator.py:295-317`
- **Compiler phases**: `openhcs/core/pipeline/compiler.py:54-275`
- **Path planner metadata injection**: `openhcs/core/pipeline/path_planner.py:396-410`
- **Validator function storage**: `openhcs/core/pipeline/funcstep_contract_validator.py:210-215`
- **Execution retrieval**: `openhcs/core/steps/function_step.py:550-556`

This architecture enables powerful patterns like lazy loading, metadata injection, and stateless execution while maintaining clean separation of concerns between compilation and execution phases.
