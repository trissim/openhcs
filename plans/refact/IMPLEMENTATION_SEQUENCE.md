# OpenHCS Refactoring Implementation Sequence

## Overview

This document outlines the precise sequence of implementation steps for the OpenHCS refactoring, based on a comprehensive code audit and analysis of the existing architecture. It supersedes REFACTORING_ROADMAP.md with more detailed and accurate implementation guidance.

## Prerequisites

- Thorough understanding of the existing OpenHCS architecture
- Review of all refactoring plan documents
- Development environment with proper testing setup

## Phase 1: Core Schema and Context Definition

*Goal: Establish the foundational structures needed for refactoring.*

### Step 1.1: Create Step Plan Schema

**Files to Create:**
- `openhcs/core/pipeline/step_plan_schema.py`

**Implementation:**
1. Implement the schema structure as outlined in `STEP_PLAN_SCHEMA_DEFINITION.md`
2. Create validation functions that verify:
   - Required fields are present
   - Special input keys match special output keys from previous steps
   - Special output keys match special input keys in later steps
   - All backend and memory type values are valid
3. Create helper functions for step plan creation and manipulation

**Doctrinal Enforcement:**
- Clause 245 — Declarative Enforcement
- Clause 500 — File Decomposition Doctrine

### Step 1.2: Implement ProcessingContext Immutability

**Files to Modify:**
- `openhcs/core/context/processing_context.py`

**Implementation:**
1. Add `_is_frozen` flag to ProcessingContext
2. Override `__setattr__` to prevent changes when frozen
3. Add `freeze()` and `is_frozen()` methods
4. Update `inject_plan()` to check frozen state
5. Remove `update_from_step_result()` method

**Doctrinal Enforcement:**
- Clause 66 — Immutability After Construction
- Clause 12 — Absolute Clean Execution

## Phase 2: Compiler and Planner Refactoring

*Goal: Update the pipeline compilation phase to generate proper step plans and freeze context.*

### Step 2.1: Update Path Planner

**Files to Modify:**
- `openhcs/core/pipeline/path_planner.py`

**Implementation:**
1. Modify to handle cross-step special key linking:
   ```python
   def plan_special_paths(steps, pipeline_context):
       # Dictionary to track special output keys and their sources
       special_output_registry = {}
       
       # First pass: Register all special outputs
       for i, step in enumerate(steps):
           step_id = step.uid
           
           if hasattr(step, "special_outputs") and step.special_outputs:
               for key in step.special_outputs:
                   # Register this step as providing output for this key
                   special_output_registry[key] = {
                       "source_step_id": step_id,
                       "pipeline_position": i
                   }
       
       # Second pass: Validate and link special inputs to outputs
       special_path_info = {}
       for i, step in enumerate(steps):
           step_id = step.uid
           step_info = {"special_inputs": {}, "special_outputs": {}}
           
           # Handle special inputs
           if hasattr(step, "special_inputs") and step.special_inputs:
               for key in step.special_inputs:
                   # Verify key exists in registry
                   if key not in special_output_registry:
                       raise ValueError(f"Step {step.name} requests special input '{key}' but no previous step provides it")
                   
                   # Verify step order
                   if special_output_registry[key]["pipeline_position"] >= i:
                       raise ValueError(f"Step {step.name} requests special input '{key}' from a step that comes after it")
                   
                   # Generate path and add to step_info
                   vfs_path = f"special://{key}/{step_id}"
                   step_info["special_inputs"][key] = {
                       "path": vfs_path,
                       "source_step_id": special_output_registry[key]["source_step_id"]
                   }
           
           # Handle special outputs
           if hasattr(step, "special_outputs") and step.special_outputs:
               for key in step.special_outputs:
                   # Generate path
                   vfs_path = f"special://{key}/{step_id}"
                   step_info["special_outputs"][key] = {
                       "path": vfs_path,
                       "target_step_ids": []  # Will be populated in a final pass
                   }
           
           special_path_info[step_id] = step_info
       
       # Final pass: Link outputs to inputs
       for i, step in enumerate(steps):
           step_id = step.uid
           
           if hasattr(step, "special_inputs") and step.special_inputs:
               for key in step.special_inputs:
                   source_step_id = special_output_registry[key]["source_step_id"]
                   # Add this step as a target to the source step's output info
                   special_path_info[source_step_id]["special_outputs"][key]["target_step_ids"].append(step_id)
       
       return special_path_info
   ```
2. Integrate the special path planning with the main path planner

### Step 2.2: Update Materialization Flag Planner

**Files to Modify:**
- `openhcs/core/pipeline/materialization_flag_planner.py`

**Implementation:**
1. Update to handle special inputs/outputs:
   ```python
   def plan_materialization_flags(steps, special_path_info):
       materialization_flags = {}
       
       for step in steps:
           step_id = step.uid
           flags = {
               "read_backend": "disk",   # Default values
               "write_backend": "disk",
               "force_disk_output": False
           }
           
           # Handle regular backends based on memory types
           # ...existing code...
           
           # Handle special inputs
           if step_id in special_path_info and special_path_info[step_id]["special_inputs"]:
               for key, info in special_path_info[step_id]["special_inputs"].items():
                   # Determine appropriate backend based on memory types
                   info["backend"] = determine_backend_for_key(key, step)
                   info["materialize"] = should_materialize(step, key)
           
           # Handle special outputs
           if step_id in special_path_info and special_path_info[step_id]["special_outputs"]:
               for key, info in special_path_info[step_id]["special_outputs"].items():
                   # Determine appropriate backend based on memory types
                   info["backend"] = determine_backend_for_key(key, step)
                   info["materialize"] = should_materialize(step, key)
           
           materialization_flags[step_id] = flags
       
       return materialization_flags
   ```
2. Ensure consistent backend selection for matching special inputs/outputs

### Step 2.3: Update Pipeline Compiler

**Files to Modify:**
- `openhcs/core/pipeline/compiler.py`

**Implementation:**
1. Modify `_create_initial_step_plans` to build plans with cross-step special IO:
   ```python
   def _create_initial_step_plans(steps, step_paths, step_flags, special_path_info):
       step_plans = {}
       
       for i, step in enumerate(steps):
           step_id = step.uid
           
           # Create basic step plan
           step_plan = {
               "step_id": step_id,
               "step_name": step.name,
               "step_type": step.__class__.__name__,
               "pipeline_position": i,
               # ...other basic fields...
           }
           
           # Add special inputs if any
           if step_id in special_path_info and special_path_info[step_id]["special_inputs"]:
               step_plan["special_inputs"] = special_path_info[step_id]["special_inputs"]
           
           # Add special outputs if any
           if step_id in special_path_info and special_path_info[step_id]["special_outputs"]:
               step_plan["special_outputs"] = special_path_info[step_id]["special_outputs"]
           
           step_plans[step_id] = step_plan
       
       return step_plans
   ```
2. Update the compilation flow to include special path planning:
   ```python
   def compile_pipeline(steps, input_dir, well_id):
       # Plan basic paths
       step_paths = plan_paths(steps, input_dir, well_id)
       
       # Plan special input/output paths and validate key matching
       special_path_info = plan_special_paths(steps)
       
       # Plan materialization flags
       materialization_flags = plan_materialization_flags(steps, special_path_info)
       
       # Create step plans
       step_plans = _create_initial_step_plans(steps, step_paths, materialization_flags, special_path_info)
       
       # Validate step plans
       validate_step_plans(step_plans)
       
       # Create immutable context
       context = ProcessingContext(step_plans=step_plans, well_id=well_id)
       context.freeze()
       
       return context
   ```
3. Call `context.freeze()` at the end of compilation
4. Import and use step plan schema validation

## Phase 3: Executor Refactoring

*Goal: Update execution phase to work with immutable context and no StepResult.*

### Step 3.1: Update Pipeline Executor

**Files to Modify:**
- `openhcs/core/pipeline/executor.py`

**Implementation:**
1. Remove StepResult validation from `execute` method
2. Remove context updating code
3. Add validation for context immutability
4. Update visualization to work with context and step plans directly
5. Modify parallel execution to preserve immutability

### Step 3.2: Update Pipeline Orchestrator

**Files to Modify:**
- `openhcs/core/orchestrator/orchestrator.py`

**Implementation:**
1. Separate compilation and execution phases in `run` method
2. Implement two-phase model:
   - First loop: compile all pipelines and create contexts
   - Second loop: execute all pipelines with compiled contexts

## Phase 4: Step Implementation Updates

*Goal: Update steps to use the new execution model with no StepResult returns.*

### Step 4.1: Update FunctionStep

**Files to Modify:**
- `openhcs/core/steps/function_step.py`

**Implementation:**
1. Update `process()` to return None instead of StepResult:
   ```python
   def process(self, context: 'ProcessingContext') -> None:
       """Process a function step using VFS for all I/O."""
       # Get step plan from context
       step_id = self.uid
       step_plan = context.step_plans[step_id]
       
       # Process using the step plan
       self._process_patterns_vfs(context, step_plan)
       
       # Return None, not StepResult
       return None
   ```

2. Implement helpers to handle special inputs/outputs:
   ```python
   def _handle_special_inputs(self, context, step_plan):
       """Load special inputs from VFS paths."""
       special_args = {}
       
       if "special_inputs" in step_plan:
           for key, info in step_plan["special_inputs"].items():
               path = info["path"]
               backend = info["backend"]
               # Load from VFS using FileManager
               data = context.filemanager.load(path, backend)
               special_args[key] = data
       
       return special_args
   
   def _handle_special_outputs(self, context, step_plan, results):
       """Save special outputs to VFS paths."""
       if not results or "special_outputs" not in step_plan:
           return
           
       for key, info in step_plan["special_outputs"].items():
           if key in results:
               path = info["path"]
               backend = info["backend"]
               # Save to VFS using FileManager
               context.filemanager.save(results[key], path, backend)
               
               # Handle materialization if needed
               if info.get("materialize", False) and backend != "disk":
                   context.filemanager.save(results[key], path, "disk")
   ```

3. Update the main processing function to use these helpers:
   ```python
   def _process_patterns_vfs(self, context, step_plan):
       """Process patterns using VFS for I/O."""
       # Get basic step configuration
       input_dir = step_plan["input_dir"]
       output_dir = step_plan["output_dir"]
       read_backend = step_plan["read_backend"]
       write_backend = step_plan["write_backend"]
       func = step_plan["func"]
       
       # Handle primary input loading
       # ... (existing code to load primary input)
       
       # Load special inputs if any
       special_args = self._handle_special_inputs(context, step_plan)
       
       # Call function with primary input and special inputs
       if special_args:
           # Return value might be special outputs
           results = func(primary_array, special_args=special_args)
       else:
           # Regular function call
           output_array = func(primary_array)
           results = {"primary": output_array}
       
       # Handle special outputs if any
       self._handle_special_outputs(context, step_plan, results)
       
       # If no special outputs or both special and primary outputs,
       # process the primary output array (if available)
       if "primary" in results:
           # Save primary output to VFS
           # ... (existing code to save primary output)
   ```

### Step 4.2: Update AbstractStep

**Files to Modify:**
- `openhcs/core/steps/abstract.py`

**Implementation:**
1. Change `process()` signature to return None
2. Update documentation to reflect new execution model

### Step 4.3: Update Specialized Steps

**Files to Modify:**
- `openhcs/core/steps/specialized/*.py`

**Implementation:**
1. Update all specialized step implementations to return None
2. Ensure proper handling of VFS paths and backends

## Phase 5: StepResult Elimination

*Goal: Completely remove StepResult and related code.*

### Step 5.1: Remove StepResult Class

**Files to Modify:**
- Delete `openhcs/core/steps/step_result.py`
- Update `openhcs/core/steps/__init__.py` to remove import

**Implementation:**
1. Remove the file entirely
2. Update all import statements in other files

### Step 5.2: Clean Up References

**Files to Modify:**
Various files with StepResult references

**Implementation:**
1. Remove all imports of StepResult
2. Remove any remaining code that references StepResult

## Phase 6: TUI Updates

*Goal: Update TUI components to work with the new architecture.*

### Step 6.1: Update FunctionPatternEditor

**Files to Modify:**
- `openhcs/tui/function_pattern_editor.py`

**Implementation:**
1. Update to handle special inputs/outputs with cross-step linking
2. Display linked steps in the UI
3. Use VFS path references consistently

### Step 6.2: Update other TUI components

**Files to Modify:**
- Various TUI files

**Implementation:**
1. Update to work with the new execution model
2. Use VFS paths consistently

## Phase 7: Configuration System Updates

*Goal: Align configuration system with new architecture.*

### Step 7.1: Update Configuration Schema

**Files to Modify:**
- `openhcs/core/config.py`

**Implementation:**
1. Update schema to match new step plan structure with cross-step special I/O linking
2. Add validation for new fields

## Testing Strategy

1. **Unit Tests:**
   - Test ProcessingContext immutability
   - Test step plan schema validation with cross-step key matching
   - Test special input/output path planning
   - Test FunctionStep special argument handling
   - Test two-phase execution

2. **Integration Tests:**
   - Create pipelines with chains of steps using special keys
   - Test error cases with mismatched or missing keys
   - Verify data flows correctly between steps

3. **Regression Tests:**
   - Ensure all existing functionality is preserved
   - Verify image processing results match previous versions

## Rollout Plan

1. Implement each phase in sequence
2. Create comprehensive tests for each phase
3. Validate before moving to the next phase
4. Document changes for downstream users