# CellProfiler Integration Architecture Design

**Date:** 2026-02-16
**Branch:** benchmark-platform
**Status:** Design Phase
**Goal:** Leak-free abstraction for CellProfiler pipeline support in OpenHCS

---

## 1. Executive Summary

OpenHCS aims to make CellProfiler obsolete by providing a cleaner, more principled architecture for high-content screening. This document captures the architectural mapping, identified abstraction leaks, and design decisions for supporting `.cppipe` pipelines in OpenHCS.

**Core Insight:** CellProfiler's stateful, mutable workspace pattern must be translated to OpenHCS's stateless, functional dataflow without semantic loss.

---

## 2. Architecture Comparison

### 2.1 CellProfiler Architecture

```
Pipeline (list of Modules)
    │
    ├── Module.run(workspace)  ← Called per image set
    │       │
    │       ├── workspace.image_set.get_image("DNA")
    │       ├── workspace.object_set.get_objects("Nuclei")
    │       ├── workspace.object_set.add_objects(cells, "Cells")
    │       └── workspace.measurements.add_measurement("Cells", "AreaShape_Area", areas)
    │
    └── Workspace: {image_set, object_set, measurements, display_data}
```

**Key Characteristics:**
- **Stateful workspace:** Modules communicate through mutable shared state
- **Named references:** Objects/images referenced by string name at runtime
- **Measurement aggregation:** Accumulates across modules into single table
- **Per-image-set execution:** One workspace per field of view

### 2.2 OpenHCS Architecture

```
Pipeline (list of FunctionSteps)
    │
    ├── FunctionStep.process(context, step_index)
    │       │
    │       ├── Load 3D stack from filemanager
    │       ├── Execute function with contract wrapper
    │       └── Save outputs to filemanager
    │
    └── ProcessingContext: {step_plans, filemanager, global_config}
```

**Key Characteristics:**
- **Stateless execution:** Steps communicate through explicit data flow
- **Compile-time wiring:** Inputs/outputs resolved at compile time
- **Functional contracts:** PURE_2D, PURE_3D, FLEXIBLE define iteration semantics
- **Per-axis execution:** One context per well (multiple sites/fields)

### 2.3 Contract System Semantics

| Contract | Input | Execution | Output |
|----------|-------|-----------|--------|
| PURE_2D | 3D stack | Unstack → f(2D) × N → Stack | 3D stack |
| PURE_3D | 3D stack | f(3D) directly | 3D stack |
| FLEXIBLE | 3D stack | If slice_by_slice: like PURE_2D, else: like PURE_3D | 3D stack |
| VOLUMETRIC_TO_SLICE | 3D stack | f(3D) → 2D | 3D stack (single slice) |

**Implementation Location:** `unified_registry.py:_execute_pure_2d`, `_execute_pure_3d`, etc.

```python
def _execute_pure_2d(self, func, image, *args, **kwargs):
    memory_type = func.output_memory_type
    slices = unstack_slices(image, memory_type, 0)
    results = [func(sl, *args, **kwargs) for sl in slices]
    return stack_slices(results, memory_type, 0)  # ← CRASH on tuples
```

---

## 3. Identified Abstraction Leaks

### Category A: Control Flow / Aggregation (Contract Layer)

| ID | Leak | Current Behavior | Required Behavior | Severity |
|----|------|------------------|-------------------|----------|
| A1 | Tuple crash | `stack_slices([(img,s,l), ...])` fails | Transpose + aggregate per-component | CRITICAL |
| A2 | No slice context | Function doesn't know which slice | `slice_index` kwarg injected | HIGH |
| A3 | No aggregation semantics | Framework guesses how to combine | Explicit `AggregationStrategy` per output | HIGH |

**A1 Details:**
- Absorbed functions return `(image_2d, stats_dataclass, labels_2d)`
- `_execute_pure_2d` collects N tuples: `[(img0,s0,l0), (img1,s1,l1), ...]`
- `stack_slices()` expects `List[ndarray]`, not `List[tuple]`
- Result: Crash at validation

**A2 Details:**
- CellProfiler: `workspace.image_number` provides context
- OpenHCS PURE_3D: `for i in range(n)` internally
- OpenHCS PURE_2D: No mechanism to pass slice index
- Result: Measurements can't correlate to slice

**A3 Details:**
- Different outputs need different aggregation:
  - Images: `List[2D] → 3D` (stack)
  - Labels: `List[2D] → 3D` (stack)
  - Measurements: `List[Dataclass] → DataFrame` (concat rows)
- Current: No declaration mechanism
- Result: Framework has no information to aggregate correctly

### Category B: Named References (Compile-Time vs Runtime)

| ID | Leak | CellProfiler Pattern | OpenHCS Status | Severity |
|----|------|---------------------|----------------|----------|
| B1 | Object naming | `get_objects("Nuclei")` | No runtime registry | MEDIUM |
| B2 | Image naming | `get_image("DNA")` | Channel index only | LOW |
| B3 | Measurement accumulation | `measurements.add()` | Per-step only | HIGH |
| B4 | Parent-child relationships | `relate_children()` | Not supported | MEDIUM |

**B1 Details:**
- CellProfiler: Objects stored in named registry, looked up at runtime
- OpenHCS: Step outputs wired at compile time
- Resolution: Compile-time symbol resolution (see Section 6)

**B3 Details:**
- CellProfiler: Multiple modules add to shared measurement table
- OpenHCS: Each step produces isolated special outputs
- Resolution: Consolidation step that merges per-step outputs

### Category C: Semantic Gaps

| ID | Gap | Description | Severity |
|----|-----|-------------|----------|
| C1 | Label arrays as first-class | Labels treated as generic data | LOW |
| C2 | Measurement naming convention | CellProfiler: `{Object}_{Category}_{Feature}` | LOW |
| C3 | Multi-step measurement collection | Steps 2,5,7 → single export | MEDIUM |
| C4 | Object-to-image association | Which image produced which labels? | LOW |

---

## 4. What We Are Certain About

### 4.1 The Contract System Is Correct

The `ProcessingContract` enum correctly separates **control flow** concerns:
- PURE_2D: Framework iterates per-slice
- PURE_3D: Function handles full stack

**This is NOT the bug.** The refactor plan's claim that "PURE_2D is for external libraries" was wrong. PURE_2D is correct for any function that expects 2D input.

### 4.2 Aggregation Is Orthogonal to Control Flow

From information-theoretic analysis:

```
Control Flow:    "How do I iterate?" (contract)
Aggregation:     "How do I combine N outputs into 1?" (strategy)

These are INDEPENDENT concerns.
```

The correct decomposition:
```
┌─────────────────────────────────────────────────────────┐
│                    CONTROL FLOW                         │
│  Contract: "How do I iterate?"                          │
│  - PURE_2D: unstack, map, stack                         │
│  - PURE_3D: pass through                                │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    DATA AGGREGATION                      │
│  Strategy: "How do I combine N outputs?"                │
│  - STACK_3D: [2D, ...] → 3D                             │
│  - CONCAT_AS_ROWS: [Dataclass, ...] → DataFrame         │
│  - COLLECT_LIST: [T, ...] → List[T]                     │
└─────────────────────────────────────────────────────────┘
```

### 4.3 Aggregation Must Be Declared, Not Inferred

The function must explicitly state how each output should be aggregated:

```python
@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(
    ("object_stats", AggregationSpec(
        materializer=MaterializationSpec(CsvOptions(...)),
        strategy=AggregationStrategy.CONCAT_AS_ROWS,
    )),
    ("labels", AggregationSpec(
        materializer=MaterializationSpec(ROIOptions()),
        strategy=AggregationStrategy.STACK_3D,
    )),
)
def identify_primary_objects(image_2d, slice_index: int, ...):
    return image_2d, stats, labels_2d
```

### 4.4 Compile-Time Wiring Over Runtime Registry

**Decision:** Named references should be resolved at compile time, not runtime.

**Rationale:**
1. Preserves OpenHCS's functional architecture
2. No hidden state between steps
3. Pipeline is statically analyzable
4. "Names" resolved once, not N times per image set

**Implementation:**
The `.cppipe → OpenHCS` converter builds a symbol table:
```
"Nuclei" → step_2.labels_output
"DNA" → input_channel_0
```

Then generates explicit wiring in the pipeline definition.

### 4.5 Existing Special Outputs Pattern Works

Current OpenHCS functions (e.g., `cell_counting_cpu.py`) demonstrate the PURE_3D pattern:
- Take 3D input
- Iterate internally over slices
- Return aggregated results

This is valid but duplicates iteration logic. The declarative PURE_2D + AggregationStrategy pattern is more principled.

---

## 5. Design Proposal: AggregationSpec

### 5.1 New Types

```python
from enum import Enum
from dataclasses import dataclass

class AggregationStrategy(Enum):
    STACK_3D = "stack_3d"           # [2D, ...] → 3D ndarray
    CONCAT_AS_ROWS = "concat_rows"  # [Dataclass, ...] → DataFrame
    COLLECT_LIST = "collect_list"   # [T, ...] → List[T]
    MERGE_DICTS = "merge_dicts"     # [Dict, ...] → Dict
    FIRST = "first"                 # [T, ...] → T
    LAST = "last"                   # [T, ...] → T

@dataclass
class AggregationSpec:
    strategy: AggregationStrategy
    materializer: MaterializationSpec
```

### 5.2 Modified special_outputs Decorator

```python
@special_outputs(
    "simple_output",  # String only: default aggregation (STACK_3D for arrays, COLLECT_LIST for others)
    ("stats", AggregationSpec(
        strategy=AggregationStrategy.CONCAT_AS_ROWS,
        materializer=MaterializationSpec(CsvOptions()),
    )),
)
```

### 5.3 Modified _execute_pure_2d

```python
def _execute_pure_2d_with_aggregation(self, func, image_3d, *args, **kwargs):
    special_outputs = getattr(func, '__special_outputs__', {})
    agg_specs = getattr(func, '__aggregation_specs__', {})
    
    slices = unstack_slices(image_3d, func.output_memory_type, 0)
    
    # Inject slice_index into kwargs if function expects it
    sig = inspect.signature(func)
    expects_slice_index = 'slice_index' in sig.parameters
    
    results = []
    for i, sl in enumerate(slices):
        if expects_slice_index:
            kwargs['slice_index'] = i
        results.append(func(sl, *args, **kwargs))
    
    # No special outputs: original behavior
    if not special_outputs or not isinstance(results[0], tuple):
        return stack_slices(results, func.output_memory_type, 0)
    
    # Transpose: [(a0,b0), (a1,b1)] → ([a0,a1], [b0,b1])
    transposed = list(zip(*results))
    
    # Apply aggregation per output
    output_keys = list(special_outputs.keys())
    aggregated = []
    for i, values in enumerate(transposed):
        key = output_keys[i] if i < len(output_keys) else None
        spec = agg_specs.get(key)
        strategy = spec.strategy if spec else _infer_strategy(values[0])
        aggregated.append(_apply_aggregation(values, strategy, func.output_memory_type))
    
    return tuple(aggregated) if len(aggregated) > 1 else aggregated[0]
```

### 5.4 Aggregation Functions

```python
def _apply_aggregation(values: List, strategy: AggregationStrategy, memory_type: str):
    if strategy == AggregationStrategy.STACK_3D:
        return stack_slices(values, memory_type, 0)
    elif strategy == AggregationStrategy.CONCAT_AS_ROWS:
        return _concat_as_rows(values)
    elif strategy == AggregationStrategy.COLLECT_LIST:
        return list(values)
    elif strategy == AggregationStrategy.MERGE_DICTS:
        return {k: v for d in values for k, v in d.items()}
    elif strategy == AggregationStrategy.FIRST:
        return values[0]
    elif strategy == AggregationStrategy.LAST:
        return values[-1]

def _concat_as_rows(values: List) -> pd.DataFrame:
    """Convert list of dataclasses to DataFrame with slice_index column."""
    import pandas as pd
    from dataclasses import asdict
    
    rows = []
    for slice_idx, value in enumerate(values):
        if hasattr(value, '__dataclass_fields__'):
            row = asdict(value)
        elif isinstance(value, dict):
            row = value
        else:
            row = {'value': value}
        row['slice_index'] = slice_idx
        rows.append(row)
    
    return pd.DataFrame(rows)

def _infer_strategy(value) -> AggregationStrategy:
    """Infer default aggregation strategy from value type."""
    import numpy as np
    if isinstance(value, np.ndarray):
        return AggregationStrategy.STACK_3D
    elif hasattr(value, '__dataclass_fields__'):
        return AggregationStrategy.CONCAT_AS_ROWS
    elif isinstance(value, dict):
        return AggregationStrategy.MERGE_DICTS
    else:
        return AggregationStrategy.COLLECT_LIST
```

---

## 6. Design Proposal: Compile-Time Symbol Resolution

### 6.1 .cppipe Parsing

The `.cppipe` file declares modules with named inputs/outputs:

```
IdentifyPrimaryObjects:[module_num]
    Select the input image:DNA
    Name the primary objects to be identified:Nuclei
    ...

IdentifySecondaryObjects:[module_num]
    Select the input objects:Nuclei
    Name the objects to be identified:Cells
    ...
```

### 6.2 Symbol Table Construction

During parsing, build a symbol table:

```python
symbol_table = {
    # Images (from NamesAndTypes module)
    "DNA": {"type": "image", "source": "input_channel_0"},
    "GFP": {"type": "image", "source": "input_channel_1"},
    
    # Objects (from Identify* modules)
    "Nuclei": {"type": "labels", "source": "step_2", "output_key": "labels"},
    "Cells": {"type": "labels", "source": "step_3", "output_key": "labels"},
    
    # Measurements (from Measure* modules)
    "Nuclei_AreaShape_Area": {"type": "measurement", "source": "step_4"},
}
```

### 6.3 Pipeline Generation

Generate OpenHCS pipeline with explicit wiring:

```python
steps = [
    # Step 0: Load images
    FunctionStep(func=load_images, ...),
    
    # Step 2: IdentifyPrimaryObjects
    FunctionStep(
        func=identify_primary_objects,
        # Wire input
        input_mapping={"image": symbol_table["DNA"]["source"]},
        # Register output in symbol table
        output_registration={"labels": ("Nuclei", "labels")},
    ),
    
    # Step 3: IdentifySecondaryObjects
    FunctionStep(
        func=identify_secondary_objects,
        # Wire inputs from symbol table
        input_mapping={
            "image": symbol_table["DNA"]["source"],
            "primary_labels": symbol_table["Nuclei"]["source"],
        },
        output_registration={"labels": ("Cells", "labels")},
    ),
    
    # Final step: Consolidate measurements
    FunctionStep(
        func=consolidate_measurements,
        input_mapping={
            "measurements": [
                symbol_table["Nuclei_AreaShape_Area"]["source"],
                symbol_table["Cells_AreaShape_Area"]["source"],
            ]
        },
    ),
]
```

### 6.4 No Runtime Registry Needed

Because all references are resolved at compile time:
- No `ObjectRegistry` in ProcessingContext
- No `NamedImageRegistry` in ProcessingContext
- Pure functional dataflow is preserved

---

## 7. Implementation Phases

### Phase 1: Fix Contract Layer (A1, A2, A3)

**Goal:** Make absorbed CellProfiler functions execute correctly.

**Tasks:**
1. Define `AggregationStrategy` enum
2. Define `AggregationSpec` dataclass
3. Extend `@special_outputs` to accept `AggregationSpec`
4. Modify `_execute_pure_2d` to handle tuples with aggregation
5. Add `slice_index` injection for functions that declare it
6. Update absorbed functions to declare aggregation strategies

**Files to Modify:**
- `openhcs/core/pipeline/function_contracts.py` - Add AggregationSpec
- `openhcs/processing/backends/lib_registry/unified_registry.py` - Modify _execute_pure_2d
- `benchmark/cellprofiler_library/functions/*.py` - Add aggregation specs

**Test Criteria:**
- `identify_primary_objects` on 3D stack produces:
  - 3D label array
  - DataFrame with per-slice measurements
- No crashes on tuple returns

### Phase 2: Symbol Table and Pipeline Generation

**Goal:** Generate OpenHCS pipeline from .cppipe file with correct wiring.

**Tasks:**
1. Extend `.cppipe` parser to extract all name references
2. Build symbol table during parsing
3. Generate pipeline with explicit input/output wiring
4. Add `consolidate_measurements` function for final output

**Files to Modify:**
- `benchmark/converter/parser.py` - Extract names
- `benchmark/converter/pipeline_generator.py` - Generate wiring
- New: `benchmark/converter/symbol_table.py`

**Test Criteria:**
- Real .cppipe file converts to working OpenHCS pipeline
- Output measurements match CellProfiler's output

### Phase 3: Absorbed Function Refactoring

**Goal:** All 88 absorbed functions use correct contracts and aggregation specs.

**Tasks:**
1. Audit all functions for correct contract (PURE_2D vs PURE_3D)
2. Add `AggregationSpec` to all functions with special outputs
3. Add `slice_index` parameter where needed
4. Verify 3D variants use PURE_3D

**Files to Modify:**
- All files in `benchmark/cellprofiler_library/functions/`

**Test Criteria:**
- All functions pass contract validation
- Aggregation produces correct output types

---

## 8. Open Questions

### 8.1 Measurement Naming Convention

**Question:** Should OpenHCS adopt CellProfiler's `{Object}_{Category}_{Feature}` convention, or use a simpler scheme?

**Options:**
- A: Adopt CellProfiler convention (compatibility)
- B: Use `{output_key}` from AggregationSpec (simplicity)
- C: Configurable per-pipeline

**Impact:** CSV column names, downstream analysis scripts

### 8.2 Multi-Site Aggregation

**Question:** CellProfiler processes one field of view at a time. OpenHCS processes one well (multiple sites). How do measurements aggregate?

**Options:**
- A: Per-site measurements, concatenated in final output
- B: Per-well aggregation (mean, sum, etc.)
- C: Both, with separate output files

**Impact:** Output file structure, statistical analysis

### 8.3 Object Relationships

**Question:** How to handle `relate_children()` pattern (parent-child object tracking)?

**Current:** Not supported
**Needed for:** IdentifySecondaryObjects, RelateObjects

**Options:**
- A: Compute on-demand as special output
- B: Store in separate relationship table
- C: Encode in label array (e.g., label ID = parent_id * 1000 + child_id)

### 8.4 3D Processing Support

**Question:** CellProfiler's 3D support is limited. How does OpenHCS handle volumetric pipelines?

**Current State:**
- Some absorbed functions have `_3d` variants
- These use PURE_3D contract

**Question:** Is this sufficient, or do we need explicit 3D CellProfiler module support?

### 8.5 Error Handling and Validation

**Question:** How to handle CellProfiler-specific errors (e.g., "no objects found")?

**Options:**
- A: Raise exception (fail the well)
- B: Log warning, return empty results
- C: Configurable behavior

### 8.6 Backward Compatibility

**Question:** Should existing OpenHCS functions be updated to use AggregationSpec?

**Current:** Functions like `count_cells_single_channel` use PURE_3D pattern
**New:** Could use PURE_2D + AggregationSpec

**Options:**
- A: Keep existing, only use for CellProfiler functions
- B: Gradually migrate existing functions
- C: Provide both patterns, let users choose

### 8.7 Performance Considerations

**Question:** Does the transpose + aggregation pattern have performance impact?

**Benchmark needed:**
- Current PURE_3D pattern
- New PURE_2D + AggregationSpec pattern
- Memory overhead of intermediate tuple lists

---

## 9. Out of Scope (For Now)

The following are explicitly out of scope for the initial implementation:

1. **UI for CellProfiler pipeline import** - CLI only initially
2. **Display/visualization modules** - Headless only
3. **CreateBatchFiles module** - OpenHCS has different parallelization model
4. **CellProfiler Analyst integration** - Different project
5. **Custom module support** - Only absorbed modules initially

---

## 10. Success Criteria

The integration is considered successful when:

1. **Functional:** A `.cppipe` file converts to an OpenHCS pipeline that produces equivalent outputs
2. **Performant:** Processing time is comparable or better than CellProfiler
3. **Maintainable:** No abstraction leaks - CellProfiler concepts are cleanly mapped
4. **Extensible:** Adding new absorbed modules is straightforward
5. **Tested:** Unit tests for aggregation, integration tests for real pipelines

---

## 11. Context for New Agents

This section provides everything a fresh agent needs to understand both architectures without additional research.

### 11.1 Essential Files to Read

**OpenHCS Core Architecture:**
```
openhcs/
├── core/
│   ├── steps/
│   │   └── function_step.py          # How steps execute, special outputs handling
│   ├── pipeline/
│   │   ├── function_contracts.py     # @special_outputs, @special_inputs decorators
│   │   └── compiler.py               # Pipeline compilation, path planning
│   ├── context/
│   │   └── processing_context.py     # ProcessingContext definition
│   ├── orchestrator/
│   │   └── orchestrator.py           # Well/site iteration, parallelization
│   └── memory/
│       └── __init__.py               # Re-exports from arraybridge
│
├── processing/
│   └── backends/
│       └── lib_registry/
│           └── unified_registry.py   # ProcessingContract, _execute_pure_2d, etc.
│
└── constants/
    └── constants.py                  # Backend, VariableComponents enums
```

**CellProfiler Integration:**
```
benchmark/
├── cellprofiler_library/
│   └── functions/                    # 88 absorbed CellProfiler modules
│       ├── identifyprimaryobjects.py # Example: PURE_2D with special outputs
│       ├── watershed.py              # Example: PURE_2D segmentation
│       └── ...
│
├── cellprofiler_source/
│   ├── library/
│   │   ├── functions/                # Cloned CP library functions
│   │   │   ├── segmentation.py       # Label formats (dense, sparse, ijv)
│   │   │   └── measurement.py        # Measurement utilities
│   │   └── opts/                     # CP option dataclasses
│   └── modules/                      # Cloned CP modules (90 files)
│
├── converter/
│   ├── parser.py                     # .cppipe file parser
│   ├── llm_converter.py              # LLM-powered module conversion
│   └── pipeline_generator.py         # Generate OpenHCS pipeline from .cppipe
│
└── cellprofiler_pipelines/
    └── ExampleHuman.cppipe           # Example pipeline for testing
```

### 11.2 OpenHCS Execution Flow (Detailed)

```
1. PipelineOrchestrator.compile_pipelines()
   │
   ├── Initialize step_plans for each step
   │   - PathPlanner generates VFS paths for inputs/outputs
   │   - Resolve special_inputs from other steps
   │   - Assign GPU resources
   │
   └── Freeze ProcessingContext (immutable for execution)

2. PipelineOrchestrator.execute_compiled_plate()
   │
   ├── For each well (parallel across workers):
   │   │
   │   └── _execute_single_axis_static(pipeline, context)
   │       │
   │       └── For each step in pipeline:
   │           │
   │           └── FunctionStep.process(context, step_index)
   │               │
   │               ├── _bulk_preload_step_images()  # Load to memory backend
   │               │
   │               ├── For each pattern group:
   │               │   │
   │               │   ├── Load slices → stack_slices() → 3D array
   │               │   │
   │               │   ├── _execute_function_core() or _execute_chain_core()
   │               │   │   │
   │               │   │   └── func(3D_array, **kwargs)
   │               │   │       │
   │               │   │       └── Contract wrapper intercepts:
   │               │   │           - PURE_2D: unstack → map → stack
   │               │   │           - PURE_3D: pass through
   │               │   │
   │               │   ├── Extract special outputs from tuple
   │               │   ├── Save special outputs to VFS (memory backend)
   │               │   └── Save main output to VFS (memory backend)
   │               │
   │               └── _bulk_writeout_step_images()  # Memory → disk/zarr
```

### 11.3 ProcessingContract Implementation

**Location:** `openhcs/processing/backends/lib_registry/unified_registry.py`

```python
class ProcessingContract(Enum):
    PURE_3D = "_execute_pure_3d"
    PURE_2D = "_execute_pure_2d"
    FLEXIBLE = "_execute_flexible"
    VOLUMETRIC_TO_SLICE = "_execute_volumetric_to_slice"

    def execute(self, registry, func, image, *args, **kwargs):
        method = getattr(registry, self.value)
        return method(func, image, *args, **kwargs)
```

**Execution methods:**
```python
def _execute_pure_3d(self, func, image, *args, **kwargs):
    """3D input → 3D output, no transformation."""
    return func(image, *args, **kwargs)

def _execute_pure_2d(self, func, image, *args, **kwargs):
    """3D input → unstack → 2D×N → stack → 3D output."""
    memory_type = func.output_memory_type
    slices = unstack_slices(image, memory_type, 0)
    results = [func(sl, *args, **kwargs) for sl in slices]  # BUG: crashes on tuples
    return stack_slices(results, memory_type, 0)

def _execute_flexible(self, func, image, *args, **kwargs):
    """Toggle between PURE_2D and PURE_3D behavior."""
    slice_by_slice = getattr(func, 'slice_by_slice', False)
    if slice_by_slice:
        return self._execute_pure_2d(func, image, *args, **kwargs)
    else:
        return self._execute_pure_3d(func, image, *args, **kwargs)
```

**How contracts are applied:**
```python
# In LibraryRegistryBase.apply_contract_wrapper()
@wraps(func)
def wrapper(image, *args, **kwargs):
    # ... inject configurable params ...
    return contract.execute(self, func, image, *args, **filtered_kwargs)
```

### 11.4 Special Outputs System

**Decorator:** `openhcs/core/pipeline/function_contracts.py`

```python
@special_outputs(
    "simple_output",  # String: no materialization
    ("stats", MaterializationSpec(CsvOptions(...))),  # With materialization
)
def my_function(image):
    return processed_image, simple_value, stats_data  # Tuple: (main, special1, special2)
```

**Execution handling:** `openhcs/core/steps/function_step.py:_execute_function_core()`

```python
raw_function_output = func_callable(main_data_arg, **final_kwargs)

if isinstance(raw_function_output, tuple):
    main_output_data = raw_function_output[0]
    returned_special_values = raw_function_output[1:]
    
    for i, (output_key, vfs_path) in enumerate(special_outputs_plan.items()):
        value_to_save = returned_special_values[i]
        context.filemanager.save(value_to_save, vfs_path, Backend.MEMORY.value)
else:
    main_output_data = raw_function_output

return main_output_data
```

**Key insight:** Special outputs are extracted AFTER the function returns. The contract layer (`_execute_pure_2d`) doesn't know about them.

### 11.5 CellProfiler Workspace Structure

**Location:** Cloned source in `benchmark/cellprofiler_source/`

```python
# CellProfiler's workspace (simplified)
class Workspace:
    def __init__(self, pipeline, image_set, object_set, measurements):
        self.image_set = image_set      # Dict-like: get_image("DNA")
        self.object_set = object_set    # Dict-like: get_objects("Nuclei")
        self.measurements = measurements  # add_measurement(object, feature, data)
        self.display_data = SimpleNamespace()
        self.pipeline = pipeline
```

**Object model:**
```python
class Objects:
    segmented: np.ndarray           # Final label array (2D or 3D)
    unedited_segmented: np.ndarray  # Before filtering
    parent_image: Image             # Reference to source image
    
    @property
    def count(self) -> int:
        return int(self.segmented.max())
    
    def relate_children(self, child_objects: 'Objects') -> Tuple[np.ndarray, np.ndarray]:
        """Returns (children_per_parent, parents_of_children)."""
        # Maps parent labels to child labels based on overlap
```

**Measurement naming:**
```python
# Format: {Object}_{Category}_{Feature}
measurements.add_measurement("Nuclei", "AreaShape_Area", areas)
measurements.add_measurement("Nuclei", "Location_Center_X", x_coords)
measurements.add_measurement("Nuclei", "Intensity_MeanIntensity_DAPI", intensities)
```

### 11.6 Absorbed Function Pattern

**Current state (buggy):**

```python
# benchmark/cellprofiler_library/functions/identifyprimaryobjects.py

@numpy(contract=ProcessingContract.PURE_2D)  # Declares: expects 2D input
@special_outputs(
    ("object_stats", csv_materializer(...)),
    ("labels", materialize_segmentation_masks),
)
def identify_primary_objects(image: np.ndarray, ...) -> Tuple[np.ndarray, PrimaryObjectStats, np.ndarray]:
    """
    Input: 2D image (because PURE_2D contract)
    Output: (2D_image, stats_dataclass, 2D_labels)
    
    Problem: When called N times via _execute_pure_2d:
    - results = [(img0, s0, l0), (img1, s1, l1), ...]
    - stack_slices(results) crashes
    """
    labels = _segment(image)
    stats = _compute_stats(labels)
    return image, stats, labels
```

**Required state (with AggregationSpec):**

```python
@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(
    ("object_stats", AggregationSpec(
        strategy=AggregationStrategy.CONCAT_AS_ROWS,
        materializer=MaterializationSpec(CsvOptions(...)),
    )),
    ("labels", AggregationSpec(
        strategy=AggregationStrategy.STACK_3D,
        materializer=MaterializationSpec(ROIOptions()),
    )),
)
def identify_primary_objects(image: np.ndarray, slice_index: int, ...) -> Tuple[np.ndarray, PrimaryObjectStats, np.ndarray]:
    """
    Input: 2D image + slice_index (injected by framework)
    Output: (2D_image, stats_dataclass, 2D_labels)
    
    Framework handles:
    - Inject slice_index
    - Collect N results
    - Transpose tuples
    - Apply aggregation strategies
    """
    labels = _segment(image)
    stats = _compute_stats(labels, slice_index)  # Use slice_index in stats
    return image, stats, labels
```

### 11.7 Key Terms Glossary

| Term | Definition |
|------|------------|
| **ProcessingContract** | Enum declaring how function handles dimensions (PURE_2D, PURE_3D, FLEXIBLE) |
| **AggregationStrategy** | (Proposed) Enum declaring how to combine N outputs into 1 |
| **special_outputs** | Decorator marking function outputs for separate VFS storage |
| **VFS (Virtual File System)** | OpenHCS's abstraction over MEMORY, DISK, ZARR backends |
| **ProcessingContext** | Immutable state container for pipeline execution |
| **step_plans** | Dict in context containing compiled execution info per step |
| **Absorbed function** | CellProfiler module converted to OpenHCS-compatible function |
| **Workspace** | CellProfiler's mutable state container (per image set) |
| **Objects** | CellProfiler's class for segmentation labels with metadata |
| **Measurements** | CellProfiler's table-like storage for per-object features |

### 11.8 Quick Reference: What to Read When

**If you need to understand:**
- How PURE_2D crashes → `unified_registry.py:_execute_pure_2d` + this doc §3
- How special outputs work → `function_step.py:_execute_function_core` + this doc §11.4
- How pipelines are compiled → `compiler.py` + `processing_context.py`
- How CellProfiler modules work → `benchmark/cellprofiler_source/modules/*.py`
- How absorbed functions are structured → `benchmark/cellprofiler_library/functions/*.py`
- How .cppipe files are parsed → `benchmark/converter/parser.py`

---

## 12. References

- CellProfiler Manual: https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-5.0.0/
- CellProfiler GitHub: https://github.com/CellProfiler/CellProfiler
- OpenHCS Architecture: `docs/architecture.md` (if exists)
- Existing Refactor Plan: `plans/cellprofiler_refactor_plan.md`
- Feasibility Study: `docs/feasibility_cellprofiler_integration.md`

---

## 13. Change Log

| Date | Author | Changes |
|------|--------|---------|
| 2026-02-16 | opencode | Initial design document |
| 2026-02-16 | opencode | Added §11 "Context for New Agents" with file paths, code snippets, glossary |
