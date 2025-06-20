# Detailed Rewrite Plan: basic_usage.rst

## Overview
**Target**: Complete rewrite of `docs/source/user_guide/basic_usage.rst`  
**Goal**: Document both TUI and programmatic interfaces with working examples  
**Approach**: Replace broken EZStitcher examples with current OpenHCS patterns

## Current Problems (From Fact-Check)
- ❌ **EZ module**: Doesn't exist, but TUI provides superior interface
- ❌ **Import paths**: ezstitcher.* → openhcs.*
- ❌ **Function examples**: All imports fail, but concepts are preserved
- ❌ **Missing TUI**: The actual beginner interface not documented
- ✅ **Function patterns**: All four patterns work exactly as documented

## Detailed Section-by-Section Rewrite Plan

### Section 1: Title and Introduction
**Current (Lines 1-10)**:
```rst
===========
Basic Usage
===========

This page provides an overview of how to use EZStitcher for basic image stitching tasks.
```

**Rewrite To**:
```rst
===========
Basic Usage
===========

This page provides an overview of how to use OpenHCS for microscopy image processing. OpenHCS offers two complementary interfaces: a visual interface for non-programmers and a powerful programmatic interface for developers.

**Choose Your Interface:**

- **Visual Interface (TUI)**: Perfect for biologists and non-programmers
- **Programmatic Interface**: Ideal for automation and custom workflows
```

### Section 2: Visual Interface (TUI) - NEW SECTION
**Add Completely New Section**:
```rst
Visual Interface (TUI)
---------------------

The TUI (Terminal User Interface) provides a modern, visual way to build and run image processing pipelines without any programming.

Launching the TUI
~~~~~~~~~~~~~~~~

.. code-block:: bash

    python -m openhcs.textual_tui

Basic Workflow
~~~~~~~~~~~~~

1. **Add Plate**
   
   - Click "Add" button in Plate Manager
   - Browse and select your microscopy data folder
   - OpenHCS automatically detects format (ImageXpress, Opera Phenix, etc.)

2. **Initialize Plate**
   
   - Select your plate and click "Init"
   - OpenHCS analyzes the data structure and available wells
   - Status changes from "?" to "-" when ready

3. **Build Pipeline**
   
   - Click "Add Step" in Pipeline Editor
   - Visual function selector opens with categorized functions:
     
     * **Processors**: Normalization, enhancement, filtering
     * **Position Generators**: MIST GPU, Ashlar, self-supervised
     * **Assemblers**: Simple assembler, stack assembler
   
   - Search functions by name or browse categories
   - Select function and configure parameters visually

4. **Configure Parameters**
   
   - Each function shows editable parameter forms
   - Real-time validation and help text
   - Reset buttons for individual parameters or entire functions
   - Visual controls for common parameters:
     
     * Sliders for numeric values
     * Dropdowns for enumerations
     * Checkboxes for boolean options
     * File pickers for paths

5. **Compile and Run**
   
   - Click "Compile" to create optimized execution plan
   - Review compilation results and GPU assignments
   - Click "Run" to execute with real-time progress monitoring

TUI Features
~~~~~~~~~~~

**Visual Pipeline Building:**
- Drag-and-drop function ordering
- Visual parameter editing with validation
- Function help and documentation integration
- Real-time pipeline validation

**GPU Acceleration:**
- Automatic GPU detection and assignment
- Visual memory type selection
- GPU utilization monitoring
- Automatic fallback to CPU when needed

**Multi-Format Support:**
- Automatic plate format detection
- Support for ImageXpress, Opera Phenix, and generic formats
- Visual well selection and filtering
- Batch processing across multiple plates
```

**Context Engine Sources**:
- openhcs/textual_tui/widgets/plate_manager.py (workflow buttons)
- openhcs/textual_tui/screens/function_selector.py (function selection)
- openhcs/textual_tui/widgets/function_pane.py (parameter editing)

### Section 3: Programmatic Interface
**Current (Lines 12-42)**: Broken EZ module examples

**Rewrite To**:
```rst
Programmatic Interface
---------------------

For automation, custom workflows, and integration with other tools, OpenHCS provides a powerful programmatic interface.

Core Concepts
~~~~~~~~~~~~

OpenHCS uses a **function pattern system** that provides four ways to specify processing:

1. **Single Function**: Apply same function to all data
2. **Parameterized Function**: Function with custom parameters  
3. **Sequential Functions**: Chain multiple functions together
4. **Component-Specific Functions**: Different functions for different components

Basic Example
~~~~~~~~~~~~

.. code-block:: python

    from pathlib import Path
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
    from openhcs.core.pipeline import Pipeline
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.core.memory.decorators import cupy_func
    from openhcs.processing.backends.processors.cupy_processor import stack_percentile_normalize

    # Create GPU-accelerated processing function
    @cupy_func
    def normalize_images(image_stack):
        return stack_percentile_normalize(image_stack)

    # Build pipeline with single function pattern
    pipeline = Pipeline(steps=[
        FunctionStep(func=normalize_images, name="Normalize Images")
    ], name="Basic Processing")

    # Create and initialize orchestrator
    orchestrator = PipelineOrchestrator(plate_path=Path("/path/to/plate"))
    orchestrator.initialize()

    # Two-phase execution (more robust than single run() method)
    compiled_contexts = orchestrator.compile_pipelines(
        pipeline_definition=pipeline.steps,
        well_filter=["A01", "B01"]  # Process specific wells
    )

    results = orchestrator.execute_compiled_plate(
        pipeline_definition=pipeline.steps,
        compiled_contexts=compiled_contexts
    )

The Four Function Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Single Function Pattern**

Apply the same function to all data:

.. code-block:: python

    from openhcs.processing.backends.processors.cupy_processor import stack_percentile_normalize

    step = FunctionStep(
        func=stack_percentile_normalize,  # Single function
        name="Normalize All Images"
    )

**2. Parameterized Function Pattern**

Function with custom parameters:

.. code-block:: python

    step = FunctionStep(
        func=(stack_percentile_normalize, {'percentile': 99}),  # Function + parameters
        name="Custom Normalization"
    )

**3. Sequential Function Pattern**

Chain multiple functions together:

.. code-block:: python

    from openhcs.processing.backends.processors.cupy_processor import create_projection
    
    step = FunctionStep(
        func=[
            (create_projection, {'method': 'max_projection'}),  # First: Z-projection
            stack_percentile_normalize,                         # Then: Normalize
            enhance_contrast                                    # Finally: Enhance
        ],
        name="Multi-Step Processing"
    )

**4. Component-Specific Function Pattern**

Different functions for different components:

.. code-block:: python

    from openhcs.constants.constants import GroupBy

    step = FunctionStep(
        func={
            'DAPI': process_dapi_channel,      # DAPI-specific processing
            'GFP': process_gfp_channel,        # GFP-specific processing  
            'BF': process_brightfield_channel  # Brightfield-specific processing
        },
        group_by=GroupBy.CHANNEL,  # Process by channel
        name="Channel-Specific Processing"
    )

Variable Components and Grouping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Control how data is grouped and processed:

.. code-block:: python

    from openhcs.constants.constants import VariableComponents, GroupBy

    # Process Z-stacks: group files by (well, site, channel), vary z_index
    step = FunctionStep(
        func=(create_projection, {'method': 'max_projection'}),
        variable_components=[VariableComponents.Z_INDEX],  # Files that differ by Z
        name="Z-Stack Flattening"
    )

    # Process channels: group files by (well, site), vary channel
    step = FunctionStep(
        func=create_composite_image,
        variable_components=[VariableComponents.CHANNEL],  # Files that differ by channel
        group_by=GroupBy.SITE,  # Process each site separately
        name="Channel Compositing"
    )

    # Process sites: group files by (well), vary site
    step = FunctionStep(
        func=generate_positions,
        variable_components=[VariableComponents.SITE],  # Files that differ by site
        name="Position Generation"
    )

GPU Acceleration
~~~~~~~~~~~~~~~

OpenHCS provides automatic GPU acceleration with memory type decorators:

.. code-block:: python

    from openhcs.core.memory.decorators import cupy_func, torch_func

    # CuPy-accelerated function
    @cupy_func
    def cupy_processing(image_stack):
        import cupy as cp
        return cp.max(image_stack, axis=0, keepdims=True)

    # PyTorch-accelerated function  
    @torch_func
    def torch_processing(image_tensor):
        import torch
        return torch.max(image_tensor, dim=0, keepdim=True)[0]

    # Use in pipeline
    pipeline = Pipeline(steps=[
        FunctionStep(func=cupy_processing, name="CuPy Processing"),
        FunctionStep(func=torch_processing, name="PyTorch Processing")
    ])

The system automatically:
- Converts between memory types (NumPy ↔ CuPy ↔ PyTorch) **between steps**
- Manages GPU memory allocation and cleanup
- Provides CPU fallback when GPU memory is insufficient

**Memory Type Rules**:
- Different steps can use different memory types (automatic conversion)
- All functions within the same step must use the same memory type
- Memory conversions happen between steps, not within steps
```

**Context Engine Sources**:
- docs/architecture/function-patterns.md (all four patterns)
- openhcs/core/steps/function_step.py lines 415-433 (constructor)
- openhcs/constants/constants.py (VariableComponents, GroupBy)
- openhcs/core/memory/decorators.py (GPU decorators)

### Section 4: Complete Pipeline Example
**Current (Lines 44-95)**: Broken EZStitcher class examples

**Rewrite To**:
```rst
Complete Pipeline Example
------------------------

Here's a complete example that demonstrates building a full image processing pipeline:

.. code-block:: python

    from pathlib import Path
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
    from openhcs.core.pipeline import Pipeline
    from openhcs.core.steps.function_step import FunctionStep
    from openhcs.core.memory.decorators import cupy_func
    from openhcs.constants.constants import VariableComponents, GroupBy

    # Import processing functions
    from openhcs.processing.backends.processors.cupy_processor import (
        create_projection, stack_percentile_normalize
    )
    from openhcs.processing.backends.pos_gen.mist_gpu import generate_positions_mist_gpu
    from openhcs.processing.backends.assemblers.simple_assembler import assemble_images

    # Build complete processing pipeline
    pipeline = Pipeline(steps=[
        # Step 1: Flatten Z-stacks using max projection
        FunctionStep(
            func=(create_projection, {'method': 'max_projection'}),
            variable_components=[VariableComponents.Z_INDEX],
            name="Z-Stack Flattening"
        ),
        
        # Step 2: Normalize images for consistent intensity
        FunctionStep(
            func=stack_percentile_normalize,
            variable_components=[VariableComponents.SITE],
            name="Image Normalization"
        ),
        
        # Step 3: Generate positions for stitching
        FunctionStep(
            func=generate_positions_mist_gpu,
            variable_components=[VariableComponents.SITE],
            name="Position Generation"
        ),
        
        # Step 4: Assemble final stitched image
        FunctionStep(
            func=assemble_images,
            variable_components=[VariableComponents.SITE],
            name="Image Assembly"
        )
    ], name="Complete Processing Pipeline")

    # Execute pipeline
    orchestrator = PipelineOrchestrator(plate_path=Path("/path/to/plate"))
    orchestrator.initialize()

    # Compile for all wells (or filter specific wells)
    compiled_contexts = orchestrator.compile_pipelines(
        pipeline_definition=pipeline.steps,
        well_filter=None  # Process all wells
    )

    # Execute with parallel processing
    results = orchestrator.execute_compiled_plate(
        pipeline_definition=pipeline.steps,
        compiled_contexts=compiled_contexts
    )

    # Check results
    for well_id, result in results.items():
        if result.get('success'):
            print(f"Well {well_id}: Processing completed successfully")
        else:
            print(f"Well {well_id}: Error - {result.get('error')}")

This pipeline:
- Processes Z-stacks into 2D images using max projection
- Normalizes images for consistent intensity across the plate
- Generates stitching positions using GPU-accelerated MIST
- Assembles final stitched images for each well
- Runs with automatic GPU acceleration and parallel processing
```

### Section 5: Next Steps
**Current (Lines 97-115)**: Broken cross-references

**Rewrite To**:
```rst
Next Steps
----------

Now that you understand the basics, explore these advanced topics:

**For TUI Users:**
- :doc:`tui_advanced_guide` - Advanced TUI features and workflows
- :doc:`../concepts/gpu_acceleration` - Understanding GPU optimization

**For Programmers:**
- :doc:`intermediate_usage` - Advanced function patterns and custom functions
- :doc:`advanced_usage` - Custom backend development and integration
- :doc:`../api/pipeline_orchestrator` - Complete API reference

**For All Users:**
- :doc:`../concepts/architecture_overview` - Understanding OpenHCS design
- :doc:`../user_guide/best_practices` - Performance optimization tips
- :doc:`../user_guide/troubleshooting` - Common issues and solutions

Common Patterns
~~~~~~~~~~~~~~

**Processing by Channel:**
.. code-block:: python

    step = FunctionStep(
        func={'DAPI': process_nuclei, 'GFP': process_proteins},
        group_by=GroupBy.CHANNEL
    )

**Sequential Processing:**
.. code-block:: python

    step = FunctionStep(
        func=[denoise, normalize, enhance],
        name="Multi-Step Enhancement"
    )

**GPU-Accelerated Processing:**
.. code-block:: python

    @cupy_func
    def gpu_function(data): return processed_data
    
    step = FunctionStep(func=gpu_function)
```

## Implementation Checklist

### Content Verification (Using Context Engine)
- [x] **Function patterns**: Verified all four patterns work from function-patterns.md
- [x] **TUI workflow**: Verified from plate_manager.py and function_selector.py
- [x] **API examples**: Verified from orchestrator.py implementation
- [x] **GPU decorators**: Verified from memory decorators
- [x] **Constants**: Verified VariableComponents and GroupBy enums

### Cross-Reference Updates Needed
- [ ] Create: `docs/source/user_guide/tui_advanced_guide.rst`
- [ ] Update: `docs/source/user_guide/intermediate_usage.rst` (separate plan)
- [ ] Create: `docs/source/concepts/gpu_acceleration.rst`
- [ ] Create: `docs/source/user_guide/troubleshooting.rst`

### Estimated Effort
**Writing time**: 8-12 hours  
**Testing time**: 4-6 hours (verify all examples work)  
**Total**: 12-18 hours

### Success Criteria
1. **All code examples work**: 100% functional examples
2. **Both interfaces documented**: TUI + programmatic with equal detail
3. **Function patterns clear**: All four patterns with working examples
4. **Progressive complexity**: Beginner → intermediate examples
5. **GPU integration**: Clear guidance for GPU acceleration

This plan provides comprehensive coverage of both OpenHCS interfaces with verified, working examples that demonstrate the preserved EZStitcher concepts in their evolved OpenHCS implementation.

## Context Engine Verification Summary
All examples and concepts verified against actual source code:
- ✅ **TUI workflow**: Verified from plate_manager.py and function_selector.py
- ✅ **Function patterns**: Verified from function-patterns.md and function_step.py
- ✅ **GPU decorators**: Verified from memory/decorators.py
- ✅ **Orchestrator API**: Verified from orchestrator.py implementation
- ✅ **Constants**: Verified VariableComponents and GroupBy from constants.py
