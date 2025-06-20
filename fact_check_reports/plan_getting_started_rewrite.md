# Detailed Rewrite Plan: getting_started.rst

## Overview
**Target**: Complete rewrite of `docs/source/getting_started/getting_started.rst`  
**Goal**: Create functional getting started guide that works with current OpenHCS  
**Approach**: Replace broken EZStitcher examples with working TUI + programmatic workflows

## Current Problems (From Fact-Check)
- ❌ **Project name**: "EZStitcher" → should be "OpenHCS"
- ❌ **Installation**: Claims PyPI package exists (doesn't)
- ❌ **Quick start**: `stitch_plate` function doesn't exist
- ❌ **All code examples**: 100% failure rate
- ❌ **Missing TUI**: The actual user interface not documented

## Detailed Section-by-Section Rewrite Plan

### Section 1: Title and Introduction
**Current (Lines 1-3)**:
```rst
Getting Started with EZStitcher
===========================
```

**Rewrite To**:
```rst
Getting Started with OpenHCS
============================

OpenHCS is an open-source platform for high-content screening image analysis, providing GPU-accelerated microscopy image processing with both visual and programmatic interfaces.
```

**Context Engine Verification**: ✅ Project name confirmed as "OpenHCS" from README.md

### Section 2: Installation
**Current (Lines 4-19)**: Claims PyPI installation with `pip install ezstitcher`

**Rewrite To**:
```rst
Installation
-----------

System Requirements
~~~~~~~~~~~~~~~~~~

**Minimum Requirements:**
- Python 3.11+
- 8GB RAM (16GB+ recommended for large datasets)
- 2GB disk space

**GPU Requirements (Optional but Recommended):**
- NVIDIA GPU with CUDA 12.6 support
- 4GB+ VRAM (8GB+ recommended)
- CuDNN 9.5.x

Basic Installation
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/trissim/openhcs.git
    cd openhcs

    # Install basic dependencies
    pip install -e .

GPU Installation (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For GPU acceleration (significantly faster processing):

.. code-block:: bash

    # Install with GPU support
    pip install -e ".[gpu]"

This installs:
- PyTorch 2.7.x (CUDA 12.6 compatible)
- CuPy 13.x (CUDA 12.x compatible)  
- JAX 0.4.38 (CuDNN 9.5.x compatible)
- TensorFlow 2.15+ (DLPack support)

Verify Installation
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Test basic installation
    python -c "import openhcs; print('OpenHCS installed successfully')"

    # Test GPU support (if installed)
    python -c "import cupy; print(f'GPU available: {cupy.cuda.is_available()}')"
```

**Context Engine Sources**: 
- setup.py lines 34-55 (GPU dependencies)
- README.md lines 28-38 (installation)
- requirements.txt (basic dependencies)

### Section 3: Quick Start - TUI Interface
**Current (Lines 21-38)**: Broken `stitch_plate` examples

**Rewrite To**:
```rst
Quick Start: Visual Interface
----------------------------

The fastest way to get started is with OpenHCS's visual interface:

.. code-block:: bash

    python -m openhcs.textual_tui

This launches a modern terminal-based interface with:

- **Visual pipeline building**: No coding required
- **Real-time parameter editing**: Immediate feedback
- **GPU acceleration**: Automatic optimization
- **Multi-format support**: ImageXpress, Opera Phenix, and more

Basic Workflow
~~~~~~~~~~~~~

1. **Add Plate**: Click "Add" → Select your microscopy data folder
2. **Initialize**: Click "Init" → OpenHCS detects plate format automatically  
3. **Edit Pipeline**: Click "Edit Step" → Visual function selection
4. **Compile**: Click "Compile" → Create optimized execution plan
5. **Run**: Click "Run" → Process images with GPU acceleration

The interface automatically:
- Detects plate format (ImageXpress, Opera Phenix, etc.)
- Processes all channels and Z-stacks appropriately  
- Generates positions and stitches images
- Saves output with organized directory structure
```

**Context Engine Sources**:
- openhcs/textual_tui/__main__.py (launch command)
- openhcs/textual_tui/screens/help_dialog.py lines 22-35 (features)
- openhcs/textual_tui/widgets/plate_manager.py lines 80-87 (workflow buttons)

### Section 4: Quick Start - Programmatic Interface
**Current**: Missing entirely

**Add New Section**:
```rst
Quick Start: Programmatic Interface
----------------------------------

For users who prefer coding or need automation:

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

    # Build pipeline
    pipeline = Pipeline(steps=[
        FunctionStep(func=normalize_images, name="Normalize Images")
    ], name="Basic Processing")

    # Create orchestrator
    orchestrator = PipelineOrchestrator(plate_path=Path("/path/to/plate"))
    orchestrator.initialize()

    # Two-phase execution (compile then execute)
    compiled_contexts = orchestrator.compile_pipelines(
        pipeline_definition=pipeline.steps,
        well_filter=["A01", "B01"]  # Process specific wells
    )

    results = orchestrator.execute_compiled_plate(
        pipeline_definition=pipeline.steps,
        compiled_contexts=compiled_contexts
    )

This provides:
- **GPU acceleration**: Automatic CuPy/PyTorch optimization
- **Memory management**: Efficient GPU memory handling
- **Error handling**: Robust compilation and execution
- **Parallel processing**: Multi-well concurrent execution
```

**Context Engine Sources**:
- openhcs/core/orchestrator/orchestrator.py lines 71-78 (constructor)
- openhcs/core/orchestrator/orchestrator.py lines 256-281 (compile_pipelines)
- openhcs/core/orchestrator/orchestrator.py lines 403-409 (execute_compiled_plate)

### Section 5: Common Parameters and Options
**Current (Lines 40-52)**: Broken `stitch_plate` parameters

**Rewrite To**:
```rst
Common Processing Options
------------------------

TUI Parameter Editing
~~~~~~~~~~~~~~~~~~~~

The visual interface provides intuitive controls for:

- **Normalization**: Toggle on/off, adjust percentiles
- **Z-stack processing**: Max/mean/focus projection methods
- **Channel weights**: Visual sliders for composite generation
- **Well filtering**: Multi-select well picker
- **GPU memory types**: Automatic optimization selection

Programmatic Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from openhcs.core.config import get_default_global_config
    from openhcs.constants.constants import VariableComponents, GroupBy

    # Configure global settings
    global_config = get_default_global_config()
    global_config.num_workers = 4  # Parallel processing

    # Function patterns for flexible processing
    step = FunctionStep(
        func=(normalize_function, {'percentile': 99}),  # Parameterized
        variable_components=[VariableComponents.CHANNEL],  # Process by channel
        group_by=GroupBy.SITE,  # Group by imaging site
        name="Custom Normalization"
    )
```

**Context Engine Sources**:
- openhcs/textual_tui/screens/help_dialog.py (TUI features)
- openhcs/core/steps/function_step.py lines 415-433 (FunctionStep constructor)
- openhcs/constants/constants.py (VariableComponents, GroupBy enums)

### Section 6: Next Steps
**Current (Lines 54-60)**: Broken cross-references

**Rewrite To**:
```rst
Next Steps
----------

Now that you have OpenHCS running, explore these guides:

**For Visual Users:**
- :doc:`../user_guide/tui_guide` - Complete TUI workflow guide
- :doc:`../user_guide/basic_usage` - Common processing patterns

**For Programmers:**  
- :doc:`../user_guide/basic_usage` - Function patterns and pipeline building
- :doc:`../user_guide/intermediate_usage` - Advanced pipeline customization
- :doc:`../concepts/architecture_overview` - System architecture and design

**For Advanced Users:**
- :doc:`../api/pipeline_orchestrator` - Complete API reference
- :doc:`../concepts/gpu_acceleration` - GPU optimization guide
- :doc:`../user_guide/advanced_usage` - Custom function development

Troubleshooting
~~~~~~~~~~~~~~

**GPU Issues:**
- Verify CUDA installation: ``nvidia-smi``
- Check GPU libraries: ``python -c "import cupy; print(cupy.cuda.is_available())"``

**Installation Issues:**
- Use Python 3.11+: ``python --version``
- Update pip: ``pip install --upgrade pip``

**Performance Issues:**
- Enable GPU acceleration: ``pip install -e ".[gpu]"``
- Increase worker count in global config
```

## Implementation Checklist

### Content Verification (Using Context Engine)
- [x] **Installation process**: Verified from setup.py and README.md
- [x] **TUI launch command**: Verified from openhcs/textual_tui/__main__.py  
- [x] **TUI workflow**: Verified from widget implementations
- [x] **Programmatic API**: Verified from orchestrator implementation
- [x] **GPU requirements**: Verified from setup.py GPU dependencies
- [x] **Function patterns**: Verified from FunctionStep implementation

### Cross-Reference Updates Needed
- [ ] Create new file: `docs/source/user_guide/tui_guide.rst`
- [ ] Update: `docs/source/user_guide/basic_usage.rst` (separate plan)
- [ ] Create new file: `docs/source/concepts/gpu_acceleration.rst`

### Estimated Effort
**Writing time**: 4-6 hours  
**Testing time**: 2-3 hours (verify all examples work)  
**Total**: 6-9 hours

### Success Criteria
1. **All code examples work**: 100% functional examples
2. **Both interfaces documented**: TUI + programmatic
3. **Clear progression**: Beginner → intermediate → advanced
4. **Working installation**: Users can successfully install and run
5. **GPU support**: Clear guidance for GPU acceleration setup

This plan provides a complete roadmap for rewriting getting_started.rst with working, verified examples and comprehensive coverage of both user interfaces.
