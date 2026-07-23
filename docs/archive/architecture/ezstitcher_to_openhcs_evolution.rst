Architectural Evolution: From EZStitcher to OpenHCS
===================================================

Overview
--------

OpenHCS evolved from its predecessor EZStitcher, transforming from a
CPU-based image stitching tool into a GPU-native scientific computing
platform. This document traces the architectural changes and design
decisions that enabled this transformation.

EZStitcher: The Architectural Foundation
----------------------------------------

Core Innovations That Carried Forward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

EZStitcher established several concepts that remain central to OpenHCS:

1. Pipeline Architecture Hierarchy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   PipelineOrchestrator
   ├── Pipeline (sequence of steps)
   │   ├── Step (single operation)
   │   ├── Step (single operation)
   │   └── Step (single operation)
   └── Pipeline (another sequence)

This hierarchical design enabled complex workflows to be built from
simple, reusable components.

2. Variable Components Pattern
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Intelligent file grouping based on microscopy metadata patterns.

.. code:: python

   # EZStitcher pattern that solved the fundamental problem:
   Step(
       func=create_projection,
       variable_components=['z_index']
   )
   # Groups: Files with same (well, channel, site) but different z_index
   # Result: [z1.tif, z2.tif, z3.tif] → stack → max_projection → single.tif

   Step(
       func=create_composite,
       variable_components=['channel'] 
   )
   # Groups: Files with same (well, site, z_index) but different channel
   # Result: [ch1.tif, ch2.tif] → stack → composite → single.tif

This pattern automatically solves the image grouping problem - how to
intelligently group related images for batch processing without manual
specification of grouping logic.

3. Function Pattern System
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Function Pattern System**: EZStitcher’s most sophisticated innovation
was its function pattern system:

.. code:: python

   # Single function
   func = normalize

   # Function with arguments
   func = (sharpen, {'amount': 1.5})

   # Sequential processing (pipeline within a step)
   func = [
       stack(sharpen),
       normalize,
       equalize_histogram
   ]

   # Component-specific processing
   func = {
       "1": process_dapi,
       "2": process_calcein
   }

   # Semantically valid nested patterns
   func = {
       "1": [                           # Channel 1: sequential processing
           (sharpen, {'amount': 1.5}),
           normalize,
           denoise_dapi
       ],
       "2": [enhance_calcein]           # Channel 2: different processing
   }

   # Note: Nested dictionaries aren't semantically valid
   # (only one group_by per step makes sense)

**The Stack Processing Evolution**: Bridged the gap between single-image
functions and stack-based processing:

.. code:: python

   # EZStitcher approach
   func = stack(gaussian)  # Transforms single-image function to stack-aware

   # OpenHCS evolution: stack_slices/unstack_slices system
   # Automatic per-slice processing with memory type management

4. Specialized Step Types
^^^^^^^^^^^^^^^^^^^^^^^^^

-  **ZFlatStep**: Z-stack flattening with projection methods
-  **CompositeStep**: Multi-channel compositing with weights
-  **PositionGenerationStep**: Tile position calculation
-  **ImageStitchingStep**: Final image assembly

EZStitcher’s Limitations
~~~~~~~~~~~~~~~~~~~~~~~~

Despite its architectural sophistication, EZStitcher hit fundamental
performance and reliability walls:

Performance Bottlenecks
^^^^^^^^^^^^^^^^^^^^^^^

-  **CPU-only processing**: Hundreds of gigabytes processed slowly
-  **Disk I/O between steps**: Every operation read/wrote from disk
-  **Memory inefficiency**: No zero-copy operations
-  **Single memory type**: Only NumPy arrays supported

Reliability Issues
^^^^^^^^^^^^^^^^^^

-  **Silent failures**: Academic code patterns that failed quietly
-  **Basic error handling**: No validation of processing chains
-  **Format brittleness**: Microscope-specific code paths

OpenHCS: The Architectural Revolution
-------------------------------------

Revolutionary Design Principles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenHCS didn’t just port EZStitcher to GPU - it fundamentally reimagined
scientific computing architecture:

1. Memory Type System
^^^^^^^^^^^^^^^^^^^^^

**Innovation**: Explicit memory type contracts with automatic
conversion.

.. code:: python

   @torch_func  # Function declares it works with PyTorch tensors
   def n2v2_denoise_torch(image: torch.Tensor) -> torch.Tensor:
       # Function receives pre-converted tensor on correct device
       device = image.device  # No device management needed
       return denoised_tensor

   @cupy_func   # Function declares it works with CuPy arrays  
   def gpu_ashlar_align_cupy(images: cp.ndarray) -> cp.ndarray:
       # Function receives pre-converted CuPy array
       return aligned_images

**Benefits**: - Functions focus on algorithms, not memory management -
Automatic conversion between memory types (CuPy ↔ PyTorch ↔ NumPy) -
Zero-copy GPU operations where possible - Compile-time validation of
memory compatibility

2. Zero-Copy GPU Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Innovation**: DLPack-based memory conversions for true zero-copy
performance.

.. code:: python

   # Before (EZStitcher): CPU roundtrip
   cupy_array → numpy_array → torch_tensor  # 2 copies, GPU→CPU→GPU

   # After (OpenHCS): Direct GPU transfer  
   cupy_array → torch_tensor  # 0 copies, GPU→GPU via DLPack

**Impact**: Orders of magnitude performance improvement for large
datasets.

3. Fail-Loudly Philosophy
^^^^^^^^^^^^^^^^^^^^^^^^^

**Innovation**: No silent degradation, explicit error handling.

.. code:: python

   # OpenHCS principle: Explicit failure over silent degradation
   def _cupy_to_torch(data, allow_cpu_roundtrip=False):
       if not allow_cpu_roundtrip:
           raise MemoryConversionError("GPU conversion failed")
       # Never silently fall back to CPU

**Contrast with Academic Code**: - Academic: Silent CPU fallback when
GPU fails - OpenHCS: Loud failure with clear error messages

4. Smell-Loop Validation
^^^^^^^^^^^^^^^^^^^^^^^^

**Innovation**: Architectural review process preventing technical debt.

::

   Plan File → Smell Review → Implementation → Validation

**Purpose**: Prevent the architectural rot that plagued EZStitcher
extensions.

5. Pipeline Compiler
^^^^^^^^^^^^^^^^^^^^

**Innovation**: Pre-execution validation of entire processing chains.

.. code:: python

   # Validates memory type compatibility before execution
   compiled_contexts = orchestrator.compile_pipelines(
       pipeline_definition=pipeline.steps,
       well_filter=wells
   )
   # Fails fast if CuPy→PyTorch conversion not supported

Architectural Continuity
~~~~~~~~~~~~~~~~~~~~~~~~

**What OpenHCS Preserved from EZStitcher**: - Pipeline → Step hierarchy
(proven architecture) - Variable components pattern (intelligent grouping
logic) - Group-by functionality (channel-specific processing) - Modular
step design (composable workflows)

**What OpenHCS Changed**: - Memory management (explicit types vs
implicit NumPy) - Error handling (fail loudly vs silent failures) -
Performance (GPU-native vs CPU-only) - Validation (compile-time checks
vs runtime surprises) - Function ecosystem (unified GPU library access
vs manual integration)

Key Innovations and Differentiators
-----------------------------------

OpenHCS introduces several new systems that make it
fundamentally different from traditional scientific computing tools.
Each system is documented in detail in dedicated architecture documents:

🔥 `Function Registry System <function_registry_system.rst>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Unified GPU function ecosystem with type-safe contracts**

The most comprehensive GPU imaging function ecosystem in scientific
computing, automatically discovering and unifying functions from
pyclesperanto, scikit-image, CuCIM, and other libraries with consistent
interfaces and memory type safety.

🖥️ `TUI System <tui_system.rst>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Advanced terminal interface**

A sophisticated Textual-based interface that works anywhere a terminal
works - unprecedented for scientific computing tools. Includes real-time
pipeline editing, live configuration management, integrated help, and
professional log monitoring.

💾 `Storage and Memory System <storage_and_memory_system.rst>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Intelligent data management for 100GB+ datasets**

Advanced Virtual File System with memory overlay capabilities, OME-ZARR
compression, and smart backend switching that automatically scales from
small experiments to massive high-content screening datasets.

⚡ `Memory Type System <memory_type_system.rst>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Zero tolerance for silent failures**

Comprehensive architecture that prevents the silent failures plaguing
academic software through explicit validation, mandatory contracts, and
clear error handling with actionable solutions.

🧬 `External Integrations Overview <external_integrations_overview.rst>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Production neuroscience research deployment**

Real-world deployment handling 100GB+ datasets in production
neuroscience research with seamless integration with Napari, Fiji, and OMERO.

These innovations work together to create a scientific computing
platform that is fundamentally different from traditional academic tools
- providing enterprise-level reliability, unprecedented scale handling,
and comprehensive GPU acceleration in a unified, user-friendly
interface.

The Collaborative AI Innovation
-------------------------------

Leveraging LLM Architectural Knowledge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The evolution from EZStitcher to OpenHCS represents a unique development
methodology:

**Traditional Approach**: Domain expert → learns software engineering →
builds tool **OpenHCS Approach**: Domain expert + AI architectural
knowledge → builds production system

Key Collaborative Patterns
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Architectural Guidance**: AI provides software engineering best
   practices
2. **Pattern Recognition**: AI identifies anti-patterns and suggests
   improvements
3. **Implementation Support**: AI helps translate architectural vision
   into code
4. **Debugging Partnership**: Systematic problem-solving combining
   domain and technical expertise

Example: Memory Type System Design
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   Human: "I need GPU processing but different libraries use different array types"
   AI: "Consider explicit memory type contracts with automatic conversion"
   Human: "How do I prevent silent CPU fallbacks?"
   AI: "Use decorators to declare memory requirements and fail loudly on violations"
   Result: @torch_func/@cupy_func decorator system

Methodological Innovation
~~~~~~~~~~~~~~~~~~~~~~~~~

| This represents a new model for scientific software development: -
  **Domain expert drives architectural vision** - **AI provides software
  engineering expertise**
| - **Iterative refinement through collaborative debugging** -
  **Real-time knowledge transfer from AI to human**

Impact and Significance
-----------------------

Technical Impact
~~~~~~~~~~~~~~~~

-  **Performance**: Orders of magnitude improvement through GPU-native
   processing
-  **Reliability**: Fail-loudly philosophy prevents silent data
   corruption
-  **Extensibility**: Memory type system enables easy addition of new
   processing functions
-  **Interoperability**: Format abstraction handles any microscope
   vendor

Scientific Impact
~~~~~~~~~~~~~~~~~

-  **Reproducibility**: Explicit validation prevents pipeline failures
-  **Accessibility**: Open-source alternative to expensive commercial
   solutions
-  **Innovation**: Enables new research through reliable, fast
   processing

Methodological Impact
~~~~~~~~~~~~~~~~~~~~~

-  **Collaborative AI Development**: Proves domain expert + AI can build
   production systems
-  **Architectural Discipline**: Shows how to prevent technical debt in
   scientific software
-  **Knowledge Transfer**: Demonstrates AI-assisted learning of software
   engineering

Lessons for Scientific Computing
--------------------------------

Architectural Principles
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Explicit over implicit**: Declare requirements clearly (memory
   types, device placement)
2. **Fail loudly over silent degradation**: Better to crash than produce
   wrong results
3. **Validation over hope**: Check compatibility before execution, not
   during
4. **Modularity over monoliths**: Composable components enable flexible
   workflows

Development Methodology
~~~~~~~~~~~~~~~~~~~~~~~

1. **Collaborative AI partnership**: Leverage AI architectural knowledge
2. **Iterative refinement**: Build, test, improve through systematic
   debugging
3. **Domain-driven design**: Let research needs drive architectural
   decisions
4. **Production mindset**: Build for reliability, not just functionality

Future Evolution
----------------

OpenHCS establishes patterns that could transform scientific computing:

Technical Directions
~~~~~~~~~~~~~~~~~~~~

-  **Multi-GPU orchestration**: Scale to larger datasets
-  **Cloud-native deployment**: Enable distributed processing
-  **Real-time processing**: Support live microscopy workflows
-  **Advanced validation**: Deeper architectural integrity checks

Methodological Directions
~~~~~~~~~~~~~~~~~~~~~~~~~

-  **AI-assisted architecture**: Deeper integration of AI in design
   decisions
-  **Collaborative debugging**: Systematic approaches to complex
   problem-solving
-  **Knowledge preservation**: Document architectural decisions and
   reasoning
-  **Community development**: Enable other researchers to contribute
   effectively

--------------

Conclusion
----------

The evolution from EZStitcher to OpenHCS demonstrates that effective
scientific software can emerge from the combination of:

1. **Deep domain expertise** (understanding real research problems)
2. **Architectural vision** (seeing beyond immediate needs)
3. **Collaborative AI development** (leveraging AI software engineering
   knowledge)
4. **Systematic methodology** (disciplined approach to complex problems)

OpenHCS proves that researchers don’t need to become software engineers
- they need to become effective collaborators with AI systems that have
architectural expertise.

**The result**: Enterprise-level scientific computing infrastructure
that enables better research through better tools.

--------------

*“The best software comes not from software engineers, but from
researchers who refuse to accept that their tools have to suck.”* -
OpenHCS Origin Story
