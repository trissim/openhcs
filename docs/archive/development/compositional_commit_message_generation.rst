Compositional Git Commit Message Generation
==========================================

Systematic methodology for generating comprehensive, technically accurate git commit messages using semantic file grouping and compositional reasoning.

Core Methodology
----------------

Semantic File Grouping
~~~~~~~~~~~~~~~~~~~~~~~

Group modified files by **functional purpose** within the system:

- **Memory Management**: Memory type conversion, stack utilities, memory wrappers
- **Processor Backends**: Backend-specific implementations (cupy, numpy, torch, etc.)
- **Core Pipeline**: Orchestration, step execution, function calling logic
- **Logging & Debugging**: Logging infrastructure, debug utilities, monitoring
- **API Interfaces**: Public APIs, function signatures, contracts
- **Documentation**: README files, API docs, architecture docs

Change Analysis Per Group
~~~~~~~~~~~~~~~~~~~~~~~~~~

For each semantic group:

1. **Read the actual code changes** - Examine the diffs, don't assume
2. **Understand the technical impact** - How do these changes affect system behavior?
3. **Identify the root cause** - What problem was being solved?
4. **Assess scope of impact** - What other parts of the system are affected?

Component Message Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create focused commit message components for each group:

- **Start with the functional area**: "Memory Management:", "CuPy Processor:", etc.
- **Describe the change type**: "Fix signature mismatch", "Add logging", "Refactor interface"
- **Explain the technical details**: What specifically was changed and why
- **Note the impact**: How this affects system behavior or fixes issues

Message Synthesis
~~~~~~~~~~~~~~~~~

Combine all components into a structured commit message:

.. code-block:: text

   <Primary Change Type>: <High-level summary>

   <Detailed description of the main change and its motivation>

   Changes by functional area:

   * <Functional Area 1>: <Component message 1>
   * <Functional Area 2>: <Component message 2>
   * <Functional Area 3>: <Component message 3>

   <Additional context, breaking changes, or follow-up notes if needed>

Example Application
-------------------

Step 1: Identify Modified Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   openhcs/processing/backends/processors/cupy_processor.py
   openhcs/processing/backends/processors/numpy_processor.py
   openhcs/processing/backends/processors/jax_processor.py
   openhcs/processing/backends/processors/pyclesperanto_processor.py
   openhcs/core/memory/stack_utils.py
   openhcs/core/steps/function_step.py

Step 2: Semantic Grouping
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Processor Backends**: cupy_processor.py, numpy_processor.py, jax_processor.py, pyclesperanto_processor.py
- **Memory Management**: stack_utils.py
- **Core Pipeline**: function_step.py

Step 3: Analyze Changes
~~~~~~~~~~~~~~~~~~~~~~

- **Processor Backends**: Fixed create_composite function signatures from List[Array] to single Array
- **Memory Management**: Added comprehensive logging for memory type conversions
- **Core Pipeline**: Added debugging logs for function execution and memory type validation

Step 4: Generate Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Processor Backends**: Fix create_composite signature mismatch across all backends
- **Memory Management**: Add detailed logging for stack/unstack operations and type conversions
- **Core Pipeline**: Add memory type validation and debugging logs for function execution

Step 5: Synthesize Final Message
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   fix: Resolve create_composite function signature mismatch causing cupy processor errors

   Fixed inconsistent function signatures across processor backends where create_composite
   expected different input formats, causing "images must be a list of CuPy arrays" errors.

   Changes by functional area:

   * Processor Backends: Standardize create_composite signature to accept single 3D array
     instead of list of arrays across cupy, numpy, jax, and pyclesperanto processors
   * Memory Management: Add comprehensive logging for stack_slices and unstack_slices
     operations to track memory type conversions between pipeline steps  
   * Core Pipeline: Add memory type validation and debugging logs in function execution
     to help diagnose type conversion issues

   All processors now consistently expect stack: Array of shape (N, Y, X) and return
   composite result of shape (1, Y, X) using weighted averaging across slices.

Benefits
--------

1. **Comprehensive Coverage**: Every change is documented and contextualized
2. **Technical Accuracy**: Messages reflect actual code changes, not assumptions
3. **Logical Organization**: Related changes are grouped for better understanding
4. **Debugging Aid**: Future developers can understand the scope and reasoning
5. **Systems Thinking**: Changes are understood in context of overall architecture

Usage
-----

This methodology should be used for:

- Complex changes spanning multiple files
- Bug fixes that require changes across different system layers
- Refactoring that affects multiple components
- Any change where the scope and impact need to be clearly communicated

For simple, single-file changes, a standard commit message format is sufficient.
