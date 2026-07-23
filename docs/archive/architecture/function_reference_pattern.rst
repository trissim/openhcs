================================
FunctionReference Pattern
================================

*Module: openhcs.core.pipeline.compiler, openhcs.formats.func_arg_prep*
*Status: STABLE*

---

Overview
========

Function patterns can contain either function names (strings) or FunctionReference objects. The FunctionReference preservation pattern ensures that function metadata (backend, tags, validation state) is maintained throughout pattern traversal and compilation.

Problem Context
===============

Traditional pattern traversal loses function metadata:

.. code-block:: python

    # Pattern with FunctionReference
    pattern = {
        'gaussian_blur': FunctionReference(
            name='gaussian_blur',
            backend='cupy',
            tags=['gpu', 'filtering'],
            validated=True
        )
    }

    # Traditional traversal extracts only name
    for func_name in pattern.keys():
        func = get_function(func_name)  # Lost backend, tags, validation!

Without metadata preservation, the compiler must re-discover function backends and re-validate, adding overhead and potential inconsistencies.

Solution: FunctionReference Injection
======================================

The compiler injects FunctionReference objects into patterns, preserving metadata:

.. code-block:: python

    from openhcs.core.pipeline.compiler import inject_function_references

    # Original pattern (strings only)
    pattern = {
        'gaussian_blur': {'sigma': 2.0},
        'threshold': {'value': 0.5}
    }

    # Inject FunctionReference objects
    enriched_pattern = inject_function_references(pattern)

    # Result: FunctionReference objects with metadata
    enriched_pattern = {
        FunctionReference(name='gaussian_blur', backend='cupy', ...): {'sigma': 2.0},
        FunctionReference(name='threshold', backend='numpy', ...): {'value': 0.5}
    }

**Key insight**: FunctionReference objects serve as enriched keys, carrying metadata through pattern traversal.

FunctionReference Structure
============================

FunctionReference Dataclass
----------------------------

.. code-block:: python

    @dataclass(frozen=True)
    class FunctionReference:
        """Immutable reference to a registered function with metadata."""

        name: str
        """Function name (e.g., 'gaussian_blur')"""

        backend: str
        """Backend name (e.g., 'cupy', 'numpy', 'skimage')"""

        tags: Tuple[str, ...]
        """Function tags (e.g., ('gpu', 'filtering'))"""

        validated: bool = False
        """Whether function has been validated"""

        signature: Optional[inspect.Signature] = None
        """Function signature for parameter validation"""

        def __hash__(self):
            """Hash based on name and backend for dict key usage."""
            return hash((self.name, self.backend))

        def __eq__(self, other):
            """Equality based on name and backend."""
            if isinstance(other, FunctionReference):
                return self.name == other.name and self.backend == other.backend
            elif isinstance(other, str):
                return self.name == other
            return False

**Design decisions**:

- **Frozen**: Immutable to enable dict key usage
- **Hashable**: Based on name + backend for dict keys
- **String equality**: Enables comparison with function names

Pattern Traversal Without iter_pattern_items
=============================================

The ``iter_pattern_items`` utility was removed to eliminate scattered traversal logic. Pattern traversal now happens inline:

Old Pattern (Removed)
----------------------

.. code-block:: python

    from openhcs.core.pipeline.pipeline_utils import iter_pattern_items

    # Scattered traversal logic
    for func, params in iter_pattern_items(pattern):
        process_function(func, params)

New Pattern (Current)
---------------------

.. code-block:: python

    # Inline traversal with FunctionReference preservation
    if isinstance(pattern, dict):
        for func_ref, params in pattern.items():
            # func_ref is FunctionReference with metadata
            process_function(func_ref, params)
    elif isinstance(pattern, list):
        for func_ref in pattern:
            # func_ref is FunctionReference
            process_function(func_ref, {})
    else:
        # Single function
        process_function(pattern, {})

**Why remove iter_pattern_items?**

- Centralized traversal logic in compiler
- Eliminated duplicate traversal implementations
- Simplified pattern handling with FunctionReference

Injection and Preservation Flow
================================

Compilation Phase
-----------------

.. code-block:: python

    class PipelineCompiler:
        def compile_step(self, step):
            """Compile step with FunctionReference injection."""

            # Get function pattern from step
            pattern = step.function

            # Inject FunctionReference objects
            enriched_pattern = self._inject_function_references(pattern)

            # Store enriched pattern in compiled step plan
            step_plan['function_pattern'] = enriched_pattern

            return step_plan

        def _inject_function_references(self, pattern):
            """Recursively inject FunctionReference into pattern."""

            if isinstance(pattern, dict):
                enriched = {}
                for func_name, params in pattern.items():
                    # Look up function in registry
                    func_obj = get_function(func_name)

                    # Create FunctionReference with metadata
                    func_ref = FunctionReference(
                        name=func_name,
                        backend=func_obj.backend,
                        tags=tuple(func_obj.tags),
                        validated=True,
                        signature=inspect.signature(func_obj)
                    )

                    enriched[func_ref] = params

                return enriched

            # Handle list and single function patterns similarly
            ...

Execution Phase
---------------

.. code-block:: python

    class FunctionStep:
        def process(self, context):
            """Execute step with FunctionReference metadata."""

            # Get enriched pattern from compiled plan
            pattern = context.step_plan['function_pattern']

            # Traverse pattern with FunctionReference objects
            for func_ref, params in pattern.items():
                # func_ref has backend, tags, signature

                # Use metadata for optimization
                if 'gpu' in func_ref.tags:
                    allocate_gpu_memory()

                # Execute function
                result = self._execute_function(func_ref, params)

**Key insight**: FunctionReference metadata enables execution optimizations without re-querying registry.

Pattern Argument Preparation
=============================

The ``func_arg_prep`` module uses FunctionReference for parameter validation:

.. code-block:: python

    from openhcs.formats.func_arg_prep import prepare_function_arguments

    def prepare_function_arguments(func_ref, user_params, context):
        """Prepare function arguments with signature validation."""

        # Use FunctionReference signature for validation
        sig = func_ref.signature

        # Validate user parameters against signature
        bound_args = sig.bind_partial(**user_params)

        # Inject context parameters
        if 'dtype_config' in sig.parameters:
            bound_args.arguments['dtype_config'] = context.dtype_config

        return bound_args.arguments

**Why use FunctionReference signature?**

- Avoids re-inspecting function object
- Signature is immutable (frozen dataclass)
- Validation is consistent across compilation and execution

Common Patterns
===============

Dict Pattern with FunctionReference
------------------------------------

.. code-block:: python

    # User defines pattern with strings
    pattern = {
        'gaussian_blur': {'sigma': 2.0},
        'threshold': {'value': 0.5}
    }

    # Compiler injects FunctionReference
    enriched = {
        FunctionReference(name='gaussian_blur', backend='cupy'): {'sigma': 2.0},
        FunctionReference(name='threshold', backend='numpy'): {'value': 0.5}
    }

    # Execution uses FunctionReference metadata
    for func_ref, params in enriched.items():
        if func_ref.backend == 'cupy':
            use_gpu_execution(func_ref, params)
        else:
            use_cpu_execution(func_ref, params)

List Pattern with FunctionReference
------------------------------------

.. code-block:: python

    # User defines list pattern
    pattern = ['gaussian_blur', 'threshold', 'erode']

    # Compiler injects FunctionReference
    enriched = [
        FunctionReference(name='gaussian_blur', backend='cupy'),
        FunctionReference(name='threshold', backend='numpy'),
        FunctionReference(name='erode', backend='skimage')
    ]

    # Execution uses FunctionReference metadata
    for func_ref in enriched:
        execute_function(func_ref)

Single Function with FunctionReference
---------------------------------------

.. code-block:: python

    # User defines single function
    pattern = 'gaussian_blur'

    # Compiler injects FunctionReference
    enriched = FunctionReference(name='gaussian_blur', backend='cupy')

    # Execution uses FunctionReference metadata
    execute_function(enriched)

Implementation Notes
====================

**🔬 Source Code**:

- FunctionReference injection: ``openhcs/core/pipeline/compiler.py`` (line 72)
- Argument preparation: ``openhcs/formats/func_arg_prep.py`` (line 21)
- Path planner integration: ``openhcs/core/pipeline/path_planner.py`` (line 101)

**🏗️ Architecture**:

- :doc:`function-registry-system` - Function registry architecture
- :doc:`pipeline-compilation-system` - Compilation flow
- :doc:`function-pattern-system-unified` - Pattern system

**📊 Performance**:

- FunctionReference creation: < 1ms per function
- Metadata lookup avoided during execution
- Signature validation cached in FunctionReference

Key Design Decisions
====================

**Why use FunctionReference as dict keys?**

Enables metadata preservation while maintaining dict pattern semantics. FunctionReference is hashable and comparable to strings.

**Why remove iter_pattern_items?**

Centralized pattern traversal in compiler eliminates duplicate logic and ensures consistent FunctionReference handling.

**Why freeze FunctionReference?**

Immutability enables dict key usage and prevents accidental modification of function metadata.

Common Gotchas
==============

- **FunctionReference equality with strings**: ``func_ref == 'gaussian_blur'`` works, but ``'gaussian_blur' == func_ref`` may not
- **Dict keys are FunctionReference**: Pattern traversal must handle FunctionReference keys, not strings
- **Signature is optional**: Not all FunctionReference objects have signature (e.g., dynamically registered functions)
- **Backend is required**: FunctionReference must have backend for proper execution routing

Debugging FunctionReference Issues
===================================

Symptom: Function Not Found
----------------------------

**Cause**: FunctionReference name doesn't match registry

**Diagnosis**:

.. code-block:: python

    # Check FunctionReference name
    logger.debug(f"Looking for function: {func_ref.name}")

    # Check registry
    available = list_functions(backend=func_ref.backend)
    logger.debug(f"Available functions: {available}")

**Fix**: Ensure FunctionReference name matches registered function name

Symptom: Wrong Backend Used
----------------------------

**Cause**: FunctionReference backend incorrect

**Diagnosis**:

.. code-block:: python

    # Check FunctionReference backend
    logger.debug(f"Function backend: {func_ref.backend}")

    # Check actual function backend
    func = get_function(func_ref.name)
    logger.debug(f"Actual backend: {func.backend}")

**Fix**: Ensure FunctionReference injection uses correct backend from registry

Symptom: Signature Validation Fails
------------------------------------

**Cause**: FunctionReference signature out of sync with actual function

**Diagnosis**:

.. code-block:: python

    # Compare signatures
    logger.debug(f"FunctionReference signature: {func_ref.signature}")

    func = get_function(func_ref.name)
    actual_sig = inspect.signature(func)
    logger.debug(f"Actual signature: {actual_sig}")

**Fix**: Re-inject FunctionReference to update signature
