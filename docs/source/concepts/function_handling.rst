.. _function-handling:

=================
Function Handling
=================

Function handling is a key aspect of OpenHCS pipeline configuration. OpenHCS supports flexible function patterns that allow you to compose complex bioimage analysis workflows.

For detailed technical information about the function pattern system, see :doc:`../architecture/function_pattern_system`.
For step configuration details, see :doc:`step`.

The ``FunctionStep`` class supports several patterns for processing functions, providing flexibility in how images are processed. This page provides a concise overview of the available patterns.

.. _function-patterns-overview:

Function Patterns Overview
--------------------------

The ``func`` parameter of the ``FunctionStep`` class can accept several types of values:

1. **Single Function**: A callable that processes 3D image arrays
2. **Function with Arguments**: A tuple of ``(function, kwargs)`` where kwargs is a dictionary of arguments
3. **List of Functions**: A sequence of functions applied one after another (function chains)
4. **Dictionary of Functions**: A mapping from component values to functions, used with ``variable_components``

.. code-block:: python

    from ezstitcher.core.steps import Step
    from ezstitcher.core.image_processor import ImageProcessor as IP
    from ezstitcher.core.utils import stack

    # 1. Single function
    step = Step(
        func=IP.stack_percentile_normalize,
        name="Normalize Images"
    )

    # 2. Function with arguments
    step = Step(
        func=(IP.stack_percentile_normalize, {
            'low_percentile': 0.1,
            'high_percentile': 99.9
        }),
        name="Normalize Images"
    )

    # 3. List of functions
    step = Step(
        func=[
            stack(IP.sharpen),              # First sharpen the images
            IP.stack_percentile_normalize   # Then normalize the intensities
        ],
        name="Enhance Images"
    )

    # 4. Dictionary of functions (with group_by)
    step = Step(
        func={
            "1": process_dapi,      # Apply process_dapi to channel 1
            "2": process_calcein    # Apply process_calcein to channel 2
        },
        name="Channel-Specific Processing",
        group_by='channel'  # Specifies that keys "1" and "2" refer to channel values
    )

.. _function-when-to-use:

When to Use Each Pattern
----------------------

**Pre-defined Steps vs. Raw Step**

For common operations, use the pre-defined step classes instead of raw Step with func parameter:

.. code-block:: python

    from ezstitcher.core.steps import ZFlatStep, CompositeStep

    # RECOMMENDED: Use ZFlatStep for Z-stack flattening
    step = ZFlatStep(method="max")  # Much cleaner than raw Step with variable_components=['z_index']

    # RECOMMENDED: Use CompositeStep for channel compositing
    step = CompositeStep(weights=[0.7, 0.3])  # Much cleaner than raw Step with variable_components=['channel']

**When to use each function pattern:**

1. **Single Function**: Use for simple operations that don't require arguments
2. **Function with Arguments**: Use when you need to customize function behavior with parameters
3. **List of Functions**: Use when you need to apply multiple processing steps in sequence
4. **Dictionary of Functions**: Use for component-specific processing (e.g., different functions for different channels)

**Key Guidelines:**

- For Z-stack flattening, use ``ZFlatStep`` instead of raw Step with variable_components=['z_index']
- For channel compositing, use ``CompositeStep`` instead of raw Step with variable_components=['channel']
- For focus detection, use ``FocusStep`` instead of manually implementing focus detection
- For channel-specific processing, use a dictionary of functions with ``group_by='channel'``
- For custom processing chains, use lists of functions

For detailed information about pre-defined steps, see :ref:`variable-components` in :doc:`step`.

.. _function-stack-utility:

The stack() Utility Function
--------------------------

The ``stack()`` utility function adapts single-image functions to work with stacks of images:

.. code-block:: python

    from ezstitcher.core.utils import stack
    from skimage.filters import gaussian

    # Use stack() to adapt a single-image function to work with a stack
    step = Step(
        func=stack(gaussian),  # Apply gaussian blur to each image in the stack
        name="Gaussian Blur"
    )

**How stack() works**: It takes a function that operates on a single image and returns a new function that applies the original function to each image in a stack.

.. _function-advanced-patterns:

Advanced Patterns
--------------

For advanced use cases, you can combine the basic patterns in various ways:

- Mix functions and function tuples in lists
- Use dictionaries of function tuples
- Create dictionaries of function lists
- Nest stack() calls within tuples or lists

For examples of these advanced patterns, see :doc:`../user_guide/advanced_usage`.

.. _function-best-practices:

Best Practices
------------

- Use pre-defined steps (ZFlatStep, CompositeStep, etc.) for common operations
- Only use raw Step with func parameter when you need custom processing
- Use the simplest pattern that meets your needs
- When using dictionaries, always specify the group_by parameter
- Use descriptive names for your steps to make your code more readable

For comprehensive best practices for function handling, see :ref:`best-practices-function-handling` in the :doc:`../user_guide/best_practices` guide.
