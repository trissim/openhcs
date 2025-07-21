============================
Core Concepts of OpenHCS
============================

Introduction to Pipeline Processing
-----------------------------------

OpenHCS is a powerful, flexible, and scalable pipeline processing system designed for complex, multi-step data workflows, with a primary focus on high-content screening (HCS) microscopy data. While its origins are in image stitching and analysis, its architecture is fundamentally general-purpose.

At its core, OpenHCS allows you to define a series of processing **steps**, which are organized into a **pipeline**. This pipeline can be executed on large, distributed datasets with features like:

-  **Virtual File System (VFS):** An abstraction layer that allows seamless switching between in-memory processing and on-disk materialization, optimizing for speed and resource usage.
-  **GPU Acceleration:** Many core functions are GPU-accelerated, enabling high-throughput processing of large images and datasets.
-  **Asynchronous Execution:** A modern `asyncio`-based engine manages concurrent operations, maximizing efficiency.
-  **Flexible Function Patterns:** A sophisticated system for defining processing logic, from simple function calls to complex, parallel workflows.

While OpenHCS can be adapted for many domains, this guide will use microscopy to illustrate its core concepts.

Key OpenHCS Concepts Illustrated with Microscopy
-----------------------------------------------

### The Pipeline: Organizing Workflows

In OpenHCS, all work is organized into a ``PipelineOrchestrator`` that manages a sequence of ``FunctionStep`` objects. This is the heart of the system.

-  **PipelineOrchestrator:** The main engine that manages data, executes steps, and handles the VFS. It is initialized with a path to the data (e.g., a microscopy plate) and a global configuration.
-  **FunctionStep:** A single unit of work in the pipeline. Each step encapsulates a Python function or a more complex "function pattern" and has its own configuration, such as a name and the data components it operates on.

### The Data Hierarchy: From Plates to Sites

In HCS, experimental data is structured hierarchically. OpenHCS understands this hierarchy, allowing you to execute steps at different levels.

-  **Plate:** The top-level container, representing, for example, a 96-well plate. The ``PipelineOrchestrator`` is typically associated with a single plate.
-  **Well:** A single compartment within a plate (e.g., A01, B02).
-  **Site (or Tile):** A specific imaging location within a well. In a tiled acquisition, a well may contain a grid of many sites (e.g., a 3x3 grid).

OpenHCS uses the ``variable_components`` parameter in a ``FunctionStep`` to determine which axis of the data to iterate over. For example, setting it to `[VariableComponents.SITE]` means the step's function will be called for each site in a well, receiving that site's corresponding image data. This allows for fine-grained control over how data is processed through the pipeline.

### Function Patterns: Defining the Work

The ``func`` parameter of a ``FunctionStep`` is its most important attribute. It defines what the step actually *does*. The power of OpenHCS comes from the flexibility of this parameter.

**1. Single Function Call:**
The simplest pattern is a single Python function. When combined with `variable_components=[VariableComponents.SITE]`, for example, this step will execute `my_function` on the image data from each site individually.

.. code-block:: python

   from openhcs.core.steps import FunctionStep
   from my_package import my_function

   step = FunctionStep(func=my_function, name="my_step")

**2. A Chain of Functions (List):**
You can provide a list of functions. OpenHCS will execute them sequentially, with the output of one function becoming the input to the next.

.. code-block:: python

   step = FunctionStep(func=[normalize_image, apply_filter], name="preprocess")

**3. Functions with Arguments (Tuple):**
To pass specific arguments to a function, use a tuple of `(function, kwargs_dict)`.

.. code-block:: python

   step = FunctionStep(
       func=[
           (normalize_image, {'low_percentile': 1.0, 'high_percentile': 99.0}),
           (tophat_filter, {'radius': 50})
       ],
       name="advanced_preprocess"
   )

**4. Channel-Specific Logic (Dictionary):**
For multi-channel imaging, you can define different processing chains for different channels using a dictionary, where keys correspond to channel indices. This allows for powerful, targeted workflows.

.. code-block:: python

   step = FunctionStep(
       func={
           '0': [(count_nuclei, {'threshold': 0.8})],  # DAPI channel
           '1': [(analyze_neurites, {'min_length': 10})] # Tubulin channel
       },
       name="channel_analysis"
   )

### Application to Microscopy Workflows

**Z-Stacks (3D Imaging):**
Z-stacks are handled within functions. For example, a `FunctionStep` could take a 3D Z-stack as input and perform a **maximum intensity projection** to convert it to a 2D image before further processing. OpenHCS provides several built-in functions for these common operations.

**Image Stitching:**
Stitching is a multi-step process in OpenHCS:
1.  A `FunctionStep` runs on each **Site** to calculate tile positions (e.g., using `ashlar_compute_tile_positions_gpu`).
2.  The calculated positions are stored by the `PipelineOrchestrator`.
3.  Another `FunctionStep` uses these positions to assemble the final stitched image for each well (e.g., using `assemble_stack_cupy`).

This modular, step-based approach makes complex workflows like stitching manageable, debuggable, and highly customizable.
