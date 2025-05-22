.. _best-practices:

===============================================
Best Practices
===============================================

This page consolidates best practices for using EZStitcher effectively. It's organized by topic to help you find the advice you need quickly.

Sections:

1. :ref:`best-practices-ez-module` - EZ module cheatsheet
2. :ref:`best-practices-pipeline` - Pipeline creation and configuration
3. :ref:`best-practices-steps` - Step configuration and usage
4. :ref:`best-practices-directory` - Directory management
5. :ref:`best-practices-function-handling` - Function handling patterns
6. :ref:`best-practices-custom-pipelines` - Custom pipeline examples

.. _best-practices-ez-module:

----------------------------------------
EZ Module in one minute
----------------------------------------

*Use for quick results with minimal code.* Only four parameters matter 90% of the time:

+ ``normalize``              → percentile stretch (1/99 by default)
+ ``flatten_z`` + ``z_method`` → convert stacks ("max", "mean", "combined" …)
+ ``channel_weights``        → which channels build reference composite

.. code-block:: python

   from ezstitcher import stitch_plate

   # Basic usage
   stitch_plate("path/to/plate")

   # With options
   stitch_plate(
       "path/to/plate",
       normalize=True,
       flatten_z=True,
       z_method="max",          # or "combined" for focus metric
       channel_weights=[0.7, 0.3, 0]
   )

*Rule of thumb* → use the EZ module for standard workflows; create custom pipelines when you need more control.

.. _best-practices-pipeline:

----------------------------------------
Pipeline Best Practices
----------------------------------------

**Pipeline Creation and Configuration**

✔ Start with **ZFlatStep → Normalize → Composite → Position → Stitch** and add/remove steps from there. See :doc:`../concepts/step` for details on different step types.

✔ Wrap repeated code in a *factory function* so notebooks stay clean.

✔ Name your pipeline (`name="Plate A - Max proj"`)—the logger shows it.

✘ Avoid inserting steps after PositionGenerationStep. It is only compatible with ImageStitchingStep.


**Pipeline Structure Best Practices**

1. **Create separate pipelines for different tasks**:
   - Position generation pipeline: process → composite → generate positions
   - Assembly pipeline: process → stitch
   - Analysis pipeline (optional): analyze stitched images

2. **Use consistent pipeline structure**:
   - Keep similar pipelines structured the same way
   - This makes code more maintainable and easier to understand

3. **Parameterize your pipelines**:
   - Create factory functions that take parameters
   - This allows you to reuse pipeline configurations across projects

4. **Save pipeline configurations as Python scripts**:
   - This allows you to version control your pipeline configurations
   - Makes it easy to reproduce results

For detailed information on pipeline construction, see :doc:`../concepts/pipeline`.

.. _best-practices-directory:

----------------------------------------
Directory Management Best Practices
----------------------------------------

**Basic Directory Guidelines**

* First step → `input_dir=orchestrator.workspace_path`.
* Omit `output_dir` unless you truly need it; EZStitcher auto‑chains.
* Use `pipeline.output_dir` when another script needs the results.

**Directory Structure Best Practices**

1. **Use the workspace path for the first step**:
   - Always use `orchestrator.workspace_path` as the input directory for the first step
   - This ensures that original data is protected from modification

2. **Specify output_dir only when you need a specific directory structure**:
   - For example, when you need to save results in a specific location
   - When you need to reference the output directory from outside the pipeline

3. **Don't specify input_dir for subsequent steps**:
   - Each step's output directory automatically becomes the next step's input directory
   - This reduces code verbosity and potential for errors

4. **Don't specify directories for steps unless needed**:
   - `PositionGenerationStep` and `ImageStitchingStep` have intelligent directory handling
   - They automatically find the right directories based on the pipeline context

5. **Use consistent directory naming**:
   - Follow the default naming conventions when possible
   - Or configure custom suffixes through PipelineConfig for consistent naming

For detailed information on directory handling, see :doc:`../concepts/directory_structure`.

.. _best-practices-steps:

----------------------------------------
Step Configuration and Usage Best Practices
----------------------------------------

**Recommended Step Order (Golden Path)**

1. **ZFlatStep / FocusStep**  - reduce stacks.
2. **Channel processing + CompositeStep** - build reference image.
3. **PositionGenerationStep** - writes CSV.
4. **ImageStitchingStep**     - uses CSV.

Anything else is an optimisation *before* or *between* 1-2.

**Step Parameter Configuration**

1. **Use Descriptive Names**:
   - Choose clear, descriptive names for your steps
   - This makes pipelines easier to understand and debug

2. **Variable Components**:
   - Use ``ZFlatStep`` instead of setting ``variable_components=['z_index']`` for Z-stack flattening
   - Use ``CompositeStep`` instead of setting ``variable_components=['channel']`` for channel compositing
   - Leave at default ``['site']`` for most other operations
   - Only set ``variable_components`` directly when you have a specific need not covered by specialized steps

3. **Directory Management**:
   - Always specify ``input_dir`` for the first step, using ``orchestrator.workspace_path``
   - Let EZStitcher handle directory resolution for subsequent steps
   - Only specify ``output_dir`` when you need a specific directory structure

4. **Parameter Validation**:
   - Ensure ``group_by`` is never the same as ``variable_components``
   - Only use ``group_by`` with dictionary functions
   - Verify that all required parameters are specified

**When to Use Specialized Steps**

For common operations, use specialized steps that encapsulate the appropriate configuration:

1. **ZFlatStep**: Use for Z-stack flattening instead of manually configuring ``variable_components=['z_index']``
2. **FocusStep**: Use for focus detection in Z-stacks
3. **CompositeStep**: Use for channel compositing instead of manually configuring ``variable_components=['channel']``

These steps provide cleaner, more readable code and ensure proper configuration. Use them with minimal parameters unless you need to override defaults.

.. _best-practices-function-handling:

----------------------------------------
Function Handling Best Practices
----------------------------------------

**Core Principle**: Always "stack-in / stack-out"—each function receives a list of images and returns a list of the **same length**.

**Function Patterns**

| Pattern     | Example                                                       | When to Use |
|-------------|---------------------------------------------------------------|-------------|
| Single fn   | `Step(func=IP.stack_percentile_normalize)`                    | When you need to apply the same processing to all images with default parameters |
| Fn + kwargs | `Step(func=(IP.tophat, {'size':15}))`                         | When you need to apply a single function with specific parameters |
| Chain       | `Step(func=[(IP.tophat,{'size':15}), IP.stack_percentile_normalize])` | When you need to apply multiple processing steps in sequence |
| Per-channel | `Step(func={'1': proc_dapi, '2': proc_gfp}, group_by='channel')` | When you need to apply different processing to different channels |

**The stack() Utility Function**

Use the `stack()` utility function to adapt single-image functions to work with stacks of images:

```python
from ezstitcher.core.utils import stack
from skimage.filters import gaussian

# Use stack() to adapt a single-image function to work with a stack
step = Step(
    name="Gaussian Blur",
    func=stack(gaussian),  # Apply gaussian blur to each image in the stack
)
```

For detailed explanation and examples of the `stack()` utility function, see :ref:`function-stack-utility` in :doc:`../concepts/function_handling`.


.. _best-practices-custom-pipelines:

----------------------------------------
Custom Pipeline Best Practices
----------------------------------------

When creating custom pipelines:

1. **Use specialized steps for common operations**:
   - ``ZFlatStep`` for Z-stack flattening
   - ``CompositeStep`` for channel compositing
   - ``PositionGenerationStep`` and ``ImageStitchingStep`` for stitching

2. **Leverage functional programming patterns**:
   - Use the ``func`` parameter to pass processing functions
   - Compose complex operations with multiple steps
   - Use ``variable_components`` and ``group_by`` for fine-grained control

3. **Follow a consistent pipeline structure**:
   - Position generation pipeline: process → composite → generate positions
   - Assembly pipeline: process → stitch
   - Analysis pipeline (optional): analyze stitched images

Example of a well-structured custom pipeline:

.. code-block:: python

   # Position generation pipeline
   pos_pipe = Pipeline(
       input_dir=orchestrator.workspace_path,
       steps=[
           ZFlatStep(),
           NormStep(),
           CompositeStep(weights=[0.7, 0.3, 0]),
           PositionGenerationStep(),
       ],
       name="Position Generation",
   )
   positions_dir = pos_pipe.steps[-1].output_dir

   # Assembly pipeline
   asm_pipe = Pipeline(
       input_dir=orchestrator.workspace_path,
       output_dir=plate_path.parent / f"{plate_path.name}_stitched",
       steps=[
           NormStep(),
           ImageStitchingStep(positions_dir=positions_dir),
       ],
       name="Assembly",
   )

--------------------------------------------------------------------
Need more depth?
--------------------------------------------------------------------

* :doc:`../concepts/pipeline`
* :doc:`../concepts/directory_structure`
* :doc:`../concepts/step`
* :doc:`../concepts/function_handling`
