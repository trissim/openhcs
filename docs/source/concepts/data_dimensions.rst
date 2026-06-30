Data Dimensions
===============

High-content microscopy data varies over semantic dimensions such as wells,
sites, channels, Z planes, and timepoints. OpenHCS keeps these dimensions as
typed workflow semantics instead of leaving each pipeline to parse filenames
manually.

Microscopy Data Dimensions
--------------------------

Typical plate data may look like this:

.. code-block:: text

   Plate/
   ├── A01_s1_w1.tif    # Well A01, Site 1, Channel 1
   ├── A01_s1_w2.tif    # Well A01, Site 1, Channel 2
   ├── A01_s2_w1.tif    # Well A01, Site 2, Channel 1
   ├── A01_s2_w2.tif    # Well A01, Site 2, Channel 2
   └── A02_s1_w1.tif    # Well A02, Site 1, Channel 1

Core dimensions:

* **Well**: sample position or experimental condition, such as ``A01``.
* **Site**: field of view within a well.
* **Channel**: fluorescence or imaging channel.
* **Z index**: optical plane in a stack.
* **Timepoint**: acquisition time in a live or longitudinal experiment.

Microscope handlers and metadata handlers expose these facts through typed
inventory and source-binding records. Agents should inspect those records before
authoring a workflow.

Variable Components
-------------------

``variable_components`` declares which semantic axes are stacked into each
callable input array.

Use it through ``LazyProcessingConfig`` on a ``FunctionStep``:

.. code-block:: python

   from openhcs.constants.constants import VariableComponents
   from openhcs.core.config import LazyProcessingConfig
   from openhcs.core.steps.function_step import FunctionStep

   step = FunctionStep(
       func=(normalize_images, {}),
       name="normalize",
       processing_config=LazyProcessingConfig(
           variable_components=[VariableComponents.SITE],
       ),
   )

In this example, site is the varying axis inside the callable's input stack.
The compiler uses the source inventory and processing config to build the work
units.

Common choices:

* ``VariableComponents.SITE``: process site stacks; this is the default mental
  model for many plate workflows.
* ``VariableComponents.CHANNEL``: stack or operate across channel variation.
* ``VariableComponents.Z_INDEX``: stack Z planes for volumetric or projection
  operations.
* ``VariableComponents.TIMEPOINT``: stack timepoints for temporal analysis.
* multiple components: stack a combination, such as site plus channel, when the
  callable's declared semantics require it.

If ``variable_components`` is empty, a callable cannot receive a meaningful
third-axis stack from source variation. In practice, use an explicit processing
config for functions that require channel, site, Z, or time semantics.

Group By
--------

``group_by`` is the routing or fanout axis for dictionary function patterns. It
does not mean "stack this axis". Stacking is owned by ``variable_components``.

Example: route different channels to different functions while each callable
still receives site-variable input:

.. code-block:: python

   from openhcs.constants.constants import GroupBy, VariableComponents
   from openhcs.core.config import LazyProcessingConfig
   from openhcs.core.steps.function_step import FunctionStep

   step = FunctionStep(
       func={
           "1": (analyze_nuclei, {}),
           "2": (analyze_neurites, {}),
       },
       name="channel_analysis",
       processing_config=LazyProcessingConfig(
           variable_components=[VariableComponents.SITE],
           group_by=GroupBy.CHANNEL,
       ),
   )

The dictionary keys are matched against the group-by component values. The
compiler prepares the grouped execution plan; runtime execution should consume
that plan rather than rediscovering routing rules.

Source Bindings And Virtual Workspaces
--------------------------------------

Real microscopes encode dimensions differently. ImageXpress, Opera Phenix,
OMERO, Bio-Formats, and OpenHCS-native layouts may use different filenames,
folders, or metadata.

OpenHCS normalizes those sources through microscope handlers, metadata
extraction rules, source bindings, and virtual workspace paths. That is why
agents should use plate inspection and source-binding tools instead of local
filename parsing.

Operational Rule
----------------

When setting up a workflow:

1. Inspect the plate or selected UI plate.
2. Confirm wells, sites, channels, Z planes, and timepoints.
3. Choose ``variable_components`` based on what the callable must receive.
4. Choose ``group_by`` only when routing a function dictionary.
5. Compile or inspect the artifact plan before running the full dataset.

Related knowledge-base documents:

* ``openhcs_core_model``
* ``openhcs_pipelines_and_steps``
* ``openhcs_function_patterns``
* ``openhcs_pipeline_compilation_system``
* ``openhcs_runtime_system_assembly_rules``
