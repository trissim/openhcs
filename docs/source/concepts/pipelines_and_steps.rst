Pipelines and Steps
===================

OpenHCS workflows are declared as pipelines of steps. A pipeline tells OpenHCS
what should happen; the compiler decides how that declaration becomes concrete
work over wells, sites, channels, storage backends, artifacts, and runtime
resources.

What Is A Pipeline?
-------------------

A pipeline is an ordered collection of processing steps. Each step consumes the
previous step's output unless its config declares a different input source.

.. code-block:: python

   from openhcs.core.steps.function_step import FunctionStep

   pipeline_steps = [
       FunctionStep(func=normalize_images, name="normalize"),
       FunctionStep(func=segment_cells, name="segment"),
       FunctionStep(func=measure_features, name="measure"),
   ]

Key properties:

* steps run in the declared order;
* data flow is explicit and compiler-planned;
* wells/sites can run in parallel;
* source format, storage, memory conversion, and artifacts are handled by
  OpenHCS infrastructure rather than ad hoc loops.

What Is A FunctionStep?
-----------------------

``FunctionStep`` is the primary step declaration. It names a callable pattern,
user parameters, and step-level configs.

.. code-block:: python

   from openhcs.constants.constants import VariableComponents
   from openhcs.core.config import LazyProcessingConfig
   from openhcs.core.steps.function_step import FunctionStep

   normalize_step = FunctionStep(
       func=(stack_percentile_normalize, {
           "low_percentile": 1.0,
           "high_percentile": 99.0,
       }),
       name="normalize",
       processing_config=LazyProcessingConfig(
           variable_components=[VariableComponents.SITE],
       ),
   )

Core fields:

* ``func``: callable, ``(callable, kwargs)``, list chain, or dictionary pattern;
* ``name``: human-readable step identity;
* ``processing_config``: lazy processing semantics such as variable components,
  grouping, and input source;
* other lazy step configs: materialization, source bindings, viewer streaming,
  well filters, dtype behavior, and related step policies.

Function Patterns
-----------------

OpenHCS supports a small set of function shapes:

* bare callable: run one function with defaults;
* tuple: run one function with explicit kwargs;
* list: run several functions in sequence;
* dictionary: route different groups, commonly channels, to different function
  patterns.

Function names, signatures, memory decorators, runtime-bound parameters, and
artifact declarations come from the function registry and callable contracts.
Agents should search and describe registry functions before authoring kwargs.

Why Not A Regular Python Loop?
------------------------------

A traditional script often mixes file discovery, image loading, GPU conversion,
analysis, output naming, and result writing in one loop. OpenHCS separates those
concerns:

* microscope handlers and source bindings interpret input data;
* ``FunctionStep`` declarations describe analysis intent;
* lazy configs describe execution semantics and inheritance;
* the compiler resolves paths, axes, artifacts, memory contracts, storage, and
  resources;
* runtime execution consumes the compiled plan.

That separation lets the same workflow run through the UI or headless runtime,
scale over plate-sized data, and remain reviewable as Python code.

Compile Before Run
------------------

Compilation is a required part of the model. It validates and prepares:

* source workspace and path planning;
* resolved ObjectState/config values;
* step input and output plans;
* artifact inputs, outputs, special IO, and materialization;
* memory type compatibility and backend choices;
* resource assignment such as GPUs and workers.

Agents should inspect the artifact plan or compile status before claiming a
pipeline is ready for a full run.

UI-Owned Versus Headless Workflows
----------------------------------

Headless execution sessions compile and run without updating the visible Plate
Manager. Use them for tests and automation.

When the user should see or continue editing the workflow in OpenHCS, use the
UI bridge path: read or apply code documents, dispatch selected-plate init,
compile, or run actions, then poll state surfaces. This preserves ObjectState
snapshots, selected rows, visible status, and output auto-add behavior.

Related Knowledge
-----------------

Read these next:

* ``openhcs_core_model``
* ``openhcs_data_dimensions``
* ``openhcs_function_patterns``
* ``openhcs_configuration_framework``
* ``openhcs_pipeline_compilation_system``
* ``openhcs_code_ui_interconversion``
