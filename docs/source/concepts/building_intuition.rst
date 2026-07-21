Building intuition
==================

Think of OpenHCS as a declaration compiler, not a loop over image files.

Sources are named inputs
------------------------

Start by identifying the plate, its execution axes, and the semantic image names
needed by the analysis. Source bindings turn files and metadata into those named
inputs. Avoid writing processing functions that search directories or parse
filenames themselves.

Steps declare intent
--------------------

An ordered ``list[FunctionStep]`` describes analysis intent. A step selects a
callable pattern and nested configuration. The compiler decides concrete paths,
source sets, artifact edges, conversions, scopes, and workers.

Use a callable chain when several operations form one processing station. Use a
dictionary pattern only when different compiled groups need different callable
patterns.

Axes answer different questions
-------------------------------

Ask these questions separately:

1. What does the stack axis represent? Declare it with
   ``variable_components``.
2. Should assembled arrays be partitioned for dictionary routing? Declare it
   with ``group_by``.
3. Does the algorithm depend on each plane independently or on the complete
   stack? The callable's ``ProcessingContract`` answers that.

Outputs are artifacts
---------------------

An image, object-label set, measurement table, relationship, or file has a typed
semantic identity. Artifact contracts declare that identity. Materialization
controls which artifacts persist outside the runtime store. Saving an
intermediate result is therefore a plan decision, not a ``force_disk_output``
flag on ``FunctionStep``.

Compile before scaling up
-------------------------

Use a representative one- or two-axis subset first. Compilation should prove
that sources resolve, callable requirements agree with processing settings,
artifacts have producers, memory conversions are possible, and runtime scopes
are complete. Then execute and inspect typed outputs before increasing the
selection.

Choose the owning layer
-----------------------

- source/file semantics: source declarations or microscope integration
- callable behavior: callable contract or owning module declaration
- storage mechanism: PolyStore
- memory conversion: ArrayBridge
- UI form behavior: pyqt-reactive
- OpenHCS workflow behavior: OpenHCS application service

Keeping behavior on its owner prevents the compiler and UI from accumulating
parallel name tables and fallback rules.

Next reading
------------

- :doc:`core_model`
- :doc:`pipelines_and_steps`
- :doc:`../architecture/source_model`
- :doc:`../architecture/artifact_contract_system`
