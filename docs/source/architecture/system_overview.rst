System overview
===============

OpenHCS separates declarations, compilation, and execution so that mutable UI
state and backend-specific configuration do not leak into workers.

End-to-end flow
---------------

.. code-block:: text

   user declarations or .cppipe
             |
             v
   PipelineConfig + list[FunctionStep]
             |
             | ObjectState resolves once
             v
   ResolvedPipelineDefinition
     + StepSnapshot tuple
             |
             | one CompilationSession per execution axis
             v
   typed CompiledStepPlan objects
     + per-step source/artifact/execution facts
             |
             v
   CompiledExecutionBundle
     + worker assignments
     + compiled runtime environment
             |
             v
   workers -> RuntimeValueStore -> materialization/export

Declaration boundary
--------------------

``PipelineConfig`` owns pipeline-level configuration, including source-binding
and processing defaults. Each ``FunctionStep`` owns a function pattern and
step-level configuration. Function patterns may be a callable, callable with
keyword arguments, a callable chain, or a dictionary pattern.

The GUI, generated Python, MCP/UI bridge, and CellProfiler importer all converge
on these declarations. There is no second runtime-pipeline object that carries
hidden semantics.

Resolution boundary
-------------------

ObjectState owns generic lazy configuration and live editing. Before compilation
continues, OpenHCS materializes one resolved step object for every declaration.
``StepSnapshot`` binds compiler identity and ObjectState scope identity to that
already-resolved step. Downstream compiler stages read semantics from the
resolved step and its callable/module declarations; they do not repeatedly ask
ObjectState to reinterpret fields.

Compilation boundary
--------------------

``CompilationSession`` ties together one ``ProcessingContext``, the resolved
steps, snapshots, ObjectState map, source-workspace projection, and mutable typed
plans for one execution axis. Its invariants prevent phases from silently
mixing steps, scopes, or contexts.

Each ``CompiledStepPlan`` is the source of truth for the compiled step. It has
typed fields for source bindings, source universe, artifact inputs and outputs,
execution group scope, function pattern, memory conversions, materialization,
backends, and streaming. Compiler stages mutate these fields
instead of maintaining parallel string-keyed dictionaries.

Execution boundary
------------------

``CompiledExecutionBundle`` carries the in-process runtime contexts, their
transport-safe projection, worker assignments, and the compiled runtime
environment. The environment includes multiprocessing start method, threading
choice, and GPU-registry initialization facts. These execution-wide facts belong
to the bundle rather than any individual ``CompiledStepPlan``. Workers execute
the bundle without resolving declaration configuration again.

Runtime data is represented by nominal value families and stored under typed
artifact keys. ``RuntimeValueStore`` is the authority for validated values,
locations, replacement, queries, and observation history. Projection strategies
adapt values to declared slice or grouping scopes without guessing from raw
array shape.

Owners and adapters
-------------------

OpenHCS owns domain wiring: microscopy sources, pipeline declarations, compiler
plans, processing contracts, artifacts, runtime values, and integrations. The
extracted packages own their generic mechanisms:

- ObjectState: lazy configuration and edit/snapshot machinery
- ArrayBridge: memory types and conversion
- PolyStore: storage, formats, ROI, and virtual-workspace primitives
- metaclass-registry: generic nominal registration
- pyqt-reactive: generic reactive forms and widgets
- python-introspect: callable and signature analysis
- ZMQRuntime: generic process and transport protocols
- pycodify: Python-source serialization

See :doc:`external_foundations` for the exact documentation boundary.

Related pages
-------------

- :doc:`nominal_ownership`
- :doc:`source_model`
- :doc:`artifact_contract_system`
- :doc:`pipeline_compilation_system`
- :doc:`runtime_value_system`
- :doc:`cellprofiler_interop`
