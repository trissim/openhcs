Pipeline compilation workflow
=============================

Compilation is a typed declaration-to-runtime lowering pass. It is not a fixed
numbered sequence and it does not produce dictionaries whose field names form a
second API.

Inputs
------

The compiler accepts an initialized ``PipelineOrchestrator`` and an ordered
``list[AbstractStep]``. Together they provide:

- a ``PipelineConfig`` and ObjectState scope topology;
- the microscope/source workspace and selected execution axes;
- callable, module, source, and artifact declarations;
- per-step processing, materialization, filtering, and streaming configuration.

Compiler authorities
--------------------

``CompilationSession`` owns cross-step compile state. One
``CompiledStepPlan`` per step and execution axis owns the resolved typed fields.
Compiler stages query declaration authorities and registry strategies; they do
not match backend names or maintain mirrored feature lists.

The main responsibilities are:

1. resolve configuration once and snapshot it;
2. lower callable patterns and contracts;
3. resolve source universes, source bindings, and main-flow dependencies;
4. plan artifact satisfaction and materialization;
5. resolve axes, grouping, memory conversion, devices, and runtime adapters;
6. validate the complete plans and assemble a ``CompiledExecutionBundle``.

Execution boundary
------------------

The runtime bundle contains worker-safe steps, frozen processing contexts, and
environment decisions. Runtime code may validate required typed fields, but it
must not repeat compiler inference. The GUI and ZMQ execution service both use
this compile-before-execute boundary.

Debugging
---------

Start with the declaration and the corresponding ``CompiledStepPlan`` field.
Then identify the compiler stage that owns that field and the nominal
declaration it queries. See :doc:`../development/pipeline_debugging_guide` and
:doc:`../architecture/pipeline_compilation_system`.
