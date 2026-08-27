Axes, grouping, and processing semantics
========================================

OpenHCS transports image data as 3D arrays, including logical single-plane
inputs. Array rank alone does not determine what the leading/third axis means or
whether a callable has per-plane or whole-stack semantics.

Five questions, five owners
---------------------------

Answer these questions independently for every authored step:

``variable_components``
  **What varies along the assembled array axis?** This may be site, channel, Z,
  or another declared component. It is the sole authority for stack-axis
  meaning.

``group_by``
  **How is an already assembled value partitioned?** With a dictionary function
  pattern, the resulting group identity also selects a callable branch. It does
  not define the stack axis or assemble a second stack.

``input_source``
  **Where does the ordinary main-flow image come from?** The two choices are the
  previous step and pipeline start. A separately named source is not a third
  ``InputSource`` value: declare it as a callable artifact input and satisfy it
  through step source bindings or a prior artifact producer.

Callable artifact contract
  **Which named semantic values does the callable consume and produce?** The
  callable or module owns exact image, object-label, measurement, relationship,
  table, grid, and external-resource ``ArtifactSpec`` declarations. The compiler
  decides whether each input is satisfied by source bindings, main flow,
  metadata, or a prior runtime producer.

Materialization plan
  **Which results must survive the execution?** Runtime-store availability lets
  downstream steps consume a typed value during the run; it does not itself
  promise a persistent file. Main-flow checkpoints use
  ``StepMaterializationConfig``. Named artifact persistence comes from artifact
  output materialization and the compiled runtime-artifact materialization plan.

Related execution declarations
------------------------------

``ProcessingContract``
  Declares how a callable depends on the stack. It is an execution-semantic
  contract, not an accepted-rank annotation.

``FunctionStepExecutionScope``
  Declares whether a callable runs once per execution axis or once per plate.
  Plate-scoped exporters are not simulated through grouping.

``ImagePayloadConsumption`` and runtime image mode
  Declare whether the callable consumes the natural payload or a composed image
  view and how that payload is presented at runtime.

Processing contracts
--------------------

``PURE_2D``
  The result for one plane is semantically independent of every other plane.
  OpenHCS may still transport or batch a 3D array, but the contract executes and
  aggregates through nominal per-slice strategies.

``PURE_3D``
  The callable depends on the full variable-component stack and executes once
  with whole-stack semantics.

``FLEXIBLE``
  The callable supports both modes through an explicit semantic-control
  parameter. The contract consumes that control and delegates to ``PURE_2D`` or
  ``PURE_3D``; callers do not infer the mode from shape.

``VOLUMETRIC_TO_SLICE``
  The callable consumes a real stack and produces a collapsed plane domain. The
  contract updates payload provenance to reflect the consumed leading axis.

Callable ownership
------------------

The callable's ``CallableContract`` declares input, output, and execution memory
roles, required variable components, allowed grouping, processing contract,
runtime-owned parameters, execution scope, runtime adapter, image mode, and
artifact declarations. Runtime-owned parameters include values supplied through
artifact, configuration, context, or adapter declarations; shared form and
catalogue analysis excludes them from authored keyword fields. The compiler
validates step configuration against these callable-owned constraints and stores
the resolved facts, including selected framework-local device bindings, on
``CompiledStepPlan``.

Runtime projection
------------------

When a contract requires per-slice execution, ``RuntimeSliceProjection`` selects
a registered strategy for each nominal value family. Image payloads, object
labels, measurements, relationships, tables, and aligned collections retain
their semantic identity during projection and aggregation. A missing projection
declaration is an error; OpenHCS does not guess from an object's shape or fields.

Common mistakes
---------------

- Treating ``group_by`` as the definition of the stack axis.
- Treating a named source binding as a third ``InputSource`` enum value.
- Treating a value's presence in the runtime store as a persistence request.
- Treating ``PURE_2D`` as “the callable only accepts rank-2 arrays.”
- Passing ``variable_components`` or ``group_by`` directly to
  ``FunctionStep`` instead of its processing configuration.
- Adding backend-specific dispatch to core rather than extending the callable or
  registered projection owner.

See :doc:`artifact_contract_system` for artifact satisfaction and persistence,
and :doc:`../guide_for_biologists/configuration_reference` for the configuration
inheritance scopes.
