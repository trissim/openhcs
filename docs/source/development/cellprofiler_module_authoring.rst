CellProfiler module authoring
=============================

CellProfiler semantics are owned by the nominal declarations in
``openhcs.interop.cellprofiler.module_declarations``. The importer and runtime
query those declarations; they must not maintain their own module-name tables.

Adding or changing a module
---------------------------

1. Find the nearest existing ``CellProfilerModule`` family with the same
   semantic behavior.
2. Extend that family so inheritance supplies shared behavior.
3. Declare settings binding, processing configuration, callable resolution,
   artifact inputs/outputs, invocation/runtime behavior, catalogue ownership,
   and measurement or relationship semantics at the module declaration.
4. Let ``AutoRegisterMeta`` place the concrete declaration in
   ``CellProfilerModule.__registry__``.
5. Keep ``pipeline_import`` as a lowering pass from parsed ``ModuleBlock``
   records to ordinary ``FunctionStep`` and ``PipelineConfig`` declarations.

Generic consumers resolve module ownership from
``CellProfilerModule.for_callable_contract``. The candidate is selected by the
canonical raw callable's complete import identity and verified by object
identity. Short-name lookup is limited to projecting attributes inside the
CellProfiler backend package; do not use it to classify compiler, runtime, UI,
or agent callables.

``CellProfilerModule`` is also the nominal callable-catalogue declaration. Each
subclass owns its ``declared_function_names()``, while the CellProfiler backend
package lazily projects the public union through that registry. Private helpers
can support an implementation, but they do not become function-browser entries.
Do not add an export list beside the module declarations.

Source modules
--------------

Images, metadata, names-and-types, and groups modules contribute directly to
``SourceBindingsConfig`` and pipeline grouping. There is no source-schema
sidecar and no generated runtime-pipeline object.

Executable modules
------------------

Executable declarations inherit ``CellProfilerModuleArtifactContracts``.
``SettingToKeywordBinding.input()`` and ``SettingToKeywordBinding.output()``
declare setting-backed roles; module leaf hooks add only the specialization the
module owns. For each parsed invocation, the mixin resolves dynamic names,
relations, and active bindings into an ordinary ``CallableContract``. Artifact
inputs may then be satisfied by source bindings, main flow, metadata, or prior
runtime producers. Outputs declare semantic type and relationships independently
of their runtime Python return position. Standard measurement outputs carry the
module declaration as their feature-vocabulary owner so later modules select
prior measurements without reconstructing ownership from an invocation name.

Keep every active setting-backed output identity in the public invocation
keywords. ``module_blocks_for_invocation()`` consumes those declaration-only
keywords before the processing callable runs, so they remain valid Python/code
transport state without becoming runtime parameters. Do not remove output
identities by scanning which downstream artifacts happen to be observed.

Testing
-------

- Parse a representative ``.cppipe`` module block.
- Verify declaration lookup through ``CellProfilerModule.__registry__``.
- Assert the lowered step/config and invocation ``CallableContract``.
- Compile the minimal pipeline and inspect exact source/artifact plans.
- Run an equivalence fixture when measurements, objects, or relationships
  change.

See :doc:`../architecture/cellprofiler_interop`.
