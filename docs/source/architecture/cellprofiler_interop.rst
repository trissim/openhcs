CellProfiler interoperability
=============================

CellProfiler interoperability is a pure translation into the same public
declarations used by native OpenHCS pipelines. It does not introduce a second
runtime, generated pipeline class, source-schema layer, or semantic sidecar.

Acknowledgment and citation
---------------------------

The interoperability implementation and its validation corpus build on the
CellProfiler project's open-source software, documentation, and public example,
tutorial, and benchmark materials. OpenHCS thanks the CellProfiler authors and
contributors and the authors of the biological datasets they distribute.
Publications using this interoperability should cite CellProfiler according to
its `official citation guidance <https://cellprofiler.org/citations>`_, including
Stirling *et al.*, *CellProfiler 4: improvements in speed, utility and
usability* (2021), `doi:10.1186/s12859-021-04344-9
<https://doi.org/10.1186/s12859-021-04344-9>`_. OpenHCS is an independent
project and is not endorsed by the CellProfiler project or the Broad Institute.

What compatibility means
------------------------

OpenHCS compatibility is semantic, not a promise to run CellProfiler itself as
a hidden subprocess. If you know CellProfiler, the familiar image names,
objects, measurements, relationships, groups, and export modules remain useful:
the importer lowers them to OpenHCS source bindings, ``FunctionStep`` objects,
artifact contracts, runtime values, and plate-scoped exporters.

The ``CellProfilerModule`` registry is the authority for the module declarations
absorbed into this OpenHCS tree. It does not inspect whichever CellProfiler
package version happens to be installed. The compatibility report projects
module, setting, source-module, corpus, and processing-contract coverage from
that registry and selected ``.cppipe`` files. It contains no execution
observations or result-equivalence records.

Official30 recipes provide broad end-to-end import, execution, and parity
evidence through separate benchmark observations. An agent must not turn that
corpus into the stronger claim that every historical CellProfiler version,
plugin module, and setting combination has been tested. For an unfamiliar
pipeline, import and compile the actual ``.cppipe`` and treat an explicit
module/setting error as the compatibility boundary.

Import flow
-----------

.. code-block:: text

   .cppipe
      -> CPPipeParser -> ModuleBlock records
      -> CellProfilerModule.require_module(name)
      -> setup modules contribute SourceBindingsConfig
      -> executable modules contribute ordinary FunctionStep declarations
      -> PipelineConfig.from_config(...)
      -> list[FunctionStep], PipelineConfig

``import_cellprofiler_pipeline`` resolves external files relative to the
pipeline directory by default or an explicit ``source_root``. Disabled modules
are ignored. Setup-only declarations do not emit a step; executable module
declarations emit one or more ordinary steps.

External model, classifier, rules, and similar resource paths are public
callable keyword values resolved by ``SettingsBinder``. They are not members of
the artifact-type registry merely because a callable consumes a file.

What the importer decides
-------------------------

The importer does encode the CellProfiler facts that are present in the parsed
pipeline and its owning module declarations. It derives source stack components
for ``variable_components``, applies callable-owned grouping constraints,
chooses previous-step versus pipeline-start main flow from source and producer
evidence, and resolves named images, objects, measurements, relationships, and
exports into exact callable artifact contracts. These are executable conversion
rules, not prose conventions.

That does not make the importer a universal native-pipeline intent oracle. The
``InputSource`` enum has only previous-step and pipeline-start main flow; an
additional named image is represented by an artifact input plus source binding
or prior producer. A ``.cppipe`` also does not choose arbitrary OpenHCS viewer
streaming, native checkpoint, VFS, or user review policy that it never declared.
Artifact presence in the runtime graph is not by itself persistent
materialisation. Native authors still review those independent choices on the
ordinary ``PipelineConfig``, ``FunctionStep``, callable contract, and compiled
artifact/materialisation plans.

Nominal module authority
------------------------

``CellProfilerModule`` is the auto-registering root for module semantics. A
module subclass owns:

- the exact CellProfiler module name
- parsed revision metadata and module-specific revision interpretation where
  implemented
- settings binding and repeated-row interpretation
- whether the module emits a step
- its public processing callable or callable batch
- required processing axes and grouping constraints
- source-setup behaviour
- callable artifact derivation, exact relations, and context advancement
- execution scope and runtime adapter behaviour
- measurement and relationship declarations

Shared behaviour composes through inheritance and module mixins. Import and
compiler code query the root registry; they do not maintain a parallel
module-name table.

Settings and public callables
-----------------------------

``SettingsBinder`` translates parsed CellProfiler settings into the public
callable's keyword arguments. The public callable remains visible to GUI/code
transport and signature analysis. During compilation an invocation-contract
provider may derive a runtime adapter or module executor, but that compile-only
object does not replace the public declaration format.

Setting-backed artifact identities, including output names, remain on that
public invocation even when the raw processing callable does not accept them.
The module declaration consumes those identity keywords while reconstructing
the exact ``ModuleBlock`` and excludes them from the later runtime call. Import
lowering therefore preserves every declared output identity through Python
source transport instead of pruning an output because a downstream observation
did not happen to reference it.

Numerical backend portability
-----------------------------

Some compatibility functions reproduce a historical NumPy numerical primitive
through an eight-lane SVML symbol. Symbol presence alone is not an execution
capability: NumPy wheels can contain AVX-512 code on a host that cannot execute
it. ``numpy_avx512_skx_svml_symbol_available()`` therefore joins two facts owned
by the loaded NumPy runtime: the complete ``AVX512_SKX`` CPU feature and the
requested symbol's presence in NumPy's loaded binary.

The CellProfiler-compatible leaf retains ownership of the symbol it requires
and of its portable Numba or NumPy implementation. The shared runtime query
does not name CellProfiler modules or choose processing semantics. Unsupported,
AVX512F-only, and non-x86 hosts select the leaf's portable implementation before
Numba lowers the call; there is no operating-system table, signal recovery,
runtime retry, or silent substitution with an unrelated backend.

Exact compile-time reconstruction
---------------------------------

``CellProfilerInvocationContractProviderFactory`` walks the resolved
``StepSnapshot`` sequence and one forward ``ArtifactDeclarationStepContext``:

.. code-block:: text

   public FunctionStep invocation + forward artifact context
       -> CallableContract canonical raw import identity
       -> CellProfilerModule.for_callable_contract(...)
       -> module_blocks_for_invocation(...)
       -> exact numbered ModuleBlock occurrence(s)
       -> invocation_callable_contract(...)
       -> compile-only runtime adapter + invocation artifact edges
       -> advance_artifact_context(...)

The canonical raw callable's complete module-and-function import identity selects
the candidate declaration, and object identity verifies the declaration-owned
callable. A same-named native callable therefore remains native. The public
callable's keyword arguments and declaration-owned setting bindings reconstruct
the exact module block. The compiler validates that reconstruction, numbers
repeated module occurrences, derives one ``CallableContract``, and advances the
same artifact context used for native callables. A native callable with unnamed
image main flow receives deterministic compiler-only provenance so a later
CellProfiler consumer can resolve its producer and group scope.

The resulting contract is indexed by step and ``FunctionInvocationKey``. It is
not stored back into the public ``FunctionStep`` and does not create a hidden
pipeline wrapper, module sidecar, or second artifact declaration graph.

Sources
-------

CellProfiler input/setup modules contribute named image-plane sources, metadata
matching, grouping fields, and other source bindings. The importer builds
``SourceBindingsConfig`` directly and folds it into ``PipelineConfig``. Source
artifact inputs are then satisfied through the normal source planner.

Artifacts and exports
---------------------

Each executable module inherits ``CellProfilerModuleArtifactContracts``. For one
parsed invocation, that mixin resolves active ``SettingToKeywordBinding``
declarations and module leaf hooks into an ordinary ``CallableContract``. There
is no parallel module-contract object. Exact semantic inputs and outputs use the
registered artifact families: special side-channel values, images, object
labels, measurements, object lineage and relationships, tables, spatial grids,
spatial graphs, and metadata. External resource paths remain bound callable
arguments.

Export modules are explicit steps. Plate-wide exporters declare
``FunctionStepExecutionScope.PLATE`` rather than relying on an implicit
post-pipeline sidecar. Materialisation and recording follow the normal artifact
graph and runtime-store paths.

CellProfiler Analyst export
---------------------------

``ExportToDatabase`` is a real plate-scoped OpenHCS step, not a placeholder or
post-run compatibility hook. Its owning module declaration selects the exact
image, object, measurement, relationship, thumbnail, and grouping artifacts.
At runtime:

.. code-block:: text

   typed RuntimeValueStore artifacts
       -> CellProfilerAnalystProjectionBuilder
       -> CPA image/object/relationship table projection
       -> CPASQLiteRenderer
       -> SQLite database + CellProfiler Analyst .properties files

The projection preserves declared table/column identities, image paths and file
names, object locations, image/object keys, group fields, relationships,
classifier metadata, channel display metadata, and optional thumbnails. The
module declares one ``SpecialArtifactType`` output with
``FileBundleOptions`` materialisation; its members are the SQLite database and
one or more ``.properties`` files. The bundle is a materialisation contract,
not an additional artifact type.

OpenHCS currently implements the SQLite database and CPA ``.properties`` route.
It rejects non-SQLite databases and custom filter rows explicitly. The current
renderer does **not** emit a CellProfiler Analyst ``.workspace`` file and does
not implement every historical per-well aggregation, plate-filter, overwrite,
or workspace-measurement setting. Some of those settings can still be present
in an imported declaration, so their presence is a known parity gap rather than
evidence that the requested side output exists.

``ExportToSpreadsheet`` is also an executable plate-scoped exporter and emits
declared measurement files from the same runtime-store model. Agents should
inspect the imported export step and resulting artifact plan, report any
non-default unsupported CPA settings, then verify the materialised files. They
must never promise CPA or workspace output merely because upstream measurement
steps compiled.

Parity evidence and safe claims
-------------------------------

There are three distinct levels of evidence:

1. **Import coverage**: every enabled module and relevant setting in the
   concrete ``.cppipe`` lowers without an unsupported declaration error.
2. **Execution coverage**: the translated pipeline compiles and runs with
   satisfied typed source/artifact contracts.
3. **Result parity**: images, labels, measurements, relationships, spreadsheets,
   SQLite tables, and CPA properties are compared under their semantic
   equivalence policies.

Only the third level supports a parity claim for the tested case. Use the
Official30 corpus as examples and regression evidence, then validate the user's
actual pipeline and representative data before describing it as equivalent.
The registry-derived compatibility report supports declarative coverage; it
does not by itself prove levels two or three.

Extension boundary
------------------

Module-specific semantics remain on the owning ``CellProfilerModule`` leaf and
its mixins. The importer, generic compiler, runtime store, and equivalence
engine consume that authority rather than dispatching on module names. See
:doc:`../development/cellprofiler_module_authoring` for the task-oriented
extension workflow.

Historical plans
----------------

The runtime unification and import/dispatch consolidation plans record the
migration that produced this boundary. They are not canonical current-state API
documentation and now live in the history archive. Unchecked items in those
snapshots describe migration-time follow-up, not current architectural
authority; current gaps must be tracked against this page and the source tree.
