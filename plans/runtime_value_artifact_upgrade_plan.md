# Runtime Value, Source Binding, and CellProfiler System Plan

**Date:** 2026-04-27
**Branch:** `benchmark-platform`
**Status:** In progress
**Supersedes:** the narrower runtime-artifact-only framing from earlier passes

## 1. Executive Summary

This branch is no longer blocked on basic compiler/runtime refactoring. That foundation is largely in place.

What remains is a system-level integration problem:

1. OpenHCS already has a typed artifact plane for produced runtime values.
2. CellProfiler also needs a typed plane for **named semantic image bindings**.
3. That source plane must fit not only the local runtime executor, but also:
   - `ObjectState` and time travel
   - `pyqt-reactive` forms and previews
   - `pycodify` round-trip code generation
   - microscope metadata/component-key semantics
   - `polystore` backend-explicit storage rules
   - direct and ZMQ execution

The central missing concept is therefore not “more wrappers” and not “more special cases in the executor”.

The central missing concept is:

**a typed, serializable, compiler-owned, GUI-compatible source-binding model for named semantic image views**

That is the main remaining semantic gap between current OpenHCS and full `.cppipe` compatibility.

---

## 2. What OpenHCS Actually Is

OpenHCS is not just the `openhcs/` package. The relevant architecture spans several companion packages and boundaries.

### 2.1 Domain/App Layer

Owned in this repo:

1. Pipeline compiler
2. Orchestrator and execution model
3. Microscope handlers and metadata interpretation
4. Runtime artifact semantics
5. GUI application and editor windows
6. CellProfiler conversion and compatibility layer

Core files:

1. [openhcs/core/orchestrator/orchestrator.py](/home/ts/code/projects/openhcs-benchmark-platform/openhcs/core/orchestrator/orchestrator.py:562)
2. [openhcs/core/context/processing_context.py](/home/ts/code/projects/openhcs-benchmark-platform/openhcs/core/context/processing_context.py:1)
3. [openhcs/core/pipeline/path_planner.py](/home/ts/code/projects/openhcs-benchmark-platform/openhcs/core/pipeline/path_planner.py:1)
4. [openhcs/core/pipeline/step_snapshot.py](/home/ts/code/projects/openhcs-benchmark-platform/openhcs/core/pipeline/step_snapshot.py:1)
5. [benchmark/converter/symbol_table.py](/home/ts/code/projects/openhcs-benchmark-platform/benchmark/converter/symbol_table.py:1)
6. [benchmark/cellprofiler_compat/module_execution.py](/home/ts/code/projects/openhcs-benchmark-platform/benchmark/cellprofiler_compat/module_execution.py:1)

### 2.2 External State/Config Layer

Owned by external local dependency:

1. `objectstate`

Responsibilities:

1. Editable state model
2. Flat dotted-path storage
3. Saved/live resolution
4. Dirty tracking
5. Time-travel DAG history
6. Scope hierarchy and delegation

Core files:

1. [/home/ts/code/projects/openhcs/external/ObjectState/src/objectstate/object_state.py](/home/ts/code/projects/openhcs/external/ObjectState/src/objectstate/object_state.py:1)
2. [/home/ts/code/projects/openhcs/external/ObjectState/src/objectstate/object_state_registry.py](/home/ts/code/projects/openhcs/external/ObjectState/src/objectstate/object_state_registry.py:1)

### 2.3 External GUI/Form Layer

Owned by external local dependency:

1. `pyqt-reactive`

Responsibilities:

1. Dataclass-driven form generation
2. ObjectState-backed editing
3. Live refresh and scoped updates
4. Window/form/view logic

Core file:

1. [/home/ts/code/projects/openhcs/external/pyqt-reactive/src/pyqt_reactive/forms/parameter_form_manager.py](/home/ts/code/projects/openhcs/external/pyqt-reactive/src/pyqt_reactive/forms/parameter_form_manager.py:1)

### 2.4 External Storage/VFS Layer

Owned by external local dependency:

1. `polystore`

Responsibilities:

1. Backend-explicit load/save
2. Memory/disk/zarr/streaming backends
3. FileManager routing
4. No implicit fallback

Core file:

1. [/home/ts/code/projects/polystore/src/polystore/filemanager.py](/home/ts/code/projects/polystore/src/polystore/filemanager.py:1)

### 2.5 External Transport Layer

Owned by external local dependency:

1. `zmqruntime`

Responsibilities:

1. Direct vs ZMQ execution transport
2. Typed progress and request/response messages
3. Queue tracking and server info

Core files:

1. [external/zmqruntime/src/zmqruntime/config.py](/home/ts/code/projects/openhcs-benchmark-platform/external/zmqruntime/src/zmqruntime/config.py:1)
2. [external/zmqruntime/src/zmqruntime/messages.py](/home/ts/code/projects/openhcs-benchmark-platform/external/zmqruntime/src/zmqruntime/messages.py:1)

### 2.6 External Introspection and Codegen Layer

Owned by external local dependencies:

1. `python-introspect`
2. `pycodify`

Responsibilities:

1. Type/signature analysis for forms and editors
2. Round-trip Python source generation
3. Code/UI interconversion integrity

Core files:

1. [external/python-introspect/src/python_introspect/__init__.py](/home/ts/code/projects/openhcs-benchmark-platform/external/python-introspect/src/python_introspect/__init__.py:1)
2. [external/pycodify/src/pycodify/__init__.py](/home/ts/code/projects/openhcs-benchmark-platform/external/pycodify/src/pycodify/__init__.py:1)

---

## 3. Current Branch Status

### 3.1 What Is Already Done

The branch has already established most of the typed runtime/compiler foundation needed for richer semantics:

1. `CompiledStepPlan` is the compiler/runtime execution SSOT.
2. Function patterns are normalized and compiled before runtime.
3. `CallableContract` centralizes callable metadata extraction.
4. Artifact graph extraction and per-invocation ownership are typed.
5. `RuntimeValue`, `RuntimeValueSchema`, and `RuntimeValueStore` exist.
6. `ArtifactKind` is preserved through compile and runtime validation.
7. Generated CellProfiler wrappers execute through the OpenHCS orchestrator/runtime path.
8. Produced images, object labels, measurements, and relationships now have real runtime representation.
9. The CellProfiler symbol table already distinguishes:
   - runtime artifact inputs
   - external image inputs
10. Minimal `.cppipe -> generate -> import -> orchestrator execute` works.
11. `.cppipe` parsing now preserves ordered typed setting records instead of only last-write dict values.
12. Converter setup modules now compile into a typed image/setup schema that lowers `NamesAndTypes` aliases into selector-bearing `source_bindings`.
13. Compiler/runtime plans now carry explicit stable step identity plus a typed main-input dependency record instead of relying purely on implicit `step_index - 1` assumptions.
    - The current field name is `step_scope_id`, but semantically this is just a compiled stable identity string copied forward from the existing step token/scope machinery.
    - Runtime execution does **not** use `ObjectState`; a later cleanup may rename this to `step_identity` or `step_node_id`.
14. Artifact input/output plans now also carry scope-based producer/source identity alongside legacy step indexes.
15. Selector-bearing runtime source resolution is now wired for the native cases OpenHCS can currently express:
    - `STEP_INPUT` bindings resolve against the current pattern-group file universe and select typed views from the current stack.
    - `PIPELINE_START` bindings resolve against the original axis file universe with inherited current-scope component constraints.
16. Metadata extraction rules are now first-class core source-binding state rather than converter-local strings:
    - compiled `StepSourceBindingsConfig` / `CompiledSourceBindingPlan` preserve typed regex-backed metadata rules
    - generated pipelines emit those rules directly
    - runtime candidate parsing augments native parser metadata from those rules instead of guessing
17. Metadata-only selectors can now resolve when the binding plan provides enough compiled metadata extraction semantics.
18. Current-scope inheritance is now opportunistic rather than rigid:
    - inherited scope fields only constrain candidates that actually expose those fields
    - this keeps pipeline-start matches usable for cases like illumination files that share folder identity but not full well/site/channel metadata
19. Metadata-based `NamesAndTypes` image-set matching now compiles into a typed cross-alias match plan:
    - the parser preserves repeated setup settings needed for match dimensions
    - escaped legacy `.cppipe` match payloads are decoded before literal parsing
    - generated `source_bindings` now carry the match plan all the way into runtime resolution
20. The `GrayToColor` absorbed-library gap is resolved through one module-level typed dispatcher instead of mode-specific registry hacks:
    - repeated stack/composite settings are preserved through a dedicated module-settings binding layer
    - `GrayToColor` source image discovery is now shared SSOT in converter code instead of ad hoc local parsing
    - BBBC021 now converts successfully with 20 processing modules and no failed absorbed modules
21. The setup/image schema is now exposed beyond the immediate converter internals:
    - generated `.cppipe` pipeline objects and prepared runtime pipelines carry the compiled pipeline-level `source_schema`
    - dataset specs can declare canonical reference `.cppipe` URLs
    - the OpenHCS benchmark adapter can resolve converted pipeline runs from either a local `.cppipe` path or a dataset-owned canonical `.cppipe` reference
22. Pipeline-level image schema ownership now lives in core OpenHCS concepts instead of converter-local dataclasses:
    - [openhcs/core/pipeline_image_schema.py](/home/ts/code/projects/openhcs-benchmark-platform/openhcs/core/pipeline_image_schema.py:1) now owns `CellProfilerImageSchema`, `ImageAssignment`, `GroupingPlan`, `ImagesRule`, and legacy alias strategies
    - [benchmark/converter/source_schema.py](/home/ts/code/projects/openhcs-benchmark-platform/benchmark/converter/source_schema.py:1) now acts as a lowering module that compiles setup modules into those core schema types
    - generated/prepared pipeline objects, symbol-table compilation, and benchmark/runtime wiring now import the schema from core directly
23. The absorbed CellProfiler import/materialization surface is now cleaner and exercised on real paths:
    - `csv_materializer(...)` is a first-class exported materialization preset used by absorbed measurement/export modules
    - `openhcs.core.memory.decorators` is a real core import surface rather than an implicit missing module
    - active and ExampleFly-relevant absorbed functions now import cleanly under unit coverage
    - the benchmark adapter now executes the real shipped `ExampleFly.cppipe` end to end and materializes non-empty CSV outputs
24. The in-tree `.cppipe` corpus is now tracked explicitly instead of implicitly:
    - shipped fixtures are classified as either supported or structurally invalid
    - adapter-level preparation failures are wrapped as `ToolExecutionError` with the original compatibility diagnostic preserved
    - `ExampleHuman.cppipe` is now asserted as a known invalid reduced fixture rather than silently encouraging weaker symbol validation
25. The accepted corpus now includes canonical BBBC021 reference pipeline snapshots in-tree:
    - `BBBC021_analysis.cppipe` and `BBBC021_illum.cppipe` now prepare successfully as supported corpus members
    - real setup-module lowering for those files is asserted under unit coverage
    - the typed image schema for those files now has explicit acceptance checks for grouping, metadata match dimensions, and selector-bearing assignments
26. PURE_2D absorbed-function execution now preserves tuple-shaped side outputs generically:
    - the shared registry slice executor now aggregates `(main_output, side_output...)` returns instead of assuming every per-slice result is a bare 2D array
    - 2D auxiliary image/label outputs restack correctly
    - tabular auxiliary outputs aggregate across slices instead of collapsing or failing
    - per-slice outputs with `slice_index` fields now get the real runtime slice index injected during aggregation
27. Real acceptance coverage now extends beyond the earlier in-tree subset:
    - canonical `BBBC021_illum.cppipe` executes through the real orchestrator/runtime path on synthetic BBBC021-shaped data
    - `ExampleFly.cppipe` CSV assertions now validate semantic headers, not just file existence
    - generated `RelateObjects` acceptance now validates concrete relationship/measurement CSV schemas
28. `NamesAndTypes` order-based image-set matching is now part of the supported semantics:
    - the setup-module compiler now lowers `Image set matching method:Order` into a typed `SourceBindingMatchPlan(method=ORDER)`
    - the runtime adapter resolves order-based pipeline-start pairings generically by matching the current step-input image-set index against the ordered target alias candidate list
    - support is implemented in the nominal core match-plan resolver path rather than as converter- or module-local glue
29. Common CellProfiler processing-module artifact contracts can now be inferred through a nominal typed pattern family instead of silently defaulting to empty contracts:
    - infrastructure modules now have an explicit no-artifact contract builder
    - common single image/object input-output shapes infer `ArtifactSpec` inputs/outputs from declared CellProfiler settings
    - unmodeled processing modules now fail loudly with a compatibility diagnostic instead of pretending they have no runtime semantics
    - canonical BBBC021 illumination modules now compile real `CorrectIlluminationCalculate` image inputs/outputs

### 3.2 What Is Still Missing

The biggest unresolved items are now:

1. Setup-module semantics are now exposed as a core-owned pipeline-level image schema during generated-pipeline preparation and benchmark execution, but they are not yet a broader editable GUI/ObjectState pipeline concept.
2. `NamesAndTypes` image-set matching semantics are now modeled for both metadata-based and order-based paths, but broader real-pipeline coverage and any remaining match variants still need work.
   - `Metadata` matching lowers into a typed cross-alias plan
   - `Order` matching now resolves through a generic runtime image-set index strategy
   - unsupported variants should continue to fail loudly until modeled natively
3. GUI/ObjectState/pycodify do not yet own richer source-binding state as an editable first-class step concept.
4. The main-input edge is now explicit in compiled plans, but the external execution model is still list-based rather than first-class graph-based.
5. The compiled identity record is semantically useful, but its current `step_scope_id` naming still reflects pre-compilation/UI vocabulary more than ideal runtime/compiler terminology.
6. Real BBBC pipelines are only partially accepted end to end.
   - live BBBC021 setup/image schema compilation now succeeds
   - live BBBC021 conversion now succeeds end to end at code-generation time
   - a BBBC021-style generated pipeline now executes through the OpenHCS orchestrator with typed named-channel bindings
   - canonical BBBC021 illumination now carries explicit inferred image artifact contracts for illumination functions instead of empty contracts
   - canonical in-tree BBBC021 reference snapshots now prepare successfully and preserve typed schema facts under direct test coverage
   - canonical `BBBC021_illum.cppipe` now executes successfully as a real converted pipeline shape
   - canonical dataset-owned `.cppipe` references can now be resolved through the benchmark adapter instead of only local ad hoc files
   - ExampleFly now executes end to end as a real shipped `.cppipe` shape and materializes measurement CSV outputs on disk
   - a generated `RelateObjects` pipeline now executes through the orchestrator and materializes both relationship and measurement CSV outputs
   - the next gap is broader corpus coverage and dataset-level validation, not basic execution enablement
7. Export and relationship-heavy semantics now have initial real-output validation with concrete CSV schema assertions, but not broad corpus-level validation.
9. The broader absorbed-function corpus still needs continued cleanup beyond the currently exercised import/materialization surface.
8. Benchmarking is no longer ahead of the remaining CellProfiler semantics, but it should still stay secondary to broader semantic validation.

---

## 4. Problem Statement

OpenHCS currently has two meaningful data planes:

1. **Primary image plane**
   - the `main_data_arg`
   - the ordinary step input/output stack flow

2. **Runtime artifact plane**
   - images produced by prior modules
   - object labels
   - measurement tables
   - relationships
   - persisted through the typed runtime store + VFS boundary

CellProfiler requires a third plane:

3. **Named semantic source plane**
   - semantic image names such as `OrigBlue`, `DNA`, `GFP`, `Actin`
   - usually views/selectors over the step input container
   - sometimes resolved from microscope metadata/component coordinates when the data is not already present in the step input container
   - distinct from runtime-produced artifacts even if both end up as the same typed image value once resolved

Today, that third plane is only partially represented:

1. the symbol table knows such names exist
2. conversion now preserves repeated setup settings and lowers setup modules into typed alias selectors
3. generated-pipeline preparation now carries a pipeline-level image schema and the benchmark adapter can target canonical dataset-owned `.cppipe` references
4. the GUI/codegen layer does not yet expose the richer selector-bearing source state as a mature editable concept
5. broader corpus-level validation still needs to prove that the compiled source plane is correct on real dataset layouts beyond the currently accepted pipelines

That is the core semantic gap.

---

## 5. Architectural Constraints

These constraints are mandatory for the remaining work.

### 5.1 No Fake Wrapper Layer

Do not solve this by building local wrapper classes around dicts.

A new type is only justified if it owns one or more of:

1. identity
2. validation invariant
3. serialization contract
4. compiler snapshot boundary
5. runtime resolution rule
6. GUI-editable state

### 5.2 No Silent Fallback

No direct-VFS fallback and no “best effort” image substitution.

If a compiled source binding cannot be resolved, runtime must fail loudly with:

1. binding name
2. step/module identity
3. expected source selector
4. axis/group scope

### 5.3 No Runtime O(n) Module-Specific Solving

Do not accumulate many `if module_name == ...` branches in the executor.

Module-specific knowledge should be compiled into declarative semantics once, then executed generically.

### 5.4 Do Not Overload the Function Pattern

The dict-of-lists function pattern already means:

1. behavior selection by component/group

It should **not** become the data-source model.

Function pattern answers:

1. what code runs for this group?

Source binding plan answers:

1. what named inputs exist for this group?

Those are related but distinct layers.

### 5.5 The GUI and Code Round-Trip Matter

Any new user-visible concept must fit:

1. `ObjectState`
2. `pyqt-reactive` forms
3. `pycodify` export/import
4. preview formatting in the pipeline editor

### 5.6 Respect Microscope/Metadata Ownership

The source of truth for real input coordinates is:

1. microscope handler
2. metadata handler
3. metadata cache
4. orchestrator component keys

CellProfiler source bindings must compile into those existing semantics, not parallel them.

### 5.7 Respect the Existing Step Input Model

OpenHCS steps already operate on a primary `main_data_arg`, which may be a multi-image or multi-dimensional container.

Source bindings must not assume that “many semantic names” means “many separately loaded arrays”.

Prefer this order of interpretation:

1. semantic name maps to a typed selector/view over the existing step input container
2. if the named data is not already present in that container, resolve it through the microscope/metadata path

So source bindings are primarily a **name-to-view / name-to-selector** layer, not a forced side-channel image loader.

### 5.8 Preserve OpenHCS Genericity

OpenHCS should not gain a “CellProfiler workspace” core abstraction.

It should gain a more generic notion of:

1. named external source bindings
2. typed artifact semantics
3. compiled input/output contracts

CellProfiler then becomes one client of those abstractions.

---

## 6. Target Architecture

### 6.1 Layering

The intended source-of-truth chain should be:

```text
Editable step/source binding config (ObjectState-visible)
    -> compiler snapshot
    -> compiled source binding plan
    -> runtime source resolver
    -> typed image/object/measurement values
    -> materialization / export
```

This must sit beside, not inside, the existing artifact chain:

```text
callable contract
    -> artifact graph
    -> compiled artifact input/output plans
    -> runtime artifact store
```

### 6.2 Core Concepts to Introduce

The remaining missing domain types are around source binding, not around artifact output.

Preferred domain split:

1. **Editable/source-layer step field**
   - a dataclass family owned by OpenHCS core
   - exposed as a real `FunctionStep` constructor field
   - serializable
   - ObjectState-friendly
   - pycodify-friendly

2. **Compiled/source-layer plan**
   - immutable
   - compiler-owned
   - no hidden dicts
   - no signature probing at runtime

3. **Runtime/source-layer resolution**
   - uses microscope metadata, component keys, and filemanager
   - returns explicit named image payloads

Candidate conceptual types:

1. `SourceBindingKind`
2. `SourceSelector`
3. `ExternalImageBinding`
4. `GroupedExternalImageBindings`
5. `StepSourceBindingsConfig`
6. `CompiledSourceBindingPlan`

These names are illustrative; exact names can change.

### 6.3 Relationship to Existing OpenHCS Concepts

### `InputSource`

`InputSource.PIPELINE_START` is a coarse step-wide source selector.

It should remain valid, but it is not enough for CellProfiler.

Correct relationship:

1. `InputSource` says which broad domain a step reads from.
2. source bindings refine which **named images** inside that domain are needed.

### `FunctionPattern`

`CompiledFunctionPattern` remains the SSOT for grouped behavior.

Source bindings should reuse the same group-key vocabulary where appropriate, but must remain a distinct plan.

### `FunctionStep`

`FunctionStep` is constructor-introspected by the UI/state system rather than declared as a dataclass.

That means source bindings should be introduced as a real first-class step field, not a hidden post-hoc attribute and not nested under `processing_config`.

Correct relationship:

1. `func` declares behavior
2. `source_bindings` declares semantic named input views/selectors
3. `processing_config` continues to own operational knobs like `group_by`, `variable_components`, and `input_source`

### `RuntimeValueStore`

Produced images continue to live in the runtime artifact plane.

External images are **not** runtime-produced artifacts. They are resolved source inputs.

However, both should converge to the same **typed image value semantics** once resolved.

That means:

1. external image binding resolution returns typed image values
2. runtime-produced images are read as typed image values
3. the executor should then be able to treat them symmetrically

### `Metadata and UI Component Selection`

The GUI already exposes metadata-backed component selection through the generic component-selection provider path.

Correct relationship:

1. metadata and microscope handlers define what coordinates/components exist
2. source bindings store typed selectors in that vocabulary
3. the UI renders human labels through the existing metadata display path

So source bindings should store stable component/metadata selectors, while the GUI displays names like `Channel 1 | DAPI` using the existing provider stack.

---

## 7. Infrastructure Module Mapping

The CellProfiler infrastructure modules should no longer be treated as vague skipped prelude.

### `Images`

Maps to:

1. image-discovery assumptions
2. input-domain description

Likely compile role:

1. validates that image-loading mode is representable in OpenHCS
2. contributes to source binding normalization

### `Metadata`

Maps to:

1. filename/metadata component interpretation
2. source selectors based on well/site/channel/z/timepoint or other metadata

Likely compile role:

1. contributes selector rules
2. validates available metadata fields against the microscope handler

### `NamesAndTypes`

This is the most important infrastructure module.

It maps:

1. CellProfiler semantic image names
2. to OpenHCS source selectors/views

This should become the primary compiler source for named external image bindings.

### `Groups`

Maps to:

1. execution partitioning
2. group-key scoping for source bindings and outputs

This should compile into:

1. grouped source binding plans
2. possibly grouped export/materialization semantics

### `SaveImages`

Should not remain a fake processing step.

It should compile into:

1. materialization/export intent for image artifacts

### `ExportToSpreadsheet`

Should compile into:

1. table materialization/export intent
2. possibly consolidation rules for measurements/relationships

---

## 8. Work Plan

The work should proceed in passes that keep source-of-truth ownership clear.

### Pass 1: Freeze the Architectural Vocabulary

**Goal:** establish the system-level semantic boundary before writing more compatibility code.

Deliverables:

1. This plan becomes the branch master plan.
2. Old local-only assumptions are retired.
3. Acceptance targets are explicit:
   - minimal generated `.cppipe`
   - real multi-image pipeline
   - BBBC021 analytical core

Acceptance:

1. No new implementation pass starts from “just patch the executor”.
2. New abstractions are evaluated against GUI, codegen, compiler, storage, and metadata concerns.

### Pass 2: Add Typed Source-Binding Domain Types

**Goal:** represent named semantic image bindings as first-class OpenHCS types.

Primary files:

1. new core module under `openhcs/core/` for source binding semantics
2. `FunctionStep` constructor surface
3. companion tests

Requirements:

1. dataclass-based
2. serializable
3. validation-rich
4. no dict wrapper theater
5. usable without CellProfiler-specific naming
6. selector-first, not loader-first
7. direct `FunctionStep` field, not hidden nested config

Acceptance:

1. Source bindings can be represented as typed Python objects.
2. They can express:
   - single named semantic image selector
   - multiple named semantic image selectors
   - optional grouped bindings
   - source selectors against metadata/component space
   - selectors over the existing step input container

### Pass 3: Attach Source Bindings to Compiler Snapshots and Compiled Plans

**Goal:** make source bindings part of compile-time SSOT.

Primary files:

1. [openhcs/core/pipeline/step_snapshot.py](/home/ts/code/projects/openhcs-benchmark-platform/openhcs/core/pipeline/step_snapshot.py:1)
2. compiled plan types
3. path planning / compiler session plumbing

Requirements:

1. `StepSnapshot` captures the saved `FunctionStep.source_bindings` value explicitly
2. compiled plans carry an immutable source-binding plan
3. no recomputation from loose string tuples during runtime
4. no hidden state outside snapshot/compiled-plan ownership

Acceptance:

1. One can inspect a compiled step plan and fully know its external image source contract.

### Pass 4: Compile CellProfiler Infrastructure Modules into the New Model

**Goal:** stop dropping `Images` / `Metadata` / `NamesAndTypes` / `Groups` as inert prelude.

Primary files:

1. `benchmark/converter/` module(s), likely a new dedicated source-plan compiler
2. [benchmark/converter/runtime_pipeline.py](/home/ts/code/projects/openhcs-benchmark-platform/benchmark/converter/runtime_pipeline.py:1)
3. [benchmark/converter/pipeline_generator.py](/home/ts/code/projects/openhcs-benchmark-platform/benchmark/converter/pipeline_generator.py:1)

Requirements:

1. artifact symbol table remains responsible for produced/runtime symbols
2. new source-plan compilation becomes responsible for external image bindings
3. these responsibilities should be separate, not muddled

Acceptance:

1. Generated pipeline artifacts include typed source binding declarations.
2. At least one multi-image `.cppipe` compiles without collapsing external image names to a tuple of raw strings.

### Pass 5: Runtime Source Resolution

**Goal:** resolve typed external image bindings through existing OpenHCS input semantics.

Primary files:

1. runtime execution path
2. CellProfiler runtime adapter/executor path
3. possibly a generic source-resolution helper in core

Requirements:

1. source resolution must use:
   - microscope handler
   - metadata cache
   - component keys
   - filemanager
2. no single-image fallback for multi-image bindings
3. failures must be explicit and typed

Acceptance:

1. `STEP_INPUT` selectors resolve against the current pattern-group file universe and select typed views from the current stack.
2. `PIPELINE_START` component selectors resolve against the original axis file universe with inherited current-scope component constraints.
3. Compiled metadata extraction rules augment native parser metadata during source resolution instead of living only in converter-local lowering.
4. Unsupported metadata-only selectors fail loudly when the compiled rule set plus native parser/source system still cannot express them.
5. External images resolve consistently under both direct and ZMQ execution.

### Pass 6: External/Produced Image Symmetry

**Goal:** make image consumption generic regardless of whether an image is external or runtime-produced.

Primary files:

1. runtime executor path
2. CellProfiler module execution policies
3. runtime image value handling

Requirements:

1. image inputs should be resolved as typed values
2. produced images and external images should share downstream semantics
3. module-specific ladders should collapse into generic binding families where possible

Candidate generic binding families:

1. single image
2. image pair
3. image set / image stack
4. image + objects
5. object set
6. measurement target

Acceptance:

1. `GrayToColor`, `OverlayOutlines`, and similar multi-input modules execute through generic source binding logic, not ad hoc fallback glue.

### Pass 7: GUI and Codegen Integration

**Goal:** ensure the new concept is not runtime-only.

Primary files:

1. `ObjectState` integration points
2. PyQt step editor and previews
3. pipeline/code export formatters
4. pipeline import/migration path

Requirements:

1. source binding config must be editable or at minimum preserved as a typed field
2. pipeline editor preview should be able to surface the presence of source bindings
3. code export/import must round-trip them

Important note:

The first implementation may keep the UI minimally exposed, but the same typed objects must already be used. Do not introduce a temporary hidden dict format that later needs replacement.

Acceptance:

1. A generated or manually authored pipeline containing source bindings can round-trip through Python code without semantic loss.

### Pass 8: Real Pipeline Acceptance

**Goal:** validate the design on real pipelines, not just synthetic tests.

Acceptance targets:

1. existing synthetic/generated `.cppipe` tests still pass
2. `ExampleFly.cppipe` end-to-end execution remains clean
3. `ExampleHuman.cppipe` either executes or fails only on clearly unsupported absorbed-module semantics
4. BBBC021 analytical core converts and executes through OpenHCS
5. benchmark adapter execution of a converted `.cppipe` uses the same path and stays green

Scope notes:

1. visualization-only modules may be compiled as explicit no-op/skip semantics if that policy is made first-class and not ad hoc
2. unsupported modules must fail loudly and specifically

### Pass 9: Relationship and Export Completion

**Goal:** finish the richer semantic outputs, not just image/object flow.

Work:

1. relationship-heavy modules
2. measurement consolidation/export semantics
3. image save/export semantics
4. real output validation

Acceptance:

1. relationship outputs are typed and materializable
2. measurement exports from converted pipelines match expected schema/semantics
3. at least one real shipped `.cppipe` and one generated relationship pipeline leave CSV outputs on disk through normal OpenHCS execution

### Pass 10: Benchmarking Last

**Goal:** only after the CellProfiler/OpenHCS semantic path is solid, make benchmarking rely on it.

Work:

1. benchmark adapter runs real converted pipelines
2. benchmark datasets carry canonical `.cppipe` references where appropriate
3. results are comparable across native OpenHCS and converted CellProfiler semantics

Acceptance:

1. benchmark path uses the same production conversion/runtime path as the integration tests
2. benchmarking is no longer ahead of semantic support
3. benchmark adapter coverage remains a thin consumer of the same converted-pipeline runtime path

---

## 9. Decisions and Rejections

### Rejected: Runtime-Only Compatibility Layer

Reason:

1. ignores GUI/codegen/state model
2. creates hidden local minima
3. encourages more fallback logic

### Rejected: Dict-Backed Workspace Emulation as Core Design

Reason:

1. wrong ownership model for OpenHCS
2. hides invariants
3. creates fake abstraction rather than real semantics

### Rejected: Overloading `func` Dict Pattern for Source Selection

Reason:

1. `func` pattern already means grouped behavior
2. mixing behavior and data-source semantics would confuse the compiler and GUI

### Preferred: Core Generic Source Binding + Thin CellProfiler Compiler

Reason:

1. keeps semantics in OpenHCS
2. lets CellProfiler remain a client
3. allows future non-CellProfiler use

---

## 10. Acceptance Checklist

The branch should be considered “architecturally ready for full CellProfiler support” only when all of the following are true:

1. external image names compile into typed source bindings, not raw string tuples
2. runtime can resolve multiple external images without fallback
3. produced and external images share a common typed image semantic model
4. source binding state is representable in ObjectState and codegen
5. at least one real multi-image `.cppipe` executes through the normal orchestrator path
6. direct and ZMQ execution both pass for the same converted pipeline
7. export/relationship semantics are validated on real outputs
8. benchmark integration is using the same semantics, not a parallel shortcut path

---

## 11. Recommended Immediate Next Pass

The next implementation pass should be:

1. widen acceptance from the current BBBC021-style generated execution path to more real-pipeline and real-data validation
2. thread the setup-module image schema farther outward so it is not trapped inside converter-local lowering
3. validate broader corpus coverage beyond the current BBBC021 / ExampleFly / generated-relationship cases
4. keep replacing hidden sequential assumptions with explicit compiled edge records where that can be done without changing the list-based pipeline/editor model

That keeps the current pass aligned with real CellProfiler semantics while still preparing the compiler/runtime for a later DAG model if it is still justified after acceptance testing.
