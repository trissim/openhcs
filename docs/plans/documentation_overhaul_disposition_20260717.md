# OpenHCS documentation overhaul disposition

Status: implemented and locally verified on 2026-07-17
Authority: current source tree and public package metadata
Audiences: users, developers, architecture maintainers

This ledger turns the July 2026 documentation audit into a finite migration. It
is a tracking artifact, not an architecture source. Current behavior belongs in
the canonical pages under `docs/source/` and in the documentation of the package
that owns the behavior.

## Disposition rules

- **Keep**: the page describes a current OpenHCS-owned boundary; verify its links
  and examples.
- **Rewrite**: preserve the useful URL or subject, but replace the obsolete model
  with the current architecture.
- **Move**: the generic behavior is owned by an extracted package. Publish it
  there and leave a temporary OpenHCS transition page plus an integration page.
- **Archive**: retain decision history outside active navigation with an explicit
  superseded/current-outcome notice.
- **Remove**: delete instructions for unsupported interfaces or nonexistent APIs
  once no compatibility value remains.

## P0: actively misleading public guidance

| Surface | Disposition | Completion evidence |
| --- | --- | --- |
| `README.md` | Rewrite installation, programmatic use, compiler, artifact, and ownership claims | Completed; commands match `pyproject.toml`, and the owner smoke suite exercises the documented package boundaries |
| `docs/source/index.rst` | Rewrite navigation and remove TUI | Completed; only supported entry points are advertised |
| `getting_started/getting_started.rst` | Rewrite | Completed; Python 3.11+, GUI-first flow, and current declaration API |
| `api/index.rst` | Rewrite | Completed; no public `Pipeline`, `run_pipeline`, or removed imports |
| `user_guide/index.rst` | Rewrite | Completed; guide status and supported entry points are explicit |
| `docs/development_setup.md` | Rewrite | Completed; explicit submodule installation matches CI and packaging |

## Canonical OpenHCS pages

### Keep and revalidate

- `concepts/data_dimensions.rst`
- `concepts/pipelines_and_steps.rst`
- `architecture/external_integrations_overview.rst`
- `architecture/streaming_boundary_and_wrappers.rst`
- `architecture/progress_runtime_projection_system.rst`
- `architecture/code_ui_interconversion.rst`
- `development/ast_refactoring_workflow.rst`

### Rewrite around current owners

- `concepts/core_model.rst`
- `concepts/module_structure.rst`
- `concepts/function_library.rst`
- `architecture/index.rst`
- `architecture/pipeline_compilation_system.rst`
- `architecture/compilation_system_detailed.rst`
- `architecture/special_io_system.rst`
- `architecture/pattern_grouping_and_special_outputs.rst`
- `architecture/function_registry_system.rst`
- `architecture/batch_workflow_service.rst`
- `architecture/plate_manager_services.rst`
- `development/runtime_system_assembly_rules.rst`
- `guides/pipeline_compilation_workflow.rst`

### New canonical subjects

- declaration-to-runtime system overview
- nominal ownership and registry strategy families
- source bindings, source universes, provenance, and workspace projection
- callable ABI, callable contracts, and module artifact contracts
- artifact graph, plan selection, satisfaction, and materialization
- one-time ObjectState resolution, snapshots, sessions, and typed step plans
- runtime values, stores, adapters, scopes, and slice projections
- axes, grouping, batching, and `ProcessingContract` locality
- CellProfiler parsing and lowering into public OpenHCS declarations
- measurement, object-label, relationship, and equivalence semantics
- extracted-package ownership and OpenHCS integration boundaries

## Extracted-package migration

| Owner | Move generic documentation | Retain in OpenHCS |
| --- | --- | --- |
| ObjectState | lazy configuration, context resolution, edit/snapshot/provenance mechanics | configuration topology, compiler resolution boundary, UI/code integration |
| ArrayBridge | memory types, converters, stack utilities, device/OOM behavior | callable decorator metadata and compiler/resource integration |
| metaclass-registry | metaclass registry, families, configuration, discovery/cache mechanics | OpenHCS nominal roots and registry strategy usage |
| PolyStore | FileManager, backends, formats, ROI, virtual workspaces, source references | source-binding, materialization, and application backend integration |
| pyqt-reactive | generic forms, widgets, managers, services, previews, window infrastructure | OpenHCS workflows, Plate Manager, editors, and adapters |
| python-introspect | signature analysis, wrapped targets, analyzer extensions | OpenHCS parameter policy and step-editor consumption |
| ZMQRuntime | process lifecycle, protocols, progress/cancellation, viewer control | OpenHCS execution and runtime-projection wiring |
| pycodify | generic source serialization, formatters, imports, collision handling | code/UI round trip and OpenHCS generation boundaries |

`omero_openhcs` owns deployment and application integration. Generic OMERO
storage, ROI, and backend behavior belongs to PolyStore.

## Archive after extracting outcomes

- CellProfiler runtime unification and consolidation plans/audits
- `architecture/tui_system.rst`
- `architecture/compilation_service.rst`
- `architecture/zmq_execution_service_extracted.rst`
- `architecture/dict_pattern_case_study.rst`
- dated semantic/refactoring audits that contradict nominal ownership

## Completion gates

- [x] OpenHCS and all eight owner sites build as strict Sphinx HTML with
  warnings as errors.
- [x] The OpenHCS build resolves first-party cross-references against inventories
  generated from the current local owner docs.
- [x] All active Python examples parse, first-party imports and concrete source
  paths are checked, and one dependency-light path per owner is runtime-smoked.
- [x] Active guidance does not advertise the TUI, a public `Pipeline` or
  `run_pipeline`, generated CellProfiler semantic sidecars, a source-schema
  layer, or string-keyed compiler plans as current APIs.
- [x] Generic package internals have one owning documentation site; OpenHCS
  retains only integration guidance and transition URLs.
- [x] Superseded originals are outside active navigation under `docs/archive/`,
  with an archive-wide supersession notice. Loose historical audit, CI incident,
  migration, conflict, coverage-plan, and readiness notes in four owner
  repositories were likewise moved under their `docs/archive/` directories.

## Local verification record

- `scripts/validate_docs.py` passed for OpenHCS, all eight owners, and
  `omero_openhcs`: 280 active files and 285 Python blocks in total, including
  owner READMEs and active contributor guides.
- `scripts/smoke_owner_quickstarts.py` passed for ObjectState, ArrayBridge,
  metaclass-registry, PolyStore, pyqt-reactive, python-introspect, ZMQRuntime,
  and pycodify.
- Nine clean Sphinx HTML builds passed with `-E -W --keep-going`; inventories
  were generated for all eight owners, and the OpenHCS build used the five
  currently configured first-party inventories.
- The MCP knowledge manifest parsed and its focused knowledge/server regression
  suite passed: 260 tests.
- The documentation workflow YAML, all eight owner `pyproject.toml` files, and
  scoped `git diff --check` validation passed.

The validator is intentionally static: it AST-parses every active Python block,
but it does not execute examples that require microscopes, OMERO, GPUs, or GUI
interaction. The smoke suite is the executable gate for one low-dependency
quick-start path per extracted owner.

## Remaining source and publication debt

- PolyStore's OMERO backend still imports the OpenHCS `FilenameParser`. That is
  a source-level ownership inversion; a nominal parser or source-projection
  protocol should be injected at the PolyStore boundary.
- `omero_openhcs` is documented as an alpha prototype, not as a compatible
  deployment path. It omits required execution fields and still uses removed
  OpenHCS imports; it needs an integration gate before release claims are made.
- The eight owner documentation edits live in separate dirty submodule
  worktrees. Publishing them requires intentional commits in each owner
  repository followed by parent gitlink updates; this overhaul does not perform
  those repository operations.
- python-introspect, ZMQRuntime, and pycodify use stable repository links until
  their documentation sites are published. ZMQRuntime now includes a Read the
  Docs configuration, but publication is external to this local overhaul.
