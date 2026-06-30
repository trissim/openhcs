# MCP Agent Experience Implementation Dry Run

Date: 2026-06-29

## Purpose

This dry run checks what happens at implementation time for the MCP agent
experience plan set. It imports the named nominal authorities and records what
they actually expose in the current checkout. If a plan needs iteration, the
implementation must iterate these authorities, not mirror their values.

## Command

Run from the repo root:

```bash
. .venv/bin/activate
python - <<'PY'
from dataclasses import fields
from inspect import signature

def show(name, value):
    print(f"{name}: {value}")

from openhcs.core.source_bindings_view import SourceInventoryProvider, SourceBindingsViewModel
from openhcs.core.source_binding_context import SourceBindingContext
from openhcs.core.source_workspace_projection import VirtualWorkspaceSourceProjection
show("SourceInventoryProvider registry", tuple(sorted(SourceInventoryProvider.__registry__)))
show("SourceBindingsViewModel.from_schema_and_bindings", signature(SourceBindingsViewModel.from_schema_and_bindings))
show("SourceBindingContext fields", tuple(f.name for f in fields(SourceBindingContext)))
show("VirtualWorkspaceSourceProjection methods", tuple(name for name in ("pipeline_start_files", "source_path_for", "source_metadata_for") if hasattr(VirtualWorkspaceSourceProjection, name)))

from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.source_bindings import CompiledSourceUniversePlan
from openhcs.core.source_load_plan import SourceLoadPlan
from openhcs.core.source_binding_selection import SourceUniverseRequest, SourceUniverseStrategy
show("CompiledStepPlan source/artifact fields", tuple(f.name for f in fields(CompiledStepPlan) if f.name.startswith("source_") or f.name.startswith("artifact_")))
show("CompiledSourceUniversePlan fields", tuple(f.name for f in fields(CompiledSourceUniversePlan)))
show("SourceLoadPlan fields", tuple(f.name for f in fields(SourceLoadPlan)))
show("SourceUniverseRequest types", tuple(t.__name__ for t in SourceUniverseRequest.registered_request_types()))
show("SourceUniverseStrategy types", tuple(t.__name__ for t in SourceUniverseStrategy.registered_strategy_types()))

from openhcs.core.artifacts import ArtifactSidecarRole, ArtifactSpec, ArtifactInputPlan, ArtifactOutputPlan
show("ArtifactSidecarRole values", tuple(role.value for role in ArtifactSidecarRole))
show("ArtifactSpec fields", tuple(f.name for f in fields(ArtifactSpec)))
show("ArtifactInputPlan fields", tuple(f.name for f in fields(ArtifactInputPlan)))
show("ArtifactOutputPlan fields", tuple(f.name for f in fields(ArtifactOutputPlan)))

from openhcs.core.config import PathPlanningConfig, VFSConfig
from openhcs.core.plate_image_inventory import PlateFileInventory
from openhcs.core.steps.function_artifact_materialization import planned_materialization_preview
show("PathPlanningConfig fields", tuple(f.name for f in fields(PathPlanningConfig)))
show("VFSConfig fields", tuple(f.name for f in fields(VFSConfig)))
show("PlateFileInventory public methods", tuple(name for name in dir(PlateFileInventory) if not name.startswith("_"))[:20])
show("planned_materialization_preview", signature(planned_materialization_preview))

from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract, ProcessingContractDeclaration
show("ProcessingContract members", tuple((member.name, member.declared_name, type(member.declaration).__name__) for member in ProcessingContract))
show("ProcessingContractDeclaration subclasses", tuple(cls.__name__ for cls in ProcessingContractDeclaration.__subclasses__()))

from openhcs.processing.backends.cellprofiler.module_classes import CellProfilerModule
show("CellProfilerModule registry count", len(CellProfilerModule.__registry__))
show("CellProfilerModule has for_module", hasattr(CellProfilerModule, "for_module"))

from openhcs.agent.capabilities import agent_capability_declarations
caps = agent_capability_declarations()
show("Agent capabilities", (len(caps), tuple((cap.name, cap.cli_command) for cap in caps[:8])))

from openhcs.mcp.dev_client_commanding import McpDevCommandSpec, CapabilityBackedCommandSpec
show("McpDevCommandSpec registry keys before command-module imports", tuple(McpDevCommandSpec.__registry__))
show("CapabilityBackedCommandSpec.for_capability_name", signature(CapabilityBackedCommandSpec.for_capability_name))

from openhcs.agent.services.knowledge_base_service import KnowledgeBaseService, default_document_specs
specs = default_document_specs()
ids = tuple(spec.document.document_id for spec in specs)
show("Knowledge docs", (len(specs), "openhcs_official30_benchmark_recipes" in ids, "openhcs_example_corpus_map" in ids))
show("KnowledgeBase official30 helpers", tuple(name for name in dir(KnowledgeBaseService) if "official30" in name))
PY
```

## Dry-Run Results From 2026-06-29

Source model:

- `SourceInventoryProvider.__registry__` is populated with five providers:
  `explicit_image_plane_sources`, `filemanager`, `local_directory`,
  `openhcs_workspace`, and `schema_context`.
- `SourceBindingsViewModel.from_schema_and_bindings` exists and takes typed
  `schema` and `bindings` keyword-only inputs.
- `SourceBindingContext` is a dataclass with typed fields for plate identity,
  schema, inventory provider, optional `.cppipe` path, and import result.
- `VirtualWorkspaceSourceProjection` exposes `pipeline_start_files`,
  `source_path_for`, and `source_metadata_for`.

Artifact/source coverage:

- `CompiledStepPlan` directly owns `source_binding_plan`,
  `source_universe_plan`, `source_load_plan`, `artifact_inputs`,
  `artifact_outputs`, and artifact group maps.
- `CompiledSourceUniversePlan` has exactly three current flags:
  `requires_step_input_selector_resolution`,
  `requires_full_pipeline_source_universe`, and
  `uses_pipeline_start_binding_origin`.
- `SourceLoadPlan` currently only exposes `zarr_config`. Plans must not promise
  richer source-load semantics until the compiler owns them.
- `SourceUniverseRequest.registered_request_types()` returns
  `StepInputSourceUniverseRequest` and `PipelineStartSourceUniverseRequest`.
- `SourceUniverseStrategy.registered_strategy_types()` returns six concrete
  strategies covering current pattern, virtual workspace, physical axis,
  axis-scoped pipeline start, virtual workspace pipeline start, and physical
  pipeline start.

Runtime sidecars:

- `ArtifactSidecarRole` currently has one value: `crop_mask`.
- `ArtifactSpec`, `ArtifactInputPlan`, and `ArtifactOutputPlan` all carry
  `sidecar_role`.
- Source-identity filename behavior is materialization behavior, not a sidecar
  role. Plans must keep those concepts separate.

VFS/storage:

- `PathPlanningConfig` and `VFSConfig` are dataclass config authorities.
- `PlateFileInventory` exposes query/record methods and should remain the file
  inventory authority.
- `planned_materialization_preview` takes a `ProcessingContext`,
  `FunctionStepExecutionPlan`, output key, and `ArtifactOutputPlan`.

Function and axis semantics:

- `ProcessingContract` lives in
  `openhcs.processing.backends.lib_registry.unified_registry`, not in
  `openhcs.core.callable_contract`.
- `ProcessingContract` members are `PURE_3D`, `PURE_2D`, `FLEXIBLE`, and
  `VOLUMETRIC_TO_SLICE`; each owns a nominal
  `ProcessingContractDeclaration`.
- `CellProfilerModule` lives in
  `openhcs.processing.backends.cellprofiler.module_classes`; it is
  `AutoRegisterMeta` backed, exposes `__registry__`, and has
  `for_module(module_name)`.

Capabilities and dev client:

- `agent_capability_declarations()` returned 79 capabilities in the dry run.
- Importing capabilities during a broad import can emit a circular-import
  warning before loading cached registry entries. Implementation should use the
  stable exported capability helpers, not early raw registry assumptions.
- `McpDevCommandSpec.__registry__` only showed `tools` before importing command
  modules. Dev-client implementation must ensure command modules are imported
  before relying on the command-spec registry.
- `CapabilityBackedCommandSpec.for_capability_name` exists and is the correct
  bridge from capability to command behavior.

Knowledge/examples:

- `default_document_specs()` returned 39 documents.
- `openhcs_official30_benchmark_recipes` and `openhcs_example_corpus_map` were
  present.
- `KnowledgeBaseService` owns existing official30 helpers; example search must
  build on those helpers and manifest specs.

## Required Plan Corrections From The Dry Run

- Function-contract plans must import `ProcessingContract` from
  `openhcs.processing.backends.lib_registry.unified_registry`.
- Sidecar plans must not imply measurement tables, object labels, or source
  identity are current `ArtifactSidecarRole` values. Current sidecar role is
  `crop_mask`; source identity remains materialization/provenance behavior.
- Artifact source-coverage plans must treat `SourceLoadPlan` as currently
  limited to `zarr_config`.
- Dev-client plans must import command modules before iterating
  `McpDevCommandSpec.__registry__`, or use capability-backed lookup directly.
- Any future richer projection must first add data to the core/compiler
  authority. MCP must not synthesize missing semantics.

