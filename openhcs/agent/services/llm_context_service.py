"""Prompt/context projection for OpenHCS authoring agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from enum import Enum
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.agent.authoring_contexts import (
    AgentCapabilityIndexContext,
    AuthoringContextDeclaration,
    CellProfilerTranslationAuthoringContext,
    CellProfilerTranslationContext,
    CoreOpenHCSModelContext,
    CustomFunctionAuthoringContext,
    CustomFunctionAuthoringRulesContext,
    CustomFunctionRuntimeContext,
    DomainExpertAssistedSetupAuthoringContext,
    DomainExpertAssistedSetupContext,
    ExampleCorpusContext,
    FirstUseAuthoringContext,
    FirstUseWorkflowContext,
    FolderOnboardingAuthoringContext,
    FolderOnboardingContext,
    HeadlessExecutionAuthoringContext,
    HeadlessExecutionContext,
    ObjectStateEditingAuthoringContext,
    ObjectStateEditingContext,
    PipelineAuthoringContext,
    PipelineAuthoringRulesContext,
    PipelineMentalModelContext,
    PipelineSystemModelContext,
    RuntimeUiCoordinationContext,
    SourceBindingWorkflowContext,
    StateCodeRoundtripContext,
    UiVisibleWorkflowAuthoringContext,
    UiVisibleWorkflowContext,
    ViewerReviewAuthoringContext,
    ViewerReviewContext,
)
from openhcs.agent.capabilities import (
    CapabilityKind,
    agent_capabilities,
    get_capability_registry,
)
from openhcs.agent.dto.authoring import AuthoringContext, AuthoringContextRequest
from openhcs.agent.dto.common import SCHEMA_VERSION
from openhcs.agent.dto.config import ConfigSchema
from openhcs.agent.dto.knowledge import KnowledgeBaseDocumentRequest
from openhcs.agent.services.architecture_projection_service import (
    CellProfilerTranslationArchitectureTopic,
)
from openhcs.agent.services.config_service import ConfigService
from openhcs.agent.services.function_catalog_service import FunctionCatalogService
from openhcs.agent.services.knowledge_base_service import KnowledgeBaseService
from openhcs.agent.ui_bridge_actions import PlateManagerAction
from openhcs.agent.ui_bridge_identities import (
    PlateManagerOrchestratorCodeDocumentIdentity,
    PlateManagerStateSurfaceIdentityDeclaration,
)


class AuthoringContextSection(ABC, metaclass=AutoRegisterMeta):
    """Nominal section rendered into authoring contexts that share its tags."""

    __registry__: ClassVar[dict[str, type["AuthoringContextSection"]]] = {}
    __registry_key__ = "section_id"
    __skip_if_no_key__ = True

    section_id: ClassVar[str | None] = None

    @classmethod
    @abstractmethod
    def render(cls, service: "AgentAuthoringContextService") -> str:
        raise NotImplementedError

    @classmethod
    def context_facets(cls) -> tuple[type[AuthoringContextDeclaration], ...]:
        return AuthoringContextDeclaration.facets_for(cls)

    @classmethod
    def belongs_to(cls, context: type["AuthoringContextDeclaration"]) -> bool:
        context_facets = context.context_facets()
        return any(facet in context_facets for facet in cls.context_facets())


class StaticAuthoringContextSection(AuthoringContextSection):
    """Authoring-context section with declaration-owned static content."""

    content: ClassVar[str]

    @classmethod
    def render(cls, service: "AgentAuthoringContextService") -> str:
        del service
        return cls.content


def render_authoring_context_sections(
    context: type[AuthoringContextDeclaration],
    service: "AgentAuthoringContextService",
) -> tuple[str, ...]:
    """Render sections whose nominal context facets belong to a context."""
    return tuple(
        section_type.render(service)
        for section_type in AuthoringContextSection.__registry__.values()
        if section_type.belongs_to(context)
    )


class KnowledgeBackedAuthoringContextSection(AuthoringContextSection):
    """Authoring-context section projected from the source-backed KB."""

    heading: ClassVar[str]
    document_id: ClassVar[str]
    document_section_id: ClassVar[str | None] = None
    max_chars: ClassVar[int] = 2_000

    @classmethod
    def render(cls, service: "AgentAuthoringContextService") -> str:
        content = service.knowledge_document_content(
            document_id=cls.document_id,
            section_id=cls.document_section_id,
            max_chars=cls.max_chars,
        )
        return f"=== {cls.heading} ===\n{content}"


class CoreModelKnowledgeSection(
    KnowledgeBackedAuthoringContextSection,
    CoreOpenHCSModelContext,
):
    section_id = "core_openhcs_model"
    heading = "OPENHCS CORE MODEL"
    document_id = "openhcs_core_model"
    document_section_id = "core-summary"
    max_chars = 3_000


class CellProfilerCompatibilityKnowledgeSection(
    KnowledgeBackedAuthoringContextSection,
    CellProfilerTranslationContext,
):
    section_id = "cellprofiler_compatibility_model"
    heading = "CELLPROFILER COMPATIBILITY MODEL"
    document_id = "openhcs_core_model"
    document_section_id = "cellprofiler-compatibility-model"
    max_chars = 2_400


class ArtifactUniverseKnowledgeSection(
    KnowledgeBackedAuthoringContextSection,
    CoreOpenHCSModelContext,
):
    section_id = "artifact_sidecar_source_universe_model"
    heading = "ARTIFACT SIDECAR AND SOURCE UNIVERSE MODEL"
    document_id = "openhcs_core_model"
    document_section_id = "artifact-sidecar-and-source-universe-model"
    max_chars = 2_400


class PipelineSystemModelSection(
    StaticAuthoringContextSection,
    PipelineSystemModelContext,
):
    section_id = "pipeline_system_model"
    content = """=== FRONTLOADED OPENHCS MODEL ===
- FunctionStep is the authored step boundary: function reference(s), step name, lazy step configs, and source_bindings.
- PipelineConfig and GlobalPipelineConfig carry lazy defaults; inspect resolved values through ObjectState/UI tools before assuming None means inactive.
- Compilation resolves FunctionStep declarations, source bindings, artifact contracts, and materialization into an execution plan; runtime tools should not bypass that boundary.
- Use architecture tools when the model is unclear: openhcs_explain_architecture with pipeline_model, cellprofiler_translation, source_semantics, or execution_runtime.
- Read knowledge docs openhcs_data_dimensions, openhcs_function_patterns, openhcs_pattern_grouping_special_outputs, and openhcs_code_ui_interconversion for axis/state/code details."""


class CellProfilerTranslationBridgeSection(
    AuthoringContextSection,
    CellProfilerTranslationContext,
):
    section_id = "cellprofiler_translation_bridge"

    @classmethod
    def render(cls, service: "AgentAuthoringContextService") -> str:
        del service
        topic = CellProfilerTranslationArchitectureTopic()
        concept_lines = "\n".join(f"- {concept}" for concept in topic.concepts_text)
        note_lines = "\n".join(
            f"- {note}" for note in topic.cellprofiler_translation_notes_text
        )
        return f"""=== CELLPROFILER MENTAL MODEL BRIDGE ===
- If you know CellProfiler, use that model first: CellProfiler compatibility is integrated into OpenHCS compilation and runtime through source bindings, FunctionStep declarations, artifact contracts, runtime values, materialization, and measurements.
- Use CellProfiler examples and .cppipe files to understand the biological intent, module order, named images/objects, measurements, and expected artifacts; then compile and validate the OpenHCS projection with artifact-plan, runtime, and viewer/inventory tools.
- Detailed architecture topic: openhcs_explain_architecture(topic_id="cellprofiler_translation").
{concept_lines}
{note_lines}"""


class RuntimeUiCoordinationSection(
    AuthoringContextSection,
    RuntimeUiCoordinationContext,
):
    section_id = "runtime_ui_coordination"

    @classmethod
    def render(cls, service: "AgentAuthoringContextService") -> str:
        del service
        code_document_id = PlateManagerOrchestratorCodeDocumentIdentity.require_value()
        state_surface_id = PlateManagerStateSurfaceIdentityDeclaration.require_value()
        workflow_names = ", ".join(
            action.value
            for action in PlateManagerAction
            if action.plate_operation is not None
        )
        return f"""=== RUNTIME AND UI COORDINATION ===
- If the OpenHCS UI is open and the user should see the work, use the UI bridge path: read/apply {code_document_id} with plate_paths and pipeline_data, then dispatch {workflow_names} through {agent_capabilities.ui_selected_plate_workflow.name}.
- Direct orchestrator sessions are headless runtime jobs: they can execute, stream to viewers, and write output plates, but they do not make PlateManager rows, ObjectState snapshots, or selected UI state visible unless the UI path is used.
- After UI-owned runs, poll {state_surface_id} with {agent_capabilities.ui_get_state_surface.name} to confirm source and output rows, then inspect/query/sample the output plate and validate viewer layers from those visible paths."""


class StateCodeRoundtripSection(
    StaticAuthoringContextSection,
    StateCodeRoundtripContext,
):
    section_id = "state_code_roundtrip"
    content = """=== OBJECTSTATE AND CODE ROUNDTRIP ===
- ObjectState is the edit/provenance layer for config, pipelines, steps, and code surfaces; use object-state-scopes and object-state-fields before assuming UI text or raw None values.
- Field markers are semantic: * means unsaved/dirty, _ means differs from defaults, inherited/resolved values show lazy/default resolution even when raw values are None.
- Code documents are live typed bidirectional UI<->code projections over ObjectState-backed UI objects, not freeform files or export/import scripts; get-code-document, validate-code-document, and apply-code-document use revision tokens to preserve reviewable Python and real-time ObjectState updates.
- UI mutations can create snapshots and branches; use window snapshots, ObjectState scopes, state surfaces, and time-travel-head to understand whether the UI is at branch head before applying edits.
- Read knowledge document openhcs_code_ui_interconversion for the current PyQt/ObjectState/code-document model."""


class CustomFunctionRuntimeSection(
    StaticAuthoringContextSection,
    CustomFunctionRuntimeContext,
):
    section_id = "custom_function_runtime"
    content = """=== CUSTOM FUNCTIONS AND RUNTIME OUTPUTS ===
- Custom functions are registry functions used inside FunctionStep.func patterns: callable, (callable, kwargs), list chains, or component/group keyed patterns.
- Dict patterns are routed by processing_config.group_by; list and single-callable patterns apply the same callable chain to each grouped 3D array.
- processing_config.variable_components are the axes stacked into each callable input array; group_by groups the remaining 3D arrays and only selects callable branches for dict patterns.
- Memory decorators such as @numpy define runtime memory conversion; FunctionStep processing_config defines grouping, axes, and input source.
- Declare non-image outputs with artifact_outputs plus MaterializationSpec/CsvOptions/JsonOptions, or ROI/image presets such as segmentation_mask_rois and tiff_stack.
- Artifact graph planning owns output names, sidecar results, materialization intent, and downstream artifact inputs; do not encode those as ad hoc files or hidden globals.
- ROI and image review flows use materialized/streamed artifacts through viewer streaming tools; after execution inspect selected-plate-files, viewer-rois, viewer-payloads, and sample-viewer-image."""


class SourceBindingWorkflowSection(
    StaticAuthoringContextSection,
    SourceBindingWorkflowContext,
):
    section_id = "source_binding_workflow"
    content = """=== SOURCE-BINDING WORKFLOW ===
- Start custom plates by inspecting real inventory: inspect-plate, query-plate-files, selected-plate-images, and selected-plate-files expose virtual paths, axes, results, and handler confidence; selected-plate review tools can target selected/source/output rows.
- Filename and folder semantics belong in MetadataExtractionRule, SourceBindingsConfig, and StepSourceBindingsConfig, not local filename parsing or raw string conventions.
- FunctionStep.source_bindings is the step-local semantic-input contract; pipeline_config.source_bindings_config is the source discovery/default contract.
- Virtual workspaces map logical OpenHCS virtual filenames to source paths and metadata; agents should query virtual paths/source metadata instead of assuming physical folder layout is execution layout.
- Compile/session inspection can expose source_workspace files with virtual_path, full_virtual_path, source_path, and source_metadata for source-bound pipelines.
- For source-bound custom plates, use reviewed Python/code-document or CellProfiler import/generator flows until the compact v1 draft-step API grows a nominal source-binding DTO. Do not pass ad hoc source_bindings dicts."""


class ExampleCorpusSection(StaticAuthoringContextSection, ExampleCorpusContext):
    section_id = "example_corpus"
    content = """=== EXAMPLE CORPUS FIRST ===
- Search/read knowledge document openhcs_example_corpus_map before inventing source bindings or pipeline structure.
- benchmark/cellprofiler_pipelines contains in-tree .cppipe examples plus checked-in OpenHCS equivalents for ExampleHuman and ExampleFly.
- benchmark/native_refs/official30_scoped_rows contains 30 native CellProfiler reference .cppipe runs; generated OpenHCS equivalents are produced by converter/adapter paths, not checked in as one static .py per case.
- benchmark/pipelines and openhcs/processing/presets/pipelines contain native OpenHCS examples and reusable preset pipelines."""


class CoreImportsSection(
    StaticAuthoringContextSection,
    PipelineAuthoringRulesContext,
):
    section_id = "core_imports"
    content = """=== CORE PIPELINE IMPORTS ===
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.config import (
    LazyProcessingConfig,
    LazyDtypeConfig,
    LazyStepSourceBindingsConfig,
    LazyStepMaterializationConfig,
    LazyNapariStreamingConfig,
    LazyFijiStreamingConfig,
    SourceBindingsConfig,
    StepSourceBindingsConfig,
)
from openhcs.core.source_bindings import (
    MetadataExtractionRule,
    MetadataSource,
    NamedSourceBinding,
    SourceBindingOrigin,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
)
from openhcs.constants.constants import VariableComponents, GroupBy
from openhcs.constants.input_source import InputSource"""


class PipelineRulesSection(
    StaticAuthoringContextSection,
    PipelineAuthoringRulesContext,
):
    section_id = "pipeline_rules"
    content = """=== PIPELINE AUTHORING RULES ===
- Author pipelines as ordered FunctionStep objects.
- Function references should come from the registry when using MCP tools.
- Search functions first, then call openhcs_describe_function before adding a step with non-default parameters.
- Rendered Python source is review/export output; the MCP draft pipeline is the canonical v1 state.
- Use LazyProcessingConfig for per-step axis/input-source semantics when needed.
- Do not pass variable_components, group_by, or input_source directly to FunctionStep; put them inside processing_config.

Minimal source-backed step:
```python
pipeline_steps = [
    FunctionStep(
        func=(registered_function, {"parameter_name": 1.0}),
        name="Denoise images",
        processing_config=LazyProcessingConfig(
            variable_components=[VariableComponents.SITE],
            group_by=GroupBy.CHANNEL,
        ),
    )
]
```"""


class ConfigSchemaHintsSection(AuthoringContextSection, PipelineAuthoringRulesContext):
    section_id = "config_schema_hints"

    @classmethod
    def render(cls, service: "AgentAuthoringContextService") -> str:
        schemas = (
            service.config_service.describe_schema("global"),
            service.config_service.describe_schema("pipeline"),
        )
        lines = [
            "=== CONFIG SCHEMA HINTS ===",
            "Use openhcs_describe_config_schema for the full reflected schema before setting non-obvious fields.",
        ]
        for schema in schemas:
            lines.extend(cls._schema_lines(schema, service.max_config_fields))
        return "\n".join(lines)

    @classmethod
    def _schema_lines(cls, schema: ConfigSchema, max_config_fields: int) -> list[str]:
        visible_fields = tuple(field for field in schema.fields if not field.ui_hidden)
        lines = [f"\n## {schema.config_type}"]
        for field in visible_fields[:max_config_fields]:
            requirement = ConfigFieldRequirementLabel.from_required(field.required)
            resolution = ConfigFieldResolutionLabel.from_lazy(field.lazy)
            lines.append(
                f"- {field.path}: {field.type_repr} ({requirement.value}{resolution.value})"
            )
        if len(visible_fields) > max_config_fields:
            remaining = len(visible_fields) - max_config_fields
            lines.append(f"- ... {remaining} more fields")
        return lines


class FunctionCatalogSection(AuthoringContextSection, PipelineAuthoringRulesContext):
    section_id = "function_catalog"

    @classmethod
    def render(cls, service: "AgentAuthoringContextService") -> str:
        page = service.function_catalog.search(
            limit=service.max_functions,
            compact_signatures=True,
        )
        lines = [
            "=== REGISTERED OPENHCS FUNCTIONS ===",
            "Use function_id values with MCP authoring tools; use imports only when rendering reviewed Python source.",
        ]
        current_library = None
        for entry in page.items:
            if entry.library != current_library:
                current_library = entry.library
                lines.append(f"\n## {entry.library}")
            summary = "" if entry.summary is None else f" - {entry.summary}"
            lines.append(f"- {entry.function_id}: `{entry.signature}`{summary}")
        if page.total > len(page.items):
            remaining = page.total - len(page.items)
            lines.append(
                f"\n... {remaining} more functions are available through openhcs_search_functions."
            )
        return "\n".join(lines)


class CustomFunctionImportsSection(
    StaticAuthoringContextSection,
    CustomFunctionAuthoringRulesContext,
):
    section_id = "custom_function_imports"
    content = """=== CORE CUSTOM FUNCTION IMPORTS ===
from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import artifact_outputs, artifact_inputs
from openhcs.processing.materialization import (
    CsvOptions,
    JsonOptions,
    MaterializationSpec,
    segmentation_mask_rois,
    tiff_stack,
)
import numpy as np"""


class CustomFunctionRulesSection(
    StaticAuthoringContextSection,
    CustomFunctionAuthoringRulesContext,
):
    section_id = "custom_function_rules"
    content = """=== CUSTOM FUNCTION RULES ===
- Decorate exactly one callable with a memory decorator such as @numpy.
- The first image-like argument is supplied by OpenHCS at runtime; do not pass it as a FunctionStep kwarg.
- Prefer concrete typed parameters with serializable defaults.
- Preserve dtype unless intentionally changing representation.
- Do not close over GUI, filesystem, or live viewer state."""


class CustomFunctionTemplateSection(
    StaticAuthoringContextSection,
    CustomFunctionAuthoringRulesContext,
):
    section_id = "custom_function_template"
    content = '''=== MINIMAL NUMPY CUSTOM FUNCTION ===
```python
from openhcs.core.memory import numpy
import numpy as np

@numpy
def robust_clip(image, low_percentile: float = 1.0, high_percentile: float = 99.0):
    """Clip intensities to robust percentiles while preserving dtype."""
    array = np.asarray(image)
    if array.size == 0:
        return image
    low = np.percentile(array, low_percentile)
    high = np.percentile(array, high_percentile)
    if high <= low:
        return image
    return np.clip(array, low, high).astype(array.dtype, copy=False)
```'''


class CustomFunctionRegistrationSection(
    StaticAuthoringContextSection,
    CustomFunctionAuthoringRulesContext,
):
    section_id = "custom_function_registration"
    content = '''=== REGISTRATION WORKFLOW ===
Use the existing public custom-function manager; do not duplicate registry or file-loading logic.

```python
from openhcs.processing.custom_functions import CustomFunctionManager

CustomFunctionManager().register_from_code(source_code, persist=True)
```

Then call openhcs_search_functions with the function name, describe the returned function_id, and draft a FunctionStep from that function_id.'''


class FirstUseOrientationSection(
    StaticAuthoringContextSection,
    FirstUseWorkflowContext,
):
    section_id = "first_use_orientation"
    content = """=== FIRST-USE OPERATIONAL ROUTES ===
- If you do not already know OpenHCS, read this first_use context before choosing tools. It is the front-door model for the compiler/runtime, UI bridge, CellProfiler compatibility, source universes, artifacts, and review workflow.
- If the user has an image folder, use the plate-data route: inspect the folder, query/sample representative files, read example/axis docs, draft the smallest useful pipeline, inspect the artifact plan, run a bounded validation, then inspect outputs.
- If the OpenHCS UI is open and the user should see or continue editing the work, use the UI-owned route: read state surfaces and code documents, validate/apply code with revision tokens, dispatch selected-plate init/compile/run, then poll state surfaces.
- If the task is unclear, use knowledge search and architecture topics before authoring. Do not ask the user to understand the full tool catalog before you have followed the smallest relevant route."""


class FolderOnboardingStepsSection(
    StaticAuthoringContextSection,
    FolderOnboardingContext,
):
    section_id = "folder_onboarding_steps"
    content = f"""=== FOLDER ONBOARDING WORKFLOW ===
- OpenHCS turns microscope folders and metadata into a typed source model and virtual workspace; inspect that model before guessing filenames or source bindings.
- Confirm wells, sites, channels, Z planes, and timepoints, then choose variable_components based on what each callable must receive and group_by only for dictionary routing.
- Inspect the folder first with {agent_capabilities.inspect_plate_path.name}; if the UI has a selected plate, use {agent_capabilities.ui_inspect_selected_plate_images.name}.
- Query and sample representative files with {agent_capabilities.query_plate_files.name}, {agent_capabilities.sample_plate_image.name}, {agent_capabilities.ui_query_selected_plate_files.name}, or {agent_capabilities.ui_sample_selected_plate_image.name} before drafting source bindings or processing steps.
- Search/read the example corpus and architecture topics before inventing source bindings. Match the folder to existing examples when possible.
- Draft the pipeline through {agent_capabilities.create_pipeline.name}, {agent_capabilities.add_function_step.name}, {agent_capabilities.validate_pipeline.name}, and {agent_capabilities.render_pipeline_source.name}, or edit the PlateManager code document when the running UI should own the state.
- Validate by compiling, inspecting the artifact plan, querying output files, and streaming outputs to a viewer before telling the domain expert the setup is complete."""


class DomainExpertAssistedSetupSection(
    StaticAuthoringContextSection,
    DomainExpertAssistedSetupContext,
):
    section_id = "domain_expert_assisted_setup"
    content = """=== DOMAIN-EXPERT ASSISTED SETUP ===
- If you do not already know OpenHCS, read openhcs_get_authoring_context(kind="first_use") first; this section assumes the core compiler/runtime and UI/code model.
- Translate biology requests into pipeline intent, but keep the operational burden on the agent: inspect the folder, infer axes, use examples, draft the workflow, compile, run a bounded validation, and show reviewable evidence.
- Treat missing channel names, site layout, Z/time semantics, and expected outputs as validation targets. Ask the expert only for domain choices that cannot be inferred from files, metadata, or examples.
- Keep source binding and config changes reviewable in Python/ObjectState so the expert can later reproduce the setup without relying on chat history."""


class UiVisibleWorkflowStepsSection(
    StaticAuthoringContextSection,
    UiVisibleWorkflowContext,
):
    section_id = "ui_visible_workflow_steps"
    content = f"""=== UI-VISIBLE WORKFLOW ===
- ObjectState is the UI state authority; code documents are live typed pycodified projections over UI-reflected objects with revision tokens, not standalone scripts.
- Discover or verify the UI bridge with {agent_capabilities.ui_list_bridges.name} and {agent_capabilities.ui_bridge_status.name}; when multiple bridges exist, pin the descriptor/connection rather than guessing.
- Read the PlateManager state surface with {agent_capabilities.ui_list_state_surfaces.name} and {agent_capabilities.ui_get_state_surface.name}; the selected/source/output rows are the UI authority for visible workflows.
- Read, validate, and apply the PlateManager code document with {agent_capabilities.ui_list_code_documents.name}, {agent_capabilities.ui_get_code_document.name}, {agent_capabilities.ui_validate_code_document.name}, and {agent_capabilities.ui_apply_code_document.name}.
- Dispatch init, compile, and run through {agent_capabilities.ui_selected_plate_workflow.name}; then poll state surfaces and operation status until the UI-visible result is clear.
- Read openhcs_code_ui_interconversion if the UI/code ownership boundary is unclear."""


class HeadlessExecutionStepsSection(
    StaticAuthoringContextSection,
    HeadlessExecutionContext,
):
    section_id = "headless_execution_steps"
    content = f"""=== HEADLESS EXECUTION WORKFLOW ===
- Use {agent_capabilities.create_orchestrator_session.name} for a draft pipeline or {agent_capabilities.create_orchestrator_session_from_pipeline_source.name} for reviewed Python source.
- Compile first with {agent_capabilities.submit_compile.name} or inspect the artifact plan with {agent_capabilities.inspect_pipeline_source_artifact_plan.name}; unresolved source bindings or artifact contracts should be fixed before full execution.
- Run with {agent_capabilities.submit_pipeline_execution.name}, poll with {agent_capabilities.get_execution_status.name}, then inspect/query/sample output plates before claiming success.
- Headless sessions do not update PlateManager selection, snapshots, or output auto-add. Use the UI-visible workflow when those are required."""


class ViewerReviewStepsSection(StaticAuthoringContextSection, ViewerReviewContext):
    section_id = "viewer_review_steps"
    content = f"""=== VIEWER REVIEW WORKFLOW ===
- Stream plate images or result artifacts with {agent_capabilities.stream_plate_files_to_viewer.name} or {agent_capabilities.ui_stream_selected_plate_files_to_viewer.name}.
- Probe or inspect the viewer with {agent_capabilities.probe_viewer_window.name}, {agent_capabilities.get_viewer_window_state.name}, and {agent_capabilities.get_viewer_window_payloads.name}.
- Validate and sample payloads with {agent_capabilities.validate_viewer_window_state.name}, {agent_capabilities.sample_viewer_window_image.name}, and {agent_capabilities.summarize_viewer_window_rois.name}; use ROI summaries and nonzero payload checks as concrete visual QA evidence."""


class CapabilityGroupIndexSection(AuthoringContextSection, AgentCapabilityIndexContext):
    section_id = "capability_group_index"

    @classmethod
    def render(cls, service: "AgentAuthoringContextService") -> str:
        del service
        registry = get_capability_registry()
        capability_by_name = {
            capability.name: capability
            for capability in registry.capabilities
        }
        lines = ["=== CAPABILITY GROUPS ==="]
        for group in registry.groups:
            tool_names = tuple(
                name
                for name in group.capability_names
                if capability_by_name[name].kind is CapabilityKind.TOOL
            )
            resource_names = tuple(
                name
                for name in group.capability_names
                if capability_by_name[name].kind is CapabilityKind.RESOURCE
            )
            lines.append(
                f"- {group.title}: {len(tool_names)} tools, {len(resource_names)} resources"
            )
            if tool_names:
                lines.append(f"  Tools: {', '.join(tool_names)}")
            if resource_names:
                lines.append(f"  Resources: {', '.join(resource_names)}")
        return "\n".join(lines)


class ObjectStateEditingStepsSection(
    StaticAuthoringContextSection,
    ObjectStateEditingContext,
):
    section_id = "objectstate_editing_steps"
    content = f"""=== OBJECTSTATE EDITING WORKFLOW ===
- List scopes and fields with {agent_capabilities.ui_list_object_state_scopes.name} and {agent_capabilities.ui_get_object_state_fields.name}; use {agent_capabilities.ui_describe_object_state_field.name} when a field's semantics are unclear.
- Apply field changes with {agent_capabilities.ui_mutate_object_state_field.name}; keep save/commit explicit through managed UI actions so dirty/default markers remain inspectable.
- Use snapshots, branches, and time-travel-head tools to avoid editing stale time-traveled state."""


class ConfigFieldRequirementLabel(Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"

    @classmethod
    def from_required(cls, required: bool) -> "ConfigFieldRequirementLabel":
        if required:
            return cls.REQUIRED
        return cls.OPTIONAL


class ConfigFieldResolutionLabel(Enum):
    LAZY = ", lazy"
    EAGER = ""

    @classmethod
    def from_lazy(cls, lazy: bool) -> "ConfigFieldResolutionLabel":
        if lazy:
            return cls.LAZY
        return cls.EAGER


class AgentAuthoringContextService:
    """Build bounded, registry-grounded context for agents authoring OpenHCS."""

    def __init__(
        self,
        function_catalog: FunctionCatalogService | None = None,
        config_service: ConfigService | None = None,
        knowledge_base: KnowledgeBaseService | None = None,
        *,
        max_functions: int = 25,
        max_config_fields: int = 8,
    ) -> None:
        self._function_catalog = function_catalog or FunctionCatalogService()
        self._config_service = config_service or ConfigService()
        self._knowledge_base = knowledge_base or KnowledgeBaseService()
        self._max_functions = max_functions
        self._max_config_fields = max_config_fields

    @property
    def function_catalog(self) -> FunctionCatalogService:
        return self._function_catalog

    @property
    def config_service(self) -> ConfigService:
        return self._config_service

    @property
    def knowledge_base(self) -> KnowledgeBaseService:
        return self._knowledge_base

    @property
    def max_functions(self) -> int:
        return self._max_functions

    @property
    def max_config_fields(self) -> int:
        return self._max_config_fields

    def get_authoring_context(self, kind: str = "pipeline") -> AuthoringContext:
        context_declaration = AuthoringContextDeclaration.from_request(kind)
        return AuthoringContext(
            schema_version=SCHEMA_VERSION,
            kind=context_declaration.require_kind(),
            content="\n\n".join(
                render_authoring_context_sections(context_declaration, self)
            ),
        )

    def knowledge_document_content(
        self,
        *,
        document_id: str,
        section_id: str | None = None,
        max_chars: int = 2_000,
    ) -> str:
        document = self.knowledge_base.get_document(
            KnowledgeBaseDocumentRequest.from_fields(
                document_id=document_id,
                section_id=section_id,
                max_chars=max_chars,
            )
        )
        if document.errors:
            messages = "; ".join(error.message for error in document.errors)
            raise RuntimeError(
                f"Failed to render authoring knowledge document {document_id!r}: {messages}"
            )
        return document.content

    def get_bounded_authoring_context(
        self,
        request: AuthoringContextRequest,
    ) -> AuthoringContext:
        if request.max_chars < 0:
            raise ValueError("max_chars must be nonnegative.")
        context = self.get_authoring_context(request.kind)
        if len(context.content) <= request.max_chars:
            return context
        return replace(
            context,
            content=(
                context.content[:request.max_chars]
                + f"\n...<truncated {len(context.content) - request.max_chars} chars>"
            ),
        )
