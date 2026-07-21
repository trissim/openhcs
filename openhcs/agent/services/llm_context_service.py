"""Prompt/context projection for OpenHCS authoring agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from enum import Enum
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.agent.authoring_contexts import (
    AuthoringContextDeclaration,
    CellProfilerTranslationContext,
    CustomFunctionAuthoringRulesContext,
    CustomFunctionRuntimeContext,
    DebuggingWorkflowContext,
    DomainExpertAssistedSetupContext,
    ExampleCorpusContext,
    FirstUseWorkflowContext,
    FolderOnboardingContext,
    HeadlessExecutionContext,
    ObjectStateEditingContext,
    PipelineAuthoringRulesContext,
    PipelineSystemModelContext,
    RuntimeUiCoordinationContext,
    SourceBindingWorkflowContext,
    StateCodeRoundtripContext,
    UiVisibleWorkflowContext,
    ViewerReviewContext,
)
from openhcs.agent.capabilities import agent_capabilities
from openhcs.agent.dto.authoring import AuthoringContext, AuthoringContextRequest
from openhcs.agent.dto.common import SCHEMA_VERSION
from openhcs.agent.dto.config import ConfigSchema
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
    PipelineDebugSessionStateSurfaceIdentityDeclaration,
    UiLiveOverviewStateSurfaceIdentityDeclaration,
)
from openhcs.ui.shared.plate_manager_code_document import (
    PlateManagerCodeNamespaceField,
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
    """Render one intent header, its matching sections, and bounded deepening links."""
    route = context.require_route()
    sections = tuple(
        section_type.render(service)
        for section_type in AuthoringContextSection.__registry__.values()
        if section_type.belongs_to(context)
    )
    header = f"=== {route.title.upper()} ===\nUse this context when {route.use_when}."
    deepening = service.render_knowledge_targets(context)
    return (header, *sections, deepening)


class PipelineSystemModelSection(
    AuthoringContextSection,
    PipelineSystemModelContext,
):
    section_id = "pipeline_system_model"

    @classmethod
    def render(cls, service: "AgentAuthoringContextService") -> str:
        del service
        return f"""=== EXECUTION MENTAL MODEL ===
- The public declaration is PipelineConfig plus an ordered list[FunctionStep]. FunctionStep.func accepts a callable, (callable, kwargs), a list chain, or a dictionary keyed by compiled group identity; nested lazy configs own processing, sources, materialization, data types, filters, and viewers.
- Five independent questions govern each step: variable_components says what varies along the assembled array axis; group_by partitions the assembled value and routes only dictionary patterns; input_source chooses previous-step or pipeline-start main flow; callable ArtifactSpec declarations name semantic inputs and outputs; compiled materialization plans decide what persists. ProcessingContract separately declares per-plane, whole-stack, flexible, or volumetric-to-slice callable semantics.
- A separately named source is not a third InputSource value: it is a declared artifact input satisfied by step source bindings or a prior producer. Runtime-store availability lets downstream steps consume a typed image, label, measurement, relationship, table, grid, or external resource, but does not itself promise a persistent file.
- Configuration has two independent inheritance axes. Scope precedence is global -> pipeline -> step, where a lazy raw None means inherit. Nominal specialization says which related policies share a default: for example, the pipeline well filter constrains the execution domain, step well-filter specializations inherit that scope unless overridden, and viewer/checkpoint filters narrow their own emission or persistence. A path-planning override is a sibling persistence policy and must not leak into viewer filtering.
- Workload filters and output filters are not interchangeable. `pipeline_config.well_filter_config` is applied before per-well contexts compile, so it reduces loading, memory, and processing. A viewer-local well filter only suppresses viewer emission after that well was processed; a materialization/path filter only suppresses its persistent output. For fast inspection, constrain the broadest safe execution scope first, then inherit or narrow at specialized consumers.
- ObjectState resolves those scope and nominal relationships once. Compilation validates sources, function patterns, artifacts, memory, scope, and workers, then emits a CompiledExecutionBundle. Workers consume that typed bundle and record typed runtime values; they do not reconstruct contracts from strings or sidecars.
- CellProfiler uses this same model: setup modules contribute source bindings and executable modules become ordinary FunctionStep declarations. Each CellProfilerModule subclass owns the semantics it declares supported, so CellProfiler knowledge transfers while parity remains pipeline-and-result evidence.
- If this model is unclear, call {agent_capabilities.explain_architecture.name} for only the relevant pipeline, source, CellProfiler, or runtime topic, or retrieve one targeted knowledge document listed below."""


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
- Detailed architecture topic: {agent_capabilities.explain_architecture.name} with topic_id="cellprofiler_translation".
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
        document_fields = ", ".join(
            field.value for field in PlateManagerCodeNamespaceField
        )
        workflow_names = ", ".join(
            action.value
            for action in PlateManagerAction
            if action.plate_operation is not None
        )
        return f"""=== RUNTIME AND UI COORDINATION ===
- If the OpenHCS UI is open and the user should see the work, use the UI bridge path: read/apply {code_document_id} as one complete document ({document_fields}), then dispatch {workflow_names} through {agent_capabilities.ui_selected_plate_workflow.name}.
- Direct orchestrator sessions are headless runtime jobs: they can execute, stream to viewers, and write output plates, but they do not make PlateManager rows, ObjectState snapshots, or selected UI state visible unless the UI path is used.
- A UI mutation receipt and a workflow terminal state are separate evidence. Retain the returned operation_id, wait once with {agent_capabilities.ui_wait_for_operation.name} for accepted/completed/rejected receipt terminality, then read {state_surface_id} with {agent_capabilities.ui_get_state_surface.name} until the selected plate's compile/run state is terminal.
- After UI-owned runs, confirm source and output rows on that state surface, then inspect/query/sample the output plate and validate viewer layers from those visible paths."""


class StateCodeRoundtripSection(
    AuthoringContextSection,
    StateCodeRoundtripContext,
):
    section_id = "state_code_roundtrip"

    @classmethod
    def render(cls, service: "AgentAuthoringContextService") -> str:
        del service
        return f"""=== OBJECTSTATE AND CODE ROUNDTRIP ===
- ObjectState is the edit/provenance layer for config, pipelines, steps, and code surfaces; list scopes and fields through {agent_capabilities.ui_list_object_state_scopes.name} and {agent_capabilities.ui_get_object_state_fields.name} before assuming UI text or raw None values.
- Field markers are semantic: * means unsaved/dirty, _ means differs from defaults, inherited/resolved values show lazy/default resolution even when raw values are None.
- Code documents are live typed bidirectional UI<->code projections over ObjectState-backed UI objects, not freeform files. Read, validate, and apply them through their declared capabilities with fresh revision tokens.
- UI mutations can create snapshots and branches; inspect the registered state surfaces and time-travel head before applying edits. Retrieve the targeted code/UI knowledge document below only when this ownership boundary needs more detail."""


class CustomFunctionRuntimeSection(
    AuthoringContextSection,
    CustomFunctionRuntimeContext,
):
    section_id = "custom_function_runtime"

    @classmethod
    def render(cls, service: "AgentAuthoringContextService") -> str:
        del service
        return f"""=== CUSTOM FUNCTIONS AND RUNTIME OUTPUTS ===
- Custom functions are registry functions used inside FunctionStep.func patterns: callable, (callable, kwargs), list chains, or component/group keyed patterns.
- A reviewed custom function becomes an ordinary registry-described callable; it does not create a separate pipeline mechanism. If {agent_capabilities.search_functions.name} has no contract-compatible implementation for an intended operation, use this route, register the callable, then describe its reflected contract before composing it.
- CLAHE, median denoising, background subtraction, and flat-field correction are example preprocessing intents, not a promised built-in catalog. Implement only the missing operation, keep it backend-agnostic at the pipeline boundary, and let its memory decorator, ProcessingContract, and artifact declarations state compatibility.
- Dict patterns are routed by processing_config.group_by; list and single-callable patterns apply the same callable chain to each grouped 3D array.
- processing_config.variable_components are the axes stacked into each callable input array; group_by groups the remaining 3D arrays and only selects callable branches for dict patterns.
- Memory decorators such as @numpy define runtime memory conversion; FunctionStep processing_config defines grouping, axes, and input source.
- Declare outputs with typed ArtifactSpec values. The decorator owns their input/output role. MeasurementsArtifactType receives schema-bearing ColumnarRows; ObjectLabelsArtifactType receives complete integer label arrays and may use segmentation_mask_rois for materialization.
- Artifact graph planning owns output names, sidecar results, materialization intent, and downstream artifact inputs; do not encode those as ad hoc files or hidden globals. Request the viewer-review context only after execution produces artifacts to inspect."""


class SourceBindingWorkflowSection(
    AuthoringContextSection,
    SourceBindingWorkflowContext,
):
    section_id = "source_binding_workflow"

    @classmethod
    def render(cls, service: "AgentAuthoringContextService") -> str:
        del service
        from openhcs.microscopes.microscope_base import (
            MicroscopeHandler,
            MicroscopeSourceSelectionRole,
        )

        handler_lines: list[str] = []
        for microscope_type, handler_type in MicroscopeHandler.__registry__.items():
            role = handler_type.source_selection_role()
            if role is MicroscopeSourceSelectionRole.DECLARED_FILE_FALLBACK:
                binding_role = "bindings own ingestion and semantic naming"
            elif handler_type.projects_declared_source_bindings():
                binding_role = "bindings may select/name handler-emitted planes"
            else:
                binding_role = "handler does not project declared bindings"
            handler_lines.append(
                f"- {microscope_type}: role={role.value}; {binding_role}. "
                f"{handler_type.source_selection_guidance()}"
            )
        registered_handlers = "\n".join(handler_lines)

        return f"""=== SOURCE-BINDING WORKFLOW ===
=== REGISTERED INGESTION OWNERS (LIVE REGISTRY) ===
{registered_handlers}
- Selection rule: prefer a format-specific owner only when its declaration-owned detection/subset policy accepts the source. Auto-detection commonly requires complete vendor metadata. When plate inspection reports format_specific_handler_candidates, read the live handler guidance: some owners permit explicit partial exports, while owners that require their complete detection contract leave loose ordinary files to SourceBindingsHandler with explicitly declared semantics.
- Separate ingestion ownership from semantic source selection. A recognized HCS layout keeps its format-specific microscope handler; CZI, OME-TIFF, and other supported rich containers use the broad Bio-Formats/store handler only when no stronger registered format-specific owner matches. SourceBindingsConfig may then name or select exact planes only when that handler declares projection support; it does not open the container or replace the handler.
- SourceBindingsHandler is the microscope-independent ingestion fallback for arbitrary image folders and unsupported layouts. If auto-detection does not recognize a folder but SourceBindingsConfig is non-empty, OpenHCS selects that nominal handler and projects the declared TIFF, PNG, JPEG, or other registered ordinary files; do not force structured CZI/OME data through this fallback.
- Start by inventorying and sampling real files. Use typed SourceFilterClause values to bound the source universe, MetadataExtractionRule named regex captures to declare well/site/channel/Z/time metadata, and NamedSourceBinding selectors plus component_identity to give semantic aliases such as DNA or GFP.
- The pipeline-level pipeline_config.source_bindings_config declares the full named physical source universe and the inputs consumed by the nominal handler projection: discovery filters, metadata, cross-alias matching, grouping metadata, stack components, explicit planes, and imported metadata tables. It may correctly contain Hoechst channel 1, MAP2 channel 2, and SMI312 channel 4. FunctionStep.source_bindings is the resolved step-local subset and order: at PIPELINE_START, an omitted value inherits that full pipeline order, while an explicit enabled step config selects and reorders a subset instead of consuming only the previous main-flow result.
- source_stack_components describe axes physically contained inside each selected file/store payload. Do not declare SITE or another source-stack axis for separate ordinary 2-D files; give those files component identities and use processing_config.variable_components to assemble the selected files into a callable stack.
- Keep three layers separate: the full pipeline source universe, the resolved/compiled ordered per-step binding plan, and the callable stack assembled according to variable_components. Callable channel indices are zero-based positions in that per-step assembled stack, not physical microscope CHANNEL values. If the step plan orders (SMI312 channel 4, Hoechst channel 1) with variable_components=CHANNEL, use callable positions 0 and 1. Source provenance remains channel 4 and channel 1, and route provenance retains those declared values when the output carries them.
- The simplest canonical MetaXpress neurite step omits the step subset, inherits (Hoechst channel 1, MAP2 channel 2, SMI312 channel 4), sets input_source=PIPELINE_START and variable_components=[CHANNEL], and uses nuclear index 0 plus neurite index 2. The efficient alternative explicitly selects/reorders (SMI312, Hoechst), giving neurite index 0 and nuclear index 1. The generic compiler regression protects any explicit ordered primary-plane subset for an implicit-main-flow callable; it is not a MetaXpress channel map.
- variable_components=[CHANNEL] assembles that stack. group_by=CHANNEL does not: group_by partitions an already assembled value and routes dictionary patterns. Because the MetaXpress callable is non-dictionary and CHANNEL already owns its assembled axis, the overlap is redundant and compilation normalizes group_by to GroupBy.NONE.
- Step source subsets/reordering select or re-enter original declared sources, especially at PIPELINE_START or a deliberate source branch. They do not reinterpret a previous-step output after stitching, Z/channel projection, or filtering changes its axes/cardinality. Follow current artifact provenance plus the downstream variable_components declaration; group_by then partitions that already assembled value.
- Source bindings compile into a source universe, per-step binding plans, and a virtual workspace. Query projected virtual paths and source_metadata rather than treating physical filenames, decoders, or storage backends as the pipeline ABI. Source-satisfied typed artifact inputs do not require invented runtime producer plans.
- Validate on representative files: inspect the full pipeline source workspace, then the resolved FunctionStep.source_bindings subset/order, then CompiledSourceBindingPlan.bindings, runtime-matched files, and current viewer routes. An unexpectedly empty compiled step plan when two bindings were selected explains why all three universe planes may flow. Do not guess a third callable index: use {agent_capabilities.get_viewer_window_payloads.name} to compare the current execution's layer/payload route keys and physical component values. MAP2 channel 2 is valid in the source workspace; it is a leaked source only on a current route for a step whose resolved/compiled plan selects SMI312 and Hoechst. A route from another submission or a step that selects MAP2 is different evidence. Conflicts and missing required aliases must be fixed in the declarations, never hidden by fallback matching.
- Through MCP, reflect PipelineConfig first. Config patches can construct the nested typed source_bindings_config; reviewed Python/code-document flows can express the full model. Use UI code documents when the running GUI must own the state, or reviewed pipeline source for an exposed headless route. Do not pass ad hoc source_bindings dictionaries or invent field names from stale examples."""


class ExampleCorpusSection(AuthoringContextSection, ExampleCorpusContext):
    section_id = "example_corpus"

    @classmethod
    def render(cls, service: "AgentAuthoringContextService") -> str:
        del service
        return f"""=== EXAMPLE CORPUS FIRST ===
- Search with {agent_capabilities.search_knowledge.name} for the biological task plus "OpenHCS Python" or "official30 recipe" before inventing source bindings or pipeline structure; follow the returned document_id and section_id rather than guessing a case name.
- The official30 knowledge target listed below is the broad current corpus. Each exact <case>-openhcs-python section is generated lazily through the public CellProfiler importer and defines PipelineConfig plus FunctionStep declarations; retrieve only the matching section with a max_chars bound large enough for the source, and require truncated=false before validating or applying it.
- The example-corpus-map target identifies the smaller current native preset authority and separates validated examples from older benchmark/debug migration evidence.
- Treat structured source or conversion errors as authoritative. Do not substitute stale checked-in scripts or raw .cppipe text when an exact generated section is unavailable."""


class PipelineRulesSection(
    AuthoringContextSection,
    PipelineAuthoringRulesContext,
):
    section_id = "pipeline_rules"

    @classmethod
    def render(cls, service: "AgentAuthoringContextService") -> str:
        del service
        return f"""=== PIPELINE AUTHORING WORKFLOW ===
- Search {agent_capabilities.search_functions.name} from the user's biological or processing intent, then call {agent_capabilities.describe_function.name} for each candidate before supplying non-default parameters. Do not start from an arbitrary function dump.
- Pipeline composition is registry- and contract-driven, not limited to named recipes or presets. Any registered callable, including one registered through the reviewed custom-function route, can participate when its reflected CallableContract is compatible with the step's array shape, ProcessingContract, memory type, runtime-bound parameters, and artifact flow. Compose compatible preprocessing as ordered FunctionSteps, or as a regular-callable list chain only when the described contracts permit it, before the analysis callable.
- Search CLAHE, median denoising, background subtraction, and flat-field correction as processing intents, then describe the exact candidates returned by the live registry. These examples do not promise that a particular backend or package is installed. If no exact compatible candidate is returned, request the custom_function context and register a reviewed implementation rather than substituting a similar search hit.
- Treat dependency readiness as live evidence. In particular, do not infer that BaSiCPy is usable from a source module, import path, or documentation match; claim it only when the exact callable is returned by {agent_capabilities.search_functions.name} and its described dependency/contract plus pipeline validation succeed.
- An opinionated analysis callable such as MetaXpress neurite outgrowth can still follow compatible preprocessing steps. The CellProfiler-derived workflow remains the more explicit module-by-module reference when comparing equivalent preprocessing and analysis stages; generated CellProfiler lowering normally keeps one module contract per FunctionStep.
- Create one typed draft with {agent_capabilities.create_pipeline.name}; add ordinary single-function steps through {agent_capabilities.add_function_step.name}. Function references come from the registry, and reflected request/config schemas own accepted values. The incremental add-step request does not express a list chain or group-keyed dictionary; for those patterns use one reviewed complete PipelineDocument/code document, or start from the exact importer-generated OpenHCS source and revise it.
- FunctionStep.func may be one callable, a callable-plus-kwargs pair, a chain, or a group-keyed dictionary. Answer the five dataflow questions in the execution model separately. Put variable_components, group_by, and input_source inside processing_config; they are not direct FunctionStep fields. Treat callable-owned artifact contracts as constraints to inspect, not fields to re-declare on the step.
- Reflect the relevant nested config path before editing. A lazy raw None usually means inherit; do not copy fields from examples or flatten pipeline, step, source, materialization, and streaming scopes.
- Validate with {agent_capabilities.validate_pipeline.name}, then render reviewed Python with {agent_capabilities.render_pipeline_source.name}. The complete PipelineDocument contains PipelineConfig plus the ordered FunctionStep list; never send a steps-only fragment or a parallel config side channel.
- Before execution, inspect the compiled artifact and materialization plans. Callable declarations own semantic image, label, measurement, relationship, table, grid, and external-resource contracts; filenames and tuple positions do not. Confirm separately which runtime values are only available during execution and which outputs have persistent targets."""


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
            (
                f"Use {agent_capabilities.describe_config_schema.name} for the "
                "top-level family map, then pass one returned path_prefix for its "
                "reflected nested schema before editing fields."
            ),
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
- Declare measurements as MeasurementsArtifactType and return ColumnarRows; declare segmentation as ObjectLabelsArtifactType and return the complete integer label array.
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
    AuthoringContextSection,
    CustomFunctionAuthoringRulesContext,
):
    section_id = "custom_function_registration"

    @classmethod
    def render(cls, service: "AgentAuthoringContextService") -> str:
        del service
        return f"""=== REGISTRATION WORKFLOW ===
- Read the targeted custom-function and artifact-contract knowledge documents below before inventing decorators or payload types.
- Register reviewed source through {agent_capabilities.register_custom_function.name}; that capability delegates to the nominal CustomFunctionManager and owns persistence and registry refresh.
- Search with {agent_capabilities.search_functions.name}, describe the returned function_id with {agent_capabilities.describe_function.name}, then request the pipeline context to add the function and validate its compiled artifact plan."""


class FirstUseOrientationSection(
    AuthoringContextSection,
    FirstUseWorkflowContext,
):
    section_id = "first_use_orientation"

    @classmethod
    def render(cls, service: "AgentAuthoringContextService") -> str:
        del service
        route_lines = []
        for declaration in AuthoringContextDeclaration.__registry__.values():
            if declaration is cls or declaration.require_kind() == "first_use":
                continue
            route = declaration.require_route()
            route_lines.append(
                f'- kind="{declaration.require_kind()}" — {route.title}: {route.use_when}.'
            )
        routes = "\n".join(route_lines)
        return f"""=== CORE MODEL IN ONE PASS ===
- OpenHCS is a typed compiler/runtime for high-content microscopy. It projects microscope data into a storage-independent virtual workspace, compiles PipelineConfig plus ordered FunctionStep declarations, resolves multidimensional grouping and artifact contracts, executes through typed runtime values, and validates results through plate and viewer evidence.
- Choose the state owner before choosing file formats or tools. If the user asks to open, show, continue, or edit work in the desktop, the UI-owned route always takes precedence over folder inspection or a headless session.

=== CHOOSE ONE TASK ROUTE ===
Request exactly one matching context with {agent_capabilities.get_authoring_context.name}; deepen into another only when the workflow reaches that boundary.
{routes}

=== SAFE FIRST ACTION ===
- Start read-only. Inspect the active surface through {agent_capabilities.list_capabilities.name}; its workflow groups, target contexts, side effects, and security metadata are the authority for what this server exposes.
- Request the matching task context before mutation. Do not read every knowledge document, enumerate every function, invent config fields, or load full-resolution image data up front.
- Before any write or execution, show the exact target and intended change, refresh revision/request tokens, obtain approval, validate, and compile before running."""


class FolderOnboardingStepsSection(
    StaticAuthoringContextSection,
    FolderOnboardingContext,
):
    section_id = "folder_onboarding_steps"
    content = f"""=== FOLDER ONBOARDING WORKFLOW ===
- This is the headless/read-only data route. If the user expects the work to appear in the desktop, stop and request kind="ui_visible_workflow"; its PlateManager state and code document remain the owner throughout onboarding.
- OpenHCS turns microscope files and metadata into a typed source model and storage-independent virtual workspace. Inspecting a path is decoder evidence only: it does not add a UI plate, choose persistent configuration, or author a pipeline.
- Keep the authoritative HCS/Bio-Formats handler for a valid native layout or rich CZI/OME container. Parser recognition alone is insufficient when that format owner requires a complete metadata detection contract; intentionally loose ordinary files may instead use SourceBindingsHandler with explicit component identities. SourceBindingsConfig may name/select planes emitted by a recognized handler only when that handler declares projection support.
- Inspect with {agent_capabilities.inspect_plate_path.name}, query a bounded representative inventory with {agent_capabilities.query_plate_files.name}, and sample a small region with {agent_capabilities.sample_plate_image.name}. Never infer dimensions from names or load a full rich container merely to identify it.
- Confirm wells, sites, channels, Z planes, timepoints, source diagnostics, and selected resolution provenance before authoring. If one container has multiple ambiguous datasets or samples, resolve the typed inspection error rather than guessing an index.
- Search/read the example corpus and architecture topics before inventing source bindings. Match the folder to existing examples when possible.
- Once the source model is proven, request kind="pipeline" for typed authoring and kind="headless_execution" for compile/run. Validate compiled source bindings and artifact plans before full execution, then query bounded outputs and request kind="viewer_review" when visual evidence matters."""


class DomainExpertAssistedSetupSection(
    StaticAuthoringContextSection,
    DomainExpertAssistedSetupContext,
):
    section_id = "domain_expert_assisted_setup"
    content = """=== DOMAIN-EXPERT ASSISTED SETUP ===
- Translate biology requests into pipeline intent, but keep the operational burden on the agent: inspect the folder, infer axes, use examples, draft the workflow, compile, run a bounded validation, and show reviewable evidence.
- Treat missing channel names, site layout, Z/time semantics, and expected outputs as validation targets. Ask the expert only for domain choices that cannot be inferred from files, metadata, or examples.
- Choose visible versus headless ownership first. Request kind="ui_visible_workflow" when the expert should see or continue editing the setup; otherwise request kind="folder_onboarding" to inspect data read-only before authoring.
- Keep source binding and config changes reviewable in Python/ObjectState so the expert can reproduce the setup without relying on chat history. Do not claim completion until compiled artifacts, bounded result samples, and viewer or measurement evidence match the requested biological outcome."""


class UiVisibleWorkflowStepsSection(
    StaticAuthoringContextSection,
    UiVisibleWorkflowContext,
):
    section_id = "ui_visible_workflow_steps"
    content = f"""=== UI-VISIBLE WORKFLOW ===
- ObjectState is the UI state authority; code documents are live typed pycodified projections over UI-reflected objects with revision tokens, not standalone scripts.
- This route takes precedence over headless folder onboarding whenever the user asks to open, inspect, or continue work in the desktop. Plate inspection remains diagnostic evidence and must not become a parallel UI setup path.
- Discover or verify the UI bridge with {agent_capabilities.ui_list_bridges.name} and {agent_capabilities.ui_bridge_status.name}; when multiple bridges exist, pin the descriptor/connection rather than guessing.
- Read {UiLiveOverviewStateSurfaceIdentityDeclaration.require_value()} first for the current ObjectState token, revision, snapshot, windows, statuses, and operations contributed by registered UI providers.
- Read the PlateManager state surface with {agent_capabilities.ui_list_state_surfaces.name} and {agent_capabilities.ui_get_state_surface.name}; the selected/source/output rows are the UI authority for visible workflows.
- Read, validate, and apply the PlateManager code document with {agent_capabilities.ui_list_code_documents.name}, {agent_capabilities.ui_get_code_document.name}, {agent_capabilities.ui_validate_code_document.name}, and {agent_capabilities.ui_apply_code_document.name}.
- Add the containing plate directory and initialize with auto-detection. Recognized HCS layouts and CZI/OME stores keep their detected handler; use SourceBindingsConfig only for semantic selection/naming after discovery, or as the SourceBindingsHandler ingestion declaration for an otherwise unrecognized arbitrary-file folder.
- For a write: read, explain, obtain approval, re-read, validate, then apply using the fresh document revision and approved confirmation policy; retain the mutation receipt and snapshot facts.
- Dispatch init, compile, and run through {agent_capabilities.ui_selected_plate_workflow.name} using its current selection revision token. Wait for the returned operation_id once with {agent_capabilities.ui_wait_for_operation.name}, then read the Plate Manager state surface for the separate workflow terminal condition.
- Retrieve the targeted code/UI knowledge document below only if the ownership boundary remains unclear."""


class HeadlessExecutionStepsSection(
    StaticAuthoringContextSection,
    HeadlessExecutionContext,
):
    section_id = "headless_execution_steps"
    content = f"""=== HEADLESS EXECUTION WORKFLOW ===
- Use {agent_capabilities.create_orchestrator_session.name} for a draft pipeline or {agent_capabilities.create_orchestrator_session_from_pipeline_source.name} for reviewed Python source. The source route requires one complete PipelineDocument with pipeline_config and pipeline_steps; never send either through a parallel side channel.
- Inspect first with {agent_capabilities.inspect_pipeline_source_artifact_plan.name}: it returns compiled source-workspace, step, group, artifact-output, and persistent-materialization plans, but not values that only exist during execution. Compile with {agent_capabilities.submit_compile.name} after those plans are sound.
- Run with {agent_capabilities.submit_pipeline_execution.name}, poll with {agent_capabilities.get_execution_status.name}, then inspect/query/sample output plates before claiming success.
- A normal completed-job status is lifecycle evidence, not a dump of RuntimeValueStore. Request kind="debugging" when an intermediate invocation value or artifact must be inspected, and kind="viewer_review" when image/label presentation is the evidence.
- Headless sessions do not update PlateManager selection, snapshots, or output auto-add. Use the UI-visible workflow when those are required."""


class DebuggingWorkflowSection(
    AuthoringContextSection,
    DebuggingWorkflowContext,
):
    section_id = "debugging_workflow"

    @classmethod
    def render(cls, service: "AgentAuthoringContextService") -> str:
        del service
        debug_surface_id = PipelineDebugSessionStateSurfaceIdentityDeclaration.require_value()
        return f"""=== DEBUGGING WORKFLOW ===
- Diagnose the first failing boundary. A declaration/config problem, compiled-plan problem, paused runtime-value problem, persisted-output problem, and viewer-presentation problem require different evidence; do not reconstruct one from another's filenames or logs.
- Pin one current evidence scope before comparing results: execution_id, debug_session_id when paused, runtime/viewer connection, step/group/invocation identity, and current route keys. Compare the resolved/compiled source-binding order and zero-based callable stack positions with the physical component values on those routes; never merge persistent layers or artifacts from another submission.
- Before execution, call {agent_capabilities.inspect_pipeline_source_artifact_plan.name}. Inspect the exact source workspace, step/group identities, typed artifact inputs/outputs, and materialization paths. This is compiled intent, not proof that a runtime value was produced.
- Before changing biological thresholds or algorithm parameters, inspect bounded raw evidence in order. Sample the exact source and routed image planes with {agent_capabilities.sample_plate_image.name} and {agent_capabilities.sample_viewer_window_image.name}; compare dtype, shape, reported intensity statistics, bounded pixels, and nonzero/saturation behavior for the same field and coordinates. Then inspect typed label arrays and ROI payloads with {agent_capabilities.get_viewer_window_payloads.name} and {agent_capabilities.summarize_viewer_window_rois.name}, including label identities, cardinality, bounds, and member relationships rather than display colors.
- Inspect schema-bearing per-object measurement rows next, not only summaries: verify row/object cardinality, object or label identifiers, source/image identity, units, representative bounded rows, and distributions or outliers against the typed label/ROI payloads. A plausible aggregate count cannot repair mismatched source order, labels, or object ownership.
- In the desktop route, select the real Plate Manager row and compile it, then read {debug_surface_id} with {agent_capabilities.ui_get_state_surface.name}. List {agent_capabilities.ui_list_actions.name} for its debug toolbar and invoke the declared Debug, Step, Run to Pause, Restart, Stop, or Inspect Runtime action only when its typed enabled state permits it.
- The debug-session surface owns phase, session/execution identity, source group, cursor, current/last frame, snapshot-store identity, and exact disabled reasons. Wait for an operation receipt with {agent_capabilities.ui_wait_for_operation.name}, then re-read the surface; do not repeatedly invoke an action while it is pending.
- Read the paused worker directly with {agent_capabilities.inspect_debug_runtime_values.name}, passing the active debug_session_id and its runtime-server connection. The bounded view shows actual invocation parameters plus runtime-value, measurement, relationship, and artifact references without bulk array transfer; follow those typed references to inspect bounded row or array content at its owner.
- A debug snapshot makes replay/inspection possible; it is not user-facing persistence. Use step materialization for an ordinary main-flow checkpoint, artifact materialization for a named typed output, or viewer review for visual evidence.
- Only after that raw evidence chain is coherent should visual interpretation with the biologist guide scientific parameter changes. Screenshots are secondary presentation evidence. If the run completed normally, use output-plate inventory/sampling and viewer validation; if it failed, keep the first typed compile/runtime error and inspect only the owning boundary before changing the pipeline."""


class ViewerReviewStepsSection(StaticAuthoringContextSection, ViewerReviewContext):
    section_id = "viewer_review_steps"
    content = f"""=== VIEWER REVIEW WORKFLOW ===
- Start from the user's scientific question and define what the final view must let them conclude. Intermediate source, mask, segmentation, and skeleton layers are diagnostic evidence; they do not replace a final result layer that communicates the biological output.
- Review one current execution in raw-evidence order: confirm execution and route identity plus resolved source order; inspect bounded source/output arrays and statistics; inspect typed label IDs and ROI payloads; reconcile schema-bearing per-object measurement rows with those objects; only then interpret the visualization with the biologist.
- When the result is a relationship, the final streamed artifact must encode that relationship directly. For example, a neuron-assignment result should give each cell body and its assigned neurites the same stable object or label identity; the viewer may derive matching display colors from that identity. Separate body-label and global-skeleton layers leave the assignment ambiguous and are not a complete user-facing visualization.
- Keep that final interpretation reproducible: produce it as a callable-owned typed image/label artifact and stream or materialize it through its compiled plan. Do not fabricate scientific relationships with viewer-only annotations. Inspect both useful intermediate layers and the final result, and treat the review as incomplete when the user cannot visually verify the requested conclusion.
- Three mechanisms are independent. A step's napari_streaming_config or fiji_streaming_config displays eligible outputs during execution; step_materialization_config persists that step's ordinary main-flow result; typed artifact materialization persists named images, labels, measurements, relationships, tables, grids, or external resources according to the compiled artifact plan.
- To view one specific step, keep pipeline/global viewer enablement disabled and set enabled=true only on that FunctionStep's chosen viewer config. Put a one-well constraint such as B03 in `pipeline_config.well_filter_config` when the goal is a fast diagnostic run: that bounds loading and execution, and the step viewer cannot emit wells that never entered the compiled domain. Leave the viewer's own well_filter inherited unless it must narrow that domain further. Use persistent=true when the detached viewer must remain after completion; display, batching, transport, and port choices also inherit unless explicitly overridden.
- If the viewer is the only desired image destination, set `pipeline_config.path_planning_config.well_filter=0` to keep the automatic final main-flow output runtime-only. This does not disable an explicitly enabled step checkpoint or named-artifact materialization; those have separate owning configs and compiled plans.
- Reflect current fields with {agent_capabilities.describe_config_schema.name} using config_type="step" and path_prefix="napari_streaming_config", "fiji_streaming_config", or "step_materialization_config"; do not copy a historical field list. Recompile after changing a viewer or materialization declaration.
- Stream existing plate images or result artifacts independently with {agent_capabilities.stream_plate_files_to_viewer.name} or {agent_capabilities.ui_stream_selected_plate_files_to_viewer.name} when rerunning a step is unnecessary.
- Probe or inspect the viewer with {agent_capabilities.probe_viewer_window.name}, {agent_capabilities.get_viewer_window_state.name}, and {agent_capabilities.get_viewer_window_payloads.name}.
- Treat layer visibility, selection, navigation, zoom, and screenshots as user-controlled presentation state. A user may change them while an agent works. Establish produced content from raw route payloads, label identities, ROI summaries, and bounded array samples; use screenshots only to assess rendering and ergonomics, never to infer that hidden data is absent or to override raw artifact identity.
- Offer compact opinionated analysis and a modular CellProfiler-derived pipeline as distinct workflow choices when both fit the task. Shared operations, successful compilation, synthetic tests, layer counts, or an earlier run do not establish output equivalence or result validation. Claim either only from its own accepted evidence, and claim equivalence only after accepted same-field executions reconcile source order, raw arrays, typed labels/ROIs, and per-object measurements for both workflows.
- Validate and sample payloads with {agent_capabilities.validate_viewer_window_state.name}, {agent_capabilities.sample_viewer_window_image.name}, and {agent_capabilities.summarize_viewer_window_rois.name}; use ROI summaries and nonzero payload checks as concrete visual QA evidence. Layer existence and nonzero pixels prove transport, not scientific adequacy: also verify that the final layer visibly encodes the requested objects, measurements, or relationships."""


class ObjectStateEditingStepsSection(
    StaticAuthoringContextSection,
    ObjectStateEditingContext,
):
    section_id = "objectstate_editing_steps"
    content = f"""=== OBJECTSTATE EDITING WORKFLOW ===
- List scopes and fields with {agent_capabilities.ui_list_object_state_scopes.name} and {agent_capabilities.ui_get_object_state_fields.name}; use {agent_capabilities.ui_describe_object_state_field.name} when a field's semantics are unclear.
- Apply field changes with {agent_capabilities.ui_mutate_object_state_field.name}; keep save/commit explicit through managed UI actions so dirty/default markers remain inspectable.
- A field-mutation request token is idempotency, not a base-revision guard. Confirm branch head, re-read the exact field immediately before one small approved mutation, and verify the field and related state surface immediately afterward; prefer a revision-checked code document for related atomic changes.
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

    def get_authoring_context(self, kind: str = "first_use") -> AuthoringContext:
        context_declaration = AuthoringContextDeclaration.from_request(kind)
        return AuthoringContext(
            schema_version=SCHEMA_VERSION,
            kind=context_declaration.require_kind(),
            content="\n\n".join(
                render_authoring_context_sections(context_declaration, self)
            ),
        )

    def render_knowledge_targets(
        self,
        context: type[AuthoringContextDeclaration],
    ) -> str:
        """Project declaration-owned deepening targets from the live KB catalog."""
        targets = context.require_route().knowledge_targets
        catalog = self.knowledge_base.list_documents()
        documents_by_id = {
            document.document_id: document for document in catalog.documents
        }
        lines = [
            "=== DEEPEN ONLY WHEN NEEDED ===",
            (
                f"Retrieve one relevant source-backed target with "
                f"{agent_capabilities.get_knowledge_document.name}; use its returned "
                "section ids to narrow further instead of loading every document."
            ),
        ]
        for target in targets:
            try:
                document = documents_by_id[target.document_id]
            except KeyError as exc:
                raise RuntimeError(
                    f"Authoring context {context.require_kind()!r} targets unknown "
                    f"knowledge document {target.document_id!r}."
                ) from exc
            target_id = target.document_id
            if target.section_id is not None:
                target_id = f"{target_id}#{target.section_id}"
            lines.append(f"- {target_id} — {document.title}: {document.summary}")
        return "\n".join(lines)

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
                context.content[: request.max_chars]
                + f"\n...<truncated {len(context.content) - request.max_chars} chars>"
            ),
        )
