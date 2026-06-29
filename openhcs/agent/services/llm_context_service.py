"""Prompt/context projection for OpenHCS authoring agents."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum

from openhcs.agent.dto.authoring import AuthoringContext, AuthoringContextRequest
from openhcs.agent.dto.common import SCHEMA_VERSION
from openhcs.agent.dto.config import ConfigSchema
from openhcs.agent.ui_bridge_actions import PlateManagerAction
from openhcs.agent.ui_bridge_identities import (
    PlateManagerOrchestratorCodeDocumentIdentity,
    PlateManagerStateSurfaceIdentityDeclaration,
)
from openhcs.agent.services.architecture_projection_service import (
    CellProfilerTranslationArchitectureTopic,
)
from openhcs.agent.services.config_service import ConfigService
from openhcs.agent.services.function_catalog_service import FunctionCatalogService


class AuthoringContextKind(Enum):
    PIPELINE = "pipeline"
    CUSTOM_FUNCTION = "custom_function"

    @classmethod
    def from_request(cls, kind: str) -> "AuthoringContextKind":
        try:
            return cls(kind.casefold())
        except ValueError as exc:
            raise ValueError(
                f"kind must be one of: {', '.join(cls.allowed_values())}"
            ) from exc

    @classmethod
    def allowed_values(cls) -> tuple[str, ...]:
        return tuple(kind.value for kind in cls)

    @property
    def rules_section(self) -> str:
        if self is AuthoringContextKind.CUSTOM_FUNCTION:
            return """=== CUSTOM FUNCTION RULES ===
- Decorate exactly one callable with a memory decorator such as @numpy.
- The first image-like argument is supplied by OpenHCS at runtime; do not pass it as a FunctionStep kwarg.
- Prefer concrete typed parameters with serializable defaults.
- Preserve dtype unless intentionally changing representation.
- Do not close over GUI, filesystem, or live viewer state."""
        return """=== PIPELINE AUTHORING RULES ===
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
        func=(gaussian_blur, {"sigma": 1.0}),
        name="Denoise images",
        processing_config=LazyProcessingConfig(
            variable_components=[VariableComponents.SITE],
            group_by=GroupBy.CHANNEL,
        ),
    )
]
```"""


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
        *,
        max_functions: int = 25,
        max_config_fields: int = 8,
    ) -> None:
        self._function_catalog = function_catalog or FunctionCatalogService()
        self._config_service = config_service or ConfigService()
        self._max_functions = max_functions
        self._max_config_fields = max_config_fields

    def get_authoring_context(self, kind: str = "pipeline") -> AuthoringContext:
        context_kind = AuthoringContextKind.from_request(kind)

        if context_kind is AuthoringContextKind.CUSTOM_FUNCTION:
            parts = [
                _state_code_roundtrip_section(),
                _custom_function_runtime_section(),
                _custom_function_imports_section(),
                context_kind.rules_section,
                _custom_function_template_section(),
                _custom_function_registration_section(),
            ]
        else:
            parts = [
                _pipeline_system_model_section(),
                _cellprofiler_translation_bridge_section(),
                _runtime_ui_coordination_section(),
                _state_code_roundtrip_section(),
                _custom_function_runtime_section(),
                _source_binding_workflow_section(),
                _example_corpus_section(),
                _core_imports_section(),
                context_kind.rules_section,
                self._config_schema_section(),
                self._function_catalog_section(),
            ]
        return AuthoringContext(
            schema_version=SCHEMA_VERSION,
            kind=context_kind.value,
            content="\n\n".join(parts),
        )

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

    def _function_catalog_section(self) -> str:
        page = self._function_catalog.search(
            limit=self._max_functions,
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
            if entry.summary is None:
                summary = ""
            else:
                summary = f" - {entry.summary}"
            lines.append(f"- {entry.function_id}: `{entry.signature}`{summary}")
        if page.total > len(page.items):
            lines.append(f"\n... {page.total - len(page.items)} more functions are available through openhcs_search_functions.")
        return "\n".join(lines)

    def _config_schema_section(self) -> str:
        schemas = (
            self._config_service.describe_schema("global"),
            self._config_service.describe_schema("pipeline"),
        )
        lines = [
            "=== CONFIG SCHEMA HINTS ===",
            "Use openhcs_describe_config_schema for the full reflected schema before setting non-obvious fields.",
        ]
        for schema in schemas:
            lines.extend(self._schema_lines(schema))
        return "\n".join(lines)

    def _schema_lines(self, schema: ConfigSchema) -> list[str]:
        visible_fields = [
            field
            for field in schema.fields
            if not field.ui_hidden
        ]
        lines = [f"\n## {schema.config_type}"]
        for field in visible_fields[:self._max_config_fields]:
            requirement = ConfigFieldRequirementLabel.from_required(field.required)
            resolution = ConfigFieldResolutionLabel.from_lazy(field.lazy)
            lines.append(
                f"- {field.path}: {field.type_repr} ({requirement.value}{resolution.value})"
            )
        if len(visible_fields) > self._max_config_fields:
            remaining = len(visible_fields) - self._max_config_fields
            lines.append(f"- ... {remaining} more fields")
        return lines


def _core_imports_section() -> str:
    return """=== CORE PIPELINE IMPORTS ===
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


def _pipeline_system_model_section() -> str:
    return """=== FRONTLOADED OPENHCS MODEL ===
- FunctionStep is the authored step boundary: function reference(s), step name, lazy step configs, and source_bindings.
- PipelineConfig and GlobalPipelineConfig carry lazy defaults; inspect resolved values through ObjectState/UI tools before assuming None means inactive.
- Compilation resolves FunctionStep declarations, source bindings, artifact contracts, and materialization into an execution plan; runtime tools should not bypass that boundary.
- Use architecture tools when the model is unclear: openhcs_explain_architecture with pipeline_model, cellprofiler_translation, source_semantics, or execution_runtime.
- Read knowledge docs openhcs_data_dimensions, openhcs_function_patterns, openhcs_pattern_grouping_special_outputs, and openhcs_code_ui_interconversion for axis/state/code details."""


def _cellprofiler_translation_bridge_section() -> str:
    topic = CellProfilerTranslationArchitectureTopic()
    concept_lines = "\n".join(
        f"- {concept}" for concept in topic.concepts_text
    )
    note_lines = "\n".join(
        f"- {note}" for note in topic.cellprofiler_translation_notes_text
    )
    return f"""=== CELLPROFILER MENTAL MODEL BRIDGE ===
- If you know CellProfiler, use that model first: CellProfiler modules, Images, Objects, Measurements, SaveImages, and ExportToDatabase are translated into OpenHCS source bindings, FunctionStep declarations, artifact contracts, runtime values, and materialized outputs.
- Use CellProfiler examples and .cppipe files to understand the biological intent, module order, named images/objects, measurements, and expected artifacts; then verify the current OpenHCS tool surface before claiming a direct import command exists.
- Detailed architecture topic: openhcs_explain_architecture(topic_id="cellprofiler_translation").
{concept_lines}
{note_lines}"""


def _runtime_ui_coordination_section() -> str:
    code_document_id = PlateManagerOrchestratorCodeDocumentIdentity.require_value()
    state_surface_id = PlateManagerStateSurfaceIdentityDeclaration.require_value()
    workflow_names = ", ".join(
        action.value
        for action in PlateManagerAction
        if action.plate_operation is not None
    )
    return f"""=== RUNTIME AND UI COORDINATION ===
- If the OpenHCS UI is open and the user should see the work, use the UI bridge path: read/apply {code_document_id} with plate_paths and pipeline_data, then dispatch {workflow_names} through openhcs_ui_selected_plate_workflow.
- Direct orchestrator sessions are headless runtime jobs: they can execute, stream to viewers, and write output plates, but they do not make PlateManager rows, ObjectState snapshots, or selected UI state visible unless the UI path is used.
- After UI-owned runs, poll {state_surface_id} to confirm source and output rows, then inspect/query/sample the output plate and validate viewer layers from those visible paths."""


def _state_code_roundtrip_section() -> str:
    return """=== OBJECTSTATE AND CODE ROUNDTRIP ===
- ObjectState is the edit/provenance layer for config, pipelines, steps, and code surfaces; use object-state-scopes and object-state-fields before assuming UI text or raw None values.
- Field markers are semantic: * means unsaved/dirty, _ means differs from defaults, inherited/resolved values show lazy/default resolution even when raw values are None.
- Code documents are bidirectional UI<->code projections with revision tokens; get-code-document, validate-code-document, and apply-code-document preserve reviewable Python and ObjectState updates.
- UI mutations can create snapshots and branches; use window snapshots, ObjectState scopes, and time-travel-head to understand whether the UI is at branch head before applying edits."""


def _custom_function_runtime_section() -> str:
    return """=== CUSTOM FUNCTIONS AND RUNTIME OUTPUTS ===
- Custom functions are registry functions used inside FunctionStep.func patterns: callable, (callable, kwargs), list chains, or component/group keyed patterns.
- Dict patterns are routed by processing_config.group_by; list and single-callable patterns use the default group but still participate in grouped artifact naming.
- processing_config.variable_components are the axes stacked into each callable input array; group_by is the partition/fanout axis for routing, execution, and output identity.
- Memory decorators such as @numpy define runtime memory conversion; FunctionStep processing_config defines grouping, axes, and input source.
- Declare non-image outputs with artifact_outputs plus MaterializationSpec/CsvOptions/JsonOptions, or ROI/image presets such as segmentation_mask_rois and tiff_stack.
- Artifact graph planning owns output names, sidecar results, materialization intent, and downstream artifact inputs; do not encode those as ad hoc files or hidden globals.
- ROI and image review flows use materialized/streamed artifacts through viewer streaming tools; after execution inspect selected-plate-files, viewer-rois, viewer-payloads, and sample-viewer-image."""


def _source_binding_workflow_section() -> str:
    return """=== SOURCE-BINDING WORKFLOW ===
- Start custom plates by inspecting real inventory: inspect-plate, query-plate-files, selected-plate-images, and selected-plate-files expose virtual paths, axes, results, and handler confidence; selected-plate review tools can target selected/source/output rows.
- Filename and folder semantics belong in MetadataExtractionRule, SourceBindingsConfig, and StepSourceBindingsConfig, not local filename parsing or raw string conventions.
- FunctionStep.source_bindings is the step-local semantic-input contract; pipeline_config.source_bindings_config is the source discovery/default contract.
- Virtual workspaces map logical OpenHCS virtual filenames to source paths and metadata; agents should query virtual paths/source metadata instead of assuming physical folder layout is execution layout.
- Compile/session inspection can expose source_workspace files with virtual_path, full_virtual_path, source_path, and source_metadata for source-bound pipelines.
- For source-bound custom plates, use reviewed Python/code-document or CellProfiler import/generator flows until the compact v1 draft-step API grows a nominal source-binding DTO. Do not pass ad hoc source_bindings dicts."""


def _example_corpus_section() -> str:
    return """=== EXAMPLE CORPUS FIRST ===
- Search/read knowledge document openhcs_example_corpus_map before inventing source bindings or pipeline structure.
- benchmark/cellprofiler_pipelines contains in-tree .cppipe examples plus checked-in OpenHCS equivalents for ExampleHuman and ExampleFly.
- benchmark/native_refs/official30_scoped_rows contains 30 native CellProfiler reference .cppipe runs; generated OpenHCS equivalents are produced by converter/adapter paths, not checked in as one static .py per case.
- benchmark/pipelines and openhcs/processing/presets/pipelines contain native OpenHCS examples and reusable preset pipelines."""


def _custom_function_imports_section() -> str:
    return """=== CORE CUSTOM FUNCTION IMPORTS ===
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


def _custom_function_template_section() -> str:
    return '''=== MINIMAL NUMPY CUSTOM FUNCTION ===
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


def _custom_function_registration_section() -> str:
    return '''=== REGISTRATION WORKFLOW ===
Use the existing public custom-function manager; do not duplicate registry or file-loading logic.

```python
from openhcs.processing.custom_functions import CustomFunctionManager

CustomFunctionManager().register_from_code(source_code, persist=True)
```

Then call openhcs_search_functions with the function name, describe the returned function_id, and draft a FunctionStep from that function_id.'''
