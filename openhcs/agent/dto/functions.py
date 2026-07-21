"""Function catalog DTOs for agent-facing OpenHCS discovery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from openhcs.agent.dto.common import AgentResultEnvelope, SCHEMA_VERSION


DEFAULT_FUNCTION_DETAIL_DOC_CHARS = 6_000


@dataclass(frozen=True)
class FunctionIdentity:
    function_id: str


@dataclass(frozen=True)
class ImportPathRef:
    import_path: str


@dataclass(frozen=True, slots=True)
class FunctionSearchRequest:
    query: str | None = None
    library: str | None = None
    limit: int = 50
    compact_signatures: bool = True


@dataclass(frozen=True, slots=True)
class FunctionDetailRequest(FunctionIdentity):
    max_doc_chars: int | None = DEFAULT_FUNCTION_DETAIL_DOC_CHARS
    compact_signature: bool = True


@dataclass(frozen=True, slots=True)
class CustomFunctionRegistrationRequest:
    """Register custom function source through the public custom-function manager."""

    source_code: str
    persist: bool = True
    compact_signature: bool = True


class FunctionParameterSource(str, Enum):
    """Nominal source of one callable parameter in agent-facing docs."""

    AGENT = "agent"
    PRIMARY_INPUT = "runtime_primary_input"
    ARTIFACT_INPUT = "runtime_artifact_input"
    RUNTIME_PARAMETER = "runtime_parameter"
    RUNTIME_ADAPTER = "runtime_adapter"


@dataclass(frozen=True, slots=True)
class FunctionCatalogEntry(FunctionIdentity, ImportPathRef):
    name: str
    module: str
    library: str
    signature: str
    summary: str | None
    backend_tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FunctionParameterSpec:
    name: str
    annotation: str | None
    default_repr: str | None
    required: bool
    supplied_by: str = "agent"
    description: str | None = None
    enum_import_path: str | None = None
    enum_members: tuple[str, ...] = ()
    enum_values: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FunctionArtifactSpec:
    name: str
    kind: str
    required: bool = True
    sidecar_role: str | None = None
    materialization_uses_source_identity_filename: bool = False


@dataclass(frozen=True, slots=True)
class CellProfilerArtifactBindingSummary:
    """One module-owned setting that declares a compile-time artifact term."""

    direction: str
    kind: str
    setting_names: tuple[str, ...]
    parameter_name: str | None = None
    runtime_parameter_name: str | None = None
    repeated: bool = False


@dataclass(frozen=True, slots=True)
class CellProfilerModuleDeclarationSummary:
    module_name: str
    declaration_class: str
    validated: bool
    function_names: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    artifact_bindings: tuple[CellProfilerArtifactBindingSummary, ...] = ()
    exact_artifact_contract_requires_compilation: bool = True


@dataclass(frozen=True, slots=True)
class FunctionRuntimeContractSummary:
    callable_kind: str
    processing_contract: str | None = None
    declared_processing_contract: str | None = None
    runtime_bound_parameters: tuple[str, ...] = ()
    required_variable_components: tuple[str, ...] = ()
    artifact_inputs: tuple[FunctionArtifactSpec, ...] = ()
    artifact_outputs: tuple[FunctionArtifactSpec, ...] = ()
    cellprofiler_module: CellProfilerModuleDeclarationSummary | None = None
    source_binding_rule: str | None = None
    materialization_rule: str | None = None
    measurement_rule: str | None = None
    pattern_compatibility_rule: str | None = None


@dataclass(frozen=True, slots=True)
class FunctionDetail:
    schema_version: str
    entry: FunctionCatalogEntry
    parameters: tuple[FunctionParameterSpec, ...]
    doc: str | None
    runtime_contract: FunctionRuntimeContractSummary | None = None
    doc_truncated: bool = False
    doc_chars: int = 0
    max_doc_chars: int | None = None


@dataclass(frozen=True, slots=True)
class FunctionCatalogPage:
    schema_version: str
    items: tuple[FunctionCatalogEntry, ...]
    total: int
    limit: int
    query: str | None = None
    library: str | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class CustomFunctionRegistrationResult(AgentResultEnvelope):
    """Result of registering custom function source through the public manager."""

    registered_count: int = 0
    persisted: bool = True
    storage_dir: str | None = None
    source_file_paths: tuple[str, ...] = ()
    functions: tuple[FunctionCatalogEntry, ...] = ()
    next_steps: tuple[str, ...] = ()


def catalog_page(
    *,
    items: tuple[FunctionCatalogEntry, ...],
    total: int,
    limit: int,
    query: str | None,
    library: str | None,
) -> FunctionCatalogPage:
    return FunctionCatalogPage(
        schema_version=SCHEMA_VERSION,
        items=items,
        total=total,
        limit=limit,
        query=query,
        library=library,
    )
