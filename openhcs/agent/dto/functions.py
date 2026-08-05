"""Function catalog DTOs for agent-facing OpenHCS discovery."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

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


class FunctionCatalogControlMessageType(str, Enum):
    """Execution-server control messages for the live function catalog."""

    READ_CATALOG = "openhcs_function_catalog_read"
    SEARCH_CATALOG = "openhcs_function_catalog_search"
    READ_DETAIL = "openhcs_function_detail_read"


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

    @property
    def membership_identity(
        self,
    ) -> tuple[str, str, str, str, str, tuple[str, ...]]:
        """Return the endpoint callable identity owned by this declaration."""

        return (
            self.function_id,
            self.import_path,
            self.name,
            self.module,
            self.library,
            self.backend_tags,
        )


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
    revision: str
    items: tuple[FunctionCatalogEntry, ...]
    total: int
    limit: int
    query: str | None = None
    library: str | None = None


@dataclass(frozen=True, slots=True)
class FunctionCatalogControlRequest:
    """Request the complete catalog owned by one execution endpoint."""

    compact_signatures: bool = True


@dataclass(frozen=True, slots=True)
class FunctionDetailControlRequest(FunctionIdentity):
    """Request one detail from an exact server catalog revision."""

    catalog_revision: str
    max_doc_chars: int | None = DEFAULT_FUNCTION_DETAIL_DOC_CHARS
    compact_signature: bool = True


@dataclass(frozen=True, slots=True)
class FunctionCatalogControlPayload:
    """Wire payload for one complete-catalog request."""

    request: FunctionCatalogControlRequest
    message_type: FunctionCatalogControlMessageType = (
        FunctionCatalogControlMessageType.READ_CATALOG
    )

    @classmethod
    def from_request(
        cls,
        request: FunctionCatalogControlRequest,
    ) -> "FunctionCatalogControlPayload":
        return cls(request=request)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "FunctionCatalogControlPayload":
        message_type = FunctionCatalogControlMessageType(str(payload["type"]))
        if message_type is not FunctionCatalogControlMessageType.READ_CATALOG:
            raise ValueError(
                f"Unsupported function catalog control type {message_type.value!r}."
            )
        request = payload["request"]
        if not isinstance(request, FunctionCatalogControlRequest):
            raise TypeError(
                "Function catalog control payload requires "
                "FunctionCatalogControlRequest, got "
                f"{type(request).__name__}."
            )
        return cls(request=request, message_type=message_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.message_type.value,
            "request": self.request,
        }


@dataclass(frozen=True, slots=True)
class FunctionDetailControlPayload:
    """Wire payload for one revision-checked function-detail request."""

    request: FunctionDetailControlRequest
    message_type: FunctionCatalogControlMessageType = (
        FunctionCatalogControlMessageType.READ_DETAIL
    )

    @classmethod
    def from_request(
        cls,
        request: FunctionDetailControlRequest,
    ) -> "FunctionDetailControlPayload":
        return cls(request=request)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "FunctionDetailControlPayload":
        message_type = FunctionCatalogControlMessageType(str(payload["type"]))
        if message_type is not FunctionCatalogControlMessageType.READ_DETAIL:
            raise ValueError(
                f"Unsupported function detail control type {message_type.value!r}."
            )
        request = payload["request"]
        if not isinstance(request, FunctionDetailControlRequest):
            raise TypeError(
                "Function detail control payload requires "
                "FunctionDetailControlRequest, got "
                f"{type(request).__name__}."
            )
        return cls(request=request, message_type=message_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.message_type.value,
            "request": self.request,
        }


@dataclass(frozen=True, slots=True)
class FunctionSearchControlPayload:
    """Wire payload carrying the existing catalog search request unchanged."""

    request: FunctionSearchRequest
    message_type: FunctionCatalogControlMessageType = (
        FunctionCatalogControlMessageType.SEARCH_CATALOG
    )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "FunctionSearchControlPayload":
        message_type = FunctionCatalogControlMessageType(str(payload["type"]))
        if message_type is not FunctionCatalogControlMessageType.SEARCH_CATALOG:
            raise ValueError(
                f"Unsupported function search control type {message_type.value!r}."
            )
        request = payload["request"]
        if not isinstance(request, FunctionSearchRequest):
            raise TypeError(
                "Function search control payload requires FunctionSearchRequest, got "
                f"{type(request).__name__}."
            )
        return cls(request=request, message_type=message_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.message_type.value,
            "request": self.request,
        }


@dataclass(frozen=True, slots=True)
class FunctionCatalogControlResponse:
    """Typed response carrying one endpoint-owned catalog projection."""

    catalog: FunctionCatalogPage

    def to_control_response(self) -> dict[str, Any]:
        return {"status": "ok", "catalog": self.catalog}

    @classmethod
    def from_control_response(
        cls,
        payload: Mapping[str, Any],
    ) -> "FunctionCatalogControlResponse":
        if payload.get("status") != "ok":
            raise RuntimeError(str(payload.get("error") or payload))
        catalog = payload["catalog"]
        if not isinstance(catalog, FunctionCatalogPage):
            raise TypeError(
                "Function catalog response requires FunctionCatalogPage, got "
                f"{type(catalog).__name__}."
            )
        return cls(catalog=catalog)


@dataclass(frozen=True, slots=True)
class FunctionDetailControlResponse:
    """Typed response carrying one revision-consistent function detail."""

    detail: FunctionDetail

    def to_control_response(self) -> dict[str, Any]:
        return {"status": "ok", "detail": self.detail}

    @classmethod
    def from_control_response(
        cls,
        payload: Mapping[str, Any],
    ) -> "FunctionDetailControlResponse":
        if payload.get("status") != "ok":
            raise RuntimeError(str(payload.get("error") or payload))
        detail = payload["detail"]
        if not isinstance(detail, FunctionDetail):
            raise TypeError(
                "Function detail response requires FunctionDetail, got "
                f"{type(detail).__name__}."
            )
        return cls(detail=detail)


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
        revision=function_catalog_revision(items),
        items=items,
        total=total,
        limit=limit,
        query=query,
        library=library,
    )


def function_catalog_revision(items: tuple[FunctionCatalogEntry, ...]) -> str:
    """Return a deterministic revision of endpoint callable membership.

    Presentation choices such as compact signatures and bounded summaries do not
    change which callable identity the endpoint owns, so they are intentionally
    excluded from the membership revision.
    """

    encoded = json.dumps(
        [item.membership_identity for item in items],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
