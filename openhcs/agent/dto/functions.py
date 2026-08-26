"""Function catalog DTOs for agent-facing OpenHCS discovery."""

from __future__ import annotations

import hashlib
import json
from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Generic, Self, TypeVar, cast

from zmqruntime.messages import MessageFields, ResponseType
from zmqruntime.startup import EndpointStartupStatus

from openhcs.agent.dto.common import SCHEMA_VERSION, AgentResultEnvelope
from openhcs.core.function_reference import FunctionReference

DEFAULT_FUNCTION_DETAIL_DOC_CHARS = 6_000


class FunctionCatalogControlMessageType(str, Enum):
    """Execution-server control messages for the live function catalog."""

    READ_CATALOG = "openhcs_function_catalog_read"
    SEARCH_CATALOG = "openhcs_function_catalog_search"
    READ_DETAIL = "openhcs_function_detail_read"
    READ_REFERENCE = "openhcs_function_reference_read"
    REGISTER_CUSTOM = "openhcs_custom_function_register"


class FunctionCatalogControlField(str, Enum):
    """OpenHCS-specific fields carried by function-catalog control messages."""

    REQUEST = "request"
    CATALOG = "catalog"
    PREPARATION = "preparation"
    DETAIL = "detail"
    REFERENCE = "reference"
    RESULT = "result"


@dataclass(frozen=True)
class FunctionIdentity:
    function_id: str


@dataclass(frozen=True)
class ImportPathRef:
    import_path: str


@dataclass(frozen=True, slots=True)
class FunctionCatalogControlRequestABC(ABC):
    """Nominal owner of one function-catalog control request identity."""

    message_type: ClassVar[FunctionCatalogControlMessageType]

    def __init_subclass__(cls) -> None:
        if "message_type" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must declare its function-catalog message_type."
            )

    @classmethod
    def from_control_payload(cls, payload: Mapping[str, Any]) -> Self:
        """Recover this exact request type from its declared wire message."""

        actual_message_type = FunctionCatalogControlMessageType(
            str(payload[MessageFields.TYPE])
        )
        if actual_message_type is not cls.message_type:
            raise ValueError(
                f"Expected function catalog control type {cls.message_type.value!r}, "
                f"got {actual_message_type.value!r}."
            )
        request = payload[FunctionCatalogControlField.REQUEST.value]
        if not isinstance(request, cls):
            raise TypeError(
                f"{cls.__name__} control payload requires {cls.__name__}, got "
                f"{type(request).__name__}."
            )
        return request


@dataclass(frozen=True, slots=True)
class FunctionSearchRequest(FunctionCatalogControlRequestABC):
    query: str | None = None
    library: str | None = None
    limit: int = 50
    compact_signatures: bool = True

    message_type = FunctionCatalogControlMessageType.SEARCH_CATALOG


@dataclass(frozen=True, slots=True)
class FunctionDetailRequest(FunctionIdentity):
    max_doc_chars: int | None = DEFAULT_FUNCTION_DETAIL_DOC_CHARS
    compact_signature: bool = True


ControlRequestT = TypeVar(
    "ControlRequestT",
    bound=FunctionCatalogControlRequestABC,
)
ControlResponseT = TypeVar("ControlResponseT")


@dataclass(frozen=True, slots=True)
class CustomFunctionRegistrationRequest(FunctionCatalogControlRequestABC):
    """Register custom function source through the public custom-function manager."""

    source_code: str
    persist: bool = True
    compact_signature: bool = True

    message_type = FunctionCatalogControlMessageType.REGISTER_CUSTOM


class FunctionParameterSource(str, Enum):
    """Nominal source of one callable parameter in agent-facing docs."""

    _runtime_description: str | None

    def __new__(
        cls,
        value: str,
        runtime_description: str | None,
    ) -> FunctionParameterSource:
        member = str.__new__(cls, value)
        member._value_ = value
        member._runtime_description = runtime_description
        return member

    AGENT = ("agent", None)
    PRIMARY_INPUT = (
        "runtime_primary_input",
        "Supplied by OpenHCS from the FunctionStep input image payload; do not "
        "pass this as a function kwarg.",
    )
    ARTIFACT_INPUT = (
        "runtime_artifact_input",
        "Supplied by OpenHCS from a declared artifact input during pipeline "
        "execution; do not pass this as a function kwarg.",
    )
    RUNTIME_PARAMETER = (
        "runtime_parameter",
        "Supplied by OpenHCS runtime execution infrastructure; do not pass this "
        "as a function kwarg.",
    )
    RUNTIME_ADAPTER = (
        "runtime_adapter",
        "Supplied by OpenHCS as a runtime adapter object; do not pass this as a "
        "function kwarg.",
    )

    @property
    def runtime_description(self) -> str | None:
        """Explain the runtime-owned source directly from its declaration."""

        return self._runtime_description


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
    supplied_by: FunctionParameterSource = FunctionParameterSource.AGENT
    description: str | None = None
    enum_import_path: str | None = None
    enum_members: tuple[str, ...] = ()
    enum_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.supplied_by, FunctionParameterSource):
            raise TypeError(
                "FunctionParameterSpec.supplied_by requires "
                f"FunctionParameterSource, got {type(self.supplied_by).__name__}."
            )


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
class FunctionCatalogControlRequest(FunctionCatalogControlRequestABC):
    """Request the complete catalog owned by one execution endpoint."""

    compact_signatures: bool = True

    message_type = FunctionCatalogControlMessageType.READ_CATALOG


@dataclass(frozen=True, slots=True)
class FunctionDetailControlRequest(
    FunctionIdentity,
    FunctionCatalogControlRequestABC,
):
    """Request one detail from an exact server catalog revision."""

    catalog_revision: str
    max_doc_chars: int | None = DEFAULT_FUNCTION_DETAIL_DOC_CHARS
    compact_signature: bool = True

    message_type = FunctionCatalogControlMessageType.READ_DETAIL


@dataclass(frozen=True, slots=True)
class FunctionReferenceControlRequest(
    FunctionIdentity,
    FunctionCatalogControlRequestABC,
):
    """Request one compiler reference from an exact server catalog revision."""

    catalog_revision: str

    message_type = FunctionCatalogControlMessageType.READ_REFERENCE


@dataclass(frozen=True, slots=True)
class FunctionCatalogControlPayload(Generic[ControlRequestT]):
    """Wire projection of one request declaration and its enum identity."""

    request: ControlRequestT

    @classmethod
    def from_request(cls, request: ControlRequestT) -> Self:
        if not isinstance(request, FunctionCatalogControlRequestABC):
            raise TypeError(
                "Function catalog control payload requires a nominal control "
                f"request, got {type(request).__name__}."
            )
        return cls(request=request)

    def to_dict(self) -> dict[str, Any]:
        return {
            MessageFields.TYPE: self.request.message_type.value,
            FunctionCatalogControlField.REQUEST.value: self.request,
        }


@dataclass(frozen=True, slots=True)
class FunctionCatalogControlResponseBase(Generic[ControlResponseT]):
    """Shared typed transport for one declared successful control response."""

    value: ControlResponseT
    field: ClassVar[FunctionCatalogControlField]
    value_type: ClassVar[type]

    def __init_subclass__(cls) -> None:
        for attribute in ("field", "value_type"):
            if attribute not in cls.__dict__:
                raise TypeError(
                    f"{cls.__name__} must declare function-catalog {attribute}."
                )

    def to_control_response(self) -> dict[str, Any]:
        return {
            MessageFields.STATUS: ResponseType.OK.value,
            self.field.value: self.value,
        }

    @classmethod
    def from_control_response(cls, payload: Mapping[str, Any]) -> Self:
        if payload.get(MessageFields.STATUS) != ResponseType.OK.value:
            raise RuntimeError(str(payload.get(MessageFields.ERROR) or payload))
        value = payload[cls.field.value]
        if not isinstance(value, cls.value_type):
            raise TypeError(
                f"Function catalog response field {cls.field.value!r} requires "
                f"{cls.value_type.__name__}, got {type(value).__name__}."
            )
        return cls(value=cast(ControlResponseT, value))


class FunctionCatalogControlResponse(
    FunctionCatalogControlResponseBase[FunctionCatalogPage]
):
    """Typed response carrying one endpoint-owned catalog projection."""

    field = FunctionCatalogControlField.CATALOG
    value_type = FunctionCatalogPage

    @property
    def catalog(self) -> FunctionCatalogPage:
        return self.value


class FunctionCatalogPreparationStatus(str, Enum):
    """Control response state while endpoint catalog preparation is active."""

    PENDING = "pending"


@dataclass(frozen=True, slots=True)
class FunctionCatalogPreparationControlResponse:
    """Typed signal that the endpoint is actively preparing its catalog."""

    status: EndpointStartupStatus
    retry_after_seconds: float = 0.1

    def to_control_response(self) -> dict[str, Any]:
        return {
            MessageFields.STATUS: FunctionCatalogPreparationStatus.PENDING.value,
            FunctionCatalogControlField.PREPARATION.value: self,
        }

    @classmethod
    def from_control_response(
        cls,
        payload: Mapping[str, Any],
    ) -> FunctionCatalogPreparationControlResponse | None:
        if (
            payload.get(MessageFields.STATUS)
            != FunctionCatalogPreparationStatus.PENDING.value
        ):
            return None
        preparation = payload.get(FunctionCatalogControlField.PREPARATION.value)
        if not isinstance(preparation, cls):
            raise TypeError(
                "Pending function catalog response requires "
                "FunctionCatalogPreparationControlResponse, got "
                f"{type(preparation).__name__}."
            )
        return preparation


class FunctionDetailControlResponse(FunctionCatalogControlResponseBase[FunctionDetail]):
    """Typed response carrying one revision-consistent function detail."""

    field = FunctionCatalogControlField.DETAIL
    value_type = FunctionDetail

    @property
    def detail(self) -> FunctionDetail:
        return self.value


class FunctionReferenceControlResponse(
    FunctionCatalogControlResponseBase[FunctionReference]
):
    """Typed response carrying one server-owned compiler reference."""

    field = FunctionCatalogControlField.REFERENCE
    value_type = FunctionReference

    @property
    def reference(self) -> FunctionReference:
        return self.value


@dataclass(frozen=True, kw_only=True, slots=True)
class CustomFunctionRegistrationResult(AgentResultEnvelope):
    """Result of registering custom function source through the public manager."""

    registered_count: int = 0
    persisted: bool = True
    storage_dir: str | None = None
    source_file_paths: tuple[str, ...] = ()
    functions: tuple[FunctionCatalogEntry, ...] = ()
    next_steps: tuple[str, ...] = ()


class CustomFunctionRegistrationControlResponse(
    FunctionCatalogControlResponseBase[CustomFunctionRegistrationResult]
):
    """Typed response carrying endpoint-owned custom registration results."""

    field = FunctionCatalogControlField.RESULT
    value_type = CustomFunctionRegistrationResult

    @property
    def result(self) -> CustomFunctionRegistrationResult:
        return self.value


def catalog_page(
    *,
    items: tuple[FunctionCatalogEntry, ...],
    catalog_items: tuple[FunctionCatalogEntry, ...],
    total: int,
    limit: int,
    query: str | None,
    library: str | None,
) -> FunctionCatalogPage:
    return FunctionCatalogPage(
        schema_version=SCHEMA_VERSION,
        revision=function_catalog_revision(catalog_items),
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
