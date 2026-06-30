"""ObjectState scope projection provider for the PyQt UI bridge."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass
from dataclasses import is_dataclass
from enum import Enum
from typing import ClassVar

from pyqt_reactive.services.scope_window_navigation import ScopeWindowNavigationService
from pyqt_reactive.services.function_pattern_code_document import (
    FunctionPatternScopeCodeDocumentDriver,
)
from pyqt_reactive.forms.parameter_form_service import ParameterFormService
from pyqt_reactive.services.window_code_document import PYTHON_MIME_TYPE
from pyqt_reactive.services.window_manager import WindowManager
from pyqt_reactive.services.window_navigation import WindowNavigationRequest
from pyqt_reactive.services.parameter_help_service import (
    parameter_help_content,
    resolved_parameter_description,
)
from python_introspect import UnifiedParameterAnalyzer

from openhcs.agent.dto.common import AgentError, JsonValue, SCHEMA_VERSION
from openhcs.agent.dto.ui_bridge import (
    UiCodeDocument,
    UiCodeDocumentApplyRequest,
    UiCodeDocumentApplyResult,
    UiCodeDocumentRequest,
    UiCodeDocumentSummary,
    UiCodeDocumentValidationRequest,
    UiCodeDocumentValidationResult,
    UiCatalogPageMetadata,
    UiMutationReceipt,
    UiObjectStateFieldFilter,
    UiObjectStateFieldProvenance,
    UiObjectStateFieldHelpRequest,
    UiObjectStateFieldHelpResult,
    UiObjectStateFieldMutationRequest,
    UiObjectStateFieldMutationResult,
    UiObjectStateFieldSummary,
    UiObjectStateScopeCatalog,
    UiObjectStateScopeIdentity,
    UiObjectStateScopeListRequest,
    UiObjectStateScopeSummary,
    UiObjectStateValuePreview,
    UiSemanticAddress,
)
from openhcs.agent.serialization import to_jsonable
from openhcs.agent.services.object_state_field_projection import (
    ObjectStateFieldFilterDeclaration,
)
from openhcs.config_framework import ObjectState, ObjectStateRegistry
from openhcs.pyqt_gui.services.ui_bridge_contracts import (
    APPLY_TIME_TRAVEL_OPT_IN_GUARD,
    CONFIRMATION_REQUIRED_GUARD,
    UiBridgeGuardPolicy,
    UiBridgeSnapshotProviderABC,
    UiCodeDocumentProviderIdentity,
    UiObjectStateScopeProviderABC,
    UiObjectStateScopeProviderIdentity,
    SnapshotBackedUiCodeDocumentProviderABC,
)
from openhcs.pyqt_gui.services.ui_agent_bridge import UiCodeDocumentApplyLabel
from openhcs.pyqt_gui.services.ui_bridge_object_state_scope_policy import (
    ObjectStateScopeVisibility,
)
from openhcs.pyqt_gui.services.ui_bridge_registry import (
    UiBridgeProviderSetABC,
    UiBridgeRegistrationContext,
)
from openhcs.pyqt_gui.services.ui_window_ids import OpenHCSUiWindowId


OBJECT_STATE_SCOPE_PROVIDER_ID = "object_state.scopes"
OBJECT_STATE_SCOPE_CODE_DOCUMENT_ID = "object_state_scope"
OBJECT_STATE_SCOPE_CODE_DOCUMENT_PREFIX = f"{OBJECT_STATE_SCOPE_CODE_DOCUMENT_ID}:"
WINDOW_CODE_DOCUMENT_ID = "window_code_document"
WINDOW_CODE_DOCUMENT_PREFIX = f"{WINDOW_CODE_DOCUMENT_ID}:"
OBJECT_STATE_FIELD_PAGE_LIMIT_MAX = 1000
OBJECT_STATE_FIELD_VALUE_REPR_LIMIT = 500
OBJECT_STATE_FIELD_VALUE_PREVIEW_LIMIT = 160
OBJECT_STATE_FIELD_VALUE_PREVIEW_ITEMS = 4

logger = logging.getLogger(__name__)


def agent_object_state_scope_id(scope_id: str | None) -> str | None:
    """Return the stable agent-facing id for an ObjectState registry scope."""

    if scope_id is None:
        return None
    return OpenHCSUiWindowId.agent_window_id_for_manager_scope(scope_id)


@dataclass(frozen=True, slots=True)
class ObjectStateScopeCodeDocumentAddress:
    """Nominal code-document address for one ObjectState scope."""

    scope_id: str

    prefix: ClassVar[str] = OBJECT_STATE_SCOPE_CODE_DOCUMENT_PREFIX
    minimum_scope_id_length: ClassVar[int] = 1

    @classmethod
    def parse(cls, document_id: str) -> "ObjectStateScopeCodeDocumentAddress":
        if not document_id.startswith(cls.prefix):
            raise ValueError(
                "ObjectState scope code document ids must start with "
                f"{cls.prefix!r}; got {document_id!r}."
            )
        scope_id = document_id[len(cls.prefix):]
        if len(scope_id) < cls.minimum_scope_id_length:
            raise ValueError("ObjectState scope code document id is missing a scope id.")
        return cls(scope_id=scope_id)

    @property
    def document_id(self) -> str:
        return f"{self.prefix}{self.scope_id}"

    @property
    def revision_key(self) -> str:
        return f"ui-code-document:{self.document_id}"

    def provider_identity(self, title: str) -> UiCodeDocumentProviderIdentity:
        return UiCodeDocumentProviderIdentity(
            document_id=self.document_id,
            widget_id=OBJECT_STATE_SCOPE_CODE_DOCUMENT_ID,
            title=title,
        )


@dataclass(frozen=True, slots=True)
class WindowCodeDocumentAddress:
    """Nominal code-document address for one open WindowManager code mode."""

    window_id: str

    prefix: ClassVar[str] = WINDOW_CODE_DOCUMENT_PREFIX

    @classmethod
    def parse(cls, document_id: str) -> "WindowCodeDocumentAddress":
        if not document_id.startswith(cls.prefix):
            raise ValueError(
                "Window code-document ids must start with "
                f"{cls.prefix!r}; got {document_id!r}."
            )
        window_id = document_id[len(cls.prefix):]
        if not window_id:
            raise ValueError("Window code-document id is missing a window id.")
        return cls(window_id=window_id)

    @classmethod
    def from_scope_id(cls, scope_id: str) -> "WindowCodeDocumentAddress":
        return cls(
            window_id=OpenHCSUiWindowId.agent_window_id_for_manager_scope(scope_id)
        )

    @property
    def document_id(self) -> str:
        return f"{self.prefix}{self.window_id}"

    @property
    def revision_key(self) -> str:
        return f"ui-code-document:{self.document_id}"

    def provider_identity(self, title: str) -> UiCodeDocumentProviderIdentity:
        return UiCodeDocumentProviderIdentity(
            document_id=self.document_id,
            widget_id=self.window_id,
            title=title,
        )


@dataclass(frozen=True, slots=True)
class ObjectStateFieldProvenanceEffect:
    """Typed effect for projecting optional ObjectState provenance."""

    @classmethod
    def from_state(
        cls,
        state: ObjectState,
        field_path: str,
    ) -> "ObjectStateFieldProvenanceEffect":
        match state.get_provenance(field_path):
            case (source_scope_id, source_type) if source_type is not None:
                return ResolvedObjectStateFieldProvenance(
                    source_scope_id=source_scope_id,
                    source_type=source_type,
                    field_name=ObjectStateScopeProjectionService.field_name(field_path),
                )
            case _:
                return MissingObjectStateFieldProvenance()

    def to_dto(self) -> UiObjectStateFieldProvenance | None:
        return None


@dataclass(frozen=True, slots=True)
class MissingObjectStateFieldProvenance(ObjectStateFieldProvenanceEffect):
    """No inherited ObjectState provenance is available for this field."""


@dataclass(frozen=True, slots=True)
class ResolvedObjectStateFieldProvenance(ObjectStateFieldProvenanceEffect):
    """Validated ObjectState provenance source for one field address."""

    source_scope_id: str | None
    source_type: type
    field_name: str

    def to_dto(self) -> UiObjectStateFieldProvenance | None:
        return UiObjectStateFieldProvenance(
            source_scope_id=agent_object_state_scope_id(self.source_scope_id),
            source_type=ObjectStateScopeProjectionService.type_qualname(self.source_type),
            source_field_path=self.source_field_path(),
        )

    def source_field_path(self) -> str | None:
        source_state = ObjectStateRegistry.get_by_scope(self.source_scope_id)
        if source_state is None:
            return None
        return source_state.project_ui_visible_field_path(
            self.source_type,
            self.field_name,
        )


@dataclass(frozen=True, slots=True)
class ObjectStateFieldSemanticProjection:
    """Shared ObjectState field semantics used by scope and widget projections."""

    state: ObjectState
    field_path: str
    raw_value: object
    resolved_value: object
    resolved_value_available: bool
    provenance: UiObjectStateFieldProvenance | None

    @classmethod
    def from_state(
        cls,
        state: ObjectState,
        field_path: str,
    ) -> "ObjectStateFieldSemanticProjection":
        raw_value = state.parameters[field_path]
        resolved_value = None
        resolved_value_available = False
        try:
            resolved_value = state.get_resolved_value(field_path)
            resolved_value_available = True
        except Exception:
            logger.debug(
                "Failed to resolve ObjectState field %s on scope %r.",
                field_path,
                state.scope_id,
                exc_info=True,
            )
        return cls(
            state=state,
            field_path=field_path,
            raw_value=raw_value,
            resolved_value=resolved_value,
            resolved_value_available=resolved_value_available,
            provenance=ObjectStateFieldProvenanceEffect.from_state(
                state,
                field_path,
            ).to_dto(),
        )

    @property
    def dirty(self) -> bool:
        return self.field_path in self.state.dirty_fields

    @property
    def signature_diff(self) -> bool:
        return self.field_path in self.state.signature_diff_fields

    @property
    def last_changed(self) -> bool:
        return self.field_path == self.state.last_changed_field

    @property
    def raw_value_is_none(self) -> bool:
        return self.raw_value is None

    @property
    def resolved_value_is_none(self) -> bool:
        return self.resolved_value_available and self.resolved_value is None

    @property
    def inherited_value(self) -> bool:
        return self.raw_value is None and (
            self.resolved_value_available and self.resolved_value is not None
        )

    @property
    def semantic_markers(self) -> tuple[str, ...]:
        markers = []
        if self.dirty:
            markers.append("*")
        if self.signature_diff:
            markers.append("_")
        return tuple(markers)

    @property
    def field_name(self) -> str:
        return self.field_path.rsplit(".", 1)[-1]

    @property
    def container_path(self) -> str:
        if "." not in self.field_path:
            return ""
        return self.field_path.rsplit(".", 1)[0]

    @property
    def raw_value_type(self) -> str:
        return self.type_name(self.raw_value)

    @property
    def resolved_value_type(self) -> str | None:
        if not self.resolved_value_available:
            return None
        return self.type_name(self.resolved_value)

    @property
    def raw_json_value(self) -> JsonValue:
        return self.json_value(self.raw_value)

    @property
    def resolved_json_value(self) -> JsonValue | None:
        if not self.resolved_value_available:
            return None
        return self.json_value(self.resolved_value)

    @property
    def raw_value_preview(self) -> UiObjectStateValuePreview:
        return self.value_preview(self.raw_value)

    @property
    def resolved_value_preview(self) -> UiObjectStateValuePreview | None:
        if not self.resolved_value_available:
            return None
        return self.value_preview(self.resolved_value)

    def semantic_address(self, *, window_id: str | None = None) -> UiSemanticAddress:
        return UiSemanticAddress(
            object_state_scope_id=self.agent_scope_id,
            field_path=self.field_path,
            window_id=window_id,
        )

    @property
    def agent_scope_id(self) -> str:
        return agent_object_state_scope_id(self.state.scope_id) or self.state.scope_id

    def to_field_summary(
        self,
        *,
        include_values: bool = False,
        include_description: bool = False,
    ) -> UiObjectStateFieldSummary:
        raw_value = None
        resolved_value = None
        if include_values:
            raw_value = self.raw_json_value
            resolved_value = self.resolved_json_value
        parameter_description = None
        if include_description:
            parameter_description = self.state.parameter_descriptions.get(
                self.field_path
            )
        return UiObjectStateFieldSummary(
            schema_version=SCHEMA_VERSION,
            address=self.semantic_address(
                window_id=OpenHCSUiWindowId.agent_window_id_for_manager_scope(
                    self.state.scope_id
                ),
            ),
            field_name=self.field_name,
            container_path=self.container_path,
            object_state_path_type=self.object_state_path_type,
            raw_value_type=self.raw_value_type,
            resolved_value_type=self.resolved_value_type,
            dirty=self.dirty,
            signature_diff=self.signature_diff,
            last_changed=self.last_changed,
            parameter_description=parameter_description,
            semantic_markers=self.semantic_markers,
            raw_value=raw_value,
            resolved_value=resolved_value,
            raw_value_preview=self.raw_value_preview,
            resolved_value_preview=self.resolved_value_preview,
            raw_value_is_none=self.raw_value_is_none,
            resolved_value_is_none=self.resolved_value_is_none,
            inherited_value=self.inherited_value,
            provenance=self.provenance,
        )

    @property
    def object_state_path_type(self) -> str:
        return self.type_qualname(self.state.type_for_path(self.field_path))

    @staticmethod
    def type_name(value) -> str:
        if value is None:
            return "None"
        return type(value).__name__

    @staticmethod
    def type_qualname(value_type: type | Callable[..., object]) -> str:
        return f"{value_type.__module__}.{value_type.__qualname__}"

    @staticmethod
    def json_value(value) -> JsonValue:
        try:
            return to_jsonable(value)
        except TypeError:
            value_repr = repr(value)
            if len(value_repr) > OBJECT_STATE_FIELD_VALUE_REPR_LIMIT:
                value_repr = (
                    f"{value_repr[:OBJECT_STATE_FIELD_VALUE_REPR_LIMIT]}"
                    "...<truncated>"
                )
            return value_repr

    @classmethod
    def value_preview(cls, value) -> UiObjectStateValuePreview:
        text, truncated = cls.preview_text(value)
        return UiObjectStateValuePreview(
            type_name=cls.type_name(value),
            is_none=value is None,
            text=text,
            truncated=truncated,
        )

    @classmethod
    def preview_text(cls, value) -> tuple[str, bool]:
        text = cls._preview_text_unbounded(value)
        truncated = len(text) > OBJECT_STATE_FIELD_VALUE_PREVIEW_LIMIT
        if truncated:
            text = (
                f"{text[:OBJECT_STATE_FIELD_VALUE_PREVIEW_LIMIT]}"
                "...<truncated>"
            )
        return text, truncated

    @classmethod
    def _preview_text_unbounded(cls, value) -> str:
        if value is None:
            return "None"
        if callable(value):
            return cls._callable_preview_text(value)
        if isinstance(value, Enum):
            return f"{type(value).__name__}.{value.name}"
        if isinstance(value, str):
            return repr(value)
        if isinstance(value, bytes):
            return f"bytes(len={len(value)})"
        if isinstance(value, Mapping):
            keys = tuple(
                cls._preview_item_text(key)
                for key in cls._first_items(value.keys())
            )
            return f"{type(value).__name__}(len={len(value)}, keys={keys!r})"
        if isinstance(value, (Sequence, Set)) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            items = tuple(
                cls._preview_item_text(item)
                for item in cls._first_items(value)
            )
            return f"{type(value).__name__}(len={len(value)}, items={items!r})"
        if is_dataclass(value) and not isinstance(value, type):
            return repr(value)
        return repr(value)

    @staticmethod
    def _first_items(values) -> tuple[object, ...]:
        items = []
        for item in values:
            items.append(item)
            if len(items) >= OBJECT_STATE_FIELD_VALUE_PREVIEW_ITEMS:
                break
        return tuple(items)

    @staticmethod
    def _preview_item_text(value) -> str:
        if callable(value):
            return ObjectStateFieldSemanticProjection._callable_preview_text(value)
        if isinstance(value, Enum):
            return f"{type(value).__name__}.{value.name}"
        return repr(value)

    @staticmethod
    def _callable_preview_text(value: Callable[..., object]) -> str:
        try:
            payload = to_jsonable(value)
        except TypeError:
            return repr(value)
        if not isinstance(payload, Mapping):
            return repr(value)
        import_path = payload.get("import_path")
        if not isinstance(import_path, str):
            return repr(value)
        return import_path


class ObjectStateScopeProjectionService:
    """Project ObjectState registry entries into bounded agent DTOs."""

    def catalog(
        self,
        request: UiObjectStateScopeListRequest,
    ) -> UiObjectStateScopeCatalog:
        visibility = ObjectStateScopeVisibility(request)
        scopes = tuple(
            self.summary(state, request)
            for state in ObjectStateRegistry.get_all()
            if visibility.includes_scope_id(state.scope_id)
        )
        return UiObjectStateScopeCatalog(
            schema_version=SCHEMA_VERSION,
            object_state_token=ObjectStateRegistry.get_token(),
            current_branch=ObjectStateRegistry.get_current_branch(),
            current_snapshot_index=ObjectStateRegistry.get_current_snapshot_index(),
            active=ObjectStateRegistry.is_time_traveling(),
            scopes=scopes,
        )

    def summary(
        self,
        state: ObjectState,
        request: UiObjectStateScopeListRequest,
    ) -> UiObjectStateScopeSummary:
        fields, field_page = self._field_page(state, request)
        return UiObjectStateScopeSummary(
            schema_version=SCHEMA_VERSION,
            identity=UiObjectStateScopeIdentity(
                object_state_scope_id=agent_object_state_scope_id(state.scope_id)
                or state.scope_id,
            ),
            object_type=type(state.object_instance).__name__,
            parameter_count=len(state.parameters),
            dirty_field_count=len(state.dirty_fields),
            signature_diff_field_count=len(state.signature_diff_fields),
            has_unsaved_changes=bool(state.dirty_fields),
            has_default_overrides=bool(state.signature_diff_fields),
            last_changed_field=state.last_changed_field,
            fields=fields,
            field_page=field_page,
        )

    def _field_page(
        self,
        state: ObjectState,
        request: UiObjectStateScopeListRequest,
    ) -> tuple[tuple[UiObjectStateFieldSummary, ...], UiCatalogPageMetadata | None]:
        if not request.include_fields:
            return (), None

        if request.field_paths:
            all_field_paths = tuple(
                field_path
                for field_path in request.field_paths
                if field_path in state.parameters
            )
        else:
            all_field_paths = tuple(sorted(state.parameters.keys()))
        if request.field_filter is not UiObjectStateFieldFilter.ALL:
            all_field_paths = tuple(
                field_path
                for field_path in all_field_paths
                if ObjectStateFieldFilterDeclaration.matches_filter(
                    request.field_filter,
                    self._field_summary(
                        state,
                        field_path,
                        include_values=False,
                        include_description=False,
                    )
                )
            )
        offset = max(0, request.field_offset)
        limit = min(max(0, request.field_limit), OBJECT_STATE_FIELD_PAGE_LIMIT_MAX)
        selected_paths = all_field_paths[offset:offset + limit]
        next_offset = offset + len(selected_paths)
        truncated = next_offset < len(all_field_paths)
        next_page_offset = None
        if truncated:
            next_page_offset = next_offset
        page = UiCatalogPageMetadata(
            limit=limit,
            offset=offset,
            returned_count=len(selected_paths),
            total_count=len(all_field_paths),
            truncated=truncated,
            next_offset=next_page_offset,
        )
        fields = tuple(
            self._field_summary(
                state,
                field_path,
                include_values=request.include_field_values,
                include_description=request.include_field_descriptions,
            )
            for field_path in selected_paths
        )
        return fields, page

    def _field_summary(
        self,
        state: ObjectState,
        field_path: str,
        *,
        include_values: bool,
        include_description: bool,
    ) -> UiObjectStateFieldSummary:
        return ObjectStateFieldSemanticProjection.from_state(
            state,
            field_path,
        ).to_field_summary(
            include_values=include_values,
            include_description=include_description,
        )

    @staticmethod
    def _resolved_type_name(state: ObjectState, field_path: str) -> str | None:
        try:
            return ObjectStateScopeProjectionService.type_name(
                state.get_resolved_value(field_path)
            )
        except Exception:
            return None

    @staticmethod
    def type_name(value) -> str:
        if value is None:
            return "None"
        return type(value).__name__

    @staticmethod
    def type_qualname(value_type: type) -> str:
        return f"{value_type.__module__}.{value_type.__qualname__}"

    @staticmethod
    def field_name(field_path: str) -> str:
        return field_path.rsplit(".", 1)[-1]

    @staticmethod
    def container_path(field_path: str) -> str:
        if "." not in field_path:
            return ""
        return field_path.rsplit(".", 1)[0]


class ObjectStateFieldHelpProjectionService:
    """Resolve ObjectState field help through the shared UI help introspection path."""

    def describe(
        self,
        request: UiObjectStateFieldHelpRequest,
    ) -> UiObjectStateFieldHelpResult:
        state = self._state_for_agent_scope_id(request.object_state_scope_id)
        if state is None:
            return self._error_result(
                request,
                AgentError(
                    code="ui_object_state_scope_not_found",
                    message=(
                        "ObjectState scope is not registered in the running UI: "
                        f"{request.object_state_scope_id!r}"
                    ),
                ),
            )
        if request.field_path not in state.parameters:
            return self._error_result(
                request,
                AgentError(
                    code="ui_object_state_field_not_found",
                    message=(
                        "ObjectState field is not registered on this scope: "
                        f"{request.field_path!r}"
                    ),
                ),
            )

        field = ObjectStateFieldSemanticProjection.from_state(
            state,
            request.field_path,
        ).to_field_summary()
        try:
            help_target = self._help_target(state, request.field_path)
            parameter_name = ObjectStateScopeProjectionService.field_name(
                request.field_path
            )
            parameter_description = resolved_parameter_description(
                help_target=help_target,
                param_name=parameter_name,
                widget_description=self._widget_description(
                    state,
                    request.field_path,
                ),
            )
            parameter_content = parameter_help_content(
                param_name=parameter_name,
                param_type=self._parameter_type(help_target, parameter_name),
                description=parameter_description,
            )
            description, description_truncated = self._bounded_text(
                parameter_content.description,
                request.max_description_chars,
            )
            return UiObjectStateFieldHelpResult(
                schema_version=SCHEMA_VERSION,
                address=field.address,
                field=field,
                object_type=ObjectStateScopeProjectionService.type_qualname(
                    type(state.object_instance)
                ),
                help_target_type=ObjectStateScopeProjectionService.type_qualname(
                    help_target
                ),
                parameter_name=parameter_name,
                summary=parameter_content.summary,
                description=description,
                description_truncated=description_truncated,
            )
        except Exception as exc:
            return self._error_result(
                request,
                AgentError.from_exception(
                    "ui_object_state_field_help_failed",
                    exc,
                ),
                field=field,
            )

    @staticmethod
    def _state_for_agent_scope_id(scope_id: str) -> ObjectState | None:
        for candidate_scope_id in OpenHCSUiWindowId.manager_scopes_for_agent_window_id(
            scope_id
        ):
            state = ObjectStateRegistry.get_by_scope(candidate_scope_id)
            if state is not None:
                return state
        return None

    @staticmethod
    def _help_target(state: ObjectState, field_path: str) -> type | Callable[..., object]:
        return state.type_for_path(field_path)

    @staticmethod
    def _parameter_type(
        help_target: type | Callable[..., object],
        parameter_name: str,
    ) -> type | None:
        parameter_info = UnifiedParameterAnalyzer.analyze(help_target).get(
            parameter_name
        )
        if parameter_info is None:
            return None
        return parameter_info.param_type

    @staticmethod
    def _widget_description(state: ObjectState, field_path: str) -> str:
        description = state.parameter_descriptions.get(field_path)
        if description is None:
            return ""
        return description

    @staticmethod
    def _bounded_text(value: str | None, max_chars: int) -> tuple[str | None, bool]:
        if value is None:
            return None, False
        bounded_max = max(0, max_chars)
        if len(value) <= bounded_max:
            return value, False
        return (
            value[:bounded_max]
            + f"\n...<truncated {len(value) - bounded_max} chars>",
            True,
        )

    @staticmethod
    def _error_result(
        request: UiObjectStateFieldHelpRequest,
        error: AgentError,
        *,
        field: UiObjectStateFieldSummary | None = None,
    ) -> UiObjectStateFieldHelpResult:
        return UiObjectStateFieldHelpResult(
            schema_version=SCHEMA_VERSION,
            address=UiSemanticAddress(
                object_state_scope_id=request.object_state_scope_id,
                field_path=request.field_path,
                window_id=request.window_id,
            ),
            field=field,
            errors=(error,),
        )


class ObjectStateFieldMutationService:
    """Mutate one ObjectState field through ObjectState's own update/reset API."""

    def mutate(
        self,
        request: UiObjectStateFieldMutationRequest,
    ) -> UiObjectStateFieldMutationResult:
        state = ObjectStateFieldHelpProjectionService._state_for_agent_scope_id(
            request.object_state_scope_id
        )
        if state is None:
            return self._error_result(
                request,
                AgentError(
                    code="ui_object_state_scope_not_found",
                    message=(
                        "ObjectState scope is not registered in the running UI: "
                        f"{request.object_state_scope_id!r}"
                    ),
                ),
            )
        if request.field_path not in state.parameters:
            return self._error_result(
                request,
                AgentError(
                    code="ui_object_state_field_not_found",
                    message=(
                        "ObjectState field is not registered on this scope: "
                        f"{request.field_path!r}"
                    ),
                ),
            )

        before = self._field_summary(state, request)
        try:
            if request.reset:
                state.reset_parameter(request.field_path)
            else:
                state.update_parameter(
                    request.field_path,
                    self._converted_value(state, request),
                )
            after = self._field_summary(state, request)
            return UiObjectStateFieldMutationResult(
                schema_version=SCHEMA_VERSION,
                address=after.address,
                mutated=True,
                reset=request.reset,
                receipt=UiMutationReceipt.accepted_for(request.request_token),
                before=before,
                after=after,
            )
        except Exception as exc:
            return self._error_result(
                request,
                AgentError.from_exception("ui_object_state_field_mutation_failed", exc),
                before=before,
            )

    @staticmethod
    def _field_summary(
        state: ObjectState,
        request: UiObjectStateFieldMutationRequest,
    ) -> UiObjectStateFieldSummary:
        return ObjectStateFieldSemanticProjection.from_state(
            state,
            request.field_path,
        ).to_field_summary(include_values=request.include_field_values)

    @staticmethod
    def _converted_value(
        state: ObjectState,
        request: UiObjectStateFieldMutationRequest,
    ) -> object:
        help_target = state.type_for_path(request.field_path)
        parameter_name = ObjectStateScopeProjectionService.field_name(
            request.field_path
        )
        parameter_info = UnifiedParameterAnalyzer.analyze(help_target).get(
            parameter_name
        )
        if parameter_info is None:
            return request.value
        return ParameterFormService().convert_value_to_type(
            request.value,
            parameter_info.param_type,
            parameter_name,
            help_target,
        )

    @staticmethod
    def _error_result(
        request: UiObjectStateFieldMutationRequest,
        error: AgentError,
        *,
        before: UiObjectStateFieldSummary | None = None,
    ) -> UiObjectStateFieldMutationResult:
        return UiObjectStateFieldMutationResult(
            schema_version=SCHEMA_VERSION,
            address=UiSemanticAddress(
                object_state_scope_id=request.object_state_scope_id,
                field_path=request.field_path,
                window_id=request.window_id,
            ),
            mutated=False,
            reset=request.reset,
            receipt=UiMutationReceipt.rejected_for(request.request_token),
            before=before,
            errors=(error,),
        )


class ObjectStateScopeProvider(UiObjectStateScopeProviderABC):
    """ObjectState scope catalog provider."""

    identity = UiObjectStateScopeProviderIdentity(
        provider_id=OBJECT_STATE_SCOPE_PROVIDER_ID,
        title="ObjectState scopes",
    )

    def __init__(self) -> None:
        self._projection = ObjectStateScopeProjectionService()

    def catalog(
        self,
        request: UiObjectStateScopeListRequest,
    ) -> UiObjectStateScopeCatalog:
        return self._projection.catalog(request)


class WindowCodeDocumentDriverBackedProvider(
    SnapshotBackedUiCodeDocumentProviderABC
):
    """Shared provider flow for WindowCodeDocumentDriver-backed documents."""

    def __init__(self, snapshot_provider: UiBridgeSnapshotProviderABC) -> None:
        self._snapshot_provider = snapshot_provider

    def read(self, request: UiCodeDocumentRequest) -> UiCodeDocument:
        try:
            address = self._address(request.document_id)
            driver = self._driver(address)
            document = driver.read_document(clean=request.clean)
            source_bytes = document.source.encode("utf-8")
            summary = self._summary_for_address(address, document.title)
            return UiCodeDocument(
                schema_version=SCHEMA_VERSION,
                summary=summary,
                source=document.source,
                mime_type=document.mime_type,
                size_bytes=len(source_bytes),
                sha256=hashlib.sha256(source_bytes).hexdigest(),
                current_revision_token=self._snapshot_provider.revision_token(
                    self._revision_key_for_address(address)
                ),
                current_snapshot=self._snapshot_provider.current_snapshot(),
                selection_mode=request.selection_mode,
                selected_scope_ids=self._selected_scope_ids(address),
            )
        except Exception as exc:
            return self._document_error(
                request,
                AgentError.from_exception("ui_code_document_read_failed", exc),
            )

    def validate(
        self,
        request: UiCodeDocumentValidationRequest,
    ) -> UiCodeDocumentValidationResult:
        try:
            address = self._address(request.document_id)
            self._driver(address).validate_source(request.source)
            return UiCodeDocumentValidationResult(
                schema_version=SCHEMA_VERSION,
                document_id=request.document_id,
                valid=True,
                normalized_scope_ids=self._selected_scope_ids(address),
            )
        except Exception as exc:
            return UiCodeDocumentValidationResult(
                schema_version=SCHEMA_VERSION,
                document_id=request.document_id,
                valid=False,
                errors=(AgentError.from_exception("ui_code_document_validation_failed", exc),),
            )

    def apply(self, request: UiCodeDocumentApplyRequest) -> UiCodeDocumentApplyResult:
        guard_error = self._apply_guard_policy(request).first_error()
        if guard_error is not None:
            return self._apply_error(request, guard_error)

        try:
            address = self._address(request.document_id)
            current_revision = self._snapshot_provider.revision_token(
                self._revision_key_for_address(address)
            )
            if request.base_revision_token != current_revision:
                return self._apply_error(
                    request,
                    AgentError(
                        code="stale_revision_token",
                        message="The UI document changed after it was read.",
                    ),
                )

            driver = self._driver(address)
            if self._source_is_current(driver, request.source):
                return self._apply_unchanged(request)
            driver.validate_source(request.source)
            ObjectStateRegistry.ensure_baseline_snapshot()
            pre_snapshot = self._snapshot_provider.current_snapshot()
            pre_head_id = self._snapshot_provider.current_branch_head_snapshot_id()
            label = UiCodeDocumentApplyLabel.resolve(
                request,
                self._provider_identity_for_label(address),
            ).value
            with ObjectStateRegistry.atomic_success(
                label,
                self._mutation_scope_id(address),
            ):
                driver.apply_source(request.source)

            post_head_id = self._snapshot_provider.current_branch_head_snapshot_id()
            if post_head_id == pre_head_id:
                return self._apply_error(
                    request,
                    AgentError(
                        code="snapshot_not_recorded",
                        message=(
                            "Applying the UI document did not record a new "
                            "ObjectState snapshot."
                        ),
                    ),
                )

            post_snapshot = self._snapshot_provider.current_snapshot()
            new_revision_token = self._snapshot_provider.revision_token(
                self._revision_key_for_address(address)
            )
            return UiCodeDocumentApplyResult(
                schema_version=SCHEMA_VERSION,
                document_id=request.document_id,
                applied=True,
                base_revision_token=request.base_revision_token,
                receipt=UiMutationReceipt.accepted_for(request.request_token),
                outcome="applied",
                new_revision_token=new_revision_token,
                current_revision_token=new_revision_token,
                current_snapshot=post_snapshot,
                undo_snapshot=pre_snapshot,
                pre_apply_snapshot=pre_snapshot,
                post_apply_snapshot=post_snapshot,
            )
        except Exception as exc:
            logger.exception(
                "Failed to apply UI code document %s",
                request.document_id,
            )
            return self._apply_error(
                request,
                AgentError.from_exception("ui_code_document_apply_failed", exc),
            )

    @staticmethod
    def _apply_guard_policy(
        request: UiCodeDocumentApplyRequest,
    ) -> UiBridgeGuardPolicy:
        return UiBridgeGuardPolicy(
            rules=(
                CONFIRMATION_REQUIRED_GUARD.bind(
                    lambda: request.confirmation_is_required(),
                ),
                APPLY_TIME_TRAVEL_OPT_IN_GUARD.bind(
                    lambda: (
                        ObjectStateRegistry.is_time_traveling()
                        and not request.apply_if_time_traveling
                    ),
                ),
            )
        )

    @staticmethod
    def _source_is_current(driver, source: str) -> bool:
        return driver.read_document(clean=True).source == source

    def _address(self, document_id: str):
        raise NotImplementedError

    def _driver(self, address):
        raise NotImplementedError

    def _summary_for_address(self, address, title: str) -> UiCodeDocumentSummary:
        raise NotImplementedError

    def _selected_scope_ids(self, address) -> tuple[str, ...]:
        raise NotImplementedError

    def _revision_key_for_address(self, address) -> str:
        raise NotImplementedError

    def _provider_identity_for_label(self, address) -> UiCodeDocumentProviderIdentity:
        raise NotImplementedError

    def _mutation_scope_id(self, address) -> str:
        raise NotImplementedError

    def _document_error(
        self,
        request: UiCodeDocumentRequest,
        error: AgentError,
    ) -> UiCodeDocument:
        source_bytes = b""
        return UiCodeDocument(
            schema_version=SCHEMA_VERSION,
            summary=self.summary(),
            source="",
            mime_type=PYTHON_MIME_TYPE,
            size_bytes=0,
            sha256=hashlib.sha256(source_bytes).hexdigest(),
            current_revision_token=self._snapshot_provider.revision_token(
                self.identity.revision_key
            ),
            current_snapshot=self._snapshot_provider.current_snapshot(),
            selection_mode=request.selection_mode,
            selected_scope_ids=(),
            errors=(error,),
        )


class ObjectStateScopeCodeDocumentProvider(WindowCodeDocumentDriverBackedProvider):
    """Dynamic code-document provider for WindowManager-backed ObjectState scopes."""

    identity = UiCodeDocumentProviderIdentity(
        document_id=OBJECT_STATE_SCOPE_CODE_DOCUMENT_ID,
        widget_id=OBJECT_STATE_SCOPE_CODE_DOCUMENT_ID,
        title="ObjectState scope code documents",
    )

    def handles(self, document_id: str) -> bool:
        return document_id == self.identity.document_id or document_id.startswith(
            OBJECT_STATE_SCOPE_CODE_DOCUMENT_PREFIX
        )

    def summaries(self) -> tuple[UiCodeDocumentSummary, ...]:
        summaries = tuple(
            self._summary_for_scope_id(scope_id)
            for scope_id in self._readable_scope_ids()
        )
        if not summaries:
            return (self.summary(),)
        return summaries

    def revision_key_for_document_id(self, document_id: str) -> str:
        return self._revision_key_for_address(self._address(document_id))

    def summary(self) -> UiCodeDocumentSummary:
        return UiCodeDocumentSummary(
            schema_version=SCHEMA_VERSION,
            identity=self.identity.as_document_identity(),
            widget_id=self.identity.widget_id,
            title=(
                "ObjectState scope code documents "
                f"({OBJECT_STATE_SCOPE_CODE_DOCUMENT_PREFIX}<scope_id>)"
            ),
            readable=False,
            writable=False,
            total_scope_count=len(ObjectStateRegistry.get_all()),
        )

    def _readable_scope_ids(self) -> tuple[str, ...]:
        return tuple(
            agent_object_state_scope_id(state.scope_id) or state.scope_id
            for state in ObjectStateRegistry.get_all()
            if self._has_code_document_driver(state.scope_id)
        )

    def _has_code_document_driver(self, scope_id: str) -> bool:
        if FunctionPatternScopeCodeDocumentDriver.handles_scope(scope_id):
            return True
        code_document_scopes = WindowManager.get_code_document_scopes()
        return any(
            candidate_scope_id in code_document_scopes
            for candidate_scope_id in OpenHCSUiWindowId.manager_scopes_for_agent_window_id(
                agent_object_state_scope_id(scope_id) or scope_id
            )
        )

    def _summary_for_scope_id(self, scope_id: str) -> UiCodeDocumentSummary:
        address = ObjectStateScopeCodeDocumentAddress(scope_id=scope_id)
        title = self._title_for_scope_id(scope_id)
        return self._summary_for_address(address, title)

    def _title_for_scope_id(self, scope_id: str) -> str:
        for candidate_scope_id in OpenHCSUiWindowId.manager_scopes_for_agent_window_id(
            scope_id
        ):
            window = WindowManager.get_window(candidate_scope_id)
            if window is not None:
                return f"Code mode - {window.windowTitle()}"
        return f"Code mode - {scope_id}"

    def _address(self, document_id: str) -> ObjectStateScopeCodeDocumentAddress:
        return ObjectStateScopeCodeDocumentAddress.parse(document_id)

    def _driver(self, address: ObjectStateScopeCodeDocumentAddress):
        scope_id = self._scope_id(address)
        if FunctionPatternScopeCodeDocumentDriver.handles_scope(scope_id):
            return FunctionPatternScopeCodeDocumentDriver(scope_id)
        return WindowManager.require_code_document_driver(scope_id)

    def _scope_id(self, address: ObjectStateScopeCodeDocumentAddress) -> str:
        for scope_id in OpenHCSUiWindowId.manager_scopes_for_agent_window_id(
            address.scope_id
        ):
            if FunctionPatternScopeCodeDocumentDriver.handles_scope(scope_id):
                return scope_id
            try:
                WindowManager.require_code_document_driver(scope_id)
                return scope_id
            except KeyError:
                continue
        for scope_id in OpenHCSUiWindowId.manager_scopes_for_agent_window_id(
            address.scope_id
        ):
            result = ScopeWindowNavigationService.navigate(
                WindowNavigationRequest(
                    scope_id=scope_id,
                    create_if_missing=True,
                )
            )
            if result.window is not None:
                return scope_id
        raise KeyError(
            f"ObjectState scope window is not available: {address.scope_id!r}"
        )

    def _summary_for_address(
        self,
        address: ObjectStateScopeCodeDocumentAddress,
        title: str,
    ) -> UiCodeDocumentSummary:
        identity = address.provider_identity(title)
        return UiCodeDocumentSummary(
            schema_version=SCHEMA_VERSION,
            identity=identity.as_document_identity(),
            widget_id=identity.widget_id,
            title=title,
            readable=True,
            writable=True,
            current_selection_count=1,
            total_scope_count=len(ObjectStateRegistry.get_all()),
        )

    def _selected_scope_ids(
        self,
        address: ObjectStateScopeCodeDocumentAddress,
    ) -> tuple[str, ...]:
        return (address.scope_id,)

    def _revision_key_for_address(
        self,
        address: ObjectStateScopeCodeDocumentAddress,
    ) -> str:
        return address.revision_key

    def _provider_identity_for_label(
        self,
        address: ObjectStateScopeCodeDocumentAddress,
    ) -> UiCodeDocumentProviderIdentity:
        return address.provider_identity(self.identity.title)

    def _mutation_scope_id(self, address: ObjectStateScopeCodeDocumentAddress) -> str:
        return self._scope_id(address)


class WindowManagerCodeDocumentProvider(WindowCodeDocumentDriverBackedProvider):
    """Discover and route live WindowManager code-mode documents."""

    identity = UiCodeDocumentProviderIdentity(
        document_id=WINDOW_CODE_DOCUMENT_ID,
        widget_id=WINDOW_CODE_DOCUMENT_ID,
        title="Open window code-mode documents",
    )

    def handles(self, document_id: str) -> bool:
        return document_id.startswith(WINDOW_CODE_DOCUMENT_PREFIX)

    def revision_key_for_document_id(self, document_id: str) -> str:
        return self._revision_key_for_address(self._address(document_id))

    def summary(self) -> UiCodeDocumentSummary:
        return UiCodeDocumentSummary(
            schema_version=SCHEMA_VERSION,
            identity=self.identity.as_document_identity(),
            widget_id=self.identity.widget_id,
            title=(
                "Open window code-mode documents "
                f"({WINDOW_CODE_DOCUMENT_PREFIX}<window_id>)"
            ),
            readable=False,
            writable=False,
            total_scope_count=len(WindowManager.get_code_document_scopes()),
        )

    def summaries(self) -> tuple[UiCodeDocumentSummary, ...]:
        scopes = WindowManager.get_code_document_scopes()
        if not scopes:
            return (self.summary(),)
        return tuple(
            self._summary_for_open_scope(scope_id)
            for scope_id in scopes
        )

    def _summary_for_open_scope(self, scope_id: str) -> UiCodeDocumentSummary:
        address = WindowCodeDocumentAddress.from_scope_id(scope_id)
        window = WindowManager.get_window(scope_id)
        title = self._open_scope_title(scope_id, address, window)
        return self._summary_for_address(address, title)

    def _open_scope_title(
        self,
        scope_id: str,
        address: WindowCodeDocumentAddress,
        window,
    ) -> str:
        if window is not None and window.windowTitle():
            return f"Code mode - {window.windowTitle()}"
        try:
            document_title = WindowManager.require_code_document_driver(
                scope_id
            ).read_document(clean=True).title
        except Exception:
            document_title = address.window_id
        return f"Code mode - {document_title}"

    def _address(self, document_id: str) -> WindowCodeDocumentAddress:
        return WindowCodeDocumentAddress.parse(document_id)

    def _driver(self, address: WindowCodeDocumentAddress):
        return WindowManager.require_code_document_driver(self._scope_id(address))

    def _scope_id(self, address: WindowCodeDocumentAddress) -> str:
        for scope_id in OpenHCSUiWindowId.manager_scopes_for_agent_window_id(
            address.window_id
        ):
            try:
                WindowManager.require_code_document_driver(scope_id)
                return scope_id
            except KeyError:
                continue
        raise KeyError(
            "Open window has no code-document driver registered: "
            f"{address.window_id!r}"
        )

    def _summary_for_address(
        self,
        address: WindowCodeDocumentAddress,
        title: str,
    ) -> UiCodeDocumentSummary:
        identity = address.provider_identity(title)
        return UiCodeDocumentSummary(
            schema_version=SCHEMA_VERSION,
            identity=identity.as_document_identity(),
            widget_id=identity.widget_id,
            title=title,
            readable=True,
            writable=True,
            current_selection_count=1,
            total_scope_count=len(WindowManager.get_code_document_scopes()),
        )

    def _selected_scope_ids(
        self,
        address: WindowCodeDocumentAddress,
    ) -> tuple[str, ...]:
        return (self._scope_id(address),)

    def _revision_key_for_address(self, address: WindowCodeDocumentAddress) -> str:
        return address.revision_key

    def _provider_identity_for_label(
        self,
        address: WindowCodeDocumentAddress,
    ) -> UiCodeDocumentProviderIdentity:
        return address.provider_identity(self.identity.title)

    def _mutation_scope_id(self, address: WindowCodeDocumentAddress) -> str:
        return self._scope_id(address)


class ObjectStateBridgeProviderSet(UiBridgeProviderSetABC):
    """Provider set for ObjectState registry projections."""

    registry_key = OBJECT_STATE_SCOPE_PROVIDER_ID

    def register(self, context: UiBridgeRegistrationContext) -> None:
        context.registry.register_object_state_scope_provider(ObjectStateScopeProvider())
        context.registry.register_code_document_provider(
            ObjectStateScopeCodeDocumentProvider(context.snapshot_provider)
        )
        context.registry.register_code_document_provider(
            WindowManagerCodeDocumentProvider(context.snapshot_provider)
        )
