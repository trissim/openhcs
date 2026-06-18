"""ObjectState scope projection provider for the PyQt UI bridge."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import ClassVar

from pyqt_reactive.services.scope_window_navigation import ScopeWindowNavigationService
from pyqt_reactive.services.window_code_document import PYTHON_MIME_TYPE
from pyqt_reactive.services.window_manager import WindowManager
from pyqt_reactive.services.window_navigation import WindowNavigationRequest

from openhcs.agent.dto.common import AgentError, SCHEMA_VERSION
from openhcs.agent.dto.ui_bridge import (
    UiCodeDocument,
    UiCodeDocumentApplyRequest,
    UiCodeDocumentApplyResult,
    UiCodeDocumentRequest,
    UiCodeDocumentSummary,
    UiCodeDocumentValidationRequest,
    UiCodeDocumentValidationResult,
    UiCatalogPageMetadata,
    UiObjectStateFieldProvenance,
    UiObjectStateFieldSummary,
    UiObjectStateScopeCatalog,
    UiObjectStateScopeIdentity,
    UiObjectStateScopeListRequest,
    UiObjectStateScopeSummary,
    UiSemanticAddress,
    UiTimeTravelRuntimeState,
)
from openhcs.config_framework import ObjectState, ObjectStateRegistry
from openhcs.pyqt_gui.services.ui_bridge_contracts import (
    APPLY_TIME_TRAVEL_OPT_IN_GUARD,
    CONFIRMATION_REQUIRED_GUARD,
    UiBridgeGuardPolicy,
    UiBridgeSnapshotProviderABC,
    UiCodeDocumentProviderABC,
    UiCodeDocumentProviderIdentity,
    UiObjectStateScopeProviderABC,
    UiObjectStateScopeProviderIdentity,
)
from openhcs.pyqt_gui.services.ui_agent_bridge import UiCodeDocumentApplyLabel
from openhcs.pyqt_gui.services.ui_bridge_object_state_scope_policy import (
    ObjectStateScopeVisibility,
)
from openhcs.pyqt_gui.services.ui_bridge_registry import (
    UiBridgeProviderSetABC,
    UiBridgeRegistrationContext,
)


OBJECT_STATE_SCOPE_PROVIDER_ID = "object_state.scopes"
OBJECT_STATE_SCOPE_CODE_DOCUMENT_ID = "object_state_scope"
OBJECT_STATE_SCOPE_CODE_DOCUMENT_PREFIX = f"{OBJECT_STATE_SCOPE_CODE_DOCUMENT_ID}:"
OBJECT_STATE_FIELD_PAGE_LIMIT_MAX = 1000


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
            source_scope_id=self.source_scope_id,
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
            time_travel_state=UiTimeTravelRuntimeState(
                active=ObjectStateRegistry.is_time_traveling()
            ),
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
                object_state_scope_id=state.scope_id,
            ),
            object_type=type(state.object_instance).__name__,
            parameter_count=len(state.parameters),
            dirty_field_count=len(state.dirty_fields),
            signature_diff_field_count=len(state.signature_diff_fields),
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

        all_field_paths = tuple(sorted(state.parameters.keys()))
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
            self._field_summary(state, field_path)
            for field_path in selected_paths
        )
        return fields, page

    def _field_summary(
        self,
        state: ObjectState,
        field_path: str,
    ) -> UiObjectStateFieldSummary:
        return UiObjectStateFieldSummary(
            schema_version=SCHEMA_VERSION,
            address=UiSemanticAddress(
                object_state_scope_id=state.scope_id,
                field_path=field_path,
                window_id=state.scope_id,
            ),
            field_name=self.field_name(field_path),
            container_path=self.container_path(field_path),
            raw_value_type=self.type_name(state.parameters[field_path]),
            resolved_value_type=self._resolved_type_name(state, field_path),
            dirty=field_path in state.dirty_fields,
            signature_diff=field_path in state.signature_diff_fields,
            last_changed=field_path == state.last_changed_field,
            provenance=ObjectStateFieldProvenanceEffect.from_state(
                state,
                field_path,
            ).to_dto(),
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


class ObjectStateScopeCodeDocumentProvider(UiCodeDocumentProviderABC):
    """Dynamic code-document provider for WindowManager-backed ObjectState scopes."""

    identity = UiCodeDocumentProviderIdentity(
        document_id=OBJECT_STATE_SCOPE_CODE_DOCUMENT_ID,
        widget_id=OBJECT_STATE_SCOPE_CODE_DOCUMENT_ID,
        title="ObjectState scope code documents",
    )

    def __init__(self, snapshot_provider: UiBridgeSnapshotProviderABC) -> None:
        self._snapshot_provider = snapshot_provider

    def handles(self, document_id: str) -> bool:
        return document_id == self.identity.document_id or document_id.startswith(
            OBJECT_STATE_SCOPE_CODE_DOCUMENT_PREFIX
        )

    def summary(self) -> UiCodeDocumentSummary:
        return UiCodeDocumentSummary(
            schema_version=SCHEMA_VERSION,
            identity=self.identity.as_document_identity(),
            widget_id=self.identity.widget_id,
            title=(
                "ObjectState scope code documents "
                f"({OBJECT_STATE_SCOPE_CODE_DOCUMENT_PREFIX}<scope_id>)"
            ),
            readable=True,
            writable=True,
            total_scope_count=len(ObjectStateRegistry.get_all()),
        )

    def read(self, request: UiCodeDocumentRequest) -> UiCodeDocument:
        try:
            address = ObjectStateScopeCodeDocumentAddress.parse(request.document_id)
            driver = self._driver(address)
            document = driver.read_document()
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
                    address.revision_key
                ),
                current_snapshot=self._snapshot_provider.current_snapshot(),
                selection_mode=request.selection_mode,
                selected_scope_ids=(address.scope_id,),
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
            address = ObjectStateScopeCodeDocumentAddress.parse(request.document_id)
            self._driver(address).validate_source(request.source)
            return UiCodeDocumentValidationResult(
                schema_version=SCHEMA_VERSION,
                document_id=request.document_id,
                valid=True,
                normalized_scope_ids=(address.scope_id,),
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
            address = ObjectStateScopeCodeDocumentAddress.parse(request.document_id)
            current_revision = self._snapshot_provider.revision_token(address.revision_key)
            if request.base_revision_token != current_revision:
                return self._apply_error(
                    request,
                    AgentError(
                        code="stale_revision_token",
                        message="The UI document changed after it was read.",
                    ),
                )

            driver = self._driver(address)
            driver.validate_source(request.source)
            pre_snapshot = self._snapshot_provider.current_snapshot()
            pre_head_id = self._snapshot_provider.current_branch_head_snapshot_id()
            ObjectStateRegistry.ensure_baseline_snapshot()
            label = UiCodeDocumentApplyLabel.resolve(
                request,
                address.provider_identity(self.identity.title),
            ).value
            with ObjectStateRegistry.atomic_success(label, address.scope_id):
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

            return UiCodeDocumentApplyResult(
                schema_version=SCHEMA_VERSION,
                document_id=request.document_id,
                applied=True,
                base_revision_token=request.base_revision_token,
                outcome="applied",
                new_revision_token=self._snapshot_provider.revision_token(
                    address.revision_key
                ),
                pre_apply_snapshot=pre_snapshot,
                post_apply_snapshot=self._snapshot_provider.current_snapshot(),
            )
        except Exception as exc:
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

    def _driver(self, address: ObjectStateScopeCodeDocumentAddress):
        result = ScopeWindowNavigationService.navigate(
            WindowNavigationRequest(
                scope_id=address.scope_id,
                create_if_missing=True,
            )
        )
        if result.window is None:
            raise KeyError(f"ObjectState scope window is not available: {address.scope_id!r}")
        return WindowManager.require_code_document_driver(address.scope_id)

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

    @staticmethod
    def _apply_error(
        request: UiCodeDocumentApplyRequest,
        error: AgentError,
    ) -> UiCodeDocumentApplyResult:
        return UiCodeDocumentApplyResult(
            schema_version=SCHEMA_VERSION,
            document_id=request.document_id,
            applied=False,
            base_revision_token=request.base_revision_token,
            errors=(error,),
        )


class ObjectStateBridgeProviderSet(UiBridgeProviderSetABC):
    """Provider set for ObjectState registry projections."""

    registry_key = OBJECT_STATE_SCOPE_PROVIDER_ID

    def register(self, context: UiBridgeRegistrationContext) -> None:
        context.registry.register_object_state_scope_provider(ObjectStateScopeProvider())
        context.registry.register_code_document_provider(
            ObjectStateScopeCodeDocumentProvider(context.snapshot_provider)
        )
