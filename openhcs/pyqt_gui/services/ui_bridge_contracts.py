"""Cycle-free contracts for PyQt UI bridge provider registration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Self

from openhcs.agent.dto.common import AgentError, SCHEMA_VERSION
from openhcs.agent.dto.ui_bridge import (
    UiCodeDocument,
    UiCodeDocumentApplyRequest,
    UiCodeDocumentApplyResult,
    UiCodeDocumentIdentity,
    UiCodeDocumentRequest,
    UiCodeDocumentSummary,
    UiCodeDocumentValidationRequest,
    UiCodeDocumentValidationResult,
    UiMutationReceipt,
    UiObjectStateScopeCatalog,
    UiObjectStateScopeListRequest,
    UiActionCatalog,
    UiActionInvokeRequest,
    UiActionInvokeResult,
    UiActionSummary,
    UiActionIdentity,
    UiStateSurfaceDocument,
    UiStateSurfaceIdentity,
    UiStateSurfaceRequest,
    UiStateSurfaceSummary,
    UiSnapshotRef,
    UiWidgetIdentity,
    UiWidgetActionInvokeRequest,
    UiWidgetActionInvokeResult,
    UiWindowCatalog,
    UiWindowCloseRequest,
    UiWindowCloseResult,
    UiWindowFocusRequest,
    UiWindowFocusResult,
    UiWindowNavigateRequest,
    UiWindowNavigateResult,
    UiWindowSnapshotRequest,
    UiWindowSnapshotResult,
    UiWidgetTreeRequest,
    UiWidgetTreeResult,
)
from openhcs.agent.ui_bridge_identities import (
    UiCodeDocumentIdentityDeclaration,
    UiStateSurfaceIdentityDeclarationBase,
    UiWidgetIdentityDeclaration,
)


class UiBridgeRegistryKeyMixin:
    """Shared AutoRegisterMeta registry-key declaration for UI bridge families."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True

    registry_key: ClassVar[str | None] = None


class UiCodeDocumentProviderABC(ABC):
    """Provider contract for one UI-owned code document."""

    identity: "UiCodeDocumentProviderIdentity"

    def handles(self, document_id: str) -> bool:
        """Return whether this provider owns a document id."""
        return self.identity.document_id == document_id

    @abstractmethod
    def summary(self) -> UiCodeDocumentSummary:
        raise NotImplementedError

    def summaries(self) -> tuple[UiCodeDocumentSummary, ...]:
        """Return catalog summaries owned by this provider."""
        return (self.summary(),)

    @abstractmethod
    def read(self, request: UiCodeDocumentRequest) -> UiCodeDocument:
        raise NotImplementedError

    @abstractmethod
    def validate(
        self,
        request: UiCodeDocumentValidationRequest,
    ) -> UiCodeDocumentValidationResult:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        request: UiCodeDocumentApplyRequest,
    ) -> UiCodeDocumentApplyResult:
        raise NotImplementedError


class UiStateSurfaceProviderABC(ABC):
    """Provider contract for one pollable UI state surface."""

    identity: "UiStateSurfaceProviderIdentity"

    @abstractmethod
    def summary(self) -> UiStateSurfaceSummary:
        raise NotImplementedError

    @abstractmethod
    def read(self, request: UiStateSurfaceRequest) -> UiStateSurfaceDocument:
        raise NotImplementedError


class UiActionProviderABC(ABC):
    """Provider contract for one widget/domain action catalog."""

    identity: "UiActionProviderIdentity"

    @abstractmethod
    def catalog(self) -> UiActionCatalog:
        raise NotImplementedError

    @abstractmethod
    def summary(self, action_id: str) -> UiActionSummary:
        raise NotImplementedError

    @abstractmethod
    def invoke(self, request: UiActionInvokeRequest) -> UiActionInvokeResult:
        raise NotImplementedError


class UiWindowProviderABC(ABC):
    """Provider contract for a catalog of focusable UI windows."""

    identity: "UiWindowProviderIdentity"

    @abstractmethod
    def catalog(self) -> UiWindowCatalog:
        raise NotImplementedError

    @abstractmethod
    def handles(self, window_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def focus(self, request: UiWindowFocusRequest) -> UiWindowFocusResult:
        raise NotImplementedError

    @abstractmethod
    def navigate(self, request: UiWindowNavigateRequest) -> UiWindowNavigateResult:
        raise NotImplementedError

    @abstractmethod
    def close(self, request: UiWindowCloseRequest) -> UiWindowCloseResult:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self, request: UiWindowSnapshotRequest) -> UiWindowSnapshotResult:
        raise NotImplementedError

    @abstractmethod
    def widget_tree(self, request: UiWidgetTreeRequest) -> UiWidgetTreeResult:
        raise NotImplementedError

    @abstractmethod
    def invoke_widget_action(
        self,
        request: UiWidgetActionInvokeRequest,
    ) -> UiWidgetActionInvokeResult:
        raise NotImplementedError


class UiObjectStateScopeProviderABC(ABC):
    """Provider contract for ObjectState registry scope projection."""

    identity: "UiObjectStateScopeProviderIdentity"

    @abstractmethod
    def catalog(
        self,
        request: UiObjectStateScopeListRequest,
    ) -> UiObjectStateScopeCatalog:
        raise NotImplementedError


class UiBridgeSnapshotProviderABC(ABC):
    """Snapshot projection behavior required by UI bridge providers."""

    @abstractmethod
    def current_snapshot(self) -> UiSnapshotRef | None:
        raise NotImplementedError

    @abstractmethod
    def current_branch_head_snapshot_id(self) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def revision_token(self, document_id: str) -> str:
        raise NotImplementedError


class SnapshotBackedUiCodeDocumentProviderABC(UiCodeDocumentProviderABC):
    """Shared apply rejection behavior for snapshot-backed code documents."""

    _snapshot_provider: UiBridgeSnapshotProviderABC

    def revision_key_for_document_id(self, document_id: str) -> str:
        del document_id
        return self.identity.revision_key

    def _apply_error(
        self,
        request: UiCodeDocumentApplyRequest,
        error: AgentError,
    ) -> UiCodeDocumentApplyResult:
        return self._apply_errors(request, (error,))

    def _apply_errors(
        self,
        request: UiCodeDocumentApplyRequest,
        errors: tuple[AgentError, ...],
    ) -> UiCodeDocumentApplyResult:
        return UiCodeDocumentApplyResult(
            schema_version=SCHEMA_VERSION,
            document_id=request.document_id,
            applied=False,
            base_revision_token=request.base_revision_token,
            receipt=UiMutationReceipt.rejected_for(request.request_token),
            current_revision_token=self._current_revision_token_for_apply_result(
                request
            ),
            current_snapshot=self._current_snapshot_for_apply_result(),
            errors=errors,
        )

    def _apply_unchanged(
        self,
        request: UiCodeDocumentApplyRequest,
    ) -> UiCodeDocumentApplyResult:
        current_revision = self._current_revision_token_for_apply_result(request)
        return UiCodeDocumentApplyResult(
            schema_version=SCHEMA_VERSION,
            document_id=request.document_id,
            applied=False,
            base_revision_token=request.base_revision_token,
            receipt=UiMutationReceipt.accepted_for(request.request_token),
            outcome="unchanged",
            new_revision_token=current_revision,
            current_revision_token=current_revision,
            current_snapshot=self._current_snapshot_for_apply_result(),
        )

    def _current_revision_token_for_apply_result(
        self,
        request: UiCodeDocumentApplyRequest,
    ) -> str | None:
        try:
            return self._snapshot_provider.revision_token(
                self.revision_key_for_document_id(request.document_id)
            )
        except Exception:
            return None

    def _current_snapshot_for_apply_result(self) -> UiSnapshotRef | None:
        try:
            return self._snapshot_provider.current_snapshot()
        except Exception:
            return None


@dataclass(frozen=True, slots=True)
class UiCodeDocumentProviderIdentity(UiCodeDocumentIdentity, UiWidgetIdentity):
    """Stable identity and derived names for one UI code document."""

    title: str

    @classmethod
    def from_declaration(
        cls,
        declaration: type[UiCodeDocumentIdentityDeclaration],
        *,
        title: str,
    ) -> Self:
        return cls(
            document_id=declaration.require_value(),
            widget_id=declaration.widget_id(),
            title=title,
        )

    @property
    def revision_key(self) -> str:
        return f"ui-code-document:{self.document_id}"

    @property
    def default_edit_label(self) -> str:
        return f"edit {self.document_id} via MCP"


@dataclass(frozen=True, slots=True)
class UiStateSurfaceProviderIdentity(UiStateSurfaceIdentity, UiWidgetIdentity):
    """Stable identity and derived names for one UI state surface."""

    title: str

    @classmethod
    def from_declaration(
        cls,
        declaration: type[UiStateSurfaceIdentityDeclarationBase],
        *,
        title: str,
    ) -> Self:
        return cls(
            surface_id=declaration.require_value(),
            widget_id=declaration.widget_id(),
            title=title,
        )

    @property
    def revision_key(self) -> str:
        return f"ui-state-surface:{self.surface_id}"


@dataclass(frozen=True, slots=True)
class UiActionProviderIdentity(UiActionIdentity):
    """Stable identity and derived names for one UI action provider."""

    title: str

    @classmethod
    def from_widget_declaration(
        cls,
        declaration: type[UiWidgetIdentityDeclaration],
        *,
        title: str,
        action_id: str | None = None,
    ) -> Self:
        return cls(
            widget_id=declaration.require_value(),
            action_id=(
                declaration.action_provider_id()
                if action_id is None
                else action_id
            ),
            title=title,
        )

    @property
    def revision_key(self) -> str:
        return f"ui-action:{self.widget_id}:{self.action_id}"


@dataclass(frozen=True, slots=True)
class UiBridgeProviderIdentity:
    """Stable identity shared by dynamic UI bridge provider catalogs."""

    provider_id: str
    title: str


@dataclass(frozen=True, slots=True)
class UiWindowProviderIdentity(UiBridgeProviderIdentity):
    """Stable identity for one dynamic UI window catalog provider."""

    @property
    def revision_key(self) -> str:
        return f"ui-window-provider:{self.provider_id}"


@dataclass(frozen=True, slots=True)
class UiObjectStateScopeProviderIdentity(UiBridgeProviderIdentity):
    """Stable identity for one ObjectState scope catalog provider."""

    @property
    def revision_key(self) -> str:
        return f"ui-object-state-provider:{self.provider_id}"


@dataclass(frozen=True, slots=True)
class UiBridgeGuardRule:
    """One fail-loud precondition for a mutating bridge request."""

    code: str
    message: str
    predicate: Callable[[], bool]

    def error_if_violated(self) -> AgentError | None:
        if self.predicate():
            return AgentError(code=self.code, message=self.message)
        return None


@dataclass(frozen=True, slots=True)
class UiBridgeGuardTemplate:
    """Declared guard identity that binds to a runtime predicate."""

    code: str
    message: str

    def bind(self, predicate: Callable[[], bool]) -> UiBridgeGuardRule:
        return UiBridgeGuardRule(
            code=self.code,
            message=self.message,
            predicate=predicate,
        )


@dataclass(frozen=True, slots=True)
class UiBridgeGuardPolicy:
    """Ordered guard authority for bridge mutation preconditions."""

    rules: tuple[UiBridgeGuardRule, ...]

    def first_error(self) -> AgentError | None:
        for rule in self.rules:
            error = rule.error_if_violated()
            if error is not None:
                return error
        return None


CONFIRMATION_REQUIRED_GUARD = UiBridgeGuardTemplate(
    code="confirmation_required",
    message="UI confirmation is required.",
)
RESTORE_TIME_TRAVEL_OPT_IN_GUARD = UiBridgeGuardTemplate(
    code="time_travel_branch_switch_requires_confirmation",
    message="Restoring while time-traveled requires allow_auto_branch=True.",
)
APPLY_TIME_TRAVEL_OPT_IN_GUARD = UiBridgeGuardTemplate(
    code="stale_time_travel_state",
    message="Applying while time-traveled requires apply_if_time_traveling=True.",
)
