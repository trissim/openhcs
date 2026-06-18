"""In-process PyQt implementation of the OpenHCS UI agent bridge."""

from __future__ import annotations

import ast
import datetime as _datetime
import hashlib
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta
from openhcs.agent.dto.common import AgentError, SCHEMA_VERSION
from openhcs.agent.dto.ui_bridge import (
    UNKNOWN_UI_BRIDGE_OPERATION_ROUTE,
    UiActionCatalog,
    UiActionIdentity,
    UiActionInvocationStatus,
    UiActionInvokeRequest,
    UiActionInvokeResult,
    UiBranchCatalog,
    UiBranchRef,
    UiBranchSwitchRequest,
    UiBridgeConnectionSpec,
    UiBridgeOperationIdentity,
    UiBridgeOperationRef,
    UiBridgeOperationRoute,
    UiBridgeOperationStatus,
    UiBridgeStatus,
    UiCodeDocument,
    UiCodeDocumentApplyRequest,
    UiCodeDocumentApplyResult,
    UiCodeDocumentCatalog,
    UiCodeDocumentIdentity,
    UiCodeDocumentRequest,
    UiCodeDocumentSelectionMode,
    UiCodeDocumentSummary,
    UiCodeDocumentValidationRequest,
    UiCodeDocumentValidationResult,
    UiMutationReceipt,
    UiObjectStateScopeCatalog,
    UiObjectStateScopeListRequest,
    UiObjectStateScopeVisibility,
    UiStateSurfaceCatalog,
    UiStateSurfaceDocument,
    UiStateSurfaceRequest,
    UiStateSurfaceSummary,
    UiSnapshotCatalog,
    UiSnapshotListRequest,
    UiSnapshotRef,
    UiSnapshotRestoreRequest,
    UiSnapshotRestoreResult,
    UiTimeTravelHeadRequest,
    UiTimeTravelRuntimeState,
    UiWindowCatalog,
    UiWindowFocusRequest,
    UiWindowFocusResult,
    UiWindowNavigateRequest,
    UiWindowNavigateResult,
    UiWindowSnapshotRequest,
    UiWindowSnapshotResult,
)
from openhcs.agent.services.ui_bridge_service import UiBridgeGatewayABC
from openhcs.config_framework.object_state import ObjectStateRegistry
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.pyqt_gui.services.ui_bridge_contracts import (
    CONFIRMATION_REQUIRED_GUARD,
    RESTORE_TIME_TRAVEL_OPT_IN_GUARD,
    UiBridgeGuardPolicy,
    UiBridgeSnapshotProviderABC,
    UiActionProviderABC,
    UiCodeDocumentProviderABC,
    UiObjectStateScopeProviderABC,
    UiStateSurfaceProviderABC,
    UiWindowProviderABC,
)
from openhcs.pyqt_gui.services.ui_bridge_object_state_scope_policy import (
    ObjectStateScopeVisibility,
)
from openhcs.pyqt_gui.services.ui_bridge_registry import (
    UiBridgeRegistrationContext,
    UiBridgeSurfaceRegistry,
)
from openhcs.pyqt_gui.services.ui_thread_dispatch import UiThreadDispatcher
from openhcs.pyqt_gui.widgets.shared.services.plate_manager_workflows import (
    PlateManagerOrchestratorCodePayload,
)


@dataclass(frozen=True, slots=True)
class CodeDocumentExecutionResult:
    """Parsed and normalized payload from one declarative UI code document."""

    payload: PlateManagerOrchestratorCodePayload

    @property
    def mutation_scope(self) -> str | None:
        if len(self.payload.plate_paths) != 1:
            return None
        return self.payload.plate_paths[0]

    def apply_namespace(self) -> dict:
        return self.payload.to_namespace()


class UiCodeDocumentSourcePolicy:
    """Validate that MCP-submitted source is declarative code-mode data."""

    allowed_import_roots = frozenset(("openhcs",))
    expected_assignments = frozenset(
        (
            "plate_paths",
            "global_config",
            "per_plate_configs",
            "pipeline_data",
        )
    )

    def validate(self, source: str) -> tuple[AgentError, ...]:
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return (AgentError.from_exception("invalid_python_source", exc),)

        visitor = DeclarativeCodeDocumentAstValidator(
            allowed_import_roots=self.allowed_import_roots,
            expected_assignments=self.expected_assignments,
        )
        visitor.visit(tree)
        return tuple(visitor.errors)


class DeclarativeCodeDocumentAstValidator(ast.NodeVisitor):
    """AST validator for the generated plate-manager code document shape."""

    def __init__(
        self,
        *,
        allowed_import_roots: frozenset[str],
        expected_assignments: frozenset[str],
    ) -> None:
        self._allowed_import_roots = allowed_import_roots
        self._expected_assignments = expected_assignments
        self._imported_names: set[str] = set()
        self.errors: list[AgentError] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root_name = alias.name.split(".", maxsplit=1)[0]
            if root_name not in self._allowed_import_roots:
                self._error("unsafe_import", f"Import is not allowed: {alias.name}")
                continue
            self._imported_names.add(alias.asname or root_name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            self._error("unsafe_import", "Relative imports are not allowed.")
            return
        module_name = node.module
        root_name = module_name.split(".", maxsplit=1)[0]
        if root_name not in self._allowed_import_roots:
            self._error("unsafe_import", f"Import is not allowed: {module_name}")
            return
        for alias in node.names:
            if alias.name == "*":
                self._error("unsafe_import", "Wildcard imports are not allowed.")
                continue
            self._imported_names.add(alias.asname or alias.name)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if not isinstance(target, ast.Name):
                self._error("unsafe_assignment", "Only named assignments are allowed.")
                continue
            if target.id not in self._expected_assignments:
                self._error(
                    "unexpected_assignment",
                    f"Unexpected assignment target: {target.id}",
                )
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._error("unsafe_assignment", "Annotated assignments are not allowed.")

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._error("unsafe_assignment", "Augmented assignments are not allowed.")

    def visit_Expr(self, node: ast.Expr) -> None:
        self._error("unsafe_statement", "Standalone expressions are not allowed.")

    def visit_Call(self, node: ast.Call) -> None:
        if not self._is_approved_constructor_call(node.func):
            self._error("unsafe_call", "Only approved constructor calls are allowed.")
            return
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            return
        if node.id in self._imported_names:
            return
        self._error("unknown_name", f"Name is not imported by the document: {node.id}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.visit(node.value)

    def visit_List(self, node: ast.List) -> None:
        self._visit_sequence(node.elts)

    visit_Tuple = visit_List
    visit_Set = visit_List

    def visit_Dict(self, node: ast.Dict) -> None:
        for key, value in zip(node.keys, node.values):
            if key is not None:
                self.visit(key)
            self.visit(value)

    def visit_Constant(self, node: ast.Constant) -> None:
        return None

    def generic_visit(self, node: ast.AST) -> None:
        safe_nodes = (
            ast.Module,
            ast.Load,
            ast.keyword,
            ast.UnaryOp,
            ast.USub,
            ast.UAdd,
        )
        if isinstance(node, safe_nodes):
            super().generic_visit(node)
            return
        self._error(
            "unsafe_statement",
            f"Unsupported source construct: {type(node).__name__}",
        )

    def _visit_sequence(self, values: list[ast.expr]) -> None:
        for value in values:
            self.visit(value)

    def _is_approved_constructor_call(self, func: ast.expr) -> bool:
        if isinstance(func, ast.Name):
            return func.id in self._imported_names and func.id[:1].isupper()
        return False

    def _error(self, code: str, message: str) -> None:
        self.errors.append(AgentError(code=code, message=message))


class UiCodeDocumentExecutionService:
    """Executes validated code-mode source through existing manager hooks."""

    def __init__(self, source_policy: UiCodeDocumentSourcePolicy | None = None) -> None:
        self._source_policy = source_policy or UiCodeDocumentSourcePolicy()

    def validate_source(self, source: str, operations) -> CodeDocumentExecutionResult:
        errors = self._source_policy.validate(source)
        if errors:
            raise UiCodeDocumentValidationError(errors)

        namespace: dict[str, object] = {}
        try:
            with operations.patch_lazy_constructors():
                exec(source, namespace)
        except TypeError as exc:
            migrated_namespace = operations.migrate_code_namespace(
                source,
                exc,
                namespace,
            )
            if migrated_namespace is None:
                raise
            namespace = migrated_namespace

        payload = PlateManagerOrchestratorCodePayload.from_namespace(namespace)
        if payload is None:
            raise UiCodeDocumentValidationError(
                (
                    AgentError(
                        code="missing_code_document_payload",
                        message="Code document did not define plate_paths and pipeline_data.",
                    ),
                )
            )

        normalized_pipeline_data = {
            plate_path: FunctionStepTransportAuthority.normalize_pipeline(pipeline_steps)
            for plate_path, pipeline_steps in payload.pipeline_data.items()
        }
        normalized_payload = replace(payload, pipeline_data=normalized_pipeline_data)
        return CodeDocumentExecutionResult(payload=normalized_payload)


@dataclass(frozen=True, slots=True)
class UiCodeDocumentValidationError(ValueError):
    errors: tuple[AgentError, ...]

    def __str__(self) -> str:
        return "; ".join(error.message for error in self.errors)


class UiBridgeBusyError(RuntimeError):
    """Raised when a mutating UI bridge operation is already running."""


class UiBridgeMutationGate:
    """Serialize mutating bridge operations for one running UI."""

    def __init__(self, tracker: "UiBridgeOperationTracker") -> None:
        self._lock = threading.Lock()
        self._tracker = tracker

    def run(
        self,
        *,
        operation_name: str,
        target_id: str | None,
        callback: Callable[[UiBridgeOperationRef], UiCodeDocumentApplyResult | UiSnapshotRestoreResult],
    ):
        if not self._lock.acquire(blocking=False):
            raise UiBridgeBusyError("A mutating UI bridge operation is already running.")
        operation = self._tracker.start(operation_name, target_id)
        try:
            result = callback(operation)
            self._tracker.complete(
                operation.identity.operation_id,
                status=UiBridgeOperationStatus.COMPLETED,
                outcome=UiBridgeMutationOutcome.from_result(result),
            )
            return result
        except Exception as exc:
            self._tracker.complete(
                operation.identity.operation_id,
                status=UiBridgeOperationStatus.FAILED,
                outcome="error",
                errors=(AgentError.from_exception("ui_bridge_operation_failed", exc),),
            )
            raise
        finally:
            self._lock.release()


class UiBridgeOperationTracker:
    """In-memory operation status projection for bridge calls."""

    def __init__(self) -> None:
        self._operations: dict[str, UiBridgeOperationRef] = {}
        self._lock = threading.Lock()

    def start(self, operation_name: str, target_id: str | None) -> UiBridgeOperationRef:
        operation_id = str(uuid.uuid4())
        operation = UiBridgeOperationRef(
            schema_version=SCHEMA_VERSION,
            identity=UiBridgeOperationIdentity(
                operation_id=operation_id,
                route=UiBridgeOperationRoute(
                    operation_name=operation_name,
                    target_id=target_id,
                ),
            ),
            status=UiBridgeOperationStatus.RUNNING.value,
            started_at_unix=time.time(),
        )
        with self._lock:
            self._operations[operation_id] = operation
        return operation

    def complete(
        self,
        operation_id: str,
        *,
        status: UiBridgeOperationStatus,
        outcome: str | None,
        errors: tuple[AgentError, ...] = (),
    ) -> UiBridgeOperationRef:
        with self._lock:
            operation = self._operations[operation_id]
            completed = replace(
                operation,
                status=status.value,
                completed_at_unix=time.time(),
                outcome=outcome,
                errors=errors,
            )
            self._operations[operation_id] = completed
        return completed

    def get(self, operation_id: str) -> UiBridgeOperationRef:
        with self._lock:
            if operation_id in self._operations:
                return self._operations[operation_id]
        return UiBridgeOperationRef(
            schema_version=SCHEMA_VERSION,
            identity=UiBridgeOperationIdentity(
                operation_id=operation_id,
                route=UNKNOWN_UI_BRIDGE_OPERATION_ROUTE,
            ),
            status=UiBridgeOperationStatus.NOT_FOUND.value,
            started_at_unix=0.0,
            completed_at_unix=0.0,
            outcome="not_found",
            errors=(
                AgentError(
                    code="unknown_ui_bridge_operation",
                    message=f"Unknown UI bridge operation: {operation_id}",
                ),
            ),
        )


class UiBridgeMutationOutcome:
    """Explicit outcome projection for bridge mutation result DTOs."""

    @staticmethod
    def from_result(
        result: UiActionInvokeResult | UiCodeDocumentApplyResult | UiSnapshotRestoreResult,
    ) -> str:
        return UiBridgeMutationOutcomeProjector.for_result_type(type(result)).outcome(result)


class UiBridgeMutationOutcomeProjector(ABC, metaclass=AutoRegisterMeta):
    """Nominal outcome projection for one bridge mutation result DTO family."""

    __registry_key__ = "result_type"
    __skip_if_no_key__ = True
    result_type: ClassVar[type | None] = None

    @classmethod
    def for_result_type(
        cls,
        result_type: type,
    ) -> "UiBridgeMutationOutcomeProjector":
        return cls.__registry__[result_type]()

    @abstractmethod
    def outcome(
        self,
        result: UiActionInvokeResult | UiCodeDocumentApplyResult | UiSnapshotRestoreResult,
    ) -> str:
        raise NotImplementedError


class UiActionInvokeOutcomeProjector(UiBridgeMutationOutcomeProjector):
    """Outcome projection for dispatched widget actions."""

    result_type = UiActionInvokeResult

    def outcome(
        self,
        result: UiActionInvokeResult | UiCodeDocumentApplyResult | UiSnapshotRestoreResult,
    ) -> str:
        return result.status


class UiCodeDocumentApplyOutcomeProjector(UiBridgeMutationOutcomeProjector):
    """Outcome projection for code-document apply mutations."""

    result_type = UiCodeDocumentApplyResult

    def outcome(
        self,
        result: UiActionInvokeResult | UiCodeDocumentApplyResult | UiSnapshotRestoreResult,
    ) -> str:
        return result.outcome


class UiSnapshotRestoreOutcomeProjector(UiBridgeMutationOutcomeProjector):
    """Outcome projection for ObjectState restore mutations."""

    result_type = UiSnapshotRestoreResult

    def outcome(
        self,
        result: UiActionInvokeResult | UiCodeDocumentApplyResult | UiSnapshotRestoreResult,
    ) -> str:
        if result.restored:
            return "restored"
        return "not_restored"


class UiObjectStateSnapshotProvider(UiBridgeSnapshotProviderABC):
    """Typed projection and restore API for ObjectState history."""

    def list_snapshots(
        self,
        request: UiSnapshotListRequest,
    ) -> UiSnapshotCatalog:
        return self.catalog(request)

    def catalog(
        self,
        visibility_request: UiObjectStateScopeVisibility | None = None,
    ) -> UiSnapshotCatalog:
        visibility = ObjectStateScopeVisibility(
            visibility_request or UiObjectStateScopeVisibility()
        )
        history = ObjectStateRegistry.get_branch_history()
        current_branch = ObjectStateRegistry.get_current_branch()
        current_index = ObjectStateRegistry.get_current_snapshot_index()
        refs = tuple(
            self._snapshot_ref(snapshot, index, history, visibility)
            for index, snapshot in enumerate(history)
            if visibility.includes_system_scopes()
            or self._visible_state_count(snapshot) > 0
        )
        return UiSnapshotCatalog(
            schema_version=SCHEMA_VERSION,
            current_branch=current_branch,
            current_snapshot_index=current_index,
            object_state_token=ObjectStateRegistry.get_token(),
            time_travel_state=UiTimeTravelRuntimeState(
                active=ObjectStateRegistry.is_time_traveling()
            ),
            snapshots=refs,
            branches=self.branch_refs(),
        )

    def branch_refs(self) -> tuple[UiBranchRef, ...]:
        branches = []
        for branch in ObjectStateRegistry.list_branches():
            branches.append(
                UiBranchRef(
                    schema_version=SCHEMA_VERSION,
                    name=str(branch["name"]),
                    head_snapshot_id=str(branch["head_id"]),
                    base_snapshot_id=str(branch["base_id"]),
                    description=str(branch["description"]),
                    is_current=bool(branch["is_current"]),
                )
            )
        return tuple(branches)

    def current_snapshot(self) -> UiSnapshotRef | None:
        history = ObjectStateRegistry.get_branch_history()
        if not history:
            return None
        current_index = ObjectStateRegistry.get_current_snapshot_index()
        if current_index == -1:
            current_index = len(history) - 1
        return self._snapshot_ref(
            history[current_index],
            current_index,
            history,
            ObjectStateScopeVisibility(
                UiObjectStateScopeVisibility(include_system_scopes=True)
            ),
        )

    def current_branch_head_snapshot_id(self) -> str | None:
        history = ObjectStateRegistry.get_branch_history()
        if not history:
            return None
        return history[-1].id

    def revision_token(self, document_id: str) -> str:
        time_travel_state = UiTimeTravelRuntimeState(
            active=ObjectStateRegistry.is_time_traveling()
        )
        parts = (
            document_id,
            str(ObjectStateRegistry.get_token()),
            ObjectStateRegistry.get_current_branch(),
            str(self.current_branch_head_snapshot_id()),
            str(ObjectStateRegistry.get_current_snapshot_index()),
            str(time_travel_state.active),
        )
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    def restore_snapshot(
        self,
        request: UiSnapshotRestoreRequest,
    ) -> UiSnapshotRestoreResult:
        guard_error = self._restore_guard_policy(request).first_error()
        if guard_error is not None:
            return self._restore_error(guard_error)

        target = self._target_snapshot(request)
        if target is None:
            return self._restore_error(
                AgentError(
                    code="snapshot_not_found",
                    message="Snapshot target was not found.",
                )
            )

        if request.snapshot_id is not None:
            restored = ObjectStateRegistry.time_travel_to_snapshot(request.snapshot_id)
        elif request.index is not None:
            restored = ObjectStateRegistry.time_travel_to(request.index)
        elif request.branch is not None:
            restored = self._restore_branch_head(request.branch)
        else:
            restored = False

        return UiSnapshotRestoreResult(
            schema_version=SCHEMA_VERSION,
            restored=restored,
            target_snapshot=target,
            current_snapshot=self.current_snapshot(),
            catalog=self.catalog(request),
        )

    def time_travel_head(
        self,
        request: UiTimeTravelHeadRequest,
    ) -> UiSnapshotRestoreResult:
        if request.confirmation_is_required():
            return self._restore_error(
                AgentError(
                    code="confirmation_required",
                    message="UI confirmation is required.",
                )
            )
        target = self.current_snapshot()
        restored = ObjectStateRegistry.time_travel_to_head()
        return UiSnapshotRestoreResult(
            schema_version=SCHEMA_VERSION,
            restored=restored,
            target_snapshot=target,
            current_snapshot=self.current_snapshot(),
            catalog=self.catalog(),
        )

    def switch_branch(
        self,
        request: UiBranchSwitchRequest,
    ) -> UiSnapshotRestoreResult:
        return self.restore_snapshot(request.as_restore_request())

    def _target_snapshot(
        self,
        request: UiSnapshotRestoreRequest,
    ) -> UiSnapshotRef | None:
        if request.snapshot_id is not None:
            return self._ref_for_snapshot_id(request.snapshot_id)
        if request.index is not None:
            history = ObjectStateRegistry.get_branch_history()
            if request.index < 0:
                index = len(history) + request.index
            else:
                index = request.index
            if 0 <= index < len(history):
                return self._snapshot_ref(
                    history[index],
                    index,
                    history,
                    ObjectStateScopeVisibility(
                        UiObjectStateScopeVisibility(include_system_scopes=True)
                    ),
                )
            return None
        if request.branch is not None:
            branches = {branch.name: branch for branch in self.branch_refs()}
            if request.branch not in branches:
                return None
            return self._ref_for_snapshot_id(branches[request.branch].head_snapshot_id)
        return None

    def _restore_branch_head(self, branch: str) -> bool:
        known_branches = {ref.name for ref in self.branch_refs()}
        if branch not in known_branches:
            return False
        return ObjectStateRegistry.switch_branch(branch)

    def _ref_for_snapshot_id(self, snapshot_id: str) -> UiSnapshotRef | None:
        history = ObjectStateRegistry.get_branch_history()
        for index, snapshot in enumerate(history):
            if snapshot.id == snapshot_id:
                return self._snapshot_ref(
                    snapshot,
                    index,
                    history,
                    ObjectStateScopeVisibility(
                        UiObjectStateScopeVisibility(include_system_scopes=True)
                    ),
                )
        return None

    def _snapshot_ref(
        self,
        snapshot,
        index: int,
        history,
        visibility: ObjectStateScopeVisibility,
    ) -> UiSnapshotRef:
        current_index = ObjectStateRegistry.get_current_snapshot_index()
        head_index = len(history) - 1
        is_head = index == head_index
        is_current = (current_index == -1 and is_head) or current_index == index
        return UiSnapshotRef(
            schema_version=SCHEMA_VERSION,
            snapshot_id=snapshot.id,
            index=index,
            branch=ObjectStateRegistry.get_current_branch(),
            parent_snapshot_id=snapshot.parent_id,
            timestamp_unix=snapshot.timestamp,
            timestamp=_datetime.datetime.fromtimestamp(
                snapshot.timestamp
            ).isoformat(timespec="milliseconds"),
            label=snapshot.label or f"Snapshot #{index}",
            num_states=(
                len(snapshot.all_states)
                if visibility.includes_system_scopes()
                else self._visible_state_count(snapshot)
            ),
            is_current=is_current,
            is_head=is_head,
            uri=f"openhcs://ui/snapshots/{snapshot.id}",
        )

    @staticmethod
    def _visible_state_count(snapshot) -> int:
        visibility = ObjectStateScopeVisibility()
        return sum(
            1
            for scope_id in snapshot.all_states
            if visibility.includes_scope_id(scope_id)
        )

    @staticmethod
    def _restore_guard_policy(
        request: UiSnapshotRestoreRequest,
    ) -> UiBridgeGuardPolicy:
        return UiBridgeGuardPolicy(
            rules=(
                CONFIRMATION_REQUIRED_GUARD.bind(
                    lambda: request.confirmation_is_required(),
                ),
                RESTORE_TIME_TRAVEL_OPT_IN_GUARD.bind(
                    lambda: (
                        ObjectStateRegistry.is_time_traveling()
                        and not request.allow_auto_branch
                    ),
                ),
            )
        )

    @staticmethod
    def _restore_error(error: AgentError) -> UiSnapshotRestoreResult:
        return UiSnapshotRestoreResult(
            schema_version=SCHEMA_VERSION,
            restored=False,
            target_snapshot=None,
            current_snapshot=None,
            catalog=None,
            errors=(error,),
        )


@dataclass(frozen=True, slots=True)
class UiCodeDocumentApplyLabel:
    """Formal defaulting authority for apply snapshot labels."""

    value: str

    @classmethod
    def resolve(
        cls,
        request: UiCodeDocumentApplyRequest,
        identity: UiCodeDocumentIdentity,
    ) -> "UiCodeDocumentApplyLabel":
        if request.snapshot_label is not None:
            return cls(request.snapshot_label)
        return cls(identity.default_edit_label)


class UiAgentBridgeService:
    """In-process UI bridge owned by the running PyQt main window."""

    def __init__(
        self,
        *,
        plate_manager=None,
        registry: UiBridgeSurfaceRegistry | None = None,
        dispatcher: UiThreadDispatcher | None = None,
        snapshot_provider: UiObjectStateSnapshotProvider | None = None,
        operation_tracker: UiBridgeOperationTracker | None = None,
    ) -> None:
        self._dispatcher = dispatcher or UiThreadDispatcher()
        self._snapshot_provider = snapshot_provider or UiObjectStateSnapshotProvider()
        self._operation_tracker = operation_tracker or UiBridgeOperationTracker()
        self._mutation_gate = UiBridgeMutationGate(self._operation_tracker)
        self._registry = registry or UiBridgeSurfaceRegistry()
        if plate_manager is not None:
            from openhcs.pyqt_gui.services.ui_bridge_plate_manager import (
                PlateManagerBridgeProviderSet,
            )

            PlateManagerBridgeProviderSet(plate_manager).register(
                UiBridgeRegistrationContext(
                    registry=self._registry,
                    snapshot_provider=self._snapshot_provider,
                )
            )

    def register_provider(self, provider: UiCodeDocumentProviderABC) -> None:
        self._registry.register_code_document_provider(provider)

    def register_state_provider(self, provider: UiStateSurfaceProviderABC) -> None:
        self._registry.register_state_surface_provider(provider)

    def register_action_provider(self, provider: UiActionProviderABC) -> None:
        self._registry.register_action_provider(provider)

    def register_window_provider(self, provider: UiWindowProviderABC) -> None:
        self._registry.register_window_provider(provider)

    def register_object_state_scope_provider(
        self,
        provider: UiObjectStateScopeProviderABC,
    ) -> None:
        self._registry.register_object_state_scope_provider(provider)

    def status(self) -> UiBridgeStatus:
        return UiBridgeStatus(
            schema_version=SCHEMA_VERSION,
            reachable=True,
            auth_required=False,
        )

    def list_documents(self) -> UiCodeDocumentCatalog:
        return self._dispatcher.call(
            lambda: UiCodeDocumentCatalog(
                schema_version=SCHEMA_VERSION,
                documents=tuple(
                    provider.summary()
                    for provider in self._registry.code_document_providers()
                ),
            )
        )

    def list_state_surfaces(self) -> UiStateSurfaceCatalog:
        return self._dispatcher.call(
            lambda: UiStateSurfaceCatalog(
                schema_version=SCHEMA_VERSION,
                surfaces=tuple(
                    provider.summary()
                    for provider in self._registry.state_surface_providers()
                ),
            )
        )

    def list_actions(self) -> UiActionCatalog:
        def catalog() -> UiActionCatalog:
            actions = []
            errors = []
            warnings = []
            for provider in self._registry.action_providers():
                provider_catalog = provider.catalog()
                actions.extend(provider_catalog.actions)
                errors.extend(provider_catalog.errors)
                warnings.extend(provider_catalog.warnings)
            return UiActionCatalog(
                schema_version=SCHEMA_VERSION,
                actions=tuple(actions),
                errors=tuple(errors),
                warnings=tuple(warnings),
            )

        return self._dispatcher.call(catalog)

    def list_windows(self) -> UiWindowCatalog:
        def catalog() -> UiWindowCatalog:
            windows = []
            errors = []
            warnings = []
            for provider in self._registry.window_providers():
                provider_catalog = provider.catalog()
                windows.extend(provider_catalog.windows)
                errors.extend(provider_catalog.errors)
                warnings.extend(provider_catalog.warnings)
            return UiWindowCatalog(
                schema_version=SCHEMA_VERSION,
                windows=tuple(windows),
                errors=tuple(errors),
                warnings=tuple(warnings),
            )

        return self._dispatcher.call(catalog)

    def list_object_state_scopes(
        self,
        request: UiObjectStateScopeListRequest,
    ) -> UiObjectStateScopeCatalog:
        def catalog() -> UiObjectStateScopeCatalog:
            scopes = []
            errors = []
            warnings = []
            for provider in self._registry.object_state_scope_providers():
                provider_catalog = provider.catalog(request)
                scopes.extend(provider_catalog.scopes)
                errors.extend(provider_catalog.errors)
                warnings.extend(provider_catalog.warnings)
            return UiObjectStateScopeCatalog(
                schema_version=SCHEMA_VERSION,
                object_state_token=ObjectStateRegistry.get_token(),
                current_branch=ObjectStateRegistry.get_current_branch(),
                current_snapshot_index=ObjectStateRegistry.get_current_snapshot_index(),
                time_travel_state=UiTimeTravelRuntimeState(
                    active=ObjectStateRegistry.is_time_traveling()
                ),
                scopes=tuple(scopes),
                errors=tuple(errors),
                warnings=tuple(warnings),
            )

        return self._dispatcher.call(catalog)

    def get_document(self, request: UiCodeDocumentRequest) -> UiCodeDocument:
        return self._dispatcher.call(lambda: self._provider(request.document_id).read(request))

    def get_state_surface(self, request: UiStateSurfaceRequest) -> UiStateSurfaceDocument:
        return self._dispatcher.call(
            lambda: self._state_provider(request.surface_id).read(request)
        )

    def invoke_action(self, request: UiActionInvokeRequest) -> UiActionInvokeResult:
        def run(operation: UiBridgeOperationRef) -> UiActionInvokeResult:
            result = self._action_provider(request.widget_id).invoke(request)
            return replace(
                result,
                receipt=replace(
                    result.receipt,
                    bridge_operation_id=operation.identity.operation_id,
                ),
            )

        try:
            return self._dispatcher.call(
                lambda: self._mutation_gate.run(
                    operation_name="invoke_action",
                    target_id=request.action_id,
                    callback=run,
                )
            )
        except UiBridgeBusyError as exc:
            return self._action_error(
                request,
                AgentError.from_exception("ui_bridge_busy", exc),
            )

    def focus_window(self, request: UiWindowFocusRequest) -> UiWindowFocusResult:
        return self._dispatcher.call(
            lambda: self._window_provider(request.window_id).focus(request)
        )

    def navigate_window(
        self,
        request: UiWindowNavigateRequest,
    ) -> UiWindowNavigateResult:
        return self._dispatcher.call(
            lambda: self._window_provider(request.window_id).navigate(request)
        )

    def snapshot_window(
        self,
        request: UiWindowSnapshotRequest,
    ) -> UiWindowSnapshotResult:
        return self._dispatcher.call(
            lambda: self._window_provider(request.window_id).snapshot(request)
        )

    def validate_document(
        self,
        request: UiCodeDocumentValidationRequest,
    ) -> UiCodeDocumentValidationResult:
        return self._dispatcher.call(
            lambda: self._provider(request.document_id).validate(request)
        )

    def apply_document(
        self,
        request: UiCodeDocumentApplyRequest,
    ) -> UiCodeDocumentApplyResult:
        def run(operation: UiBridgeOperationRef) -> UiCodeDocumentApplyResult:
            result = self._provider(request.document_id).apply(request)
            return replace(result, operation_id=operation.identity.operation_id)

        try:
            return self._dispatcher.call(
                lambda: self._mutation_gate.run(
                    operation_name="apply_code_document",
                    target_id=request.document_id,
                    callback=run,
                )
            )
        except UiBridgeBusyError as exc:
            return self._apply_document_error(
                request,
                AgentError.from_exception("ui_bridge_busy", exc),
            )

    def list_snapshots(self, request: UiSnapshotListRequest) -> UiSnapshotCatalog:
        return self._dispatcher.call(lambda: self._snapshot_provider.list_snapshots(request))

    def restore_snapshot(
        self,
        request: UiSnapshotRestoreRequest,
    ) -> UiSnapshotRestoreResult:
        return self._dispatcher.call(
            lambda: self._mutation_gate.run(
                operation_name="restore_snapshot",
                target_id=request.snapshot_id,
                callback=lambda operation: self._snapshot_provider.restore_snapshot(
                    request
                ),
            )
        )

    def time_travel_head(
        self,
        request: UiTimeTravelHeadRequest,
    ) -> UiSnapshotRestoreResult:
        return self._dispatcher.call(
            lambda: self._mutation_gate.run(
                operation_name="time_travel_head",
                target_id=None,
                callback=lambda operation: self._snapshot_provider.time_travel_head(
                    request
                ),
            )
        )

    def list_branches(self) -> UiBranchCatalog:
        return self._dispatcher.call(
            lambda: UiBranchCatalog(
                schema_version=SCHEMA_VERSION,
                current_branch=ObjectStateRegistry.get_current_branch(),
                branches=self._snapshot_provider.branch_refs(),
            )
        )

    def switch_branch(self, request: UiBranchSwitchRequest) -> UiSnapshotRestoreResult:
        return self._dispatcher.call(
            lambda: self._mutation_gate.run(
                operation_name="switch_branch",
                target_id=request.branch,
                callback=lambda operation: self._snapshot_provider.switch_branch(request),
            )
        )

    def get_operation_status(self, operation_id: str) -> UiBridgeOperationRef:
        return self._operation_tracker.get(operation_id)

    def _provider(self, document_id: str) -> UiCodeDocumentProviderABC:
        return self._registry.code_document_provider(document_id)

    def _state_provider(self, surface_id: str) -> UiStateSurfaceProviderABC:
        return self._registry.state_surface_provider(surface_id)

    def _action_provider(self, widget_id: str) -> UiActionProviderABC:
        return self._registry.action_provider(widget_id)

    def _window_provider(self, window_id: str) -> UiWindowProviderABC:
        return self._registry.window_provider(window_id)

    @staticmethod
    def _apply_document_error(
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

    @staticmethod
    def _action_error(
        request: UiActionInvokeRequest,
        error: AgentError,
    ) -> UiActionInvokeResult:
        return UiActionInvokeResult(
            schema_version=SCHEMA_VERSION,
            identity=UiActionIdentity(
                widget_id=request.widget_id,
                action_id=request.action_id,
            ),
            status=UiActionInvocationStatus.REJECTED.value,
            receipt=UiMutationReceipt(
                request_token=request.request_token,
                accepted=False,
            ),
            errors=(error,),
        )


class InProcessUiBridgeGateway(UiBridgeGatewayABC):
    """Gateway adapter for tests or same-process MCP embedding."""

    registry_key = "in_process"

    def __init__(self, bridge: UiAgentBridgeService) -> None:
        self._bridge = bridge

    def status(self, connection: UiBridgeConnectionSpec) -> UiBridgeStatus:
        del connection
        return self._bridge.status()

    def list_documents(self, connection: UiBridgeConnectionSpec) -> UiCodeDocumentCatalog:
        del connection
        return self._bridge.list_documents()

    def list_state_surfaces(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> UiStateSurfaceCatalog:
        del connection
        return self._bridge.list_state_surfaces()

    def list_actions(self, connection: UiBridgeConnectionSpec) -> UiActionCatalog:
        del connection
        return self._bridge.list_actions()

    def list_windows(self, connection: UiBridgeConnectionSpec) -> UiWindowCatalog:
        del connection
        return self._bridge.list_windows()

    def list_object_state_scopes(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiObjectStateScopeListRequest,
    ) -> UiObjectStateScopeCatalog:
        del connection
        return self._bridge.list_object_state_scopes(request)

    def get_document(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiCodeDocumentRequest,
    ) -> UiCodeDocument:
        del connection
        return self._bridge.get_document(request)

    def get_state_surface(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiStateSurfaceRequest,
    ) -> UiStateSurfaceDocument:
        del connection
        return self._bridge.get_state_surface(request)

    def invoke_action(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiActionInvokeRequest,
    ) -> UiActionInvokeResult:
        del connection
        return self._bridge.invoke_action(request)

    def focus_window(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWindowFocusRequest,
    ) -> UiWindowFocusResult:
        del connection
        return self._bridge.focus_window(request)

    def navigate_window(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWindowNavigateRequest,
    ) -> UiWindowNavigateResult:
        del connection
        return self._bridge.navigate_window(request)

    def snapshot_window(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiWindowSnapshotRequest,
    ) -> UiWindowSnapshotResult:
        del connection
        return self._bridge.snapshot_window(request)

    def validate_document(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiCodeDocumentValidationRequest,
    ) -> UiCodeDocumentValidationResult:
        del connection
        return self._bridge.validate_document(request)

    def apply_document(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiCodeDocumentApplyRequest,
    ) -> UiCodeDocumentApplyResult:
        del connection
        return self._bridge.apply_document(request)

    def list_snapshots(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiSnapshotListRequest,
    ) -> UiSnapshotCatalog:
        del connection
        return self._bridge.list_snapshots(request)

    def restore_snapshot(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiSnapshotRestoreRequest,
    ) -> UiSnapshotRestoreResult:
        del connection
        return self._bridge.restore_snapshot(request)

    def time_travel_head(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiTimeTravelHeadRequest,
    ) -> UiSnapshotRestoreResult:
        del connection
        return self._bridge.time_travel_head(request)

    def list_branches(self, connection: UiBridgeConnectionSpec) -> UiBranchCatalog:
        del connection
        return self._bridge.list_branches()

    def switch_branch(
        self,
        connection: UiBridgeConnectionSpec,
        request: UiBranchSwitchRequest,
    ) -> UiSnapshotRestoreResult:
        del connection
        return self._bridge.switch_branch(request)

    def get_operation_status(
        self,
        connection: UiBridgeConnectionSpec,
        operation_id: str,
    ) -> UiBridgeOperationRef:
        del connection
        return self._bridge.get_operation_status(operation_id)
