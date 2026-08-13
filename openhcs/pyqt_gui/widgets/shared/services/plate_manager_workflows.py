"""Workflow services owned by the plate manager widget."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, ClassVar

from metaclass_registry import AutoRegisterMeta
from objectstate.object_state import ObjectState, ObjectStateRegistry
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.core.orchestrator.orchestrator import OrchestratorState
from openhcs.core.selection import (
    SelectedAllSelectionMode,
    SelectedScopeIdsCarrier,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.ui.shared.plate_scope_identity import PlateScopeIdentity
from openhcs.pyqt_gui.services.plate_manager_root_state import (
    root_orchestrator_scope_ids,
)
from openhcs.pyqt_gui.services.plate_manager_row import PlateManagerRow
from openhcs.pyqt_gui.services.pipeline_object_state_binding import (
    PipelineObjectStateBinding,
)
from pyqt_reactive.widgets.shared.manager_workflows import (
    ManagerCodeExecutionWorkflow,
    ManagerDeletionWorkflow,
)
from openhcs.pyqt_gui.widgets.shared.services.gui_event_bus_broadcast import (
    GuiEventBusBroadcaster,
)
from openhcs.ui.shared.plate_manager_code_document import (
    PlateManagerCodeDocumentAuthority,
    PlateManagerOrchestratorCodePayload,
)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget


@dataclass(frozen=True, kw_only=True, slots=True)
class PlateManagerCodeMutationScope(
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Proof-bearing collection scope attached to a code-document apply."""

    __registry_key__ = "mode"
    __skip_if_no_key__ = True

    mode: ClassVar[SelectedAllSelectionMode | None] = None
    selected_scope_ids: tuple[str, ...] = ()

    @classmethod
    def all(cls) -> "PlateManagerCodeMutationScope":
        return AllPlateManagerCodeMutationScope()

    @classmethod
    def from_carrier(
        cls,
        carrier: SelectedScopeIdsCarrier,
        *,
        default: SelectedAllSelectionMode = SelectedAllSelectionMode.SELECTED,
    ) -> "PlateManagerCodeMutationScope":
        mode = SelectedAllSelectionMode(carrier.resolved_selection_mode(default))
        return cls.__registry__[mode](
            selected_scope_ids=tuple(carrier.selected_scope_ids),
        )

    def require_payload_scope(self, plate_paths: tuple[str, ...]) -> None:
        """Reject selected documents that exceed their read-time authority."""

        if not self.selected_scope_ids:
            return
        if frozenset(plate_paths) != frozenset(self.selected_scope_ids):
            raise ValueError(
                "Selected-scope code documents must preserve their selected plate "
                "scope IDs. Read the document again to change the mutation scope."
            )

    @abstractmethod
    def synchronize(
        self,
        workflow: "PlateManagerCodeWorkflow",
        payload: PlateManagerOrchestratorCodePayload,
    ) -> None:
        """Synchronize visible rows within the declared mutation scope."""


class SelectedPlateManagerCodeMutationScope(PlateManagerCodeMutationScope):
    """Upsert selected rows while preserving every unmentioned plate."""

    mode = SelectedAllSelectionMode.SELECTED

    def synchronize(
        self,
        workflow: "PlateManagerCodeWorkflow",
        payload: PlateManagerOrchestratorCodePayload,
    ) -> None:
        self.require_payload_scope(payload.plate_paths)
        workflow.ensure_plate_entries(list(payload.plate_paths))


class AllPlateManagerCodeMutationScope(PlateManagerCodeMutationScope):
    """Make the complete visible collection match an all-scope document."""

    mode = SelectedAllSelectionMode.ALL

    def synchronize(
        self,
        workflow: "PlateManagerCodeWorkflow",
        payload: PlateManagerOrchestratorCodePayload,
    ) -> None:
        workflow.sync_plate_entries(payload.plate_paths)


@dataclass(frozen=True, slots=True)
class PlateManagerCodeWorkflow(ManagerCodeExecutionWorkflow):
    """Applies edited orchestrator code to plate-manager state."""

    workflow_key = "plate_manager"
    manager: "PlateManagerWidget"
    mutation_scope: PlateManagerCodeMutationScope = field(
        default_factory=PlateManagerCodeMutationScope.all
    )

    def migration_namespace(self, code: str, error: Exception) -> dict | None:
        del code, error
        return None

    def apply_namespace(self, namespace) -> bool:
        payload = PlateManagerCodeDocumentAuthority.from_namespace(namespace)
        self.manager.require_pipeline_definition_mutation_allowed()

        self.mutation_scope.synchronize(self, payload)

        self.apply_global_config(payload.global_pipeline_config)
        self.apply_per_plate_configs(payload.per_plate_configs)
        self.apply_pipeline_data(payload.pipeline_data)
        return True

    def validate_namespace(self, namespace) -> bool:
        PlateManagerCodeDocumentAuthority.from_namespace(namespace)
        return True

    def sync_plate_entries(self, plate_paths: tuple[str, ...]) -> None:
        """Make visible plate rows match the code document's plate path list."""
        requested_paths = tuple(dict.fromkeys(str(path) for path in plate_paths))
        root_state = self.manager._ensure_root_state()
        current_paths = tuple(root_orchestrator_scope_ids(root_state))
        requested_path_set = set(requested_paths)
        removed_paths = tuple(
            path for path in current_paths if path not in requested_path_set
        )

        if removed_paths:
            removed_rows = [
                PlateManagerRow.from_scope(
                    path,
                    cppipe_path=self.manager._cppipe_path_for_scope_id(path),
                )
                for path in removed_paths
            ]
            if not self.manager.deletion_workflow.validate(removed_rows):
                raise ValueError("Cannot remove plates while execution is in progress.")
            self.manager.deletion_workflow.delete(removed_rows)

        self.ensure_plate_entries(list(requested_paths))
        selection_changed = self.reconcile_selection(requested_paths)
        self.manager.update_item_list()
        if selection_changed:
            self.manager.plate_selected.emit(self.manager.selected_plate_path)

    def ensure_plate_entries(self, plate_paths: list[str]) -> None:
        if not plate_paths:
            return

        root_state = self.manager._ensure_root_state()
        current_paths = root_orchestrator_scope_ids(root_state)
        existing_paths = set(current_paths)
        new_paths = list(current_paths)
        added_count = 0

        for plate_path in plate_paths:
            plate_str = str(plate_path)
            if plate_str in existing_paths:
                continue
            identity = PlateScopeIdentity.from_scope_id(plate_str)
            self.manager._create_orchestrator_for_plate(
                plate_str,
                plate_root=identity.plate_root,
                cppipe_path=identity.cppipe_path,
            )
            plate_name = identity.display_name
            new_paths.append(plate_str)
            existing_paths.add(plate_str)
            added_count += 1
            logger.info("Added plate '%s' from orchestrator code", plate_name)

        if not added_count:
            return

        with ObjectStateRegistry.atomic("register orchestrators"):
            root_state.update_parameter("orchestrator_scope_ids", new_paths)

        if self.manager.item_list:
            self.manager.update_item_list()
        status_message = f"Added {added_count} plate(s) from orchestrator code"
        self.manager.status_message.emit(status_message)
        logger.info(status_message)

    def reconcile_selection(self, requested_paths: tuple[str, ...]) -> bool:
        """Align semantic selection with the applied plate document."""
        current_selection = self.manager.selected_plate_path
        if requested_paths:
            if current_selection in requested_paths:
                return False
            self.manager.selected_plate_path = requested_paths[0]
            return True

        if current_selection:
            self.manager.selected_plate_path = ""
            return True
        return False

    def apply_global_config(self, global_config: GlobalPipelineConfig) -> None:
        self.manager.require_pipeline_definition_mutation_allowed()
        self.manager.global_config = global_config

        global_state = ObjectStateRegistry.get_by_scope("")
        if global_state:
            global_state.update_object_instance(global_config)

        for row in self.manager.plates:
            orchestrator = ObjectStateRegistry.get_object(row.scope_id)
            if orchestrator:
                self.manager._update_orchestrator_global_config(
                    row.scope_id,
                    orchestrator,
                    global_config,
                )

        self.manager.service_adapter.set_global_config(global_config)
        self.manager.global_config_changed.emit()
        GuiEventBusBroadcaster(self.manager.event_bus).config_changed(global_config)

    def apply_per_plate_configs(
        self,
        per_plate_configs: dict[str, PipelineConfig],
    ) -> None:
        self.manager.require_pipeline_definition_mutation_allowed()
        last_pipeline_config = None
        for plate_path, pipeline_config in per_plate_configs.items():
            plate_key = str(plate_path)
            self.manager.plate_configs[plate_key] = pipeline_config

            orchestrator = ObjectStateRegistry.get_object(plate_key)
            if orchestrator:
                orchestrator.apply_pipeline_config(pipeline_config)
                self._update_pipeline_config_delegate_state(
                    plate_key,
                    pipeline_config,
                )
                effective_config = orchestrator.get_effective_config()
                self.manager.orchestrator_config_changed.emit(
                    str(orchestrator.plate_path),
                    effective_config,
                )
                logger.debug(
                    "Applied per-plate pipeline config to orchestrator: %s",
                    orchestrator.plate_path,
                )
            else:
                logger.info(
                    "Stored pipeline config for %s; will apply when initialized.",
                    plate_key,
                )
            last_pipeline_config = pipeline_config

        if last_pipeline_config:
            GuiEventBusBroadcaster(self.manager.event_bus).config_changed(
                last_pipeline_config
            )

    @staticmethod
    def _pipeline_config_delegate_state(plate_key: str) -> ObjectState:
        state = ObjectStateRegistry.get_by_scope(plate_key)
        if state is None:
            raise RuntimeError(
                "Per-plate PipelineConfig update requires an existing orchestrator "
                f"ObjectState for scope {plate_key!r}."
            )
        if not state.has_delegate:
            raise RuntimeError(
                "Per-plate PipelineConfig update requires an orchestrator "
                "ObjectState delegated to pipeline_config; got "
                f"{type(state.object_instance).__name__} at scope {plate_key!r}."
            )
        if not isinstance(state.saved_object, PipelineConfig):
            raise RuntimeError(
                "Per-plate PipelineConfig update requires a PipelineConfig delegate; "
                f"got {type(state.saved_object).__name__} at scope {plate_key!r}."
            )
        return state

    @classmethod
    def _update_pipeline_config_delegate_state(
        cls,
        plate_key: str,
        pipeline_config: PipelineConfig,
    ) -> None:
        cls._pipeline_config_delegate_state(plate_key).update_object_instance(
            pipeline_config
        )

    def apply_pipeline_data(self, pipeline_data: dict[str, list[FunctionStep]]) -> None:
        self.manager.require_pipeline_definition_mutation_allowed()
        for plate_path, submitted_steps in pipeline_data.items():
            pipeline_steps = list(submitted_steps)
            PipelineObjectStateBinding.update_plate_steps(plate_path, pipeline_steps)
            logger.debug(
                "Updated pipeline for %s with %d steps",
                plate_path,
                len(pipeline_steps),
            )
            self.invalidate_orchestrator_compilation_state(plate_path)
            if plate_path == self.manager.selected_plate_path:
                GuiEventBusBroadcaster(self.manager.event_bus).pipeline_changed(
                    pipeline_steps
                )
                self.manager.status_message.emit(
                    f"Loaded {len(pipeline_steps)} steps from plate-manager code document"
                )

        self.manager.pipeline_data_changed.emit()

    def invalidate_orchestrator_compilation_state(self, plate_path: str) -> None:
        if plate_path in self.manager.plate_compiled_data:
            self.manager.emit_compiled_state(plate_path, None)
            logger.debug("Cleared compiled data for %s", plate_path)
        self.manager.clear_plate_execution_tracking(plate_path)

        orchestrator = ObjectStateRegistry.get_object(plate_path)
        if (
            orchestrator
            and orchestrator.state.has_completed_initialization
            and orchestrator.state is not OrchestratorState.READY
        ):
            orchestrator._state = OrchestratorState.READY
            self.manager.orchestrator_state_changed.emit(
                plate_path,
                OrchestratorState.READY,
            )


@dataclass(frozen=True, slots=True)
class PlateManagerDeletionWorkflow(ManagerDeletionWorkflow):
    """Validates and deletes plates plus their registered ObjectState scopes."""

    workflow_key = "plate_manager"
    manager: "PlateManagerWidget"

    def validate(self, items: list[PlateManagerRow]) -> bool:
        if not self.manager.is_any_plate_running():
            return True
        self.manager.service_adapter.show_error_dialog(
            "Cannot delete plates while execution is in progress.\n"
            "Please stop execution first."
        )
        return False

    def delete(self, items: list[PlateManagerRow]) -> None:
        paths_to_delete = {row.scope_id for row in items}
        plate_names = [row.name for row in items]
        label = f"delete plate{'s' if len(items) > 1 else ''} {', '.join(plate_names)}"

        with ObjectStateRegistry.atomic(label):
            root_state = self.manager._ensure_root_state()
            current_paths = root_orchestrator_scope_ids(root_state)
            remaining_paths = [
                path for path in current_paths if path not in paths_to_delete
            ]
            root_state.update_parameter(
                "orchestrator_scope_ids",
                remaining_paths,
            )

            for path in paths_to_delete:
                path_str = str(path)
                count = ObjectStateRegistry.unregister_scope_and_descendants(path_str)
                logger.debug(
                    "Cascade unregistered %d ObjectState(s) for deleted plate: %s",
                    count,
                    path,
                )

                if path_str in self.manager.plate_configs:
                    del self.manager.plate_configs[path_str]
                    logger.debug("Deleted plate_configs entry for: %s", path)

        next_selection = self._selection_after_delete(
            previous_selection=self.manager.selected_plate_path,
            deleted_paths=paths_to_delete,
            remaining_paths=remaining_paths,
        )
        if next_selection is None:
            return
        self.manager.selected_plate_path = next_selection
        self.manager.plate_selected.emit(next_selection)

    def _selection_after_delete(
        self,
        *,
        previous_selection: str,
        deleted_paths: set[str],
        remaining_paths: list[str],
    ) -> str | None:
        if previous_selection and previous_selection not in deleted_paths:
            if previous_selection in remaining_paths:
                return None
        if remaining_paths:
            return remaining_paths[0]
        return ""
