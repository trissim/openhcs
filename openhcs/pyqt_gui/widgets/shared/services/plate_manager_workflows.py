"""Workflow services owned by the plate manager widget."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from openhcs.config_framework.object_state import ObjectStateRegistry
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.core.orchestrator.orchestrator import OrchestratorState
from openhcs.core.steps.function_step import FunctionStep
from openhcs.interop.cellprofiler.runtime.generated_pipeline import (
    CellProfilerGeneratedStepFunctionSpec,
    CellProfilerPipelineRuntimeRebinder,
)
from openhcs.pyqt_gui.services.plate_scope_identity import PlateScopeIdentity
from openhcs.pyqt_gui.services.plate_manager_root_state import (
    root_orchestrator_scope_ids,
)
from openhcs.pyqt_gui.services.plate_manager_row import PlateManagerRow
from pyqt_reactive.widgets.shared.manager_workflows import (
    ManagerCodeExecutionWorkflow,
    ManagerDeletionWorkflow,
)
from openhcs.pyqt_gui.widgets.shared.services.gui_event_bus_broadcast import (
    GuiEventBusBroadcaster,
)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget


class PlateManagerCodeNamespaceField(str, Enum):
    """Field authority for executed plate-manager code namespaces."""

    GLOBAL_CONFIG = "global_config"
    PER_PLATE_CONFIGS = "per_plate_configs"
    PIPELINE_DATA = "pipeline_data"
    PLATE_PATHS = "plate_paths"


class RemovedPlateManagerCodeNamespaceField(str, Enum):
    """Removed plate-manager code fields with explicit migration messages."""

    PIPELINE_CONFIG = "pipeline_config"

    @classmethod
    def reject_present_fields(cls, namespace: dict) -> None:
        for field in cls:
            if field.value in namespace:
                raise ValueError(field.error_message())

    def error_message(self) -> str:
        if self is RemovedPlateManagerCodeNamespaceField.PIPELINE_CONFIG:
            return (
                f"{self.value} is not a plate-manager code document field; "
                "use per_plate_configs keyed by plate path."
            )
        raise RuntimeError(f"Unhandled removed plate-manager field: {self.value}.")


@dataclass(frozen=True, slots=True)
class PlateManagerOrchestratorCodePayload:
    """Authoritative payload for plate-manager orchestrator code documents."""

    plate_paths: tuple[str, ...]
    pipeline_data: dict[str, list[FunctionStep]]
    global_pipeline_config: GlobalPipelineConfig | None = None
    per_plate_configs: dict[str, PipelineConfig] | None = None

    @classmethod
    def from_namespace(
        cls,
        namespace: dict,
    ) -> "PlateManagerOrchestratorCodePayload | None":
        RemovedPlateManagerCodeNamespaceField.reject_present_fields(namespace)

        plate_paths_field = PlateManagerCodeNamespaceField.PLATE_PATHS.value
        pipeline_data_field = PlateManagerCodeNamespaceField.PIPELINE_DATA.value
        if plate_paths_field not in namespace or pipeline_data_field not in namespace:
            return None

        plate_paths = tuple(str(path) for path in namespace[plate_paths_field])
        pipeline_data = {
            str(plate_path): list(pipeline_steps)
            for plate_path, pipeline_steps in namespace[pipeline_data_field].items()
        }
        global_config = namespace.get(
            PlateManagerCodeNamespaceField.GLOBAL_CONFIG.value
        )
        per_plate_configs = namespace.get(
            PlateManagerCodeNamespaceField.PER_PLATE_CONFIGS.value
        )

        if global_config is not None and not isinstance(
            global_config, GlobalPipelineConfig
        ):
            raise TypeError("global_config must be a GlobalPipelineConfig.")
        if per_plate_configs is not None:
            per_plate_configs = {
                str(plate_path): pipeline_config
                for plate_path, pipeline_config in per_plate_configs.items()
            }
            for pipeline_config in per_plate_configs.values():
                if not isinstance(pipeline_config, PipelineConfig):
                    raise TypeError(
                        "per_plate_configs values must be PipelineConfig instances."
                    )

        return cls(
            plate_paths=plate_paths,
            pipeline_data=pipeline_data,
            global_pipeline_config=global_config,
            per_plate_configs=per_plate_configs,
        )

    def to_namespace(self) -> dict:
        namespace = {
            PlateManagerCodeNamespaceField.PLATE_PATHS.value: list(self.plate_paths),
            PlateManagerCodeNamespaceField.PIPELINE_DATA.value: self.pipeline_data,
        }
        if self.global_pipeline_config is not None:
            namespace[PlateManagerCodeNamespaceField.GLOBAL_CONFIG.value] = (
                self.global_pipeline_config
            )
        if self.per_plate_configs is not None:
            namespace[PlateManagerCodeNamespaceField.PER_PLATE_CONFIGS.value] = (
                self.per_plate_configs
            )
        return namespace


@dataclass(frozen=True, slots=True)
class PlateManagerCodeWorkflow(ManagerCodeExecutionWorkflow):
    """Applies edited orchestrator code to plate-manager state."""

    workflow_key = "plate_manager"
    manager: "PlateManagerWidget"

    def migration_namespace(self, code: str, error: Exception) -> dict | None:
        del code, error
        return None

    def apply_namespace(self, namespace: dict) -> bool:
        payload = PlateManagerOrchestratorCodePayload.from_namespace(namespace)
        if payload is None:
            return False

        self.sync_plate_entries(payload.plate_paths)

        if payload.global_pipeline_config is not None:
            self.apply_global_config(payload.global_pipeline_config)

        if payload.per_plate_configs is not None:
            self.apply_per_plate_configs(payload.per_plate_configs)

        self.apply_pipeline_data(payload.pipeline_data)
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
            self.manager._create_orchestrator_for_plate(plate_str)
            plate_name = Path(plate_str).name or plate_str
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
        last_pipeline_config = None
        for plate_path, pipeline_config in per_plate_configs.items():
            plate_key = str(plate_path)
            self.manager.plate_configs[plate_key] = pipeline_config

            orchestrator = ObjectStateRegistry.get_object(plate_key)
            if orchestrator:
                orchestrator.apply_pipeline_config(pipeline_config)
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

    def apply_pipeline_data(self, pipeline_data: dict[str, list[FunctionStep]]) -> None:
        plate_pipeline_editor = self.manager.plate_pipeline_editor
        if plate_pipeline_editor is None:
            logger.warning("No pipeline editor available to update pipeline data")
            self.manager.pipeline_data_changed.emit()
            return

        current_plate = plate_pipeline_editor.current_plate
        for plate_path, pipeline_steps in pipeline_data.items():
            pipeline_steps = self.runtime_bound_pipeline_for_plate(
                plate_path,
                pipeline_steps,
            )
            plate_pipeline_editor.update_pipeline_for_plate(
                plate_path,
                pipeline_steps,
            )
            logger.debug(
                "Updated pipeline for %s with %d steps",
                plate_path,
                len(pipeline_steps),
            )
            self.invalidate_orchestrator_compilation_state(plate_path)

            if plate_path != current_plate:
                continue
            plate_pipeline_editor.pipeline_steps = pipeline_steps
            plate_pipeline_editor.update_item_list()
            plate_pipeline_editor.pipeline_changed.emit(pipeline_steps)
            GuiEventBusBroadcaster(self.manager.event_bus).pipeline_changed(
                pipeline_steps
            )
            logger.debug(
                "Triggered UI cascade refresh for current plate: %s",
                plate_path,
            )

        self.manager.pipeline_data_changed.emit()

    def runtime_bound_pipeline_for_plate(
        self,
        plate_path: str,
        pipeline_steps: list[FunctionStep],
    ) -> list[FunctionStep]:
        """Preserve generated CellProfiler artifact bindings through code mode."""
        if not self.pipeline_contains_cellprofiler_steps(pipeline_steps):
            return pipeline_steps

        plate_pipeline_editor = self.manager.plate_pipeline_editor
        if plate_pipeline_editor is None:
            raise RuntimeError(
                "Cannot apply CellProfiler pipeline code without a pipeline editor."
            )

        import_result = self.cellprofiler_import_result_for_plate(
            plate_pipeline_editor,
            plate_path,
        )
        if import_result is None:
            identity = PlateScopeIdentity.from_scope_id(plate_path)
            if identity.cppipe_path is None:
                return pipeline_steps
            raise RuntimeError(
                "Cannot apply CellProfiler pipeline code for "
                f"{plate_path!r} because the .cppipe import context is not loaded. "
                "Initialize the plate before editing or applying generated pipeline code."
            )

        return CellProfilerPipelineRuntimeRebinder.from_import_result(
            import_result,
        ).rebind(pipeline_steps)

    @staticmethod
    def pipeline_contains_cellprofiler_steps(
        pipeline_steps: list[FunctionStep],
    ) -> bool:
        """Return whether any edited step references absorbed CellProfiler callables."""
        return any(
            isinstance(step, FunctionStep)
            and CellProfilerGeneratedStepFunctionSpec(step.func).metadata() is not None
            for step in pipeline_steps
        )

    @staticmethod
    def cellprofiler_import_result_for_plate(
        plate_pipeline_editor,
        plate_path: str,
    ):
        """Return the import result already associated with a logical plate scope."""
        plate_key = str(plate_path)
        if plate_key in plate_pipeline_editor.cellprofiler_import_results_by_plate:
            return plate_pipeline_editor.cellprofiler_import_results_by_plate[plate_key]
        if plate_pipeline_editor.current_plate == plate_key:
            return plate_pipeline_editor.cellprofiler_import_result
        return None

    def invalidate_orchestrator_compilation_state(self, plate_path: str) -> None:
        if plate_path in self.manager.plate_compiled_data:
            del self.manager.plate_compiled_data[plate_path]
            logger.debug("Cleared compiled data for %s", plate_path)

        orchestrator = ObjectStateRegistry.get_object(plate_path)
        if orchestrator and orchestrator.state == OrchestratorState.COMPILED:
            orchestrator._state = OrchestratorState.READY
            self.manager.orchestrator_state_changed.emit(plate_path, "READY")


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
