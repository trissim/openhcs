from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from openhcs.agent.dto.ui_bridge import (
    UiCodeDocumentSelectionMode,
    UiStateSurfaceRequest,
)
from openhcs.agent.ui_bridge_actions import PlateManagerAction
from openhcs.agent.ui_bridge_identities import (
    PlateManagerLiveMeasurementsStateSurfaceIdentityDeclaration,
    UiStateSurfaceIdentityDeclarationBase,
)
from objectstate.object_state import ObjectStateRegistry
from openhcs.core.artifacts import MeasurementsArtifactType
from openhcs.core.component_group_scope import RuntimeExecutionAxisScope
from openhcs.core.progress import (
    ProgressEvent,
    ProgressIdentity,
    ProgressPhase,
    ProgressStatus,
)
from openhcs.core.progress.live_measurements import (
    LiveMeasurementProgressPayload,
    LiveMeasurementTablePreview,
)
from openhcs.core.runtime_artifact_values import ArtifactKey
from openhcs.core.runtime_stores import (
    RuntimeArtifactAddress,
    RuntimeArtifactLocation,
)
from openhcs.pyqt_gui.services.ui_agent_bridge import UiAgentBridgeService
from openhcs.pyqt_gui.services.ui_bridge_contracts import (
    state_surface_declaration_for_identity,
)
from openhcs.pyqt_gui.services.ui_bridge_live_overview import (
    LiveOverviewBridgeProviderSet,
)
from openhcs.pyqt_gui.services.ui_bridge_plate_manager import (
    PlateManagerBridgeProviderSet,
)
from openhcs.pyqt_gui.services.ui_bridge_registry import CompositeUiBridgeProviderSet
from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget
from openhcs.pyqt_gui.widgets.shared.services.live_measurement_progress_service import (
    LiveMeasurementAvailableNotification,
)
from openhcs.pyqt_gui.windows.live_measurements_window import LiveMeasurementTableModel


@dataclass(frozen=True, slots=True)
class _PlateRow:
    scope_id: str


class _Manager:
    def __init__(self) -> None:
        self.plates = [_PlateRow("plate-1"), _PlateRow("plate-2")]
        self.selected = [self.plates[0]]
        self.live_measurement_model = LiveMeasurementTableModel()

    def get_selected_items(self) -> list[_PlateRow]:
        return list(self.selected)


class _InlineDispatcher:
    def call(self, callback):
        return callback()


def _notification() -> LiveMeasurementAvailableNotification:
    preview = LiveMeasurementTablePreview(
        address=RuntimeArtifactAddress(
            key=ArtifactKey(
                name="PerNeuronMeasurements",
                artifact_type=MeasurementsArtifactType,
                scope=RuntimeExecutionAxisScope(axis_id="B03"),
            ),
            location=RuntimeArtifactLocation(
                path="/runtime/per_neuron_measurements.pkl",
                backend="memory",
            ),
            value_type="MeasurementTable",
        ),
        columns=("label_id", "neurite_length_px"),
        rows=(
            {"label_id": 1, "neurite_length_px": 160.0},
            {"label_id": 2, "neurite_length_px": 55.0},
        ),
        row_count=3,
        truncated_rows=True,
        truncated_columns=False,
        object_name="neurons",
        source_image_name="SMI312",
    )
    payload = LiveMeasurementProgressPayload(
        previews=(preview,),
        preview_count=1,
        truncated_previews=False,
    )
    event = ProgressEvent(
        identity=ProgressIdentity(
            execution_id="execution-1",
            plate_id="plate-1",
            axis_id="B03",
            step_name="Neurite Outgrowth",
        ),
        phase=ProgressPhase.STEP_COMPLETED,
        status=ProgressStatus.SUCCESS,
        percent=100.0,
        completed=1,
        total=1,
        timestamp=1.0,
        pid=1234,
        context=payload.to_context(),
    )
    return LiveMeasurementAvailableNotification(event=event, payload=payload)


def _bridge(manager: _Manager) -> UiAgentBridgeService:
    return UiAgentBridgeService(
        provider_set=CompositeUiBridgeProviderSet(
            (
                PlateManagerBridgeProviderSet(manager),
                LiveOverviewBridgeProviderSet(),
            )
        ),
        dispatcher=_InlineDispatcher(),
    )


def _live_measurement_declaration():
    return state_surface_declaration_for_identity(
        PlateManagerWidget.UI_STATE_SURFACE_DECLARATIONS,
        PlateManagerLiveMeasurementsStateSurfaceIdentityDeclaration,
    )


def test_live_measurement_surface_is_declared_by_owning_widget() -> None:
    declaration = _live_measurement_declaration()

    assert declaration.surface_id == "plate_manager.live_measurements"
    assert (
        declaration.identity
        is PlateManagerLiveMeasurementsStateSurfaceIdentityDeclaration
    )
    assert declaration.payload_schema == "openhcs.ui.live_measurements_state.v1"
    assert declaration.related_action_ids == (PlateManagerAction.VIEW_RESULTS.value,)
    assert declaration.surface_id in {
        declaration_type.require_value()
        for declaration_type in UiStateSurfaceIdentityDeclarationBase.__registry__.values()
    }
    assert not hasattr(declaration.identity, "widget_identity")


def test_ast_inventory_finds_each_protocol_surface_only_on_its_widget_owner() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    declaration_identities: set[str] = set()
    declaration_count = 0

    for path in (repository_root / "openhcs" / "pyqt_gui").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for owner in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            for statement in owner.body:
                if not isinstance(statement, ast.Assign) or not any(
                    isinstance(target, ast.Name)
                    and target.id == "UI_STATE_SURFACE_DECLARATIONS"
                    for target in statement.targets
                ):
                    continue
                declarations = [
                    node
                    for node in ast.walk(statement.value)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "UiOwnedStateSurfaceDeclaration"
                ]
                declaration_count += len(declarations)
                for declaration in declarations:
                    identity_keyword = next(
                        keyword
                        for keyword in declaration.keywords
                        if keyword.arg == "identity"
                    )
                    assert isinstance(identity_keyword.value, ast.Name)
                    declaration_identities.add(identity_keyword.value.id)

    protocol_identities = {
        declaration_type.__name__
        for declaration_type in UiStateSurfaceIdentityDeclarationBase.__registry__.values()
        if issubclass(declaration_type, UiStateSurfaceIdentityDeclarationBase)
    }
    assert declaration_count == len(protocol_identities)
    assert declaration_identities == protocol_identities
    assert all(
        not hasattr(declaration_type, "widget_identity")
        for declaration_type in UiStateSurfaceIdentityDeclarationBase.__registry__.values()
        if issubclass(declaration_type, UiStateSurfaceIdentityDeclarationBase)
    )


def test_live_measurement_surface_projects_exact_bounded_preview_and_revision() -> None:
    ObjectStateRegistry.clear()
    manager = _Manager()
    manager.live_measurement_model.add_notification(_notification())
    bridge = _bridge(manager)
    declaration = _live_measurement_declaration()

    catalog = bridge.list_state_surfaces()
    state = bridge.get_state_surface(
        UiStateSurfaceRequest(
            surface_id=declaration.surface_id,
            selection_mode=UiCodeDocumentSelectionMode.ALL.value,
        )
    )
    poll = bridge.get_state_surface(
        UiStateSurfaceRequest(
            surface_id=declaration.surface_id,
            selection_mode=UiCodeDocumentSelectionMode.ALL.value,
            base_revision_token=state.current_revision_token,
        )
    )

    assert declaration.surface_id in {surface.surface_id for surface in catalog.surfaces}
    assert state.payload_schema == declaration.payload_schema
    assert state.payload["retained_entry_count"] == 1
    assert state.payload["visible_entry_count"] == 1
    assert state.payload["total_row_count"] == 3
    entry = state.payload["entries"][0]
    assert entry["execution_id"] == "execution-1"
    assert entry["plate_id"] == "plate-1"
    assert entry["step_name"] == "Neurite Outgrowth"
    assert entry["preview"]["columns"] == ["label_id", "neurite_length_px"]
    assert entry["preview"]["rows"][0] == {
        "label_id": 1,
        "neurite_length_px": 160.0,
    }
    assert entry["preview"]["row_count"] == 3
    assert entry["preview"]["truncated_rows"] is True
    assert entry["preview"]["object_name"] == "neurons"
    assert poll.unchanged is True


def test_live_measurement_surface_honors_plate_selection_and_populates_overview() -> None:
    ObjectStateRegistry.clear()
    manager = _Manager()
    manager.live_measurement_model.add_notification(_notification())
    bridge = _bridge(manager)
    declaration = _live_measurement_declaration()

    selected = bridge.get_state_surface(
        UiStateSurfaceRequest(
            surface_id=declaration.surface_id,
            selection_mode=UiCodeDocumentSelectionMode.SELECTED.value,
        )
    )
    overview = bridge.get_state_surface(
        UiStateSurfaceRequest(surface_id="ui_live_overview.state")
    )
    section = next(
        item
        for item in overview.payload["sections"]
        if item["section_id"] == declaration.surface_id
    )

    assert selected.selected_scope_ids == ("plate-1",)
    assert selected.payload["visible_entry_count"] == 1
    assert section["metrics"][0] == {
        "key": "tables",
        "label": "tables",
        "value": "1",
    }
    assert section["metrics"][1]["value"] == "3"
    assert section["items"][0]["label"] == (
        "Neurite Outgrowth: PerNeuronMeasurements"
    )

    manager.selected = [manager.plates[1]]
    filtered = bridge.get_state_surface(
        UiStateSurfaceRequest(
            surface_id=declaration.surface_id,
            selection_mode=UiCodeDocumentSelectionMode.SELECTED.value,
        )
    )

    assert filtered.selected_scope_ids == ("plate-2",)
    assert filtered.payload["visible_entry_count"] == 0
    assert filtered.payload["entries"] == []
