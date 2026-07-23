from openhcs.core.orchestrator.orchestrator import OrchestratorState
from openhcs.core.progress.projection import (
    PlateRuntimeIdentity,
    PlateRuntimeProjection,
    PlateRuntimeState,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_state import (
    TerminalExecutionStatus,
)
from openhcs.pyqt_gui.widgets.shared.services.plate_status_presenter import (
    PlateStatusPresenter,
)


def _runtime_plate(state: PlateRuntimeState, percent: float = 0.0) -> PlateRuntimeProjection:
    return PlateRuntimeProjection(
        identity=PlateRuntimeIdentity(execution_id="exec-1", plate_id="/tmp/plate"),
        state=state,
        percent=percent,
        axis_progress=tuple(),
        latest_timestamp=1.0,
    )


def test_runtime_projection_is_canonical_over_local_flags():
    prefix = PlateStatusPresenter.build_status_prefix(
        orchestrator_state=OrchestratorState.EXEC_FAILED,
        is_init_pending=True,
        is_compile_pending=True,
        is_execution_active=False,
        terminal_status=TerminalExecutionStatus.FAILED,
        queue_position=3,
        runtime_projection=_runtime_plate(PlateRuntimeState.COMPILING, 42.0),
    )

    assert prefix == "⏳ Compiling 42.0%"


def test_queue_position_used_when_runtime_projection_absent():
    prefix = PlateStatusPresenter.build_status_prefix(
        orchestrator_state=None,
        is_init_pending=False,
        is_compile_pending=False,
        is_execution_active=False,
        terminal_status=None,
        queue_position=2,
        runtime_projection=None,
    )

    assert prefix == "⏳ Queued 0.0% (q#2)"


def test_pending_status_when_active_without_runtime_or_queue():
    prefix = PlateStatusPresenter.build_status_prefix(
        orchestrator_state=None,
        is_init_pending=False,
        is_compile_pending=False,
        is_execution_active=True,
        terminal_status=None,
        queue_position=None,
        runtime_projection=None,
    )

    assert prefix == "⏳ Pending"


def test_orchestrator_fallback_for_idle_case():
    prefix = PlateStatusPresenter.build_status_prefix(
        orchestrator_state=OrchestratorState.COMPILED,
        is_init_pending=False,
        is_compile_pending=False,
        is_execution_active=False,
        terminal_status=None,
        queue_position=None,
        runtime_projection=None,
    )

    assert prefix == "✓ Compiled"
