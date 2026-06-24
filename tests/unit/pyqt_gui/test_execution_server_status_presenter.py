from openhcs.core.progress.projection import (
    ExecutionRuntimeProjection,
    PlateRuntimeIdentity,
    PlateRuntimeProjection,
    PlateRuntimeState,
)
from openhcs.pyqt_gui.widgets.shared.services.execution_server_status_presenter import (
    ExecutionServerStatusPresenter,
)


def test_execution_server_status_presenter_returns_ready_when_no_plates():
    presenter = ExecutionServerStatusPresenter()
    projection = ExecutionRuntimeProjection()

    view = presenter.build_status_text(
        projection=projection,
        server_info=None,
    )

    assert view.text == "Ready"


def test_execution_server_status_presenter_includes_projection_counts():
    presenter = ExecutionServerStatusPresenter()
    plate_one = PlateRuntimeProjection(
        identity=PlateRuntimeIdentity(execution_id="exec-1", plate_id="/tmp/p1"),
        state=PlateRuntimeState.COMPILING,
        percent=40.0,
        axis_progress=tuple(),
        latest_timestamp=1.0,
    )
    plate_two = PlateRuntimeProjection(
        identity=PlateRuntimeIdentity(execution_id="exec-2", plate_id="/tmp/p2"),
        state=PlateRuntimeState.EXECUTING,
        percent=85.0,
        axis_progress=tuple(),
        latest_timestamp=1.0,
    )
    projection = ExecutionRuntimeProjection(
        by_plate_latest={"/tmp/p1": plate_one, "/tmp/p2": plate_two},
        compiling_count=1,
        executing_count=1,
        compiled_count=0,
        complete_count=0,
        overall_percent=62.5,
    )
    view = presenter.build_status_text(
        projection=projection,
        server_info=None,
    )

    assert (
        view.text
        == "Server: ⏳ 1 compiling, ⚙️ 1 executing | 2 plates | avg 62.5%"
    )


def test_execution_server_status_presenter_includes_failed_projection_count():
    presenter = ExecutionServerStatusPresenter()
    failed_plate = PlateRuntimeProjection(
        identity=PlateRuntimeIdentity(execution_id="exec-3", plate_id="/tmp/p3"),
        state=PlateRuntimeState.FAILED,
        percent=10.0,
        axis_progress=tuple(),
        latest_timestamp=1.0,
    )
    projection = ExecutionRuntimeProjection(
        by_plate_latest={"/tmp/p3": failed_plate},
        failed_count=1,
        overall_percent=10.0,
    )

    view = presenter.build_status_text(
        projection=projection,
        server_info=None,
    )

    assert view.text == "Server: ❌ 1 failed | 1 plates | avg 10.0%"
