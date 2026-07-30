"""Image Browser streaming configuration ownership regressions."""

from types import SimpleNamespace

from PyQt6 import sip
from objectstate import ObjectState, ObjectStateRegistry
from PyQt6.QtTest import QTest

from pyqt_reactive.widgets.shared.responsive_layout_widgets import (
    ResponsiveParameterRow,
)
from pyqt_reactive.widgets.structural_table import StructuralMaskedContainerTarget

from objectstate.global_config import set_global_config_for_editing
from openhcs.core.config import (
    GlobalPipelineConfig,
    PipelineConfig,
    StreamingConfig,
    StreamingDefaults,
)
from openhcs.pyqt_gui.widgets.image_browser import (
    ImageBrowserConfig,
    ImageBrowserWidget,
)


def test_image_browser_state_resolves_plate_then_global(tmp_path) -> None:
    """The browser is a sibling of steps under its associated PipelineConfig."""

    ObjectStateRegistry.clear()
    global_config = GlobalPipelineConfig(
        streaming_defaults=StreamingDefaults(host="global-host"),
    )
    set_global_config_for_editing(GlobalPipelineConfig, global_config)
    try:
        global_state = ObjectState(global_config, scope_id="")
        ObjectStateRegistry.register(global_state, _skip_snapshot=True)

        plate_path = tmp_path / "plate"
        plate_state = ObjectState(
            PipelineConfig(),
            scope_id=str(plate_path),
            parent_state=global_state,
        )
        ObjectStateRegistry.register(plate_state, _skip_snapshot=True)

        browser = SimpleNamespace(config=ImageBrowserConfig(), scope_id=None)
        browser_state = ImageBrowserWidget._create_state_for_orchestrator(
            browser,
            SimpleNamespace(plate_path=plate_path),
        )

        assert browser_state._parent_state is plate_state
        for config_key in StreamingConfig.supported_config_keys():
            path = f"{config_key}.host"
            assert browser_state.get_resolved_value(path) == "global-host"

            plate_state.update_parameter(path, f"plate-{config_key}")
            assert browser_state.get_resolved_value(path) == f"plate-{config_key}"

            browser_state.update_parameter(path, f"browser-{config_key}")
            assert browser_state.get_resolved_value(path) == f"browser-{config_key}"
    finally:
        ObjectStateRegistry.clear()
        set_global_config_for_editing(GlobalPipelineConfig, GlobalPipelineConfig())


def test_image_browser_edits_advance_live_saved_baseline() -> None:
    """A live streaming edit has no separate unsaved configuration state."""

    state = ObjectState(ImageBrowserConfig())
    path = "napari_streaming_config.port"
    state.update_parameter(path, 6001)
    assert state.is_raw_dirty
    assert state.get_saved_resolved_value(path) != 6001

    browser = SimpleNamespace(state=state)
    ImageBrowserWidget._on_parameter_changed(browser, path, 6001)

    assert state.get_resolved_value(path) == 6001
    assert state.get_saved_resolved_value(path) == 6001
    assert not state.is_raw_dirty
    assert state.dirty_fields == set()


def test_streaming_enableable_is_title_only_and_targets_its_config_group(qapp) -> None:
    """Each registered viewer owns one title checkbox and its group flash target."""

    ObjectStateRegistry.clear()
    browser = ImageBrowserWidget()
    browser.resize(1400, 900)
    browser.show()
    config_keys = StreamingConfig.supported_config_keys()

    try:
        for _ in range(300):
            qapp.processEvents()
            ready = len(browser.tabbed_form.tab_forms) == len(config_keys)
            if ready:
                ready = all(
                    config_key in form.nested_managers
                    and "enabled" in form.nested_managers[config_key].widgets
                    and form.widgets[config_key].title_layout.isAncestorOf(
                        form.nested_managers[config_key].widgets["enabled"]
                    )
                    for config_key, form in zip(
                        config_keys,
                        browser.tabbed_form.tab_forms,
                        strict=True,
                    )
                )
            if ready:
                break
            QTest.qWait(10)

        assert ready
        # Let responsive-row timers run after enabled-title relocation and force
        # the same width transitions that previously pulled the checkbox and its
        # Reset control back into the otherwise-empty source row.
        for width in (900, 1400, 700, 1400):
            browser.resize(width, 900)
            QTest.qWait(75)
            qapp.processEvents()

        for index, (config_key, form) in enumerate(
            zip(config_keys, browser.tabbed_form.tab_forms, strict=True)
        ):
            browser.tabbed_form.tab_widget.setCurrentIndex(index)
            QTest.qWait(75)
            qapp.processEvents()

            nested = form.nested_managers[config_key]
            container = form.widgets[config_key]
            enabled_widget = nested.widgets["enabled"]
            enabled_label = nested.labels["enabled"]
            enabled_reset = nested.reset_buttons["enabled"]

            source_rows = [
                row
                for row in container.findChildren(ResponsiveParameterRow)
                if any(
                    widget in (enabled_widget, enabled_label, enabled_reset)
                    for widget, _stretch in (
                        *row._left_widgets,
                        *row._right_widgets,
                    )
                )
            ]
            visible_empty_rows = [
                row
                for row in container.findChildren(ResponsiveParameterRow)
                if row.isVisibleTo(browser)
                and row._row1_layout.count() == 0
                and row._row2_layout.count() == 0
            ]

            assert container.title_layout.isAncestorOf(enabled_widget)
            assert container.title_layout.isAncestorOf(enabled_reset)
            assert enabled_widget.isVisibleTo(browser)
            assert enabled_label.isHidden()
            assert nested.labels["enabled"] is enabled_label
            assert enabled_label.parentWidget() is None
            assert not sip.isdeleted(enabled_label)
            assert not sip.isdeleted(enabled_widget)
            assert not sip.isdeleted(enabled_reset)
            assert not source_rows
            assert not visible_empty_rows

            target = nested.field_flash_target("enabled")
            assert isinstance(target, StructuralMaskedContainerTarget)
            assert target.container is container

            port = 6100 + index
            nested.update_parameter("port", port)
            qapp.processEvents()
            path = f"{config_key}.port"
            assert browser.state.get_resolved_value(path) == port
            assert browser.state.get_saved_resolved_value(path) == port
            assert not container._title_label.text().startswith("* ")
    finally:
        browser.cleanup()
        browser.close()
        browser.deleteLater()
        ObjectStateRegistry.clear()
