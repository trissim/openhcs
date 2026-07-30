from __future__ import annotations

from dataclasses import dataclass

import pytest
from PyQt6.QtWidgets import QDialog, QMessageBox

from objectstate import (
    ObjectState,
    ObjectStateRegistry,
    get_live_global_config,
    get_saved_global_config,
    set_live_global_config,
    set_saved_global_config,
)
from openhcs.core.config import GlobalPipelineConfig
from openhcs.pyqt_gui.windows.config_edit_session import ConfigEditSession
from openhcs.pyqt_gui.windows.config_window import (
    ConfigSaveParticipant,
    ConfigWindow,
    ConfigWindowTabSpec,
)


@dataclass(frozen=True)
class _FirstConfig:
    value: int = 1


@dataclass(frozen=True)
class _SecondConfig:
    value: int = 1


@pytest.fixture(autouse=True)
def _clear_object_state_registry():
    ObjectStateRegistry.clear()
    yield
    ObjectStateRegistry.clear()


@pytest.fixture
def suppress_save_dialog(monkeypatch):
    messages: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, _title, message: messages.append(message),
    )
    return messages


def _dispose_window(window: ConfigWindow, qapp) -> None:
    window.before_managed_close()
    window.deleteLater()
    qapp.processEvents()


def test_code_edit_updates_only_live_global_context_and_cancel_restores_it() -> None:
    saved = GlobalPipelineConfig(num_workers=1)
    prior_live = GlobalPipelineConfig(num_workers=2)
    edited = GlobalPipelineConfig(num_workers=3)
    set_saved_global_config(GlobalPipelineConfig, saved)
    set_live_global_config(GlobalPipelineConfig, prior_live)
    session = ConfigEditSession(
        state=ObjectState(saved, scope_id=""),
        original_config=saved,
    )

    session.apply_code_edit_context(edited)

    assert get_saved_global_config(GlobalPipelineConfig) is saved
    assert get_live_global_config(GlobalPipelineConfig) is edited

    assert session.restore_global_context_if_dirty() is True
    assert get_saved_global_config(GlobalPipelineConfig) is saved
    assert get_live_global_config(GlobalPipelineConfig) is prior_live


def test_save_guard_rejection_has_no_transaction_side_effects(
    qapp,
    suppress_save_dialog,
) -> None:
    state = ObjectState(_FirstConfig(), scope_id="guard")
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    effects: list[tuple[str, object]] = []

    def reject_mutation() -> None:
        raise RuntimeError("mutation rejected")

    window = ConfigWindow(
        tabs=(
            ConfigWindowTabSpec(
                state=state,
                save_participant=ConfigSaveParticipant(
                    apply=lambda value: effects.append(("apply", value)),
                    rollback=lambda value: effects.append(("rollback", value)),
                ),
                before_mutation=reject_mutation,
            ),
        ),
        scope_id="guard",
    )

    try:
        state.update_parameter("value", 2)
        qapp.processEvents()
        token_before = ObjectStateRegistry.get_token()
        saved_before = state.saved_object
        parameters_before = dict(state.parameters)
        dirty_before = state.is_raw_dirty

        window.save_config(close_window=False)

        assert ObjectStateRegistry.get_token() == token_before
        assert state.saved_object is saved_before
        assert state.parameters == parameters_before
        assert state.is_raw_dirty is dirty_before
        assert effects == []
        assert suppress_save_dialog == [
            "Failed to save configuration:\nmutation rejected"
        ]
    finally:
        _dispose_window(window, qapp)


def test_failure_after_first_mark_saved_restores_every_state_exactly(
    qapp,
    monkeypatch,
    suppress_save_dialog,
) -> None:
    first_saved = _FirstConfig()
    second_saved = _SecondConfig()
    first_state = ObjectState(first_saved, scope_id="first")
    second_state = ObjectState(second_saved, scope_id="second")
    ObjectStateRegistry.register(first_state, _skip_snapshot=True)
    ObjectStateRegistry.register(second_state, _skip_snapshot=True)
    window = ConfigWindow(
        tabs=(
            ConfigWindowTabSpec(state=first_state),
            ConfigWindowTabSpec(state=second_state),
        ),
        scope_id="",
    )

    try:
        first_state.update_parameter("value", 2)
        second_state.update_parameter("value", 3)
        first_parameters = dict(first_state.parameters)
        second_parameters = dict(second_state.parameters)
        monkeypatch.setattr(
            second_state,
            "mark_saved",
            lambda: (_ for _ in ()).throw(
                RuntimeError("second mark_saved failed")
            ),
        )

        window.save_config(close_window=False)

        assert first_state.saved_object is first_saved
        assert second_state.saved_object is second_saved
        assert first_state.parameters == first_parameters
        assert second_state.parameters == second_parameters
        assert first_state.is_raw_dirty is True
        assert second_state.is_raw_dirty is True
        assert suppress_save_dialog == [
            "Failed to save configuration:\nsecond mark_saved failed"
        ]
    finally:
        _dispose_window(window, qapp)


def test_failed_participant_restores_saved_live_and_session_cancel_baseline(
    qapp,
    suppress_save_dialog,
) -> None:
    saved = GlobalPipelineConfig(num_workers=1)
    prior_live = GlobalPipelineConfig(num_workers=2)
    set_saved_global_config(GlobalPipelineConfig, saved)
    set_live_global_config(GlobalPipelineConfig, prior_live)
    state = ObjectState(saved, scope_id="")
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    rollback_values: list[object] = []
    external_config = {"current": saved}
    restore_observations: list[tuple[object, object, object, object]] = []

    def fail_apply(value: object) -> None:
        external_config["current"] = value
        raise RuntimeError("participant failed")

    def rollback(value: object) -> None:
        rollback_values.append(value)
        external_config["current"] = value

    window = ConfigWindow(
        tabs=(
            ConfigWindowTabSpec(
                state=state,
                save_participant=ConfigSaveParticipant(
                    apply=fail_apply,
                    rollback=rollback,
                ),
            ),
        ),
        scope_id="",
    )

    try:
        state.update_parameter("num_workers", 3)
        window.active_tab.session.mark_global_field_changed()
        pre_save_live = get_live_global_config(GlobalPipelineConfig)
        assert pre_save_live is not prior_live
        state.on_state_changed(
            lambda _paths: restore_observations.append(
                (
                    state.saved_object,
                    external_config["current"],
                    get_saved_global_config(GlobalPipelineConfig),
                    get_live_global_config(GlobalPipelineConfig),
                )
            )
        )

        window.save_config(close_window=False)

        assert state.saved_object is saved
        assert state.parameters["num_workers"] == 3
        assert state.is_raw_dirty is True
        assert get_saved_global_config(GlobalPipelineConfig) is saved
        assert get_live_global_config(GlobalPipelineConfig) is pre_save_live
        assert rollback_values == [saved]
        assert window.active_tab.session.global_context_dirty is True
        restored_observations = [
            observation
            for observation in restore_observations
            if observation[0] is saved
        ]
        assert restored_observations
        assert restored_observations[-1] == (
            saved,
            saved,
            saved,
            pre_save_live,
        )

        window.restore_managed_state()

        assert state.saved_object is saved
        assert state.is_raw_dirty is False
        assert get_saved_global_config(GlobalPipelineConfig) is saved
        assert get_live_global_config(GlobalPipelineConfig) is prior_live
        assert suppress_save_dialog == [
            "Failed to save configuration:\nparticipant failed"
        ]
    finally:
        _dispose_window(window, qapp)


def test_failed_global_context_publication_restores_session_and_both_stores(
    qapp,
    monkeypatch,
    suppress_save_dialog,
) -> None:
    saved = GlobalPipelineConfig(num_workers=1)
    prior_live = GlobalPipelineConfig(num_workers=2)
    set_saved_global_config(GlobalPipelineConfig, saved)
    set_live_global_config(GlobalPipelineConfig, prior_live)
    state = ObjectState(saved, scope_id="")
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    window = ConfigWindow(
        tabs=(ConfigWindowTabSpec(state=state),),
        scope_id="",
    )

    try:
        state.update_parameter("num_workers", 3)
        session = window.active_tab.session
        session.mark_global_field_changed()
        pre_save_live = get_live_global_config(GlobalPipelineConfig)
        original_publish = ConfigEditSession.publish_saved_global_config

        def publish_then_fail(current_session, value: object) -> None:
            original_publish(current_session, value)
            if current_session is session:
                raise RuntimeError("global publication failed")

        monkeypatch.setattr(
            ConfigEditSession,
            "publish_saved_global_config",
            publish_then_fail,
        )

        window.save_config(close_window=False)

        assert state.saved_object is saved
        assert state.parameters["num_workers"] == 3
        assert state.is_raw_dirty is True
        assert get_saved_global_config(GlobalPipelineConfig) is saved
        assert get_live_global_config(GlobalPipelineConfig) is pre_save_live
        assert session.global_context_dirty is True
        assert suppress_save_dialog == [
            "Failed to save configuration:\nglobal publication failed"
        ]

        window.restore_managed_state()
        assert get_saved_global_config(GlobalPipelineConfig) is saved
        assert get_live_global_config(GlobalPipelineConfig) is prior_live
    finally:
        _dispose_window(window, qapp)


def test_participant_failure_compensates_attempted_effects_in_reverse_order(
    qapp,
    suppress_save_dialog,
) -> None:
    first_saved = _FirstConfig()
    second_saved = _SecondConfig()
    first_state = ObjectState(first_saved, scope_id="first")
    second_state = ObjectState(second_saved, scope_id="second")
    ObjectStateRegistry.register(first_state, _skip_snapshot=True)
    ObjectStateRegistry.register(second_state, _skip_snapshot=True)
    effects: list[tuple[str, object]] = []

    def fail_second(value: object) -> None:
        effects.append(("apply_second", value))
        raise RuntimeError("second participant failed")

    window = ConfigWindow(
        tabs=(
            ConfigWindowTabSpec(
                state=first_state,
                save_participant=ConfigSaveParticipant(
                    apply=lambda value: effects.append(("apply_first", value)),
                    rollback=lambda value: effects.append(("rollback_first", value)),
                ),
            ),
            ConfigWindowTabSpec(
                state=second_state,
                save_participant=ConfigSaveParticipant(
                    apply=fail_second,
                    rollback=lambda value: effects.append(("rollback_second", value)),
                ),
            ),
        ),
        scope_id="",
    )

    try:
        first_state.update_parameter("value", 2)
        second_state.update_parameter("value", 3)

        window.save_config(close_window=False)

        assert [name for name, _value in effects] == [
            "apply_first",
            "apply_second",
            "rollback_second",
            "rollback_first",
        ]
        assert first_state.saved_object is first_saved
        assert second_state.saved_object is second_saved
        assert first_state.is_raw_dirty is True
        assert second_state.is_raw_dirty is True
        assert suppress_save_dialog == [
            "Failed to save configuration:\nsecond participant failed"
        ]
    finally:
        _dispose_window(window, qapp)


def test_postcommit_window_failure_does_not_roll_back_commit(
    qapp,
    monkeypatch,
    suppress_save_dialog,
) -> None:
    state = ObjectState(_FirstConfig(), scope_id="postcommit")
    ObjectStateRegistry.register(state, _skip_snapshot=True)
    applied: list[object] = []
    rolled_back: list[object] = []
    window = ConfigWindow(
        tabs=(
            ConfigWindowTabSpec(
                state=state,
                save_participant=ConfigSaveParticipant(
                    apply=applied.append,
                    rollback=rolled_back.append,
                ),
            ),
        ),
        scope_id="postcommit",
    )

    try:
        state.update_parameter("value", 2)
        monkeypatch.setattr(
            window,
            "accept_committed_state",
            lambda: (_ for _ in ()).throw(
                RuntimeError("postcommit close failed")
            ),
        )

        window.save_config()

        assert state.saved_object == _FirstConfig(value=2)
        assert state.is_raw_dirty is False
        assert applied == [state.saved_object]
        assert rolled_back == []
        assert suppress_save_dialog == []
    finally:
        _dispose_window(window, qapp)


def test_inactive_tab_factory_failure_is_visible_and_rejects_window(
    qapp,
    monkeypatch,
) -> None:
    first_state = ObjectState(_FirstConfig(), scope_id="first")
    second_state = ObjectState(_SecondConfig(), scope_id="second")
    messages: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, _title, message: messages.append(message),
    )
    original_materialize = ConfigWindow._materialize_tab

    def materialize_or_fail(self, tab):
        if tab.spec.label == "_SecondConfig":
            raise RuntimeError("inactive tab build failed")
        return original_materialize(self, tab)

    monkeypatch.setattr(ConfigWindow, "_materialize_tab", materialize_or_fail)
    window = ConfigWindow(
        tabs=(
            ConfigWindowTabSpec(state=first_state),
            ConfigWindowTabSpec(state=second_state),
        ),
        scope_id="",
    )

    window.show()
    qapp.processEvents()
    first_view = window._tabs[0].view
    assert first_view is not None
    assert window._tabs[1].view is None

    window._tab_body.set_current_index(1)
    qapp.processEvents()

    assert window._tab_body.current_index() == 0
    assert window._tabs[0].view is first_view
    assert window._tabs[1].view is None
    assert window.result() == QDialog.DialogCode.Rejected
    assert messages == [
        "Failed to construct _SecondConfig:\ninactive tab build failed"
    ]


def test_progressive_form_build_failure_is_visible_and_rejects_window(
    qapp,
    monkeypatch,
) -> None:
    state = ObjectState(_FirstConfig(), scope_id="progressive")
    messages: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, _title, message: messages.append(message),
    )
    window = ConfigWindow(
        tabs=(ConfigWindowTabSpec(state=state),),
        scope_id="progressive",
    )
    window.show()
    qapp.processEvents()

    window.active_tab.form_manager.form_build_failed.emit(
        RuntimeError("later row failed")
    )
    qapp.processEvents()

    assert window.result() == QDialog.DialogCode.Rejected
    assert messages == [
        "Failed to construct _FirstConfig:\nlater row failed"
    ]
