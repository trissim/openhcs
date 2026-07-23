"""Change-filtering tests for code-document form synchronization."""

from __future__ import annotations

from dataclasses import dataclass

from openhcs.core.steps.function_step import FunctionStep
from openhcs.ui.shared.code_editor_form_updater import CodeEditorFormUpdater


@dataclass(frozen=True)
class _NestedConfig:
    enabled: bool | None = None
    label: str | None = None


@dataclass(frozen=True)
class _RootConfig:
    name: str | None = None
    nested: _NestedConfig | None = None


class _RecordingFormManager:
    def __init__(
        self,
        parameters: dict[str, object],
        *,
        nested_managers: dict[str, "_RecordingFormManager"] | None = None,
    ) -> None:
        self.parameters = parameters
        self.nested_managers = nested_managers or {}
        self.updated: list[tuple[str, object]] = []

    def update_parameter(self, field_name: str, value: object) -> None:
        self.updated.append((field_name, value))
        self.parameters[field_name] = value


def test_update_form_from_instance_dispatches_only_changed_nested_leaf() -> None:
    nested_manager = _RecordingFormManager(
        {
            "enabled": None,
            "label": "same",
        }
    )
    root_manager = _RecordingFormManager(
        {
            "name": "same",
            "nested": _NestedConfig(enabled=None, label="same"),
        },
        nested_managers={"nested": nested_manager},
    )

    CodeEditorFormUpdater.update_form_from_instance(
        root_manager,
        _RootConfig(
            name="same",
            nested=_NestedConfig(enabled=False, label="same"),
        ),
    )

    assert root_manager.updated == []
    assert nested_manager.updated == [("enabled", False)]


def test_update_form_from_instance_skips_fully_unchanged_document() -> None:
    nested_manager = _RecordingFormManager(
        {
            "enabled": None,
            "label": "same",
        }
    )
    root_manager = _RecordingFormManager(
        {
            "name": "same",
            "nested": _NestedConfig(enabled=None, label="same"),
        },
        nested_managers={"nested": nested_manager},
    )

    CodeEditorFormUpdater.update_form_from_instance(
        root_manager,
        _RootConfig(
            name="same",
            nested=_NestedConfig(enabled=None, label="same"),
        ),
    )

    assert root_manager.updated == []
    assert nested_manager.updated == []


def test_update_form_from_instance_preserves_explicit_reset_to_none() -> None:
    root_manager = _RecordingFormManager(
        {
            "name": "override",
            "nested": None,
        }
    )

    CodeEditorFormUpdater.update_form_from_instance(
        root_manager,
        _RootConfig(name=None, nested=None),
    )

    assert root_manager.updated == [("name", None)]


def test_update_form_from_instance_updates_unexpanded_dataclass_once() -> None:
    original_nested = _NestedConfig(enabled=None, label="same")
    replacement_nested = _NestedConfig(enabled=True, label="same")
    root_manager = _RecordingFormManager(
        {
            "name": "same",
            "nested": original_nested,
        }
    )

    CodeEditorFormUpdater.update_form_from_instance(
        root_manager,
        _RootConfig(name="same", nested=replacement_nested),
    )

    assert root_manager.updated == [("nested", replacement_nested)]


def test_update_form_from_non_dataclass_step_dispatches_only_changed_field() -> None:
    root_manager = _RecordingFormManager(
        {
            "name": "Original",
            "description": None,
            "enabled": True,
        }
    )

    CodeEditorFormUpdater.update_form_from_instance(
        root_manager,
        FunctionStep(name="Renamed"),
    )

    assert root_manager.updated == [("name", "Renamed")]
