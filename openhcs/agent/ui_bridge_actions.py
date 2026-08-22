"""Agent-facing UI bridge action declarations."""

from __future__ import annotations

from enum import Enum


class PlateOperation(str, Enum):
    """Closed set of batch operations that validate visible plate rows."""

    INIT = "init"
    COMPILE = "compile"
    RUN = "run"


class ManagerButtonPresentationMixin:
    """Project an action declaration into the generic manager-button contract."""

    value: str
    label: str
    tooltip: str

    @property
    def button_config(self) -> tuple[str, str, str]:
        return self.label, self.value, self.tooltip


class MainWindowAction(str, Enum):
    """Closed set of agent-facing main-window actions."""

    title: str
    side_effects: tuple[str, ...]
    confirmation_required: bool

    def __new__(
        cls,
        value: str,
        title: str,
        side_effects: tuple[str, ...],
        confirmation_required: bool,
    ) -> "MainWindowAction":
        member = str.__new__(cls, value)
        member._value_ = value
        member.title = title
        member.side_effects = side_effects
        member.confirmation_required = confirmation_required
        return member

    CHECK_FOR_UPDATES = (
        "check_for_updates",
        "Check for Updates",
        ("checks_trusted_release_service", "may_open_update_confirmation"),
        False,
    )


class PlateManagerAction(ManagerButtonPresentationMixin, str, Enum):
    """Closed set of PlateManager button actions and agent-facing semantics."""

    side_effects: tuple[str, ...]
    confirmation_required: bool
    plate_operation: PlateOperation | None

    def __new__(
        cls,
        value: str,
        label: str,
        tooltip: str,
        side_effects: tuple[str, ...],
        confirmation_required: bool,
        plate_operation: PlateOperation | None,
    ) -> "PlateManagerAction":
        member = str.__new__(cls, value)
        member._value_ = value
        member.label = label
        member.tooltip = tooltip
        member.side_effects = side_effects
        member.confirmation_required = confirmation_required
        member.plate_operation = plate_operation
        return member

    ADD_PLATE = (
        "add_plate",
        "Add",
        "Add new plate directory",
        ("opens_file_dialog", "mutates_plate_collection"),
        True,
        None,
    )
    DELETE_PLATE = (
        "del_plate",
        "Del",
        "Delete selected plates",
        ("mutates_plate_collection",),
        True,
        None,
    )
    EDIT_CONFIG = (
        "edit_config",
        "Edit",
        "Edit plate configuration",
        ("opens_config_window", "may_mutate_plate_config"),
        True,
        None,
    )
    INIT_PLATE = (
        "init_plate",
        "Init",
        "Initialize selected plates",
        ("starts_initialization_workflow",),
        True,
        PlateOperation.INIT,
    )
    COMPILE_PLATE = (
        "compile_plate",
        "Compile",
        "Compile plate pipelines",
        ("starts_compile_workflow",),
        True,
        PlateOperation.COMPILE,
    )
    RUN_PLATE = (
        "run_plate",
        "Run",
        "Run/Stop plate execution",
        ("starts_or_stops_execution_workflow",),
        True,
        PlateOperation.RUN,
    )
    CODE_PLATE = (
        "code_plate",
        "Code",
        "Generate Python code",
        ("opens_code_document_window",),
        False,
        None,
    )
    VIEW_RESULTS = (
        "view_results",
        "Results",
        "View live measurement results",
        ("opens_results_window",),
        False,
        None,
    )
    VIEW_METADATA = (
        "view_metadata",
        "Viewer",
        "View plate metadata",
        ("opens_metadata_window",),
        False,
        None,
    )
