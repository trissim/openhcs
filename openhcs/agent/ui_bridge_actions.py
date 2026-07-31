"""Agent-facing UI bridge action declarations."""

from __future__ import annotations

from enum import Enum


class PlateOperation(str, Enum):
    """Closed set of batch operations that validate visible plate rows."""

    INIT = "init"
    COMPILE = "compile"
    RUN = "run"


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


class PlateManagerAction(str, Enum):
    """Closed set of PlateManager button actions and agent-facing semantics."""

    side_effects: tuple[str, ...]
    confirmation_required: bool
    plate_operation: PlateOperation | None

    def __new__(
        cls,
        value: str,
        side_effects: tuple[str, ...],
        confirmation_required: bool,
        plate_operation: PlateOperation | None,
    ) -> "PlateManagerAction":
        member = str.__new__(cls, value)
        member._value_ = value
        member.side_effects = side_effects
        member.confirmation_required = confirmation_required
        member.plate_operation = plate_operation
        return member

    ADD_PLATE = (
        "add_plate",
        ("opens_file_dialog", "mutates_plate_collection"),
        True,
        None,
    )
    DELETE_PLATE = (
        "del_plate",
        ("mutates_plate_collection",),
        True,
        None,
    )
    EDIT_CONFIG = (
        "edit_config",
        ("opens_config_window", "may_mutate_plate_config"),
        True,
        None,
    )
    INIT_PLATE = (
        "init_plate",
        ("starts_initialization_workflow",),
        True,
        PlateOperation.INIT,
    )
    COMPILE_PLATE = (
        "compile_plate",
        ("starts_compile_workflow",),
        True,
        PlateOperation.COMPILE,
    )
    RUN_PLATE = (
        "run_plate",
        ("starts_or_stops_execution_workflow",),
        True,
        PlateOperation.RUN,
    )
    CODE_PLATE = (
        "code_plate",
        ("opens_code_document_window",),
        False,
        None,
    )
    VIEW_RESULTS = (
        "view_results",
        ("opens_results_window",),
        False,
        None,
    )
    VIEW_METADATA = (
        "view_metadata",
        ("opens_metadata_window",),
        False,
        None,
    )
