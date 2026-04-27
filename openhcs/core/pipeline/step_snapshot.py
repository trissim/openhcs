"""Typed compiler snapshots for resolved pipeline steps."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from openhcs.core.config import WellFilterConfig
from openhcs.core.source_bindings import (
    EMPTY_SOURCE_BINDINGS,
    StepSourceBindingsConfig,
)
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.lib_registry.unified_registry import (
    LibraryRegistryBase,
)


@dataclass(frozen=True, slots=True)
class StepProcessingSnapshot:
    """ObjectState-resolved processing config facts used by the compiler."""

    variable_components: Sequence[Any]
    group_by: Any
    input_source: Any
    config: Any


@dataclass(frozen=True, slots=True)
class StepWellFilterSnapshot:
    """ObjectState-resolved well filter attached to one step config root."""

    root: str
    well_filter: Any
    well_filter_mode: Any


@dataclass(frozen=True, slots=True)
class StepSnapshot:
    """Compiler input for one already-resolved pipeline step.

    The normal compiler path has already converted ObjectState to a resolved step
    object before this snapshot is built. This type does not call to_object();
    it captures the saved ObjectState values that downstream compiler phases need.
    """

    index: int
    scope_id: str
    name: str
    step_type: str
    enabled: bool
    is_function_step: bool
    func: Any
    source_bindings: StepSourceBindingsConfig
    processing: StepProcessingSnapshot
    materialization_config: Any
    injectable_values: Mapping[str, Any]
    well_filters: tuple[StepWellFilterSnapshot, ...] = ()

    @classmethod
    def from_resolved_step(
        cls,
        *,
        index: int,
        step: AbstractStep,
        step_state: Any,
    ) -> "StepSnapshot":
        """Build a snapshot from a resolved step plus its saved ObjectState."""
        processing = StepProcessingSnapshot(
            variable_components=_saved_value(
                step_state,
                "processing_config.variable_components",
                index,
            ),
            group_by=_saved_value(
                step_state,
                "processing_config.group_by",
                index,
            ),
            input_source=_saved_value(
                step_state,
                "processing_config.input_source",
                index,
            ),
            config=_saved_value(step_state, "processing_config", index),
        )

        injectable_values = {
            param_name: _saved_value(step_state, param_name, index)
            for param_name, _, _ in LibraryRegistryBase.INJECTABLE_PARAMS
        }

        return cls(
            index=index,
            scope_id=step_state.scope_id,
            name=step.name,
            step_type=step.__class__.__name__,
            enabled=bool(_saved_value(step_state, "enabled", index)),
            is_function_step=isinstance(step, FunctionStep),
            func=step.func if isinstance(step, FunctionStep) else None,
            source_bindings=(
                _saved_value(step_state, "source_bindings", index)
                if isinstance(step, FunctionStep)
                else EMPTY_SOURCE_BINDINGS
            ),
            processing=processing,
            materialization_config=_saved_value(
                step_state,
                "step_materialization_config",
                index,
            ),
            injectable_values=MappingProxyType(injectable_values),
            well_filters=_build_well_filter_snapshots(step_state, index),
        )

    @property
    def variable_components(self) -> Sequence[Any]:
        return self.processing.variable_components

    @property
    def group_by(self) -> Any:
        return self.processing.group_by

    @property
    def input_source(self) -> Any:
        return self.processing.input_source

    @property
    def processing_config(self) -> Any:
        return self.processing.config


def build_step_snapshots(
    steps: Sequence[AbstractStep],
    step_state_map: Mapping[int, Any],
) -> tuple[StepSnapshot, ...]:
    """Build compiler snapshots for already-resolved steps."""
    snapshots: list[StepSnapshot] = []
    for index, step in enumerate(steps):
        try:
            step_state = step_state_map[index]
        except KeyError as exc:
            raise ValueError(
                f"Missing ObjectState for resolved step {index} "
                f"({step.name})."
            ) from exc
        snapshots.append(
            StepSnapshot.from_resolved_step(
                index=index,
                step=step,
                step_state=step_state,
            )
        )
    return tuple(snapshots)


def _build_well_filter_snapshots(
    step_state: Any,
    step_index: int,
) -> tuple[StepWellFilterSnapshot, ...]:
    roots: list[str] = []
    for path, value_type in _path_to_type_map(step_state, step_index).items():
        if "." in path:
            continue
        if isinstance(value_type, type) and issubclass(value_type, WellFilterConfig):
            roots.append(path)

    snapshots: list[StepWellFilterSnapshot] = []
    for root in sorted(roots):
        well_filter = _saved_value(
            step_state,
            f"{root}.well_filter",
            step_index,
        )
        if well_filter is None:
            continue
        snapshots.append(
            StepWellFilterSnapshot(
                root=root,
                well_filter=well_filter,
                well_filter_mode=_saved_value(
                    step_state,
                    f"{root}.well_filter_mode",
                    step_index,
                ),
            )
        )
    return tuple(snapshots)


def _path_to_type_map(step_state: Any, step_index: int) -> Mapping[str, Any]:
    path_to_type = step_state._path_to_type
    if not isinstance(path_to_type, Mapping):
        raise TypeError(
            f"Step {step_index} ObjectState _path_to_type must be a mapping, "
            f"got {type(path_to_type).__name__}."
        )
    return path_to_type


def _saved_value(step_state: Any, path: str, step_index: int) -> Any:
    try:
        return step_state.get_saved_resolved_value(path)
    except Exception as exc:
        raise ValueError(
            f"Step {step_index} snapshot requires saved ObjectState value "
            f"'{path}'."
        ) from exc
