"""Typed compiler snapshots for resolved pipeline steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from openhcs.core.config import (
    ProcessingConfig,
    StepMaterializationConfig,
    WellFilterConfig,
)
from openhcs.core.pipeline.step_config_universe import StepConfigUniverse
from openhcs.core.runtime_invocation import RuntimeParameterBinding
from openhcs.core.source_bindings import (
    EMPTY_SOURCE_BINDINGS,
    StepSourceBindingsConfig,
)
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_step import FunctionStep

if TYPE_CHECKING:
    from objectstate import ObjectState


@dataclass(frozen=True, slots=True)
class StepWellFilterSnapshot:
    """ObjectState-resolved well filter attached to one step config root."""

    config: WellFilterConfig


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
    configs: StepConfigUniverse

    @classmethod
    def from_resolved_step(
        cls,
        *,
        index: int,
        step: AbstractStep,
        step_state: "ObjectState",
    ) -> "StepSnapshot":
        """Build a snapshot from a resolved step plus its saved ObjectState."""
        configs = StepConfigUniverse.from_object_state(step_state)

        return cls(
            index=index,
            scope_id=step_state.scope_id,
            name=step.name,
            step_type=step.__class__.__name__,
            enabled=bool(step.enabled),
            is_function_step=isinstance(step, FunctionStep),
            func=step.func if isinstance(step, FunctionStep) else None,
            configs=configs,
        )

    @property
    def variable_components(self) -> Sequence[Any]:
        return self.processing_config.variable_components

    @property
    def group_by(self) -> Any:
        return self.processing_config.group_by

    @property
    def input_source(self) -> Any:
        return self.processing_config.input_source

    @property
    def processing_config(self) -> ProcessingConfig:
        return self.configs.require(ProcessingConfig, step_index=self.index)

    @property
    def source_bindings(self) -> StepSourceBindingsConfig:
        if not self.is_function_step:
            return EMPTY_SOURCE_BINDINGS
        return self.configs.require(StepSourceBindingsConfig, step_index=self.index)

    @property
    def materialization_config(self) -> StepMaterializationConfig:
        return self.configs.require(StepMaterializationConfig, step_index=self.index)

    @property
    def callable_runtime_config_bindings(self) -> tuple[RuntimeParameterBinding, ...]:
        return self.configs.runtime_parameter_bindings()

    @property
    def well_filters(self) -> tuple[StepWellFilterSnapshot, ...]:
        return tuple(
            StepWellFilterSnapshot(config=config)
            for config in self.configs.instances_of(WellFilterConfig)
            if config.well_filter is not None
        )


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
