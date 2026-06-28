"""Typed runtime view over compiled FunctionStep plans."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from openhcs.constants.constants import VALID_GPU_MEMORY_TYPES, VariableComponents
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.compiled_step_plan import (
    ArtifactInputPlans,
    ArtifactOutputPlans,
    CompiledStepPlan,
    InputConversionPlan,
    MaterializedOutputPlan,
    RuntimeArtifactMaterializationPlan,
    SequentialRuntimeFilterPlan,
)
from openhcs.core.config import StreamingConfig
from openhcs.core.function_patterns import CompiledFunctionPattern
from openhcs.core.source_bindings import CompiledSourceBindingPlan
from openhcs.core.source_bindings import CompiledSourceUniversePlan
from openhcs.core.source_load_plan import SourceLoadPlan
from openhcs.core.step_dependencies import StepInputDependency
from openhcs.core.steps.function_io import create_image_path_getter

logger = logging.getLogger(__name__)


AxisPathGetter = Callable[[str | Path, str], list[str]]


@dataclass(frozen=True)
class FunctionStepExecutionPlan:
    """Typed runtime snapshot of one compiled FunctionStep plan."""

    step_index: int
    step_scope_id: str | None
    step_name: str
    axis_id: str
    input_dir: Path
    output_dir: Path
    variable_components: Sequence[VariableComponents]
    group_by: Any
    sequential_filter_plan: SequentialRuntimeFilterPlan
    main_input_dependency: StepInputDependency
    source_binding_plan: CompiledSourceBindingPlan
    source_universe_plan: CompiledSourceUniversePlan
    source_load_plan: SourceLoadPlan
    artifact_inputs: ArtifactInputPlans
    artifact_outputs: ArtifactOutputPlans
    read_backend: str
    write_backend: str
    input_memory_type: str
    output_memory_type: str
    zarr_config: Mapping[str, Any] | None
    device_id: int | None
    get_paths_for_axis: AxisPathGetter
    pipeline_position: int
    output_plate_root: str
    sub_dir: str
    analysis_results_dir: str | None
    input_conversion: InputConversionPlan | None
    materialized_output: MaterializedOutputPlan | None
    runtime_artifact_materialization: RuntimeArtifactMaterializationPlan
    streaming_configs: tuple[StreamingConfig, ...]
    compiled_function_pattern: CompiledFunctionPattern
    artifact_inputs_by_group: Mapping[Any, ArtifactInputPlans]
    artifact_outputs_by_group: Mapping[Any, ArtifactOutputPlans]

    @classmethod
    def from_context(
        cls,
        context: ProcessingContext,
        step_index: int,
    ) -> "FunctionStepExecutionPlan":
        compiled_plan: CompiledStepPlan = context.step_plans[step_index]
        step_name = compiled_plan.step_name
        axis_id = compiled_plan.axis_id
        input_dir = _require_path(compiled_plan.input_dir, "input_dir", compiled_plan)
        output_dir = _require_path(compiled_plan.output_dir, "output_dir", compiled_plan)

        if not all([axis_id, input_dir, output_dir]):
            raise ValueError(f"Plan missing essential keys for step {step_index}")

        variable_components = compiled_plan.variable_components
        if variable_components is None:
            variable_components = [VariableComponents.SITE]
            logger.warning(
                "Step %s (%s) had None variable_components, using default [SITE]",
                step_index,
                step_name,
            )

        input_memory_type = _require_value(
            compiled_plan.input_memory_type,
            "input_memory_type",
            compiled_plan,
        )
        output_memory_type = _require_value(
            compiled_plan.output_memory_type,
            "output_memory_type",
            compiled_plan,
        )
        requires_gpu = (
            input_memory_type in VALID_GPU_MEMORY_TYPES
            or output_memory_type in VALID_GPU_MEMORY_TYPES
        )
        device_id = compiled_plan.gpu_id if requires_gpu else None

        get_paths_for_axis = create_image_path_getter(
            axis_id,
            context.filemanager,
            context.microscope_handler,
        )

        return cls(
            step_index=step_index,
            step_scope_id=compiled_plan.step_scope_id,
            step_name=step_name,
            axis_id=axis_id,
            input_dir=input_dir,
            output_dir=output_dir,
            variable_components=variable_components,
            group_by=compiled_plan.group_by,
            sequential_filter_plan=compiled_plan.sequential_filter_plan,
            main_input_dependency=compiled_plan.main_input_dependency,
            source_binding_plan=compiled_plan.source_binding_plan,
            source_universe_plan=compiled_plan.source_universe_plan,
            source_load_plan=compiled_plan.source_load_plan,
            artifact_inputs=compiled_plan.artifact_inputs,
            artifact_outputs=compiled_plan.artifact_outputs,
            read_backend=_require_value(compiled_plan.read_backend, "read_backend", compiled_plan),
            write_backend=_require_value(compiled_plan.write_backend, "write_backend", compiled_plan),
            input_memory_type=input_memory_type,
            output_memory_type=output_memory_type,
            zarr_config=compiled_plan.zarr_config,
            device_id=device_id,
            get_paths_for_axis=get_paths_for_axis,
            pipeline_position=compiled_plan.pipeline_position or step_index,
            output_plate_root=_require_value(
                compiled_plan.output_plate_root,
                "output_plate_root",
                compiled_plan,
            ),
            sub_dir=_require_value(compiled_plan.sub_dir, "sub_dir", compiled_plan),
            analysis_results_dir=compiled_plan.analysis_results_dir,
            input_conversion=compiled_plan.input_conversion,
            materialized_output=compiled_plan.materialized_output,
            runtime_artifact_materialization=(
                compiled_plan.runtime_artifact_materialization
            ),
            streaming_configs=tuple(compiled_plan.streaming_configs.values()),
            compiled_function_pattern=_require_value(
                compiled_plan.compiled_function_pattern,
                "compiled_function_pattern",
                compiled_plan,
            ),
            artifact_inputs_by_group=compiled_plan.artifact_inputs_by_group,
            artifact_outputs_by_group=compiled_plan.artifact_outputs_by_group,
        )

    @property
    def variable_component_values(self) -> list[str]:
        return [component.value for component in self.variable_components]

    @property
    def variable_component_names(self) -> list[str]:
        return [component.name for component in self.variable_components]

    @property
    def group_by_value(self) -> str | None:
        return self.group_by.value if self.group_by else None

    @property
    def group_by_name(self) -> str | None:
        return self.group_by.name if self.group_by else None

    @property
    def group_projects_runtime_plane(self) -> bool:
        """Return whether the current group axis is a runtime-slice stack axis."""
        group_by_value = self.group_by_value
        if group_by_value is None:
            return False
        return any(
            component.value == group_by_value
            for component in self.variable_components
        )

    @property
    def input_conversion_dir(self) -> Path:
        return self._require_input_conversion().output_dir

    @property
    def has_input_conversion(self) -> bool:
        return self.input_conversion is not None

    @property
    def input_conversion_backend(self) -> str:
        return self._require_input_conversion().backend

    @property
    def input_conversion_uses_virtual_workspace(self) -> bool:
        return self._require_input_conversion().uses_virtual_workspace

    @property
    def input_conversion_original_subdir(self) -> str:
        return self._require_input_conversion().original_subdir

    @property
    def has_materialized_output(self) -> bool:
        return self.materialized_output is not None

    @property
    def materialized_output_dir(self) -> Path:
        return self._require_materialized_output().output_dir

    @property
    def materialized_backend(self) -> str:
        return self._require_materialized_output().backend

    @property
    def materialized_plate_root(self) -> str:
        return self._require_materialized_output().plate_root

    @property
    def materialized_sub_dir(self) -> str:
        return self._require_materialized_output().sub_dir

    @property
    def materialized_analysis_results_dir(self) -> str | None:
        return self._require_materialized_output().analysis_results_dir

    @property
    def artifact_analysis_output_dir(self) -> Path:
        output_dir = (
            self.materialized_analysis_results_dir
            if self.has_materialized_output
            else self.analysis_results_dir
        )
        if output_dir is None:
            raise ValueError(
                f"Step {self.step_index} ({self.step_name}) has no analysis results directory."
            )
        return Path(output_dir)

    @property
    def artifact_images_dir(self) -> str:
        if self.has_materialized_output:
            return str(self.materialized_output_dir)
        return str(self.output_dir)

    def _require_input_conversion(self) -> InputConversionPlan:
        if self.input_conversion is None:
            raise ValueError(
                f"Step {self.step_index} ({self.step_name}) has no input conversion plan."
            )
        return self.input_conversion

    def _require_materialized_output(self) -> MaterializedOutputPlan:
        if self.materialized_output is None:
            raise ValueError(
                f"Step {self.step_index} ({self.step_name}) has no materialized output plan."
            )
        return self.materialized_output


def _require_path(
    value: Path | str | None,
    field_name: str,
    plan: CompiledStepPlan,
) -> Path:
    return Path(_require_value(value, field_name, plan))


def _require_value(value: Any, field_name: str, plan: CompiledStepPlan) -> Any:
    if value is None:
        raise ValueError(
            f"Compiled plan for step {plan.step_index} ({plan.step_name}) is missing {field_name}."
        )
    return value
