"""Typed runtime view over compiled FunctionStep plans."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from openhcs.constants.constants import VALID_GPU_MEMORY_TYPES, VariableComponents
from openhcs.core.artifacts import ArtifactInputPlan, ArtifactOutputPlan
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.function_patterns import FunctionInvocationKey, FunctionInvocationPlan
from openhcs.core.steps.function_io import create_image_path_getter


logger = logging.getLogger(__name__)


AxisPathGetter = Callable[[str | Path, str], list[str]]
ArtifactInputPlans = Mapping[str, ArtifactInputPlan]
ArtifactOutputPlans = Mapping[str, ArtifactOutputPlan]


@dataclass(frozen=True)
class InputConversionPlan:
    """Typed input-conversion section of a FunctionStep plan."""

    output_dir: Path
    backend: str
    uses_virtual_workspace: bool
    original_subdir: str


@dataclass(frozen=True)
class MaterializedOutputPlan:
    """Typed materialized-output section of a FunctionStep plan."""

    output_dir: Path
    backend: str
    plate_root: str
    sub_dir: str
    analysis_results_dir: str | None


@dataclass(frozen=True)
class FunctionStepExecutionPlan:
    """Typed runtime snapshot of one compiled FunctionStep plan."""

    step_index: int
    step_name: str
    axis_id: str
    input_dir: Path
    output_dir: Path
    variable_components: Sequence[VariableComponents]
    group_by: Any
    func: Any
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
    streaming_configs: tuple[Any, ...]
    function_invocation_plans: Mapping[FunctionInvocationKey, FunctionInvocationPlan]
    artifact_inputs_by_group: Mapping[Any, ArtifactInputPlans]
    artifact_outputs_by_group: Mapping[Any, ArtifactOutputPlans]

    @classmethod
    def from_context(
        cls,
        context: ProcessingContext,
        step_index: int,
    ) -> "FunctionStepExecutionPlan":
        compiled_plan = context.step_plans[step_index]
        step_name = compiled_plan["step_name"]
        axis_id = compiled_plan["axis_id"]
        input_dir = Path(compiled_plan["input_dir"])
        output_dir = Path(compiled_plan["output_dir"])

        if not all([axis_id, input_dir, output_dir]):
            raise ValueError(f"Plan missing essential keys for step {step_index}")

        variable_components = compiled_plan["variable_components"]
        if variable_components is None:
            variable_components = [VariableComponents.SITE]
            logger.warning(
                "Step %s (%s) had None variable_components, using default [SITE]",
                step_index,
                step_name,
            )

        input_memory_type = compiled_plan["input_memory_type"]
        output_memory_type = compiled_plan["output_memory_type"]
        requires_gpu = (
            input_memory_type in VALID_GPU_MEMORY_TYPES
            or output_memory_type in VALID_GPU_MEMORY_TYPES
        )
        device_id = compiled_plan["gpu_id"] if requires_gpu else None

        get_paths_for_axis = create_image_path_getter(
            axis_id,
            context.filemanager,
            context.microscope_handler,
        )

        return cls(
            step_index=step_index,
            step_name=step_name,
            axis_id=axis_id,
            input_dir=input_dir,
            output_dir=output_dir,
            variable_components=variable_components,
            group_by=compiled_plan["group_by"],
            func=compiled_plan["func"],
            artifact_inputs=compiled_plan["artifact_inputs"],
            artifact_outputs=compiled_plan["artifact_outputs"],
            read_backend=compiled_plan["read_backend"],
            write_backend=compiled_plan["write_backend"],
            input_memory_type=input_memory_type,
            output_memory_type=output_memory_type,
            zarr_config=compiled_plan["zarr_config"],
            device_id=device_id,
            get_paths_for_axis=get_paths_for_axis,
            pipeline_position=compiled_plan.get("pipeline_position", step_index),
            output_plate_root=compiled_plan["output_plate_root"],
            sub_dir=compiled_plan["sub_dir"],
            analysis_results_dir=compiled_plan.get("analysis_results_dir"),
            input_conversion=_input_conversion_from_mapping(compiled_plan, input_dir),
            materialized_output=_materialized_output_from_mapping(compiled_plan),
            streaming_configs=_streaming_configs_from_mapping(compiled_plan),
            function_invocation_plans=compiled_plan.get("function_invocation_plans", {}),
            artifact_inputs_by_group=compiled_plan.get("artifact_inputs_by_group", {}),
            artifact_outputs_by_group=compiled_plan.get("artifact_outputs_by_group", {}),
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


def _input_conversion_from_mapping(
    compiled_plan: Mapping[str, Any],
    input_dir: Path,
) -> InputConversionPlan | None:
    if "input_conversion_dir" not in compiled_plan:
        return None

    return InputConversionPlan(
        output_dir=Path(compiled_plan["input_conversion_dir"]),
        backend=compiled_plan["input_conversion_backend"],
        uses_virtual_workspace=bool(
            compiled_plan.get("input_conversion_uses_virtual_workspace", False)
        ),
        original_subdir=compiled_plan.get(
            "input_conversion_original_subdir",
            input_dir.name,
        ),
    )


def _materialized_output_from_mapping(
    compiled_plan: Mapping[str, Any],
) -> MaterializedOutputPlan | None:
    if "materialized_output_dir" not in compiled_plan:
        return None

    return MaterializedOutputPlan(
        output_dir=Path(compiled_plan["materialized_output_dir"]),
        backend=compiled_plan["materialized_backend"],
        plate_root=compiled_plan["materialized_plate_root"],
        sub_dir=compiled_plan["materialized_sub_dir"],
        analysis_results_dir=compiled_plan.get("materialized_analysis_results_dir"),
    )


def _streaming_configs_from_mapping(compiled_plan: Mapping[str, Any]) -> tuple[Any, ...]:
    from openhcs.core.config import StreamingConfig

    return tuple(
        config for config in compiled_plan.values() if isinstance(config, StreamingConfig)
    )
