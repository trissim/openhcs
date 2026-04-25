"""Typed runtime view over compiled FunctionStep plans."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

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
class FunctionStepExecutionPlan:
    """Typed runtime projection of one compiled FunctionStep plan."""

    raw: MutableMapping[str, Any]
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

    @classmethod
    def from_context(
        cls,
        context: ProcessingContext,
        step_index: int,
    ) -> "FunctionStepExecutionPlan":
        raw = context.step_plans[step_index]
        step_name = raw["step_name"]
        axis_id = raw["axis_id"]
        input_dir = Path(raw["input_dir"])
        output_dir = Path(raw["output_dir"])

        if not all([axis_id, input_dir, output_dir]):
            raise ValueError(f"Plan missing essential keys for step {step_index}")

        variable_components = raw["variable_components"]
        if variable_components is None:
            variable_components = [VariableComponents.SITE]
            raw["variable_components"] = variable_components
            logger.warning(
                "Step %s (%s) had None variable_components, using default [SITE]",
                step_index,
                step_name,
            )

        input_memory_type = raw["input_memory_type"]
        output_memory_type = raw["output_memory_type"]
        requires_gpu = (
            input_memory_type in VALID_GPU_MEMORY_TYPES
            or output_memory_type in VALID_GPU_MEMORY_TYPES
        )
        device_id = raw["gpu_id"] if requires_gpu else None

        get_paths_for_axis = create_image_path_getter(
            axis_id,
            context.filemanager,
            context.microscope_handler,
        )
        raw["get_paths_for_axis"] = get_paths_for_axis

        return cls(
            raw=raw,
            step_index=step_index,
            step_name=step_name,
            axis_id=axis_id,
            input_dir=input_dir,
            output_dir=output_dir,
            variable_components=variable_components,
            group_by=raw["group_by"],
            func=raw["func"],
            artifact_inputs=raw["artifact_inputs"],
            artifact_outputs=raw["artifact_outputs"],
            read_backend=raw["read_backend"],
            write_backend=raw["write_backend"],
            input_memory_type=input_memory_type,
            output_memory_type=output_memory_type,
            zarr_config=raw["zarr_config"],
            device_id=device_id,
            get_paths_for_axis=get_paths_for_axis,
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
        return Path(self.raw["input_conversion_dir"])

    @property
    def has_input_conversion(self) -> bool:
        return "input_conversion_dir" in self.raw

    @property
    def input_conversion_backend(self) -> str:
        return self.raw["input_conversion_backend"]

    @property
    def input_conversion_uses_virtual_workspace(self) -> bool:
        return self.raw["input_conversion_uses_virtual_workspace"]

    @property
    def input_conversion_original_subdir(self) -> str:
        return self.raw["input_conversion_original_subdir"]

    @property
    def pipeline_position(self) -> int:
        return self.raw["pipeline_position"]

    @property
    def output_plate_root(self) -> str:
        return self.raw["output_plate_root"]

    @property
    def sub_dir(self) -> str:
        return self.raw["sub_dir"]

    @property
    def analysis_results_dir(self) -> str | None:
        return self.raw.get("analysis_results_dir")

    @property
    def has_materialized_output(self) -> bool:
        return "materialized_output_dir" in self.raw

    @property
    def materialized_output_dir(self) -> Path:
        return Path(self.raw["materialized_output_dir"])

    @property
    def materialized_backend(self) -> str:
        return self.raw["materialized_backend"]

    @property
    def materialized_plate_root(self) -> str:
        return self.raw["materialized_plate_root"]

    @property
    def materialized_sub_dir(self) -> str:
        return self.raw["materialized_sub_dir"]

    @property
    def materialized_analysis_results_dir(self) -> str | None:
        return self.raw.get("materialized_analysis_results_dir")

    @property
    def artifact_analysis_output_dir(self) -> Path:
        key = (
            "materialized_analysis_results_dir"
            if self.has_materialized_output
            else "analysis_results_dir"
        )
        return Path(self.raw[key])

    @property
    def artifact_images_dir(self) -> str:
        if self.has_materialized_output:
            return str(self.materialized_output_dir)
        return str(self.output_dir)

    @property
    def streaming_configs(self) -> list[Any]:
        from openhcs.core.config import StreamingConfig

        return [
            config
            for config in self.raw.values()
            if isinstance(config, StreamingConfig)
        ]

    @property
    def function_invocation_plans(
        self,
    ) -> Mapping[FunctionInvocationKey, FunctionInvocationPlan]:
        return self.raw.get("function_invocation_plans", {})

    @property
    def artifact_inputs_by_group(self) -> Mapping[Any, ArtifactInputPlans]:
        return self.raw.get("artifact_inputs_by_group", {})

    @property
    def artifact_outputs_by_group(self) -> Mapping[Any, ArtifactOutputPlans]:
        return self.raw.get("artifact_outputs_by_group", {})
