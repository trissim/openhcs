"""CellProfiler-compatible RunImageJMacro backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from openhcs.core.aligned_image_payload import (
    AlignedImageSliceContext,
    AlignedImageStack,
    pack_aligned_image_outputs,
)
from openhcs.core.artifacts import (
    ArtifactSpecCollection,
    ArtifactSpecRelation,
    GroupLineageSourceRelation,
    ImageArtifactType,
)
from openhcs.core.memory.decorators import numpy
from openhcs.core.config import FijiStreamingConfig
from openhcs.core.pipeline.function_contracts import composed_image_payload
from openhcs.core.runtime_image_values import image_payload_data
from openhcs.core.vfs_protocol import PlateInputFile
from openhcs.interop.cellprofiler.module_settings import (
    BoundModuleSettings,
)
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.setting_names import setting_values
from openhcs.interop.cellprofiler.settings_binder import SettingToKeywordBinding
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.runtime.fiji_macro_runtime import FijiMacroExecutionRequest

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.parser import ModuleBlock


@composed_image_payload
@numpy(contract=ProcessingContract.FLEXIBLE)
def run_imagej_macro(
    image: np.ndarray,
    macro_path: PlateInputFile = "macro.ijm",
    input_filenames: tuple[str, ...] = ("input_0.tiff",),
    output_filenames: tuple[str, ...] = ("output_0.tiff",),
    output_image_names: tuple[str, ...] = ("ImageJOutput",),
    directory_variable: str = "Directory",
    macro_variables: Mapping[str, str] | None = None,
    fiji_streaming_config: FijiStreamingConfig = FijiStreamingConfig(),
) -> np.ndarray | AlignedImageStack:
    """Execute a macro in the managed Fiji runtime and return declared outputs.

    Args:
        macro_path: Plate-relative path to the ImageJ macro file to execute.
        macro_variables: Optional variable names and string values injected before
            the macro runs.
    """

    if not input_filenames:
        raise ValueError("RunImagejMacro requires at least one input image group.")
    if not output_filenames:
        raise ValueError("RunImagejMacro requires at least one output image group.")
    if len(output_filenames) != len(output_image_names):
        raise ValueError(
            "RunImagejMacro output filenames and image names must have identical "
            f"cardinality, got {len(output_filenames)} and "
            f"{len(output_image_names)}."
        )
    if any(not name.strip() for name in output_image_names):
        raise ValueError("RunImagejMacro output image names cannot be blank.")

    image_data = np.asarray(image_payload_data(image))
    if len(input_filenames) > 1 and image_data.ndim < 3:
        raise ValueError(
            "RunImagejMacro with multiple input groups requires an explicit "
            f"leading image axis, got shape {image_data.shape!r}."
        )
    input_images = (
        (image_data,)
        if len(input_filenames) == 1
        else tuple(image_data[index] for index in range(image_data.shape[0]))
    )
    if len(input_images) != len(input_filenames):
        raise ValueError(
            "RunImagejMacro requires one leading image plane per input filename; "
            f"got {len(input_images)} image(s) and {len(input_filenames)} filename(s)."
        )

    output_images = FijiMacroExecutionRequest.from_arrays(
        macro_path=macro_path,
        input_filenames=input_filenames,
        output_filenames=output_filenames,
        directory_variable=directory_variable,
        macro_variables={} if macro_variables is None else macro_variables,
        input_images=tuple(np.asarray(input_image) for input_image in input_images),
    ).send(fiji_streaming_config)

    return pack_aligned_image_outputs(
        output_images,
        slice_contexts=tuple(
            AlignedImageSliceContext.independent_main_flow(
                image_name,
                artifact_kind=ImageArtifactType.value,
            )
            for image_name in output_image_names
        ),
    )


class RunImagejMacroModule(
    CellProfilerModule,
):
    module_name = "RunImagejMacro"
    function_name = "run_imagej_macro"
    validated = True
    confidence = 0.95

    hidden_count_setting = "Hidden"
    executable_directory_setting = "Executable directory"
    executable_file_setting = "Executable"
    macro_directory_setting = "Macro directory"
    macro_file_setting = "Macro"
    directory_variable_setting = (
        "What variable in your macro defines the folder ImageJ should use?"
    )
    input_image_setting = "Select an image to send to your macro"
    input_filename_setting = "What should this image temporarily saved as?"
    output_filename_setting = "What is the image filename CellProfiler should load?"
    output_image_setting = "What should CellProfiler call the loaded image?"
    macro_variable_name_setting = "What variable name is your macro expecting?"
    macro_variable_value_setting = "What value should this variable have?"

    input_image_binding = SettingToKeywordBinding.input(
        input_image_setting,
        ImageArtifactType,
        repeated=True,
    )
    output_image_binding = SettingToKeywordBinding.output(
        output_image_setting,
        ImageArtifactType,
        "output_image_names",
        repeated=True,
    )
    setting_bindings = (
        input_image_binding,
        output_image_binding,
        SettingToKeywordBinding(
            input_filename_setting,
            "input_filenames",
            repeated=True,
        ),
        SettingToKeywordBinding(
            output_filename_setting,
            "output_filenames",
            repeated=True,
        ),
        SettingToKeywordBinding(
            directory_variable_setting,
            "directory_variable",
        ),
    )
    ignored_settings = (
        hidden_count_setting,
        executable_directory_setting,
        executable_file_setting,
        macro_directory_setting,
        macro_file_setting,
        macro_variable_name_setting,
        macro_variable_value_setting,
    )

    @dataclass(frozen=True, slots=True)
    class InputGroup:
        image_name: str
        filename: str

    @dataclass(frozen=True, slots=True)
    class OutputGroup:
        filename: str
        image_name: str

    @classmethod
    def input_groups(cls, module: "ModuleBlock") -> tuple[InputGroup, ...]:
        image_names = setting_values(module, cls.input_image_setting)
        filenames = setting_values(module, cls.input_filename_setting)
        if not image_names or len(image_names) != len(filenames):
            raise ValueError(
                f"RunImagejMacro({module.module_num}) requires one filename for "
                f"each input image, got {image_names!r} and {filenames!r}."
            )
        return tuple(
            cls.InputGroup(image_name, filename)
            for image_name, filename in zip(image_names, filenames, strict=True)
        )

    @classmethod
    def output_groups(cls, module: "ModuleBlock") -> tuple[OutputGroup, ...]:
        filenames = setting_values(module, cls.output_filename_setting)
        image_names = setting_values(module, cls.output_image_setting)
        if not image_names or len(image_names) != len(filenames):
            raise ValueError(
                f"RunImagejMacro({module.module_num}) requires one image name for "
                f"each output filename, got {filenames!r} and {image_names!r}."
            )
        return tuple(
            cls.OutputGroup(filename, image_name)
            for filename, image_name in zip(filenames, image_names, strict=True)
        )

    @classmethod
    def macro_variables(cls, module: "ModuleBlock") -> dict[str, str]:
        names = setting_values(module, cls.macro_variable_name_setting)
        values = setting_values(module, cls.macro_variable_value_setting)
        if len(names) != len(values):
            raise ValueError(
                f"RunImagejMacro({module.module_num}) macro variable names and "
                f"values do not pair exactly: {names!r}, {values!r}."
            )
        return dict(zip(names, values, strict=True))

    @classmethod
    def validate_hidden_counts(
        cls,
        module: "ModuleBlock",
        *,
        input_count: int,
        output_count: int,
        variable_count: int,
    ) -> None:
        count_literals = setting_values(module, cls.hidden_count_setting)
        if not count_literals:
            return
        if len(count_literals) != 3:
            raise ValueError(
                f"RunImagejMacro({module.module_num}) requires exactly three "
                f"Hidden count rows, got {count_literals!r}."
            )
        declared_counts = tuple(int(value) for value in count_literals)
        actual_counts = (input_count, output_count, variable_count)
        if declared_counts != actual_counts:
            raise ValueError(
                f"RunImagejMacro({module.module_num}) declares group counts "
                f"{declared_counts!r}, but parsed {actual_counts!r}."
            )

    @classmethod
    def postprocess_bound_settings(
        cls,
        module: "ModuleBlock",
        bound: BoundModuleSettings,
    ) -> BoundModuleSettings:
        input_groups = cls.input_groups(module)
        output_groups = cls.output_groups(module)
        macro_variables = cls.macro_variables(module)
        cls.validate_hidden_counts(
            module,
            input_count=len(input_groups),
            output_count=len(output_groups),
            variable_count=len(macro_variables),
        )
        return bound.with_kwargs(
            {
                "input_filenames": tuple(group.filename for group in input_groups),
                "output_filenames": tuple(group.filename for group in output_groups),
                "output_image_names": tuple(
                    group.image_name for group in output_groups
                ),
                "macro_variables": macro_variables,
            }
        ).with_consumed_settings(
            cls.hidden_count_setting,
            cls.input_image_setting,
            cls.input_filename_setting,
            cls.output_filename_setting,
            cls.output_image_setting,
            cls.macro_variable_name_setting,
            cls.macro_variable_value_setting,
        )

    @classmethod
    def artifact_contract_inputs(
        cls,
        module,
        *,
        invocation_key,
        step_context,
    ):
        input_groups = cls.input_groups(module)
        cls.validate_hidden_counts(
            module,
            input_count=len(input_groups),
            output_count=len(cls.output_groups(module)),
            variable_count=len(cls.macro_variables(module)),
        )
        return super().artifact_contract_inputs(
            module,
            invocation_key=invocation_key,
            step_context=step_context,
        )

    @classmethod
    def artifact_contract_outputs(
        cls,
        module,
        *,
        invocation_key,
        step_context,
        artifact_inputs,
    ):
        cls.output_groups(module)
        return super().artifact_contract_outputs(
            module,
            invocation_key=invocation_key,
            step_context=step_context,
            artifact_inputs=artifact_inputs,
        )

    @classmethod
    def artifact_output_relations(
        cls,
        module,
        *,
        invocation_key,
        step_context,
        binding,
        name,
        artifact_inputs: ArtifactSpecCollection,
        output_position: int,
    ) -> tuple[ArtifactSpecRelation, ...]:
        """Inherit invocation scope from the first declared macro input."""

        del cls, module, invocation_key, step_context, binding, name, output_position
        image_inputs = artifact_inputs.for_artifact_type(ImageArtifactType).specs
        if not image_inputs:
            raise ValueError("RunImagejMacro requires at least one image input.")
        return (GroupLineageSourceRelation(source=image_inputs[0].ref()),)
