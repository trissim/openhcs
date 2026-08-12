"""Executable CellProfiler-compatible SaveImages declaration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Annotated, ClassVar, TypeVar
from urllib.parse import quote

import numpy as np
from python_introspect import set_signature_analysis_target

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSidecarRole,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRelation,
    GroupLineageSourceRelation,
    ImageArtifactType,
    MaterializationSourceIdentityRelation,
    MeasurementsArtifactType,
)
from openhcs.core.image_file_serialization import image_payload_as_uint8
from openhcs.core.memory import numpy
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
)
from openhcs.core.pipeline.function_contracts import (
    runtime_bound_parameters,
    special_inputs,
)
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.runtime_batch_contracts import SliceIndexRuntimeParameter
from openhcs.core.runtime_image_values import (
    image_payload_data,
    image_payload_metadata,
    with_image_payload_data,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    MeasurementRowValueField,
    MeasurementScope,
    MeasurementSubject,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    MeasurementArtifactOutputModule,
    PlaneRuntimeArtifactModule,
)
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    optional_setting_value,
    setting_values,
    split_symbol_names,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    cellprofiler_enum_setting_parser,
    coerce_cellprofiler_enum,
    parse_cellprofiler_bool,
    parse_cellprofiler_int,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
)
from openhcs.processing.materialization import (
    ImageFileOptions,
    MaterializationSpec,
    MaterializedFilenameIdentity,
)

if TYPE_CHECKING:
    from openhcs.core.function_patterns import (
        FunctionInvocationKey,
        NormalizedFunctionItem,
    )
    from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
    from openhcs.interop.cellprofiler.parser import ModuleBlock


class SaveImagesImageKind(str, Enum):
    """Image payload kind supported by the OpenHCS SaveImages callable."""

    IMAGE = "image"


class SaveImagesFilenameMethod(str, Enum):
    """CellProfiler filename construction modes supported by SaveImages."""

    FROM_IMAGE_FILENAME = "from_image_filename"
    SEQUENTIAL_NUMBERS = "sequential_numbers"
    SINGLE_NAME = "single_name"

    def relative_path_template(
        self,
        *,
        directory: str | None,
        single_name: str,
        sequential_prefix: str,
        sequential_digits: int,
        suffix: str,
        file_format: "SaveImagesFileFormat",
    ) -> str | None:
        """Return the generic relative template for this naming mode."""

        if self is SaveImagesFilenameMethod.FROM_IMAGE_FILENAME:
            return None
        if self is SaveImagesFilenameMethod.SINGLE_NAME:
            filename = f"{single_name}{suffix}{file_format.value}"
        else:
            filename = (
                f"{sequential_prefix}{{index:0{sequential_digits}d}}"
                f"{suffix}{file_format.value}"
            )
        return _relative_template(directory, filename)


class SaveImagesFileFormat(str, Enum):
    """SaveImages file suffixes delegated to generic serialization formats."""

    PNG = ".png"
    TIFF = ".tiff"
    NPY = ".npy"
    JPEG = ".jpeg"
    HDF5 = ".h5"


class SaveImagesBitDepth(str, Enum):
    """Pixel conversion performed on the materialized SaveImages copy."""

    NATIVE = "native"
    UINT8 = "uint8"
    UINT16 = "uint16"
    FLOAT32 = "float32"

    def convert(self, payload: RuntimeArrayData) -> RuntimeArrayData:
        """Return a converted payload while retaining image provenance."""

        data = np.asarray(image_payload_data(payload))
        if self is SaveImagesBitDepth.NATIVE:
            converted = data
        elif self is SaveImagesBitDepth.UINT8:
            converted = image_payload_as_uint8(data)
        elif self is SaveImagesBitDepth.UINT16:
            converted = _image_payload_as_uint16(data)
        else:
            converted = data.astype(np.float32, copy=False)
        return with_image_payload_data(payload, converted)


class SaveImagesWhen(str, Enum):
    """Cycle selection declared by a SaveImages module."""

    EVERY_CYCLE = "every_cycle"
    FIRST_CYCLE = "first_cycle"
    LAST_CYCLE = "last_cycle"


class SaveImagesSeriesAxis(str, Enum):
    """Source component used by sequential SaveImages filenames."""

    TIMEPOINT = "timepoint"
    Z_INDEX = "z_index"


class SaveImagesOutputLocation(str, Enum):
    """Output-directory roots representable by generic materialization."""

    DEFAULT_OUTPUT_FOLDER = "default_output_folder"
    DEFAULT_OUTPUT_SUBFOLDER = "default_output_subfolder"

    @classmethod
    def relative_directory_from_literal(cls, value: str | None) -> str | None:
        """Lower one CellProfiler output location to a relative directory."""

        if value is None:
            return None
        normalized = value.strip()
        if not normalized or normalized.lower() == "none":
            return None
        if "|" not in normalized:
            return normalized
        location_literal, directory_literal = normalized.split("|", 1)
        location = coerce_cellprofiler_enum(cls, location_literal)
        directory = directory_literal.strip()
        if location is cls.DEFAULT_OUTPUT_FOLDER:
            if directory.lower() in {"", ".", "none"}:
                return None
            raise ValueError(
                "SaveImages default output-folder settings cannot also declare "
                f"a relative directory: {value!r}."
            )
        if not directory or directory.lower() == "none":
            raise ValueError(
                "SaveImages default output sub-folder requires a relative path."
            )
        return directory


SaveImagesImageKind.IMAGE.cellprofiler_literals = ("Image",)
SaveImagesFilenameMethod.FROM_IMAGE_FILENAME.cellprofiler_literals = (
    "From image filename",
)
SaveImagesFilenameMethod.SEQUENTIAL_NUMBERS.cellprofiler_literals = (
    "Sequential numbers",
)
SaveImagesFilenameMethod.SINGLE_NAME.cellprofiler_literals = ("Single name",)
SaveImagesFileFormat.PNG.cellprofiler_literals = ("png",)
SaveImagesFileFormat.TIFF.cellprofiler_literals = ("tif", "tiff")
SaveImagesFileFormat.NPY.cellprofiler_literals = ("npy",)
SaveImagesFileFormat.JPEG.cellprofiler_literals = ("jpg", "jpeg")
SaveImagesFileFormat.HDF5.cellprofiler_literals = ("h5", "hdf5")
SaveImagesBitDepth.NATIVE.cellprofiler_literals = ("Raw", "raw")
SaveImagesBitDepth.UINT8.cellprofiler_literals = ("8-bit integer",)
SaveImagesBitDepth.UINT16.cellprofiler_literals = ("16-bit integer",)
SaveImagesBitDepth.FLOAT32.cellprofiler_literals = ("32-bit floating point",)
SaveImagesWhen.EVERY_CYCLE.cellprofiler_literals = ("Every cycle",)
SaveImagesWhen.FIRST_CYCLE.cellprofiler_literals = ("First cycle",)
SaveImagesWhen.LAST_CYCLE.cellprofiler_literals = ("Last cycle",)
SaveImagesSeriesAxis.TIMEPOINT.cellprofiler_literals = ("T (Time)",)
SaveImagesSeriesAxis.Z_INDEX.cellprofiler_literals = ("Z (Slice)",)
SaveImagesOutputLocation.DEFAULT_OUTPUT_FOLDER.cellprofiler_literals = (
    "Default Output Folder",
)
SaveImagesOutputLocation.DEFAULT_OUTPUT_SUBFOLDER.cellprofiler_literals = (
    "Default Output Folder sub-folder",
)


EnumT = TypeVar("EnumT", bound=Enum)


def _enum_setting(
    module: "ModuleBlock",
    setting: str | SettingNameFamily,
    enum_type: type[EnumT],
    default: EnumT,
) -> EnumT:
    value = optional_setting_value(module, setting)
    if value is None:
        return default
    return coerce_cellprofiler_enum(enum_type, value)


def _bool_setting(
    module: "ModuleBlock",
    setting: str | SettingNameFamily,
    default: bool,
) -> bool:
    value = optional_setting_value(module, setting)
    return default if value is None else parse_cellprofiler_bool(value)


def _int_setting(
    module: "ModuleBlock",
    setting: str | SettingNameFamily,
    default: int,
) -> int:
    value = optional_setting_value(module, setting)
    return default if value is None else parse_cellprofiler_int(value)


def _optional_text(value: str) -> str | None:
    normalized = value.strip()
    return None if normalized.lower() in {"", "none"} else normalized


def _parse_output_location_setting(value: str) -> str | None:
    return SaveImagesOutputLocation.relative_directory_from_literal(value)


def _relative_template(directory: str | None, filename: str) -> str:
    if not filename:
        raise ValueError("SaveImages filename cannot be empty.")
    if directory is None:
        return filename
    return str(PurePosixPath(directory) / filename)


def _image_payload_as_uint16(payload: object) -> np.ndarray:
    array = np.asarray(payload)
    if array.dtype == np.uint16:
        return array
    if array.dtype == np.bool_:
        return array.astype(np.uint16) * np.uint16(65535)
    if np.issubdtype(array.dtype, np.integer):
        minimum = int(array.min(initial=0))
        maximum = int(array.max(initial=0))
        if minimum >= 0 and maximum <= 1:
            converted = array.astype(np.uint16, copy=False)
            converted *= np.uint16(65535)
            return converted
        if minimum >= 0 and maximum <= np.iinfo(np.uint16).max:
            return array.astype(np.uint16, copy=False)
        return np.clip(array, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    values = array.astype(np.float64, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0 or (float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0):
        values = values * 65535.0
    sanitized = np.nan_to_num(values, nan=0.0, posinf=65535.0, neginf=0.0)
    return np.rint(np.clip(sanitized, 0.0, 65535.0)).astype(np.uint16)


class SaveImagesRecordedMeasurementFeature(str, Enum):
    """CellProfiler image features emitted when file recording is enabled."""

    FILE_NAME = "FileName"
    PATH_NAME = "PathName"
    URL = "URL"

    def feature_name(self, image_name: str) -> str:
        normalized = image_name.strip()
        if not normalized:
            raise ValueError("SaveImages recorded image name cannot be empty.")
        return f"{self.value}_{normalized}"


class SaveImagesRecordedMeasurementSourceRelation(ArtifactSpecRelation):
    """Identify the exact saved image named by recorded file measurements."""

    relation_key: ClassVar[str] = "save_images_recorded_measurement_source"
    target_artifact_type = MeasurementsArtifactType

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.source.artifact_type is not ImageArtifactType:
            raise ValueError(
                "SaveImages recorded measurements require an image source, got "
                f"{self.source.artifact_type.value}:{self.source.name}."
            )

    def measurement_subject(self) -> MeasurementSubject:
        """Declare the saved image as the owner of recorded file measurements."""

        return MeasurementSubject(MeasurementScope.IMAGE, self.source.name)


@dataclass(frozen=True, slots=True)
class SaveImagesRecordedMeasurementRow:
    """One image-scoped filename, pathname, or URL measurement row."""

    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
    feature_name: Annotated[str, MeasurementRowAxisField.FEATURE_NAME]
    result_value: Annotated[str, MeasurementRowValueField.RESULT_VALUE]


class SaveImagesModule(
    PlaneRuntimeArtifactModule,
    MeasurementArtifactOutputModule,
    CellProfilerModule,
):
    """Axis-scoped image conversion and generic file materialization."""

    module_name = "SaveImages"
    function_name = "save_images"
    function_variants = ("save_images_with_measurements",)
    validated = True
    confidence = 1.0

    image_kind_setting = SettingNameFamily("Select the type of image to save")
    source_image_setting = SettingNameFamily("Select the image to save")
    filename_method_setting = SettingNameFamily(
        "Select method for constructing file names"
    )
    filename_source_image_setting = SettingNameFamily(
        "Select image name for file prefix"
    )
    single_file_name_setting = SettingNameFamily(
        "Enter single file name",
        aliases=("Enter file prefix",),
    )
    number_of_digits_setting = SettingNameFamily("Number of digits")
    append_suffix_setting = SettingNameFamily("Append a suffix to the image file name?")
    filename_suffix_setting = SettingNameFamily("Text to append to the image name")
    file_format_setting = SettingNameFamily("Saved file format")
    output_location_setting = SettingNameFamily("Output file location")
    bit_depth_setting = SettingNameFamily("Image bit depth")
    overwrite_setting = SettingNameFamily("Overwrite existing files without warning?")
    when_to_save_setting = SettingNameFamily("When to save")
    record_file_setting = SettingNameFamily(
        "Record the file and path information to the saved image?"
    )
    create_subfolders_setting = SettingNameFamily(
        "Create subfolders in the output folder?"
    )
    base_image_folder_setting = SettingNameFamily("Base image folder")
    series_axis_setting = SettingNameFamily("How to save the series")
    lossless_compression_setting = SettingNameFamily("Save with lossless compression?")
    materialized_image_setting = SettingNameFamily(
        "OpenHCS materialized image artifact"
    )

    filename_source_image_binding = SettingToKeywordBinding.input(
        filename_source_image_setting,
        ImageArtifactType,
        parse=_optional_text,
    )
    selected_image_binding = SettingToKeywordBinding.input(
        source_image_setting,
        ImageArtifactType,
        runtime_parameter_name="image_to_save",
    )
    materialized_image_binding = SettingToKeywordBinding.output(
        materialized_image_setting,
        ImageArtifactType,
        "materialized_image_artifact_name",
    )
    setting_bindings = (
        filename_source_image_binding,
        selected_image_binding,
        materialized_image_binding,
        SettingToKeywordBinding(
            image_kind_setting,
            "image_kind",
            cellprofiler_enum_setting_parser(SaveImagesImageKind),
        ),
        SettingToKeywordBinding(
            filename_method_setting,
            "filename_method",
            cellprofiler_enum_setting_parser(SaveImagesFilenameMethod),
        ),
        SettingToKeywordBinding(single_file_name_setting, "single_file_name"),
        SettingToKeywordBinding(
            number_of_digits_setting,
            "number_of_digits",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            append_suffix_setting,
            "append_suffix",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(filename_suffix_setting, "filename_suffix"),
        SettingToKeywordBinding(
            file_format_setting,
            "file_format",
            cellprofiler_enum_setting_parser(SaveImagesFileFormat),
        ),
        SettingToKeywordBinding(
            output_location_setting,
            "output_location",
            _parse_output_location_setting,
        ),
        SettingToKeywordBinding(
            bit_depth_setting,
            "bit_depth",
            cellprofiler_enum_setting_parser(SaveImagesBitDepth),
        ),
        SettingToKeywordBinding(
            overwrite_setting,
            "overwrite",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            when_to_save_setting,
            "when_to_save",
            cellprofiler_enum_setting_parser(SaveImagesWhen),
        ),
        SettingToKeywordBinding(
            record_file_setting,
            "record_file_and_path",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            create_subfolders_setting,
            "create_subfolders",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            base_image_folder_setting,
            "base_image_folder",
            _optional_text,
        ),
        SettingToKeywordBinding(
            series_axis_setting,
            "series_axis",
            cellprofiler_enum_setting_parser(SaveImagesSeriesAxis),
        ),
        SettingToKeywordBinding(
            lossless_compression_setting,
            "lossless_compression",
            parse_cellprofiler_bool,
        ),
    )

    @classmethod
    def uses_cellprofiler_runtime_adapter(cls) -> bool:
        """SaveImages is a generic OpenHCS array callable."""

        return False

    @classmethod
    def records_file_and_path(cls, module: "ModuleBlock") -> bool:
        """Return whether this declaration owns the recorded measurement port."""

        return _bool_setting(module, cls.record_file_setting, False)

    @classmethod
    def derives_missing_output_identity(
        cls,
        binding: SettingToKeywordBinding,
    ) -> bool:
        """Defer the materialized image identity until its block is numbered."""

        return (
            binding is not cls.materialized_image_binding
            and super().derives_missing_output_identity(binding)
        )

    @classmethod
    def postprocess_bound_settings(cls, module, bound):
        """Carry the selected image identity into the active row-producing callable."""

        bound = super().postprocess_bound_settings(module, bound)
        if not cls.records_file_and_path(module):
            return bound
        selected_names = cls.artifact_names_for_binding(
            module,
            cls.selected_image_binding,
        )
        if len(selected_names) != 1:
            raise ValueError(
                f"SaveImages({module.module_num}) requires exactly one recorded "
                f"image identity, got {selected_names!r}."
            )
        return bound.with_kwargs({"saved_image_name": selected_names[0]})

    @classmethod
    def resolve_function(
        cls,
        module: "ModuleBlock",
        *,
        contract,
        source_bindings,
    ):
        """Select the callable whose return ABI matches file-recording topology."""

        if cls.records_file_and_path(module):
            return cls.require_callable(cls.function_variants[0])
        return super().resolve_function(
            module,
            contract=contract,
            source_bindings=source_bindings,
        )

    @classmethod
    def _derived_identity_setting_records(
        cls,
        *,
        invocation,
        block_position,
        existing_records,
        step_context,
    ):
        """Reconstruct file-recording topology from the public callable variant."""

        from openhcs.interop.cellprofiler.parser import ModuleSetting

        setting_key = cls.normalize_setting_name(cls.record_file_setting.canonical)
        own_records = (
            ()
            if setting_key in cls._normalized_record_setting_names(existing_records)
            else (
                ModuleSetting(
                    cls.record_file_setting.canonical,
                    (
                        "Yes"
                        if invocation.contract.function_name == cls.function_variants[0]
                        else "No"
                    ),
                ),
            )
        )
        return (
            *own_records,
            *super()._derived_identity_setting_records(
                invocation=invocation,
                block_position=block_position,
                existing_records=(*existing_records, *own_records),
                step_context=step_context,
            ),
        )

    @classmethod
    def finalize_module_blocks_for_invocation(
        cls,
        blocks, *,
        invocation: "NormalizedFunctionItem",
        step_context: "ArtifactDeclarationStepContext",
    ) -> tuple[ModuleBlock, ...]:
        """Select the exact saved-image source named by the active public callable."""

        blocks = super().finalize_module_blocks_for_invocation(
            blocks, invocation=invocation,
            step_context=step_context,
        )
        if invocation.contract.function_name != cls.function_variants[0]:
            return blocks
        if "saved_image_name" not in invocation.kwargs_dict:
            raise ValueError(
                "save_images_with_measurements requires public saved_image_name "
                "identity."
            )
        saved_image_name = str(invocation.kwargs_dict["saved_image_name"]).strip()
        matching_blocks = tuple(
            block
            for block in blocks
            if cls.artifact_names_for_binding(
                block,
                cls.selected_image_binding,
            )
            == (saved_image_name,)
        )
        if len(matching_blocks) != 1:
            raise ValueError(
                "save_images_with_measurements requires one exact selected image "
                f"matching saved_image_name={saved_image_name!r}, got "
                f"{tuple(cls.artifact_names_for_binding(block, cls.selected_image_binding) for block in blocks)!r}."
            )
        return matching_blocks

    @classmethod
    def active_artifact_bindings(
        cls,
        module: "ModuleBlock | None" = None,
        *,
        invocation_key: "FunctionInvocationKey | None" = None,
    ) -> tuple[SettingToKeywordBinding, ...]:
        """Declare the filename source only when source-derived naming is active."""

        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        filename_method = _enum_setting(
            module,
            cls.filename_method_setting,
            SaveImagesFilenameMethod,
            SaveImagesFilenameMethod.FROM_IMAGE_FILENAME,
        )
        if filename_method is SaveImagesFilenameMethod.FROM_IMAGE_FILENAME:
            filename_source = optional_setting_value(
                module,
                cls.filename_source_image_setting,
            )
            normalized_filename_source = (
                None if filename_source is None else _optional_text(filename_source)
            )
            if (
                normalized_filename_source is not None
                and normalized_filename_source
                not in cls.artifact_names_for_binding(
                    module,
                    cls.selected_image_binding,
                )
            ):
                return bindings
        return tuple(
            binding
            for binding in bindings
            if binding is not cls.filename_source_image_binding
        )

    @classmethod
    def _available_artifact_input_specs(
        cls,
        *,
        binding: SettingToKeywordBinding,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
    ) -> tuple[ArtifactSpec, ...]:
        """Preserve every scoped image choice when selection is omitted."""

        producer_specs = super()._available_artifact_input_specs(
            binding=binding,
            invocation_key=invocation_key,
            step_context=step_context,
        )
        if binding is not cls.selected_image_binding:
            return producer_specs
        return ArtifactSpecCollection(
            spec.for_plan_type(ArtifactInputPlan)
            for spec in (
                *step_context.main_flow_artifacts.of_artifact_type(
                    binding.require_artifact_type()
                ),
                *producer_specs,
            )
            if spec.sidecar_role is None
        ).unique(conflict_context="SaveImages selectable image")

    @classmethod
    def main_flow_output_specs(
        cls,
        main_flow_candidates: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        """The converted export copy never replaces ordinary main flow."""

        del cls, main_flow_candidates
        return ()

    @classmethod
    def artifact_contract_outputs(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        artifact_inputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpec, ...]:
        """Declare the materialized image artifact absent from CellProfiler settings."""

        selected_image_names = cls.artifact_names_for_binding(
            module,
            cls.selected_image_binding,
        )
        if len(selected_image_names) != 1:
            raise ValueError(
                f"SaveImages({module.module_num}) requires exactly one selected "
                f"image, got {selected_image_names!r}."
            )
        selected_image = artifact_inputs.require_by_name_and_artifact_type(
            selected_image_names[0],
            ImageArtifactType,
        )
        names = tuple(
            name
            for value in setting_values(module, cls.materialized_image_setting)
            for name in split_symbol_names(value)
        )
        if len(names) > 1:
            raise ValueError(
                f"SaveImages({module.module_num}) declares multiple materialized "
                f"image identities: {names!r}."
            )
        output_name = (
            names[0]
            if names
            else cls.canonical_numbered_module_output_artifact_name(
                module,
                artifact_type=ImageArtifactType,
                output_position=0,
            )
        )
        filename_source_names = (
            cls.artifact_names_for_binding(
                module,
                cls.filename_source_image_binding,
            )
            if cls.filename_source_image_binding
            in cls.artifact_bindings_for(
                module,
                invocation_key=invocation_key,
                plan_type=ArtifactInputPlan,
                artifact_type=ImageArtifactType,
            )
            else ()
        )
        if len(filename_source_names) > 1:
            raise ValueError(
                f"SaveImages({module.module_num}) declares multiple filename "
                f"source images: {filename_source_names!r}."
            )
        if filename_source_names:
            filename_source = artifact_inputs.require_by_name_and_artifact_type(
                filename_source_names[0],
                ImageArtifactType,
            )
        elif (
            _enum_setting(
                module,
                cls.filename_method_setting,
                SaveImagesFilenameMethod,
                SaveImagesFilenameMethod.FROM_IMAGE_FILENAME,
            )
            is SaveImagesFilenameMethod.FROM_IMAGE_FILENAME
        ):
            filename_source = selected_image
        else:
            filename_source = None
        relations = (
            ()
            if filename_source is None
            else (
                MaterializationSourceIdentityRelation(
                    source=filename_source.ref(),
                ),
            )
        )
        saved_image = ArtifactSpec.output_preserving_source_stack_scope(
            output_name,
            ImageArtifactType,
            selected_image,
            relations=relations,
            sidecar_role=ArtifactSidecarRole.MATERIALIZED_IMAGE_COPY,
            materialization=cls._materialization_spec(
                module,
                output_name=output_name,
            ),
        )
        if not cls.records_file_and_path(module):
            return (saved_image,)
        recorded_measurements = ArtifactSpec.output(
            cls.measurement_artifact_name(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
            ),
            MeasurementsArtifactType,
            measurement_feature_owner=cls,
            relations=(
                SaveImagesRecordedMeasurementSourceRelation(
                    source=selected_image.ref()
                ),
                ArtifactSpecRelation(source=saved_image.ref()),
                GroupLineageSourceRelation(source=selected_image.ref()),
            ),
        )
        return saved_image, recorded_measurements

    @classmethod
    def _materialization_spec(
        cls,
        module: "ModuleBlock",
        *,
        output_name: str,
    ) -> MaterializationSpec:
        image_kind = _enum_setting(
            module,
            cls.image_kind_setting,
            SaveImagesImageKind,
            SaveImagesImageKind.IMAGE,
        )
        if image_kind is not SaveImagesImageKind.IMAGE:
            raise ValueError(
                f"SaveImages({module.module_num}) supports image payloads only."
            )
        if _bool_setting(module, cls.create_subfolders_setting, False):
            raise ValueError(
                f"SaveImages({module.module_num}) source-relative subfolder "
                "replication is not representable by ImageFileOptions."
            )
        digits = _int_setting(module, cls.number_of_digits_setting, 4)
        if digits <= 0:
            raise ValueError("SaveImages sequential filename digits must be positive.")

        filename_method = _enum_setting(
            module,
            cls.filename_method_setting,
            SaveImagesFilenameMethod,
            SaveImagesFilenameMethod.FROM_IMAGE_FILENAME,
        )
        file_format = _enum_setting(
            module,
            cls.file_format_setting,
            SaveImagesFileFormat,
            SaveImagesFileFormat.TIFF,
        )
        append_suffix = _bool_setting(module, cls.append_suffix_setting, False)
        suffix = (
            optional_setting_value(module, cls.filename_suffix_setting) or ""
            if append_suffix
            else ""
        )
        single_name = (
            optional_setting_value(module, cls.single_file_name_setting) or "SavedImage"
        ).strip()
        sequential_prefix = single_name
        directory = SaveImagesOutputLocation.relative_directory_from_literal(
            optional_setting_value(module, cls.output_location_setting)
        )
        relative_path_template = filename_method.relative_path_template(
            directory=directory,
            single_name=single_name,
            sequential_prefix=sequential_prefix,
            sequential_digits=digits,
            suffix=suffix,
            file_format=file_format,
        )
        filename_identity = (
            MaterializedFilenameIdentity.SOURCE_IDENTITY
            if filename_method is SaveImagesFilenameMethod.FROM_IMAGE_FILENAME
            else MaterializedFilenameIdentity.ARTIFACT_NAME
        )
        options = ImageFileOptions(
            filename_suffix=f"{suffix}{file_format.value}",
            filename_identity=filename_identity,
            relative_path_template=relative_path_template,
        )
        materialization = MaterializationSpec(options)
        materialization.candidate_paths(f"{output_name}.pkl")
        return materialization


def _converted_saved_image(
    image_to_save: RuntimeArrayData,
    *,
    image_kind: SaveImagesImageKind,
    bit_depth: SaveImagesBitDepth,
) -> RuntimeArrayData:
    if image_kind is not SaveImagesImageKind.IMAGE:
        raise ValueError("save_images supports image payloads only.")
    return bit_depth.convert(image_to_save)


def _recorded_save_images_rows(
    filename_source: RuntimeArrayData,
    *,
    slice_index: int,
    saved_image_name: str,
    filename_method: SaveImagesFilenameMethod,
    single_file_name: str,
    number_of_digits: int,
    append_suffix: bool,
    filename_suffix: str,
    file_format: SaveImagesFileFormat,
    output_location: str | None,
) -> DataclassMeasurementColumnarRows:
    suffix = filename_suffix if append_suffix else ""
    if filename_method is SaveImagesFilenameMethod.FROM_IMAGE_FILENAME:
        source_path = image_payload_metadata(filename_source).source_path
        source_stem = (
            PurePosixPath(str(source_path).replace("\\", "/")).stem
            if source_path is not None
            else saved_image_name
        )
        filename = f"{source_stem}{suffix}{file_format.value}"
    elif filename_method is SaveImagesFilenameMethod.SEQUENTIAL_NUMBERS:
        filename = (
            f"{single_file_name}{1:0{number_of_digits}d}{suffix}{file_format.value}"
        )
    else:
        filename = f"{single_file_name}{suffix}{file_format.value}"
    relative_path = _relative_template(output_location, filename)
    pathname = PurePosixPath(relative_path).parent.as_posix()
    if pathname == ".":
        pathname = ""
    values = {
        SaveImagesRecordedMeasurementFeature.FILE_NAME: filename,
        SaveImagesRecordedMeasurementFeature.PATH_NAME: pathname,
        SaveImagesRecordedMeasurementFeature.URL: f"file:{quote(relative_path)}",
    }
    rows = tuple(
        SaveImagesRecordedMeasurementRow(
            slice_index=slice_index,
            feature_name=feature.feature_name(saved_image_name),
            result_value=values[feature],
        )
        for feature in SaveImagesRecordedMeasurementFeature
    )
    return DataclassMeasurementColumnarRows(
        rows,
        row_type=SaveImagesRecordedMeasurementRow,
    )


@numpy(contract=ProcessingContract.PURE_3D)
@special_inputs("image_to_save")
def save_images(
    image: RuntimeArrayData,
    *,
    image_to_save: RuntimeArrayData,
    image_kind: SaveImagesImageKind = SaveImagesImageKind.IMAGE,
    filename_method: SaveImagesFilenameMethod = (
        SaveImagesFilenameMethod.FROM_IMAGE_FILENAME
    ),
    single_file_name: str = "SavedImage",
    number_of_digits: int = 4,
    append_suffix: bool = False,
    filename_suffix: str = "",
    file_format: SaveImagesFileFormat = SaveImagesFileFormat.TIFF,
    output_location: str | None = None,
    bit_depth: SaveImagesBitDepth = SaveImagesBitDepth.NATIVE,
    overwrite: bool = True,
    when_to_save: SaveImagesWhen = SaveImagesWhen.EVERY_CYCLE,
    create_subfolders: bool = False,
    base_image_folder: str | None = None,
    series_axis: SaveImagesSeriesAxis = SaveImagesSeriesAxis.TIMEPOINT,
    lossless_compression: bool = True,
) -> tuple[RuntimeArrayData, RuntimeArrayData]:
    """Prepare a selected image or object set for saving without replacing the pipeline image.

    Args:
        image_to_save: Image or object labels written by the configured save step.
    """

    del (
        filename_method,
        single_file_name,
        number_of_digits,
        append_suffix,
        filename_suffix,
        file_format,
        output_location,
        overwrite,
        when_to_save,
        create_subfolders,
        base_image_folder,
        series_axis,
        lossless_compression,
    )
    return image, _converted_saved_image(
        image_to_save,
        image_kind=image_kind,
        bit_depth=bit_depth,
    )


@numpy(contract=ProcessingContract.PURE_3D)
@special_inputs("image_to_save")
@runtime_bound_parameters(SliceIndexRuntimeParameter)
def save_images_with_measurements(
    image: RuntimeArrayData,
    *,
    image_to_save: RuntimeArrayData,
    saved_image_name: str,
    image_kind: SaveImagesImageKind = SaveImagesImageKind.IMAGE,
    filename_method: SaveImagesFilenameMethod = (
        SaveImagesFilenameMethod.FROM_IMAGE_FILENAME
    ),
    single_file_name: str = "SavedImage",
    number_of_digits: int = 4,
    append_suffix: bool = False,
    filename_suffix: str = "",
    file_format: SaveImagesFileFormat = SaveImagesFileFormat.TIFF,
    output_location: str | None = None,
    bit_depth: SaveImagesBitDepth = SaveImagesBitDepth.NATIVE,
    overwrite: bool = True,
    when_to_save: SaveImagesWhen = SaveImagesWhen.EVERY_CYCLE,
    create_subfolders: bool = False,
    base_image_folder: str | None = None,
    series_axis: SaveImagesSeriesAxis = SaveImagesSeriesAxis.TIMEPOINT,
    lossless_compression: bool = True,
    slice_index: int = 0,
) -> tuple[RuntimeArrayData, RuntimeArrayData, DataclassMeasurementColumnarRows]:
    """Prepare an image for saving and record its output-file measurements.

    Args:
        saved_image_name: Image name recorded in the saved-file measurement rows.
    """

    del (
        overwrite,
        when_to_save,
        create_subfolders,
        base_image_folder,
        series_axis,
        lossless_compression,
    )
    converted = _converted_saved_image(
        image_to_save,
        image_kind=image_kind,
        bit_depth=bit_depth,
    )
    rows = _recorded_save_images_rows(
        image,
        slice_index=slice_index,
        saved_image_name=saved_image_name,
        filename_method=filename_method,
        single_file_name=single_file_name,
        number_of_digits=number_of_digits,
        append_suffix=append_suffix,
        filename_suffix=filename_suffix,
        file_format=file_format,
        output_location=output_location,
    )
    return image, converted, rows


set_signature_analysis_target(save_images_with_measurements, save_images)
