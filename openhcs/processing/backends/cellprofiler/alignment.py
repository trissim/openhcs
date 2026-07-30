"""Alignment backends for CellProfiler-compatible processing."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, ClassVar
import numpy as np
from metaclass_registry import AutoRegisterMeta
from scipy.fftpack import fft2, ifft2
from openhcs.constants.constants import GroupBy, MemoryType, VariableComponents
from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
)
from openhcs.core.memory.decorators import numpy
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
    MeasurementSparseColumnarRows,
)
from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    ImageMeasurementSubjectRelation,
    SourceStackLineageSourceRelation,
)
from openhcs.core.pipeline.function_contracts import (
    required_variable_components,
)
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    RuntimeMeasurementFeature,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxisValueProjection,
)
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    MaskedImagePayload,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)
from openhcs.interop.cellprofiler.module_settings import (
    BoundModuleSettings,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    MeasurementArtifactOutputModule,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    optional_setting_value,
    required_setting_value,
    setting_name_matches,
    setting_names,
    setting_values,
)
from openhcs.interop.cellprofiler.settings_binder import SettingToKeywordBinding
from openhcs.interop.cellprofiler.cellprofiler_literals import (
    cellprofiler_enum_from_literal,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    DeclaredImageOutputPayloadMeasurementRecordMixin,
    FieldDerivedMeasurementFeatureModule,
    MeasurementFeatureRecord,
    NoObjectNameMeasurementRecordMixin,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    ModuleOwnedResultMeasurementRows,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    CellProfilerBackendAuthority,
)
from openhcs.processing.backends.cellprofiler.alignment_mutual_information_offset import (
    mutual_information_offset_numba,
    mutual_information_offset_unmasked_numba,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


class _AlignShiftFieldRole(str, Enum):
    """Routing roles carried by the Align producer row schema."""

    IMAGE_OUTPUT_INDEX = "image_output_index"


class AlignOutputMeasurementRecordRowsMixin(FieldDerivedMeasurementFeatureModule):
    """Declares Align measurement rows on the module MRO."""

    measurement_feature_family = "Align"
    measurement_feature_token_aliases = (("shift", "shift"),)

    class MeasurementFeature(RuntimeMeasurementFeature):
        """Shift features emitted from producer-annotated Align result fields."""

        X_SHIFT = "Xshift"
        Y_SHIFT = "Yshift"

    @dataclass(frozen=True, slots=True)
    class MeasurementRecord(MeasurementFeatureRecord):
        """Raw Align measurements before CP row projection."""

        slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
        source_image_name: Annotated[str, MeasurementRowAxisField.SOURCE_IMAGE_NAME]
        x_shift: int
        y_shift: int

    @dataclass(frozen=True, slots=True)
    class MeasurementRows(ModuleOwnedResultMeasurementRows):
        """Project absorbed Align results into CP image measurement rows."""

        image_output_names: tuple[str, ...]

        @classmethod
        def for_request(cls, module_type, request):
            image_output_names = (
                request.callable_contract.artifact_outputs.names_of_artifact_type(
                    ImageArtifactType
                )
            )
            return cls(
                request.output_value,
                module_type=module_type,
                image_output_names=image_output_names,
            )

        def rows(self) -> MeasurementSparseColumnarRows:
            records: list[dict[str, object]] = []
            slice_field = self.source_field_annotated_by(
                AlignShiftMeasurement,
                MeasurementRowAxisField.SLICE_INDEX,
            )
            output_index_field = self.source_field_annotated_by(
                AlignShiftMeasurement,
                _AlignShiftFieldRole.IMAGE_OUTPUT_INDEX,
            )
            shift_fields = self.source_fields_annotated_with(
                AlignShiftMeasurement,
                RuntimeMeasurementFeature,
            )
            if not shift_fields:
                raise TypeError(
                    "AlignShiftMeasurement must annotate at least one shift feature."
                )
            for result in self.source_rows().iter_row_mappings():
                output_index = int(result[output_index_field.name])
                if output_index < 0 or output_index >= len(self.image_output_names):
                    raise ValueError(
                        f"Align measurement output_index {output_index} does not match "
                        f"declared image outputs {self.image_output_names!r}."
                    )
                source_image_name = self.image_output_names[output_index]
                records.append(
                    {
                        slice_field.name: int(result[slice_field.name]),
                        MeasurementRowAxisField.SOURCE_IMAGE_NAME.value: source_image_name,
                        **{
                            self.module_type.measurement_feature_name(
                                feature.feature_name,
                                source_image_name,
                            ): int(result[field_spec.name])
                            for field_spec, feature in shift_fields
                        },
                    }
                )
            fields = (
                slice_field,
                FieldSpec(MeasurementRowAxisField.SOURCE_IMAGE_NAME.value, str),
                *(
                    FieldSpec(
                        self.module_type.measurement_feature_name(
                            feature.feature_name,
                            output_name,
                        ),
                        field_spec.dtype,
                        required=False,
                    )
                    for output_name in self.image_output_names
                    for field_spec, feature in shift_fields
                ),
            )
            return MeasurementSparseColumnarRows.from_rows(
                records,
                fields=fields,
            )


class AlignModule(
    AlignOutputMeasurementRecordRowsMixin,
    NoObjectNameMeasurementRecordMixin,
    DeclaredImageOutputPayloadMeasurementRecordMixin,
    MeasurementArtifactOutputModule,
):
    module_name = "Align"
    function_name = "align"
    validated = True
    group_by = GroupBy.SITE
    confidence = 1.0

    class Method(Enum):
        """Image-registration metric supported by CellProfiler Align."""

        MUTUAL_INFORMATION = "Mutual Information"
        NORMALIZED_CROSS_CORRELATION = "Normalized Cross Correlation"

    @classmethod
    def measurement_output_relations(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key,
        step_context,
        artifact_inputs: ArtifactSpecCollection,
    ):
        """Declare every aligned image as a subject of Align measurements."""

        return (
            *super().measurement_output_relations(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
            ),
            *(
                ImageMeasurementSubjectRelation(
                    source=ArtifactSpec.output(
                        output_name,
                        ImageArtifactType,
                    ).ref()
                )
                for output_name in cls.image_output_names(module)
            ),
        )

    method_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Select the alignment method"
    )
    crop_mode_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Crop mode",
        aliases=("Crop output images to retain just the aligned regions?",),
    )
    first_input_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Select the first input image"
    )
    first_output_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Name the first output image"
    )
    second_input_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Select the second input image"
    )
    second_output_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Name the second output image"
    )
    additional_input_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Select the additional image"
    )
    additional_output_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Name the output image"
    )
    additional_mode_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Select how the alignment is to be applied"
    )
    fixed_image_input_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        SettingToKeywordBinding.input(first_input_setting, ImageArtifactType),
        SettingToKeywordBinding.input(second_input_setting, ImageArtifactType),
    )
    fixed_image_output_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        SettingToKeywordBinding.output(first_output_setting, ImageArtifactType),
        SettingToKeywordBinding.output(second_output_setting, ImageArtifactType),
    )

    class AdditionalMode(str, Enum):
        SIMILARLY = "Similarly"

        @classmethod
        def from_literal(
            cls, value: "AlignModule.AdditionalMode | str"
        ) -> "AlignModule.AdditionalMode":
            return cellprofiler_enum_from_literal(cls, value)

    class CropMode(str, Enum):
        KEEP_SIZE = "Keep size"
        CROP_TO_ALIGNED_REGION = "Crop to aligned region"
        PAD_IMAGES = "Pad images"

        @classmethod
        def from_literal(
            cls, value: "AlignModule.CropMode | str"
        ) -> "AlignModule.CropMode":
            return cellprofiler_enum_from_literal(
                cls,
                value,
                aliases={
                    "yes": cls.CROP_TO_ALIGNED_REGION,
                    "true": cls.CROP_TO_ALIGNED_REGION,
                    "no": cls.KEEP_SIZE,
                    "false": cls.KEEP_SIZE,
                },
            )

    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        *fixed_image_input_bindings,
        SettingToKeywordBinding.input(
            additional_input_setting,
            ImageArtifactType,
            repeated=True,
        ),
        *fixed_image_output_bindings,
        SettingToKeywordBinding.output(
            additional_output_setting,
            ImageArtifactType,
            repeated=True,
        ),
        SettingToKeywordBinding(method_setting, "method"),
        SettingToKeywordBinding(
            crop_mode_setting,
            "crop_mode",
            CropMode.from_literal,
        ),
        SettingToKeywordBinding(
            additional_mode_setting,
            "additional_alignment_modes",
            AdditionalMode.from_literal,
        ),
    )

    @classmethod
    def postprocess_bound_settings(
        cls,
        module: "ModuleBlock",
        bound: BoundModuleSettings,
    ) -> BoundModuleSettings:
        method_text = optional_setting_value(module, cls.method_setting)
        kwargs: dict[str, Any] = {
            "method": (
                cls.Method(method_text)
                if method_text is not None
                else cls.Method.MUTUAL_INFORMATION
            ),
            "crop_mode": cls.crop_mode(module),
        }
        additional_modes = cls.additional_alignment_modes(module)
        if additional_modes:
            kwargs["additional_alignment_modes"] = additional_modes
        return bound.with_kwargs(kwargs)

    @classmethod
    def image_input_names(cls, module: "ModuleBlock") -> tuple[str, ...]:
        return (
            *(
                required_setting_value(module, binding.setting_name)
                for binding in cls.fixed_image_input_bindings
            ),
            *setting_values(module, cls.additional_input_setting),
        )

    @classmethod
    def image_output_names(cls, module: "ModuleBlock") -> tuple[str, ...]:
        return (
            *(
                required_setting_value(module, binding.setting_name)
                for binding in cls.fixed_image_output_bindings
            ),
            *setting_values(module, cls.additional_output_setting),
        )

    @classmethod
    def crop_mode(cls, module: "ModuleBlock") -> "AlignModule.CropMode":
        return cls.CropMode.from_literal(
            optional_setting_value(module, cls.crop_mode_setting) or "No"
        )

    @classmethod
    def additional_alignment_modes(
        cls, module: "ModuleBlock"
    ) -> tuple["AlignModule.AdditionalMode", ...]:
        additional_inputs = setting_values(module, cls.additional_input_setting)
        additional_outputs = setting_values(module, cls.additional_output_setting)
        if len(additional_inputs) != len(additional_outputs):
            raise ValueError(
                f"Module Align({module.module_num}) has {len(additional_inputs)} additional inputs but {len(additional_outputs)} additional outputs."
            )
        raw_modes = setting_values(module, cls.additional_mode_setting)
        if not raw_modes:
            return (cls.AdditionalMode.SIMILARLY,) * len(additional_inputs)
        modes = tuple((cls.AdditionalMode.from_literal(value) for value in raw_modes))
        if len(modes) != len(additional_inputs):
            raise ValueError(
                f"Module Align({module.module_num}) has {len(modes)} additional alignment modes for {len(additional_inputs)} additional images."
            )
        return modes

    @classmethod
    def _derived_identity_setting_records(
        cls,
        *,
        invocation,
        block_position,
        existing_records,
        step_context,
    ):
        """Derive one output identity for every reconstructed Align input."""
        from openhcs.interop.cellprofiler.parser import ModuleSetting

        del invocation
        fixed_inputs = tuple(
            cls._setting_record_values(existing_records, binding.setting_name)
            for binding in cls.fixed_image_input_bindings
        )
        additional_inputs = cls._setting_record_values(
            existing_records, cls.additional_input_setting
        )
        if any(len(names) != 1 for names in fixed_inputs):
            raise ValueError(
                "Align reconstruction requires exactly one first and second image."
            )

        expected_outputs = (
            *((binding.setting_name, 1) for binding in cls.fixed_image_output_bindings),
            (cls.additional_output_setting, len(additional_inputs)),
        )
        records: list[ModuleSetting] = []
        output_position = 0
        for setting, expected_count in expected_outputs:
            existing_names = cls._setting_record_values(existing_records, setting)
            if len(existing_names) > expected_count:
                raise ValueError(
                    f"Align output setting {setting_names(setting)[0]!r} declares "
                    f"{len(existing_names)} names for {expected_count} inputs."
                )
            for _ in range(expected_count - len(existing_names)):
                records.append(
                    ModuleSetting(
                        setting_names(setting)[0],
                        cls.canonical_output_artifact_name(
                            artifact_type=ImageArtifactType,
                            output_position=output_position + len(existing_names),
                            block_position=block_position,
                            step_context=step_context,
                        ),
                    )
                )
                existing_names = (*existing_names, records[-1].value)
            output_position += expected_count
        return tuple(records)

    @staticmethod
    def _setting_record_values(records, setting) -> tuple[str, ...]:
        return tuple(
            str(record.value)
            for record in records
            if setting_name_matches(record.name, setting)
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
    ):
        """Preserve each aligned output's corresponding input stack scope."""
        del (
            invocation_key,
            step_context,
            binding,
        )
        image_inputs = artifact_inputs.for_artifact_type(ImageArtifactType).specs
        if output_position >= len(image_inputs):
            raise ValueError(
                f"Align output {name!r} at position {output_position} has no "
                f"corresponding input in {image_inputs!r}."
            )
        return (
            SourceStackLineageSourceRelation(
                source=image_inputs[output_position].ref()
            ),
        )


class AlignmentBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Alignment operations keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def mutual_information_offset(
        self,
        reference_pixels: np.ndarray,
        moving_pixels: np.ndarray,
        reference_mask: np.ndarray,
        moving_mask: np.ndarray,
    ) -> tuple[int, int]:
        """Return column/row offset maximizing mutual information."""


class NumbaNumpyAlignmentBackendStrategy(AlignmentBackendStrategy):
    """Numba-accelerated NumPy alignment primitives."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        pixels = np.arange(16, dtype=np.float64).reshape((4, 4))
        mask = np.ones(pixels.shape, dtype=np.bool_)
        self.mutual_information_offset(pixels, pixels, mask, mask)

    def mutual_information_offset(
        self,
        reference_pixels: np.ndarray,
        moving_pixels: np.ndarray,
        reference_mask: np.ndarray,
        moving_mask: np.ndarray,
    ) -> tuple[int, int]:
        max_shape = np.maximum(reference_pixels.shape, moving_pixels.shape)
        reshaped_reference_pixels = _reshape_image(reference_pixels, max_shape)
        reshaped_moving_pixels = _reshape_image(moving_pixels, max_shape)
        reshaped_reference_mask = _reshape_image(reference_mask, max_shape)
        reshaped_moving_mask = _reshape_image(moving_mask, max_shape)
        if bool(np.all(reshaped_reference_mask)) and bool(np.all(reshaped_moving_mask)):
            return mutual_information_offset_unmasked_numba(
                np.asarray(reshaped_reference_pixels, dtype=np.float64),
                np.asarray(reshaped_moving_pixels, dtype=np.float64),
            )
        return mutual_information_offset_numba(
            np.asarray(reshaped_reference_pixels, dtype=np.float64),
            np.asarray(reshaped_moving_pixels, dtype=np.float64),
            np.asarray(reshaped_reference_mask, dtype=np.bool_),
            np.asarray(reshaped_moving_mask, dtype=np.bool_),
        )


AlignAdditionalModes = tuple[AlignModule.AdditionalMode, ...]
AlignImageGeometry = tuple[tuple[int, int], tuple[int, int]]
AlignGeometryPair = tuple[AlignImageGeometry, AlignImageGeometry]


@dataclass(frozen=True, slots=True)
class AlignShiftMeasurement:
    """Per-output translation reported by CellProfiler Align."""

    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
    output_index: Annotated[int, _AlignShiftFieldRole.IMAGE_OUTPUT_INDEX]
    x_shift: Annotated[int, AlignModule.MeasurementFeature.X_SHIFT]
    y_shift: Annotated[int, AlignModule.MeasurementFeature.Y_SHIFT]


@dataclass(frozen=True, slots=True)
class TranslationOffsetRequest:
    """Inputs for Align translation-offset computation."""

    reference_image: np.ndarray
    moving_image: np.ndarray
    method: AlignModule.Method
    first_mask: np.ndarray | None
    second_mask: np.ndarray | None
    alignment_backend_provider: BackendProviderInput

    def offset(self) -> tuple[int, int]:
        """Return integer row/column offsets in CellProfiler's native convention."""
        reference_pixels = np.asarray(self.reference_image, dtype=float)
        moving_pixels = np.asarray(self.moving_image, dtype=float)
        if reference_pixels.ndim != 2 or moving_pixels.ndim != 2:
            raise ValueError(
                "Align offset computation requires explicitly projected 2-D "
                "registration images."
            )
        if self.method is AlignModule.Method.NORMALIZED_CROSS_CORRELATION:
            column_offset, row_offset = cross_correlation_offset(
                reference_pixels, moving_pixels
            )
        else:
            selected_backend = AlignmentBackendStrategy.for_memory_type(
                backend_provider=self.alignment_backend_provider
            )
            column_offset, row_offset = selected_backend.mutual_information_offset(
                reference_pixels,
                moving_pixels,
                (
                    np.ones(reference_pixels.shape, dtype=bool)
                    if self.first_mask is None
                    else np.asarray(self.first_mask, dtype=bool)
                ),
                (
                    np.ones(moving_pixels.shape, dtype=bool)
                    if self.second_mask is None
                    else np.asarray(self.second_mask, dtype=bool)
                ),
            )
        return (int(row_offset), int(column_offset))


@dataclass(frozen=True, slots=True)
class AlignOutputRequest:
    """Nominal request for applying one Align output geometry."""

    image: np.ndarray
    mask: np.ndarray | None
    metadata: ImagePayloadMetadata
    offset: tuple[int, int]
    shape: tuple[int, int]

    def aligned_payload(self) -> np.ndarray | MaskedImagePayload:
        output_shape = tuple(self.shape) + tuple(np.asarray(self.image).shape[2:])
        output = np.zeros(output_shape, dtype=np.asarray(self.image).dtype)
        source_view, output_view = offset_slice(
            np.asarray(self.image), output, *self.offset
        )
        output_view[...] = source_view
        source_mask = (
            np.ones(np.asarray(self.image).shape[:2], dtype=bool)
            if self.mask is None
            else np.asarray(self.mask, dtype=bool)
        )
        output_mask = np.zeros(tuple(self.shape), dtype=bool)
        source_mask_view, output_mask_view = offset_slice(
            source_mask, output_mask, *self.offset
        )
        output_mask_view[...] = source_mask_view
        return self.metadata.payload_with(
            output, None if np.all(output_mask) else output_mask
        )


@dataclass(frozen=True, slots=True)
class SimilarlyAlignedOutputGeometry:
    """Geometry for applying the second Align transform to additional images."""

    additional_image: np.ndarray
    second_image: np.ndarray
    second_offset: tuple[int, int]
    second_shape: tuple[int, int]
    crop_mode: AlignModule.CropMode

    @property
    def geometry(self) -> tuple[tuple[int, int], tuple[int, int]]:
        if self.crop_mode is AlignModule.CropMode.KEEP_SIZE:
            return (
                self.second_offset,
                tuple(np.asarray(self.additional_image).shape[:2]),
            )
        if tuple(np.asarray(self.additional_image).shape[:2]) != tuple(
            np.asarray(self.second_image).shape[:2]
        ):
            raise ValueError(
                "Align additional images with non-keep-size crop modes must share the second input image spatial shape."
            )
        return (self.second_offset, self.second_shape)


class AlignCropModeStrategy(
    EnumKeyedStrategyMixin[AlignModule.CropMode], ABC, metaclass=AutoRegisterMeta
):
    """Nominal strategy family for legacy Align crop modes."""

    __registry_key__ = "crop_mode_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "crop_mode"
    __enum_label_attr__ = "crop_mode_label"
    crop_mode: ClassVar[AlignModule.CropMode | None] = None
    crop_mode_label: ClassVar[str | None] = None

    @classmethod
    def for_crop_mode(cls, crop_mode: AlignModule.CropMode) -> "AlignCropModeStrategy":
        return cls.for_enum_member(crop_mode)

    @abstractmethod
    def apply(
        self,
        offsets: AlignImageGeometry,
        shapes: AlignImageGeometry,
    ) -> AlignGeometryPair:
        """Return first/second image outputs for one crop mode."""


class KeepSizeAlignCropModeStrategy(AlignCropModeStrategy):
    """Keep aligned images in their original shape."""

    crop_mode = AlignModule.CropMode.KEEP_SIZE

    def apply(
        self,
        offsets: AlignImageGeometry,
        shapes: AlignImageGeometry,
    ) -> AlignGeometryPair:
        return (offsets, shapes)


class PadImagesAlignCropModeStrategy(AlignCropModeStrategy):
    """Pad both images to preserve all shifted content."""

    crop_mode = AlignModule.CropMode.PAD_IMAGES

    def apply(
        self,
        offsets: AlignImageGeometry,
        shapes: AlignImageGeometry,
    ) -> AlignGeometryPair:
        offsets_array = np.asarray(offsets, dtype=int)
        shapes_array = np.asarray(shapes, dtype=int)
        offsets_array = offsets_array - np.min(offsets_array, axis=0)[np.newaxis, :]
        shapes_array = shapes_array + offsets_array
        output_shape = np.max(shapes_array, axis=0)
        output_shapes = np.tile(output_shape, (len(shapes), 1))
        return (
            tuple(tuple(int(value) for value in row) for row in offsets_array),
            tuple(tuple(int(value) for value in row) for row in output_shapes),
        )


class CropToOverlapAlignCropModeStrategy(AlignCropModeStrategy):
    """Crop both images to the overlapping aligned region."""

    crop_mode = AlignModule.CropMode.CROP_TO_ALIGNED_REGION

    def apply(
        self,
        offsets: AlignImageGeometry,
        shapes: AlignImageGeometry,
    ) -> AlignGeometryPair:
        offsets_array = np.asarray(offsets, dtype=int)
        shapes_array = np.asarray(shapes, dtype=int)
        offsets_array = offsets_array - np.max(offsets_array, axis=0)[np.newaxis, :]
        shapes_array = shapes_array + offsets_array
        output_shape = np.min(shapes_array, axis=0)
        output_shapes = np.tile(output_shape, (len(shapes), 1))
        return (
            tuple(tuple(int(value) for value in row) for row in offsets_array),
            tuple(tuple(int(value) for value in row) for row in output_shapes),
        )


@dataclass(frozen=True, slots=True)
class AlignExecution:
    """Execute legacy CellProfiler Align semantics for stacked image payloads."""

    image: object
    method: AlignModule.Method
    crop_mode: AlignModule.CropMode
    additional_alignment_modes: AlignAdditionalModes
    alignment_backend_provider: BackendProviderInput

    def execute(
        self,
    ) -> tuple[AlignedImageStack, DataclassMeasurementColumnarRows]:
        """Return aligned image payloads followed by shift measurements."""
        input_data = np.asarray(image_payload_data(self.image))
        input_metadata = image_payload_metadata(self.image)
        if input_metadata.plane_axis is None:
            raise ValueError("Align requires a declared input image plane axis.")
        plane_count = input_metadata.source_provenance.source_plane_count
        if plane_count < 2:
            raise ValueError(
                "Align requires at least two declared source image planes."
            )
        plane_projection = RuntimePlaneAxisValueProjection.preserve(
            axis=input_metadata.plane_axis,
            axis_size=plane_count,
        )
        plane_projection.validate_shape(input_data.shape, value_name="Align input")
        image_payloads = tuple(
            RuntimeSliceProjection.value_for_slice(
                self.image,
                plane_projection.selected_plane(index),
            )
            for index in range(plane_count)
        )
        images = tuple(
            np.asarray(image_payload_data(payload)) for payload in image_payloads
        )
        first_image, second_image = images[:2]
        metadata = tuple(image_payload_metadata(payload) for payload in image_payloads)
        masks = tuple(
            self.spatial_mask(payload, image, plane_metadata)
            for payload, image, plane_metadata in zip(
                image_payloads,
                images,
                metadata,
                strict=True,
            )
        )
        registration_images = tuple(
            self.registration_image(image, plane_metadata)
            for image, plane_metadata in zip(images, metadata, strict=True)
        )
        additional_count = len(images) - 2
        if additional_count == 0:
            if self.additional_alignment_modes:
                raise ValueError(
                    "Align got additional alignment modes without extra images."
                )
            additional_modes: tuple[AlignModule.AdditionalMode, ...] = ()
        elif not self.additional_alignment_modes:
            additional_modes = (
                AlignModule.AdditionalMode.SIMILARLY,
            ) * additional_count
        else:
            additional_modes = tuple(
                AlignModule.AdditionalMode.from_literal(mode)
                for mode in self.additional_alignment_modes
            )
            if len(additional_modes) != additional_count:
                raise ValueError(
                    "Align additional alignment mode count must match additional "
                    f"image count; got {len(additional_modes)} modes for "
                    f"{additional_count} images."
                )
        row_offset, column_offset = TranslationOffsetRequest(
            reference_image=registration_images[0],
            moving_image=registration_images[1],
            method=self.method,
            first_mask=masks[0],
            second_mask=masks[1],
            alignment_backend_provider=self.alignment_backend_provider,
        ).offset()
        normalized_crop_mode = self.crop_mode
        offsets, shapes = AlignCropModeStrategy.for_crop_mode(
            normalized_crop_mode
        ).apply(
            ((0, 0), (row_offset, column_offset)),
            (first_image.shape[:2], second_image.shape[:2]),
        )
        outputs = [
            AlignOutputRequest(
                image=first_image,
                mask=masks[0],
                metadata=metadata[0],
                offset=offsets[0],
                shape=shapes[0],
            ).aligned_payload(),
            AlignOutputRequest(
                image=second_image,
                mask=masks[1],
                metadata=metadata[1],
                offset=offsets[1],
                shape=shapes[1],
            ).aligned_payload(),
        ]
        additional_measurements: list[AlignShiftMeasurement] = []
        for output_index, (
            additional_image,
            additional_mask,
            additional_metadata,
            mode,
        ) in enumerate(
            zip(images[2:], masks[2:], metadata[2:], additional_modes, strict=True),
            start=2,
        ):
            if mode is not AlignModule.AdditionalMode.SIMILARLY:
                raise ValueError(
                    f"Unsupported Align additional-image mode {mode.value!r}."
                )
            additional_offset, additional_shape = SimilarlyAlignedOutputGeometry(
                additional_image=additional_image,
                second_image=second_image,
                second_offset=offsets[1],
                second_shape=shapes[1],
                crop_mode=normalized_crop_mode,
            ).geometry
            outputs.append(
                AlignOutputRequest(
                    image=additional_image,
                    mask=additional_mask,
                    metadata=additional_metadata,
                    offset=additional_offset,
                    shape=additional_shape,
                ).aligned_payload()
            )
            additional_measurements.append(
                AlignShiftMeasurement(
                    slice_index=0,
                    output_index=output_index,
                    x_shift=int(-additional_offset[1]),
                    y_shift=int(-additional_offset[0]),
                )
            )
        measurements = (
            AlignShiftMeasurement(
                slice_index=0,
                output_index=0,
                x_shift=int(-offsets[0][1]),
                y_shift=int(-offsets[0][0]),
            ),
            AlignShiftMeasurement(
                slice_index=0,
                output_index=1,
                x_shift=int(-offsets[1][1]),
                y_shift=int(-offsets[1][0]),
            ),
            *additional_measurements,
        )
        return (
            AlignedImageStack(tuple(outputs)),
            DataclassMeasurementColumnarRows(
                measurements,
                row_type=AlignShiftMeasurement,
            ),
        )

    @staticmethod
    def registration_image(
        image: np.ndarray,
        metadata: ImagePayloadMetadata,
    ) -> np.ndarray:
        """Return 2-D registration pixels from declared channel semantics."""
        channel_axis = metadata.normalized_source_channel_axis(image)
        if channel_axis is None:
            if image.ndim != 2:
                raise ValueError(
                    "Align image planes without a source channel axis must be 2-D."
                )
            return np.asarray(image, dtype=float)
        projected = np.mean(np.asarray(image, dtype=float), axis=channel_axis)
        if projected.ndim != 2:
            raise ValueError(
                "Align source channel projection must produce a 2-D image."
            )
        return projected

    @staticmethod
    def spatial_mask(
        payload: object,
        image: np.ndarray,
        metadata: ImagePayloadMetadata,
    ) -> np.ndarray | None:
        """Return a 2-D mask using the image plane's declared channel axis."""
        mask = image_payload_mask(payload)
        if mask is None:
            return None
        mask_array = metadata.mask_domain(image).broadcast_to_data(mask)
        channel_axis = metadata.normalized_source_channel_axis(image)
        if channel_axis is not None:
            mask_array = np.all(mask_array, axis=channel_axis)
        if mask_array.ndim != 2:
            raise ValueError("Align image-plane masks must resolve to 2-D.")
        return np.asarray(mask_array, dtype=bool)


def cross_correlation_offset(
    reference_pixels: np.ndarray, moving_pixels: np.ndarray
) -> tuple[int, int]:
    shape = np.maximum(reference_pixels.shape, moving_pixels.shape)
    fft_shape = shape * 2
    row_grid, column_grid = np.mgrid[-shape[0] : shape[0], -shape[1] : shape[1]]
    overlap_count = np.abs(row_grid * column_grid).astype(float)
    overlap_count[overlap_count < 1] = 1
    reference_pixels = reference_pixels - np.mean(reference_pixels)
    moving_pixels = moving_pixels - np.mean(moving_pixels)
    reference_fft = fft2(reference_pixels, fft_shape.tolist())
    moving_fft = fft2(moving_pixels, fft_shape.tolist())
    correlation = ifft2(reference_fft * moving_fft.conj()).real
    ref_rows, ref_columns = reference_pixels.shape
    ref_sum = np.zeros(fft_shape)
    ref_sum[:ref_rows, :ref_columns] = cumsum_quadrant(
        reference_pixels, row_forwards=False, column_forwards=False
    )
    ref_sum[:ref_rows, -ref_columns:] = cumsum_quadrant(
        reference_pixels, row_forwards=False, column_forwards=True
    )
    ref_sum[-ref_rows:, :ref_columns] = cumsum_quadrant(
        reference_pixels, row_forwards=True, column_forwards=False
    )
    ref_sum[-ref_rows:, -ref_columns:] = cumsum_quadrant(
        reference_pixels, row_forwards=True, column_forwards=True
    )
    ref_mean = ref_sum / overlap_count
    moving_rows, moving_columns = moving_pixels.shape
    moving_sum = np.zeros(fft_shape)
    moving_sum[:moving_rows, :moving_columns] = cumsum_quadrant(
        moving_pixels, row_forwards=False, column_forwards=False
    )
    moving_sum[:moving_rows, -moving_columns:] = cumsum_quadrant(
        moving_pixels, row_forwards=False, column_forwards=True
    )
    moving_sum[-moving_rows:, :moving_columns] = cumsum_quadrant(
        moving_pixels, row_forwards=True, column_forwards=False
    )
    moving_sum[-moving_rows:, -moving_columns:] = cumsum_quadrant(
        moving_pixels, row_forwards=True, column_forwards=True
    )
    moving_mean = np.fliplr(np.flipud(moving_sum)) / overlap_count
    ref_sd = np.sum(reference_pixels**2) - ref_mean**2 * np.prod(shape)
    moving_sd = np.sum(moving_pixels**2) - moving_mean**2 * np.prod(shape)
    sd = np.sqrt(np.maximum(ref_sd * moving_sd, 0))
    normalized = np.divide(
        correlation, sd, out=np.zeros_like(correlation), where=sd != 0
    )
    normalized[(overlap_count < np.prod(shape) / 2) & (sd < np.mean(sd) / 100)] = 0
    row_offset, column_offset = np.unravel_index(np.argmax(normalized), fft_shape)
    if row_offset > reference_pixels.shape[0]:
        row_offset -= int(fft_shape[0])
    if column_offset > reference_pixels.shape[1]:
        column_offset -= int(fft_shape[1])
    return (int(column_offset), int(row_offset))


def cumsum_quadrant(
    values: np.ndarray, *, row_forwards: bool, column_forwards: bool
) -> np.ndarray:
    if row_forwards:
        values = values.cumsum(0)
    else:
        values = np.flipud(np.flipud(values).cumsum(0))
    if column_forwards:
        return values.cumsum(1)
    return np.fliplr(np.fliplr(values).cumsum(1))


def offset_slice(
    source: np.ndarray, target: np.ndarray, row_offset: int, column_offset: int
) -> tuple[np.ndarray, np.ndarray]:
    if row_offset < 0:
        height = min(source.shape[0] + row_offset, target.shape[0])
        source_row_start = -row_offset
        target_row_start = 0
    else:
        height = min(source.shape[0], target.shape[0] - row_offset)
        source_row_start = 0
        target_row_start = row_offset
    if column_offset < 0:
        width = min(source.shape[1] + column_offset, target.shape[1])
        source_column_start = -column_offset
        target_column_start = 0
    else:
        width = min(source.shape[1], target.shape[1] - column_offset)
        source_column_start = 0
        target_column_start = column_offset
    if height <= 0 or width <= 0:
        empty = (slice(0, 0), slice(0, 0))
        return (source[empty], target[empty])
    source_slices = (
        slice(source_row_start, source_row_start + height),
        slice(source_column_start, source_column_start + width),
        *(slice(None),) * max(0, source.ndim - 2),
    )
    target_slices = (
        slice(target_row_start, target_row_start + height),
        slice(target_column_start, target_column_start + width),
        *(slice(None),) * max(0, target.ndim - 2),
    )
    return (source[source_slices], target[target_slices])


def prepare_align() -> None:
    """Compile alignment backend kernels outside measured execution."""
    reference = np.zeros((32, 32), dtype=np.float32)
    moving = np.zeros((32, 32), dtype=np.float32)
    reference[8:20, 9:21] = 1.0
    moving[9:21, 8:20] = 1.0
    TranslationOffsetRequest(
        reference_image=reference,
        moving_image=moving,
        method=AlignModule.Method.MUTUAL_INFORMATION,
        first_mask=None,
        second_mask=None,
        alignment_backend_provider=DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    ).offset()


@required_variable_components(VariableComponents.CHANNEL)
@numpy(contract=ProcessingContract.PURE_3D)
def align(
    image: np.ndarray,
    *,
    method: AlignModule.Method = AlignModule.Method.MUTUAL_INFORMATION,
    crop_mode: AlignModule.CropMode = AlignModule.CropMode.KEEP_SIZE,
    additional_alignment_modes: AlignAdditionalModes = (),
    alignment_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[AlignedImageStack, DataclassMeasurementColumnarRows]:
    """Align primary images and apply declared additional-image shifts."""
    return AlignExecution(
        image=image,
        method=method,
        crop_mode=crop_mode,
        additional_alignment_modes=additional_alignment_modes,
        alignment_backend_provider=alignment_backend_provider,
    ).execute()


align.__openhcs_prepare__ = prepare_align


def _reshape_image(source: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
    if tuple(source.shape) == tuple(new_shape):
        return source
    result = np.zeros(new_shape, source.dtype)
    result[: source.shape[0], : source.shape[1]] = source
    return result


__all__ = public_names_from_objects(
    AlignExecution,
    AlignShiftMeasurement,
    AlignmentBackendStrategy,
    NumbaNumpyAlignmentBackendStrategy,
    align,
    prepare_align,
)
