"""Alignment backends for CellProfiler-compatible processing."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, ClassVar
import numpy as np
from metaclass_registry import AutoRegisterMeta
from scipy.fftpack import fft2, ifft2
from openhcs.constants.constants import MemoryType
from openhcs.core.memory.decorators import numpy
from openhcs.core.artifacts import ArtifactSpec, ArtifactSpecCollection, ImageArtifactType
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_semantics import MeasurementRowAxisField
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    MaskedImagePayload,
    RuntimeImagePayloadContext,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)
from openhcs.interop.cellprofiler.module_declarations import (
    ProcessingContract,
    ImageArtifactInputCapability,
    ImageArtifactInputModule,
    ImageArtifactOutputCapability,
    ImageArtifactOutputModule,
    MeasurementArtifactOutputCapability,
    MeasurementArtifactOutputModule,
    ModuleSettingsSourceModule,
)
from openhcs.interop.cellprofiler.setting_names import (
    optional_setting_value,
    required_setting_value,
    setting_values,
)
from openhcs.interop.cellprofiler.cellprofiler_literals import (
    cellprofiler_enum_from_literal,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    DeclaredImageOutputPayloadMeasurementRecordMixin,
    FieldDerivedMeasurementFeatureModule,
    MeasurementFeatureRecord,
    NoFieldsMeasurementRecordMixin,
    NoObjectNameMeasurementRecordMixin,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    CellProfilerMeasurementStatField,
    ModuleOwnedResultMeasurementRows,
)
from openhcs.interop.cellprofiler.runtime.main_flow import (
    CellProfilerMainFlowReplacementPolicyMixin,
)
from openhcs.interop.cellprofiler.runtime.payload_types import CellProfilerKwargDict
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


class AlignOutputMeasurementRecordRowsMixin(FieldDerivedMeasurementFeatureModule):
    """Declares Align measurement rows on the module MRO."""

    measurement_feature_family = "Align"
    measurement_feature_token_aliases = (("shift", "shift"),)

    class MeasurementStatField(CellProfilerMeasurementStatField):
        """Absorbed Align result fields."""

        OUTPUT_INDEX = "output_index"
        SLICE_INDEX = MeasurementRowAxisField.SLICE_INDEX.value
        X_SHIFT = "x_shift"
        Y_SHIFT = "y_shift"

    @dataclass(frozen=True, slots=True)
    class MeasurementRecord(MeasurementFeatureRecord):
        """Raw Align measurements before CP row projection."""

        slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
        source_image_name: Annotated[str, MeasurementRowAxisField.SOURCE_IMAGE_NAME]
        x_shift: float
        y_shift: float

    @dataclass(frozen=True, slots=True)
    class MeasurementRows(ModuleOwnedResultMeasurementRows):
        """Project absorbed Align results into CP image measurement rows."""

        registry_key = "align"
        image_output_names: tuple[str, ...]

        @classmethod
        def for_request(cls, module_type, request):
            image_output_names = tuple(
                spec.name
                for spec in ArtifactSpecCollection(
                    request.declared_outputs
                ).of_artifact_type(ImageArtifactType)
            )
            return cls(
                request.output_value,
                module_type=module_type,
                image_output_names=image_output_names,
            )

        def rows(self) -> list[CellProfilerKwargDict]:
            records: list[AlignOutputMeasurementRecordRowsMixin.MeasurementRecord] = []
            stat_field = self.stat_field_type
            record_type = self.module_type.MeasurementRecord
            for result in self.source_rows():
                output_index = int(
                    self.row_value(result, stat_field.OUTPUT_INDEX, 0)
                )
                if output_index < 0 or output_index >= len(self.image_output_names):
                    raise ValueError(
                        f"Align measurement output_index {output_index} does not match "
                        f"declared image outputs {self.image_output_names!r}."
                    )
                slice_index = int(
                    self.row_value(result, stat_field.SLICE_INDEX, 0)
                )
                records.append(
                    record_type(
                        slice_index=slice_index,
                        source_image_name=self.image_output_names[output_index],
                        x_shift=float(self.row_value(result, stat_field.X_SHIFT, 0.0)),
                        y_shift=float(self.row_value(result, stat_field.Y_SHIFT, 0.0)),
                    )
                )
            return self.module_type.measurement_feature_rows_from_records(tuple(records))


class AlignMainFlowReplacementMixin(CellProfilerMainFlowReplacementPolicyMixin):
    """Align publishes its declared aligned image outputs to downstream flow."""

    def replaces_main_flow(self, image_outputs: tuple[ArtifactSpec, ...]) -> bool:
        return bool(image_outputs)


class AlignModule(
    AlignMainFlowReplacementMixin,
    AlignOutputMeasurementRecordRowsMixin,
    NoObjectNameMeasurementRecordMixin,
    DeclaredImageOutputPayloadMeasurementRecordMixin,
    NoFieldsMeasurementRecordMixin,
    ModuleSettingsSourceModule,
    ImageArtifactInputModule,
    ImageArtifactOutputModule,
    MeasurementArtifactOutputModule,
):
    module_name = "Align"
    function_name = "align"
    validated = True
    contract = ProcessingContract.PURE_3D
    confidence = 1.0
    method_setting = "Select the alignment method"
    v2_crop_setting = "Crop output images to retain just the aligned regions?"
    crop_mode_setting = "Crop mode"
    first_input_setting = "Select the first input image"
    first_output_setting = "Name the first output image"
    second_input_setting = "Select the second input image"
    second_output_setting = "Name the second output image"
    additional_input_setting = "Select the additional image"
    additional_output_setting = "Name the output image"
    additional_mode_setting = "Select how the alignment is to be applied"
    image_input_settings = (
        first_input_setting,
        second_input_setting,
        additional_input_setting,
    )
    image_output_settings = (
        first_output_setting,
        second_output_setting,
        additional_output_setting,
    )

    @classmethod
    def compile_time_required_artifact_input_settings(cls):
        return tuple(
            (
                setting,
                capability_type,
            )
            for setting, capability_type in super().compile_time_required_artifact_input_settings()
            if setting != cls.additional_input_setting
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

    @classmethod
    def settings_source(cls, module: "ModuleBlock") -> "CellProfilerKwargs":
        kwargs: dict[str, Any] = {
            "method": optional_setting_value(module, cls.method_setting)
            or "Mutual Information",
            "crop_mode": cls.crop_mode(module).value,
        }
        additional_modes = cls.additional_alignment_modes(module)
        if additional_modes:
            kwargs["additional_alignment_modes"] = tuple(
                (mode.value for mode in additional_modes)
            )
        return kwargs

    @classmethod
    def image_input_names(cls, module: "ModuleBlock") -> tuple[str, ...]:
        return (
            required_setting_value(module, cls.first_input_setting),
            required_setting_value(module, cls.second_input_setting),
            *setting_values(module, cls.additional_input_setting),
        )

    @classmethod
    def image_output_names(cls, module: "ModuleBlock") -> tuple[str, ...]:
        return (
            required_setting_value(module, cls.first_output_setting),
            required_setting_value(module, cls.second_output_setting),
            *setting_values(module, cls.additional_output_setting),
        )

    @classmethod
    def crop_mode(cls, module: "ModuleBlock") -> "AlignModule.CropMode":
        return cls.CropMode.from_literal(
            optional_setting_value(module, cls.crop_mode_setting)
            or optional_setting_value(module, cls.v2_crop_setting)
            or "No"
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
    def artifact_contract(cls, assembler, builder, module):
        inputs = [
            ImageArtifactInputCapability.bind_artifact(cls, builder, module, ImageArtifactInputCapability.spec(name))
            for name in cls.image_input_names(module)
        ]
        output_names = cls.image_output_names(module)
        outputs = [
            ImageArtifactOutputCapability.bind_artifact(
                cls,
                builder,
                module,
                ArtifactSpec.output_preserving_source_stack_scope(
                    output_name,
                    ImageArtifactType,
                    ArtifactSpec.input(input_name, ImageArtifactType),
                ),
            )
            for input_name, output_name in zip(
                cls.image_input_names(module), output_names, strict=True
            )
        ]
        outputs.append(
            MeasurementArtifactOutputCapability.bind_artifact(cls, builder, module, MeasurementArtifactOutputCapability.spec(cls.measurement_artifact_name(module)))
        )
        return assembler.assemble_contract(
            module, builder, inputs=inputs, outputs=outputs
        )


from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer


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


AlignAdditionalModes = tuple[AlignModule.AdditionalMode | str, ...]
AlignImageGeometry = tuple[tuple[int, int], tuple[int, int]]
AlignGeometryPair = tuple[AlignImageGeometry, AlignImageGeometry]


@dataclass(frozen=True, slots=True)
class AlignCropRequest:
    """Inputs shared by Align crop-mode strategies."""

    offsets: AlignImageGeometry
    shapes: AlignImageGeometry


@dataclass(frozen=True, slots=True)
class AlignShiftMeasurement:
    """Per-output translation reported by CellProfiler Align."""

    slice_index: int
    output_index: int
    x_shift: float
    y_shift: float


@dataclass(frozen=True, slots=True)
class TranslationOffsetRequest:
    """Inputs for Align translation-offset computation."""

    reference_image: np.ndarray
    moving_image: np.ndarray
    method: str
    first_mask: np.ndarray | None
    second_mask: np.ndarray | None
    alignment_backend_provider: BackendProviderInput

    def offset(self) -> tuple[int, int]:
        """Return integer row/column offsets in CellProfiler's native convention."""
        reference_pixels = alignment_pixels(self.reference_image)
        moving_pixels = alignment_pixels(self.moving_image)
        if self.method.strip().lower() == "normalized cross correlation":
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
                alignment_mask(self.first_mask, reference_pixels.shape),
                alignment_mask(self.second_mask, moving_pixels.shape),
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
        return RuntimeImagePayloadContext(
            output,
            mask=None if np.all(output_mask) else output_mask,
            metadata=self.metadata,
        ).payload()


@dataclass(frozen=True, slots=True)
class AlignGeometryProjection:
    """Projection from mutable offset/shape arrays to immutable Align geometry."""

    offsets: np.ndarray
    shapes: np.ndarray

    def as_pair(self) -> AlignGeometryPair:
        return (
            tuple((tuple((int(value) for value in row)) for row in self.offsets)),
            tuple((tuple((int(value) for value in row)) for row in self.shapes)),
        )


@dataclass(frozen=True, slots=True)
class AlignInputPayloads:
    """Stacked Align image payload, mask, and metadata projection."""

    payload: object

    @property
    def images(self) -> tuple[np.ndarray, ...]:
        data = np.asarray(image_payload_data(self.payload))
        if not hasattr(data, "ndim") or data.ndim not in (3, 4) or data.shape[0] < 2:
            raise ValueError("Align requires at least two stacked image inputs.")
        return tuple((data[index] for index in range(data.shape[0])))

    def masks(self, images: tuple[np.ndarray, ...]) -> tuple[np.ndarray | None, ...]:
        mask = image_payload_mask(self.payload)
        if mask is None:
            return (None,) * len(images)
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.ndim == 2:
            return (mask_array,) * len(images)
        if mask_array.ndim == 3 and mask_array.shape[0] == len(images):
            return tuple((mask_array[index] for index in range(len(images))))
        spatial_shapes = {
            tuple(np.asarray(input_image).shape[:2]) for input_image in images
        }
        if len(spatial_shapes) == 1 and mask_array.shape[:2] == next(
            iter(spatial_shapes)
        ):
            return (mask_array,) * len(images)
        raise ValueError(
            "Align mask must be shared 2D mask or one mask per stacked image."
        )

    def metadata(self, count: int) -> tuple[ImagePayloadMetadata, ...]:
        metadata = image_payload_metadata(self.payload)
        return tuple((metadata.for_source_plane(index) for index in range(count)))


@dataclass(frozen=True, slots=True)
class AlignAdditionalModePlan:
    """Validated Align mode plan for additional images."""

    modes: AlignAdditionalModes
    additional_count: int

    @property
    def normalized_modes(self) -> tuple[AlignModule.AdditionalMode, ...]:
        if self.additional_count == 0:
            if self.modes:
                raise ValueError(
                    "Align got additional alignment modes without extra images."
                )
            return ()
        if not self.modes:
            return (AlignModule.AdditionalMode.SIMILARLY,) * self.additional_count
        normalized = tuple(
            (AlignModule.AdditionalMode.from_literal(mode) for mode in self.modes)
        )
        if len(normalized) != self.additional_count:
            raise ValueError(
                f"Align additional alignment mode count must match additional image count; got {len(normalized)} modes for {self.additional_count} images."
            )
        return normalized


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
    def apply(self, request: AlignCropRequest) -> AlignGeometryPair:
        """Return first/second image outputs for one crop mode."""


class KeepSizeAlignCropModeStrategy(AlignCropModeStrategy):
    """Keep aligned images in their original shape."""

    crop_mode = AlignModule.CropMode.KEEP_SIZE

    def apply(self, request: AlignCropRequest) -> AlignGeometryPair:
        return (request.offsets, request.shapes)


class PadImagesAlignCropModeStrategy(AlignCropModeStrategy):
    """Pad both images to preserve all shifted content."""

    crop_mode = AlignModule.CropMode.PAD_IMAGES

    def apply(self, request: AlignCropRequest) -> AlignGeometryPair:
        return align_offsets_for_padding(request.offsets, request.shapes)


class CropToOverlapAlignCropModeStrategy(AlignCropModeStrategy):
    """Crop both images to the overlapping aligned region."""

    crop_mode = AlignModule.CropMode.CROP_TO_ALIGNED_REGION

    def apply(self, request: AlignCropRequest) -> AlignGeometryPair:
        return align_offsets_for_cropping(request.offsets, request.shapes)


@dataclass(frozen=True, slots=True)
class AlignExecution:
    """Execute legacy CellProfiler Align semantics for stacked image payloads."""

    image: object
    method: str
    crop_mode: AlignModule.CropMode | str
    additional_alignment_modes: AlignAdditionalModes
    alignment_backend_provider: BackendProviderInput

    def execute(self) -> tuple[object, ...]:
        """Return aligned image payloads followed by shift measurements."""
        payloads = AlignInputPayloads(self.image)
        images = payloads.images
        first_image, second_image = images[:2]
        masks = payloads.masks(images)
        metadata = payloads.metadata(len(images))
        additional_modes = AlignAdditionalModePlan(
            self.additional_alignment_modes, len(images) - 2
        ).normalized_modes
        row_offset, column_offset = TranslationOffsetRequest(
            reference_image=first_image,
            moving_image=second_image,
            method=self.method,
            first_mask=masks[0],
            second_mask=masks[1],
            alignment_backend_provider=self.alignment_backend_provider,
        ).offset()
        normalized_crop_mode = AlignModule.CropMode.from_literal(self.crop_mode)
        offsets, shapes = align_offsets(
            ((0, 0), (row_offset, column_offset)),
            (first_image.shape[:2], second_image.shape[:2]),
            normalized_crop_mode,
        )
        outputs = list(
            crop_mode_outputs(
                first_image,
                second_image,
                first_mask=masks[0],
                second_mask=masks[1],
                first_metadata=metadata[0],
                second_metadata=metadata[1],
                offsets=offsets,
                shapes=shapes,
            )
        )
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
                    x_shift=float(-additional_offset[1]),
                    y_shift=float(-additional_offset[0]),
                )
            )
        measurements = (
            AlignShiftMeasurement(
                slice_index=0,
                output_index=0,
                x_shift=float(-offsets[0][1]),
                y_shift=float(-offsets[0][0]),
            ),
            AlignShiftMeasurement(
                slice_index=0,
                output_index=1,
                x_shift=float(-offsets[1][1]),
                y_shift=float(-offsets[1][0]),
            ),
            *additional_measurements,
        )
        return (*outputs, measurements)


def align_offsets(
    offsets: AlignImageGeometry,
    shapes: AlignImageGeometry,
    crop_mode: AlignModule.CropMode,
) -> AlignGeometryPair:
    return AlignCropModeStrategy.for_crop_mode(crop_mode).apply(
        AlignCropRequest(offsets=offsets, shapes=shapes)
    )


def align_offsets_for_cropping(
    offsets: AlignImageGeometry, shapes: AlignImageGeometry
) -> AlignGeometryPair:
    offsets_array = np.asarray(offsets, dtype=int)
    shapes_array = np.asarray(shapes, dtype=int)
    offsets_array = offsets_array - np.max(offsets_array, axis=0)[np.newaxis, :]
    shapes_array = shapes_array + offsets_array
    output_shape = np.min(shapes_array, axis=0)
    return AlignGeometryProjection(
        offsets=offsets_array, shapes=np.tile(output_shape, (len(shapes), 1))
    ).as_pair()


def align_offsets_for_padding(
    offsets: AlignImageGeometry, shapes: AlignImageGeometry
) -> AlignGeometryPair:
    offsets_array = np.asarray(offsets, dtype=int)
    shapes_array = np.asarray(shapes, dtype=int)
    offsets_array = offsets_array - np.min(offsets_array, axis=0)[np.newaxis, :]
    shapes_array = shapes_array + offsets_array
    output_shape = np.max(shapes_array, axis=0)
    return AlignGeometryProjection(
        offsets=offsets_array, shapes=np.tile(output_shape, (len(shapes), 1))
    ).as_pair()


def crop_mode_outputs(
    first_image: np.ndarray,
    second_image: np.ndarray,
    *,
    first_mask: np.ndarray | None,
    second_mask: np.ndarray | None,
    first_metadata: ImagePayloadMetadata,
    second_metadata: ImagePayloadMetadata,
    offsets: AlignImageGeometry,
    shapes: AlignImageGeometry,
) -> tuple[np.ndarray | MaskedImagePayload, np.ndarray | MaskedImagePayload]:
    return (
        AlignOutputRequest(
            image=first_image,
            mask=first_mask,
            metadata=first_metadata,
            offset=offsets[0],
            shape=shapes[0],
        ).aligned_payload(),
        AlignOutputRequest(
            image=second_image,
            mask=second_mask,
            metadata=second_metadata,
            offset=offsets[1],
            shape=shapes[1],
        ).aligned_payload(),
    )


def alignment_pixels(image: np.ndarray) -> np.ndarray:
    pixels = np.asarray(image, dtype=float)
    if pixels.ndim == 3:
        return np.mean(pixels, axis=2)
    return pixels


def alignment_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    return np.asarray(mask, dtype=bool)


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
        method="Mutual Information",
        first_mask=None,
        second_mask=None,
        alignment_backend_provider=DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    ).offset()


@numpy(contract=ProcessingContract.PURE_3D)
@special_outputs(
    (
        "align_measurements",
        csv_materializer(
            fields=["slice_index", "output_index", "x_shift", "y_shift"],
            analysis_type="alignment",
        ),
    )
)
def align(
    image: np.ndarray,
    *,
    method: str = "Mutual Information",
    crop_mode: AlignModule.CropMode | str = AlignModule.CropMode.KEEP_SIZE,
    additional_alignment_modes: AlignAdditionalModes = (),
    alignment_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[object, ...]:
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
