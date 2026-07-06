"""Shared CellProfiler image-plane geometry semantics."""

from __future__ import annotations
from dataclasses import replace
from enum import Enum
from openhcs.core.artifacts import (
    ArtifactSpecRef,
    GroupLineageSourceRelation,
    ArtifactSpec,
    ImageArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeImageInputOrigin,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerKwargDict,
    CellProfilerRuntimeValue,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileLogger,
)
from openhcs.interop.cellprofiler.runtime.special_input_policies import (
    CellProfilerSpecialInputPolicyMixin,
    SpecialInputBindingRequest,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.interop.cellprofiler.setting_names import SettingNameFamily
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    cellprofiler_enum_value_setting_parser,
    normalize_cellprofiler_setting_name,
    parse_cellprofiler_bool,
    parse_cellprofiler_float,
    parse_cellprofiler_int,
)
from openhcs.interop.cellprofiler.module_declarations import (
    ProcessingContract,
    BinderSettingsSourceModule,
    BoundModuleSettings,
    CellProfilerModule,
    ImageArtifactInputModule,
    ImageArtifactOutputModule,
    ModuleSettingsSourceModule,
    ObjectArtifactInputModule,
    ScopedMeasurementModule,
    StructuringElementSettingsModule,
)
from openhcs.interop.cellprofiler.setting_names import (
    optional_setting_value,
    required_setting_value,
    setting_values,
    split_symbol_names,
)
from openhcs.interop.cellprofiler.cellprofiler_literals import (
    cellprofiler_enum_from_literal,
)
from openhcs.processing.backends.cellprofiler.thresholding import (
    ThresholdSettingsModule,
)


class MaskImageSource(Enum):
    """Mask source domains exposed by MaskImage settings."""

    OBJECTS = "objects"
    IMAGE = "image"


class MaskImageSpecialInputPolicy(CellProfilerSpecialInputPolicyMixin):
    """Bind mask object labels in the current runtime plane."""

    def binding_current_image(
        self,
        *,
        current_image: CellProfilerRuntimeValue,
        primary_image: CellProfilerRuntimeValue | None,
    ) -> CellProfilerRuntimeValue:
        """Align mask labels to the image being masked."""
        return primary_image if primary_image is not None else current_image

    def bind(self, request: SpecialInputBindingRequest) -> CellProfilerKwargDict:
        if len(request.parameter_names) != len(request.special_input_specs):
            raise NotImplementedError(
                f"{request.module_name} declares special_inputs {list(request.parameter_names)}, but compiled runtime inputs are {[spec.name for spec in request.special_input_specs]}."
            )
        bound: CellProfilerKwargDict = {}
        alignment_image: CellProfilerRuntimeValue | None = None
        deferred_object_specs: list[tuple[str, ArtifactSpec]] = []
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "mask_image_special_input_specs",
            0.0,
            special_specs=tuple(
                (
                    (spec.artifact_type.value, spec.name)
                    for spec in request.special_input_specs
                )
            ),
            runtime_specs=tuple(
                (
                    (spec.artifact_type.value, spec.name)
                    for spec in request.runtime_inputs
                )
            ),
        )
        for parameter_name, spec in zip(
            request.parameter_names, request.special_input_specs, strict=True
        ):
            if spec.artifact_type is ObjectLabelsArtifactType:
                deferred_object_specs.append((parameter_name, spec))
                continue
            value = (
                request.runtime_value_without_current_image_projection(spec)
                if spec.artifact_type is ImageArtifactType
                and request.binding_scope.image_origin(spec)
                is RuntimeImageInputOrigin.RUNTIME
                else request.runtime_value(spec)
            )
            bound[parameter_name] = value
            if spec.artifact_type is ImageArtifactType and alignment_image is None:
                alignment_image = value
        if alignment_image is None:
            for spec in request.runtime_inputs:
                if (
                    spec.artifact_type is ImageArtifactType
                    and request.binding_scope.image_origin(spec)
                    is RuntimeImageInputOrigin.RUNTIME
                ):
                    alignment_image = (
                        request.runtime_value_without_current_image_projection(spec)
                    )
                    break
        for parameter_name, spec in deferred_object_specs:
            bound[parameter_name] = (
                request.current_image_aligned_object_label_runtime_value(
                    spec, alignment_image=alignment_image
                )
            )
        return bound


class MaskImageModule(
    MaskImageSpecialInputPolicy,
    ImageArtifactInputModule,
    ImageArtifactOutputModule,
    ObjectArtifactInputModule,
    CellProfilerModule,
):
    module_name = "MaskImage"
    function_name = "mask_image"
    validated = True
    contract = ProcessingContract.FLEXIBLE
    confidence = 1.0
    image_input_settings = ("Select the input image", "Select image for mask")
    object_input_settings = ("Select object for mask",)
    image_output_settings = ("Name the output image",)
    ignored_settings = (
        "Select the input image",
        "Name the output image",
        "Select object for mask",
        "Select image for mask",
    )
    setting_bindings = (
        SettingToKeywordBinding(
            "Use objects or an image as a mask?",
            "mask_source",
            cellprofiler_enum_value_setting_parser(MaskImageSource),
        ),
        SettingToKeywordBinding(
            "Invert the mask?", "invert_mask", parse_cellprofiler_bool
        ),
    )

    @classmethod
    def artifact_contract_outputs(cls, builder, module):
        primary_image_names = cls._mask_image_primary_image_names(module)
        outputs = []
        for output in super().artifact_contract_outputs(builder, module):
            output_spec = output.artifact_spec
            if (
                output_spec.artifact_type is ImageArtifactType
                and len(primary_image_names) == 1
            ):
                output_spec = replace(
                    output_spec,
                    relations=(
                        *output_spec.relations,
                        GroupLineageSourceRelation(
                            source=ArtifactSpecRef.input(
                                primary_image_names[0],
                                ImageArtifactType,
                            )
                        ),
                    ),
                )
                output = builder.declare_artifact(output_spec, module)
            outputs.append(output)
        return tuple(outputs)

    @classmethod
    def _mask_image_primary_image_names(cls, module):
        primary_setting = cls.image_input_setting_names()[0]
        declared_setting = cls.declared_setting_name(
            cls.declared_setting_value(primary_setting)
        )
        return tuple(
            name
            for value in setting_values(module, declared_setting)
            for name in split_symbol_names(value)
        )


class ResizeModule(
    ImageArtifactInputModule, ImageArtifactOutputModule, BinderSettingsSourceModule
):
    module_name = "Resize"
    function_name = "resize"
    validated = True
    function_variants = ("resize_volumetric",)
    contract = ProcessingContract.FLEXIBLE
    confidence = 1.0
    image_input_settings = (
        "Select the input image",
        "Select the image with the desired dimensions",
    )
    image_output_settings = ("Name the output image",)
    method_setting = SettingNameFamily("Resizing method")
    factor_setting = SettingNameFamily("Resizing factor")
    factor_x_setting = SettingNameFamily("X Resizing factor")
    factor_y_setting = SettingNameFamily("Y Resizing factor")
    factor_z_setting = SettingNameFamily("Z Resizing factor")
    width_setting = SettingNameFamily(
        "Width of the final image", aliases=("Width (x) of the final image",)
    )
    height_setting = SettingNameFamily(
        "Height of the final image", aliases=("Height (y) of the final image",)
    )
    planes_setting = SettingNameFamily("# of planes (z) in the final image")
    interpolation_setting = SettingNameFamily("Interpolation method")
    volumetric_settings = (factor_z_setting, planes_setting)

    @classmethod
    def settings_source(
        cls, module: "ModuleBlock", binder: "SettingsBinder"
    ) -> "CellProfilerKwargs":
        del binder
        kwargs: dict[str, Any] = {}
        resizing_method = optional_setting_value(module, cls.method_setting)
        if resizing_method is not None:
            kwargs["resize_method"] = cls.resize_method(resizing_method).value
        resizing_factor = optional_setting_value(module, cls.factor_setting)
        if resizing_factor is not None:
            factor = parse_cellprofiler_float(resizing_factor)
            kwargs["resizing_factor_x"] = factor
            kwargs["resizing_factor_y"] = factor
        factor_x = optional_setting_value(module, cls.factor_x_setting)
        if factor_x is not None:
            kwargs["resizing_factor_x"] = parse_cellprofiler_float(factor_x)
        factor_y = optional_setting_value(module, cls.factor_y_setting)
        if factor_y is not None:
            kwargs["resizing_factor_y"] = parse_cellprofiler_float(factor_y)
        factor_z = optional_setting_value(module, cls.factor_z_setting)
        if factor_z is not None:
            kwargs["resizing_factor_z"] = parse_cellprofiler_float(factor_z)
        width = optional_setting_value(module, cls.width_setting)
        if width is not None:
            kwargs["specific_width"] = parse_cellprofiler_int(width)
        height = optional_setting_value(module, cls.height_setting)
        if height is not None:
            kwargs["specific_height"] = parse_cellprofiler_int(height)
        planes = optional_setting_value(module, cls.planes_setting)
        if planes is not None:
            kwargs["specific_planes"] = parse_cellprofiler_int(planes)
        interpolation = optional_setting_value(module, cls.interpolation_setting)
        if interpolation is not None:
            kwargs["interpolation"] = coerce_cellprofiler_enum(
                InterpolationMethod, interpolation
            ).value
        return kwargs

    @staticmethod
    def resize_method(value: str) -> ResizeMethod:
        normalized = value.strip().lower()
        if "fraction" in normalized or "multiple" in normalized:
            return ResizeMethod.BY_FACTOR
        if (
            "specific" in normalized
            or "dimension" in normalized
            or "manual" in normalized
        ):
            return ResizeMethod.TO_SIZE
        return coerce_cellprofiler_enum(ResizeMethod, value)

    @classmethod
    def resolve_function(
        cls, module: "ModuleBlock", *, default_function_name: str | None = None
    ) -> "ResolvedModuleFunction":
        del default_function_name
        function_name = (
            cls.function_variants[0]
            if any(
                (setting_values(module, setting) for setting in cls.volumetric_settings)
            )
            else str(cls.function_name)
        )
        return super().resolve_function(module, default_function_name=function_name)


class TileModule(
    ImageArtifactInputModule,
    ImageArtifactOutputModule,
    ModuleSettingsSourceModule,
):
    module_name = "Tile"
    function_name = "tile"
    validated = True
    contract = ProcessingContract.PURE_3D
    confidence = 1.0
    image_input_settings = (
        "Select an input image",
        "Select an additional image to tile",
    )
    image_output_settings = ("Name the output image",)

    @staticmethod
    def settings_source(module: "ModuleBlock") -> "CellProfilerKwargs":
        assembly_method = module.get_setting(
            "Tile assembly method", "Within cycles"
        ).strip()
        normalized_method = normalize_cellprofiler_setting_name(assembly_method)
        if normalized_method != "within_cycles":
            raise NotImplementedError(
                f"Tile assembly method is not supported by the converter: {assembly_method!r}"
            )
        return {
            "rows": 1,
            "columns": 1,
            "place_first": "top_left",
            "tile_style": "row",
            "meander": False,
            "auto_rows": False,
            "auto_columns": True,
        }


from dataclasses import replace
from dataclasses import dataclass
from enum import Enum
import warnings
from typing import Any
from typing import Tuple
import numpy as np
from openhcs.core.aligned_image_payload import (
    aligned_payload_slice,
    payload_slices_for_alignment,
)
from openhcs.core.image_shapes import (
    is_color_image_slice,
    is_color_image_stack,
    is_grayscale_image_stack,
    is_grayscale_volume_slice,
    trailing_spatial_target_shape,
)
from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_values import image_mask_for_data_domain
from openhcs.core.runtime_values import image_payload_data
from openhcs.core.runtime_values import image_payload_mask
from openhcs.core.runtime_values import image_payload_metadata
from openhcs.core.runtime_values import RuntimeImagePayloadContext
from openhcs.core.source_matching import SourceImageSetIdentity
from openhcs.core.source_plane_alignment import (
    SourcePayloadPlaneIdentitySequence,
    SourcePlaneIdentitySequenceAlignment,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.core.pipeline.function_contracts import special_inputs
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer


class TileMethod(Enum):
    WITHIN_CYCLES = "within_cycles"
    ACROSS_CYCLES = "across_cycles"


class PlaceFirst(Enum):
    TOP_LEFT = "top_left"
    BOTTOM_LEFT = "bottom_left"
    TOP_RIGHT = "top_right"
    BOTTOM_RIGHT = "bottom_right"

    @property
    def row_from_bottom(self) -> bool:
        return self.value.startswith("bottom_")

    @property
    def column_from_right(self) -> bool:
        return self.value.endswith("_right")


class TileStyle(Enum):
    ROW = "row"
    COLUMN = "column"


class ResizeMethod(Enum):
    BY_FACTOR = "by_factor"
    TO_SIZE = "to_size"


class InterpolationMethod(Enum):
    NEAREST_NEIGHBOR = "nearest_neighbor"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"


class MaskSource(Enum):
    """CellProfiler MaskImage source type."""

    OBJECTS = "objects"
    IMAGE = "image"


class FlipMethod(Enum):
    NONE = "none"
    LEFT_TO_RIGHT = "left_to_right"
    TOP_TO_BOTTOM = "top_to_bottom"
    BOTH = "both"


class RotateMethod(Enum):
    NONE = "none"
    ANGLE = "angle"
    COORDINATES = "coordinates"


class AlignmentDirection(Enum):
    HORIZONTALLY = "horizontally"
    VERTICALLY = "vertically"


@dataclass(frozen=True, slots=True)
class TileSettings:
    rows: int
    columns: int
    place_first: PlaceFirst
    tile_style: TileStyle
    meander: bool
    auto_rows: bool
    auto_columns: bool

    def geometry(self, image_count: int) -> "TileGeometry":
        grid_rows, grid_columns = tile_grid_dimensions(
            image_count, self.rows, self.columns, self.auto_rows, self.auto_columns
        )
        return TileGeometry(
            rows=grid_rows,
            columns=grid_columns,
            tile_style=self.tile_style,
            place_first=self.place_first,
            meander=self.meander,
        )


@dataclass(frozen=True, slots=True)
class TileGeometry:
    rows: int
    columns: int
    tile_style: TileStyle
    place_first: PlaceFirst
    meander: bool

    @property
    def tile_count(self) -> int:
        return self.rows * self.columns

    def coordinates(self, image_index: int) -> tuple[int, int]:
        """Return row/column coordinates for one tile index."""
        if self.tile_style == TileStyle.ROW:
            tile_i = int(image_index / self.columns)
            tile_j = image_index % self.columns
            if self.meander and tile_i % 2 == 1:
                tile_j = self.columns - tile_j - 1
        else:
            tile_i = image_index % self.rows
            tile_j = int(image_index / self.rows)
            if self.meander and tile_j % 2 == 1:
                tile_i = self.rows - tile_i - 1
        if self.place_first.row_from_bottom:
            tile_i = self.rows - tile_i - 1
        if self.place_first.column_from_right:
            tile_j = self.columns - tile_j - 1
        return (tile_i, tile_j)


@dataclass(frozen=True, slots=True)
class ResizeGeometry:
    """CellProfiler resize geometry for pixels and per-pixel validity masks."""

    output_shape: tuple[int, ...]
    interpolation_order: int

    @classmethod
    def from_parameters(
        cls,
        input_shape: tuple[int, ...],
        *,
        resize_method: ResizeMethod,
        resizing_factors: tuple[float, ...],
        specific_shape: tuple[int, ...],
        interpolation: InterpolationMethod,
    ) -> "ResizeGeometry":
        if resize_method is ResizeMethod.BY_FACTOR:
            output_shape = tuple(
                (
                    int(np.round(axis_size * factor))
                    for axis_size, factor in zip(
                        input_shape, resizing_factors, strict=True
                    )
                )
            )
        else:
            output_shape = specific_shape
        return cls(
            output_shape=output_shape,
            interpolation_order=cls.resolve_interpolation_order(interpolation),
        )

    @classmethod
    def from_trailing_spatial_parameters(
        cls,
        input_shape: tuple[int, ...],
        *,
        resize_method: ResizeMethod,
        resizing_factors: tuple[float, ...],
        specific_shape: tuple[int, ...],
        interpolation: InterpolationMethod,
    ) -> "ResizeGeometry":
        resizing_factors, specific_shape = cls.effective_trailing_spatial_parameters(
            input_shape=input_shape,
            resizing_factors=resizing_factors,
            specific_shape=specific_shape,
        )
        spatial_rank = len(resizing_factors)
        if resize_method is ResizeMethod.BY_FACTOR:
            spatial_shape = input_shape[-spatial_rank:]
            output_spatial_shape = tuple(
                (
                    int(np.round(axis_size * factor))
                    for axis_size, factor in zip(
                        spatial_shape, resizing_factors, strict=True
                    )
                )
            )
        else:
            output_spatial_shape = specific_shape
        return cls(
            output_shape=trailing_spatial_target_shape(
                input_shape, output_spatial_shape
            ),
            interpolation_order=cls.resolve_interpolation_order(interpolation),
        )

    @staticmethod
    def effective_trailing_spatial_parameters(
        *,
        input_shape: tuple[int, ...],
        resizing_factors: tuple[float, ...],
        specific_shape: tuple[int, ...],
    ) -> tuple[tuple[float, ...], tuple[int, ...]]:
        """Project declared trailing spatial parameters onto the runtime array rank."""
        spatial_rank = min(len(input_shape), len(resizing_factors), len(specific_shape))
        if spatial_rank <= 0:
            raise ValueError(
                "Resize requires at least one runtime axis and one declared spatial axis."
            )
        return (resizing_factors[-spatial_rank:], specific_shape[-spatial_rank:])

    @staticmethod
    def resolve_interpolation_order(interpolation: InterpolationMethod) -> int:
        if interpolation is InterpolationMethod.NEAREST_NEIGHBOR:
            return 0
        if interpolation is InterpolationMethod.BILINEAR:
            return 1
        if interpolation is InterpolationMethod.BICUBIC:
            return 3
        raise TypeError(f"Unsupported Resize interpolation {interpolation!r}.")

    def resize_pixels(self, pixels: Any) -> np.ndarray:
        import skimage.transform

        return skimage.transform.resize(
            pixels,
            self.output_shape,
            order=self.interpolation_order,
            mode="symmetric",
            preserve_range=True,
        ).astype(np.asarray(pixels).dtype, copy=False)

    def resize_mask(
        self, mask: Any | None, *, input_shape: tuple[int, ...] | None = None
    ) -> np.ndarray | None:
        """Resize CP image validity masks using the same geometry as pixels."""
        import scipy.ndimage as ndi

        if mask is None:
            if input_shape is None:
                return None
            mask_array = np.ones(input_shape, dtype=bool)
        else:
            mask_array = np.asarray(mask, dtype=bool)
        output_shape = self.output_shape[-mask_array.ndim :]
        zoom = tuple(
            (
                output_size / input_size
                for output_size, input_size in zip(
                    output_shape, mask_array.shape, strict=True
                )
            )
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="It is recommended to use mode = grid-constant instead of constant when grid_mode is True.",
                category=UserWarning,
            )
            resized = ndi.zoom(
                mask_array.astype(np.float32),
                zoom,
                order=0,
                mode="constant",
                grid_mode=True,
            )
        return resized.astype(bool, copy=False)

    def resize_payload(self, image: Any) -> Any:
        pixels = image_payload_data(image)
        output_pixels = self.resize_pixels(pixels)
        mask = image_mask_for_data_domain(source_payload=image, data=pixels)
        metadata = image_payload_metadata(image).without_spatial_domain()
        if self.interpolation_order != 0:
            metadata = metadata.without_unit_interval_intensity_scale()
        return RuntimeImagePayloadContext(
            output_pixels,
            mask=self.resize_mask(mask, input_shape=tuple(np.asarray(pixels).shape)),
            metadata=metadata,
        ).payload()


@dataclass(frozen=True, slots=True)
class RotationResult:
    slice_index: int
    rotation_angle: float


@dataclass(frozen=True, slots=True)
class CellProfilerPlaneGeometry:
    """One CellProfiler XY plane coordinate system."""

    shape: tuple[int, int]
    spatial_rank: int = 2

    @classmethod
    def from_image_plane(cls, image: np.ndarray) -> "CellProfilerPlaneGeometry":
        image_array = collapse_singleton_plane_stack(
            np.asarray(image_payload_data(image))
        )
        if image_array.ndim not in {2, 3}:
            raise ValueError(
                f"CellProfiler image planes must be 2D grayscale, ZYX grayscale, or HWC color; got shape {image_array.shape!r}."
            )
        if is_grayscale_volume_slice(image_array):
            return cls(
                tuple((int(axis) for axis in image_array.shape[-2:])), spatial_rank=3
            )
        if image_array.ndim == 3 and (not is_color_image_slice(image_array)):
            raise ValueError(
                f"CellProfiler 3D image planes must be HWC color; got shape {image_array.shape!r}."
            )
        return cls(tuple((int(axis) for axis in image_array.shape[:2])))

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        if self.spatial_rank == 2:
            return self.shape
        if self.spatial_rank == 3:
            return self.shape
        raise ValueError(f"Unsupported CellProfiler spatial rank {self.spatial_rank}.")

    def binary_mask(
        self, mask: np.ndarray, *, threshold: float = 0.5, labels: bool = False
    ) -> np.ndarray:
        mask_array = binary_mask_plane(mask, threshold=threshold, labels=labels)
        if self.spatial_rank == 3 and mask_array.ndim == 3:
            return align_volume_mask_to_shape(mask_array, self.shape)
        if self.spatial_rank == 3 and mask_array.ndim == 2:
            return align_binary_mask_to_shape(mask_array, self.shape)
        return align_binary_mask_to_shape(mask_array, self.shape)

    def label_plane(self, labels: np.ndarray) -> np.ndarray:
        return align_label_plane_to_shape(labels.astype(np.int32), self.shape)


@dataclass(frozen=True, slots=True)
class CellProfilerImageMaskPlane:
    """One image plane paired with a binary mask in the same XY geometry."""

    image: np.ndarray
    mask: np.ndarray

    def __post_init__(self) -> None:
        geometry = CellProfilerPlaneGeometry.from_image_plane(self.image)
        if tuple(self.mask.shape[-2:]) != geometry.shape:
            raise ValueError(
                f"CellProfilerImageMaskPlane mask shape must match image spatial shape; got mask {self.mask.shape!r} for image {geometry.shape!r}."
            )
        if geometry.spatial_rank == 2 and self.mask.ndim != 2:
            raise ValueError(
                f"CellProfilerImageMaskPlane 2D images require 2D masks; got mask {self.mask.shape!r}."
            )


def aligned_image_mask_planes(
    image: np.ndarray, mask: np.ndarray, *, threshold: float = 0.5, labels: bool = False
) -> tuple[CellProfilerImageMaskPlane, ...]:
    """Align a mask payload to each image plane using CellProfiler slice rules."""
    image_planes = payload_slices_for_alignment(image)
    mask_planes = payload_slices_for_alignment(mask)
    source_aligned_mask_planes = source_aligned_target_planes(image_planes, mask_planes)
    if source_aligned_mask_planes is not None:
        mask_planes = source_aligned_mask_planes
    if len(image_planes) == 1 and len(mask_planes) > 1:
        image_plane = image_planes[0]
        geometry = CellProfilerPlaneGeometry.from_image_plane(image_plane)
        matched_mask_plane = single_source_aligned_target_plane(
            image_plane, mask_planes
        )
        if matched_mask_plane is not None:
            return (
                CellProfilerImageMaskPlane(
                    image=image_plane,
                    mask=geometry.binary_mask(
                        matched_mask_plane, threshold=threshold, labels=labels
                    ),
                ),
            )
        if geometry.spatial_rank == 3:
            projected_volume_mask = np.stack(
                tuple(
                    (
                        geometry.binary_mask(
                            mask_plane, threshold=threshold, labels=labels
                        )
                        for mask_plane in mask_planes
                    )
                ),
                axis=0,
            )
            return (
                CellProfilerImageMaskPlane(
                    image=image_plane, mask=projected_volume_mask
                ),
            )
        projected_mask = np.any(
            np.stack(
                tuple(
                    (
                        geometry.binary_mask(
                            mask_plane, threshold=threshold, labels=labels
                        )
                        for mask_plane in mask_planes
                    )
                )
            ),
            axis=0,
        )
        return (CellProfilerImageMaskPlane(image=image_plane, mask=projected_mask),)
    if len(mask_planes) not in {1, len(image_planes)}:
        projected_mask_planes = _project_volume_mask_planes(
            image_planes, mask_planes, threshold=threshold, labels=labels
        )
        if projected_mask_planes is not None:
            return projected_mask_planes
        projected_mask_planes = _project_flat_mask_plane_groups(
            image_planes, mask_planes, threshold=threshold, labels=labels
        )
        if projected_mask_planes is not None:
            return projected_mask_planes
        raise ValueError(
            f"CellProfiler mask payload must have one plane or match image plane count; got image count {len(image_planes)} and mask count {len(mask_planes)}."
        )
    return tuple(
        (
            CellProfilerImageMaskPlane(
                image=image_plane,
                mask=CellProfilerPlaneGeometry.from_image_plane(
                    image_plane
                ).binary_mask(
                    aligned_payload_slice(mask_planes, plane_index),
                    threshold=threshold,
                    labels=labels,
                ),
            )
            for plane_index, image_plane in enumerate(image_planes)
        )
    )


def source_aligned_target_planes(
    image_planes: tuple[Any, ...], target_planes: tuple[Any, ...]
) -> tuple[Any, ...] | None:
    """Return target planes ordered by shared source-plane identity."""
    target_indexes = SourcePlaneIdentitySequenceAlignment(
        SourcePayloadPlaneIdentitySequence.from_payloads(
            image_planes, SourceImageSetIdentity.DEFAULT_POLICY
        ),
        SourcePayloadPlaneIdentitySequence.from_payloads(
            target_planes, SourceImageSetIdentity.DEFAULT_POLICY
        ),
    ).target_indexes_for_image_planes()
    if target_indexes is None:
        return None
    return tuple((target_planes[index] for index in target_indexes))


def single_source_aligned_target_plane(
    image_plane: Any, target_planes: tuple[Any, ...]
) -> Any | None:
    """Return one target plane matching a single image plane by source identity."""
    aligned = source_aligned_target_planes((image_plane,), target_planes)
    if aligned is None:
        return None
    if len(aligned) != 1:
        raise ValueError(
            "Source-plane alignment returned an invalid single-plane result."
        )
    return aligned[0]


def _project_volume_mask_planes(
    image_planes: tuple[Any, ...],
    mask_planes: tuple[Any, ...],
    *,
    threshold: float,
    labels: bool,
) -> tuple[CellProfilerImageMaskPlane, ...] | None:
    """Project stacked volume masks onto each image plane when ranks permit."""
    image_count = len(image_planes)
    if image_count <= 1:
        return None
    mask_arrays = tuple((np.asarray(image_payload_data(mask)) for mask in mask_planes))
    if not mask_arrays or any((mask.ndim < 3 for mask in mask_arrays)):
        return None
    if any((mask.shape[0] != image_count for mask in mask_arrays)):
        return _project_all_volume_masks_to_image_planes(
            image_planes, mask_arrays, threshold=threshold, labels=labels
        )
    return tuple(
        (
            CellProfilerImageMaskPlane(
                image=image_plane,
                mask=np.any(
                    np.stack(
                        tuple(
                            (
                                CellProfilerPlaneGeometry.from_image_plane(
                                    image_plane
                                ).binary_mask(
                                    mask_array[plane_index],
                                    threshold=threshold,
                                    labels=labels,
                                )
                                for mask_array in mask_arrays
                            )
                        )
                    ),
                    axis=0,
                ),
            )
            for plane_index, image_plane in enumerate(image_planes)
        )
    )


def _project_flat_mask_plane_groups(
    image_planes: tuple[Any, ...],
    mask_planes: tuple[Any, ...],
    *,
    threshold: float,
    labels: bool,
) -> tuple[CellProfilerImageMaskPlane, ...] | None:
    """Project flattened grouped mask stacks onto matching image-plane indices."""
    image_count = len(image_planes)
    mask_count = len(mask_planes)
    if image_count <= 1 or mask_count <= image_count:
        return None
    if mask_count % image_count != 0:
        return None
    group_count = mask_count // image_count
    return tuple(
        (
            CellProfilerImageMaskPlane(
                image=image_plane,
                mask=np.any(
                    np.stack(
                        tuple(
                            (
                                CellProfilerPlaneGeometry.from_image_plane(
                                    image_plane
                                ).binary_mask(
                                    mask_planes[
                                        group_index * image_count + plane_index
                                    ],
                                    threshold=threshold,
                                    labels=labels,
                                )
                                for group_index in range(group_count)
                            )
                        )
                    ),
                    axis=0,
                ),
            )
            for plane_index, image_plane in enumerate(image_planes)
        )
    )


def _project_all_volume_masks_to_image_planes(
    image_planes: tuple[Any, ...],
    mask_arrays: tuple[np.ndarray, ...],
    *,
    threshold: float,
    labels: bool,
) -> tuple[CellProfilerImageMaskPlane, ...]:
    """Collapse all mask leading axes, then broadcast the XY mask to each image."""
    return tuple(
        (
            CellProfilerImageMaskPlane(
                image=image_plane,
                mask=np.any(
                    np.stack(
                        tuple(
                            (
                                _project_mask_array_to_geometry(
                                    mask_array,
                                    CellProfilerPlaneGeometry.from_image_plane(
                                        image_plane
                                    ),
                                    threshold=threshold,
                                    labels=labels,
                                )
                                for mask_array in mask_arrays
                            )
                        )
                    ),
                    axis=0,
                ),
            )
            for image_plane in image_planes
        )
    )


def _project_mask_array_to_geometry(
    mask_array: np.ndarray,
    geometry: CellProfilerPlaneGeometry,
    *,
    threshold: float,
    labels: bool,
) -> np.ndarray:
    """Project every leading-axis mask plane into one XY geometry."""
    mask_planes = mask_array.reshape((-1, *mask_array.shape[-2:]))
    return np.any(
        np.stack(
            tuple(
                (
                    geometry.binary_mask(mask_plane, threshold=threshold, labels=labels)
                    for mask_plane in mask_planes
                )
            )
        ),
        axis=0,
    )


def restore_image_mask_planes(
    original_image: np.ndarray, masked_planes: tuple[np.ndarray, ...]
) -> np.ndarray:
    """Restore masked image planes to the original image payload rank."""
    if not masked_planes:
        raise ValueError("Cannot restore an empty CellProfiler image plane set.")
    if not _is_stack_payload(original_image) and len(masked_planes) == 1:
        return masked_planes[0]
    return np.stack(masked_planes).astype(masked_planes[0].dtype, copy=False)


def binary_mask_plane(
    mask: np.ndarray, *, threshold: float = 0.5, labels: bool = False
) -> np.ndarray:
    """Convert one CellProfiler mask/label plane to a 2D boolean mask."""
    mask = collapse_singleton_plane_stack(np.asarray(mask))
    if labels:
        return mask > 0
    if is_color_image_slice(mask):
        return np.any(mask > threshold, axis=-1)
    unique_values = np.unique(mask)
    if len(unique_values) <= 2 and set(unique_values).issubset({0, 1, False, True}):
        return mask > 0
    return mask > threshold


def align_binary_mask_to_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor align a boolean mask to an XY shape."""
    if mask.shape == shape:
        return mask.astype(bool, copy=False)
    return resize_nearest(mask.astype(np.uint8), shape).astype(bool)


def align_volume_mask_to_shape(
    mask: np.ndarray, shape_yx: tuple[int, int]
) -> np.ndarray:
    """Nearest-neighbor align every Z plane of a ZYX boolean mask."""
    if mask.shape[-2:] == shape_yx:
        return mask.astype(bool, copy=False)
    return np.stack(
        tuple((align_binary_mask_to_shape(plane, shape_yx) for plane in mask)), axis=0
    )


def align_label_plane_to_shape(
    labels: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Nearest-neighbor align a dense label plane to an XY shape."""
    labels = collapse_singleton_plane_stack(np.asarray(labels))
    if 0 in labels.shape:
        return np.zeros(shape, dtype=np.int32)
    if labels.shape == shape:
        return labels.astype(np.int32, copy=False)
    return resize_nearest(labels, shape).astype(np.int32)


def resize_nearest(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a discrete 2D payload without interpolation artifacts."""
    from skimage.transform import resize

    return resize(image, shape, order=0, preserve_range=True, anti_aliasing=False)


def collapse_singleton_plane_stack(payload: Any) -> Any:
    """Collapse one-plane label/mask stacks to CellProfiler's 2D plane form."""
    payload_array = np.asarray(payload)
    if payload_array.ndim == 3 and payload_array.shape[0] == 1:
        return payload_array[0]
    return payload


@numpy(contract=ProcessingContract.FLEXIBLE)
@special_inputs("mask")
def mask_image(
    image: np.ndarray,
    mask: np.ndarray,
    mask_source: MaskSource = MaskSource.IMAGE,
    invert_mask: bool = False,
    binary_threshold: float = 0.5,
) -> np.ndarray:
    """Mask an image using CellProfiler image/object mask semantics."""
    mask_source = coerce_cellprofiler_enum(MaskSource, mask_source)
    masked_plane_results = tuple(
        (
            masked_image_plane(plane.image, plane.mask, invert_mask=invert_mask)
            for plane in aligned_image_mask_planes(
                image,
                mask,
                threshold=binary_threshold,
                labels=mask_source is MaskSource.OBJECTS,
            )
        )
    )
    masked_data = restore_image_mask_planes(
        image_payload_data(image), tuple((result[0] for result in masked_plane_results))
    )
    output_mask = restore_image_mask_planes(
        image_payload_data(image), tuple((result[1] for result in masked_plane_results))
    )
    return RuntimeImagePayloadContext(
        masked_data,
        mask=output_mask,
        metadata=replace(image_payload_metadata(image), mask_defines_border=True),
    ).payload()


def masked_image_plane(
    image: np.ndarray, binary_mask: np.ndarray, *, invert_mask: bool
) -> tuple[np.ndarray, np.ndarray]:
    image_data = collapse_singleton_plane_stack(image_payload_data(image))
    if invert_mask:
        binary_mask = ~binary_mask
    projected_mask = image_mask_for_data_domain(
        source_payload=image_data, data=image_data, explicit_mask=binary_mask
    )
    if projected_mask is None:
        raise ValueError(
            f"MaskImage mask cannot be projected into the image data domain; got mask={np.shape(binary_mask)!r}, image={np.shape(image_data)!r}."
        )
    binary_mask = np.asarray(projected_mask, dtype=bool)
    existing_mask = image_payload_mask(image)
    if existing_mask is not None:
        existing_projected_mask = image_mask_for_data_domain(
            source_payload=image_data,
            data=image_data,
            explicit_mask=collapse_singleton_plane_stack(existing_mask),
        )
        if existing_projected_mask is not None:
            binary_mask = binary_mask & np.asarray(existing_projected_mask, dtype=bool)
    masked = image_data.copy()
    masked[~binary_mask] = 0
    return (masked, np.asarray(binary_mask, dtype=bool))


@numpy(contract=ProcessingContract.PURE_2D)
def mask_image_with_binary(image: np.ndarray, invert_mask: bool = False) -> np.ndarray:
    """Return a binary mask plane, optionally inverted."""
    binary_mask = image > 0.5
    if invert_mask:
        binary_mask = ~binary_mask
    return binary_mask.astype(np.float32)


@numpy(contract=ProcessingContract.PURE_3D)
def mask_image_stacked(
    image: np.ndarray, invert_mask: bool = False, binary_threshold: float = 0.5
) -> np.ndarray:
    """Mask an image where image[0] is pixels and image[1] is mask."""
    img = image[0]
    mask = image[1]
    binary_mask = binary_mask_plane(mask, threshold=binary_threshold)
    if invert_mask:
        binary_mask = ~binary_mask
    result = img.copy()
    result[~binary_mask] = 0
    return result[np.newaxis, ...]


def tile_grid_dimensions(
    image_count: int, rows: int, columns: int, auto_rows: bool, auto_columns: bool
) -> tuple[int, int]:
    """Calculate CellProfiler Tile grid dimensions from auto/manual settings."""
    if auto_rows:
        if auto_columns:
            row_count = int(np.sqrt(image_count))
            column_count = int((image_count + row_count - 1) / row_count)
            return (row_count, column_count)
        column_count = columns
        row_count = int((image_count + column_count - 1) / column_count)
        return (row_count, column_count)
    if auto_columns:
        row_count = rows
        column_count = int((image_count + row_count - 1) / row_count)
        return (row_count, column_count)
    return (rows, columns)


def put_tile(
    pixels: np.ndarray,
    output_pixels: np.ndarray,
    image_index: int,
    geometry: TileGeometry,
) -> None:
    """Place one image plane into a CellProfiler Tile output montage."""
    tile_height = int(output_pixels.shape[0] / geometry.rows)
    tile_width = int(output_pixels.shape[1] / geometry.columns)
    tile_i, tile_j = geometry.coordinates(image_index)
    tile_i *= tile_height
    tile_j *= tile_width
    img_height = min(tile_height, pixels.shape[0])
    img_width = min(tile_width, pixels.shape[1])
    output_pixels[tile_i : tile_i + img_height, tile_j : tile_j + img_width] = pixels[
        :img_height, :img_width
    ]


def tile_output_shape(
    image: np.ndarray, output_height: int, output_width: int
) -> tuple[int, ...]:
    """Return CellProfiler Tile output shape for grayscale or color stacks."""
    if image.ndim == 4:
        return (output_height, output_width, image.shape[3])
    return (output_height, output_width)


@numpy(contract=ProcessingContract.PURE_3D)
def tile(
    image: np.ndarray,
    rows: int = 8,
    columns: int = 12,
    place_first: PlaceFirst = PlaceFirst.TOP_LEFT,
    tile_style: TileStyle = TileStyle.ROW,
    meander: bool = False,
    auto_rows: bool = False,
    auto_columns: bool = False,
) -> np.ndarray:
    """Tile multiple images together to form a CellProfiler montage image."""
    if image.ndim not in {3, 4}:
        raise ValueError(
            f"Tile expects an image stack shaped (N, H, W) or (N, H, W, C), got {image.shape!r}."
        )
    num_images = image.shape[0]
    if num_images == 0:
        raise ValueError("No images provided for tiling")
    geometry = TileSettings(
        rows=rows,
        columns=columns,
        place_first=place_first,
        tile_style=tile_style,
        meander=meander,
        auto_rows=auto_rows,
        auto_columns=auto_columns,
    ).geometry(num_images)
    if geometry.tile_count < num_images:
        raise ValueError(
            f"Grid size ({geometry.rows}x{geometry.columns}={geometry.tile_count}) is too small for {num_images} images"
        )
    tile_height = image.shape[1]
    tile_width = image.shape[2]
    output_height = tile_height * geometry.rows
    output_width = tile_width * geometry.columns
    output_pixels = np.zeros(
        tile_output_shape(image, output_height, output_width), dtype=image.dtype
    )
    for image_index in range(num_images):
        put_tile(image[image_index], output_pixels, image_index, geometry)
    return output_pixels[np.newaxis, ...]


@numpy(contract=ProcessingContract.PURE_2D)
def resize(
    image: np.ndarray,
    resize_method: ResizeMethod = ResizeMethod.BY_FACTOR,
    resizing_factor_x: float = 0.25,
    resizing_factor_y: float = 0.25,
    specific_width: int = 100,
    specific_height: int = 100,
    interpolation: InterpolationMethod = InterpolationMethod.NEAREST_NEIGHBOR,
) -> np.ndarray:
    """Resize a CellProfiler image plane by factor or explicit dimensions."""
    resize_method = coerce_cellprofiler_enum(ResizeMethod, resize_method)
    interpolation = coerce_cellprofiler_enum(InterpolationMethod, interpolation)
    pixels = image_payload_data(image)
    geometry = ResizeGeometry.from_parameters(
        tuple(np.asarray(pixels).shape[:2]),
        resize_method=resize_method,
        resizing_factors=(resizing_factor_y, resizing_factor_x),
        specific_shape=(specific_height, specific_width),
        interpolation=interpolation,
    )
    return geometry.resize_payload(image)


@numpy(contract=ProcessingContract.PURE_3D)
def resize_volumetric(
    image: np.ndarray,
    resize_method: ResizeMethod = ResizeMethod.BY_FACTOR,
    resizing_factor_x: float = 0.25,
    resizing_factor_y: float = 0.25,
    resizing_factor_z: float = 0.25,
    specific_width: int = 100,
    specific_height: int = 100,
    specific_planes: int = 10,
    interpolation: InterpolationMethod = InterpolationMethod.NEAREST_NEIGHBOR,
) -> np.ndarray:
    """Resize a CellProfiler ZYX image volume by factor or explicit dimensions."""
    resize_method = coerce_cellprofiler_enum(ResizeMethod, resize_method)
    interpolation = coerce_cellprofiler_enum(InterpolationMethod, interpolation)
    pixels = image_payload_data(image)
    geometry = ResizeGeometry.from_trailing_spatial_parameters(
        tuple(np.asarray(pixels).shape),
        resize_method=resize_method,
        resizing_factors=(resizing_factor_z, resizing_factor_y, resizing_factor_x),
        specific_shape=(specific_planes, specific_height, specific_width),
        interpolation=interpolation,
    )
    return geometry.resize_payload(image)


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(
    (
        "rotation_results",
        csv_materializer(
            fields=["slice_index", "rotation_angle"], analysis_type="rotation"
        ),
    )
)
def flip_and_rotate(
    image: np.ndarray,
    flip_method: FlipMethod = FlipMethod.NONE,
    rotate_method: RotateMethod = RotateMethod.NONE,
    rotation_angle: float = 0.0,
    first_pixel_x: int = 0,
    first_pixel_y: int = 0,
    second_pixel_x: int = 0,
    second_pixel_y: int = 100,
    alignment_direction: AlignmentDirection = AlignmentDirection.HORIZONTALLY,
    crop_rotated_edges: bool = True,
) -> Tuple[np.ndarray, RotationResult]:
    """Flip and/or rotate a CellProfiler image plane."""
    from scipy.ndimage import rotate as scipy_rotate

    pixel_data = image.copy()
    if flip_method != FlipMethod.NONE:
        if flip_method == FlipMethod.LEFT_TO_RIGHT:
            pixel_data = np.flip(pixel_data, axis=1)
        elif flip_method == FlipMethod.TOP_TO_BOTTOM:
            pixel_data = np.flip(pixel_data, axis=0)
        elif flip_method == FlipMethod.BOTH:
            pixel_data = np.flip(np.flip(pixel_data, axis=1), axis=0)
    angle = 0.0
    if rotate_method != RotateMethod.NONE:
        if rotate_method == RotateMethod.ANGLE:
            angle = rotation_angle
        elif rotate_method == RotateMethod.COORDINATES:
            xdiff = second_pixel_x - first_pixel_x
            ydiff = second_pixel_y - first_pixel_y
            if alignment_direction == AlignmentDirection.VERTICALLY:
                angle = -np.arctan2(ydiff, xdiff) * 180.0 / np.pi
            else:
                angle = np.arctan2(xdiff, ydiff) * 180.0 / np.pi
        if angle != 0.0:
            pixel_data = scipy_rotate(pixel_data, angle, reshape=True, order=1)
            if crop_rotated_edges:
                crop_mask = (
                    scipy_rotate(np.ones(image.shape[:2]), angle, reshape=True) > 0.5
                )
                half = (np.array(crop_mask.shape) // 2).astype(int)
                quartercrop = crop_mask[half[0] :, half[1] :]
                ci = np.cumsum(quartercrop, 0)
                cj = np.cumsum(quartercrop, 1)
                carea_d = ci * cj
                carea_d[quartercrop == 0] = 0
                quartercrop_u = crop_mask[
                    crop_mask.shape[0] - half[0] - 1 :: -1, half[1] :
                ]
                ci = np.cumsum(quartercrop_u, 0)
                cj = np.cumsum(quartercrop_u, 1)
                carea_u = ci * cj
                carea_u[quartercrop_u == 0] = 0
                min_shape = min(carea_d.shape[0], carea_u.shape[0])
                carea = carea_d[:min_shape] + carea_u[:min_shape]
                if carea.size > 0:
                    max_carea = np.max(carea)
                    if max_carea > 0:
                        max_area_idx = np.argwhere(carea == max_carea)[0] + half
                        min_i = max(crop_mask.shape[0] - max_area_idx[0] - 1, 0)
                        max_i = max_area_idx[0] + 1
                        min_j = max(crop_mask.shape[1] - max_area_idx[1] - 1, 0)
                        max_j = max_area_idx[1] + 1
                        pixel_data = pixel_data[min_i:max_i, min_j:max_j]
    return (
        pixel_data.astype(np.float32),
        RotationResult(slice_index=0, rotation_angle=angle),
    )


def _is_stack_payload(payload: Any) -> bool:
    return is_grayscale_image_stack(payload) or is_color_image_stack(payload)


class FlipAndRotateModule(CellProfilerModule):
    module_name = "FlipAndRotate"
    function_name = "flip_and_rotate"
    validated = True
    confidence = 1.0
