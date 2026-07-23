"""Shared CellProfiler image-plane geometry semantics."""

from __future__ import annotations

from typing import Annotated, TYPE_CHECKING, ClassVar

from openhcs.core.steps.function_runtime import RuntimeCallableKwargs
from collections.abc import Callable
from dataclasses import replace
from enum import Enum
from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    GroupLineageSourceRelation,
    ImageArtifactType,
    ImageMeasurementSubjectRelation,
    InputStackBroadcastSourceRelation,
    ObjectLabelsArtifactType,
    SourceStackLineageSourceRelation,
)
from openhcs.core.source_bindings import StepSourceBindingsConfig
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
from openhcs.interop.cellprofiler.module_settings import (
    BoundModuleSettings,
)
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    MeasurementFeatureRecord,
    NoObjectNameMeasurementRecordMixin,
    ProducedImageMeasurementRecordMixin,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    FormattingMeasurementFeatureTemplate,
    ModuleOwnedResultMeasurementRows,
)
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
    MeasurementProjectedColumnarRows,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
)
from openhcs.interop.cellprofiler.setting_names import (
    optional_setting_value,
    required_setting_value,
    setting_values,
)
from openhcs.core.callable_contract import CallableContract

if TYPE_CHECKING:
    from openhcs.core.function_patterns import FunctionInvocationKey
    from openhcs.interop.cellprofiler.parser import ModuleBlock
    from openhcs.interop.cellprofiler.settings_binder import SettingsBinder


class MaskSource(Enum):
    """Mask source domains exposed by MaskImage settings."""

    OBJECTS = "objects"
    IMAGE = "image"


class ResizeDimensionSpecification(Enum):
    """Resize target-size authorities exposed by CellProfiler settings."""

    MANUAL = "Manual"
    IMAGE = "Image"


class MaskImageModule(
    ObjectArtifactInputModule,
    CellProfilerModule,
):
    module_name = "MaskImage"
    function_name = "mask_image"
    validated = True
    confidence = 1.0
    input_image_setting = SettingNameFamily("Select the input image")
    masking_image_setting = SettingNameFamily("Select image for mask")
    masking_objects_setting = SettingNameFamily("Select object for mask")
    output_image_setting = SettingNameFamily("Name the output image")
    mask_source_setting = SettingNameFamily("Use objects or an image as a mask?")
    input_image_binding = SettingToKeywordBinding.input(
        input_image_setting, ImageArtifactType
    )
    masking_image_binding = SettingToKeywordBinding.input(
        masking_image_setting,
        ImageArtifactType,
        runtime_parameter_name="mask",
    )
    masking_objects_binding = SettingToKeywordBinding.input(
        masking_objects_setting,
        ObjectLabelsArtifactType,
        runtime_parameter_name="mask",
    )
    output_image_binding = SettingToKeywordBinding.output(
        output_image_setting, ImageArtifactType
    )
    mask_source_binding = SettingToKeywordBinding(
        mask_source_setting,
        "mask_source",
        cellprofiler_enum_value_setting_parser(MaskSource),
    )
    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (input_image_binding, masking_image_binding, masking_objects_binding,output_image_binding,mask_source_binding,
        SettingToKeywordBinding(
            "Invert the mask?", "invert_mask", parse_cellprofiler_bool
        ),)

    @classmethod
    def active_artifact_bindings(
        cls,
        module: "ModuleBlock | None" = None,
        *,
        invocation_key: "FunctionInvocationKey | None" = None,
    ) -> tuple[SettingToKeywordBinding, ...]:
        """Return the image or object mask selected by this module."""

        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        if cls._mask_source(module) is MaskSource.IMAGE:
            inactive = cls.masking_objects_binding
        else:
            inactive = cls.masking_image_binding
        return tuple(
            binding for binding in bindings if binding is not inactive
        )

    @classmethod
    def _mask_source(cls, module: "ModuleBlock") -> MaskSource:
        values = setting_values(module, cls.mask_source_setting)
        if len(values) > 1:
            raise ValueError(
                f"MaskImage declares multiple mask-source rows: {values!r}."
            )
        return coerce_cellprofiler_enum(
            MaskSource,
            values[0] if values else MaskSource.IMAGE.value,
        )

    @classmethod
    def finalize_artifact_contract_inputs(
        cls,
        module,
        *,
        invocation_key,
        step_context,
        artifact_inputs: ArtifactSpecCollection,
    ):
        inputs = ArtifactSpecCollection(
            super().finalize_artifact_contract_inputs(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
            )
        )
        if cls._mask_source(module) is not MaskSource.IMAGE:
            return inputs.specs

        image_names = setting_values(module, cls.input_image_setting)
        mask_names = setting_values(module, cls.masking_image_setting)
        if len(image_names) != 1 or len(mask_names) != 1:
            raise ValueError(
                f"Module {module.name}({module.module_num}) requires exactly one "
                "primary image and one image-mask input to declare stack broadcast; "
                f"got primary={image_names!r}, mask={mask_names!r}."
            )
        image = inputs.require_by_name_and_artifact_type(
            image_names[0],
            ImageArtifactType,
        )
        mask = inputs.require_by_name_and_artifact_type(
            mask_names[0],
            ImageArtifactType,
        )
        if mask.ref() == image.ref():
            raise ValueError(
                f"Module {module.name}({module.module_num}) cannot declare image "
                f"{image.name!r} as both the primary image and image-mask input; "
                "stack broadcast requires an exact distinct source."
            )
        related_mask = mask.with_group_scope_relation(
            InputStackBroadcastSourceRelation(source=image.ref())
        )
        return tuple(
            related_mask if spec.ref() == mask.ref() else spec for spec in inputs.specs
        )


def _parse_resize_method_setting(value: str) -> str:
    normalized = value.strip().lower()
    if "fraction" in normalized or "multiple" in normalized:
        return "by_factor"
    if "specific" in normalized or "dimension" in normalized or "manual" in normalized:
        return "to_size"
    return coerce_cellprofiler_enum(ResizeMethod, value).value


def _parse_resize_interpolation_setting(value: str) -> str:
    return coerce_cellprofiler_enum(InterpolationMethod, value).value


def _parse_tile_first_corner_setting(value: str) -> str:
    return coerce_cellprofiler_enum(PlaceFirst, value).value


def _parse_tile_direction_setting(value: str) -> str:
    return coerce_cellprofiler_enum(TileStyle, value).value


class ResizeModule(
    CellProfilerModule):
    module_name = "Resize"
    function_name = "resize"
    validated = True
    function_variants = ("resize_volumetric",)
    confidence = 1.0
    input_image_setting = SettingNameFamily("Select the input image")
    desired_dimensions_image_setting = SettingNameFamily(
        "Select the image with the desired dimensions"
    )
    output_image_setting = SettingNameFamily("Name the output image")
    input_image_binding = SettingToKeywordBinding.input(
        input_image_setting, ImageArtifactType
    )
    desired_dimensions_image_binding = SettingToKeywordBinding.input(
        desired_dimensions_image_setting, ImageArtifactType
    )
    output_image_binding = SettingToKeywordBinding.output(
        output_image_setting, ImageArtifactType
    )
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
    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (input_image_binding,
        desired_dimensions_image_binding,output_image_binding,SettingToKeywordBinding(
            method_setting,
            "resize_method",
            _parse_resize_method_setting,
        ),
        SettingToKeywordBinding(
            factor_x_setting,
            "resizing_factor_x",
            parse_cellprofiler_float,
        ),
        SettingToKeywordBinding(
            factor_y_setting,
            "resizing_factor_y",
            parse_cellprofiler_float,
        ),
        SettingToKeywordBinding(
            factor_z_setting,
            "resizing_factor_z",
            parse_cellprofiler_float,
        ),
        SettingToKeywordBinding(
            width_setting,
            "specific_width",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            height_setting,
            "specific_height",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            planes_setting,
            "specific_planes",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            interpolation_setting,
            "interpolation",
            _parse_resize_interpolation_setting,
        ),)
    dimension_specification_setting = SettingNameFamily(
        "Method to specify the dimensions"
    )
    additional_image_count_setting = SettingNameFamily("Additional image count")
    volumetric_settings = (factor_z_setting, planes_setting)

    @classmethod
    def bind_settings(cls, module, *, binder):
        """Bind scalar rows and the legacy shared X/Y factor row."""

        dimension_method = optional_setting_value(
            module, cls.dimension_specification_setting
        )
        resizing_method = optional_setting_value(module, cls.method_setting)
        if (
            dimension_method is not None
            and resizing_method is not None
            and cls.resize_method(resizing_method) is ResizeMethod.TO_SIZE
            and "manual" not in dimension_method.casefold()
        ):
            raise NotImplementedError(
                "Resize-to-size currently requires manually declared dimensions."
            )
        bound = cls._bind_declared_settings(module, binder=binder)
        bound = bound.with_kwargs(cls._resize_kwargs(module, binder))
        bound = bound.with_consumed_settings(
            cls.factor_setting,
            cls.dimension_specification_setting,
            cls.additional_image_count_setting,
        )
        return cls._finalize_bound_settings(module, binder=binder, bound=bound)

    @classmethod
    def _resize_kwargs(
        cls, module: "ModuleBlock", binder: "SettingsBinder"
    ) -> "RuntimeCallableKwargs":
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
        return ResizeMethod(_parse_resize_method_setting(value))

    @classmethod
    def resolve_function(
        cls,
        module: "ModuleBlock",
        *,
        contract: "CallableContract",
        source_bindings: "StepSourceBindingsConfig",
    ) -> Callable[..., object]:
        del contract, source_bindings
        function_name = (
            cls.function_variants[0]
            if any(
                (setting_values(module, setting) for setting in cls.volumetric_settings)
            )
            else str(cls.function_name)
        )
        return cls.require_callable(function_name)

    @classmethod
    def active_artifact_bindings(
        cls,
        module: "ModuleBlock | None" = None,
        *,
        invocation_key: "FunctionInvocationKey | None" = None,
    ) -> tuple[SettingToKeywordBinding, ...]:
        """Declare the size-reference role selected by Resize behavior."""

        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        resize_method = optional_setting_value(module, cls.method_setting)
        dimension_specification = optional_setting_value(
            module,
            cls.dimension_specification_setting,
        )
        uses_dimensions_image = (
            resize_method is not None
            and cls.resize_method(resize_method) is ResizeMethod.TO_SIZE
            and dimension_specification is not None
            and coerce_cellprofiler_enum(
                ResizeDimensionSpecification,
                dimension_specification,
            )
            is ResizeDimensionSpecification.IMAGE
        )
        return tuple(
            binding
            for binding in bindings
            if uses_dimensions_image
            or binding is not cls.desired_dimensions_image_binding
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
        """Preserve plane identity unless volumetric resizing changes that axis."""

        del invocation_key, step_context, binding, name, output_position
        source = artifact_inputs.require_by_name_and_artifact_type(
            required_setting_value(module, cls.input_image_setting),
            ImageArtifactType,
        )
        relation_type = SourceStackLineageSourceRelation
        if any(setting_values(module, item) for item in cls.volumetric_settings):
            resize_method = cls.resize_method(
                required_setting_value(module, cls.method_setting)
            )
            if (
                resize_method is ResizeMethod.TO_SIZE
                or parse_cellprofiler_float(
                    required_setting_value(module, cls.factor_z_setting)
                )
                != 1.0
            ):
                relation_type = GroupLineageSourceRelation
        return (relation_type(source=source.ref()),)


def _parse_tile_assembly_method(value: str) -> str:
    normalized = normalize_cellprofiler_setting_name(value)
    if normalized != "within_cycles":
        raise NotImplementedError(
            f"Tile assembly method is not supported by the converter: {value!r}"
        )
    return normalized


class TileModule(
    CellProfilerModule):
    module_name = "Tile"
    function_name = "tile"
    validated = True
    confidence = 1.0
    input_image_setting = SettingNameFamily("Select an input image")
    additional_image_setting = SettingNameFamily("Select an additional image to tile")
    output_image_setting = SettingNameFamily("Name the output image")
    assembly_method_setting = SettingNameFamily("Tile assembly method")
    rows_setting = SettingNameFamily("Final number of rows")
    columns_setting = SettingNameFamily("Final number of columns")
    first_corner_setting = SettingNameFamily("Image corner to begin tiling")
    direction_setting = SettingNameFamily("Direction to begin tiling")
    meander_setting = SettingNameFamily("Use meander mode?")
    auto_rows_setting = SettingNameFamily("Automatically calculate number of rows?")
    auto_columns_setting = SettingNameFamily(
        "Automatically calculate number of columns?"
    )
    input_image_binding = SettingToKeywordBinding.input(
        input_image_setting, ImageArtifactType
    )
    additional_image_binding = SettingToKeywordBinding.input(
        additional_image_setting,
        ImageArtifactType,
        repeated=True,
    )
    output_image_binding = SettingToKeywordBinding.output(
        output_image_setting, ImageArtifactType
    )
    assembly_method_binding = SettingToKeywordBinding(
        assembly_method_setting,
        "tile_assembly_method",
        _parse_tile_assembly_method,
    )
    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (input_image_binding, additional_image_binding, output_image_binding,assembly_method_binding,
        SettingToKeywordBinding(rows_setting, "rows", parse_cellprofiler_int),
        SettingToKeywordBinding(columns_setting, "columns", parse_cellprofiler_int),
        SettingToKeywordBinding(
            first_corner_setting,
            "place_first",
            _parse_tile_first_corner_setting,
        ),
        SettingToKeywordBinding(
            direction_setting,
            "tile_style",
            _parse_tile_direction_setting,
        ),
        SettingToKeywordBinding(
            meander_setting,
            "meander",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            auto_rows_setting,
            "auto_rows",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            auto_columns_setting,
            "auto_columns",
            parse_cellprofiler_bool,
        ),)

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
        """Preserve the primary tiled image's source-stack identity."""
        del module, invocation_key, step_context, binding, name, output_position
        image_inputs = artifact_inputs.for_artifact_type(ImageArtifactType).specs
        if not image_inputs:
            raise ValueError("Tile requires a primary image input.")
        return (SourceStackLineageSourceRelation(source=image_inputs[0].ref()),)

    @classmethod
    def postprocess_bound_settings(
        cls, module: "ModuleBlock", bound: "BoundModuleSettings"
    ) -> "BoundModuleSettings":
        additional_names = setting_values(module, cls.additional_image_setting)
        if not additional_names:
            return bound
        return bound.with_kwargs(
            {cls.additional_image_binding.require_parameter_name(): additional_names}
        )


from dataclasses import dataclass
from enum import Enum
from typing import Any
import numpy as np
from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    payload_slices_for_alignment,
)
from openhcs.core.image_shapes import (
    trailing_spatial_target_shape,
)
from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_image_values import (
    image_mask_for_data_domain,
)
from openhcs.core.runtime_image_values import (
    image_payload_data,
)
from openhcs.core.runtime_image_values import (
    image_payload_mask,
)
from openhcs.core.runtime_image_values import (
    image_payload_metadata,
)
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.runtime_object_labels import (
    object_label_dense_array,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
)
from openhcs.core.pipeline.function_contracts import special_inputs
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


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


FlipMethod.NONE.cellprofiler_literals = ("Do not flip",)
FlipMethod.LEFT_TO_RIGHT.cellprofiler_literals = ("Left to right",)
FlipMethod.TOP_TO_BOTTOM.cellprofiler_literals = ("Top to bottom",)
FlipMethod.BOTH.cellprofiler_literals = ("Left to right and top to bottom",)
RotateMethod.NONE.cellprofiler_literals = ("Do not rotate",)
RotateMethod.ANGLE.cellprofiler_literals = ("Enter angle",)
RotateMethod.COORDINATES.cellprofiler_literals = ("Enter coordinates",)


def _parse_flip_coordinate_pair(value: str) -> tuple[int, int]:
    parts = tuple(part.strip() for part in value.split(","))
    if len(parts) != 2:
        raise ValueError(
            f"FlipAndRotate coordinate setting must be x,y, got {value!r}."
        )
    return (int(float(parts[0])), int(float(parts[1])))


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
        if mask is None:
            if input_shape is None:
                return None
            mask_array = np.ones(input_shape, dtype=bool)
        else:
            mask_array = np.asarray(mask, dtype=bool)
        output_shape = self.output_shape[-mask_array.ndim :]
        resized = mask_array
        valid_axes: list[np.ndarray] = []
        for axis, (input_size, output_size) in enumerate(
            zip(mask_array.shape, output_shape, strict=True)
        ):
            grid_positions = (
                (np.arange(output_size, dtype=np.float64) + 0.5)
                * input_size
                / output_size
            )
            source_indices = np.clip(
                np.floor(grid_positions).astype(np.intp), 0, input_size - 1
            )
            resized = np.take(resized, source_indices, axis=axis)
            valid_axes.append(
                (grid_positions >= 0.5) & (grid_positions <= input_size - 0.5)
            )
        for axis, valid in enumerate(valid_axes):
            axis_shape = [1] * resized.ndim
            axis_shape[axis] = valid.size
            resized &= valid.reshape(axis_shape)
        return resized

    def resize_payload(self, image: Any) -> Any:
        pixels = image_payload_data(image)
        output_pixels = self.resize_pixels(pixels)
        mask = image_mask_for_data_domain(source_payload=image, data=pixels)
        metadata = image_payload_metadata(image)
        output_spatial_shape = metadata.spatial_shape_yx(output_pixels)
        if output_spatial_shape is None:
            raise ValueError("Resize output does not declare two spatial axes.")
        metadata = metadata.with_spatial_resize(output_spatial_shape)
        if self.interpolation_order != 0:
            metadata = metadata.without_unit_interval_intensity_scale()
        return metadata.payload_with(
            output_pixels,
            self.resize_mask(mask, input_shape=tuple(np.asarray(pixels).shape)),
        )


@dataclass(frozen=True, slots=True)
class RotationResult(MeasurementFeatureRecord):
    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
    rotation_angle: float


@dataclass(frozen=True, slots=True)
class CellProfilerPlaneGeometry:
    """One CellProfiler XY plane coordinate system."""

    shape: tuple[int, int]

    @classmethod
    def from_image_plane(cls, image: Any) -> "CellProfilerPlaneGeometry":
        image_array = np.asarray(image_payload_data(image))
        metadata = image_payload_metadata(image)
        if metadata.plane_axis is not None:
            raise ValueError(
                "CellProfiler plane geometry requires an already-projected image; "
                f"got declared {metadata.plane_axis.value!r} plane axis."
            )
        channel_axis = metadata.normalized_source_channel_axis(image_array)
        expected_rank = 2 if channel_axis is None else 3
        if image_array.ndim != expected_rank:
            raise ValueError(
                "CellProfiler XY plane storage does not match its declared channel "
                f"layout: shape {image_array.shape!r}, channel axis {channel_axis!r}."
            )
        spatial_shape = metadata.spatial_shape_yx(image_array)
        if spatial_shape is None:
            raise ValueError(
                "CellProfiler XY plane metadata does not declare two spatial axes."
            )
        return cls(spatial_shape)

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return self.shape

    def binary_mask(
        self,
        mask: np.ndarray | ObjectLabelValue,
        *,
        threshold: float = 0.5,
        labels: bool = False,
    ) -> np.ndarray:
        mask_array = binary_mask_plane(mask, threshold=threshold, labels=labels)
        if tuple(mask_array.shape) != self.spatial_shape:
            raise ValueError(
                "CellProfiler mask shape must exactly match the selected image "
                f"domain; got {mask_array.shape!r} for {self.spatial_shape!r}."
            )
        return mask_array.astype(bool, copy=False)

    def label_plane(self, labels: np.ndarray) -> np.ndarray:
        return align_label_plane_to_shape(
            labels.astype(np.int32),
            self.spatial_shape,
        )


@dataclass(frozen=True, slots=True)
class CellProfilerImageMaskPlane:
    """One image plane paired with a binary mask in the same XY geometry."""

    image: np.ndarray
    mask: np.ndarray

    def __post_init__(self) -> None:
        geometry = CellProfilerPlaneGeometry.from_image_plane(self.image)
        if tuple(self.mask.shape) != geometry.spatial_shape:
            raise ValueError(
                f"CellProfilerImageMaskPlane mask shape must match image spatial shape; got mask {self.mask.shape!r} for image {geometry.shape!r}."
            )


def aligned_image_mask_planes(
    image: np.ndarray,
    mask: np.ndarray | ObjectLabelValue,
    *,
    threshold: float = 0.5,
    labels: bool = False,
) -> tuple[CellProfilerImageMaskPlane, ...]:
    """Pair image and mask planes with exact declared runtime cardinality."""
    image_planes = payload_slices_for_alignment(image)
    mask_planes = payload_slices_for_alignment(mask)
    if len(image_planes) != len(mask_planes):
        raise ValueError(
            "CellProfiler image and mask cardinalities must exactly match after "
            f"typed runtime projection; got {len(image_planes)} image plane(s) "
            f"and {len(mask_planes)} mask plane(s)."
        )
    return tuple(
        CellProfilerImageMaskPlane(
            image=image_plane,
            mask=CellProfilerPlaneGeometry.from_image_plane(image_plane).binary_mask(
                mask_plane,
                threshold=threshold,
                labels=labels,
            ),
        )
        for image_plane, mask_plane in zip(image_planes, mask_planes, strict=True)
    )


def restore_image_mask_planes(
    original_image: np.ndarray, masked_planes: tuple[np.ndarray, ...]
) -> Any:
    """Restore masked planes through the image payload's declared owner."""
    if not masked_planes:
        raise ValueError("Cannot restore an empty CellProfiler image plane set.")
    if isinstance(original_image, AlignedImageStack):
        if len(masked_planes) != len(original_image.slices):
            raise ValueError(
                "Aligned image result cardinality must exactly match its owner: "
                f"{len(masked_planes)} != {len(original_image.slices)}."
            )
        return original_image.with_slices(masked_planes)
    if len(masked_planes) == 1:
        return masked_planes[0]
    raise ValueError(
        "Multiple masked image planes require an AlignedImageStack owner; "
        f"got {len(masked_planes)} unowned planes."
    )


def binary_mask_plane(
    mask: np.ndarray | ObjectLabelValue,
    *,
    threshold: float = 0.5,
    labels: bool = False,
) -> np.ndarray:
    """Convert one CellProfiler mask/label plane to a 2D boolean mask."""
    mask_array = np.asarray(
        object_label_dense_array(mask)
        if isinstance(mask, ObjectLabelValue)
        else image_payload_data(mask)
    )
    if labels:
        if mask_array.ndim != 2:
            raise ValueError(
                "CellProfiler object masks require one projected 2-D label plane, "
                f"got shape {mask_array.shape!r}."
            )
        return mask_array > 0
    channel_axis = image_payload_metadata(mask).normalized_source_channel_axis(mask)
    if channel_axis is not None:
        return np.any(mask_array > threshold, axis=channel_axis)
    if mask_array.ndim != 2:
        raise ValueError(
            "CellProfiler image masks require a projected 2-D image or an "
            "explicit source channel axis; "
            f"got shape {mask_array.shape!r}."
        )
    return mask_array > threshold


def align_binary_mask_to_shape(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Validate a boolean mask against an explicitly selected image domain."""
    if tuple(mask.shape) != tuple(shape):
        raise ValueError(
            "CellProfiler binary mask shape must exactly match the selected "
            f"image domain; got {mask.shape!r} for {shape!r}."
        )
    return mask.astype(bool, copy=False)


def align_volume_mask_to_shape(
    mask: np.ndarray, shape_yx: tuple[int, int]
) -> np.ndarray:
    """Validate a volume mask's spatial shape after domain selection."""
    if mask.ndim != 3 or tuple(mask.shape[-2:]) != tuple(shape_yx):
        raise ValueError(
            "CellProfiler volume mask must be ZYX with the selected XY shape; "
            f"got {mask.shape!r} for {shape_yx!r}."
        )
    return mask.astype(bool, copy=False)


def align_label_plane_to_shape(
    labels: np.ndarray, shape: tuple[int, ...]
) -> np.ndarray:
    """Validate dense labels against an explicitly selected image domain."""
    labels = np.asarray(labels)
    if tuple(labels.shape) != tuple(shape):
        raise ValueError(
            "CellProfiler label shape must exactly match the selected image "
            f"domain; got {labels.shape!r} for {shape!r}."
        )
    return labels.astype(np.int32, copy=False)


@numpy(contract=ProcessingContract.FLEXIBLE)
@special_inputs("mask")
def mask_image(
    image: np.ndarray,
    mask: np.ndarray | ObjectLabelValue,
    mask_source: MaskSource = MaskSource.IMAGE,
    invert_mask: bool = False,
    binary_threshold: float = 0.5,
) -> np.ndarray:
    """Mask an image using CellProfiler image/object mask semantics.

    Args:
        mask: Mask image when ``mask_source`` is ``Image``, or object-label value
            when it is ``Objects``; its spatial shape must match the input image.
        binary_threshold: Image-mask pixels above this value are foreground;
            ignored when masking from object labels.
    """
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
    return replace(
        image_payload_metadata(image), mask_defines_border=True
    ).payload_with(masked_data, output_mask)


def masked_image_plane(
    image: np.ndarray, binary_mask: np.ndarray, *, invert_mask: bool
) -> tuple[np.ndarray, np.ndarray]:
    image_data = image_payload_data(image)
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
            explicit_mask=existing_mask,
        )
        if existing_projected_mask is not None:
            binary_mask = binary_mask & np.asarray(existing_projected_mask, dtype=bool)
    masked = image_data.copy()
    masked[~binary_mask] = 0
    return (masked, np.asarray(binary_mask, dtype=bool))


@numpy(contract=ProcessingContract.PURE_2D)
def mask_image_with_binary(
    image: np.ndarray, invert_mask: bool = False
) -> RuntimeArrayData:
    """Return a binary mask plane, optionally inverted."""
    binary_mask = image_payload_data(image) > 0.5
    if invert_mask:
        binary_mask = ~binary_mask
    return image_payload_metadata(image).payload_with(
        binary_mask.astype(np.float32),
        image_payload_mask(image),
    )


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


@numpy(contract=ProcessingContract.FLEXIBLE)
def tile(
    image: np.ndarray,
    rows: int = 8,
    columns: int = 12,
    place_first: PlaceFirst = PlaceFirst.TOP_LEFT,
    tile_style: TileStyle = TileStyle.ROW,
    meander: bool = False,
    auto_rows: bool = False,
    auto_columns: bool = False,
) -> RuntimeArrayData:
    """Tile multiple images together to form a CellProfiler montage image."""
    image_data = image_payload_data(image)
    if image_data.ndim not in {3, 4}:
        raise ValueError(
            "Tile expects an image stack shaped (N, H, W) or (N, H, W, C), "
            f"got {image_data.shape!r}."
        )
    num_images = image_data.shape[0]
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
    tile_height = image_data.shape[1]
    tile_width = image_data.shape[2]
    output_height = tile_height * geometry.rows
    output_width = tile_width * geometry.columns
    output_pixels = np.zeros(
        tile_output_shape(image_data, output_height, output_width),
        dtype=image_data.dtype,
    )
    for image_index in range(num_images):
        put_tile(image_data[image_index], output_pixels, image_index, geometry)
    metadata = image_payload_metadata(image)
    if metadata.plane_axis is not None:
        metadata = metadata.collapse_leading_plane_axis()
    return (
        metadata.replace_fields(
            source_channel_axis=-1 if image_data.ndim == 4 else None
        )
        .with_spatial_resize((output_height, output_width))
        .payload_with(output_pixels, None)
    )


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
) -> tuple[RuntimeArrayData, DataclassMeasurementColumnarRows]:
    """Flip and/or rotate a CellProfiler image plane.

    Args:
        flip_method: Reflection to apply before any rotation.
        rotate_method: Choose no rotation, a direct angle, or alignment from two
            pixel coordinates.
        rotation_angle: Counterclockwise rotation in degrees for angle mode.
        first_pixel_x: Horizontal coordinate of the first alignment point.
        first_pixel_y: Vertical coordinate of the first alignment point.
        second_pixel_x: Horizontal coordinate of the second alignment point.
        second_pixel_y: Vertical coordinate of the second alignment point.
        alignment_direction: Orient the line through the two alignment points
            horizontally or vertically in coordinate mode.
        crop_rotated_edges: Remove padded rotation borders by retaining the
            largest centered rectangular image region.
    """
    from scipy.ndimage import rotate as scipy_rotate

    image_data = image_payload_data(image)
    pixel_data = image_data.copy()
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
                    scipy_rotate(np.ones(image_data.shape[:2]), angle, reshape=True)
                    > 0.5
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
        image_payload_metadata(image)
        .with_spatial_resize(pixel_data.shape[:2])
        .payload_with(pixel_data.astype(np.float32), None),
        DataclassMeasurementColumnarRows(
            (RotationResult(slice_index=0, rotation_angle=angle),),
            row_type=RotationResult,
        ),
    )


class FlipAndRotateModule(
    NoObjectNameMeasurementRecordMixin,
    ProducedImageMeasurementRecordMixin,
    MeasurementArtifactOutputModule,
):
    module_name = "FlipAndRotate"
    function_name = "flip_and_rotate"
    validated = True
    confidence = 1.0

    input_image_setting = SettingNameFamily("Select the input image")
    output_image_setting = SettingNameFamily("Name the output image")
    flip_method_setting = SettingNameFamily("Select method to flip image")
    rotate_method_setting = SettingNameFamily("Select method to rotate image")
    crop_rotated_edges_setting = SettingNameFamily("Crop away the rotated edges?")
    calculate_rotation_setting = SettingNameFamily("Calculate rotation")
    first_pixel_setting = SettingNameFamily(
        "Enter coordinates of the top or left pixel"
    )
    second_pixel_setting = SettingNameFamily(
        "Enter the coordinates of the bottom or right pixel"
    )
    alignment_direction_setting = SettingNameFamily(
        "Select how the specified points should be aligned"
    )
    rotation_angle_setting = SettingNameFamily("Enter angle of rotation")

    input_image_binding = SettingToKeywordBinding.input(
        input_image_setting,
        ImageArtifactType,
    )
    output_image_binding = SettingToKeywordBinding.output(
        output_image_setting,
        ImageArtifactType,
    )
    setting_bindings = (
        input_image_binding,
        output_image_binding,
        SettingToKeywordBinding(
            flip_method_setting,
            "flip_method",
            cellprofiler_enum_value_setting_parser(FlipMethod),
        ),
        SettingToKeywordBinding(
            rotate_method_setting,
            "rotate_method",
            cellprofiler_enum_value_setting_parser(RotateMethod),
        ),
        SettingToKeywordBinding(
            crop_rotated_edges_setting,
            "crop_rotated_edges",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            alignment_direction_setting,
            "alignment_direction",
            cellprofiler_enum_value_setting_parser(AlignmentDirection),
        ),
        SettingToKeywordBinding(
            rotation_angle_setting,
            "rotation_angle",
            parse_cellprofiler_float,
        ),
    )
    ignored_settings = (calculate_rotation_setting,)

    class MeasurementFeatureTemplate(FormattingMeasurementFeatureTemplate):
        """Exact native image measurement emitted by FlipAndRotate."""

        ROTATION = ("Rotation_{output_image_name}", float)

    @dataclass(frozen=True, slots=True)
    class MeasurementRows(ModuleOwnedResultMeasurementRows):
        """Project the rotation result through its declared output image name."""

        output_image_name: str

        @classmethod
        def for_request(cls, module_type, request):
            output_names = request.callable_contract.artifact_outputs.names_of_artifact_type(
                ImageArtifactType
            )
            if len(output_names) != 1:
                raise ValueError(
                    f"{module_type.__name__} requires exactly one image output, "
                    f"got {output_names!r}."
                )
            return cls(
                request.output_value,
                module_type=module_type,
                output_image_name=output_names[0],
            )

        def rows(self) -> MeasurementProjectedColumnarRows:
            source_rows = self.source_rows()
            slice_field = self.source_field_annotated_by(
                RotationResult,
                MeasurementRowAxisField.SLICE_INDEX,
            )
            angle_field = self.source_field("rotation_angle")
            feature = self.module_type.MeasurementFeatureTemplate.ROTATION
            feature_name = feature.feature_name(
                output_image_name=self.output_image_name,
            )
            return MeasurementProjectedColumnarRows(
                {
                    slice_field.name: source_rows.column_values(slice_field.name),
                    feature_name: source_rows.column_values(angle_field.name),
                },
                fields=(
                    slice_field,
                    feature.field_spec(feature_name),
                ),
            )

    @classmethod
    def bind_settings(cls, module, *, binder):
        """Bind scalar settings and expand CellProfiler coordinate pairs."""

        bound = cls._bind_declared_settings(module, binder=binder)
        coordinate_kwargs: RuntimeCallableKwargs = {}
        first_pixel = optional_setting_value(module, cls.first_pixel_setting)
        if first_pixel is not None:
            first_pixel_x, first_pixel_y = _parse_flip_coordinate_pair(first_pixel)
            coordinate_kwargs.update(
                first_pixel_x=first_pixel_x,
                first_pixel_y=first_pixel_y,
            )
        second_pixel = optional_setting_value(module, cls.second_pixel_setting)
        if second_pixel is not None:
            second_pixel_x, second_pixel_y = _parse_flip_coordinate_pair(second_pixel)
            coordinate_kwargs.update(
                second_pixel_x=second_pixel_x,
                second_pixel_y=second_pixel_y,
            )
        bound = bound.with_kwargs(coordinate_kwargs)
        bound = bound.with_consumed_settings(
            cls.first_pixel_setting,
            cls.second_pixel_setting,
        )
        return cls._finalize_bound_settings(module, binder=binder, bound=bound)

    @classmethod
    def measurement_output_relations(
        cls,
        module,
        *,
        invocation_key,
        step_context,
        artifact_inputs: ArtifactSpecCollection,
    ):
        """Declare the transformed image as the native rotation subject."""

        inherited = super().measurement_output_relations(
            module,
            invocation_key=invocation_key,
            step_context=step_context,
            artifact_inputs=artifact_inputs,
        )
        output_name = optional_setting_value(module, cls.output_image_setting)
        if output_name is None:
            return inherited
        return (
            *inherited,
            ImageMeasurementSubjectRelation(
                source=ArtifactSpec.output(output_name, ImageArtifactType).ref()
            ),
        )
