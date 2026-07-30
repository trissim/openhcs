"""Crop geometry semantics for CellProfiler-compatible processing."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, ClassVar
from metaclass_registry import AutoRegisterMeta
import numpy as np
from openhcs.core.artifacts import (
    ArtifactSidecarRole,
    ArtifactSidecarSourceRelation,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactType,
    ImageArtifactType,
    InputStackBroadcastSourceRelation,
    ObjectLabelsArtifactType,
)
from openhcs.core.memory.decorators import numpy as numpy_decorator
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
)
from openhcs.core.pipeline.function_contracts import special_inputs
from openhcs.core.runtime_measurements import RuntimeMeasurementFeature
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
    SourceQualifiedWideMeasurementRowsModule,
)
from openhcs.interop.cellprofiler.setting_names import (
    optional_setting_value,
    required_setting_value,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    NoObjectNameMeasurementRecordMixin,
    ProducedImageMeasurementRecordMixin,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    cellprofiler_enum_value_setting_parser,
    parse_cellprofiler_float,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

if TYPE_CHECKING:
    from openhcs.core.function_patterns import FunctionInvocationKey
    from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
    from openhcs.interop.cellprofiler.parser import ModuleBlock


class CropModule(
    NoObjectNameMeasurementRecordMixin,
    ProducedImageMeasurementRecordMixin,
    SourceQualifiedWideMeasurementRowsModule,
    ObjectArtifactInputModule,
    MeasurementArtifactOutputModule,
):
    module_name = "Crop"
    function_name = "crop"
    validated = True
    confidence = 1.0
    cycle_pattern_setting: ClassVar[str] = "Apply which cycle's cropping pattern?"
    ignored_settings = (cycle_pattern_setting,)
    measurement_category_prefixes = (("crop",),)
    measurement_record_excluded_fields = frozenset({"fraction_retained"})

    class MeasurementFeature(RuntimeMeasurementFeature):
        """Feature families emitted by Crop."""

        AREA_RETAINED_AFTER_CROPPING = (
            "AreaRetainedAfterCropping",
            (),
            (),
            (),
            "area_retained",
        )
        ORIGINAL_IMAGE_AREA = (
            "OriginalImageArea",
            (),
            (),
            (),
            "original_area",
        )
    shape_setting: ClassVar[str] = "Select the cropping shape"
    method_setting: ClassVar[str] = "Select the cropping method"
    removal_setting: ClassVar[str] = "Remove empty rows and columns?"
    input_image_setting: ClassVar[str] = "Select the input image"
    output_image_setting: ClassVar[str] = "Name the output image"
    mask_image_setting: ClassVar[str] = "Select the masking image"
    previous_image_setting: ClassVar[str] = "Select the image with a cropping mask"
    objects_setting: ClassVar[str] = "Select the objects"
    left_right_setting: ClassVar[str] = "Left and right rectangle positions"
    top_bottom_setting: ClassVar[str] = "Top and bottom rectangle positions"
    ellipse_center_setting: ClassVar[str] = "Coordinates of ellipse center"
    ellipse_x_radius_setting: ClassVar[str] = "Ellipse radius, X direction"
    ellipse_y_radius_setting: ClassVar[str] = "Ellipse radius, Y direction"
    input_image_binding = SettingToKeywordBinding.input(input_image_setting, ImageArtifactType)
    mask_image_binding = SettingToKeywordBinding.input(
        mask_image_setting, ImageArtifactType, runtime_parameter_name="topology_inputs"
    )
    previous_image_binding = SettingToKeywordBinding.input(
        previous_image_setting,
        ImageArtifactType,
        runtime_parameter_name="topology_inputs",
        sidecar_role=ArtifactSidecarRole.CROP_MASK,
    )
    output_image_binding = SettingToKeywordBinding.output(output_image_setting, ImageArtifactType)
    objects_binding = SettingToKeywordBinding.input(
        objects_setting, ObjectLabelsArtifactType, runtime_parameter_name="topology_inputs"
    )

    class Shape(str, Enum):
        RECTANGLE = "Rectangle"
        ELLIPSE = "Ellipse"
        IMAGE = "Image"
        OBJECTS = "Objects"
        CROPPING = "Previous cropping"

        @property
        def topology_artifact_type(self) -> type[ArtifactType] | None:
            """Return the exact compiled artifact type required by this shape."""

            if self in (type(self).RECTANGLE, type(self).ELLIPSE):
                return None
            if self in (type(self).IMAGE, type(self).CROPPING):
                return ImageArtifactType
            if self is type(self).OBJECTS:
                return ObjectLabelsArtifactType
            raise AssertionError(f"Unsupported Crop shape {self!r}.")

        def validate_topology_specs(
            self,
            specs: tuple[ArtifactSpec, ...],
        ) -> None:
            """Validate exact declared topology-input cardinality and role."""

            artifact_type = self.topology_artifact_type
            expected_count = 0 if artifact_type is None else 1
            if len(specs) != expected_count:
                raise ValueError(
                    f"Crop shape {self.value!r} requires {expected_count} topology "
                    f"input(s), got {tuple(spec.ref() for spec in specs)!r}."
                )
            if (
                artifact_type is not None
                and specs[0].artifact_type is not artifact_type
            ):
                raise TypeError(
                    f"Crop shape {self.value!r} requires "
                    f"{artifact_type.__name__}, got "
                    f"{specs[0].artifact_type.__name__}."
                )

        def runtime_topology_inputs(
            self,
            values: tuple[np.ndarray | ObjectLabelValue, ...],
        ) -> tuple[np.ndarray | None, ObjectLabelValue | None]:
            """Bind the canonical topology sequence to its declared Crop role."""

            artifact_type = self.topology_artifact_type
            expected_count = 0 if artifact_type is None else 1
            if len(values) != expected_count:
                raise ValueError(
                    f"Crop shape {self.value!r} requires {expected_count} topology "
                    f"input(s), got {len(values)}."
                )
            if artifact_type is None:
                return (None, None)
            value = values[0]
            if artifact_type is ObjectLabelsArtifactType:
                if not isinstance(value, ObjectLabelValue):
                    raise TypeError(
                        f"Crop shape {self.value!r} requires ObjectLabelValue, got "
                        f"{type(value).__name__}."
                    )
                return (None, value)
            if isinstance(value, ObjectLabelValue):
                raise TypeError(
                    f"Crop shape {self.value!r} requires an image topology input, "
                    "not ObjectLabelValue."
                )
            return (value, None)

    class Method(str, Enum):
        COORDINATES = "Coordinates"
        MOUSE = "Mouse"

        @property
        def is_coordinate_based(self) -> bool:
            return self is type(self).COORDINATES

    class RemovalMethod(str, Enum):
        NO = "No"
        EDGES = "Edges"
        ALL = "All"

        @property
        def removes_empty_rows_or_columns(self) -> bool:
            return self is not type(self).NO

        @property
        def removes_internal_empty_rows_or_columns(self) -> bool:
            return self is type(self).ALL

    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        input_image_binding,
        mask_image_binding,
        previous_image_binding,
        output_image_binding,
        objects_binding,
        SettingToKeywordBinding(
            shape_setting,
            "crop_shape",
            cellprofiler_enum_value_setting_parser(Shape),
        ),
        SettingToKeywordBinding(
            method_setting,
            "cropping_method",
            cellprofiler_enum_value_setting_parser(Method),
        ),
        SettingToKeywordBinding(
            removal_setting,
            "removal_method",
            cellprofiler_enum_value_setting_parser(RemovalMethod),
        ),
        SettingToKeywordBinding(
            left_right_setting,
            "left_right_rectangle_positions",
        ),
        SettingToKeywordBinding(
            top_bottom_setting,
            "top_bottom_rectangle_positions",
        ),
        SettingToKeywordBinding(ellipse_center_setting, "ellipse_center"),
        SettingToKeywordBinding(
            ellipse_x_radius_setting,
            "ellipse_x_radius",
            parse_cellprofiler_float,
        ),
        SettingToKeywordBinding(
            ellipse_y_radius_setting,
            "ellipse_y_radius",
            parse_cellprofiler_float,
        ),
    )

    @classmethod
    def shape(cls, module: "ModuleBlock") -> "CropModule.Shape":
        return cls.Shape(
            optional_setting_value(module, cls.shape_setting)
            or cls.Shape.RECTANGLE.value
        )

    @classmethod
    def method(cls, module: "ModuleBlock") -> "CropModule.Method":
        return cls.Method(
            optional_setting_value(module, cls.method_setting)
            or cls.Method.COORDINATES.value
        )

    @classmethod
    def removal_method(cls, module: "ModuleBlock") -> "CropModule.RemovalMethod":
        return cls.RemovalMethod(
            optional_setting_value(module, cls.removal_setting)
            or cls.RemovalMethod.NO.value
        )

    @classmethod
    def input_image_name(cls, module: "ModuleBlock") -> str:
        return required_setting_value(module, cls.input_image_setting)

    @classmethod
    def output_image_name(cls, module: "ModuleBlock") -> str:
        return required_setting_value(module, cls.output_image_setting)

    @classmethod
    def topology_input_binding(
        cls,
        module: "ModuleBlock",
    ) -> SettingToKeywordBinding | None:
        """Return the exact setting binding for the selected Crop topology role."""

        shape = cls.shape(module)
        if shape in (cls.Shape.RECTANGLE, cls.Shape.ELLIPSE):
            return None
        if shape is cls.Shape.IMAGE:
            return cls.mask_image_binding
        if shape is cls.Shape.OBJECTS:
            return cls.objects_binding
        if shape is cls.Shape.CROPPING:
            return cls.previous_image_binding
        raise AssertionError(f"Unsupported Crop shape {shape!r}.")

    @classmethod
    def active_artifact_bindings(cls, module=None, *, invocation_key=None):
        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        topology_binding = cls.topology_input_binding(module)
        return tuple(
            binding
            for binding in bindings
            if binding is not cls.mask_image_binding or binding is topology_binding
            if binding is not cls.objects_binding or binding is topology_binding
            if binding is not cls.previous_image_binding or binding is topology_binding
        )

    @classmethod
    def previous_cropping_image_name(cls, module: "ModuleBlock") -> str | None:
        from openhcs.interop.cellprofiler.setting_names import OptionalSettingSymbol

        return OptionalSettingSymbol(module, cls.previous_image_setting).value

    @classmethod
    def finalize_artifact_contract_inputs(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        artifact_inputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpec, ...]:
        inputs = ArtifactSpecCollection(
            super().finalize_artifact_contract_inputs(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
            )
        )
        shape = cls.shape(module)
        topology_binding = cls.topology_input_binding(module)
        if topology_binding is None:
            topology_inputs: tuple[ArtifactSpec, ...] = ()
        else:
            topology_parameter_name = (
                topology_binding.require_runtime_parameter_name()
            )
            topology_inputs = tuple(
                spec
                for spec in inputs
                if spec.parameter_name == topology_parameter_name
            )
        shape.validate_topology_specs(topology_inputs)
        if topology_binding is None:
            return inputs.specs

        primary_image = inputs.require_by_name_and_artifact_type(
            cls.input_image_name(module),
            ImageArtifactType,
        )
        (topology_input,) = topology_inputs
        if (
            shape is cls.Shape.CROPPING
            and topology_input.sidecar_role is not ArtifactSidecarRole.CROP_MASK
        ):
            raise ValueError(
                f"Crop({module.module_num}) requires one exact crop-mask sidecar, "
                f"got {topology_input!r}."
            )
        if shape.topology_artifact_type is not ImageArtifactType:
            return inputs.specs

        projected_topology = topology_input.with_group_scope_relation(
            InputStackBroadcastSourceRelation(source=primary_image.ref())
        )
        return tuple(
            projected_topology if spec.ref() == topology_input.ref() else spec
            for spec in inputs.specs
        )

    @classmethod
    def artifact_contract_outputs(
        cls,
        module,
        *,
        invocation_key,
        step_context,
        artifact_inputs: ArtifactSpecCollection,
    ):
        outputs = super().artifact_contract_outputs(
            module,
            invocation_key=invocation_key,
            step_context=step_context,
            artifact_inputs=artifact_inputs,
        )
        output_name = cls.output_image_name(module)
        primary_output = ArtifactSpecCollection(
            outputs
        ).require_by_name_and_artifact_type(output_name, ImageArtifactType)
        sidecar = ArtifactSpec.output(
            ArtifactSidecarRole.CROP_MASK.name_for(output_name),
            ImageArtifactType,
            relations=(
                *primary_output.relations,
                ArtifactSidecarSourceRelation(source=primary_output.ref()),
            ),
            sidecar_role=ArtifactSidecarRole.CROP_MASK,
        )
        declared: list[ArtifactSpec] = []
        for output in outputs:
            declared.append(output)
            if output.name == output_name and output.artifact_type is ImageArtifactType:
                declared.append(sidecar)
        if sidecar not in declared:
            raise ValueError(
                f"Crop({module.module_num}) did not declare primary image output "
                f"{output_name!r}."
            )
        return tuple(declared)


CropBoundaryPair = tuple[int | None, int | None] | None


@dataclass(frozen=True, slots=True)
class CropMeasurement:
    """Measurements from one Crop invocation."""

    slice_index: int
    original_area: int
    area_retained: int
    fraction_retained: float


@dataclass(frozen=True, slots=True)
class CropMaskRequest:
    """Nominal crop-mask construction request."""

    orig_image_pixels: np.ndarray
    mask_plane: np.ndarray | None
    crop_shape: CropModule.Shape
    cropping_method: CropModule.Method
    left_right_rectangle_positions: CropBoundaryPair
    top_bottom_rectangle_positions: CropBoundaryPair
    ellipse_center: tuple[float, float] | None
    ellipse_x_radius: float | None
    ellipse_y_radius: float | None
    cropping_labels: ObjectLabelValue | None

@dataclass(frozen=True, slots=True)
class CropSpatialBounds:
    """Spatial context for a crop output in the parent image domain."""

    offset_yx: tuple[int, int]
    output_shape_yx: tuple[int, int]
    first_last_yx: tuple[int, int, int, int] | None


class CropShapeMaskStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal strategy family for CellProfiler Crop shape modes."""

    __registry_key__ = "crop_shape_label"
    __skip_if_no_key__ = True
    crop_shape_label: ClassVar[str | None] = None
    crop_shape: ClassVar[CropModule.Shape | None] = None

    @classmethod
    def for_shape(cls, crop_shape: CropModule.Shape) -> "CropShapeMaskStrategy":
        strategy_type = cls.__registry__.get(crop_shape.value)
        if strategy_type is None:
            raise NotImplementedError(
                f"Unsupported CellProfiler Crop shape {crop_shape.value!r}."
            )
        return strategy_type()

    @abstractmethod
    def mask(self, request: CropMaskRequest) -> np.ndarray:
        """Return a boolean crop mask for one shape mode."""

    def validate_crop_mask(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Validate and normalize a shape-mode crop mask against input image XY."""
        crop_mask = np.asarray(mask).astype(bool)
        if crop_mask.shape != image.shape[:2]:
            raise ValueError(
                f"Crop mask shape must match input image XY shape; got mask {crop_mask.shape!r} for image {image.shape[:2]!r}."
            )
        return crop_mask


class PreviousCroppingMaskStrategy(CropShapeMaskStrategy):
    """Use the prior Crop sidecar mask."""

    crop_shape = CropModule.Shape.CROPPING
    crop_shape_label = crop_shape.value

    def mask(self, request: CropMaskRequest) -> np.ndarray:
        if request.mask_plane is None:
            raise ValueError("Crop Previous cropping mode requires a crop-mask plane.")
        return self.validate_crop_mask(request.mask_plane, request.orig_image_pixels)


class ImageMaskCropMaskStrategy(CropShapeMaskStrategy):
    """Use a supplied image mask."""

    crop_shape = CropModule.Shape.IMAGE
    crop_shape_label = crop_shape.value

    def mask(self, request: CropMaskRequest) -> np.ndarray:
        if request.mask_plane is None:
            raise ValueError("Crop image-mask mode requires a mask-image plane.")
        return self.validate_crop_mask(
            request.mask_plane > 0, request.orig_image_pixels
        )


class ObjectMaskCropMaskStrategy(CropShapeMaskStrategy):
    """Use supplied object labels as the crop mask."""

    crop_shape = CropModule.Shape.OBJECTS
    crop_shape_label = crop_shape.value

    def mask(self, request: CropMaskRequest) -> np.ndarray:
        if request.cropping_labels is None:
            raise ValueError("Crop object-mask mode requires cropping_labels.")
        return self.validate_crop_mask(
            object_label_crop_mask(request.cropping_labels, request.orig_image_pixels),
            request.orig_image_pixels,
        )


class RectangleCropMaskStrategy(CropShapeMaskStrategy):
    """Build a rectangular coordinate crop mask."""

    crop_shape = CropModule.Shape.RECTANGLE
    crop_shape_label = crop_shape.value

    def mask(self, request: CropMaskRequest) -> np.ndarray:
        require_coordinate_cropping(request)
        left, right = rectangle_pair(
            request.left_right_rectangle_positions, "left_right_rectangle_positions"
        )
        top, bottom = rectangle_pair(
            request.top_bottom_rectangle_positions, "top_bottom_rectangle_positions"
        )
        return rectangle_cropping(request.orig_image_pixels, (left, right, top, bottom))


class EllipseCropMaskStrategy(CropShapeMaskStrategy):
    """Build an elliptical coordinate crop mask."""

    crop_shape = CropModule.Shape.ELLIPSE
    crop_shape_label = crop_shape.value

    def mask(self, request: CropMaskRequest) -> np.ndarray:
        require_coordinate_cropping(request)
        if (
            request.ellipse_center is None
            or request.ellipse_x_radius is None
            or request.ellipse_y_radius is None
        ):
            raise ValueError("Crop ellipse mode requires center and X/Y radii.")
        return ellipse_cropping(
            request.orig_image_pixels,
            float_pair(request.ellipse_center, "ellipse_center"),
            (float(request.ellipse_x_radius), float(request.ellipse_y_radius)),
        )


@dataclass(frozen=True, slots=True)
class CropImageRequest:
    """Nominal request for CellProfiler Crop row/column projection."""

    image: np.ndarray
    crop_mask: np.ndarray
    crop_internal: bool = False

    def cropped_pixels(self) -> np.ndarray:
        i_histogram = self.crop_mask.sum(axis=1)
        i_cumsum = np.cumsum(i_histogram != 0)
        j_histogram = self.crop_mask.sum(axis=0)
        j_cumsum = np.cumsum(j_histogram != 0)
        if i_cumsum[-1] == 0:
            return np.zeros((0, 0), dtype=self.image.dtype)
        if self.crop_internal:
            i_keep = np.argwhere(i_histogram > 0).flatten()
            j_keep = np.argwhere(j_histogram > 0).flatten()
            return self.image[i_keep, :][:, j_keep].copy()
        i_first = int(np.argwhere(i_cumsum == 1)[0][0])
        i_last = int(np.argwhere(i_cumsum == i_cumsum.max())[0][0])
        j_first = int(np.argwhere(j_cumsum == 1)[0][0])
        j_last = int(np.argwhere(j_cumsum == j_cumsum.max())[0][0])
        return self.image[i_first : i_last + 1, j_first : j_last + 1].copy()


@dataclass(frozen=True, slots=True)
class CropRequest:
    """Executable CellProfiler Crop request."""

    image: np.ndarray
    mask_plane: np.ndarray | None = None
    crop_shape: CropModule.Shape = CropModule.Shape.RECTANGLE
    cropping_method: CropModule.Method = CropModule.Method.COORDINATES
    removal_method: CropModule.RemovalMethod = CropModule.RemovalMethod.NO
    left_right_rectangle_positions: CropBoundaryPair = None
    top_bottom_rectangle_positions: CropBoundaryPair = None
    ellipse_center: tuple[float, float] | None = None
    ellipse_x_radius: float | None = None
    ellipse_y_radius: float | None = None
    cropping_labels: ObjectLabelValue | None = None

    def execute(
        self,
    ) -> tuple[np.ndarray, np.ndarray, DataclassMeasurementColumnarRows]:
        input_pixels = image_payload_data(self.image)
        input_image_mask = image_payload_mask(self.image)
        input_metadata = image_payload_metadata(self.image)
        orig_image_pixels = np.asarray(input_pixels)
        input_mask_plane = None
        orig_image_mask = (
            None
            if input_image_mask is None
            else np.asarray(input_image_mask, dtype=bool)
        )
        if self.mask_plane is not None:
            input_mask_plane = image_payload_data(self.mask_plane)
        request = CropMaskRequest(
            orig_image_pixels=orig_image_pixels,
            mask_plane=input_mask_plane,
            crop_shape=self.crop_shape,
            cropping_method=self.cropping_method,
            left_right_rectangle_positions=self.left_right_rectangle_positions,
            top_bottom_rectangle_positions=self.top_bottom_rectangle_positions,
            ellipse_center=self.ellipse_center,
            ellipse_x_radius=self.ellipse_x_radius,
            ellipse_y_radius=self.ellipse_y_radius,
            cropping_labels=self.cropping_labels,
        )
        removal_method = self.removal_method
        cropping = CropShapeMaskStrategy.for_shape(request.crop_shape).mask(request)
        cropped_mask = cropped_mask_for(cropping, None, removal_method)
        output_image_mask = cropped_image_mask(
            cropping, cropped_mask, orig_image_mask, removal_method
        )
        cropped_pixel_data = cropped_image_pixels(
            orig_image_pixels, cropping, cropped_mask, removal_method
        )
        input_shape_yx = tuple((int(value) for value in orig_image_pixels.shape[:2]))
        crop_bounds = crop_spatial_bounds(cropping, removal_method)
        output_metadata = crop_output_metadata(
            input_metadata, input_shape_yx=input_shape_yx, bounds=crop_bounds
        )
        original_area = int(np.prod(orig_image_pixels.shape[:2]))
        area_retained = int(np.sum(cropping))
        measurements = CropMeasurement(
            slice_index=0,
            original_area=original_area,
            area_retained=area_retained,
            fraction_retained=area_retained / original_area if original_area else 0.0,
        )
        if not removal_method.removes_empty_rows_or_columns:
            output_metadata = replace(output_metadata, mask_defines_border=False)
        return (
            output_metadata.payload_with(cropped_pixel_data, output_image_mask),
            cropping,
            DataclassMeasurementColumnarRows(
                (measurements,),
                row_type=CropMeasurement,
            ),
        )


def require_coordinate_cropping(request: CropMaskRequest) -> None:
    """Validate that a shape strategy is configured for coordinate cropping."""
    if request.cropping_method.is_coordinate_based:
        return
    raise NotImplementedError(
        f"Headless OpenHCS execution supports coordinate Crop, not {request.cropping_method.value!r}."
    )


def ellipse_cropping(
    orig_image_pixels: np.ndarray,
    ellipse_center: tuple[float, float],
    ellipse_radius: tuple[float, float],
) -> np.ndarray:
    """Return a CP-compatible elliptical crop mask."""
    x_center, y_center = ellipse_center
    x_radius, y_radius = ellipse_radius
    x_max = orig_image_pixels.shape[1]
    y_max = orig_image_pixels.shape[0]
    if x_radius > y_radius:
        dist_x = np.sqrt(x_radius**2 - y_radius**2)
        dist_y = 0
        major_radius = x_radius
    else:
        dist_x = 0
        dist_y = np.sqrt(y_radius**2 - x_radius**2)
        major_radius = y_radius
    focus_1_x, focus_1_y = (x_center - dist_x, y_center - dist_y)
    focus_2_x, focus_2_y = (x_center + dist_x, y_center + dist_y)
    y, x = np.mgrid[0:y_max, 0:x_max]
    d1 = np.sqrt((x - focus_1_x) ** 2 + (y - focus_1_y) ** 2)
    d2 = np.sqrt((x - focus_2_x) ** 2 + (y - focus_2_y) ** 2)
    return d1 + d2 <= major_radius * 2


def rectangle_cropping(
    orig_image_pixels: np.ndarray,
    bounding_box: tuple[int | None, int | None, int | None, int | None],
) -> np.ndarray:
    """Return a CP-compatible rectangular crop mask."""
    cropping = np.ones(orig_image_pixels.shape[:2], bool)
    left, right, top, bottom = bounding_box
    if left and left > 0:
        cropping[:, :left] = False
    if right and right < cropping.shape[1]:
        cropping[:, right:] = False
    if top and top > 0:
        cropping[:top, :] = False
    if bottom and bottom < cropping.shape[0]:
        cropping[bottom:, :] = False
    return cropping


def cropped_mask_for(
    cropping: np.ndarray,
    mask: np.ndarray | None,
    removal_method: CropModule.RemovalMethod,
) -> np.ndarray:
    """Return the output mask for a crop result."""
    if not removal_method.removes_empty_rows_or_columns:
        return cropping if mask is None else mask
    if mask is not None:
        return mask
    return CropImageRequest(
        image=cropping,
        crop_mask=cropping,
        crop_internal=removal_method.removes_internal_empty_rows_or_columns,
    ).cropped_pixels()


def cropped_image_pixels(
    orig_image_pixels: np.ndarray,
    cropping: np.ndarray,
    mask: np.ndarray | None,
    removal_method: CropModule.RemovalMethod,
) -> np.ndarray:
    """Return cropped image pixels using CP removal semantics."""
    if not removal_method.removes_empty_rows_or_columns:
        cropped_pixel_data = orig_image_pixels.copy()
        cropped_pixel_data[~cropping] = 0
        return cropped_pixel_data
    cropped_pixel_data = CropImageRequest(
        image=orig_image_pixels,
        crop_mask=cropping,
        crop_internal=removal_method.removes_internal_empty_rows_or_columns,
    ).cropped_pixels()
    if mask is not None:
        cropped_pixel_data[~mask.astype(bool)] = 0
    return cropped_pixel_data


def cropped_image_mask(
    cropping: np.ndarray,
    mask: np.ndarray | None,
    orig_image_mask: np.ndarray | None,
    removal_method: CropModule.RemovalMethod,
) -> np.ndarray:
    """Return the output image mask using CellProfiler Crop semantics."""
    crop_mask = cropped_mask_for(cropping, mask, removal_method)
    if orig_image_mask is None:
        return crop_mask
    if not removal_method.removes_empty_rows_or_columns:
        return np.asarray(orig_image_mask, dtype=bool) & crop_mask
    crop_internal = removal_method.removes_internal_empty_rows_or_columns
    cropped_parent_mask = CropImageRequest(
        image=np.asarray(orig_image_mask, dtype=bool),
        crop_mask=cropping,
        crop_internal=crop_internal,
    ).cropped_pixels()
    return np.asarray(cropped_parent_mask, dtype=bool) & crop_mask


def crop_spatial_bounds(
    cropping: np.ndarray, removal_method: CropModule.RemovalMethod
) -> CropSpatialBounds:
    """Return crop bounds in parent-image coordinates."""
    input_shape = tuple((int(value) for value in cropping.shape[:2]))
    if not removal_method.removes_empty_rows_or_columns:
        return CropSpatialBounds(
            offset_yx=(0, 0),
            output_shape_yx=input_shape,
            first_last_yx=(0, input_shape[0] - 1, 0, input_shape[1] - 1),
        )
    row_indexes = np.flatnonzero(np.sum(cropping, axis=1) > 0)
    column_indexes = np.flatnonzero(np.sum(cropping, axis=0) > 0)
    if row_indexes.size == 0 or column_indexes.size == 0:
        return CropSpatialBounds(
            offset_yx=(0, 0), output_shape_yx=(0, 0), first_last_yx=None
        )
    first_row = int(row_indexes[0])
    last_row = int(row_indexes[-1])
    first_column = int(column_indexes[0])
    last_column = int(column_indexes[-1])
    if removal_method.removes_internal_empty_rows_or_columns:
        output_shape = (int(row_indexes.size), int(column_indexes.size))
    else:
        output_shape = (last_row - first_row + 1, last_column - first_column + 1)
    return CropSpatialBounds(
        offset_yx=(first_row, first_column),
        output_shape_yx=output_shape,
        first_last_yx=(first_row, last_row, first_column, last_column),
    )


def crop_output_metadata(
    metadata: ImagePayloadMetadata,
    *,
    input_shape_yx: tuple[int, int],
    bounds: CropSpatialBounds,
) -> ImagePayloadMetadata:
    """Return image metadata with exact crop-local physical edge context."""
    if bounds.first_last_yx is None:
        physical_edges = (False, False, False, False)
    else:
        first_row, last_row, first_column, last_column = bounds.first_last_yx
        top_edge, bottom_edge, left_edge, right_edge = (
            metadata.physical_border_edges_for_shape(input_shape_yx)
        )
        physical_edges = (
            top_edge and first_row == 0,
            bottom_edge and last_row == input_shape_yx[0] - 1,
            left_edge and first_column == 0,
            right_edge and last_column == input_shape_yx[1] - 1,
        )
    return metadata.with_spatial_crop(
        input_shape_yx=input_shape_yx,
        output_shape_yx=bounds.output_shape_yx,
        offset_yx=bounds.offset_yx,
        physical_border_edges_yx=physical_edges,
    )


@numpy_decorator(contract=ProcessingContract.PURE_2D)
@special_inputs("topology_inputs")
def crop(
    image: np.ndarray,
    topology_inputs: tuple[np.ndarray | ObjectLabelValue, ...] = (),
    crop_shape: CropModule.Shape = CropModule.Shape.RECTANGLE,
    cropping_method: CropModule.Method = CropModule.Method.COORDINATES,
    removal_method: CropModule.RemovalMethod = CropModule.RemovalMethod.NO,
    left_right_rectangle_positions: CropBoundaryPair = None,
    top_bottom_rectangle_positions: CropBoundaryPair = None,
    ellipse_center: tuple[float, float] | None = None,
    ellipse_x_radius: float | None = None,
    ellipse_y_radius: float | None = None,
) -> tuple[np.ndarray, np.ndarray, DataclassMeasurementColumnarRows]:
    """Crop an image and return its CellProfiler crop-mask sidecar.

    Args:
        topology_inputs: Mode-dependent crop source: one mask image for ``Image``,
            one prior crop-mask sidecar for ``Previous cropping``, or one object-
            label value for ``Objects``; empty for rectangle and ellipse modes.
    """
    mask_plane, cropping_labels = crop_shape.runtime_topology_inputs(topology_inputs)
    return CropRequest(
        image=image,
        mask_plane=mask_plane,
        crop_shape=crop_shape,
        cropping_method=cropping_method,
        removal_method=removal_method,
        left_right_rectangle_positions=left_right_rectangle_positions,
        top_bottom_rectangle_positions=top_bottom_rectangle_positions,
        ellipse_center=ellipse_center,
        ellipse_x_radius=ellipse_x_radius,
        ellipse_y_radius=ellipse_y_radius,
        cropping_labels=cropping_labels,
    ).execute()


@numpy_decorator(contract=ProcessingContract.PURE_2D)
def crop_simple(
    image: np.ndarray,
    crop_top: int = 0,
    crop_bottom: int = 0,
    crop_left: int = 0,
    crop_right: int = 0,
) -> np.ndarray:
    """Simple rectangular crop by pixel counts removed from each edge."""
    height, width = image.shape
    y_start = max(0, min(crop_top, height - 1))
    y_end = height - crop_bottom if crop_bottom > 0 else height
    y_end = max(y_start + 1, min(y_end, height))
    x_start = max(0, min(crop_left, width - 1))
    x_end = width - crop_right if crop_right > 0 else width
    x_end = max(x_start + 1, min(x_end, width))
    return image[y_start:y_end, x_start:x_end].copy()


def object_label_crop_mask(labels: ObjectLabelValue, image: np.ndarray) -> np.ndarray:
    """Return a foreground mask from one already-projected label plane."""
    label_array = object_label_dense_array(labels)
    image_shape = tuple(np.asarray(image).shape[:2])
    if label_array.ndim != 2:
        raise ValueError(
            "Crop object labels must be projected to one 2-D plane, got "
            f"shape {label_array.shape!r}."
        )
    if label_array.shape != image_shape:
        raise ValueError(
            "Crop projected object labels must match the input image XY shape; "
            f"got labels {label_array.shape!r} and image {image_shape!r}."
        )
    return label_array > 0


def rectangle_pair(value: CropBoundaryPair, name: str) -> tuple[int | None, int | None]:
    """Validate and normalize a rectangle boundary pair."""
    if value is None:
        return (None, None)
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {value!r}.")
    return (value[0], value[1])


def float_pair(value: tuple[float, float], name: str) -> tuple[float, float]:
    """Validate and normalize a coordinate pair."""
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {value!r}.")
    return (float(value[0]), float(value[1]))


__all__ = [
    "CropBoundaryPair",
    "CropImageRequest",
    "CropMaskRequest",
    "CropMeasurement",
    "CropRequest",
    "CropShapeMaskStrategy",
    "CropSpatialBounds",
    "crop",
    "crop_simple",
    "cropped_image_pixels",
    "cropped_mask_for",
    "crop_output_metadata",
    "crop_spatial_bounds",
    "ellipse_cropping",
    "float_pair",
    "object_label_crop_mask",
    "rectangle_cropping",
    "rectangle_pair",
    "require_coordinate_cropping",
]
