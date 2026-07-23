"""Morphology backend strategies for CellProfiler-compatible processing.

This module is the OpenHCS processing-backend seam for CellProfiler-compatible
semantics.  The default implementation is independent NumPy/SciPy/skimage code;
the optional Centrosome provider is allowed for matching legacy morphology
behavior when explicitly requested.
"""

from __future__ import annotations
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING
import numpy as np
from openhcs.core.artifacts import (
    ImageArtifactType,
    ArtifactSpecCollection,
    GroupLineageSourceRelation,
    ObjectLabelsArtifactType,
    SourceStackLineageSourceRelation,
)
from openhcs.core.source_bindings import StepSourceBindingsConfig
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.core.measurement_row_materialization import (
    MeasurementSparseColumnarRows,
)
from openhcs.interop.cellprofiler.setting_names import SettingNameFamily
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    cellprofiler_enum_value_setting_parser,
    normalize_cellprofiler_setting_name,
    parse_cellprofiler_bool,
    parse_cellprofiler_float,
    parse_cellprofiler_int,
)
from openhcs.interop.cellprofiler.runtime.primary_image_input_policies import (
    ObjectLabelDrivenPrimaryImageInputPolicy,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    LabelsObjectInputPolicy,
)
from openhcs.interop.cellprofiler.runtime.output_contexts import (
    InputObjectLabelWithoutParentImageOutputSourceContextPolicyMixin,
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
    ObjectArtifactOutputModule,
    ObjectLineageTransformContractModule,
    PlaneRuntimeArtifactModule,
)
from openhcs.interop.cellprofiler.module_structuring_element_settings import (
    StructuringElementSettingBinding,
    StructuringElementSettingsModule,
)
from openhcs.interop.cellprofiler.setting_names import (
    normalized_symbol_name,
    optional_setting_value,
    required_setting_value,
    setting_values,
)
from openhcs.interop.cellprofiler.cellprofiler_literals import (
    cellprofiler_enum_from_literal,
)

if TYPE_CHECKING:
    from openhcs.core.callable_contract import CallableContract
    from openhcs.interop.cellprofiler.parser import ModuleBlock


class CombineObjectsMethod(Enum):
    """Overlap policies exposed by CombineObjects settings."""

    MERGE = "merge"
    PRESERVE = "preserve"
    DISCARD = "discard"
    SEGMENT = "segment"


class MaskObjectsOverlapHandling(Enum):
    """CellProfiler MaskObjects overlap handling mode."""

    MASK = "keep_overlapping_region"
    KEEP = "keep"
    REMOVE = "remove"
    REMOVE_PERCENTAGE = "remove_depending_on_overlap"


class MaskObjectsNumberingChoice(Enum):
    """CellProfiler MaskObjects output label numbering mode."""

    RENUMBER = "renumber"
    RETAIN = "retain"


class ImageStructuringElementModule(
    StructuringElementSettingsModule,
):
    """Shared declaration for image morphology modules with one image output."""

    input_image_setting = SettingNameFamily("Select the input image")
    output_image_setting = SettingNameFamily("Name the output image")
    setting_bindings = (
        SettingToKeywordBinding.input(input_image_setting, ImageArtifactType),
        SettingToKeywordBinding.output(output_image_setting, ImageArtifactType),
    )

    @classmethod
    def postprocess_bound_settings(
        cls, module: "ModuleBlock", bound: BoundModuleSettings
    ) -> BoundModuleSettings:
        setting = cls.declared_structuring_element_setting(module)
        footprint = build_structuring_element(
            setting.structuring_element,
            setting.size,
        )
        return BoundModuleSettings(
            {
                **bound.kwargs,
                SliceBySliceRuntimeParameter.require_parameter_name(): (
                    footprint.ndim == 2
                ),
            },
            bound.unmapped_kwargs,
            bound.setting_coverage,
        )


class ClosingModule(ImageStructuringElementModule):
    module_name = "Closing"
    function_name = "closing"
    validated = True
    confidence = 1.0


class ObjectTransformContractModule(
    PlaneRuntimeArtifactModule,
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
    ObjectArtifactOutputModule,
):
    """Shared declaration for object modules that emit measurements plus objects."""

    input_objects_setting = SettingNameFamily(
        "Select the input objects",
        aliases=("Select the input object", "Select objects to be masked"),
    )
    output_objects_setting = SettingNameFamily(
        "Name the output objects",
        aliases=("Name the output object", "Name the masked objects"),
    )
    input_objects_binding = SettingToKeywordBinding.input(
        input_objects_setting, ObjectLabelsArtifactType, runtime_parameter_name="labels"
    )
    output_objects_binding = SettingToKeywordBinding.output(
        output_objects_setting,
        ObjectLabelsArtifactType,
    )
    setting_bindings = (
        input_objects_binding,
        output_objects_binding,
    )


class ObjectLineageTransformModule(ObjectLineageTransformContractModule):
    """Shared declaration for object transforms that also emit parent-child lineage."""

    input_objects_setting = SettingNameFamily(
        "Select the input object", aliases=("Select the input objects",)
    )
    output_objects_setting = SettingNameFamily(
        "Name the output object", aliases=("Name the output objects",)
    )
    input_objects_binding = SettingToKeywordBinding.input(
        input_objects_setting, ObjectLabelsArtifactType, runtime_parameter_name="labels"
    )
    output_objects_binding = SettingToKeywordBinding.output(
        output_objects_setting,
        ObjectLabelsArtifactType,
    )
    setting_bindings = (
        input_objects_binding,
        output_objects_binding,
    )


class CombineobjectsModule(
    PlaneRuntimeArtifactModule,
    ObjectArtifactInputModule,
    MeasurementArtifactOutputModule,
    ObjectArtifactOutputModule,
    CellProfilerModule,
):
    module_name = "Combineobjects"
    function_name = "combineobjects"
    validated = True
    confidence = 1.0
    first_objects_setting = SettingNameFamily("Select initial object set")
    second_objects_setting = SettingNameFamily("Select object set to combine")
    output_objects_setting = SettingNameFamily("Name the combined object set")
    first_objects_binding = SettingToKeywordBinding.input(
        first_objects_setting, ObjectLabelsArtifactType, runtime_parameter_name="object_labels"
    )
    output_objects_binding = SettingToKeywordBinding.output(
        output_objects_setting,
        ObjectLabelsArtifactType,
    )
    setting_bindings = (
        first_objects_binding,
        SettingToKeywordBinding.input(
            second_objects_setting,
            ObjectLabelsArtifactType,
            runtime_parameter_name="object_labels",
        ),
        output_objects_binding,
        SettingToKeywordBinding(
            "Select how to handle overlapping objects",
            "method",
            cellprofiler_enum_value_setting_parser(CombineObjectsMethod),
        ),
    )

    @classmethod
    def artifact_output_relations(
        cls,
        module,
        *,
        binding,
        name,
        invocation_key,
        step_context,
        artifact_inputs: ArtifactSpecCollection,
        output_position,
    ):
        """Use the declared initial object set as the combined-label domain."""

        if binding is not cls.output_objects_binding:
            return super().artifact_output_relations(
                module,
                binding=binding,
                name=name,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
                output_position=output_position,
            )
        del name, invocation_key, step_context, output_position
        (source_name,) = cls.artifact_names_for_binding(
            module,
            cls.first_objects_binding,
        )
        source = artifact_inputs.require_by_name_and_artifact_type(
            source_name,
            ObjectLabelsArtifactType,
        )
        return (SourceStackLineageSourceRelation(source=source.ref()),)


class DilateImageModule(ImageStructuringElementModule):
    module_name = "DilateImage"
    function_name = "dilate_image"
    validated = True
    aliases = ("Dilation",)
    confidence = 1.0


class ZStackFunctionVariantModule:
    """Select the declared volumetric callable for a source Z stack."""

    @classmethod
    def resolve_function(
        cls,
        module: "ModuleBlock",
        *,
        contract: "CallableContract",
        source_bindings: "StepSourceBindingsConfig",
    ) -> Callable[..., object]:
        del module, contract
        function_name = (
            cls.function_variants[0]
            if AllComponents.Z_INDEX in source_bindings.source_stack_components
            else str(cls.function_name)
        )
        return cls.require_callable(function_name)


class DilateObjectsModule(
    ZStackFunctionVariantModule,
    ObjectTransformContractModule,
    StructuringElementSettingsModule,
):
    module_name = "DilateObjects"
    function_name = "dilate_objects"
    function_variants = ("dilate_objects_3d",)
    validated = True
    confidence = 1.0
    structuring_element_binding = StructuringElementSettingBinding(
        shape_keyword="structuring_element_shape",
        size_keyword="structuring_element_size",
    )


class ErodeImageModule(ImageStructuringElementModule):
    module_name = "ErodeImage"
    function_name = "erode_image"
    validated = True
    aliases = ("Erosion",)
    confidence = 1.0


class ErodeObjectsModule(
    ObjectLabelDrivenPrimaryImageInputPolicy,
    LabelsObjectInputPolicy,
    ObjectLineageTransformModule,
    StructuringElementSettingsModule,
):
    module_name = "ErodeObjects"
    function_name = "erode_objects"
    validated = True
    confidence = 1.0
    setting_bindings = (
        SettingToKeywordBinding(
            "Prevent object removal", "preserve_midpoints", parse_cellprofiler_bool
        ),
        SettingToKeywordBinding(
            "Relabel resulting objects", "relabel_objects", parse_cellprofiler_bool
        ),
    )


class CellProfilerExpandShrinkOperation(Enum):
    """Closed CellProfiler UI operation dialect for ExpandOrShrinkObjects."""

    SHRINK_TO_POINT = "Shrink objects to a point"
    EXPAND_UNTIL_TOUCHING = "Expand objects until touching"
    ADD_DIVIDING_LINES = "Add partial dividing lines between objects"
    SHRINK_DEFINED_PIXELS = "Shrink objects by a specified number of pixels"
    SHRINK_BY_MEASUREMENT = "Shrink objects by a previous measurement"
    EXPAND_DEFINED_PIXELS = "Expand objects by a specified number of pixels"
    EXPAND_BY_MEASUREMENT = "Expand objects by a previous measurement"
    SKELETONIZE = "Skeletonize each object"
    DESPUR = "Remove spurs"


class ExpandShrinkMode(Enum):
    """Runtime mode literals consumed by ExpandOrShrinkObjects execution."""

    EXPAND_DEFINED_PIXELS = "expand_defined_pixels"
    EXPAND_INFINITE = "expand_infinite"
    SHRINK_DEFINED_PIXELS = "shrink_defined_pixels"
    SHRINK_TO_POINT = "shrink_to_point"
    ADD_DIVIDING_LINES = "add_dividing_lines"
    DESPUR = "despur"
    SKELETONIZE = "skeletonize"


class ExpandOrShrinkObjectsModule(ObjectTransformContractModule):
    module_name = "ExpandOrShrinkObjects"
    function_name = "expand_or_shrink_objects"
    validated = True
    confidence = 1.0
    input_objects_setting = SettingNameFamily("Select the input objects")
    output_objects_setting = SettingNameFamily("Name the output objects")
    setting_bindings = (
        SettingToKeywordBinding(
            "Select the operation",
            "mode",
            lambda value: (
                ExpandShrinkOperationStrategy.mode_for_cellprofiler_operation(
                    value
                ).value
            ),
        ),
        SettingToKeywordBinding(
            "Number of pixels by which to expand or shrink",
            "iterations",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            "Fill holes in objects so that all objects shrink to a single point?",
            "fill_holes",
            parse_cellprofiler_bool,
        ),
    )


class MaskObjectsModule(
    ObjectLabelDrivenPrimaryImageInputPolicy,
    ObjectLineageTransformContractModule,
):
    module_name = "MaskObjects"
    function_name = "mask_objects"
    validated = True
    confidence = 1.0
    outline_retention_setting = "Retain outlines of the resulting objects?"
    outline_image_setting = "Name the outline image"
    input_objects_setting = SettingNameFamily(
        "Select the input objects", aliases=("Select objects to be masked",)
    )
    output_objects_setting = SettingNameFamily(
        "Name the output objects", aliases=("Name the masked objects",)
    )
    masking_image_setting = SettingNameFamily("Select the masking image")
    masking_objects_setting = SettingNameFamily("Select the masking object")
    input_objects_binding = SettingToKeywordBinding.input(
        input_objects_setting, ObjectLabelsArtifactType, runtime_parameter_name="labels"
    )
    output_objects_binding = SettingToKeywordBinding.output(
        output_objects_setting, ObjectLabelsArtifactType
    )
    masking_image_binding = SettingToKeywordBinding.input(
        masking_image_setting, ImageArtifactType, runtime_parameter_name="mask"
    )
    masking_objects_binding = SettingToKeywordBinding.input(
        masking_objects_setting, ObjectLabelsArtifactType, runtime_parameter_name="mask"
    )

    @classmethod
    def primary_image_domain_input_binding(cls) -> SettingToKeywordBinding:
        """Use the objects being masked, not the masking object, as the domain."""

        return cls.input_objects_binding

    setting_bindings = (
        masking_image_binding,
        input_objects_binding,
        masking_objects_binding,
        output_objects_binding,
        SettingToKeywordBinding(
            "Handling of objects that are partially masked",
            "overlap_handling",
            cellprofiler_enum_value_setting_parser(MaskObjectsOverlapHandling),
        ),
        SettingToKeywordBinding(
            "Fraction of object that must overlap",
            "overlap_fraction",
            parse_cellprofiler_float,
        ),
        SettingToKeywordBinding(
            "Numbering of resulting objects",
            "numbering",
            cellprofiler_enum_value_setting_parser(MaskObjectsNumberingChoice),
        ),
        SettingToKeywordBinding(
            "Invert the mask?", "invert_mask", parse_cellprofiler_bool
        ),
    )
    ignored_settings = (
        "Mask using a region defined by other objects or by binary image",
        outline_retention_setting,
    )

    @classmethod
    def ignored_settings_for(
        cls, module: "ModuleBlock"
    ) -> tuple[str | "SettingNameFamily", ...]:
        ignored = tuple(cls.ignored_settings)
        if cls.setting_value(module, cls.outline_retention_setting) == "No":
            return (*ignored, cls.outline_image_setting)
        return ignored

    @classmethod
    def _masking_artifact_names(cls, module):
        if module is None:
            return None, None
        masking_image = normalized_symbol_name(
            optional_setting_value(module, cls.masking_image_setting) or ""
        )
        masking_objects = normalized_symbol_name(
            optional_setting_value(module, cls.masking_objects_setting) or ""
        )
        if masking_image is not None and masking_objects is not None:
            raise ValueError(
                "MaskObjects cannot declare both masking image and object inputs."
            )
        return masking_image, masking_objects

    @classmethod
    def active_artifact_bindings(cls, module=None, *, invocation_key=None):
        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        masking_image, _ = cls._masking_artifact_names(module)
        _, masking_objects = cls._masking_artifact_names(module)
        return tuple(
            binding
            for binding in bindings
            if masking_image is not None or binding is not cls.masking_image_binding
            if masking_objects is not None or binding is not cls.masking_objects_binding
        )


class OpeningModule(ImageStructuringElementModule):
    module_name = "Opening"
    function_name = "opening"
    validated = True
    confidence = 1.0


class RemoveHolesModule(
    ZStackFunctionVariantModule,
    CellProfilerModule,
):
    module_name = "RemoveHoles"
    function_name = "remove_holes"
    validated = True
    function_variants = ("remove_holes_3d",)
    confidence = 1.0
    input_image_setting = SettingNameFamily("Select the input image")
    output_image_setting = SettingNameFamily("Name the output image")
    setting_bindings = (
        SettingToKeywordBinding.input(input_image_setting, ImageArtifactType),
        SettingToKeywordBinding.output(output_image_setting, ImageArtifactType),
        SettingToKeywordBinding(
            "Size of holes to fill", "diameter", parse_cellprofiler_float
        ),
    )


class ResizeObjectsModule(
    ObjectLabelDrivenPrimaryImageInputPolicy,
    LabelsObjectInputPolicy,
    InputObjectLabelWithoutParentImageOutputSourceContextPolicyMixin,
    ObjectLineageTransformModule,
):
    module_name = "ResizeObjects"
    function_name = "resize_objects"
    validated = True
    function_variants = ("resize_objects_3d",)
    confidence = 1.0
    desired_dimensions_image_setting = SettingNameFamily(
        "Select the image with the desired dimensions"
    )
    method_setting = SettingNameFamily("Method")
    factor_z_setting = SettingNameFamily("Z Factor")
    planes_setting = SettingNameFamily("Planes (Z)")
    volumetric_settings = (factor_z_setting, planes_setting)
    ignored_settings = (desired_dimensions_image_setting,)
    setting_bindings = (
        SettingToKeywordBinding(
            "Method", "method", normalize_cellprofiler_setting_name
        ),
        SettingToKeywordBinding("X Factor", "factor_x", parse_cellprofiler_float),
        SettingToKeywordBinding("Y Factor", "factor_y", parse_cellprofiler_float),
        SettingToKeywordBinding(factor_z_setting, "factor_z", parse_cellprofiler_float),
        SettingToKeywordBinding("Width (X)", "width", parse_cellprofiler_int),
        SettingToKeywordBinding("Height (Y)", "height", parse_cellprofiler_int),
        SettingToKeywordBinding(planes_setting, "planes", parse_cellprofiler_int),
    )

    @classmethod
    def uses_volumetric_resize(cls, module: "ModuleBlock") -> bool:
        """Return whether this declaration changes the input label-stack depth."""

        return any(
            setting_values(module, setting) for setting in cls.volumetric_settings
        )

    @classmethod
    def changes_stack_depth(cls, module: "ModuleBlock") -> bool:
        """Return whether the declared resize changes the object stack depth."""

        method = cellprofiler_enum_from_literal(
            ResizeObjectsMethod,
            required_setting_value(module, cls.method_setting),
        )
        if method is ResizeObjectsMethod.DIMENSIONS:
            return True
        return (
            parse_cellprofiler_float(
                required_setting_value(module, cls.factor_z_setting)
            )
            != 1.0
        )

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
            if cls.uses_volumetric_resize(module)
            else str(cls.function_name)
        )
        return cls.require_callable(function_name)

    @classmethod
    def artifact_output_relations(
        cls,
        module,
        *,
        binding,
        name,
        invocation_key,
        step_context,
        artifact_inputs: ArtifactSpecCollection,
        output_position,
    ):
        inherited = super().artifact_output_relations(
            module,
            binding=binding,
            name=name,
            invocation_key=invocation_key,
            step_context=step_context,
            artifact_inputs=artifact_inputs,
            output_position=output_position,
        )
        if not cls.uses_volumetric_resize(module) or not cls.changes_stack_depth(
            module
        ):
            return inherited
        (source_name,) = cls.artifact_names_for_binding(
            module,
            cls.input_objects_binding,
        )
        source = artifact_inputs.require_by_name_and_artifact_type(
            source_name,
            ObjectLabelsArtifactType,
        )
        return (GroupLineageSourceRelation(source=source.ref()),)


from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import logging
import os
import time
from typing import ClassVar
from metaclass_registry import AutoRegisterMeta
from numba import njit
from openhcs.constants.constants import (
    AllComponents,
    MemoryType,
)
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.callable_contract import runtime_image_execution_mode
from openhcs.core.memory.decorators import numpy as numpy_decorator
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
)
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelInputExecutionMode,
    object_label_input_execution_mode,
    special_inputs,
    )
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.image_shapes import (
    apply_over_trailing_spatial_axes,
    trailing_spatial_factors,
    trailing_spatial_target_shape,
)
from openhcs.core.runtime_object_label_domains import (
    DenseObjectLabelConsecutiveRelabelingStrategy,
    ExplicitObjectLabelDomainDeclaration,
    ObjectLabelDomain,
    ObjectLabelIdDomainStrategy,
    PresentObjectLabelIdsDomainDeclaration,
)
from openhcs.core.runtime_relationships import (
    DirectedObjectRelationshipPayload,
    object_label_identity_lineage_payload,
    object_label_parent_child_payload,
)
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_metadata,
    with_image_payload_data,
)
from openhcs.core.runtime_object_labels import (
    object_label_value_with_dense_labels,
)
from openhcs.core.source_metadata import SourceVoxelSpacing
from openhcs.core.source_spatial_domain import SourceSpatialDomainAdapter
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.cellprofiler.enum_attributes import (
    CellProfilerEnumAttributeMixin,
)
from openhcs.processing.backends.cellprofiler.relationships import (
    ObjectRelationshipBackendStrategy,
)
from openhcs.processing.backends.cellprofiler.structuring_elements import (
    StructuringElement,
    StructuringElementInput,
    StructuringElementSize,
    adapt_structuring_element_rank,
    build_structuring_element,
)
from openhcs.processing.backends.cellprofiler.worm_geometry import (
    branchpoints as _branchpoints,
    endpoints as _endpoints,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    CellProfilerBackendAuthority,
)
from openhcs.processing.backends.analysis.region_properties import (
    LabelRegionPropertiesBackendStrategy,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
    SliceBySliceRuntimeParameter,
)

HolePredicate = Callable[[int, bool], bool]
ConnectivityStructureBuilder = Callable[[int], np.ndarray]
LabelBoundingBox = tuple[int, tuple[slice, ...]]
LabelBoundingBoxes = list[LabelBoundingBox]
SCIPY_CONSTANT_BOUNDARY_MODE = "constant"
MORPH_CONVOLUTION_MODE = "constant"
MORPH_CONVEX_HULL_OPERATION = "convex_hull"
MORPHOLOGY_STRATEGY_REGISTRY_KEY = "strategy_label"
SPARSE_CUBIC_BOOLEAN_RESAMPLE_RADIUS = 2.0
EIGHT_NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
FOUR_CONNECTED_KERNEL = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
logger = logging.getLogger(__name__)


class MorphOperation(Enum):
    """CellProfiler Morph operation names."""

    BRANCHPOINTS = "branchpoints"
    BRIDGE = "bridge"
    CLEAN = "clean"
    CONVEX_HULL = MORPH_CONVEX_HULL_OPERATION
    DIAG = "diag"
    DISTANCE = "distance"
    ENDPOINTS = "endpoints"
    FILL = "fill"
    HBREAK = "hbreak"
    MAJORITY = "majority"
    OPENLINES = "openlines"
    REMOVE = "remove"
    SHRINK = "shrink"
    SKELPE = "skelpe"
    SPUR = "spur"
    THICKEN = "thicken"
    THIN = "thin"
    VBREAK = "vbreak"


class RepeatMode(Enum):
    """CellProfiler Morph repeat policies."""

    ONCE = "once"
    FOREVER = "forever"
    CUSTOM = "custom"


class ResizeObjectsMethod(Enum):
    """CellProfiler ResizeObjects size policy."""

    DIMENSIONS = ("dimensions", "to_size", "manual")
    FACTOR = ("factor", "by_factor")

    def __new__(cls, value: str, *cellprofiler_literals: str):
        member = object.__new__(cls)
        member._value_ = value
        member.cellprofiler_literals = cellprofiler_literals
        return member


class FillMode(Enum):
    HOLES = "holes"
    CONVEX_HULL = MORPH_CONVEX_HULL_OPERATION


class MaskChoice(Enum):
    """MaskObjects mask source kind."""

    OBJECTS = "objects"
    IMAGE = "image"


@dataclass(frozen=True, slots=True)
class FillObjectsRequest:
    """Complete request for one FillObjects mode implementation."""

    image: np.ndarray
    label_array: np.ndarray
    diameter: float
    morphology_backend_provider: BackendProviderInput


class FillObjectsModeStrategy(
    EnumKeyedStrategyMixin[FillMode], ABC, metaclass=AutoRegisterMeta
):
    """Nominal implementation for one FillObjects mode."""

    __registry_key__ = MORPHOLOGY_STRATEGY_REGISTRY_KEY
    __skip_if_no_key__ = True
    __enum_member_attr__ = "mode"
    mode: ClassVar[FillMode | None] = None
    strategy_label: ClassVar[str | None] = None

    @classmethod
    def for_mode(cls, mode: FillMode | str) -> "FillObjectsModeStrategy":
        return cls.for_enum_member(coerce_cellprofiler_enum(FillMode, mode))

    @abstractmethod
    def fill(self, request: FillObjectsRequest) -> np.ndarray:
        """Return labels transformed by this fill mode."""


class FillObjectHolesStrategy(FillObjectsModeStrategy):
    """Fill holes within each object below the configured area threshold."""

    mode = FillMode.HOLES

    def fill(self, request: FillObjectsRequest) -> np.ndarray:
        from skimage.morphology import remove_small_holes

        filled_labels = np.zeros_like(request.label_array)
        max_hole_area = np.pi * (request.diameter / 2.0) ** 2
        region_props = (
            LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
                request.label_array
            )
        )
        for label_id in region_props.label:
            label_int = int(label_id)
            obj_mask = request.label_array == label_int
            filled_mask = remove_small_holes(
                obj_mask, area_threshold=int(max_hole_area), connectivity=1
            )
            filled_labels[filled_mask] = label_int
        return filled_labels


class FillObjectConvexHullStrategy(FillObjectsModeStrategy):
    """Replace each object support with its convex hull."""

    mode = FillMode.CONVEX_HULL

    def fill(self, request: FillObjectsRequest) -> np.ndarray:
        filled_labels = np.zeros_like(request.label_array)
        morphology = MorphologyBackendStrategy.for_callable(
            fill_objects, backend_provider=request.morphology_backend_provider
        )
        region_props = (
            LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
                request.label_array
            )
        )
        for index, label_id in enumerate(region_props.label):
            label_int = int(label_id)
            obj_mask = request.label_array == label_int
            minr = int(region_props.bbox_min_y[index])
            minc = int(region_props.bbox_min_x[index])
            maxr = int(region_props.bbox_max_y[index])
            maxc = int(region_props.bbox_max_x[index])
            obj_crop = obj_mask[minr:maxr, minc:maxc]
            if obj_crop.sum() > 2:
                hull = morphology.convex_hull_image(obj_crop)
                filled_labels[minr:maxr, minc:maxc][hull] = label_int
            else:
                filled_labels[obj_mask] = label_int
        return filled_labels


@dataclass(frozen=True, slots=True)
class PlaneStats:
    """Base summary row for operations reported per runtime plane."""

    slice_index: int


@dataclass(frozen=True, slots=True)
class ObjectRemovalStats(PlaneStats):
    """Shared object-removal count summary."""

    objects_removed: int


@dataclass(frozen=True, slots=True)
class ObjectCountTransitionStats(PlaneStats):
    """Shared input/output object-count transition summary."""

    input_object_count: int
    output_object_count: int


@dataclass(frozen=True, slots=True)
class ObjectRemovalTransitionStats(ObjectCountTransitionStats):
    """Object-count transition that also reports removed object count."""

    objects_removed: int


@dataclass(frozen=True, slots=True)
class ResizeObjectsStats:
    slice_index: int
    original_height: int
    original_width: int
    new_height: int
    new_width: int
    object_count: int


@dataclass(frozen=True, slots=True)
class ResizeObjects3DStats:
    slice_index: int
    original_depth: int
    original_height: int
    original_width: int
    new_depth: int
    new_height: int
    new_width: int
    object_count: int


@dataclass(frozen=True, slots=True)
class ResizeObjectsRequest:
    """Runtime resize configuration independent of public module signature shape."""

    labels: np.ndarray
    method: ResizeObjectsMethod
    factor_x: float
    factor_y: float
    factor_z: float
    width: int
    height: int
    planes: int

    def target_shape(self) -> tuple[int, ...]:
        return resize_objects_target_shape(
            self.labels.shape, planes=self.planes, height=self.height, width=self.width
        )

    def zoom_factors(self) -> tuple[float, ...]:
        if self.method == ResizeObjectsMethod.DIMENSIONS:
            return tuple(
                np.divide(np.multiply(1.0, self.target_shape()), self.labels.shape)
            )
        return resize_objects_zoom_factors(
            self.labels.ndim,
            factor_z=self.factor_z,
            factor_y=self.factor_y,
            factor_x=self.factor_x,
        )


@dataclass(frozen=True, slots=True)
class ErosionStats(ObjectRemovalTransitionStats):
    pass


@dataclass(frozen=True, slots=True)
class DilationStats:
    slice_index: int
    object_count: int
    mean_area_before: float
    mean_area_after: float


@dataclass(frozen=True, slots=True)
class DilationStats3D:
    object_count: int
    mean_volume_before: float
    mean_volume_after: float


@dataclass(frozen=True, slots=True)
class CentroidStats:
    slice_index: int
    object_count: int


@dataclass(frozen=True)
class MorphOperationRequest:
    """Execution context shared by registered Morph operation strategies."""

    image: np.ndarray
    iterations: int
    rescale_values: bool
    line_length: int
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION
    memory_type: MemoryType = MemoryType.NUMPY


NeighborConvolutionTransition = Callable[[np.ndarray, np.ndarray], np.ndarray]


class MorphOperationStrategy(
    EnumKeyedStrategyMixin[MorphOperation], ABC, metaclass=AutoRegisterMeta
):
    """Registered implementation authority for CellProfiler Morph operations."""

    __enum_member_attr__ = "operation"
    operation: ClassVar[MorphOperation | None] = None

    @abstractmethod
    def apply(self, request: MorphOperationRequest) -> np.ndarray:
        """Apply this operation to the runtime request."""


class RepeatModeStrategy(
    EnumKeyedStrategyMixin[RepeatMode], ABC, metaclass=AutoRegisterMeta
):
    """Registered iteration-count policy for CellProfiler Morph repeat modes."""

    __enum_member_attr__ = "repeat_mode"
    repeat_mode: ClassVar[RepeatMode | None] = None

    @abstractmethod
    def repeat_count(self, custom_repeats: int) -> int:
        """Return the concrete repeat count for this policy."""


class OnceRepeatModeStrategy(RepeatModeStrategy):
    """Run a Morph operation once."""

    repeat_mode = RepeatMode.ONCE

    def repeat_count(self, custom_repeats: int) -> int:
        del custom_repeats
        return 1


class ForeverRepeatModeStrategy(RepeatModeStrategy):
    """CellProfiler's bounded approximation of FOREVER repeat mode."""

    repeat_mode = RepeatMode.FOREVER

    def repeat_count(self, custom_repeats: int) -> int:
        del custom_repeats
        return 10000


class CustomRepeatModeStrategy(RepeatModeStrategy):
    """Run a Morph operation a declared number of times."""

    repeat_mode = RepeatMode.CUSTOM

    def repeat_count(self, custom_repeats: int) -> int:
        return custom_repeats


def _ensure_binary(image: np.ndarray) -> np.ndarray:
    if image.dtype != bool:
        return image != 0
    return image


class IterativeConvolutionMorphOperationStrategy(MorphOperationStrategy):
    """Template strategy for Morph operations driven by neighbor convolution."""

    kernel: ClassVar[np.ndarray | None] = None
    transition: ClassVar[NeighborConvolutionTransition | None] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.operation is not None and (cls.kernel is None or cls.transition is None):
            raise TypeError(
                f"{cls.__name__} must declare kernel and transition for Morph"
            )

    def apply(self, request: MorphOperationRequest) -> np.ndarray:
        from scipy.ndimage import convolve

        kernel = type(self).kernel
        transition = type(self).transition
        if kernel is None or transition is None:
            raise TypeError(f"{type(self).__name__} cannot run convolutional Morph")
        result = _ensure_binary(request.image).astype(np.float32)
        for _ in range(request.iterations):
            neighbor_count = convolve(
                result.astype(np.uint8), kernel, mode=MORPH_CONVOLUTION_MODE, cval=0
            )
            result = transition(result, neighbor_count)
        return result


def _bridge(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    from scipy.ndimage import convolve

    result = _ensure_binary(image).astype(np.float32)
    patterns = [
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]),
        np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
        np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]]),
        np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]]),
    ]
    for _ in range(iterations):
        for pattern in patterns:
            match = convolve(result, pattern, mode=MORPH_CONVOLUTION_MODE, cval=0)
            result = np.where(match == 2, 1.0, result)
    return result


def _convex_hull(
    image: np.ndarray, morphology: "MorphologyBackendStrategy"
) -> np.ndarray:
    binary = _ensure_binary(image)
    if not np.any(binary):
        return np.zeros_like(image, dtype=np.float32)
    return morphology.convex_hull_image(binary).astype(np.float32)


def _diag(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    from scipy.ndimage import binary_dilation

    result = _ensure_binary(image).astype(np.float32)
    struct = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool)
    for _ in range(iterations):
        dilated = binary_dilation(result > 0, structure=struct)
        result = np.maximum(result, dilated.astype(np.float32))
    return result


def _distance(image: np.ndarray, rescale: bool = True) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    binary = _ensure_binary(image)
    dist = distance_transform_edt(binary)
    if rescale and dist.max() > 0:
        dist = dist / dist.max()
    return dist.astype(np.float32)


def _hbreak(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    from scipy.ndimage import convolve

    result = _ensure_binary(image).astype(np.float32)
    pattern = np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]], dtype=np.float32)
    for _ in range(iterations):
        match = convolve(result, pattern, mode=MORPH_CONVOLUTION_MODE, cval=0)
        result = np.where((match >= 6) & (result > 0), 0.0, result)
    return result


def _majority(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    from scipy.ndimage import convolve

    result = _ensure_binary(image).astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.float32)
    for _ in range(iterations):
        neighbor_sum = convolve(result, kernel, mode=MORPH_CONVOLUTION_MODE, cval=0)
        result = (neighbor_sum >= 5).astype(np.float32)
    return result


class OpenLineStructuringElement(ABC, metaclass=AutoRegisterMeta):
    """Registered structuring-element authority for Morph OPENLINES angles."""

    __registry_key__ = "angle"
    __skip_if_no_key__ = True
    angle: ClassVar[int | None] = None

    @classmethod
    def registered_elements(cls) -> tuple["OpenLineStructuringElement", ...]:
        return tuple((strategy_type() for strategy_type in cls.__registry__.values()))

    @abstractmethod
    def structure(self, line_length: int) -> np.ndarray:
        """Return this angle's line structuring element."""


class HorizontalOpenLineStructuringElement(OpenLineStructuringElement):
    """Horizontal OPENLINES structuring element."""

    angle = 0

    def structure(self, line_length: int) -> np.ndarray:
        return np.ones((1, line_length), dtype=bool)


class RisingDiagonalOpenLineStructuringElement(OpenLineStructuringElement):
    """Rising diagonal OPENLINES structuring element."""

    angle = 45

    def structure(self, line_length: int) -> np.ndarray:
        return np.eye(line_length, dtype=bool)


class VerticalOpenLineStructuringElement(OpenLineStructuringElement):
    """Vertical OPENLINES structuring element."""

    angle = 90

    def structure(self, line_length: int) -> np.ndarray:
        return np.ones((line_length, 1), dtype=bool)


class FallingDiagonalOpenLineStructuringElement(OpenLineStructuringElement):
    """Falling diagonal OPENLINES structuring element."""

    angle = 135

    def structure(self, line_length: int) -> np.ndarray:
        return np.fliplr(np.eye(line_length, dtype=bool))


def _openlines(image: np.ndarray, line_length: int = 3) -> np.ndarray:
    from scipy.ndimage import binary_dilation, binary_erosion

    binary = _ensure_binary(image)
    result = np.zeros_like(binary)
    for element in OpenLineStructuringElement.registered_elements():
        struct = element.structure(line_length)
        eroded = binary_erosion(binary, structure=struct)
        dilated = binary_dilation(eroded, structure=struct)
        result = result | dilated
    return result.astype(np.float32)


def _shrink(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    from skimage.morphology import thin

    binary = _ensure_binary(image)
    return thin(binary, max_num_iter=iterations).astype(np.float32)


def _skelpe(image: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize

    binary = _ensure_binary(image)
    return skeletonize(binary).astype(np.float32)


def _thicken(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    from scipy.ndimage import binary_dilation

    result = _ensure_binary(image)
    for _ in range(iterations):
        result = binary_dilation(result)
    return result.astype(np.float32)


def _thin(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    from skimage.morphology import thin

    binary = _ensure_binary(image)
    return thin(binary, max_num_iter=iterations).astype(np.float32)


def _vbreak(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    from scipy.ndimage import convolve

    result = _ensure_binary(image).astype(np.float32)
    pattern = np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=np.float32)
    for _ in range(iterations):
        match = convolve(result, pattern, mode=MORPH_CONVOLUTION_MODE, cval=0)
        result = np.where((match >= 6) & (result > 0), 0.0, result)
    return result


def convex_hull_morph_operation(request: MorphOperationRequest) -> np.ndarray:
    """Run Morph CONVEX_HULL through the configured morphology backend."""
    morphology = MorphologyBackendStrategy.for_memory_type(
        request.memory_type, backend_provider=request.backend_provider
    )
    return _convex_hull(request.image, morphology)


class BranchpointsMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.BRANCHPOINTS

    def apply(self, request: MorphOperationRequest) -> np.ndarray:
        return _branchpoints(request.image)


class BridgeMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.BRIDGE

    def apply(self, request: MorphOperationRequest) -> np.ndarray:
        return _bridge(request.image, request.iterations)


class CleanMorphOperationStrategy(IterativeConvolutionMorphOperationStrategy):
    operation = MorphOperation.CLEAN
    kernel = EIGHT_NEIGHBOR_KERNEL
    transition = staticmethod(
        lambda result, neighbor_count: np.where(neighbor_count == 0, 0.0, result)
    )


class ConvexHullMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.CONVEX_HULL

    def apply(self, request: MorphOperationRequest) -> np.ndarray:
        return convex_hull_morph_operation(request)


class DiagMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.DIAG

    def apply(self, request: MorphOperationRequest) -> np.ndarray:
        return _diag(request.image, request.iterations)


class DistanceMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.DISTANCE

    def apply(self, request: MorphOperationRequest) -> np.ndarray:
        return _distance(request.image, request.rescale_values)


class EndpointsMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.ENDPOINTS

    def apply(self, request: MorphOperationRequest) -> np.ndarray:
        return _endpoints(request.image)


class FillMorphOperationStrategy(IterativeConvolutionMorphOperationStrategy):
    operation = MorphOperation.FILL
    kernel = EIGHT_NEIGHBOR_KERNEL
    transition = staticmethod(
        lambda result, neighbor_count: np.where(neighbor_count == 8, 1.0, result)
    )


class HBreakMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.HBREAK

    def apply(self, request: MorphOperationRequest) -> np.ndarray:
        return _hbreak(request.image, request.iterations)


class MajorityMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.MAJORITY

    def apply(self, request: MorphOperationRequest) -> np.ndarray:
        return _majority(request.image, request.iterations)


class OpenLinesMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.OPENLINES

    def apply(self, request: MorphOperationRequest) -> np.ndarray:
        return _openlines(request.image, request.line_length)


class RemoveMorphOperationStrategy(IterativeConvolutionMorphOperationStrategy):
    operation = MorphOperation.REMOVE
    kernel = FOUR_CONNECTED_KERNEL
    transition = staticmethod(
        lambda result, neighbor_count: np.where(neighbor_count == 4, 0.0, result)
    )


class ShrinkMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.SHRINK

    def apply(self, request: MorphOperationRequest) -> np.ndarray:
        return _shrink(request.image, request.iterations)


class SkelpeMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.SKELPE

    def apply(self, request: MorphOperationRequest) -> np.ndarray:
        return _skelpe(request.image)


class SpurMorphOperationStrategy(IterativeConvolutionMorphOperationStrategy):
    operation = MorphOperation.SPUR
    kernel = EIGHT_NEIGHBOR_KERNEL
    transition = staticmethod(
        lambda result, neighbor_count: np.where(
            (neighbor_count == 1) & (result > 0), 0.0, result
        )
    )


class ThickenMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.THICKEN

    def apply(self, request: MorphOperationRequest) -> np.ndarray:
        return _thicken(request.image, request.iterations)


class ThinMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.THIN

    def apply(self, request: MorphOperationRequest) -> np.ndarray:
        return _thin(request.image, request.iterations)


class VBreakMorphOperationStrategy(MorphOperationStrategy):
    operation = MorphOperation.VBREAK

    def apply(self, request: MorphOperationRequest) -> np.ndarray:
        return _vbreak(request.image, request.iterations)


def apply_morph_operation(
    image: np.ndarray,
    operation: MorphOperation = MorphOperation.THIN,
    repeat_mode: RepeatMode = RepeatMode.ONCE,
    custom_repeats: int = 2,
    rescale_values: bool = True,
    line_length: int = 3,
    morphology_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    memory_type: MemoryType = MemoryType.NUMPY,
) -> np.ndarray:
    """Apply one CellProfiler Morph operation through registered backend policies."""
    iterations = RepeatModeStrategy.for_enum_member(repeat_mode).repeat_count(
        custom_repeats
    )
    return MorphOperationStrategy.for_enum_member(operation).apply(
        MorphOperationRequest(
            image=image,
            iterations=iterations,
            rescale_values=rescale_values,
            line_length=line_length,
            backend_provider=morphology_backend_provider,
            memory_type=memory_type,
        )
    )


@numpy_decorator(contract=ProcessingContract.PURE_2D)
def morph(
    image: np.ndarray,
    operation: MorphOperation = MorphOperation.THIN,
    repeat_mode: RepeatMode = RepeatMode.ONCE,
    custom_repeats: int = 2,
    rescale_values: bool = True,
    line_length: int = 3,
    morphology_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> np.ndarray:
    """Decorated CellProfiler Morph entrypoint backed by registered strategies.

    Args:
        operation: Morphological operation to apply to the binary foreground.
        repeat_mode: Iteration policy: once, a bounded 10,000-pass approximation
            of ``forever``, or ``custom``.
        custom_repeats: Number of passes used when ``repeat_mode`` is ``custom``.
        rescale_values: Normalize the distance-transform result to 0..1 when using
            the ``distance`` operation.
        line_length: Structuring-line length in pixels for the ``openlines``
            operation.
    """
    return apply_morph_operation(
        image=image,
        operation=operation,
        repeat_mode=repeat_mode,
        custom_repeats=custom_repeats,
        rescale_values=rescale_values,
        line_length=line_length,
        morphology_backend_provider=morphology_backend_provider,
    )


def _morph_image_pixels(
    image: np.ndarray,
    structuring_element: StructuringElement | str,
    size: int,
    operation: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    pixels = np.asarray(image)
    footprint = build_structuring_element(structuring_element, size)
    footprint = adapt_structuring_element_rank(footprint, pixels.ndim)
    return operation(pixels, footprint)


def _morph_image_payload(
    image: np.ndarray,
    structuring_element: StructuringElement | str,
    size: int,
    operation: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    pixel_data = image_payload_data(image)
    result = _morph_image_pixels(pixel_data, structuring_element, size, operation)
    return with_image_payload_data(
        image,
        result.astype(pixel_data.dtype, copy=False),
        metadata=image_payload_metadata(image).without_unit_interval_intensity_scale(),
    )


@runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
@numpy_decorator(contract=ProcessingContract.FLEXIBLE)
def closing(
    image: np.ndarray,
    structuring_element: StructuringElementInput = StructuringElement.DISK,
    size: StructuringElementSize = 3,
    morphology_backend_provider: BackendProviderInput = CellProfilerBackendProvider.NATIVE,
    *,
    slice_by_slice: bool = True,
) -> np.ndarray:
    """Apply CellProfiler-compatible grayscale closing to an image plane."""
    morphology = MorphologyBackendStrategy.for_callable(
        closing, backend_provider=morphology_backend_provider
    )
    return _morph_image_payload(
        image,
        structuring_element,
        size,
        morphology.grayscale_closing,
    )


@runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
@numpy_decorator(contract=ProcessingContract.FLEXIBLE)
def opening(
    image: np.ndarray,
    structuring_element: StructuringElementInput = StructuringElement.DISK,
    size: StructuringElementSize = 3,
    morphology_backend_provider: BackendProviderInput = CellProfilerBackendProvider.NATIVE,
    *,
    slice_by_slice: bool = True,
) -> np.ndarray:
    """Apply CellProfiler-compatible grayscale opening to an image plane."""
    morphology = MorphologyBackendStrategy.for_callable(
        opening, backend_provider=morphology_backend_provider
    )
    return _morph_image_payload(
        image,
        structuring_element,
        size,
        morphology.grayscale_opening,
    )


@runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
@numpy_decorator(contract=ProcessingContract.FLEXIBLE)
def dilate_image(
    image: np.ndarray,
    structuring_element: StructuringElementInput = StructuringElement.DISK,
    size: StructuringElementSize = 3,
    *,
    slice_by_slice: bool = True,
) -> np.ndarray:
    """Apply grayscale dilation to an image plane."""
    from skimage.morphology import dilation

    dilated = _morph_image_pixels(
        image,
        structuring_element,
        size,
        lambda spatial_image, footprint: dilation(spatial_image, footprint),
    )
    return dilated.astype(image.dtype)


@runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
@numpy_decorator(contract=ProcessingContract.FLEXIBLE)
def erode_image(
    image: np.ndarray,
    structuring_element: StructuringElementInput = StructuringElement.DISK,
    size: StructuringElementSize = 3,
    *,
    slice_by_slice: bool = True,
) -> np.ndarray:
    """Apply grayscale erosion to an image plane."""
    from skimage.morphology import erosion

    eroded = _morph_image_pixels(
        image,
        structuring_element,
        size,
        lambda spatial_image, footprint: erosion(spatial_image, footprint),
    )
    return eroded.astype(image.dtype, copy=False)


@numpy_decorator(contract=ProcessingContract.PURE_2D)
def remove_holes(image: np.ndarray, diameter: float = 1.0) -> np.ndarray:
    """Fill binary holes smaller than the CellProfiler diameter threshold."""
    return HoleRemovalDiameterPolicy(diameter=diameter, volumetric=False).apply(image)


@numpy_decorator(contract=ProcessingContract.PURE_3D)
def remove_holes_3d(image: np.ndarray, diameter: float = 1.0) -> np.ndarray:
    """Fill volumetric holes smaller than the CellProfiler diameter threshold."""
    return HoleRemovalDiameterPolicy(diameter=diameter, volumetric=True).apply(image)


@numpy_decorator(contract=ProcessingContract.PURE_2D)
def morphological_skeleton_2d(image: np.ndarray) -> np.ndarray:
    """Compute the 2-D morphological skeleton of a binary image."""
    from skimage.morphology import skeletonize

    return skeletonize(image > 0).astype(np.float32)


@numpy_decorator(contract=ProcessingContract.PURE_3D)
def morphological_skeleton_3d(image: np.ndarray) -> np.ndarray:
    """Compute the 3-D morphological skeleton of a binary volume."""
    from skimage.morphology import skeletonize_3d

    return skeletonize_3d(image > 0).astype(np.float32)


@numpy_decorator(contract=ProcessingContract.PURE_2D)
def morphologicalskeleton(image: np.ndarray) -> np.ndarray:
    """Compute CellProfiler MorphologicalSkeleton on one image plane."""
    from skimage.morphology import skeletonize

    return skeletonize(image > 0).astype(np.float32)


@dataclass(frozen=True, slots=True)
class HoleRemovalDiameterPolicy:
    """CellProfiler RemoveHoles diameter threshold semantics."""

    diameter: float
    volumetric: bool = False

    @property
    def threshold(self) -> int:
        radius = self.diameter / 2.0
        if self.volumetric:
            threshold = 4.0 / 3.0 * np.pi * radius**3
        else:
            threshold = np.pi * radius**2
        return max(1, int(threshold))

    def binary_image(self, image: np.ndarray) -> np.ndarray:
        from skimage import img_as_bool

        if image.dtype.kind == "f":
            return img_as_bool(image)
        if image.dtype.kind in ("u", "i"):
            return image > 0
        return image.astype(bool)

    def apply(self, image: np.ndarray) -> np.ndarray:
        import skimage.morphology

        result = skimage.morphology.remove_small_holes(
            self.binary_image(image), area_threshold=self.threshold
        )
        return result.astype(np.float32)


def face_connected_component_structure(ndim: int) -> np.ndarray:
    """Return the SciPy face-connectivity structure for an nd label image."""
    from scipy import ndimage as ndi

    return ndi.generate_binary_structure(ndim, 1)


def full_connected_component_structure(ndim: int) -> np.ndarray:
    """Return the full 3-wide neighborhood structure for an nd label image."""
    return np.ones((3,) * ndim, dtype=bool)


class ConnectedComponentConnectivity(ABC, metaclass=AutoRegisterMeta):
    """Registered structuring-element policy for connected components."""

    __registry_key__ = "connectivity"
    __skip_if_no_key__ = True
    connectivity: ClassVar[int | None] = None
    structure_builder: ClassVar[ConnectivityStructureBuilder | None] = None

    @classmethod
    def for_connectivity(cls, connectivity: int) -> "ConnectedComponentConnectivity":
        strategy_type = cls.__registry__.get(connectivity)
        if strategy_type is None:
            raise ValueError(
                f"Unsupported connected-component connectivity: {connectivity}"
            )
        return strategy_type()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.connectivity is not None and cls.structure_builder is None:
            raise TypeError(
                f"{cls.__name__} must declare structure_builder for connectivity"
            )

    def structure(self, ndim: int) -> np.ndarray:
        structure_builder = type(self).structure_builder
        if structure_builder is None:
            raise TypeError(f"{type(self).__name__} cannot build connectivity")
        return structure_builder(ndim)


class FaceConnectedComponents(ConnectedComponentConnectivity):
    """Face-connected components."""

    connectivity = 1
    structure_builder = staticmethod(face_connected_component_structure)


class FullConnectedComponents(ConnectedComponentConnectivity):
    """Fully connected components over a 3-wide neighborhood."""

    connectivity = 2
    structure_builder = staticmethod(full_connected_component_structure)


class CellProfilerDeclumpMethod(Enum):
    """Typed declumping modes that affect morphology backend geometry."""

    INTENSITY = "intensity"
    SHAPE = "shape"


class FillHolesOption(CellProfilerEnumAttributeMixin, Enum):
    """CellProfiler IdentifyPrimaryObjects hole-fill phase policy."""

    __cellprofiler_attribute_names__ = ("fill_before_declump", "fill_after_declump")
    NEVER = ("never", False, False)
    AFTER_BOTH = ("after_both", True, True)
    AFTER_DECLUMP = ("after_declump", False, True)

    def before_declump_requested(self, *, use_advanced_settings: bool) -> bool:
        """Return whether CP fills binary foreground holes before declumping."""
        return not use_advanced_settings or self.fill_before_declump

    def after_declump_requested(self, *, use_advanced_settings: bool) -> bool:
        """Return whether CP fills labeled-object holes after declumping/filtering."""
        return not use_advanced_settings or self.fill_after_declump


CELLPROFILER_LOW_RES_AUTO_MAXIMA_SUPPRESSION_SIZE = 7.0


@dataclass(frozen=True, slots=True)
class DeclumpingMaximaGeometry:
    """CellProfiler declumping maxima resize and suppression geometry."""

    image_resize_factor: float
    suppress_size: float

    @classmethod
    def from_cellprofiler_settings(
        cls,
        *,
        min_diameter: int,
        low_res_maxima: bool,
        automatic_suppression: bool,
        maxima_suppression_size: float,
    ) -> "DeclumpingMaximaGeometry":
        if min_diameter > 10 and low_res_maxima:
            image_resize_factor = 10.0 / float(min_diameter)
            if automatic_suppression:
                return cls(
                    image_resize_factor,
                    CELLPROFILER_LOW_RES_AUTO_MAXIMA_SUPPRESSION_SIZE,
                )
            return cls(
                image_resize_factor,
                manual_declumping_size(maxima_suppression_size) * image_resize_factor
                + 0.5,
            )
        if automatic_suppression:
            return cls(1.0, float(min_diameter) / 1.5)
        return cls(1.0, manual_declumping_size(maxima_suppression_size))


@dataclass(frozen=True, slots=True)
class SparseBooleanCubicMapCoordinatesThreshold:
    """Sparse evaluator for thresholded cubic boolean coordinate resampling."""

    source: np.ndarray
    target_shape: tuple[int, int]
    divisor: float
    threshold: float = 0.5

    def execute(self) -> np.ndarray:
        """Return the thresholded cubic-resampled boolean image."""
        from scipy import ndimage as ndi

        source_array = np.asarray(self.source, dtype=bool)
        points = np.argwhere(source_array)
        if points.size == 0:
            return np.zeros(self.target_shape, dtype=bool)
        radius = SPARSE_CUBIC_BOOLEAN_RESAMPLE_RADIUS
        window_diameter = int(np.ceil(radius * 2.0 * float(self.divisor))) + 3
        dense_area = int(self.target_shape[0]) * int(self.target_shape[1])
        sparse_area = int(points.shape[0]) * window_diameter * window_diameter
        if sparse_area >= dense_area:
            coordinates = _declumping_resize_coordinates(
                self.target_shape, self.divisor
            )
            return (
                ndi.map_coordinates(source_array.astype(float), coordinates)
                > self.threshold
            )
        coefficients = ndi.spline_filter(source_array.astype(float), order=3)
        output = np.zeros(self.target_shape, dtype=bool)
        for source_y, source_x in points:
            self._evaluate_source_window(
                output, coefficients, int(source_y), int(source_x), radius
            )
        return output

    def _evaluate_source_window(
        self,
        output: np.ndarray,
        coefficients: np.ndarray,
        source_y: int,
        source_x: int,
        radius: float,
    ) -> None:
        """Evaluate one sparse source point's affected target window."""
        from scipy import ndimage as ndi

        y_start = max(0, int(np.floor((float(source_y) - radius) * self.divisor)))
        y_stop = min(
            self.target_shape[0],
            int(np.ceil((float(source_y) + radius) * self.divisor)) + 1,
        )
        x_start = max(0, int(np.floor((float(source_x) - radius) * self.divisor)))
        x_stop = min(
            self.target_shape[1],
            int(np.ceil((float(source_x) + radius) * self.divisor)) + 1,
        )
        if y_start >= y_stop or x_start >= x_stop:
            return
        coordinates = _declumping_resize_coordinates(
            (y_stop - y_start, x_stop - x_start), self.divisor
        )
        y_coordinates = coordinates[0] + float(y_start) / self.divisor
        x_coordinates = coordinates[1] + float(x_start) / self.divisor
        output[y_start:y_stop, x_start:x_stop] |= (
            ndi.map_coordinates(
                coefficients, (y_coordinates, x_coordinates), prefilter=False
            )
            > self.threshold
        )


def manual_declumping_size(size: float) -> float:
    """Return the configured manual CP declumping size."""
    size = float(size)
    if size <= 0:
        return 0.0
    return size


class MorphologyBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Nominal morphology operations keyed by OpenHCS memory type."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @classmethod
    def for_memory_type(
        cls,
        memory_type: MemoryType | str = MemoryType.NUMPY,
        *,
        backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
        prefer_centrosome: bool = False,
    ) -> "MorphologyBackendStrategy":
        if prefer_centrosome:
            if backend_provider not in (
                DEFAULT_CELLPROFILER_BACKEND_SELECTION,
                CellProfilerBackendProvider.CENTROSOME,
            ):
                raise ValueError(
                    f"prefer_centrosome=True conflicts with explicit backend_provider={backend_provider!r}"
                )
            backend_provider = CellProfilerBackendProvider.CENTROSOME
        return super().for_memory_type(memory_type, backend_provider=backend_provider)

    @classmethod
    def for_callable(
        cls,
        func: object,
        *,
        backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
        prefer_centrosome: bool = False,
    ) -> "MorphologyBackendStrategy":
        if prefer_centrosome:
            if backend_provider not in (
                DEFAULT_CELLPROFILER_BACKEND_SELECTION,
                CellProfilerBackendProvider.CENTROSOME,
            ):
                raise ValueError(
                    f"prefer_centrosome=True conflicts with explicit backend_provider={backend_provider!r}"
                )
            backend_provider = CellProfilerBackendProvider.CENTROSOME
        return super().for_callable(func, backend_provider=backend_provider)

    @abstractmethod
    def connected_components(
        self, mask: np.ndarray, *, connectivity: int = 2
    ) -> tuple[np.ndarray, int]:
        """Label foreground components in a binary 2-D mask."""

    @abstractmethod
    def disk_footprint(self, radius: float) -> np.ndarray:
        """Return a 2-D disk footprint."""

    @abstractmethod
    def declumping_suppression_footprint(
        self,
        suppress_size: float,
        *,
        min_diameter: float,
        declump_method: CellProfilerDeclumpMethod,
    ) -> np.ndarray:
        """Return the local-maxima suppression footprint for declumping."""

    @abstractmethod
    def grayscale_opening(self, image: np.ndarray, footprint: np.ndarray) -> np.ndarray:
        """Return grayscale morphological opening for a 2-D image."""

    @abstractmethod
    def grayscale_closing(self, image: np.ndarray, footprint: np.ndarray) -> np.ndarray:
        """Return grayscale morphological closing for a 2-D image."""

    @abstractmethod
    def erode_labeled_objects(
        self, labels: np.ndarray, footprint: np.ndarray
    ) -> np.ndarray:
        """Erode labeled objects while preserving label identities."""

    @abstractmethod
    def block_labels(
        self, image_shape: tuple[int, int], block_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Partition a 2-D plane into square block labels."""

    @abstractmethod
    def blockwise_minimum(
        self, image: np.ndarray, mask: np.ndarray | None, block_size: int
    ) -> np.ndarray:
        """Broadcast the masked minimum of each CellProfiler block to its pixels."""

    @abstractmethod
    def fix_labeled_result(self, values: np.ndarray) -> np.ndarray:
        """Normalize scipy.ndimage labeled reductions to an ndarray."""

    @abstractmethod
    def fill_labeled_holes(
        self,
        labels: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        size_predicate: HolePredicate | None = None,
    ) -> np.ndarray:
        """Fill enclosed background components."""

    @abstractmethod
    def fill_labeled_holes_below_size(
        self, labels: np.ndarray, maximum_hole_size: int
    ) -> np.ndarray:
        """Fill enclosed background components smaller than a size limit."""

    @abstractmethod
    def restore_removed_declump_basins(
        self,
        pre_declump_labels: np.ndarray,
        labels_before_size_filter: np.ndarray,
        labels_after_size_filter: np.ndarray,
    ) -> np.ndarray:
        """Restore removed watershed basins with one surviving original identity."""

    @abstractmethod
    def local_maxima_by_label(
        self, image: np.ndarray, labels: np.ndarray, footprint: np.ndarray
    ) -> np.ndarray:
        """Find local maxima independently within each positive label."""

    @abstractmethod
    def smooth_image_for_declumping(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        filter_size: float,
        *,
        declump_method: CellProfilerDeclumpMethod = CellProfilerDeclumpMethod.SHAPE,
        suppress_size: float | None = None,
        min_diameter: float | None = None,
    ) -> np.ndarray:
        """Smooth an image using CP's mask-corrected declumping convention."""

    def declumping_smoothing_kernel(
        self,
        filter_size: float,
        *,
        declump_method: CellProfilerDeclumpMethod = CellProfilerDeclumpMethod.SHAPE,
        suppress_size: float | None = None,
        min_diameter: float | None = None,
    ) -> np.ndarray:
        """Return the CP Gaussian kernel used for declumping smoothing."""
        if filter_size == 0:
            return np.empty((0,), dtype=np.float64)
        sigma_divisor = _declumping_smoothing_sigma_divisor(
            declump_method=declump_method,
            suppress_size=suppress_size,
            min_diameter=min_diameter,
        )
        sigma = float(filter_size) / sigma_divisor
        half_width = max(int(float(filter_size) / 2.0), 1)
        offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
        kernel = (
            1.0 / np.sqrt(2.0 * np.pi) / sigma * np.exp(-0.5 * offsets**2 / sigma**2)
        )
        return np.ascontiguousarray(kernel, dtype=np.float64)

    @abstractmethod
    def convex_hull_image(self, mask: np.ndarray) -> np.ndarray:
        """Return the binary convex hull of a 2-D mask."""

    @abstractmethod
    def declumping_seed_points(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        footprint: np.ndarray,
        image_resize_factor: float,
    ) -> np.ndarray:
        """Find CellProfiler-compatible declumping seed points."""

    @abstractmethod
    def shrink_components_to_seed_points(self, mask: np.ndarray) -> np.ndarray:
        """Represent each connected component by one seed point."""

    @abstractmethod
    def relabel_sequential(self, labels: np.ndarray) -> tuple[np.ndarray, int]:
        """Compact positive labels to 1..N."""


class NumpyMorphologyBackendStrategy(MorphologyBackendStrategy):
    """Independent NumPy/SciPy/skimage morphology backend."""

    backend_key = CellProfilerBackendAuthority.backend_key(MemoryType.NUMPY)
    memory_type = MemoryType.NUMPY
    is_default_backend = False

    def connected_components(
        self, mask: np.ndarray, *, connectivity: int = 2
    ) -> tuple[np.ndarray, int]:
        return _scipy_connected_components(mask, connectivity=connectivity)

    def disk_footprint(self, radius: float) -> np.ndarray:
        return _scipy_disk_footprint(radius)

    def declumping_suppression_footprint(
        self,
        suppress_size: float,
        *,
        min_diameter: float,
        declump_method: CellProfilerDeclumpMethod,
    ) -> np.ndarray:
        radius = _declumping_suppression_radius(
            suppress_size, min_diameter=min_diameter, declump_method=declump_method
        )
        return _scipy_disk_footprint(radius)

    def grayscale_closing(self, image: np.ndarray, footprint: np.ndarray) -> np.ndarray:
        return _skimage_grayscale_closing(image, footprint)

    def grayscale_opening(self, image: np.ndarray, footprint: np.ndarray) -> np.ndarray:
        return _skimage_grayscale_opening(image, footprint)

    def erode_labeled_objects(
        self, labels: np.ndarray, footprint: np.ndarray
    ) -> np.ndarray:
        return _scipy_erode_labeled_objects(labels, footprint)

    def block_labels(
        self, image_shape: tuple[int, int], block_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        return _scipy_block_labels(image_shape, block_size)

    def blockwise_minimum(
        self, image: np.ndarray, mask: np.ndarray | None, block_size: int
    ) -> np.ndarray:
        return _scipy_blockwise_minimum(image, mask, block_size, morphology=self)

    def fix_labeled_result(self, values: np.ndarray) -> np.ndarray:
        return _scipy_fix_labeled_result(values)

    def fill_labeled_holes(
        self,
        labels: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        size_predicate: HolePredicate | None = None,
    ) -> np.ndarray:
        return self._scipy_fill_labeled_holes(
            labels,
            mask=mask,
            size_predicate=size_predicate,
        )

    def fill_labeled_holes_below_size(
        self, labels: np.ndarray, maximum_hole_size: int
    ) -> np.ndarray:
        return self._scipy_fill_labeled_holes(
            labels, size_predicate=lambda size, _is_foreground: size < maximum_hole_size
        )

    def _scipy_fill_labeled_holes(
        self,
        labels: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        size_predicate: HolePredicate | None = None,
    ) -> np.ndarray:
        from scipy import ndimage as ndi

        array = np.asarray(labels)
        foreground = array != 0
        background = ~foreground
        if mask is not None:
            background &= np.asarray(mask, dtype=bool)
        if not background.any():
            return array.copy()
        structure = ndi.generate_binary_structure(array.ndim, 1)
        background_labels, component_count = ndi.label(background, structure=structure)
        if component_count == 0:
            return array.copy()
        border_ids = _border_component_ids(background_labels)
        candidate_ids = set(range(1, component_count + 1)) - border_ids
        if size_predicate is not None:
            sizes = np.bincount(
                background_labels.ravel(), minlength=component_count + 1
            )
            candidate_ids = {
                component_id
                for component_id in candidate_ids
                if size_predicate(int(sizes[component_id]), False)
            }
        if not candidate_ids:
            return array.copy()
        fill_mask = np.isin(background_labels, tuple(sorted(candidate_ids)))
        if array.dtype == bool or np.array_equal(
            np.unique(array), np.array([False, True])
        ):
            output = foreground.copy()
            output[fill_mask] = True
            return output.astype(array.dtype, copy=False)
        _, nearest_indices = ndi.distance_transform_edt(
            background, return_distances=True, return_indices=True
        )
        output = array.copy()
        output[fill_mask] = array[
            tuple((axis_indices[fill_mask] for axis_indices in nearest_indices))
        ]
        return output

    def restore_removed_declump_basins(
        self,
        pre_declump_labels: np.ndarray,
        labels_before_size_filter: np.ndarray,
        labels_after_size_filter: np.ndarray,
    ) -> np.ndarray:
        return _restore_removed_declump_basins_numba(
            np.ascontiguousarray(pre_declump_labels, dtype=np.int64),
            np.ascontiguousarray(labels_before_size_filter, dtype=np.int64),
            np.ascontiguousarray(labels_after_size_filter, dtype=np.int64),
        ).astype(np.asarray(labels_after_size_filter).dtype, copy=False)

    def local_maxima_by_label(
        self, image: np.ndarray, labels: np.ndarray, footprint: np.ndarray
    ) -> np.ndarray:
        return _scipy_local_maxima_by_label(image, labels, footprint)

    def smooth_image_for_declumping(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        filter_size: float,
        *,
        declump_method: CellProfilerDeclumpMethod = CellProfilerDeclumpMethod.SHAPE,
        suppress_size: float | None = None,
        min_diameter: float | None = None,
    ) -> np.ndarray:
        return _scipy_smooth_image_for_declumping(
            image,
            mask,
            self.declumping_smoothing_kernel(
                filter_size,
                declump_method=declump_method,
                suppress_size=suppress_size,
                min_diameter=min_diameter,
            ),
        )

    def convex_hull_image(self, mask: np.ndarray) -> np.ndarray:
        return _skimage_convex_hull_image(mask)

    def declumping_seed_points(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        footprint: np.ndarray,
        image_resize_factor: float,
    ) -> np.ndarray:
        return _scipy_declumping_seed_points(
            image, labels, footprint, image_resize_factor, morphology=self
        )

    def shrink_components_to_seed_points(self, mask: np.ndarray) -> np.ndarray:
        return _scipy_shrink_components_to_seed_points(mask)

    def relabel_sequential(self, labels: np.ndarray) -> tuple[np.ndarray, int]:
        return _scipy_relabel_sequential(labels)


class CentrosomeNumpyMorphologyBackendStrategy(NumpyMorphologyBackendStrategy):
    """Optional centrosome provider for NumPy-memory morphology."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.CENTROSOME
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.CENTROSOME
    is_default_backend = False

    def disk_footprint(self, radius: float) -> np.ndarray:
        from centrosome.cpmorphology import strel_disk

        return strel_disk(radius)

    def block_labels(
        self, image_shape: tuple[int, int], block_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        from centrosome.cpmorphology import block

        block_size = max(1, int(block_size))
        return block(image_shape, (block_size, block_size))

    def fix_labeled_result(self, values: np.ndarray) -> np.ndarray:
        from centrosome.cpmorphology import fixup_scipy_ndimage_result

        return fixup_scipy_ndimage_result(values)

    def fill_labeled_holes(
        self,
        labels: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        size_predicate: HolePredicate | None = None,
    ) -> np.ndarray:
        from centrosome.cpmorphology import fill_labeled_holes

        if size_predicate is None:
            return fill_labeled_holes(labels, mask=mask)
        return fill_labeled_holes(labels, mask=mask, size_fn=size_predicate)

    def fill_labeled_holes_below_size(
        self, labels: np.ndarray, maximum_hole_size: int
    ) -> np.ndarray:
        return self.fill_labeled_holes(
            labels, size_predicate=lambda size, _is_foreground: size < maximum_hole_size
        )

    def local_maxima_by_label(
        self, image: np.ndarray, labels: np.ndarray, footprint: np.ndarray
    ) -> np.ndarray:
        from centrosome.cpmorphology import is_local_maximum

        return np.asarray(is_local_maximum(image, labels, footprint), dtype=bool)

    def convex_hull_image(self, mask: np.ndarray) -> np.ndarray:
        from centrosome.cpmorphology import convex_hull_image

        return np.asarray(convex_hull_image(mask), dtype=bool)

    def shrink_components_to_seed_points(self, mask: np.ndarray) -> np.ndarray:
        from centrosome.cpmorphology import binary_shrink

        return np.asarray(binary_shrink(mask), dtype=bool)

    def relabel_sequential(self, labels: np.ndarray) -> tuple[np.ndarray, int]:
        from centrosome.cpmorphology import relabel

        relabeled, count = relabel(labels)
        return (relabeled, int(count))


class NumbaNumpyMorphologyBackendStrategy(NumpyMorphologyBackendStrategy):
    """Numba-accelerated NumPy morphology backend."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        mask = np.array(
            [[False, True, False], [True, True, False], [False, False, True]],
            dtype=np.bool_,
        )
        labels = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 2]], dtype=np.int32)
        image = np.arange(9, dtype=np.float64).reshape((3, 3))
        footprint = np.ones((3, 3), dtype=np.bool_)
        self.connected_components(mask, connectivity=2)
        self.fill_labeled_holes(labels)
        self.erode_labeled_objects(labels, footprint)
        self.local_maxima_by_label(image, labels, footprint)
        self.smooth_image_for_declumping(image, mask, 1.0)
        self.smooth_image_for_declumping(
            image, np.ones(mask.shape, dtype=np.bool_), 1.0
        )

    def connected_components(
        self, mask: np.ndarray, *, connectivity: int = 2
    ) -> tuple[np.ndarray, int]:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.ndim != 2:
            return self._connected_components_planewise(
                mask_array, connectivity=connectivity
            )
        if connectivity != 2:
            return super().connected_components(mask_array, connectivity=connectivity)
        return _foreground_components_2d_numba(np.ascontiguousarray(mask_array))

    def _connected_components_planewise(
        self, mask: np.ndarray, *, connectivity: int
    ) -> tuple[np.ndarray, int]:
        if mask.ndim < 2:
            raise ValueError("Connected components requires at least two dimensions.")
        labels = np.zeros(mask.shape, dtype=np.int32)
        plane_count = int(np.prod(mask.shape[:-2], dtype=np.int64))
        source_planes = mask.reshape((plane_count, *mask.shape[-2:]))
        target_planes = labels.reshape((plane_count, *mask.shape[-2:]))
        label_offset = 0
        for plane_index in range(plane_count):
            plane_labels, plane_count_labels = self.connected_components(
                source_planes[plane_index], connectivity=connectivity
            )
            if plane_count_labels:
                target_planes[plane_index] = np.where(
                    plane_labels > 0, plane_labels + label_offset, 0
                )
                label_offset += plane_count_labels
        return (labels, label_offset)

    def convex_hull_image(self, mask: np.ndarray) -> np.ndarray:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.ndim != 2:
            raise NotImplementedError(
                "Numba morphology backend currently supports 2-D convex hulls."
            )
        return _convex_hull_image_numba(np.ascontiguousarray(mask_array))

    def grayscale_closing(self, image: np.ndarray, footprint: np.ndarray) -> np.ndarray:
        return self._grayscale_morphology(image, footprint, first_pass_is_dilation=True)

    def grayscale_opening(self, image: np.ndarray, footprint: np.ndarray) -> np.ndarray:
        return self._grayscale_morphology(image, footprint, first_pass_is_dilation=False)

    def _grayscale_morphology(
        self,
        image: np.ndarray,
        footprint: np.ndarray,
        *,
        first_pass_is_dilation: bool,
    ) -> np.ndarray:
        image_array = np.asarray(image)
        footprint_array = np.asarray(footprint, dtype=bool)
        if image_array.ndim > 2:
            footprint_2d = footprint_array.reshape(footprint_array.shape[-2:])
            return apply_over_trailing_spatial_axes(
                image_array,
                2,
                lambda plane: self._grayscale_morphology(
                    plane,
                    footprint_2d,
                    first_pass_is_dilation=first_pass_is_dilation,
                ),
            )
        if image_array.ndim != 2 or footprint_array.ndim != 2:
            raise NotImplementedError(
                "Numba morphology backend currently supports 2-D grayscale morphology."
            )
        footprint_offsets = FootprintOffsetTable.from_footprint(
            footprint_array, dimension_policy=FOOTPRINT_OFFSET_2D_POLICY
        )
        return _grayscale_morphology_2d_numba(
            np.ascontiguousarray(image_array),
            footprint_offsets.y_offsets,
            footprint_offsets.x_offsets,
            first_pass_is_dilation,
        )

    def block_labels(
        self, image_shape: tuple[int, int], block_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(image_shape) != 2:
            raise NotImplementedError(
                "Numba morphology backend currently supports 2-D block labels."
            )
        height, width = image_shape
        return _block_labels_2d_numba(int(height), int(width), max(1, int(block_size)))

    def blockwise_minimum(
        self, image: np.ndarray, mask: np.ndarray | None, block_size: int
    ) -> np.ndarray:
        image_array = np.asarray(image)
        if image_array.ndim not in (2, 3):
            return super().blockwise_minimum(image_array, mask, block_size)
        mask_array = (
            np.empty((0, 0), dtype=np.bool_)
            if mask is None
            else np.asarray(mask, dtype=np.bool_)
        )
        if mask is not None and mask_array.shape != image_array.shape[:2]:
            raise ValueError(
                f"Blockwise minimum mask must match image spatial shape; got mask {mask_array.shape!r} for image {image_array.shape!r}."
            )
        return _blockwise_minimum_numba(
            np.ascontiguousarray(image_array),
            np.ascontiguousarray(mask_array),
            mask is not None,
            max(1, int(block_size)),
        )

    def erode_labeled_objects(
        self, labels: np.ndarray, footprint: np.ndarray
    ) -> np.ndarray:
        labels_array = np.asarray(labels)
        footprint_array = np.asarray(footprint, dtype=bool)
        if labels_array.ndim not in (2, 3) or footprint_array.ndim != labels_array.ndim:
            return super().erode_labeled_objects(labels_array, footprint_array)
        offsets = FootprintOffsetTable.from_footprint(
            footprint_array, dimension_policy=FOOTPRINT_OFFSET_2D_OR_3D_POLICY
        ).offsets
        return _erode_labeled_objects_numba(
            np.ascontiguousarray(labels_array), offsets
        ).astype(labels_array.dtype, copy=False)

    def local_maxima_by_label(
        self, image: np.ndarray, labels: np.ndarray, footprint: np.ndarray
    ) -> np.ndarray:
        image_array = np.ascontiguousarray(image)
        labels_array = np.ascontiguousarray(labels)
        footprint_offsets = FootprintOffsetTable.from_footprint(
            footprint, dimension_policy=FOOTPRINT_OFFSET_2D_POLICY
        )
        offset_distances = np.sum(
            footprint_offsets.offsets * footprint_offsets.offsets, axis=1
        )
        offset_order = np.argsort(offset_distances, kind="stable")
        return _local_maxima_by_label_numba(
            image_array,
            labels_array,
            np.ascontiguousarray(footprint_offsets.y_offsets[offset_order]),
            np.ascontiguousarray(footprint_offsets.x_offsets[offset_order]),
        )

    def smooth_image_for_declumping(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        filter_size: float,
        *,
        declump_method: CellProfilerDeclumpMethod = CellProfilerDeclumpMethod.SHAPE,
        suppress_size: float | None = None,
        min_diameter: float | None = None,
    ) -> np.ndarray:
        image_array = np.asarray(image)
        mask_array = np.asarray(mask, dtype=bool)
        if image_array.ndim != 2 or mask_array.ndim != 2:
            return self._smooth_image_for_declumping_planewise(
                image_array,
                mask_array,
                filter_size,
                declump_method=declump_method,
                suppress_size=suppress_size,
                min_diameter=min_diameter,
            )
        if image_array.shape != mask_array.shape:
            raise ValueError(
                f"Declumping smoothing mask must match the image shape; got mask {mask_array.shape!r} for image {image_array.shape!r}."
            )
        kernel = self.declumping_smoothing_kernel(
            filter_size,
            declump_method=declump_method,
            suppress_size=suppress_size,
            min_diameter=min_diameter,
        )
        if kernel.size == 0:
            return image_array
        if bool(np.all(mask_array)):
            return _smooth_image_for_declumping_full_mask_numba(
                np.ascontiguousarray(image_array), kernel
            )
        return _smooth_image_for_declumping_numba(
            np.ascontiguousarray(image_array), np.ascontiguousarray(mask_array), kernel
        )

    def _smooth_image_for_declumping_planewise(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        filter_size: float,
        *,
        declump_method: CellProfilerDeclumpMethod,
        suppress_size: float | None,
        min_diameter: float | None,
    ) -> np.ndarray:
        if image.ndim < 2 or mask.ndim < 2:
            raise ValueError("Declumping smoothing requires at least two dimensions.")
        if image.shape != mask.shape:
            raise ValueError(
                f"Declumping smoothing mask must match the image shape; got mask {mask.shape!r} for image {image.shape!r}."
            )
        smoothed = np.empty_like(image)
        plane_count = int(np.prod(image.shape[:-2], dtype=np.int64))
        image_planes = image.reshape((plane_count, *image.shape[-2:]))
        mask_planes = mask.reshape((plane_count, *mask.shape[-2:]))
        target_planes = smoothed.reshape((plane_count, *image.shape[-2:]))
        for plane_index in range(plane_count):
            target_planes[plane_index] = self.smooth_image_for_declumping(
                image_planes[plane_index],
                mask_planes[plane_index],
                filter_size,
                declump_method=declump_method,
                suppress_size=suppress_size,
                min_diameter=min_diameter,
            )
        return smoothed

    def fill_labeled_holes(
        self,
        labels: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        size_predicate: HolePredicate | None = None,
    ) -> np.ndarray:
        labels_array = np.asarray(labels)
        excluded_background = None
        if mask is not None:
            mask_array = np.asarray(mask, dtype=bool)
            if mask_array.shape != labels_array.shape:
                raise ValueError(
                    f"Hole-fill mask shape must match labels shape; got {mask_array.shape!r} for {labels_array.shape!r}."
                )
            excluded_background = (labels_array == 0) & ~mask_array
            labels_array = np.where(
                excluded_background,
                1,
                labels_array,
            )
        if labels_array.ndim != 2:
            filled = self._fill_labeled_holes_planewise(
                labels_array, size_predicate=size_predicate
            )
        else:
            filled = self._fill_labeled_holes_2d(
                labels_array, size_predicate=size_predicate
            )
        if excluded_background is not None and np.any(excluded_background):
            filled = filled.copy()
            filled[excluded_background] = 0
        return filled

    def _fill_labeled_holes_planewise(
        self, labels: np.ndarray, *, size_predicate: HolePredicate | None = None
    ) -> np.ndarray:
        if labels.ndim < 2:
            raise ValueError("Hole filling requires at least two dimensions.")
        filled = np.empty_like(labels)
        plane_count = int(np.prod(labels.shape[:-2], dtype=np.int64))
        source_planes = labels.reshape((plane_count, *labels.shape[-2:]))
        target_planes = filled.reshape((plane_count, *labels.shape[-2:]))
        for plane_index in range(plane_count):
            target_planes[plane_index] = self._fill_labeled_holes_2d(
                source_planes[plane_index], size_predicate=size_predicate
            )
        return filled

    def _fill_labeled_holes_2d(
        self, labels: np.ndarray, *, size_predicate: HolePredicate | None = None
    ) -> np.ndarray:
        components, sizes, touches_border, component_count = (
            _background_components_2d_numba(np.ascontiguousarray(labels))
        )
        fill_flags = np.zeros(component_count + 1, dtype=np.bool_)
        for component_id in range(1, component_count + 1):
            if touches_border[component_id]:
                continue
            if size_predicate is None or size_predicate(
                int(sizes[component_id]), False
            ):
                fill_flags[component_id] = True
        if not np.any(fill_flags):
            return labels
        if labels.dtype == np.bool_:
            return _fill_binary_holes_from_components_numba(
                np.ascontiguousarray(labels), components, fill_flags
            )
        return _fill_labeled_holes_single_label_components_numba(
            np.ascontiguousarray(labels), components, fill_flags
        )

    def restore_removed_declump_basins(
        self,
        pre_declump_labels: np.ndarray,
        labels_before_size_filter: np.ndarray,
        labels_after_size_filter: np.ndarray,
    ) -> np.ndarray:
        pre_declump = np.asarray(pre_declump_labels)
        before = np.asarray(labels_before_size_filter)
        after = np.asarray(labels_after_size_filter)
        if pre_declump.ndim != 2 or before.ndim != 2 or after.ndim != 2:
            raise NotImplementedError(
                "Numba morphology backend currently supports 2-D declump basin restoration."
            )
        if pre_declump.shape != before.shape or before.shape != after.shape:
            raise ValueError(
                f"Declump basin restoration inputs must have identical shapes; got {pre_declump.shape!r}, {before.shape!r}, and {after.shape!r}."
            )
        return _restore_removed_declump_basins_numba(
            np.ascontiguousarray(pre_declump, dtype=np.int64),
            np.ascontiguousarray(before, dtype=np.int64),
            np.ascontiguousarray(after, dtype=np.int64),
        ).astype(after.dtype, copy=False)

    def fill_labeled_holes_below_size(
        self, labels: np.ndarray, maximum_hole_size: int
    ) -> np.ndarray:
        labels_array = np.asarray(labels)
        if labels_array.ndim != 2:
            return self._fill_labeled_holes_below_size_planewise(
                labels_array, maximum_hole_size
            )
        return self._fill_labeled_holes_below_size_2d(labels_array, maximum_hole_size)

    def _fill_labeled_holes_below_size_planewise(
        self, labels: np.ndarray, maximum_hole_size: int
    ) -> np.ndarray:
        if labels.ndim < 2:
            raise ValueError("Hole filling requires at least two dimensions.")
        filled = np.empty_like(labels)
        plane_count = int(np.prod(labels.shape[:-2], dtype=np.int64))
        source_planes = labels.reshape((plane_count, *labels.shape[-2:]))
        target_planes = filled.reshape((plane_count, *labels.shape[-2:]))
        for plane_index in range(plane_count):
            target_planes[plane_index] = self._fill_labeled_holes_below_size_2d(
                source_planes[plane_index], maximum_hole_size
            )
        return filled

    def _fill_labeled_holes_below_size_2d(
        self, labels: np.ndarray, maximum_hole_size: int
    ) -> np.ndarray:
        components, sizes, touches_border, component_count = (
            _background_components_2d_numba(np.ascontiguousarray(labels))
        )
        fill_flags = _hole_fill_flags_below_size_numba(
            sizes, touches_border, component_count, int(maximum_hole_size)
        )
        if labels.dtype == np.bool_:
            return _fill_binary_holes_from_components_numba(
                np.ascontiguousarray(labels), components, fill_flags
            )
        return _fill_labeled_holes_single_label_components_numba(
            np.ascontiguousarray(labels), components, fill_flags
        )

    def shrink_components_to_seed_points(self, mask: np.ndarray) -> np.ndarray:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.ndim != 2:
            raise NotImplementedError(
                "Numba morphology backend currently supports 2-D seed shrinking."
            )
        return _binary_shrink_2d_numba(
            np.ascontiguousarray(mask_array), _binary_shrink_table_stack()
        )

    def declumping_seed_points(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        footprint: np.ndarray,
        image_resize_factor: float,
    ) -> np.ndarray:
        image_array = np.asarray(image)
        labels_array = np.asarray(labels)
        if image_array.ndim != 2 or labels_array.ndim != 2:
            return self._declumping_seed_points_planewise(
                image_array, labels_array, footprint, image_resize_factor
            )
        if image_array.shape != labels_array.shape:
            raise ValueError(
                "image and labels must have identical shapes for declumping seed extraction"
            )
        if float(image_resize_factor) != 1.0:
            return super().declumping_seed_points(
                image_array, labels_array, footprint, image_resize_factor
            )
        maxima = self.local_maxima_by_label(image_array, labels_array, footprint)
        maxima[np.asarray(image_array) <= 0] = 0
        return self.shrink_components_to_seed_points(maxima)

    def _declumping_seed_points_planewise(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        footprint: np.ndarray,
        image_resize_factor: float,
    ) -> np.ndarray:
        if image.ndim < 2 or labels.ndim < 2:
            raise ValueError(
                "Declumping seed extraction requires at least two dimensions."
            )
        if image.shape != labels.shape:
            raise ValueError(
                "image and labels must have identical shapes for declumping seed extraction"
            )
        seeds = np.empty(labels.shape, dtype=bool)
        plane_count = int(np.prod(labels.shape[:-2], dtype=np.int64))
        image_planes = image.reshape((plane_count, *image.shape[-2:]))
        label_planes = labels.reshape((plane_count, *labels.shape[-2:]))
        seed_planes = seeds.reshape((plane_count, *labels.shape[-2:]))
        for plane_index in range(plane_count):
            seed_planes[plane_index] = self.declumping_seed_points(
                image_planes[plane_index],
                label_planes[plane_index],
                footprint,
                image_resize_factor,
            )
        return seeds

    def relabel_sequential(self, labels: np.ndarray) -> tuple[np.ndarray, int]:
        labels_array = np.asarray(labels)
        label_dtype = (
            labels_array.dtype
            if np.issubdtype(labels_array.dtype, np.integer)
            else np.dtype(np.int64)
        )
        if labels_array.ndim > 2:
            relabeled_planes, count = _relabel_sequential_3d_numba(
                np.ascontiguousarray(
                    labels_array.reshape((-1, *labels_array.shape[-2:])),
                    dtype=label_dtype,
                )
            )
            return (relabeled_planes.reshape(labels_array.shape), int(count))
        if labels_array.ndim != 2:
            raise ValueError("Relabeling requires at least two dimensions.")
        return _relabel_sequential_numba(
            np.ascontiguousarray(labels_array, dtype=label_dtype)
        )


class OpenCVNumpyMorphologyBackendStrategy(NumbaNumpyMorphologyBackendStrategy):
    """OpenCV-accelerated NumPy morphology backend."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.OPENCV
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.OPENCV
    is_default_backend = False

    def grayscale_closing(self, image: np.ndarray, footprint: np.ndarray) -> np.ndarray:
        return self._opencv_morphology(image, footprint, operation="closing")

    def grayscale_opening(self, image: np.ndarray, footprint: np.ndarray) -> np.ndarray:
        return self._opencv_morphology(image, footprint, operation="opening")

    def _opencv_morphology(
        self, image: np.ndarray, footprint: np.ndarray, *, operation: str
    ) -> np.ndarray:
        import cv2

        image_array = np.asarray(image)
        footprint_array = np.asarray(footprint, dtype=np.uint8)
        if image_array.ndim > 2:
            footprint_2d = footprint_array.reshape(footprint_array.shape[-2:])
            return apply_over_trailing_spatial_axes(
                image_array,
                2,
                lambda plane: self._opencv_morphology(
                    plane,
                    footprint_2d,
                    operation=operation,
                ),
            )
        op = cv2.MORPH_OPEN if operation == "opening" else cv2.MORPH_CLOSE
        result = cv2.morphologyEx(
            np.ascontiguousarray(image_array),
            op,
            footprint_array,
            borderType=cv2.BORDER_REFLECT,
        )
        return np.asarray(result, dtype=image_array.dtype)


def _scipy_disk_footprint(radius: float) -> np.ndarray:
    radius = max(0.0, float(radius))
    extent = int(radius)
    y, x = np.ogrid[-extent : extent + 1, -extent : extent + 1]
    return x * x + y * y <= radius * radius


def _declumping_suppression_radius(
    suppress_size: float,
    *,
    min_diameter: float,
    declump_method: CellProfilerDeclumpMethod,
) -> float:
    size = max(1.0, float(suppress_size))
    return max(1.0, size - 0.5)


def _scipy_block_labels(
    image_shape: tuple[int, int], block_size: int
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    block_size = max(1, int(block_size))
    row_blocks = max(1, int(np.floor(float(height) / float(block_size))))
    column_blocks = max(1, int(np.floor(float(width) / float(block_size))))
    labels = np.empty((height, width), dtype=np.int32)
    indexes: list[int] = []
    for row in range(row_blocks):
        y_start = int(np.ceil(float(row * height) / float(row_blocks)))
        y_stop = int(np.ceil(float((row + 1) * height) / float(row_blocks)))
        for column in range(column_blocks):
            x_start = int(np.ceil(float(column * width) / float(column_blocks)))
            x_stop = int(np.ceil(float((column + 1) * width) / float(column_blocks)))
            label = row * column_blocks + column
            labels[y_start:y_stop, x_start:x_stop] = label
            indexes.append(label)
    return (labels, np.asarray(indexes, dtype=np.int32))


def _scipy_blockwise_minimum(
    image: np.ndarray,
    mask: np.ndarray | None,
    block_size: int,
    *,
    morphology: MorphologyBackendStrategy,
) -> np.ndarray:
    from scipy.ndimage import minimum

    image_array = np.asarray(image)
    labels, indexes = morphology.block_labels(image_array.shape[:2], block_size)
    labels = labels.copy()
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != image_array.shape[:2]:
            raise ValueError(
                f"Blockwise minimum mask must match image spatial shape; got mask {mask_array.shape!r} for image {image_array.shape!r}."
            )
        labels[~mask_array] = -1
    valid = labels != -1
    result = np.zeros(image_array.shape, dtype=image_array.dtype)
    if not np.any(valid):
        return result
    if image_array.ndim == 2:
        minima = morphology.fix_labeled_result(minimum(image_array, labels, indexes))
        result[valid] = minima[labels[valid]]
        return result
    if image_array.ndim != 3:
        raise NotImplementedError(
            "Blockwise minimum currently supports 2-D images or 3-D color images."
        )
    for channel in range(image_array.shape[2]):
        minima = morphology.fix_labeled_result(
            minimum(image_array[:, :, channel], labels, indexes)
        )
        result[valid, channel] = minima[labels[valid]]
    return result


def _scipy_erode_labeled_objects(
    labels: np.ndarray, footprint: np.ndarray
) -> np.ndarray:
    import scipy.ndimage

    labels_array = np.asarray(labels)
    contours = scipy.ndimage.morphological_gradient(
        labels_array, footprint=np.asarray(footprint, dtype=bool)
    )
    return labels_array * (contours == 0)


@njit(cache=True)
def _block_labels_2d_numba(
    height: int, width: int, block_size: int
) -> tuple[np.ndarray, np.ndarray]:
    row_blocks = max(1, int(np.floor(float(height) / float(block_size))))
    column_blocks = max(1, int(np.floor(float(width) / float(block_size))))
    labels = np.empty((height, width), dtype=np.int32)
    indexes = np.empty(row_blocks * column_blocks, dtype=np.int32)
    for row in range(row_blocks):
        y_start = int(np.ceil(float(row * height) / float(row_blocks)))
        y_stop = int(np.ceil(float((row + 1) * height) / float(row_blocks)))
        for column in range(column_blocks):
            x_start = int(np.ceil(float(column * width) / float(column_blocks)))
            x_stop = int(np.ceil(float((column + 1) * width) / float(column_blocks)))
            label = row * column_blocks + column
            indexes[label] = label
            for y in range(y_start, y_stop):
                for x in range(x_start, x_stop):
                    labels[y, x] = label
    return (labels, indexes)


@njit(cache=True)
def _blockwise_minimum_numba(
    image: np.ndarray, mask: np.ndarray, has_mask: bool, block_size: int
) -> np.ndarray:
    height = image.shape[0]
    width = image.shape[1]
    row_blocks = max(1, int(np.floor(float(height) / float(block_size))))
    column_blocks = max(1, int(np.floor(float(width) / float(block_size))))
    label_count = row_blocks * column_blocks
    output = np.zeros(image.shape, dtype=image.dtype)
    if image.ndim == 2:
        minima = np.empty(label_count, dtype=image.dtype)
        has_value = np.zeros(label_count, dtype=np.bool_)
        for row in range(row_blocks):
            y_start = int(np.ceil(float(row * height) / float(row_blocks)))
            y_stop = int(np.ceil(float((row + 1) * height) / float(row_blocks)))
            for column in range(column_blocks):
                x_start = int(np.ceil(float(column * width) / float(column_blocks)))
                x_stop = int(
                    np.ceil(float((column + 1) * width) / float(column_blocks))
                )
                label = row * column_blocks + column
                for y in range(y_start, y_stop):
                    for x in range(x_start, x_stop):
                        if has_mask and (not mask[y, x]):
                            continue
                        value = image[y, x]
                        if not has_value[label] or value < minima[label]:
                            minima[label] = value
                            has_value[label] = True
                if has_value[label]:
                    value = minima[label]
                    for y in range(y_start, y_stop):
                        for x in range(x_start, x_stop):
                            if not has_mask or mask[y, x]:
                                output[y, x] = value
        return output
    channel_count = image.shape[2]
    minima = np.empty((label_count, channel_count), dtype=image.dtype)
    has_value = np.zeros(label_count, dtype=np.bool_)
    for row in range(row_blocks):
        y_start = int(np.ceil(float(row * height) / float(row_blocks)))
        y_stop = int(np.ceil(float((row + 1) * height) / float(row_blocks)))
        for column in range(column_blocks):
            x_start = int(np.ceil(float(column * width) / float(column_blocks)))
            x_stop = int(np.ceil(float((column + 1) * width) / float(column_blocks)))
            label = row * column_blocks + column
            for y in range(y_start, y_stop):
                for x in range(x_start, x_stop):
                    if has_mask and (not mask[y, x]):
                        continue
                    if not has_value[label]:
                        for channel in range(channel_count):
                            minima[label, channel] = image[y, x, channel]
                        has_value[label] = True
                    else:
                        for channel in range(channel_count):
                            value = image[y, x, channel]
                            if value < minima[label, channel]:
                                minima[label, channel] = value
            if has_value[label]:
                for y in range(y_start, y_stop):
                    for x in range(x_start, x_stop):
                        if not has_mask or mask[y, x]:
                            for channel in range(channel_count):
                                output[y, x, channel] = minima[label, channel]
    return output


@njit(cache=True)
def _erode_labeled_objects_numba(labels: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    output = np.zeros(labels.shape, dtype=labels.dtype)
    if labels.ndim == 2:
        height, width = labels.shape
        for y in range(height):
            for x in range(width):
                label = labels[y, x]
                if label == 0:
                    continue
                keep = True
                for offset_index in range(offsets.shape[0]):
                    yy = y + offsets[offset_index, 0]
                    xx = x + offsets[offset_index, 1]
                    if yy < 0 or xx < 0 or yy >= height or (xx >= width):
                        continue
                    if labels[yy, xx] != label:
                        keep = False
                        break
                if keep:
                    output[y, x] = label
        return output
    z_size, y_size, x_size = labels.shape
    for z in range(z_size):
        for y in range(y_size):
            for x in range(x_size):
                label = labels[z, y, x]
                if label == 0:
                    continue
                keep = True
                for offset_index in range(offsets.shape[0]):
                    zz = z + offsets[offset_index, 0]
                    yy = y + offsets[offset_index, 1]
                    xx = x + offsets[offset_index, 2]
                    if (
                        zz < 0
                        or yy < 0
                        or xx < 0
                        or (zz >= z_size)
                        or (yy >= y_size)
                        or (xx >= x_size)
                    ):
                        continue
                    if labels[zz, yy, xx] != label:
                        keep = False
                        break
                if keep:
                    output[z, y, x] = label
    return output


def _scipy_connected_components(
    mask: np.ndarray, *, connectivity: int = 2
) -> tuple[np.ndarray, int]:
    from scipy import ndimage as ndi

    mask_array = np.asarray(mask, dtype=bool)
    structure = ConnectedComponentConnectivity.for_connectivity(connectivity).structure(
        mask_array.ndim
    )
    labels, count = ndi.label(mask_array, structure=structure)
    return (labels.astype(np.int32, copy=False), int(count))


def _scipy_fix_labeled_result(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 0:
        return values.reshape(1)
    return values


def _scipy_local_maxima_by_label(
    image: np.ndarray, labels: np.ndarray, footprint: np.ndarray
) -> np.ndarray:
    from scipy import ndimage as ndi

    image_array = np.asarray(image)
    labels_array = np.asarray(labels)
    maxima = np.zeros(labels_array.shape, dtype=bool)
    if image_array.shape != labels_array.shape:
        raise ValueError(
            "image and labels must have identical shapes for labeled local maxima"
        )
    for label_id, bounds in _positive_label_bounding_boxes(labels_array):
        label_crop = labels_array[bounds] == label_id
        image_crop = image_array[bounds]
        masked_image = np.where(label_crop, image_crop, -np.inf)
        local_max = ndi.maximum_filter(
            masked_image,
            footprint=footprint,
            mode=SCIPY_CONSTANT_BOUNDARY_MODE,
            cval=-np.inf,
        )
        maxima[bounds] |= label_crop & (image_crop == local_max)
    return maxima


def _scipy_smooth_image_for_declumping(
    image: np.ndarray, mask: np.ndarray, kernel: np.ndarray
) -> np.ndarray:
    import scipy.ndimage

    if kernel.size == 0:
        return image

    def convolve(array: np.ndarray) -> np.ndarray:
        output = scipy.ndimage.convolve1d(
            array, kernel, axis=0, mode=SCIPY_CONSTANT_BOUNDARY_MODE
        )
        return scipy.ndimage.convolve1d(
            output, kernel, axis=1, mode=SCIPY_CONSTANT_BOUNDARY_MODE
        )

    mask_array = np.asarray(mask, dtype=bool)
    edge_array = convolve(mask_array.astype(float))
    masked_image = np.asarray(image).copy()
    masked_image[~mask_array] = 0
    smoothed_image = convolve(masked_image)
    valid = mask_array & (edge_array != 0)
    masked_image[valid] = smoothed_image[valid] / edge_array[valid]
    return masked_image


def _declumping_smoothing_sigma_divisor(
    *,
    declump_method: CellProfilerDeclumpMethod,
    suppress_size: float | None,
    min_diameter: float | None,
) -> float:
    return 2.35


def _skimage_convex_hull_image(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import convex_hull_image

    return np.asarray(convex_hull_image(np.asarray(mask, dtype=bool)), dtype=bool)


def _skimage_grayscale_closing(image: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    from skimage.morphology import closing as skimage_closing

    image_array = np.asarray(image)
    return np.asarray(
        skimage_closing(image_array, np.asarray(footprint, dtype=bool)),
        dtype=image_array.dtype,
    )


def _skimage_grayscale_opening(image: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    from skimage.morphology import opening as skimage_opening

    image_array = np.asarray(image)
    return np.asarray(
        skimage_opening(image_array, np.asarray(footprint, dtype=bool)),
        dtype=image_array.dtype,
    )


def _scipy_declumping_seed_points(
    image: np.ndarray,
    labels: np.ndarray,
    footprint: np.ndarray,
    image_resize_factor: float,
    *,
    morphology: MorphologyBackendStrategy,
) -> np.ndarray:
    from scipy import ndimage as ndi

    image_array = np.asarray(image)
    labels_array = np.asarray(labels)
    if image_array.shape != labels_array.shape:
        raise ValueError(
            "image and labels must have identical shapes for declumping seed extraction"
        )
    if image_resize_factor < 1.0:
        shape = np.maximum(
            1, np.ceil(np.asarray(image_array.shape) * float(image_resize_factor))
        ).astype(int)
        coordinates = _declumping_resize_coordinates(
            (int(shape[0]), int(shape[1])), float(image_resize_factor)
        )
        resized_image = ndi.map_coordinates(image_array, coordinates)
        resized_labels = ndi.map_coordinates(labels_array, coordinates, order=0).astype(
            labels_array.dtype, copy=False
        )
    else:
        resized_image = image_array
        resized_labels = labels_array
    maxima = morphology.local_maxima_by_label(resized_image, resized_labels, footprint)
    maxima[resized_image <= 0] = 0
    if image_resize_factor < 1.0:
        inverse_resize_factor = float(image_array.shape[0]) / float(maxima.shape[0])
        coordinates = _declumping_resize_coordinates(
            (int(image_array.shape[0]), int(image_array.shape[1])),
            inverse_resize_factor,
        )
        maxima = SparseBooleanCubicMapCoordinatesThreshold(
            maxima,
            (int(image_array.shape[0]), int(image_array.shape[1])),
            inverse_resize_factor,
        ).execute()
    return morphology.shrink_components_to_seed_points(maxima)


@lru_cache(maxsize=128)
def _declumping_resize_coordinates(
    target_shape: tuple[int, int], divisor: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return cached CellProfiler-style coordinate grids for declumping resize."""
    return tuple(
        np.mgrid[0 : target_shape[0], 0 : target_shape[1]].astype(float)
        / float(divisor)
    )


def _positive_label_bounding_boxes(labels: np.ndarray) -> LabelBoundingBoxes:
    positive_coords = np.nonzero(labels > 0)
    if not positive_coords[0].size:
        return []
    label_values = labels[positive_coords]
    order = np.argsort(label_values, kind="stable")
    sorted_labels = label_values[order]
    sorted_coords = tuple((axis_coords[order] for axis_coords in positive_coords))
    change_offsets = np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1
    group_starts = np.concatenate(([0], change_offsets))
    group_ends = np.concatenate((change_offsets, [sorted_labels.size]))
    boxes: LabelBoundingBoxes = []
    for start, end in zip(group_starts, group_ends):
        bounds = tuple(
            (
                slice(
                    int(axis_coords[start:end].min()),
                    int(axis_coords[start:end].max()) + 1,
                )
                for axis_coords in sorted_coords
            )
        )
        boxes.append((int(sorted_labels[start]), bounds))
    return boxes


def _scipy_shrink_components_to_seed_points(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    mask_array = np.asarray(mask, dtype=bool)
    components, component_count = ndi.label(
        mask_array, structure=np.ones((3,) * mask_array.ndim, dtype=bool)
    )
    seeds = np.zeros(mask_array.shape, dtype=bool)
    for component_id, component_slice in enumerate(
        ndi.find_objects(components, max_label=component_count), start=1
    ):
        if component_slice is None:
            continue
        component_crop = components[component_slice] == component_id
        coords = np.argwhere(component_crop)
        if coords.size == 0:
            continue
        centroid = coords.mean(axis=0)
        nearest = int(np.argmin(np.sum((coords - centroid) ** 2, axis=1)))
        seed_coord = tuple(
            (
                int(axis_slice.start or 0) + int(coord)
                for axis_slice, coord in zip(
                    component_slice, coords[nearest], strict=True
                )
            )
        )
        seeds[seed_coord] = True
    return seeds


@lru_cache(maxsize=1)
def _binary_shrink_table_stack() -> np.ndarray:
    binary_shrink = BinaryShrinkPatternAlgebra
    erode_table = np.array(
        [
            binary_shrink.pattern_center(index)
            and binary_shrink.component_count(index & ~16) != 1
            for index in range(512)
        ],
        dtype=np.bool_,
    )
    erode_table[binary_shrink.index_of(np.ones((3, 3), dtype=bool))] = True
    tables = (
        erode_table
        | binary_shrink.make_table(
            False,
            np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=bool),
            np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=bool),
        )
        & binary_shrink.make_table(
            False,
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool),
            np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=bool),
        ),
        erode_table
        | binary_shrink.make_table(
            False,
            np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=bool),
            np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
        )
        & binary_shrink.make_table(
            False,
            np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=bool),
            np.array([[0, 0, 1], [1, 1, 0], [1, 1, 0]], dtype=bool),
        ),
        erode_table
        | binary_shrink.make_table(
            False,
            np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]], dtype=bool),
            np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=bool),
        )
        & binary_shrink.make_table(
            False,
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
            np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=bool),
        ),
        erode_table
        | binary_shrink.make_table(
            False,
            np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
            np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
        )
        & binary_shrink.make_table(
            False,
            np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=bool),
            np.array([[0, 1, 1], [0, 1, 1], [1, 0, 0]], dtype=bool),
        ),
    )
    return np.ascontiguousarray(np.stack(tables), dtype=np.bool_)


class BinaryShrinkPatternAlgebra:
    """Owns the 3x3 bit-pattern algebra used by binary shrink lookup tables."""

    @staticmethod
    def pattern_center(index: int) -> bool:
        return bool(index & 16)

    @classmethod
    def component_count(cls, index: int) -> int:
        pattern = cls.pattern_of(index)
        visited = np.zeros((3, 3), dtype=bool)
        components = 0
        for row in range(3):
            for col in range(3):
                if not pattern[row, col] or visited[row, col]:
                    continue
                components += 1
                stack: list[tuple[int, int]] = [(row, col)]
                visited[row, col] = True
                while stack:
                    current_row, current_col = stack.pop()
                    for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        next_row = current_row + delta_row
                        next_col = current_col + delta_col
                        if (
                            next_row < 0
                            or next_row >= 3
                            or next_col < 0
                            or (next_col >= 3)
                            or visited[next_row, next_col]
                            or (not pattern[next_row, next_col])
                        ):
                            continue
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))
        return components

    @staticmethod
    def pattern_of(index: int) -> np.ndarray:
        pattern = np.zeros((3, 3), dtype=bool)
        bit = 1
        for row in range(3):
            for col in range(3):
                pattern[row, col] = bool(index & bit)
                bit <<= 1
        return pattern

    @staticmethod
    def index_of(pattern: np.ndarray) -> int:
        index = 0
        bit = 1
        for row in range(3):
            for col in range(3):
                if pattern[row, col]:
                    index += bit
                bit <<= 1
        return index

    @staticmethod
    def make_table(value: bool, pattern: np.ndarray, care: np.ndarray) -> np.ndarray:
        table = np.empty(512, dtype=np.bool_)
        for index in range(512):
            matches = True
            bit = 1
            for row in range(3):
                for col in range(3):
                    if care[row, col] and bool(index & bit) != bool(pattern[row, col]):
                        matches = False
                        break
                    bit <<= 1
                if not matches:
                    break
            table[index] = value if matches else not value
        return table


def _scipy_relabel_sequential(labels: np.ndarray) -> tuple[np.ndarray, int]:
    labels_array = np.asarray(labels)
    positive = np.unique(labels_array[labels_array > 0])
    output = np.zeros(labels_array.shape, dtype=np.int32)
    for new_label, old_label in enumerate(positive, start=1):
        output[labels_array == old_label] = new_label
    return (output, int(positive.size))


@dataclass(frozen=True, slots=True)
class FootprintOffsetDimensionPolicy:
    """Allowed footprint dimensionality and failure message for offset tables."""

    supported_dimensions: tuple[int, ...]
    message: str

    def validate(self, footprint: np.ndarray) -> None:
        if footprint.ndim not in self.supported_dimensions:
            raise NotImplementedError(self.message)


FOOTPRINT_OFFSET_2D_POLICY = FootprintOffsetDimensionPolicy(
    supported_dimensions=(2,),
    message="CellProfiler-compatible morphology currently supports 2-D footprints.",
)
FOOTPRINT_OFFSET_2D_OR_3D_POLICY = FootprintOffsetDimensionPolicy(
    supported_dimensions=(2, 3),
    message="CellProfiler-compatible morphology currently supports 2-D and 3-D footprints.",
)


@dataclass(frozen=True, slots=True)
class FootprintOffsetTable:
    """Contiguous centered offsets for Numba morphology kernels."""

    offsets: np.ndarray

    @classmethod
    def from_footprint(
        cls, footprint: np.ndarray, *, dimension_policy: FootprintOffsetDimensionPolicy
    ) -> "FootprintOffsetTable":
        footprint_array = np.asarray(footprint, dtype=bool)
        dimension_policy.validate(footprint_array)
        center = np.asarray(footprint_array.shape, dtype=np.int64) // 2
        coords = np.argwhere(footprint_array).astype(np.int64)
        return cls(np.ascontiguousarray(coords - center))

    @property
    def y_offsets(self) -> np.ndarray:
        return self.offsets[:, 0]

    @property
    def x_offsets(self) -> np.ndarray:
        return self.offsets[:, 1]


def _border_component_ids(component_labels: np.ndarray) -> set[int]:
    border_values: list[np.ndarray] = []
    for axis in range(component_labels.ndim):
        border_values.append(np.take(component_labels, 0, axis=axis).ravel())
        border_values.append(np.take(component_labels, -1, axis=axis).ravel())
    return {
        int(component_id)
        for component_id in np.concatenate(border_values)
        if component_id != 0
    }


@njit(cache=True)
def _grayscale_morphology_2d_numba(
    image: np.ndarray,
    offset_rows: np.ndarray,
    offset_cols: np.ndarray,
    first_pass_is_dilation: bool,
) -> np.ndarray:
    height, width = image.shape
    intermediate = np.empty_like(image)
    output = np.empty_like(image)
    footprint_size = offset_rows.size
    for row in range(height):
        for col in range(width):
            best = image[
                _reflect_index_1d(row + int(offset_rows[0]), height),
                _reflect_index_1d(col + int(offset_cols[0]), width),
            ]
            for offset_index in range(1, footprint_size):
                value = image[
                    _reflect_index_1d(row + int(offset_rows[offset_index]), height),
                    _reflect_index_1d(col + int(offset_cols[offset_index]), width),
                ]
                if (
                    first_pass_is_dilation
                    and value > best
                    or (not first_pass_is_dilation and value < best)
                ):
                    best = value
            intermediate[row, col] = best
    for row in range(height):
        for col in range(width):
            best = intermediate[
                _reflect_index_1d(row + int(offset_rows[0]), height),
                _reflect_index_1d(col + int(offset_cols[0]), width),
            ]
            for offset_index in range(1, footprint_size):
                value = intermediate[
                    _reflect_index_1d(row + int(offset_rows[offset_index]), height),
                    _reflect_index_1d(col + int(offset_cols[offset_index]), width),
                ]
                if (
                    first_pass_is_dilation
                    and value < best
                    or (not first_pass_is_dilation and value > best)
                ):
                    best = value
            output[row, col] = best
    return output


@njit(cache=True)
def _reflect_index_1d(index: int, size: int) -> int:
    if size <= 1:
        return 0
    reflected = index
    while reflected < 0 or reflected >= size:
        if reflected < 0:
            reflected = -reflected - 1
        else:
            reflected = 2 * size - reflected - 1
    return reflected


@njit(cache=True)
def _convex_hull_image_numba(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    output = np.zeros((height, width), dtype=np.bool_)
    point_count = 0
    for y in range(height):
        for x in range(width):
            if mask[y, x]:
                point_count += 1
    if point_count == 0:
        return output
    row_count2 = height * 2 + 1
    min_col_by_row = np.empty(row_count2, dtype=np.int64)
    max_col_by_row = np.empty(row_count2, dtype=np.int64)
    point_capacity = max(2, row_count2 * 2)
    point_y = np.empty(point_capacity, dtype=np.int64)
    point_x = np.empty(point_capacity, dtype=np.int64)
    hull_y = np.empty(point_capacity * 2, dtype=np.int64)
    hull_x = np.empty(point_capacity * 2, dtype=np.int64)
    point_count = _collect_convex_hull_diamond_extreme_points_numba(
        mask, min_col_by_row, max_col_by_row, point_y, point_x
    )
    if point_count == 0:
        return output
    hull_count = _monotone_chain_hull_numba(
        point_y, point_x, point_count, hull_y, hull_x
    )
    _paint_convex_hull_mask_numba(output, hull_y, hull_x, hull_count)
    return output


@njit(cache=True)
def _collect_convex_hull_diamond_extreme_points_numba(
    mask: np.ndarray,
    min_col_by_row: np.ndarray,
    max_col_by_row: np.ndarray,
    point_y: np.ndarray,
    point_x: np.ndarray,
) -> int:
    height, width = mask.shape
    row_count2 = height * 2 + 1
    for row_index in range(row_count2):
        min_col_by_row[row_index] = 9223372036854775807
        max_col_by_row[row_index] = -9223372036854775807
    for y in range(height):
        for x in range(width):
            if mask[y, x]:
                _add_convex_hull_diamond_vertex_numba(
                    min_col_by_row, max_col_by_row, 2 * y - 1, 2 * x
                )
                _add_convex_hull_diamond_vertex_numba(
                    min_col_by_row, max_col_by_row, 2 * y + 1, 2 * x
                )
                _add_convex_hull_diamond_vertex_numba(
                    min_col_by_row, max_col_by_row, 2 * y, 2 * x - 1
                )
                _add_convex_hull_diamond_vertex_numba(
                    min_col_by_row, max_col_by_row, 2 * y, 2 * x + 1
                )
    point_count = 0
    for row_index in range(row_count2):
        max_col = max_col_by_row[row_index]
        if max_col < -9223372036854775800:
            continue
        row2 = row_index - 1
        min_col = min_col_by_row[row_index]
        point_y[point_count] = row2
        point_x[point_count] = min_col
        point_count += 1
        if max_col != min_col:
            point_y[point_count] = row2
            point_x[point_count] = max_col
            point_count += 1
    return point_count


@njit(cache=True)
def _add_convex_hull_diamond_vertex_numba(
    min_col_by_row: np.ndarray, max_col_by_row: np.ndarray, row2: int, col2: int
) -> None:
    row_index = row2 + 1
    if col2 < min_col_by_row[row_index]:
        min_col_by_row[row_index] = col2
    if col2 > max_col_by_row[row_index]:
        max_col_by_row[row_index] = col2


@njit(cache=True)
def _cross_convex_hull_points_numba(
    ay: int, ax: int, by: int, bx: int, cy: int, cx: int
) -> int:
    return (by - ay) * (cx - ax) - (bx - ax) * (cy - ay)


@njit(cache=True)
def _monotone_chain_hull_numba(
    point_y: np.ndarray,
    point_x: np.ndarray,
    point_count: int,
    hull_y: np.ndarray,
    hull_x: np.ndarray,
) -> int:
    if point_count <= 1:
        if point_count == 1:
            hull_y[0] = point_y[0]
            hull_x[0] = point_x[0]
        return point_count
    hull_count = 0
    for index in range(point_count):
        py = point_y[index]
        px = point_x[index]
        while (
            hull_count >= 2
            and _cross_convex_hull_points_numba(
                hull_y[hull_count - 2],
                hull_x[hull_count - 2],
                hull_y[hull_count - 1],
                hull_x[hull_count - 1],
                py,
                px,
            )
            <= 0
        ):
            hull_count -= 1
        hull_y[hull_count] = py
        hull_x[hull_count] = px
        hull_count += 1
    lower_count = hull_count
    for index in range(point_count - 2, -1, -1):
        py = point_y[index]
        px = point_x[index]
        while (
            hull_count > lower_count
            and _cross_convex_hull_points_numba(
                hull_y[hull_count - 2],
                hull_x[hull_count - 2],
                hull_y[hull_count - 1],
                hull_x[hull_count - 1],
                py,
                px,
            )
            <= 0
        ):
            hull_count -= 1
        hull_y[hull_count] = py
        hull_x[hull_count] = px
        hull_count += 1
    if hull_count > 1:
        hull_count -= 1
    return hull_count


@njit(cache=True)
def _paint_convex_hull_mask_numba(
    output: np.ndarray, hull_y: np.ndarray, hull_x: np.ndarray, hull_count: int
) -> None:
    if hull_count <= 0:
        return
    if hull_count == 1:
        if hull_y[0] % 2 != 0 or hull_x[0] % 2 != 0:
            return
        y = hull_y[0] // 2
        x = hull_x[0] // 2
        if y >= 0 and y < output.shape[0] and (x >= 0) and (x < output.shape[1]):
            output[y, x] = True
        return
    min_row2 = hull_y[0]
    max_row2 = hull_y[0]
    min_col2 = hull_x[0]
    max_col2 = hull_x[0]
    for index in range(1, hull_count):
        row2 = hull_y[index]
        col2 = hull_x[index]
        if row2 < min_row2:
            min_row2 = row2
        if row2 > max_row2:
            max_row2 = row2
        if col2 < min_col2:
            min_col2 = col2
        if col2 > max_col2:
            max_col2 = col2
    if hull_count == 2:
        _paint_convex_hull_line_mask_numba(
            output,
            hull_y[0],
            hull_x[0],
            hull_y[1],
            hull_x[1],
            min_row2,
            max_row2,
            min_col2,
            max_col2,
        )
        return
    area2 = 0
    for index in range(hull_count):
        next_index = 0 if index == hull_count - 1 else index + 1
        area2 += hull_y[index] * hull_x[next_index]
        area2 -= hull_y[next_index] * hull_x[index]
    positive_orientation = area2 >= 0
    image_height, image_width = output.shape
    min_y = max(0, _ceil_div2_numba(min_row2))
    max_y = min(image_height - 1, _floor_div2_numba(max_row2))
    min_x = max(0, _ceil_div2_numba(min_col2))
    max_x = min(image_width - 1, _floor_div2_numba(max_col2))
    for y in range(min_y, max_y + 1):
        query_row2 = y * 2
        for x in range(min_x, max_x + 1):
            query_col2 = x * 2
            inside = True
            for index in range(hull_count):
                next_index = 0 if index == hull_count - 1 else index + 1
                cross = _cross_convex_hull_points_numba(
                    hull_y[index],
                    hull_x[index],
                    hull_y[next_index],
                    hull_x[next_index],
                    query_row2,
                    query_col2,
                )
                if positive_orientation:
                    if cross < 0:
                        inside = False
                        break
                elif cross > 0:
                    inside = False
                    break
            if inside:
                output[y, x] = True


@njit(cache=True)
def _ceil_div2_numba(value: int) -> int:
    if value >= 0:
        return (value + 1) // 2
    return value // 2


@njit(cache=True)
def _floor_div2_numba(value: int) -> int:
    if value >= 0:
        return value // 2
    return -((-value + 1) // 2)


@njit(cache=True)
def _paint_convex_hull_line_mask_numba(
    output: np.ndarray,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    min_row2: int,
    max_row2: int,
    min_col2: int,
    max_col2: int,
) -> None:
    dy = y1 - y0
    dx = x1 - x0
    length2 = dy * dy + dx * dx
    if length2 == 0:
        if y0 % 2 == 0 and x0 % 2 == 0:
            y = y0 // 2
            x = x0 // 2
            if y >= 0 and y < output.shape[0] and (x >= 0) and (x < output.shape[1]):
                output[y, x] = True
        return
    image_height, image_width = output.shape
    min_y = max(0, _ceil_div2_numba(min_row2))
    max_y = min(image_height - 1, _floor_div2_numba(max_row2))
    min_x = max(0, _ceil_div2_numba(min_col2))
    max_x = min(image_width - 1, _floor_div2_numba(max_col2))
    for y in range(min_y, max_y + 1):
        query_row2 = y * 2
        for x in range(min_x, max_x + 1):
            query_col2 = x * 2
            dot = (query_row2 - y0) * dy + (query_col2 - x0) * dx
            if dot < 0 or dot > length2:
                continue
            cross = dy * (query_col2 - x0) - dx * (query_row2 - y0)
            if cross == 0:
                output[y, x] = True


@njit(cache=True)
def _local_maxima_by_label_numba(
    image: np.ndarray, labels: np.ndarray, offset_y: np.ndarray, offset_x: np.ndarray
) -> np.ndarray:
    height, width = image.shape
    maxima = np.zeros((height, width), dtype=np.bool_)
    for y in range(height):
        for x in range(width):
            label = labels[y, x]
            if label <= 0:
                continue
            current = image[y, x]
            max_value = -np.inf
            for offset_index in range(offset_y.size):
                neighbor_y = y + offset_y[offset_index]
                neighbor_x = x + offset_x[offset_index]
                if (
                    neighbor_y < 0
                    or neighbor_y >= height
                    or neighbor_x < 0
                    or (neighbor_x >= width)
                ):
                    continue
                if labels[neighbor_y, neighbor_x] != label:
                    continue
                value = image[neighbor_y, neighbor_x]
                if value > current:
                    max_value = value
                    break
                if value > max_value:
                    max_value = value
            maxima[y, x] = current == max_value
    return maxima


@njit(cache=True)
def _smooth_image_for_declumping_numba(
    image: np.ndarray, mask: np.ndarray, kernel: np.ndarray
) -> np.ndarray:
    height, width = image.shape
    radius = kernel.size // 2
    edge_vertical = np.empty((height, width), dtype=np.float64)
    image_vertical = np.empty((height, width), dtype=np.float64)
    edge_array = np.empty((height, width), dtype=np.float64)
    smoothed_image = np.empty((height, width), dtype=np.float64)
    for y in range(height):
        for x in range(width):
            edge_sum = 0.0
            image_sum = 0.0
            for kernel_index in range(kernel.size):
                iy = y + kernel_index - radius
                if iy < 0 or iy >= height:
                    continue
                kernel_value = kernel[kernel_index]
                if mask[iy, x]:
                    edge_sum += kernel_value
                    image_sum += float(image[iy, x]) * kernel_value
            edge_vertical[y, x] = edge_sum
            image_vertical[y, x] = image_sum
    for y in range(height):
        for x in range(width):
            edge_sum = 0.0
            image_sum = 0.0
            for kernel_index in range(kernel.size):
                ix = x + kernel_index - radius
                if ix < 0 or ix >= width:
                    continue
                kernel_value = kernel[kernel_index]
                edge_sum += edge_vertical[y, ix] * kernel_value
                image_sum += image_vertical[y, ix] * kernel_value
            edge_array[y, x] = edge_sum
            smoothed_image[y, x] = image_sum
    output = np.empty_like(image)
    for y in range(height):
        for x in range(width):
            if mask[y, x]:
                edge_value = edge_array[y, x]
                if edge_value != 0.0:
                    output[y, x] = smoothed_image[y, x] / edge_value
                else:
                    output[y, x] = image[y, x]
            else:
                output[y, x] = 0
    return output


@njit(cache=True, fastmath=True)
def _smooth_image_for_declumping_full_mask_numba(
    image: np.ndarray, kernel: np.ndarray
) -> np.ndarray:
    height, width = image.shape
    radius = kernel.size // 2
    kernel_size = kernel.size
    edge_y = np.empty(height, dtype=np.float64)
    edge_x = np.empty(width, dtype=np.float64)
    image_vertical = np.empty((height, width), dtype=np.float64)
    for y in range(height):
        edge_sum = 0.0
        for kernel_index in range(kernel.size):
            iy = y + kernel_index - radius
            if iy >= 0 and iy < height:
                edge_sum += kernel[kernel_index]
        edge_y[y] = edge_sum
    for x in range(width):
        edge_sum = 0.0
        for kernel_index in range(kernel.size):
            ix = x + kernel_index - radius
            if ix >= 0 and ix < width:
                edge_sum += kernel[kernel_index]
        edge_x[x] = edge_sum
    full_y_start = radius
    full_y_stop = height - radius
    full_x_start = radius
    full_x_stop = width - radius
    for y in range(height):
        if y >= full_y_start and y < full_y_stop:
            y0 = y - radius
            for x in range(width):
                image_sum = 0.0
                for kernel_index in range(kernel_size):
                    image_sum += (
                        float(image[y0 + kernel_index, x]) * kernel[kernel_index]
                    )
                image_vertical[y, x] = image_sum
        else:
            for x in range(width):
                image_sum = 0.0
                for kernel_index in range(kernel_size):
                    iy = y + kernel_index - radius
                    if iy < 0 or iy >= height:
                        continue
                    image_sum += float(image[iy, x]) * kernel[kernel_index]
                image_vertical[y, x] = image_sum
    output = np.empty_like(image)
    for y in range(height):
        edge_y_value = edge_y[y]
        for x in range(full_x_start):
            image_sum = 0.0
            for kernel_index in range(kernel_size):
                ix = x + kernel_index - radius
                if ix < 0 or ix >= width:
                    continue
                image_sum += image_vertical[y, ix] * kernel[kernel_index]
            edge_value = edge_y_value * edge_x[x]
            if edge_value != 0.0:
                output[y, x] = image_sum / edge_value
            else:
                output[y, x] = image[y, x]
        for x in range(full_x_start, full_x_stop):
            x0 = x - radius
            image_sum = 0.0
            for kernel_index in range(kernel_size):
                image_sum += image_vertical[y, x0 + kernel_index] * kernel[kernel_index]
            edge_value = edge_y_value * edge_x[x]
            if edge_value != 0.0:
                output[y, x] = image_sum / edge_value
            else:
                output[y, x] = image[y, x]
        for x in range(full_x_stop, width):
            image_sum = 0.0
            for kernel_index in range(kernel_size):
                ix = x + kernel_index - radius
                if ix < 0 or ix >= width:
                    continue
                image_sum += image_vertical[y, ix] * kernel[kernel_index]
            edge_value = edge_y_value * edge_x[x]
            if edge_value != 0.0:
                output[y, x] = image_sum / edge_value
            else:
                output[y, x] = image[y, x]
    return output


@njit(cache=True)
def _foreground_components_2d_numba(mask: np.ndarray) -> tuple[np.ndarray, int]:
    height, width = mask.shape
    capacity = height * width
    labels = np.zeros((height, width), dtype=np.int32)
    queue_y = np.empty(capacity, dtype=np.int64)
    queue_x = np.empty(capacity, dtype=np.int64)
    component_count = 0
    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y, start_x] or labels[start_y, start_x] != 0:
                continue
            component_count += 1
            head = 0
            tail = 1
            queue_y[0] = start_y
            queue_x[0] = start_x
            labels[start_y, start_x] = component_count
            while head < tail:
                y = queue_y[head]
                x = queue_x[head]
                head += 1
                for dy in range(-1, 2):
                    ny = y + dy
                    if ny < 0 or ny >= height:
                        continue
                    for dx in range(-1, 2):
                        if dy == 0 and dx == 0:
                            continue
                        nx = x + dx
                        if nx < 0 or nx >= width:
                            continue
                        if not mask[ny, nx] or labels[ny, nx] != 0:
                            continue
                        labels[ny, nx] = component_count
                        queue_y[tail] = ny
                        queue_x[tail] = nx
                        tail += 1
    return (labels, component_count)


@njit(cache=True)
def _background_components_2d_numba(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    height, width = labels.shape
    capacity = height * width
    components = np.zeros((height, width), dtype=np.int32)
    sizes = np.zeros(capacity + 1, dtype=np.int64)
    touches_border = np.zeros(capacity + 1, dtype=np.bool_)
    queue_y = np.empty(capacity, dtype=np.int64)
    queue_x = np.empty(capacity, dtype=np.int64)
    component_count = 0
    for start_y in range(height):
        for start_x in range(width):
            if labels[start_y, start_x] != 0 or components[start_y, start_x] != 0:
                continue
            component_count += 1
            head = 0
            tail = 1
            queue_y[0] = start_y
            queue_x[0] = start_x
            components[start_y, start_x] = component_count
            while head < tail:
                y = queue_y[head]
                x = queue_x[head]
                head += 1
                sizes[component_count] += 1
                if y == 0 or y == height - 1 or x == 0 or (x == width - 1):
                    touches_border[component_count] = True
                if y > 0 and labels[y - 1, x] == 0 and (components[y - 1, x] == 0):
                    components[y - 1, x] = component_count
                    queue_y[tail] = y - 1
                    queue_x[tail] = x
                    tail += 1
                if (
                    y + 1 < height
                    and labels[y + 1, x] == 0
                    and (components[y + 1, x] == 0)
                ):
                    components[y + 1, x] = component_count
                    queue_y[tail] = y + 1
                    queue_x[tail] = x
                    tail += 1
                if x > 0 and labels[y, x - 1] == 0 and (components[y, x - 1] == 0):
                    components[y, x - 1] = component_count
                    queue_y[tail] = y
                    queue_x[tail] = x - 1
                    tail += 1
                if (
                    x + 1 < width
                    and labels[y, x + 1] == 0
                    and (components[y, x + 1] == 0)
                ):
                    components[y, x + 1] = component_count
                    queue_y[tail] = y
                    queue_x[tail] = x + 1
                    tail += 1
    return (components, sizes, touches_border, component_count)


@njit(cache=True)
def _hole_fill_flags_below_size_numba(
    sizes: np.ndarray,
    touches_border: np.ndarray,
    component_count: int,
    maximum_hole_size: int,
) -> np.ndarray:
    fill_flags = np.zeros(component_count + 1, dtype=np.bool_)
    for component_id in range(1, component_count + 1):
        fill_flags[component_id] = (
            not touches_border[component_id] and sizes[component_id] < maximum_hole_size
        )
    return fill_flags


@njit(cache=True)
def _fill_binary_holes_from_components_numba(
    labels: np.ndarray, components: np.ndarray, fill_flags: np.ndarray
) -> np.ndarray:
    height, width = labels.shape
    has_fillable_component = False
    for component in range(fill_flags.size):
        if fill_flags[component]:
            has_fillable_component = True
            break
    if not has_fillable_component:
        return labels
    output = labels.copy()
    for y in range(height):
        for x in range(width):
            component = components[y, x]
            if component > 0 and fill_flags[component]:
                output[y, x] = True
    return output


@njit(cache=True)
def _fill_labeled_holes_single_label_components_numba(
    labels: np.ndarray, components: np.ndarray, fill_flags: np.ndarray
) -> np.ndarray:
    height, width = labels.shape
    component_labels = np.zeros(fill_flags.size, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            component = components[y, x]
            if component <= 0 or not fill_flags[component]:
                continue
            if y > 0:
                _record_component_boundary_label_numba(
                    component_labels, component, int(labels[y - 1, x])
                )
            if y + 1 < height:
                _record_component_boundary_label_numba(
                    component_labels, component, int(labels[y + 1, x])
                )
            if x > 0:
                _record_component_boundary_label_numba(
                    component_labels, component, int(labels[y, x - 1])
                )
            if x + 1 < width:
                _record_component_boundary_label_numba(
                    component_labels, component, int(labels[y, x + 1])
                )
    has_fillable_component = False
    for component in range(component_labels.size):
        if component_labels[component] > 0:
            has_fillable_component = True
            break
    if not has_fillable_component:
        return labels
    output = labels.copy()
    for y in range(height):
        for x in range(width):
            component = components[y, x]
            if component > 0 and fill_flags[component]:
                label = component_labels[component]
                if label > 0:
                    output[y, x] = label
    return output


@njit(cache=True)
def _max_label_2d_numba(labels: np.ndarray) -> int:
    height, width = labels.shape
    max_label = 0
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label > max_label:
                max_label = label
    return max_label


@njit(cache=True)
def _restore_removed_declump_basins_numba(
    pre_declump_labels: np.ndarray,
    labels_before_size_filter: np.ndarray,
    labels_after_size_filter: np.ndarray,
) -> np.ndarray:
    height, width = labels_before_size_filter.shape
    output = labels_after_size_filter.copy()
    max_pre_declump_label = _max_label_2d_numba(pre_declump_labels)
    component_surviving_label = np.zeros(max_pre_declump_label + 1, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            pre_label = int(pre_declump_labels[y, x])
            after_label = int(labels_after_size_filter[y, x])
            if pre_label <= 0 or after_label <= 0:
                continue
            current_label = component_surviving_label[pre_label]
            if current_label == 0:
                component_surviving_label[pre_label] = after_label
            elif current_label != after_label:
                component_surviving_label[pre_label] = -1
    visited = np.zeros((height, width), dtype=np.bool_)
    stack_y = np.empty(height * width, dtype=np.int64)
    stack_x = np.empty(height * width, dtype=np.int64)
    component_y = np.empty(height * width, dtype=np.int64)
    component_x = np.empty(height * width, dtype=np.int64)
    for start_y in range(height):
        for start_x in range(width):
            if visited[start_y, start_x]:
                continue
            if (
                labels_before_size_filter[start_y, start_x] <= 0
                or labels_after_size_filter[start_y, start_x] > 0
            ):
                visited[start_y, start_x] = True
                continue
            stack_size = 1
            stack_y[0] = start_y
            stack_x[0] = start_x
            visited[start_y, start_x] = True
            component_size = 0
            boundary_label = 0
            has_multiple_boundary_labels = False
            pre_declump_label = 0
            has_multiple_pre_declump_labels = False
            while stack_size:
                stack_size -= 1
                y = stack_y[stack_size]
                x = stack_x[stack_size]
                component_y[component_size] = y
                component_x[component_size] = x
                component_size += 1
                current_pre_declump_label = int(pre_declump_labels[y, x])
                if current_pre_declump_label <= 0:
                    has_multiple_pre_declump_labels = True
                elif pre_declump_label == 0:
                    pre_declump_label = current_pre_declump_label
                elif pre_declump_label != current_pre_declump_label:
                    has_multiple_pre_declump_labels = True
                for dy in range(-1, 2):
                    yy = y + dy
                    if yy < 0 or yy >= height:
                        continue
                    for dx in range(-1, 2):
                        if dy == 0 and dx == 0:
                            continue
                        xx = x + dx
                        if xx < 0 or xx >= width:
                            continue
                        neighbor_after = labels_after_size_filter[yy, xx]
                        if neighbor_after > 0:
                            if boundary_label == 0:
                                boundary_label = neighbor_after
                            elif boundary_label != neighbor_after:
                                has_multiple_boundary_labels = True
                            continue
                        if visited[yy, xx]:
                            continue
                        if (
                            labels_before_size_filter[yy, xx] > 0
                            and labels_after_size_filter[yy, xx] == 0
                        ):
                            visited[yy, xx] = True
                            stack_y[stack_size] = yy
                            stack_x[stack_size] = xx
                            stack_size += 1
            if (
                boundary_label <= 0
                or has_multiple_boundary_labels
                or pre_declump_label <= 0
                or has_multiple_pre_declump_labels
            ):
                continue
            surviving_label = component_surviving_label[pre_declump_label]
            if surviving_label <= 0 or surviving_label != boundary_label:
                continue
            for index in range(component_size):
                output[component_y[index], component_x[index]] = boundary_label
    return output


@njit(cache=True)
def _record_component_boundary_label_numba(
    component_labels: np.ndarray, component: int, label: int
) -> None:
    if label <= 0:
        return
    current = component_labels[component]
    if current == 0:
        component_labels[component] = label
    elif current != label:
        component_labels[component] = -1


@njit(cache=True)
def _fill_labeled_holes_from_components_numba(
    labels: np.ndarray, components: np.ndarray, fill_flags: np.ndarray
) -> np.ndarray:
    height, width = labels.shape
    capacity = height * width
    output = labels.copy()
    queue_y = np.empty(capacity, dtype=np.int64)
    queue_x = np.empty(capacity, dtype=np.int64)
    head = 0
    tail = 0
    for y in range(height):
        for x in range(width):
            component = components[y, x]
            if component <= 0 or not fill_flags[component]:
                continue
            label = _first_adjacent_foreground_label_numba(labels, y, x)
            if label == 0:
                continue
            output[y, x] = label
            queue_y[tail] = y
            queue_x[tail] = x
            tail += 1
    while head < tail:
        y = queue_y[head]
        x = queue_x[head]
        head += 1
        component = components[y, x]
        label = output[y, x]
        if (
            y > 0
            and components[y - 1, x] == component
            and fill_flags[component]
            and (output[y - 1, x] == 0)
        ):
            output[y - 1, x] = label
            queue_y[tail] = y - 1
            queue_x[tail] = x
            tail += 1
        if (
            y + 1 < height
            and components[y + 1, x] == component
            and fill_flags[component]
            and (output[y + 1, x] == 0)
        ):
            output[y + 1, x] = label
            queue_y[tail] = y + 1
            queue_x[tail] = x
            tail += 1
        if (
            x > 0
            and components[y, x - 1] == component
            and fill_flags[component]
            and (output[y, x - 1] == 0)
        ):
            output[y, x - 1] = label
            queue_y[tail] = y
            queue_x[tail] = x - 1
            tail += 1
        if (
            x + 1 < width
            and components[y, x + 1] == component
            and fill_flags[component]
            and (output[y, x + 1] == 0)
        ):
            output[y, x + 1] = label
            queue_y[tail] = y
            queue_x[tail] = x + 1
            tail += 1
    return output


@njit(cache=True)
def _first_adjacent_foreground_label_numba(labels: np.ndarray, y: int, x: int):
    height, width = labels.shape
    for dy in range(-1, 2):
        neighbor_y = y + dy
        if neighbor_y < 0 or neighbor_y >= height:
            continue
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            neighbor_x = x + dx
            if neighbor_x < 0 or neighbor_x >= width:
                continue
            label = labels[neighbor_y, neighbor_x]
            if label != 0:
                return label
    return labels[y, x]


@njit(cache=True)
def _binary_shrink_2d_numba(mask: np.ndarray, tables: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    current = np.zeros((height + 2, width + 2), dtype=np.bool_)
    capacity = height * width
    coords_y = np.empty(capacity, dtype=np.int64)
    coords_x = np.empty(capacity, dtype=np.int64)
    removed_y = np.empty(capacity, dtype=np.int64)
    removed_x = np.empty(capacity, dtype=np.int64)
    count = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            padded_y = y + 1
            padded_x = x + 1
            current[padded_y, padded_x] = True
            coords_y[count] = padded_y
            coords_x[count] = padded_x
            count += 1
    iterations = count
    for _iteration in range(iterations):
        pixel_count = count
        for table_index in range(4):
            table = tables[table_index]
            new_count = 0
            removed_count = 0
            for coord_index in range(count):
                y = coords_y[coord_index]
                x = coords_x[coord_index]
                if not current[y, x]:
                    continue
                pattern_index = 0
                bit = 1
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if current[y + dy, x + dx]:
                            pattern_index += bit
                        bit <<= 1
                if table[pattern_index]:
                    coords_y[new_count] = y
                    coords_x[new_count] = x
                    new_count += 1
                else:
                    removed_y[removed_count] = y
                    removed_x[removed_count] = x
                    removed_count += 1
            for removed_index in range(removed_count):
                current[removed_y[removed_index], removed_x[removed_index]] = False
            count = new_count
        if count == pixel_count:
            break
    output = np.zeros((height, width), dtype=np.bool_)
    for coord_index in range(count):
        output[coords_y[coord_index] - 1, coords_x[coord_index] - 1] = True
    return output


@njit(cache=True)
def _seed_points_from_components_numba(
    components: np.ndarray, component_count: int
) -> np.ndarray:
    height, width = components.shape
    counts = np.zeros(component_count + 1, dtype=np.int64)
    sum_y = np.zeros(component_count + 1, dtype=np.float64)
    sum_x = np.zeros(component_count + 1, dtype=np.float64)
    for y in range(height):
        for x in range(width):
            component = components[y, x]
            if component <= 0:
                continue
            counts[component] += 1
            sum_y[component] += y
            sum_x[component] += x
    best_distance = np.empty(component_count + 1, dtype=np.float64)
    best_y = np.full(component_count + 1, -1, dtype=np.int64)
    best_x = np.full(component_count + 1, -1, dtype=np.int64)
    for component in range(component_count + 1):
        best_distance[component] = np.inf
    for y in range(height):
        for x in range(width):
            component = components[y, x]
            if component <= 0:
                continue
            centroid_y = sum_y[component] / counts[component]
            centroid_x = sum_x[component] / counts[component]
            dy = y - centroid_y
            dx = x - centroid_x
            distance = dy * dy + dx * dx
            if distance < best_distance[component]:
                best_distance[component] = distance
                best_y[component] = y
                best_x[component] = x
    seeds = np.zeros((height, width), dtype=np.bool_)
    for component in range(1, component_count + 1):
        y = best_y[component]
        x = best_x[component]
        if y >= 0 and x >= 0:
            seeds[y, x] = True
    return seeds


@njit(cache=True)
def _relabel_sequential_flat_numba(labels: np.ndarray) -> tuple[np.ndarray, int]:
    max_label = 0
    for index in range(labels.size):
        label = labels[index]
        if label > max_label:
            max_label = label
    if max_label <= 0:
        return (np.zeros(labels.shape, dtype=np.int32), 0)
    present = np.zeros(max_label + 1, dtype=np.bool_)
    for index in range(labels.size):
        label = labels[index]
        if label > 0:
            present[label] = True
    mapping = np.zeros(max_label + 1, dtype=np.int32)
    count = 0
    for label in range(1, max_label + 1):
        if present[label]:
            count += 1
            mapping[label] = count
    output = np.zeros(labels.shape, dtype=np.int32)
    for index in range(labels.size):
        label = labels[index]
        if label > 0:
            output[index] = mapping[label]
    return (output, count)


@njit(cache=True)
def _relabel_sequential_numba(labels: np.ndarray) -> tuple[np.ndarray, int]:
    height, width = labels.shape
    flat_output, count = _relabel_sequential_flat_numba(labels.reshape(height * width))
    return (flat_output.reshape((height, width)), count)


@njit(cache=True)
def _relabel_sequential_3d_numba(labels: np.ndarray) -> tuple[np.ndarray, int]:
    plane_count, height, width = labels.shape
    flat_output, count = _relabel_sequential_flat_numba(
        labels.reshape(plane_count * height * width)
    )
    return (flat_output.reshape((plane_count, height, width)), count)


class ExpandShrinkOperationStrategy(
    EnumKeyedStrategyMixin[ExpandShrinkMode], ABC, metaclass=AutoRegisterMeta
):
    """Nominal CellProfiler ExpandOrShrinkObjects operation strategy."""

    __registry_key__ = MORPHOLOGY_STRATEGY_REGISTRY_KEY
    __skip_if_no_key__ = True
    __enum_member_attr__ = "mode"
    mode: ClassVar[ExpandShrinkMode | None] = None
    strategy_label: ClassVar[str | None] = None
    cellprofiler_operations: ClassVar[
        tuple[CellProfilerExpandShrinkOperation, ...]
    ] = ()

    @classmethod
    def for_mode(cls, mode: ExpandShrinkMode | str) -> "ExpandShrinkOperationStrategy":
        resolved = coerce_cellprofiler_enum(ExpandShrinkMode, mode)
        return cls.for_enum_member(resolved)

    @classmethod
    def mode_for_cellprofiler_operation(
        cls, operation: CellProfilerExpandShrinkOperation | str
    ) -> ExpandShrinkMode:
        """Return the runtime mode declared for one CellProfiler operation."""
        resolved = coerce_cellprofiler_enum(
            CellProfilerExpandShrinkOperation, operation
        )
        matches = tuple(
            (
                strategy_type.mode
                for strategy_type in cls.registered_strategy_types()
                if resolved in strategy_type.cellprofiler_operations
            )
        )
        if len(matches) != 1 or not isinstance(matches[0], ExpandShrinkMode):
            raise ValueError(
                f"Expected exactly one ExpandOrShrinkObjects mode for operation {resolved.value!r}; found {len(matches)}."
            )
        return matches[0]

    @abstractmethod
    def apply(
        self, labels: np.ndarray, *, iterations: int, fill_holes: bool
    ) -> np.ndarray:
        """Return transformed labels for this operation mode."""

    @staticmethod
    def apply_label_planes(
        labels: np.ndarray, operation: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        output = np.empty_like(labels, dtype=np.int32)
        label_planes = labels.reshape((-1, *labels.shape[-2:]))
        output_planes = output.reshape((-1, *output.shape[-2:]))
        for plane_index in range(label_planes.shape[0]):
            output_planes[plane_index] = operation(label_planes[plane_index])
        return output


class ExpandDefinedPixelsStrategy(ExpandShrinkOperationStrategy):
    """Expand labeled objects by a fixed pixel radius."""

    mode = ExpandShrinkMode.EXPAND_DEFINED_PIXELS
    cellprofiler_operations = (
        CellProfilerExpandShrinkOperation.EXPAND_DEFINED_PIXELS,
        CellProfilerExpandShrinkOperation.EXPAND_BY_MEASUREMENT,
    )

    def apply(
        self, labels: np.ndarray, *, iterations: int, fill_holes: bool
    ) -> np.ndarray:
        return self.expand_defined_pixels(labels, iterations)

    def expand_defined_pixels(self, labels: np.ndarray, iterations: int) -> np.ndarray:
        """Expand labeled objects by a defined number of pixels."""
        from scipy.ndimage import distance_transform_edt

        if iterations <= 0:
            return labels.copy()
        labels_int = labels.astype(np.int32, copy=False)
        if labels_int.ndim > 2:
            return self.apply_label_planes(
                labels_int, lambda plane: self.expand_defined_pixels(plane, iterations)
            )
        if _labels_are_points_numba(np.ascontiguousarray(labels_int)):
            return _expand_point_labels_defined_pixels_numba(
                np.ascontiguousarray(labels_int), int(iterations)
            )
        result = labels_int.copy()
        background = labels_int == 0
        distances, indices = distance_transform_edt(background, return_indices=True)
        expand_mask = background & (distances <= iterations)
        result[expand_mask] = labels_int[
            indices[0][expand_mask], indices[1][expand_mask]
        ]
        return result


class ExpandInfiniteStrategy(ExpandShrinkOperationStrategy):
    """Expand labeled objects until all background is assigned."""

    mode = ExpandShrinkMode.EXPAND_INFINITE
    cellprofiler_operations = (CellProfilerExpandShrinkOperation.EXPAND_UNTIL_TOUCHING,)

    def apply(
        self, labels: np.ndarray, *, iterations: int, fill_holes: bool
    ) -> np.ndarray:
        return _expand_until_touching(labels)


class ShrinkDefinedPixelsStrategy(ExpandShrinkOperationStrategy):
    """Shrink labeled objects by a fixed pixel radius."""

    mode = ExpandShrinkMode.SHRINK_DEFINED_PIXELS
    cellprofiler_operations = (
        CellProfilerExpandShrinkOperation.SHRINK_DEFINED_PIXELS,
        CellProfilerExpandShrinkOperation.SHRINK_BY_MEASUREMENT,
    )

    def apply(
        self, labels: np.ndarray, *, iterations: int, fill_holes: bool
    ) -> np.ndarray:
        return _shrink_defined_pixels(labels, iterations, fill_holes)


class ShrinkToPointStrategy(ExpandShrinkOperationStrategy):
    """Shrink each object to its center point."""

    mode = ExpandShrinkMode.SHRINK_TO_POINT
    cellprofiler_operations = (CellProfilerExpandShrinkOperation.SHRINK_TO_POINT,)

    def apply(
        self, labels: np.ndarray, *, iterations: int, fill_holes: bool
    ) -> np.ndarray:
        return self.shrink_to_point(labels, fill_holes)

    def shrink_to_point(self, labels: np.ndarray, fill: bool) -> np.ndarray:
        """Shrink each labeled object to a single point at its centroid."""
        labels_int = labels.astype(np.int32, copy=False)
        if labels_int.ndim > 2:
            return self.apply_label_planes(
                labels_int, lambda plane: self.shrink_to_point(plane, fill)
            )
        if labels_int.size == 0 or int(labels_int.max()) <= 0:
            return np.zeros_like(labels_int)
        return _shrink_to_point_numba(np.ascontiguousarray(labels_int))


class AddDividingLinesStrategy(ExpandShrinkOperationStrategy):
    """Remove touching object boundary pixels."""

    mode = ExpandShrinkMode.ADD_DIVIDING_LINES
    cellprofiler_operations = (CellProfilerExpandShrinkOperation.ADD_DIVIDING_LINES,)

    def apply(
        self, labels: np.ndarray, *, iterations: int, fill_holes: bool
    ) -> np.ndarray:
        return _add_dividing_lines(labels)


class DespurStrategy(ExpandShrinkOperationStrategy):
    """Remove object spurs by repeated opening."""

    mode = ExpandShrinkMode.DESPUR
    cellprofiler_operations = (CellProfilerExpandShrinkOperation.DESPUR,)

    def apply(
        self, labels: np.ndarray, *, iterations: int, fill_holes: bool
    ) -> np.ndarray:
        return _despur(labels, iterations)


class SkeletonizeStrategy(ExpandShrinkOperationStrategy):
    """Reduce each object to a skeleton."""

    mode = ExpandShrinkMode.SKELETONIZE
    cellprofiler_operations = (CellProfilerExpandShrinkOperation.SKELETONIZE,)

    def apply(
        self, labels: np.ndarray, *, iterations: int, fill_holes: bool
    ) -> np.ndarray:
        return _skeletonize_labels(labels)


@njit(cache=True)
def _labels_are_points_numba(labels: np.ndarray) -> bool:
    height, width = labels.shape
    max_label = _max_label_2d_numba(labels)
    if max_label <= 0:
        return True
    counts = np.zeros(max_label + 1, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0:
                continue
            counts[label] += 1
            if counts[label] > 1:
                return False
    return True


@njit(cache=True)
def _expand_point_labels_defined_pixels_numba(
    labels: np.ndarray, radius: int
) -> np.ndarray:
    height, width = labels.shape
    output = labels.copy()
    radius_squared = radius * radius
    initial_distance = radius_squared + 1
    best_distance = np.full(labels.shape, initial_distance, dtype=np.int32)
    best_y = np.full(labels.shape, 2147483647, dtype=np.int32)
    best_x = np.full(labels.shape, 2147483647, dtype=np.int32)
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0:
                continue
            for dy in range(-radius, radius + 1):
                yy = y + dy
                if yy < 0 or yy >= height:
                    continue
                for dx in range(-radius, radius + 1):
                    xx = x + dx
                    if xx < 0 or xx >= width:
                        continue
                    distance = dy * dy + dx * dx
                    if distance > radius_squared:
                        continue
                    if distance < best_distance[yy, xx] or (
                        distance == best_distance[yy, xx]
                        and (
                            x < best_x[yy, xx]
                            or (x == best_x[yy, xx] and y < best_y[yy, xx])
                        )
                    ):
                        best_distance[yy, xx] = distance
                        best_y[yy, xx] = y
                        best_x[yy, xx] = x
                        output[yy, xx] = label
    return output


def _expand_until_touching(labels: np.ndarray) -> np.ndarray:
    """Expand labeled objects until they touch."""
    from scipy.ndimage import distance_transform_edt

    if labels.ndim > 2:
        return ExpandShrinkOperationStrategy.apply_label_planes(
            labels, _expand_until_touching
        )
    if labels.max() == 0:
        return labels.copy()
    mask = labels > 0
    _distances, indices = distance_transform_edt(~mask, return_indices=True)
    return labels[indices[0], indices[1]]


def _shrink_defined_pixels(
    labels: np.ndarray, iterations: int, fill: bool
) -> np.ndarray:
    """Shrink labeled objects by a defined number of pixels."""
    if iterations <= 0:
        return labels.copy()
    original = labels.astype(np.int32, copy=False)
    if original.ndim > 2:
        return ExpandShrinkOperationStrategy.apply_label_planes(
            original, lambda plane: _shrink_defined_pixels(plane, iterations, fill)
        )
    result = original.copy()
    for _ in range(iterations):
        same_neighbors = np.zeros(result.shape, dtype=bool)
        center = result[1:-1, 1:-1]
        same_neighbors[1:-1, 1:-1] = (
            (center > 0)
            & (center == result[:-2, 1:-1])
            & (center == result[2:, 1:-1])
            & (center == result[1:-1, :-2])
            & (center == result[1:-1, 2:])
        )
        result = np.where(same_neighbors, result, 0).astype(np.int32, copy=False)
    if fill:
        _restore_eroded_objects_to_centroids(original, result)
    return result


def _restore_eroded_objects_to_centroids(
    original: np.ndarray, eroded: np.ndarray
) -> None:
    """Preserve one centroid pixel for labels fully removed by shrinking."""
    region_props = LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
        original.astype(np.int32, copy=False)
    )
    if region_props.label.size == 0:
        return
    remaining_ids = set(
        (int(label_id) for label_id in np.unique(eroded) if label_id > 0)
    )
    for index, label_id in enumerate(region_props.label):
        label_int = int(label_id)
        if label_int in remaining_ids:
            continue
        cy = int(region_props.centroid_y[index])
        cx = int(region_props.centroid_x[index])
        eroded[cy, cx] = label_int


@njit(cache=True)
def _shrink_to_point_numba(labels: np.ndarray) -> np.ndarray:
    height, width = labels.shape
    max_label = _max_label_2d_numba(labels)
    y_sums = np.zeros(max_label + 1, dtype=np.float64)
    x_sums = np.zeros(max_label + 1, dtype=np.float64)
    counts = np.zeros(max_label + 1, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0:
                continue
            y_sums[label] += float(y)
            x_sums[label] += float(x)
            counts[label] += 1
    result = np.zeros(labels.shape, dtype=np.int32)
    for label in range(1, max_label + 1):
        count = counts[label]
        if count <= 0:
            continue
        cy = int(y_sums[label] / float(count))
        cx = int(x_sums[label] / float(count))
        if cy < 0:
            cy = 0
        elif cy >= height:
            cy = height - 1
        if cx < 0:
            cx = 0
        elif cx >= width:
            cx = width - 1
        result[cy, cx] = label
    return result


def _add_dividing_lines(labels: np.ndarray) -> np.ndarray:
    """Add 1-pixel dividing lines between touching objects."""
    from scipy.ndimage import maximum_filter, minimum_filter

    if labels.ndim > 2:
        return ExpandShrinkOperationStrategy.apply_label_planes(
            labels, _add_dividing_lines
        )
    if labels.max() == 0:
        return labels.copy()
    result = labels.copy()
    max_filt = maximum_filter(labels, size=3)
    min_filt = minimum_filter(labels, size=3)
    boundary = (max_filt != min_filt) & (min_filt > 0)
    result[boundary] = 0
    return result


def _despur(labels: np.ndarray, iterations: int) -> np.ndarray:
    """Remove spurs from labeled objects."""
    from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

    if iterations <= 0:
        return labels.copy()
    if labels.ndim > 2:
        return ExpandShrinkOperationStrategy.apply_label_planes(
            labels, lambda plane: _despur(plane, iterations)
        )
    result = np.zeros_like(labels)
    struct = generate_binary_structure(2, 1)
    for label_id in range(1, labels.max() + 1):
        obj_mask = labels == label_id
        opened = binary_erosion(obj_mask, structure=struct, iterations=iterations)
        opened = binary_dilation(opened, structure=struct, iterations=iterations)
        result[opened] = label_id
    return result


def _skeletonize_labels(labels: np.ndarray) -> np.ndarray:
    """Reduce labeled objects to their skeletons."""
    from skimage.morphology import skeletonize

    if labels.ndim > 2:
        return ExpandShrinkOperationStrategy.apply_label_planes(
            labels, _skeletonize_labels
        )
    result = np.zeros_like(labels)
    for label_id in range(1, labels.max() + 1):
        obj_mask = labels == label_id
        skeleton = skeletonize(obj_mask)
        result[skeleton] = label_id
    return result


def prepare_expand_or_shrink_objects() -> None:
    """Compile kernels used by common object expansion/shrink modes."""
    labels = np.zeros((16, 16), dtype=np.int32)
    labels[2:5, 3:7] = 1
    labels[8:12, 9:14] = 2
    points = ShrinkToPointStrategy().shrink_to_point(labels, False)
    ExpandDefinedPixelsStrategy().expand_defined_pixels(points, 2)


@numpy_decorator(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
def expand_or_shrink_objects(
    image: np.ndarray,
    labels: ObjectLabelValue,
    mode: ExpandShrinkMode | str = ExpandShrinkMode.EXPAND_DEFINED_PIXELS,
    iterations: int = 1,
    fill_holes: bool = True,
) -> tuple[object, MeasurementSparseColumnarRows, ObjectLabelValue]:
    """Expand or shrink labeled objects using CellProfiler-compatible semantics.

    Args:
        labels: Object-label plane whose regions are expanded, shrunk, skeletonized,
            or reduced to points according to ``mode``.
    """
    labels_int = object_label_dense_array(labels, dtype=np.int32)
    operation = ExpandShrinkOperationStrategy.for_mode(mode)
    result_labels = operation.apply(
        labels_int, iterations=iterations, fill_holes=fill_holes
    )
    return (
        image,
        MeasurementSparseColumnarRows.from_rows((), fields=()),
        object_label_value_with_dense_labels(
            labels,
            result_labels.astype(np.int32, copy=False),
        ),
    )


expand_or_shrink_objects.__openhcs_prepare__ = prepare_expand_or_shrink_objects


@dataclass(frozen=True, slots=True)
class MaskObjectsStats(ObjectRemovalStats):
    """MaskObjects count summary for one runtime plane."""

    original_object_count: int
    remaining_object_count: int


@dataclass(frozen=True, slots=True)
class MaskObjectsPlaneResult:
    """MaskObjects result for one runtime plane."""

    labels: np.ndarray
    stats: MaskObjectsStats
    relationships: DirectedObjectRelationshipPayload


class MaskObjectsOverlapHandlingStrategy(
    EnumKeyedStrategyMixin[MaskObjectsOverlapHandling], ABC, metaclass=AutoRegisterMeta
):
    """Apply one MaskObjects overlap-handling policy to a label plane."""

    __registry_key__ = MORPHOLOGY_STRATEGY_REGISTRY_KEY
    __skip_if_no_key__ = True
    __enum_member_attr__ = "overlap_handling"
    __enum_label_attr__ = MORPHOLOGY_STRATEGY_REGISTRY_KEY
    overlap_handling: ClassVar[MaskObjectsOverlapHandling | None] = None
    strategy_label: ClassVar[str | None] = None

    @classmethod
    def for_choice(
        cls, overlap_handling: MaskObjectsOverlapHandling
    ) -> "MaskObjectsOverlapHandlingStrategy":
        return cls.for_enum_member(overlap_handling)

    @abstractmethod
    def apply(
        self,
        label_image: np.ndarray,
        masked_labels: np.ndarray,
        binary_mask: np.ndarray,
        *,
        nobjects: int,
        overlap_fraction: float,
    ) -> np.ndarray:
        """Return labels after applying the overlap policy."""


class MaskObjectsMaskOverlapStrategy(MaskObjectsOverlapHandlingStrategy):
    overlap_handling = MaskObjectsOverlapHandling.MASK

    def apply(
        self,
        label_image: np.ndarray,
        masked_labels: np.ndarray,
        binary_mask: np.ndarray,
        *,
        nobjects: int,
        overlap_fraction: float,
    ) -> np.ndarray:
        del label_image, nobjects, overlap_fraction
        return masked_labels * binary_mask.astype(masked_labels.dtype)


class MaskObjectsKeepOverlapStrategy(MaskObjectsOverlapHandlingStrategy):
    overlap_handling = MaskObjectsOverlapHandling.KEEP

    def apply(
        self,
        label_image: np.ndarray,
        masked_labels: np.ndarray,
        binary_mask: np.ndarray,
        *,
        nobjects: int,
        overlap_fraction: float,
    ) -> np.ndarray:
        del overlap_fraction
        import scipy.ndimage as ndi

        object_indices = np.arange(1, nobjects + 1, dtype=np.int32)
        pixel_counts = np.atleast_1d(
            ndi.sum(binary_mask.astype(np.float64), label_image, object_indices)
        )
        keep = pixel_counts > 0
        keep_lookup = np.concatenate([[False], keep])
        masked_labels[~keep_lookup[label_image]] = 0
        return masked_labels


class MaskObjectsRemoveOverlapStrategy(MaskObjectsOverlapHandlingStrategy):
    overlap_handling = MaskObjectsOverlapHandling.REMOVE

    def apply(
        self,
        label_image: np.ndarray,
        masked_labels: np.ndarray,
        binary_mask: np.ndarray,
        *,
        nobjects: int,
        overlap_fraction: float,
    ) -> np.ndarray:
        del overlap_fraction
        import scipy.ndimage as ndi

        object_indices = np.arange(1, nobjects + 1, dtype=np.int32)
        pixel_counts = np.atleast_1d(
            ndi.sum(binary_mask.astype(np.float64), label_image, object_indices)
        )
        total_pixels = np.atleast_1d(
            ndi.sum(
                np.ones(label_image.shape, dtype=np.float64),
                label_image,
                object_indices,
            )
        )
        keep = pixel_counts == total_pixels
        keep_lookup = np.concatenate([[False], keep])
        masked_labels[~keep_lookup[label_image]] = 0
        return masked_labels


class MaskObjectsRemovePercentageOverlapStrategy(MaskObjectsOverlapHandlingStrategy):
    overlap_handling = MaskObjectsOverlapHandling.REMOVE_PERCENTAGE

    def apply(
        self,
        label_image: np.ndarray,
        masked_labels: np.ndarray,
        binary_mask: np.ndarray,
        *,
        nobjects: int,
        overlap_fraction: float,
    ) -> np.ndarray:
        import scipy.ndimage as ndi

        object_indices = np.arange(1, nobjects + 1, dtype=np.int32)
        pixel_counts = np.atleast_1d(
            ndi.sum(binary_mask.astype(np.float64), label_image, object_indices)
        )
        total_pixels = np.atleast_1d(
            ndi.sum(
                np.ones(label_image.shape, dtype=np.float64),
                label_image,
                object_indices,
            )
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            fractions = np.where(total_pixels > 0, pixel_counts / total_pixels, 0)
        keep = fractions >= overlap_fraction
        keep_lookup = np.concatenate([[False], keep])
        masked_labels[~keep_lookup[label_image]] = 0
        return masked_labels


class MaskObjectsNumberingStrategy(
    EnumKeyedStrategyMixin[MaskObjectsNumberingChoice], ABC, metaclass=AutoRegisterMeta
):
    """Apply one MaskObjects label-numbering policy."""

    __registry_key__ = MORPHOLOGY_STRATEGY_REGISTRY_KEY
    __skip_if_no_key__ = True
    __enum_member_attr__ = "numbering"
    __enum_label_attr__ = MORPHOLOGY_STRATEGY_REGISTRY_KEY
    numbering: ClassVar[MaskObjectsNumberingChoice | None] = None
    strategy_label: ClassVar[str | None] = None

    @classmethod
    def for_choice(
        cls, numbering: MaskObjectsNumberingChoice
    ) -> "MaskObjectsNumberingStrategy":
        return cls.for_enum_member(numbering)

    @abstractmethod
    def apply(
        self, masked_labels: np.ndarray, *, nobjects: int
    ) -> tuple[np.ndarray, int]:
        """Return labels and remaining object count after numbering."""


class MaskObjectsRenumberStrategy(MaskObjectsNumberingStrategy):
    numbering = MaskObjectsNumberingChoice.RENUMBER

    def apply(
        self, masked_labels: np.ndarray, *, nobjects: int
    ) -> tuple[np.ndarray, int]:
        unique_labels = np.unique(masked_labels[masked_labels != 0])
        if len(unique_labels) == 0:
            return (masked_labels, 0)
        indexer = np.zeros(nobjects + 1, dtype=np.int32)
        indexer[unique_labels] = np.arange(1, len(unique_labels) + 1, dtype=np.int32)
        return (indexer[masked_labels], len(unique_labels))


class MaskObjectsRetainNumberingStrategy(MaskObjectsNumberingStrategy):
    numbering = MaskObjectsNumberingChoice.RETAIN

    def apply(
        self, masked_labels: np.ndarray, *, nobjects: int
    ) -> tuple[np.ndarray, int]:
        del nobjects
        return (masked_labels, len(np.unique(masked_labels[masked_labels != 0])))


@dataclass(frozen=True, slots=True)
class MaskObjectsPlaneOperation:
    """CellProfiler MaskObjects semantics for one aligned object-label plane."""

    overlap_handling: MaskObjectsOverlapHandling
    overlap_fraction: float
    numbering: MaskObjectsNumberingChoice
    invert_mask: bool
    relationship_backend: ObjectRelationshipBackendStrategy

    def apply(
        self, label_image: object, mask: object, *, slice_index: int = 0
    ) -> MaskObjectsPlaneResult:
        (label_image, mask), adapters = SourceSpatialDomainAdapter.aligned_values(
            (label_image, mask)
        )
        label_adapter = adapters[0]
        label_image = np.asarray(label_image, dtype=np.int32)
        mask = np.asarray(mask)
        binary_mask = mask > 0 if mask.max() > 1 else mask.astype(bool)
        if self.invert_mask:
            binary_mask = ~binary_mask
        masked_labels = label_image.copy()
        nobjects = int(np.max(label_image))
        if nobjects == 0:
            return MaskObjectsPlaneResult(
                labels=(
                    label_adapter.extract_source_array(
                        masked_labels,
                        spatial_axes_yx=label_adapter.spatial_axes_yx,
                    )
                ),
                stats=MaskObjectsStats(
                    slice_index=slice_index,
                    original_object_count=0,
                    remaining_object_count=0,
                    objects_removed=0,
                ),
                relationships=DirectedObjectRelationshipPayload(
                    source_ids=(), target_ids=()
                ),
            )
        masked_labels = MaskObjectsOverlapHandlingStrategy.for_choice(
            self.overlap_handling
        ).apply(
            label_image,
            masked_labels,
            binary_mask,
            nobjects=nobjects,
            overlap_fraction=self.overlap_fraction,
        )
        masked_labels, remaining_count = MaskObjectsNumberingStrategy.for_choice(
            self.numbering
        ).apply(masked_labels, nobjects=nobjects)
        return MaskObjectsPlaneResult(
            labels=(
                label_adapter.extract_source_array(
                    masked_labels,
                    spatial_axes_yx=label_adapter.spatial_axes_yx,
                )
            ),
            stats=MaskObjectsStats(
                slice_index=slice_index,
                original_object_count=nobjects,
                remaining_object_count=remaining_count,
                objects_removed=nobjects - remaining_count,
            ),
            relationships=self.relationship_backend.parent_child_payload_from_labels(
                label_image, masked_labels
            ),
        )


@dataclass(frozen=True, slots=True)
class MaskObjectsOutputLabels:
    """Typed MaskObjects label output preserving input object-label semantics."""

    source: ObjectLabelValue
    labels: np.ndarray

    def value(self) -> ObjectLabelValue:
        source_domain = self.source.object_label_domain()
        return object_label_value_with_dense_labels(
            self.source,
            self.labels,
            domain_declaration=PresentObjectLabelIdsDomainDeclaration(
                scope=source_domain.scope,
                plane_projection=self.source.declared_plane_projection(),
            ),
        )


@numpy_decorator(contract=ProcessingContract.FLEXIBLE)
@object_label_input_execution_mode(ObjectLabelInputExecutionMode.MATCH_IMAGE_STACK)
@special_inputs("labels", "mask")
def mask_objects(
    image: np.ndarray,
    labels: ObjectLabelValue,
    mask: np.ndarray | ObjectLabelValue,
    overlap_handling: MaskObjectsOverlapHandling = MaskObjectsOverlapHandling.MASK,
    overlap_fraction: float = 0.5,
    numbering: MaskObjectsNumberingChoice = MaskObjectsNumberingChoice.RENUMBER,
    invert_mask: bool = False,
    relationship_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[
    np.ndarray,
    DataclassMeasurementColumnarRows,
    ObjectLabelValue,
    DirectedObjectRelationshipPayload,
]:
    """Mask object labels while preserving OpenHCS object-label domain semantics.

    Args:
        labels: Source object labels whose regions are clipped, retained, or removed
            according to the overlap policy.
        mask: Binary image or object-label mask selecting the spatial region used
            to evaluate each source object.
    """
    overlap_handling = coerce_cellprofiler_enum(
        MaskObjectsOverlapHandling, overlap_handling
    )
    numbering = coerce_cellprofiler_enum(MaskObjectsNumberingChoice, numbering)
    relationship_backend = ObjectRelationshipBackendStrategy.for_memory_type(
        backend_provider=relationship_backend_provider
    )
    operation = MaskObjectsPlaneOperation(
        overlap_handling=overlap_handling,
        overlap_fraction=overlap_fraction,
        numbering=numbering,
        invert_mask=invert_mask,
        relationship_backend=relationship_backend,
    )
    stack_slice_count = (
        labels.runtime_slice_plane_count()
        if isinstance(labels, ObjectLabelValue)
        else None
    )
    if stack_slice_count is not None:
        (label_stack, mask_stack), stack_adapters = (
            SourceSpatialDomainAdapter.aligned_values((labels, mask))
        )
        if (
            label_stack.ndim != 3
            or mask_stack.ndim != 3
            or label_stack.shape[0] != stack_slice_count
            or mask_stack.shape[0] != stack_slice_count
        ):
            raise ValueError(
                "Plane-scoped object-label mask alignment requires dense stacks "
                f"with exactly {stack_slice_count} declared planes; got "
                f"{label_stack.shape!r} and {mask_stack.shape!r}."
            )
        plane_results = tuple(
            operation.apply(
                label_stack[slice_index],
                mask_stack[slice_index],
                slice_index=slice_index,
            )
            for slice_index in range(stack_slice_count)
        )
        label_adapter = stack_adapters[0]
        masked_stack = label_adapter.extract_source_array(
            np.stack([result.labels for result in plane_results], axis=0),
            spatial_axes_yx=label_adapter.spatial_axes_yx,
        )
        if not isinstance(labels, ObjectLabelValue):
            raise TypeError(
                "Runtime-slice MaskObjects labels require an ObjectLabelValue."
            )
        source_domain = labels.object_label_domain()
        masked_payload = object_label_value_with_dense_labels(
            labels,
            masked_stack,
            domain_declaration=PresentObjectLabelIdsDomainDeclaration(
                scope=source_domain.scope,
                plane_projection=labels.declared_plane_projection(),
            ),
        )
        relationships = DirectedObjectRelationshipPayload(
            source_ids=tuple(
                parent_id
                for result in plane_results
                for parent_id in result.relationships.source_ids
            ),
            target_ids=tuple(
                child_id
                for result in plane_results
                for child_id in result.relationships.target_ids
            ),
            slice_indices=tuple(
                slice_index
                for slice_index, result in enumerate(plane_results)
                for _child_id in result.relationships.target_ids
            ),
            slice_count=stack_slice_count,
        )
        return (
            image,
            DataclassMeasurementColumnarRows(
                tuple(result.stats for result in plane_results),
                row_type=MaskObjectsStats,
            ),
            masked_payload,
            relationships,
        )
    result = operation.apply(labels, mask)
    if not isinstance(labels, ObjectLabelValue):
        raise TypeError("MaskObjects labels require an ObjectLabelValue.")
    masked_labels = MaskObjectsOutputLabels(labels, result.labels).value()
    return (
        image,
        DataclassMeasurementColumnarRows(
            (result.stats,),
            row_type=MaskObjectsStats,
        ),
        masked_labels,
        result.relationships,
    )


@dataclass(frozen=True, slots=True)
class CombineObjectsStats:
    """CellProfiler CombineObjects summary row."""

    slice_index: int
    method: str
    input_objects_x: int
    input_objects_y: int
    output_objects: int


class CombineObjectsStrategy(
    EnumKeyedStrategyMixin[CombineObjectsMethod], ABC, metaclass=AutoRegisterMeta
):
    """Nominal object-label combination strategy."""

    __registry_key__ = "method_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "method"
    method_label: ClassVar[str | None] = None
    method: ClassVar[CombineObjectsMethod | None] = None

    @classmethod
    def for_method(cls, method: CombineObjectsMethod | str) -> "CombineObjectsStrategy":
        return cls.for_enum_member(
            coerce_cellprofiler_enum(CombineObjectsMethod, method)
        )

    @abstractmethod
    def combine(self, labels_x: np.ndarray, labels_y: np.ndarray) -> np.ndarray:
        """Return combined labels for this policy."""

    def result(
        self, labels_x: np.ndarray, labels_y: np.ndarray
    ) -> tuple[CombineObjectsStats, np.ndarray]:
        combined_labels = self.combine(labels_x, labels_y)
        method = type(self).method
        if method is None:
            raise TypeError(f"{type(self).__name__} must declare method.")
        return (
            CombineObjectsStats(
                slice_index=0,
                method=method.value,
                input_objects_x=positive_dense_label_count(labels_x),
                input_objects_y=positive_dense_label_count(labels_y),
                output_objects=positive_dense_label_count(combined_labels),
            ),
            combined_labels,
        )


class MergeCombineObjectsStrategy(CombineObjectsStrategy):
    """Merge overlapping objects from two label images into single objects."""

    method = CombineObjectsMethod.MERGE

    def combine(self, labels_x: np.ndarray, labels_y: np.ndarray) -> np.ndarray:
        from scipy.ndimage import label as scipy_label

        combined_binary = ((labels_x > 0) | (labels_y > 0)).astype(np.uint8)
        merged_labels, _ = scipy_label(combined_binary)
        return merged_labels.astype(np.int32)


class PreserveCombineObjectsStrategy(CombineObjectsStrategy):
    """Preserve labels_x and add non-overlapping objects from labels_y."""

    method = CombineObjectsMethod.PRESERVE

    def combine(self, labels_x: np.ndarray, labels_y: np.ndarray) -> np.ndarray:
        result = labels_x.copy().astype(np.int32)
        max_label = labels_x.max()
        non_overlapping_mask = (labels_y > 0) & (labels_x == 0)
        if non_overlapping_mask.any():
            y_labels_in_mask = np.unique(labels_y[non_overlapping_mask])
            y_labels_in_mask = y_labels_in_mask[y_labels_in_mask > 0]
            for index, y_label in enumerate(y_labels_in_mask):
                y_object_mask = (labels_y == y_label) & non_overlapping_mask
                result[y_object_mask] = max_label + index + 1
        return result


class DiscardCombineObjectsStrategy(CombineObjectsStrategy):
    """Discard objects from labels_x that overlap labels_y."""

    method = CombineObjectsMethod.DISCARD

    def combine(self, labels_x: np.ndarray, labels_y: np.ndarray) -> np.ndarray:
        from scipy.ndimage import label as scipy_label

        overlap_mask = (labels_x > 0) & (labels_y > 0)
        overlapping_labels = np.unique(labels_x[overlap_mask])
        result = labels_x.copy().astype(np.int32)
        for label_id in overlapping_labels:
            if label_id > 0:
                result[labels_x == label_id] = 0
        if result.max() > 0:
            result, _ = scipy_label(result > 0)
        return result.astype(np.int32)


class SegmentCombineObjectsStrategy(CombineObjectsStrategy):
    """Segment labels_x using labels_y as watershed markers."""

    method = CombineObjectsMethod.SEGMENT

    def combine(self, labels_x: np.ndarray, labels_y: np.ndarray) -> np.ndarray:
        from scipy.ndimage import distance_transform_edt
        from skimage.segmentation import watershed

        binary_x = labels_x > 0
        if not binary_x.any():
            return np.zeros_like(labels_x, dtype=np.int32)
        distance = distance_transform_edt(binary_x)
        markers = labels_y.copy()
        markers[~binary_x] = 0
        if markers.max() == 0:
            return labels_x.astype(np.int32)
        return watershed(-distance, markers, mask=binary_x).astype(np.int32)


@numpy_decorator(contract=ProcessingContract.PURE_2D)
@special_inputs("object_labels")
def combineobjects(
    image: np.ndarray,
    object_labels: tuple[ObjectLabelValue, ...],
    method: CombineObjectsMethod | str = CombineObjectsMethod.MERGE,
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows, ObjectLabelValue]:
    """Combine objects from two label images using CellProfiler policies.

    Args:
        object_labels: Exactly two object-label inputs, ordered as the base labels
            and the incoming labels interpreted by ``method``.
    """
    if len(object_labels) != 2:
        raise ValueError(
            f"CombineObjects requires exactly two object-label inputs, got {len(object_labels)}."
        )
    labels_x, labels_y = (
        object_label_dense_array(value, dtype=np.int32) for value in object_labels
    )
    stats, combined_labels = CombineObjectsStrategy.for_method(method).result(
        labels_x, labels_y
    )
    return (
        image,
        DataclassMeasurementColumnarRows((stats,), row_type=CombineObjectsStats),
        object_label_value_with_dense_labels(object_labels[0], combined_labels),
    )


class SplitOrMergeOperation(Enum):
    """CellProfiler SplitOrMergeObjects top-level operation."""

    MERGE = "merge"
    SPLIT = "split"


class SplitOrMergeMergeMethod(Enum):
    """CellProfiler SplitOrMergeObjects merge selection."""

    DISTANCE = "distance"
    PER_PARENT = "per_parent"


class SplitOrMergeOutputObjectType(Enum):
    """CellProfiler SplitOrMergeObjects per-parent output mode."""

    DISCONNECTED = "disconnected"
    CONVEX_HULL = MORPH_CONVEX_HULL_OPERATION


class SplitOrMergeIntensityMethod(Enum):
    """CellProfiler SplitOrMergeObjects guide-image criterion."""

    CENTROIDS = "centroids"
    CLOSEST_POINT = "closest_point"


class SplitOrMergeInputTopology(Enum):
    """Exact public callable ABI selected by SplitOrMergeObjects settings."""

    LABELS_ONLY = "split_or_merge_objects"
    GUIDE_IMAGE = "split_or_merge_objects_with_guide_image"
    PARENT_OBJECTS = "split_or_merge_objects_per_parent"

    @classmethod
    def from_values(
        cls,
        *,
        operation: SplitOrMergeOperation | str,
        merge_method: SplitOrMergeMergeMethod | str,
        use_guide_image: bool,
    ) -> "SplitOrMergeInputTopology":
        operation_member = coerce_cellprofiler_enum(
            SplitOrMergeOperation,
            operation,
        )
        if operation_member is SplitOrMergeOperation.SPLIT:
            return cls.LABELS_ONLY
        merge_method_member = coerce_cellprofiler_enum(
            SplitOrMergeMergeMethod,
            merge_method,
        )
        if merge_method_member is SplitOrMergeMergeMethod.PER_PARENT:
            return cls.PARENT_OBJECTS
        return cls.GUIDE_IMAGE if use_guide_image else cls.LABELS_ONLY


@dataclass(frozen=True, slots=True)
class SplitOrMergeStats(ObjectCountTransitionStats):
    """CellProfiler SplitOrMergeObjects summary row."""

    operation: str


@dataclass(frozen=True, slots=True)
class SplitOrMergeRequest:
    """Complete semantic request for SplitOrMergeObjects."""

    image: np.ndarray
    labels: np.ndarray
    operation: SplitOrMergeOperation
    merge_method: SplitOrMergeMergeMethod
    output_object_type: SplitOrMergeOutputObjectType
    distance_threshold: int
    use_guide_image: bool
    minimum_intensity_fraction: float
    intensity_method: SplitOrMergeIntensityMethod
    parent_labels: np.ndarray | None
    morphology_backend_provider: BackendProviderInput

    @property
    def input_object_count(self) -> int:
        return positive_dense_label_count(self.labels)


class SplitOrMergeOperationStrategy(
    EnumKeyedStrategyMixin[SplitOrMergeOperation], ABC, metaclass=AutoRegisterMeta
):
    """Nominal implementation for one SplitOrMergeObjects operation."""

    __registry_key__ = MORPHOLOGY_STRATEGY_REGISTRY_KEY
    __skip_if_no_key__ = True
    __enum_member_attr__ = "operation"
    operation: ClassVar[SplitOrMergeOperation]
    strategy_label: ClassVar[str | None] = None

    @classmethod
    def for_operation(
        cls, operation: SplitOrMergeOperation | str
    ) -> "SplitOrMergeOperationStrategy":
        return cls.for_enum_member(
            coerce_cellprofiler_enum(SplitOrMergeOperation, operation)
        )

    @abstractmethod
    def execute(self, request: SplitOrMergeRequest) -> np.ndarray:
        """Return output labels for the operation."""


class SplitObjectsStrategy(SplitOrMergeOperationStrategy):
    operation = SplitOrMergeOperation.SPLIT

    def execute(self, request: SplitOrMergeRequest) -> np.ndarray:
        from scipy.ndimage import label as scipy_label

        output_labels, _ = scipy_label(
            request.labels > 0, structure=np.ones((3, 3), bool)
        )
        return output_labels


class MergeObjectsStrategy(SplitOrMergeOperationStrategy):
    operation = SplitOrMergeOperation.MERGE

    def execute(self, request: SplitOrMergeRequest) -> np.ndarray:
        if request.operation is not SplitOrMergeOperation.MERGE:
            raise ValueError(
                f"MergeObjectsStrategy cannot execute {request.operation!r}."
            )
        return SplitOrMergeMergeMethodStrategy.for_method(request.merge_method).merge(
            request
        )


class SplitOrMergeMergeMethodStrategy(
    EnumKeyedStrategyMixin[SplitOrMergeMergeMethod], ABC, metaclass=AutoRegisterMeta
):
    """Nominal implementation for one SplitOrMergeObjects merge method."""

    __registry_key__ = MORPHOLOGY_STRATEGY_REGISTRY_KEY
    __skip_if_no_key__ = True
    __enum_member_attr__ = "method"
    method: ClassVar[SplitOrMergeMergeMethod]
    strategy_label: ClassVar[str | None] = None

    @classmethod
    def for_method(
        cls, method: SplitOrMergeMergeMethod | str
    ) -> "SplitOrMergeMergeMethodStrategy":
        return cls.for_enum_member(
            coerce_cellprofiler_enum(SplitOrMergeMergeMethod, method)
        )

    @abstractmethod
    def merge(self, request: SplitOrMergeRequest) -> np.ndarray:
        """Return output labels for the merge method."""


class DistanceSplitOrMergeMergeMethodStrategy(SplitOrMergeMergeMethodStrategy):
    method = SplitOrMergeMergeMethod.DISTANCE

    def merge(self, request: SplitOrMergeRequest) -> np.ndarray:
        from scipy.ndimage import distance_transform_edt, label as scipy_label

        mask = request.labels > 0
        if request.distance_threshold > 0:
            distance = distance_transform_edt(~mask)
            mask = distance < request.distance_threshold / 2.0 + 1
        output_labels, _ = scipy_label(mask, structure=np.ones((3, 3), bool))
        output_labels[request.labels == 0] = 0
        if request.use_guide_image:
            output_labels = SplitOrMergeGuideImageFilter().filter(
                request.labels,
                output_labels,
                request.image,
                request.minimum_intensity_fraction,
                request.intensity_method,
            )
        return DenseObjectLabelConsecutiveRelabelingStrategy.for_labels(
            output_labels
        ).relabel(output_labels)


class ParentSplitOrMergeMergeMethodStrategy(SplitOrMergeMergeMethodStrategy):
    method = SplitOrMergeMergeMethod.PER_PARENT

    def merge(self, request: SplitOrMergeRequest) -> np.ndarray:
        if request.parent_labels is None:
            raise ValueError(
                "parent_labels are required when merge_method is PER_PARENT"
            )
        from skimage.measure import regionprops

        output_labels = np.zeros_like(request.labels)
        for prop in regionprops(request.labels):
            child_mask = request.labels == prop.label
            parent_values = request.parent_labels[child_mask]
            parent_values = parent_values[parent_values > 0]
            if len(parent_values) > 0:
                output_labels[child_mask] = np.bincount(parent_values).argmax()
            else:
                output_labels[child_mask] = prop.label
        if request.output_object_type == SplitOrMergeOutputObjectType.CONVEX_HULL:
            output_labels = SplitOrMergeConvexHull().labels(
                output_labels,
                MorphologyBackendStrategy.for_callable(
                    split_or_merge_objects,
                    backend_provider=request.morphology_backend_provider,
                ),
            )
        return DenseObjectLabelConsecutiveRelabelingStrategy.for_labels(
            output_labels
        ).relabel(output_labels)


class SplitOrMergeGuideImageFilter:
    """Guide-image filtering policy for distance-based object merging."""

    def filter(
        self,
        original_labels: np.ndarray,
        merged_labels: np.ndarray,
        image: np.ndarray,
        minimum_intensity_fraction: float,
        intensity_method: SplitOrMergeIntensityMethod,
    ) -> np.ndarray:
        if intensity_method is not SplitOrMergeIntensityMethod.CLOSEST_POINT:
            return merged_labels.copy()
        from scipy.ndimage import distance_transform_edt, label as scipy_label

        _, indices = distance_transform_edt(original_labels == 0, return_indices=True)
        closest_i, closest_j = indices
        object_intensity = image[closest_i, closest_j] * minimum_intensity_fraction
        valid_mask = (original_labels > 0) | (image >= object_intensity)
        output_labels, _ = scipy_label(
            valid_mask & (merged_labels > 0), structure=np.ones((3, 3), bool)
        )
        output_labels[original_labels == 0] = 0
        return output_labels


class SplitOrMergeConvexHull:
    """Convex-hull fill policy for per-parent merged labels."""

    def labels(
        self, labels: np.ndarray, morphology: MorphologyBackendStrategy
    ) -> np.ndarray:
        output = np.zeros_like(labels)
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]
        for label_id in unique_labels:
            mask = labels == label_id
            coords = np.argwhere(mask)
            if len(coords) < 3:
                output[mask] = label_id
                continue
            min_row = int(coords[:, 0].min())
            max_row = int(coords[:, 0].max()) + 1
            min_col = int(coords[:, 1].min())
            max_col = int(coords[:, 1].max()) + 1
            hull = morphology.convex_hull_image(mask[min_row:max_row, min_col:max_col])
            output[min_row:max_row, min_col:max_col][hull] = label_id
        return output


def _execute_split_or_merge_objects(
    image: np.ndarray,
    labels: ObjectLabelValue,
    *,
    operation: SplitOrMergeOperation,
    merge_method: SplitOrMergeMergeMethod,
    output_object_type: SplitOrMergeOutputObjectType,
    distance_threshold: int,
    use_guide_image: bool,
    minimum_intensity_fraction: float,
    intensity_method: SplitOrMergeIntensityMethod,
    parent_labels: ObjectLabelValue | None,
    morphology_backend_provider: BackendProviderInput,
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows, ObjectLabelValue]:
    """Execute one exact SplitOrMergeObjects public ABI."""
    labels_array = object_label_dense_array(labels, dtype=np.int32)
    parent_array = (
        None
        if parent_labels is None
        else object_label_dense_array(parent_labels, dtype=np.int32)
    )
    request = SplitOrMergeRequest(
        image=image,
        labels=labels_array,
        operation=coerce_cellprofiler_enum(SplitOrMergeOperation, operation),
        merge_method=coerce_cellprofiler_enum(SplitOrMergeMergeMethod, merge_method),
        output_object_type=coerce_cellprofiler_enum(
            SplitOrMergeOutputObjectType, output_object_type
        ),
        distance_threshold=distance_threshold,
        use_guide_image=use_guide_image,
        minimum_intensity_fraction=minimum_intensity_fraction,
        intensity_method=coerce_cellprofiler_enum(
            SplitOrMergeIntensityMethod, intensity_method
        ),
        parent_labels=parent_array,
        morphology_backend_provider=morphology_backend_provider,
    )
    output_labels = SplitOrMergeOperationStrategy.for_operation(
        request.operation
    ).execute(request)
    stats = SplitOrMergeStats(
        slice_index=0,
        input_object_count=int(request.input_object_count),
        output_object_count=int(positive_dense_label_count(output_labels)),
        operation=request.operation.value,
    )
    return (
        image,
        DataclassMeasurementColumnarRows((stats,), row_type=SplitOrMergeStats),
        object_label_value_with_dense_labels(
            labels,
            output_labels.astype(np.int32, copy=False),
            domain_declaration=PresentObjectLabelIdsDomainDeclaration(),
        ),
    )


@numpy_decorator(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
def split_or_merge_objects(
    image: np.ndarray,
    labels: ObjectLabelValue,
    operation: SplitOrMergeOperation = SplitOrMergeOperation.MERGE,
    merge_method: SplitOrMergeMergeMethod = SplitOrMergeMergeMethod.DISTANCE,
    output_object_type: SplitOrMergeOutputObjectType = SplitOrMergeOutputObjectType.DISCONNECTED,
    distance_threshold: int = 0,
    use_guide_image: bool = False,
    minimum_intensity_fraction: float = 0.9,
    intensity_method: SplitOrMergeIntensityMethod = SplitOrMergeIntensityMethod.CENTROIDS,
    morphology_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows, ObjectLabelValue]:
    """Split objects or merge them by distance without a guide image.

    Args:
        labels: Object-label plane to split into connected regions or merge by the
            configured distance policy.
    """
    return _execute_split_or_merge_objects(
        image,
        labels,
        operation=operation,
        merge_method=merge_method,
        output_object_type=output_object_type,
        distance_threshold=distance_threshold,
        use_guide_image=use_guide_image,
        minimum_intensity_fraction=minimum_intensity_fraction,
        intensity_method=intensity_method,
        parent_labels=None,
        morphology_backend_provider=morphology_backend_provider,
    )


@numpy_decorator(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
def split_or_merge_objects_with_guide_image(
    image: np.ndarray,
    labels: ObjectLabelValue,
    operation: SplitOrMergeOperation = SplitOrMergeOperation.MERGE,
    merge_method: SplitOrMergeMergeMethod = SplitOrMergeMergeMethod.DISTANCE,
    output_object_type: SplitOrMergeOutputObjectType = SplitOrMergeOutputObjectType.DISCONNECTED,
    distance_threshold: int = 0,
    use_guide_image: bool = True,
    minimum_intensity_fraction: float = 0.9,
    intensity_method: SplitOrMergeIntensityMethod = SplitOrMergeIntensityMethod.CENTROIDS,
    morphology_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows, ObjectLabelValue]:
    """Merge objects by distance using the declared guide image.

    Args:
        labels: Object-label plane whose distance-based merges are filtered by the
            intensities in the primary guide image.
    """
    return _execute_split_or_merge_objects(
        image,
        labels,
        operation=operation,
        merge_method=merge_method,
        output_object_type=output_object_type,
        distance_threshold=distance_threshold,
        use_guide_image=use_guide_image,
        minimum_intensity_fraction=minimum_intensity_fraction,
        intensity_method=intensity_method,
        parent_labels=None,
        morphology_backend_provider=morphology_backend_provider,
    )


@numpy_decorator(contract=ProcessingContract.PURE_2D)
@special_inputs("labels", "parent_labels")
def split_or_merge_objects_per_parent(
    image: np.ndarray,
    labels: ObjectLabelValue,
    parent_labels: ObjectLabelValue,
    operation: SplitOrMergeOperation = SplitOrMergeOperation.MERGE,
    merge_method: SplitOrMergeMergeMethod = SplitOrMergeMergeMethod.PER_PARENT,
    output_object_type: SplitOrMergeOutputObjectType = SplitOrMergeOutputObjectType.DISCONNECTED,
    distance_threshold: int = 0,
    use_guide_image: bool = False,
    minimum_intensity_fraction: float = 0.9,
    intensity_method: SplitOrMergeIntensityMethod = SplitOrMergeIntensityMethod.CENTROIDS,
    morphology_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows, ObjectLabelValue]:
    """Merge child objects through the declared parent-object input.

    Args:
        labels: Child object-label plane whose regions are grouped by overlapping
            parent identity.
        parent_labels: Parent object-label plane assigning each child to the parent
            used for the per-parent merge.
    """
    return _execute_split_or_merge_objects(
        image,
        labels,
        operation=operation,
        merge_method=merge_method,
        output_object_type=output_object_type,
        distance_threshold=distance_threshold,
        use_guide_image=use_guide_image,
        minimum_intensity_fraction=minimum_intensity_fraction,
        intensity_method=intensity_method,
        parent_labels=parent_labels,
        morphology_backend_provider=morphology_backend_provider,
    )


def positive_dense_label_count(labels: np.ndarray) -> int:
    """Return the count of positive labels present in a dense label image."""
    return int(len(np.unique(labels)) - (1 if 0 in labels else 0))


def dense_label_area_statistics(labels: np.ndarray) -> tuple[float, float, float]:
    """Return mean, median, and total positive-label area."""
    areas = np.bincount(np.asarray(labels).ravel())[1:]
    positive_areas = areas[areas > 0]
    if positive_areas.size == 0:
        return (0.0, 0.0, 0.0)
    return (
        float(np.mean(positive_areas)),
        float(np.median(positive_areas)),
        float(np.sum(positive_areas)),
    )


def filter_labels_below_minimum_diameter(
    labels: np.ndarray, min_diameter: float
) -> np.ndarray:
    min_area = np.pi * float(min_diameter) ** 2 / 4.0
    labels_array = np.ascontiguousarray(labels)
    areas = np.bincount(np.asarray(labels_array).ravel())
    return filter_labels_by_area_numba(
        labels_array, np.ascontiguousarray(areas), float(min_area), np.inf
    )


def filter_labels_above_maximum_diameter(
    labels: np.ndarray, max_diameter: float
) -> np.ndarray:
    max_area = np.pi * float(max_diameter) ** 2 / 4.0
    labels_array = np.ascontiguousarray(labels)
    areas = np.bincount(np.asarray(labels_array).ravel())
    return filter_labels_by_area_numba(
        labels_array, np.ascontiguousarray(areas), 0.0, float(max_area)
    )


def filter_labels_by_diameter_range(
    labels: np.ndarray, min_diameter: float, max_diameter: float
) -> tuple[np.ndarray, np.ndarray]:
    min_area = np.pi * float(min_diameter) ** 2 / 4.0
    max_area = np.pi * float(max_diameter) ** 2 / 4.0
    labels_array = np.ascontiguousarray(labels)
    areas = np.ascontiguousarray(np.bincount(np.asarray(labels_array).ravel()))
    return filter_labels_by_diameter_range_numba(
        labels_array, areas, float(min_area), float(max_area)
    )


def filter_labels_by_area_numba(
    labels: np.ndarray, areas: np.ndarray, min_area: float, max_area: float
) -> np.ndarray:
    if labels.ndim == 2:
        return _filter_labels_by_area_2d_numba(labels, areas, min_area, max_area)
    if labels.ndim == 3:
        return _filter_labels_by_area_3d_numba(labels, areas, min_area, max_area)
    raise ValueError(
        f"IdentifyPrimaryObjects area filtering expects 2-D planes or stacked planes, got shape {labels.shape!r}."
    )


@njit(cache=True)
def _filter_labels_by_area_2d_numba(
    labels: np.ndarray, areas: np.ndarray, min_area: float, max_area: float
) -> np.ndarray:
    output = labels.copy()
    height, width = labels.shape
    for row in range(height):
        for col in range(width):
            label = int(labels[row, col])
            if label <= 0:
                continue
            area = float(areas[label])
            if area < min_area or area > max_area:
                output[row, col] = 0
    return output


@njit(cache=True)
def _filter_labels_by_area_3d_numba(
    labels: np.ndarray, areas: np.ndarray, min_area: float, max_area: float
) -> np.ndarray:
    output = labels.copy()
    plane_count, height, width = labels.shape
    for plane_index in range(plane_count):
        for row in range(height):
            for col in range(width):
                label = int(labels[plane_index, row, col])
                if label <= 0:
                    continue
                area = float(areas[label])
                if area < min_area or area > max_area:
                    output[plane_index, row, col] = 0
    return output


def filter_labels_by_diameter_range_numba(
    labels: np.ndarray, areas: np.ndarray, min_area: float, max_area: float
) -> tuple[np.ndarray, np.ndarray]:
    if labels.ndim == 2:
        return _filter_labels_by_diameter_range_2d_numba(
            labels, areas, min_area, max_area
        )
    if labels.ndim == 3:
        return _filter_labels_by_diameter_range_3d_numba(
            labels, areas, min_area, max_area
        )
    raise ValueError(
        f"IdentifyPrimaryObjects size filtering expects 2-D planes or stacked planes, got shape {labels.shape!r}."
    )


@njit(cache=True)
def _filter_labels_by_diameter_range_2d_numba(
    labels: np.ndarray, areas: np.ndarray, min_area: float, max_area: float
) -> tuple[np.ndarray, np.ndarray]:
    small_removed = labels.copy()
    final = labels.copy()
    height, width = labels.shape
    for row in range(height):
        for col in range(width):
            label = int(labels[row, col])
            if label <= 0:
                continue
            area = float(areas[label])
            if area < min_area:
                small_removed[row, col] = 0
                final[row, col] = 0
            elif area > max_area:
                final[row, col] = 0
    return (small_removed, final)


@njit(cache=True)
def _filter_labels_by_diameter_range_3d_numba(
    labels: np.ndarray, areas: np.ndarray, min_area: float, max_area: float
) -> tuple[np.ndarray, np.ndarray]:
    small_removed = labels.copy()
    final = labels.copy()
    plane_count, height, width = labels.shape
    for plane_index in range(plane_count):
        for row in range(height):
            for col in range(width):
                label = int(labels[plane_index, row, col])
                if label <= 0:
                    continue
                area = float(areas[label])
                if area < min_area:
                    small_removed[plane_index, row, col] = 0
                    final[plane_index, row, col] = 0
                elif area > max_area:
                    final[plane_index, row, col] = 0
    return (small_removed, final)


def filter_border_objects(
    labeled_image: np.ndarray,
    *,
    image_mask: np.ndarray | None,
    image_metadata: ImagePayloadMetadata = ImagePayloadMetadata(),
) -> np.ndarray:
    """Remove labels touching the physical border or masked image border."""
    labeled_array = np.asarray(labeled_image)
    if labeled_array.ndim != 2:
        raise ValueError(
            "IdentifyPrimaryObjects border filtering requires one 2D label plane; "
            "the PURE_2D processing contract owns plane projection."
        )
    height, width = labeled_array.shape[:2]
    physical_edges = image_metadata.physical_border_edges_for_shape((height, width))
    output, removed_physical = filter_physical_border_objects_numba(
        np.ascontiguousarray(labeled_array),
        bool(physical_edges[0]),
        bool(physical_edges[1]),
        bool(physical_edges[2]),
        bool(physical_edges[3]),
    )
    if removed_physical:
        return output
    if image_mask is None or image_metadata.mask_defines_border is False:
        return output
    from scipy import ndimage as ndi

    max_label = int(output.max())
    if max_label <= 0:
        return output
    mask = np.asarray(image_mask, dtype=bool)
    if mask.shape != labeled_array.shape:
        raise ValueError(
            "IdentifyPrimaryObjects mask and label plane shapes must match exactly; "
            f"got {mask.shape!r} and {labeled_array.shape!r}."
        )
    mask_border = np.logical_not(ndi.binary_erosion(mask, border_value=1)) & mask
    masked_border_labels = output[mask_border].astype(np.int64, copy=False)
    masked_border_histogram = np.bincount(masked_border_labels, minlength=max_label + 1)
    labels_to_remove = np.flatnonzero(masked_border_histogram[1:] > 0) + 1
    if labels_to_remove.size:
        output[np.isin(output, labels_to_remove)] = 0
    return output


@njit(cache=True)
def filter_physical_border_objects_numba(
    labels: np.ndarray, top: bool, bottom: bool, left: bool, right: bool
) -> tuple[np.ndarray, bool]:
    height, width = labels.shape
    max_label = _max_label_2d_numba(labels)
    if max_label <= 0:
        return (labels, False)
    remove = np.zeros(max_label + 1, dtype=np.bool_)
    if top and height > 0:
        for x in range(width):
            label = int(labels[0, x])
            if label > 0:
                remove[label] = True
    if bottom and height > 0:
        for x in range(width):
            label = int(labels[height - 1, x])
            if label > 0:
                remove[label] = True
    if left and width > 0:
        for y in range(height):
            label = int(labels[y, 0])
            if label > 0:
                remove[label] = True
    if right and width > 0:
        for y in range(height):
            label = int(labels[y, width - 1])
            if label > 0:
                remove[label] = True
    any_removed = False
    for label in range(1, max_label + 1):
        if remove[label]:
            any_removed = True
            break
    if not any_removed:
        return (labels, False)
    output = labels.copy()
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label > 0 and remove[label]:
                output[y, x] = 0
    return (output, True)


def profile_function_runtime_enabled() -> bool:
    return os.environ.get(PROFILE_RUNTIME_ENV, "").lower() in {"1", "true", "yes"}


def log_function_runtime_profile(label: str, seconds: float, **fields: object) -> None:
    if not profile_function_runtime_enabled():
        return
    field_text = " ".join((f"{key}={value}" for key, value in fields.items()))
    logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)


@numpy_decorator(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
def erode_objects(
    image: np.ndarray,
    labels: ObjectLabelValue,
    structuring_element: StructuringElementInput = StructuringElement.DISK,
    size: StructuringElementSize = 1,
    preserve_midpoints: bool = True,
    relabel_objects: bool = False,
) -> tuple[
    np.ndarray,
    DataclassMeasurementColumnarRows,
    ObjectLabelValue,
    DirectedObjectRelationshipPayload,
]:
    """Erode CellProfiler object labels while preserving optional midpoints.

    Args:
        labels: Two-dimensional object labels to erode while retaining or relabeling
            IDs according to the midpoint settings.
    """
    from skimage.measure import label as relabel
    from openhcs.processing.backends.cellprofiler.structuring_elements import (
        adapt_structuring_element_rank,
    )

    total_started_at = time.perf_counter()
    source_labels = labels
    labels = object_label_dense_array(source_labels, dtype=np.int32)
    footprint = adapt_structuring_element_rank(
        build_structuring_element(structuring_element, size), labels.ndim
    )
    phase_started_at = time.perf_counter()
    input_labels = ObjectLabelIdDomainStrategy.for_value(labels).present_ids(labels)
    input_count = len(input_labels)
    log_function_runtime_profile(
        "erode_objects_input_labels", time.perf_counter() - phase_started_at
    )
    phase_started_at = time.perf_counter()
    eroded = MorphologyBackendStrategy.for_memory_type().erode_labeled_objects(
        labels, footprint
    )
    log_function_runtime_profile(
        "erode_objects_backend", time.perf_counter() - phase_started_at
    )
    eroded_labels = ObjectLabelIdDomainStrategy.for_value(eroded).present_ids(eroded)
    if preserve_midpoints:
        phase_started_at = time.perf_counter()
        missing_labels = np.setdiff1d(
            np.asarray(input_labels, dtype=np.int64),
            np.asarray(eroded_labels, dtype=np.int64),
            assume_unique=True,
        )
        preservation = MidpointPreservationPolicy.for_footprint(footprint)
        eroded = preservation.preserve_missing_labels(labels, eroded, missing_labels)
        log_function_runtime_profile(
            "erode_objects_preserve_midpoints",
            time.perf_counter() - phase_started_at,
            missing=len(missing_labels),
            policy=type(preservation).__name__,
        )
        output_labels = input_labels
    else:
        output_labels = eroded_labels
    if relabel_objects:
        phase_started_at = time.perf_counter()
        eroded = relabel(eroded > 0).astype(labels.dtype)
        output_labels = tuple(range(1, int(eroded.max()) + 1))
        log_function_runtime_profile(
            "erode_objects_relabel", time.perf_counter() - phase_started_at
        )
    output_count = len(output_labels)
    stats = ErosionStats(
        slice_index=0,
        input_object_count=input_count,
        output_object_count=output_count,
        objects_removed=input_count - output_count,
    )
    phase_started_at = time.perf_counter()
    eroded_value = object_label_value_with_dense_labels(
        source_labels,
        eroded,
        domain_declaration=ExplicitObjectLabelDomainDeclaration(
            ObjectLabelDomain(declared_object_ids=output_labels)
        ),
    )
    if relabel_objects:
        relationship = object_label_parent_child_payload(source_labels, eroded_value)
    else:
        relationship = object_label_identity_lineage_payload(
            source_labels, eroded_value
        )
    log_function_runtime_profile(
        "erode_objects_lineage", time.perf_counter() - phase_started_at
    )
    log_function_runtime_profile(
        "erode_objects_total", time.perf_counter() - total_started_at
    )
    return (
        image,
        DataclassMeasurementColumnarRows((stats,), row_type=ErosionStats),
        eroded_value,
        relationship,
    )


class MidpointPreservationPolicy:
    """CellProfiler midpoint preservation for labels lost during erosion."""

    def preserve_missing_labels(
        self, labels: np.ndarray, eroded: np.ndarray, missing_labels: np.ndarray
    ) -> np.ndarray:
        for label_id in missing_labels:
            label_positions = np.argwhere(labels == label_id)
            if label_positions.size == 0:
                continue
            lower = label_positions.min(axis=0)
            upper = label_positions.max(axis=0) + 1
            expanded_lower = np.maximum(lower - 1, 0)
            expanded_upper = np.minimum(upper + 1, labels.shape)
            expanded_slices = tuple(
                (
                    slice(int(start), int(stop))
                    for start, stop in zip(expanded_lower, expanded_upper, strict=True)
                )
            )
            inner_slices = tuple(
                (
                    slice(int(start - expanded_start), int(stop - expanded_start))
                    for start, stop, expanded_start in zip(
                        lower, upper, expanded_lower, strict=True
                    )
                )
            )
            output_slices = tuple(
                (
                    slice(int(start), int(stop))
                    for start, stop in zip(lower, upper, strict=True)
                )
            )
            binary = labels[expanded_slices] == label_id
            midpoint = self.midpoint_distance(binary)[inner_slices]
            eroded_region = eroded[output_slices]
            eroded_region[midpoint == np.max(midpoint)] = label_id
        return eroded

    def midpoint_distance(self, binary: np.ndarray) -> np.ndarray:
        import scipy.ndimage

        return scipy.ndimage.distance_transform_edt(binary)

    @classmethod
    def for_footprint(cls, footprint: np.ndarray) -> "MidpointPreservationPolicy":
        if SimpleDiskMidpointPreservationPolicy.matches(footprint):
            return SimpleDiskMidpointPreservationPolicy()
        return cls()


class SimpleDiskMidpointPreservationPolicy(MidpointPreservationPolicy):
    """CellProfiler's optimized disk-1 behavior restores entire missing labels."""

    @classmethod
    def matches(cls, footprint: np.ndarray) -> bool:
        import skimage.morphology

        return (
            footprint.ndim == 2
            and footprint.shape == (3, 3)
            and np.array_equal(footprint, skimage.morphology.disk(1))
        )

    def preserve_missing_labels(
        self, labels: np.ndarray, eroded: np.ndarray, missing_labels: np.ndarray
    ) -> np.ndarray:
        return eroded + labels * np.isin(labels, missing_labels)


@numpy_decorator(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
def dilate_objects(
    image: np.ndarray,
    labels: ObjectLabelValue,
    structuring_element_shape: StructuringElementInput = StructuringElement.DISK,
    structuring_element_size: StructuringElementSize = 1,
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows, ObjectLabelValue]:
    """Dilate labels with CellProfiler's higher-label-overwrites policy.

    Args:
        labels: Two-dimensional object labels whose regions are expanded by the
            selected structuring element.
    """
    from scipy.ndimage import grey_dilation

    label_array = object_label_dense_array(labels, dtype=np.int32)
    props_before = LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
        label_array
    )
    mean_area_before = (
        float(np.mean(props_before.area)) if props_before.label.size else 0.0
    )
    footprint = build_structuring_element(
        structuring_element_shape, structuring_element_size
    )
    footprint = adapt_structuring_element_rank(footprint, label_array.ndim)
    dilated_labels = grey_dilation(label_array, footprint=footprint)
    props_after = LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
        dilated_labels.astype(np.int32, copy=False)
    )
    mean_area_after = (
        float(np.mean(props_after.area)) if props_after.label.size else 0.0
    )
    stats = DilationStats(
        slice_index=0,
        object_count=int(props_after.label.size),
        mean_area_before=mean_area_before,
        mean_area_after=mean_area_after,
    )
    return (
        image,
        DataclassMeasurementColumnarRows((stats,), row_type=DilationStats),
        object_label_value_with_dense_labels(
            labels, dilated_labels.astype(np.int32, copy=False)
        ),
    )


@numpy_decorator(contract=ProcessingContract.PURE_3D)
@object_label_input_execution_mode(ObjectLabelInputExecutionMode.FULL_STACK)
@special_inputs("labels")
def dilate_objects_3d(
    image: np.ndarray,
    labels: ObjectLabelValue,
    structuring_element_shape: StructuringElementInput = StructuringElement.BALL,
    structuring_element_size: StructuringElementSize = 1,
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows, ObjectLabelValue]:
    """Dilate 3D labels with CellProfiler's higher-label-overwrites policy.

    Args:
        labels: Three-dimensional object-label volume whose regions are expanded
            by the selected volumetric structuring element.
    """
    from scipy.ndimage import grey_dilation
    from skimage.measure import regionprops

    label_array = object_label_dense_array(labels, dtype=np.int32)
    props_before = regionprops(label_array)
    volumes_before = [prop.area for prop in props_before]
    mean_volume_before = float(np.mean(volumes_before)) if volumes_before else 0.0
    footprint = build_structuring_element(
        structuring_element_shape, structuring_element_size
    )
    dilated_labels = grey_dilation(label_array, footprint=footprint)
    props_after = regionprops(dilated_labels)
    volumes_after = [prop.area for prop in props_after]
    mean_volume_after = float(np.mean(volumes_after)) if volumes_after else 0.0
    stats = DilationStats3D(
        object_count=len(props_after),
        mean_volume_before=mean_volume_before,
        mean_volume_after=mean_volume_after,
    )
    return (
        image,
        DataclassMeasurementColumnarRows((stats,), row_type=DilationStats3D),
        object_label_value_with_dense_labels(
            labels, dilated_labels.astype(np.int32, copy=False)
        ),
    )


@numpy_decorator(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
def fill_objects(
    image: np.ndarray,
    labels: ObjectLabelValue,
    mode: FillMode = FillMode.HOLES,
    diameter: float = 64.0,
    morphology_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> ObjectLabelValue:
    """Fill object holes or replace objects with convex hull labels.

    Args:
        labels: Object-label plane whose internal holes or concavities are filled.
        mode: Fill holes up to the diameter threshold or replace each object with
            its convex hull.
        diameter: Maximum hole diameter in pixels for ``holes`` mode; converted to
            a circular area threshold.
    """
    label_array = object_label_dense_array(labels, dtype=np.int32)
    if label_array.max() == 0:
        return object_label_value_with_dense_labels(labels, label_array.copy())
    filled_labels = FillObjectsModeStrategy.for_mode(mode).fill(
        FillObjectsRequest(
            image=image,
            label_array=label_array,
            diameter=diameter,
            morphology_backend_provider=morphology_backend_provider,
        )
    )
    return object_label_value_with_dense_labels(
        labels,
        filled_labels.astype(label_array.dtype, copy=False),
    )


@numpy_decorator(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
def shrink_to_object_centers(
    image: np.ndarray, labels: ObjectLabelValue
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows, ObjectLabelValue]:
    """Transform labeled objects into single-pixel centroid labels.

    Args:
        labels: Two-dimensional object labels reduced to one labeled centroid pixel
            per region.
    """
    label_array = object_label_dense_array(labels, dtype=np.int32)
    region_props = LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
        label_array
    )
    output_labels = np.zeros_like(label_array, dtype=np.int32)
    for index, label_id in enumerate(region_props.label):
        centroid_int = (
            int(round(float(region_props.centroid_y[index]))),
            int(round(float(region_props.centroid_x[index]))),
        )
        if all(
            (
                0 <= centroid_int[axis] < label_array.shape[axis]
                for axis in range(len(centroid_int))
            )
        ):
            output_labels[centroid_int] = int(label_id)
    return (
        image,
        DataclassMeasurementColumnarRows(
            (
                CentroidStats(
                    slice_index=0,
                    object_count=int(region_props.label.size),
                ),
            ),
            row_type=CentroidStats,
        ),
        object_label_value_with_dense_labels(labels, output_labels),
    )


@numpy_decorator(contract=ProcessingContract.PURE_3D)
@object_label_input_execution_mode(ObjectLabelInputExecutionMode.FULL_STACK)
@special_inputs("labels")
def shrink_to_object_centers_3d(
    image: np.ndarray, labels: ObjectLabelValue
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows, ObjectLabelValue]:
    """Transform 3D labeled objects into single-voxel centroid labels.

    Args:
        labels: Three-dimensional object labels reduced to one labeled centroid
            voxel per region.
    """
    from skimage.measure import regionprops

    label_array = object_label_dense_array(labels, dtype=np.int32)
    props = regionprops(label_array)
    output_labels = np.zeros_like(label_array, dtype=np.int32)
    for region in props:
        centroid_int = tuple((int(round(coordinate)) for coordinate in region.centroid))
        if all(
            (
                0 <= centroid_int[axis] < label_array.shape[axis]
                for axis in range(len(centroid_int))
            )
        ):
            output_labels[centroid_int] = region.label
    return (
        image,
        DataclassMeasurementColumnarRows(
            (CentroidStats(slice_index=0, object_count=len(props)),),
            row_type=CentroidStats,
        ),
        object_label_value_with_dense_labels(labels, output_labels),
    )


@numpy_decorator(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
def resize_objects(
    image: np.ndarray,
    labels: ObjectLabelValue,
    method: ResizeObjectsMethod = ResizeObjectsMethod.FACTOR,
    factor_x: float = 0.25,
    factor_y: float = 0.25,
    factor_z: float = 1.0,
    width: int = 100,
    height: int = 100,
    planes: int = 10,
) -> tuple[
    np.ndarray,
    DataclassMeasurementColumnarRows,
    ObjectLabelValue,
    DirectedObjectRelationshipPayload,
]:
    """Resize object labels by CellProfiler nearest-neighbor label semantics.

    Args:
        labels: Two-dimensional object labels to resample by factors or an explicit
            output width and height.
    """
    source_labels = labels
    labels = object_label_dense_array(source_labels, dtype=np.int32)
    original_shape = labels.shape
    request = ResizeObjectsRequest(
        labels=labels,
        method=coerce_cellprofiler_enum(ResizeObjectsMethod, method),
        factor_x=factor_x,
        factor_y=factor_y,
        factor_z=factor_z,
        width=width,
        height=height,
        planes=planes,
    )
    resized_labels = resize_object_labels_nearest(labels, request.zoom_factors())
    unique_labels = np.unique(resized_labels)
    object_count = len(unique_labels[unique_labels > 0])
    stats = ResizeObjectsStats(
        slice_index=0,
        original_height=original_shape[-2],
        original_width=original_shape[-1],
        new_height=resized_labels.shape[-2],
        new_width=resized_labels.shape[-1],
        object_count=object_count,
    )
    output_labels = object_label_value_with_dense_labels(
        source_labels,
        resized_labels,
        domain_declaration=PresentObjectLabelIdsDomainDeclaration(),
        source_spatial_domain=(
            source_labels.object_label_source_spatial_domain().with_spatial_resize(
                resized_labels.shape[-2:]
            )
        ),
    )
    output_labels = output_labels.with_variants(
        output_labels.variant_data,
        parent_image_source_voxel_spacing=SourceVoxelSpacing(),
    )
    relationship = object_label_identity_lineage_payload(source_labels, output_labels)
    return (
        image,
        DataclassMeasurementColumnarRows((stats,), row_type=ResizeObjectsStats),
        output_labels,
        relationship,
    )


def resize_objects_target_shape(
    shape: tuple[int, ...], *, planes: int, height: int, width: int
) -> tuple[int, ...]:
    spatial_shape = (planes, height, width) if len(shape) >= 3 else (height, width)
    return trailing_spatial_target_shape(shape, spatial_shape)


def resize_objects_zoom_factors(
    ndim: int, *, factor_z: float, factor_y: float, factor_x: float
) -> tuple[float, ...]:
    spatial_factors = (
        (factor_z, factor_y, factor_x) if ndim >= 3 else (factor_y, factor_x)
    )
    return trailing_spatial_factors(ndim, spatial_factors)


def resize_object_labels_nearest(
    labels: np.ndarray, zoom_factors: tuple[float, ...]
) -> np.ndarray:
    """Resize dense object labels with SciPy order-0 nearest-neighbor geometry."""
    label_array = np.asarray(labels)
    target_shape = tuple(
        int(round(axis_size * zoom_factor))
        for axis_size, zoom_factor in zip(label_array.shape, zoom_factors, strict=True)
    )
    if any(axis_size <= 0 for axis_size in target_shape):
        from scipy.ndimage import zoom

        return zoom(label_array, zoom_factors, order=0, mode="nearest").astype(np.int32)
    resized = label_array
    for axis, target_size in enumerate(target_shape):
        source_size = resized.shape[axis]
        if source_size == target_size:
            continue
        if source_size == 1:
            source_indices = np.zeros(target_size, dtype=np.intp)
        elif target_size == 1:
            source_indices = np.zeros(1, dtype=np.intp)
        else:
            source_indices = np.rint(
                np.arange(target_size, dtype=np.float64)
                * float(source_size - 1)
                / float(target_size - 1)
            ).astype(np.intp)
        resized = np.take(resized, source_indices, axis=axis)
    return resized.astype(np.int32, copy=False)


@numpy_decorator(contract=ProcessingContract.PURE_3D)
@object_label_input_execution_mode(ObjectLabelInputExecutionMode.FULL_STACK)
@special_inputs("labels")
def resize_objects_3d(
    image: np.ndarray,
    labels: ObjectLabelValue,
    method: ResizeObjectsMethod = ResizeObjectsMethod.FACTOR,
    factor_x: float = 0.25,
    factor_y: float = 0.25,
    factor_z: float = 0.25,
    width: int = 100,
    height: int = 100,
    planes: int = 10,
) -> tuple[
    np.ndarray,
    DataclassMeasurementColumnarRows,
    ObjectLabelValue,
    DirectedObjectRelationshipPayload,
]:
    """Resize 3D object labels by CellProfiler nearest-neighbor semantics.

    Args:
        labels: Three-dimensional object-label volume to resample by factors or an
            explicit plane, height, and width shape.
    """
    source_labels = labels
    labels = object_label_dense_array(source_labels, dtype=np.int32)
    original_shape = labels.shape
    request = ResizeObjectsRequest(
        labels=labels,
        method=coerce_cellprofiler_enum(ResizeObjectsMethod, method),
        factor_x=factor_x,
        factor_y=factor_y,
        factor_z=factor_z,
        width=width,
        height=height,
        planes=planes,
    )
    resized_labels = resize_object_labels_nearest(labels, request.zoom_factors())
    output_ids = ObjectLabelIdDomainStrategy.for_value(resized_labels).present_ids(
        resized_labels
    )
    object_count = len(output_ids)
    stats = ResizeObjects3DStats(
        slice_index=0,
        original_depth=original_shape[0],
        original_height=original_shape[1],
        original_width=original_shape[2],
        new_depth=resized_labels.shape[0],
        new_height=resized_labels.shape[1],
        new_width=resized_labels.shape[2],
        object_count=object_count,
    )
    output_labels = object_label_value_with_dense_labels(
        source_labels,
        resized_labels,
        domain_declaration=ExplicitObjectLabelDomainDeclaration(
            ObjectLabelDomain(declared_object_ids=output_ids)
        ),
        source_spatial_domain=(
            source_labels.object_label_source_spatial_domain().with_spatial_resize(
                resized_labels.shape[-2:]
            )
        ),
    )
    output_labels = output_labels.with_variants(
        output_labels.variant_data,
        parent_image_source_voxel_spacing=SourceVoxelSpacing(),
    )
    relationship = object_label_identity_lineage_payload(source_labels, output_labels)
    return (
        image,
        DataclassMeasurementColumnarRows((stats,), row_type=ResizeObjects3DStats),
        output_labels,
        relationship,
    )


class FillObjectsModule(
    ObjectLabelDrivenPrimaryImageInputPolicy,
    LabelsObjectInputPolicy,
    ObjectArtifactInputModule,
    ObjectArtifactOutputModule,
    CellProfilerModule,
):
    module_name = "FillObjects"
    function_name = "fill_objects"
    validated = True
    confidence = 1.0
    input_objects_setting = SettingNameFamily("Select the input objects")
    output_objects_setting = SettingNameFamily("Name the output objects")
    input_objects_binding = SettingToKeywordBinding.input(
        input_objects_setting,
        ObjectLabelsArtifactType,
        runtime_parameter_name="labels",
    )
    setting_bindings = (
        input_objects_binding,
        SettingToKeywordBinding.output(
            output_objects_setting, ObjectLabelsArtifactType
        ),
    )


class MorphModule(
    CellProfilerModule
):
    module_name = "Morph"
    function_name = "morph"
    validated = True
    confidence = 1.0
    input_image_setting = SettingNameFamily("Select the input image")
    output_image_setting = SettingNameFamily("Name the output image")
    setting_bindings = (
        SettingToKeywordBinding.input(input_image_setting, ImageArtifactType),
        SettingToKeywordBinding.output(output_image_setting, ImageArtifactType),
    )


class MorphologicalskeletonModule(
    ZStackFunctionVariantModule,
    CellProfilerModule,
):
    module_name = "Morphologicalskeleton"
    function_name = "morphologicalskeleton"
    function_variants = ("morphological_skeleton_3d",)
    validated = True
    confidence = 0.95
    input_image_setting = SettingNameFamily("Select the input image")
    output_image_setting = SettingNameFamily("Name the output image")
    setting_bindings = (
        SettingToKeywordBinding.input(input_image_setting, ImageArtifactType),
        SettingToKeywordBinding.output(output_image_setting, ImageArtifactType),
    )


class ShrinkToObjectCentersModule(
    ZStackFunctionVariantModule,
    ObjectLabelDrivenPrimaryImageInputPolicy,
    LabelsObjectInputPolicy,
    ObjectTransformContractModule,
):
    module_name = "ShrinkToObjectCenters"
    function_name = "shrink_to_object_centers"
    function_variants = ("shrink_to_object_centers_3d",)
    validated = True
    confidence = 1.0


class SplitOrMergeObjectsModule(
    PlaneRuntimeArtifactModule,
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
    ObjectArtifactOutputModule,
):
    module_name = "SplitOrMergeObjects"
    function_name = "split_or_merge_objects"
    function_variants = (
        "split_or_merge_objects_with_guide_image",
        "split_or_merge_objects_per_parent",
    )
    validated = True
    confidence = 1.0
    input_objects_setting = SettingNameFamily("Select the input objects")
    output_objects_setting = SettingNameFamily("Name the new objects")
    guide_image_setting = SettingNameFamily(
        "Select the grayscale image to guide merging"
    )
    parent_objects_setting = SettingNameFamily("Select the parent object")
    input_objects_binding = SettingToKeywordBinding.input(
        input_objects_setting, ObjectLabelsArtifactType, runtime_parameter_name="labels"
    )
    output_objects_binding = SettingToKeywordBinding.output(
        output_objects_setting,
        ObjectLabelsArtifactType,
    )
    parent_objects_binding = SettingToKeywordBinding.input(
        parent_objects_setting, ObjectLabelsArtifactType, runtime_parameter_name="parent_labels"
    )
    guide_image_binding = SettingToKeywordBinding.input(
        guide_image_setting, ImageArtifactType
    )
    operation_binding = SettingToKeywordBinding(
        "Operation",
        "operation",
        cellprofiler_enum_value_setting_parser(SplitOrMergeOperation),
    )
    merge_method_binding = SettingToKeywordBinding(
        "Merging method",
        "merge_method",
        cellprofiler_enum_value_setting_parser(SplitOrMergeMergeMethod),
    )
    use_guide_image_binding = SettingToKeywordBinding(
        "Merge using a grayscale image?",
        "use_guide_image",
        parse_cellprofiler_bool,
    )
    setting_bindings = (
        input_objects_binding,
        parent_objects_binding,
        output_objects_binding,
        guide_image_binding,
        operation_binding,
        SettingToKeywordBinding(
            "Maximum distance within which to merge objects",
            "distance_threshold",
            parse_cellprofiler_int,
        ),
        use_guide_image_binding,
        SettingToKeywordBinding(
            "Minimum intensity fraction",
            "minimum_intensity_fraction",
            parse_cellprofiler_float,
        ),
        SettingToKeywordBinding(
            "Method to find object intensity",
            "intensity_method",
            cellprofiler_enum_value_setting_parser(SplitOrMergeIntensityMethod),
        ),
        merge_method_binding,
        SettingToKeywordBinding(
            "Output object type",
            "output_object_type",
            cellprofiler_enum_value_setting_parser(SplitOrMergeOutputObjectType),
        ),
    )

    @classmethod
    def input_topology(cls, module: "ModuleBlock") -> SplitOrMergeInputTopology:
        return SplitOrMergeInputTopology.from_values(
            operation=required_setting_value(
                module, cls.operation_binding.setting_name
            ),
            merge_method=required_setting_value(
                module,
                cls.merge_method_binding.setting_name,
            ),
            use_guide_image=parse_cellprofiler_bool(
                required_setting_value(
                    module,
                    cls.use_guide_image_binding.setting_name,
                )
            ),
        )

    @classmethod
    def active_artifact_bindings(cls, module=None, *, invocation_key=None):
        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        topology = cls.input_topology(module)
        return tuple(
            binding
            for binding in bindings
            if topology is SplitOrMergeInputTopology.GUIDE_IMAGE
            or binding is not cls.guide_image_binding
            if topology is SplitOrMergeInputTopology.PARENT_OBJECTS
            or binding is not cls.parent_objects_binding
        )

    @classmethod
    def artifact_output_relations(
        cls,
        module,
        *,
        binding,
        name,
        invocation_key,
        step_context,
        artifact_inputs: ArtifactSpecCollection,
        output_position,
    ):
        if binding is not cls.output_objects_binding:
            return super().artifact_output_relations(
                module,
                binding=binding,
                name=name,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
                output_position=output_position,
            )
        del name, invocation_key, step_context, output_position
        source_names = cls.artifact_names_for_binding(
            module,
            cls.input_objects_binding,
        )
        if len(source_names) != 1:
            raise ValueError(
                f"SplitOrMergeObjects requires exactly one source object artifact, "
                f"got {source_names!r}."
            )
        source = artifact_inputs.require_by_name_and_artifact_type(
            source_names[0],
            ObjectLabelsArtifactType,
        )
        return (SourceStackLineageSourceRelation(source=source.ref()),)

    @classmethod
    def resolve_function(
        cls,
        module: "ModuleBlock",
        *,
        contract: "CallableContract",
        source_bindings: "StepSourceBindingsConfig",
    ) -> Callable[..., object]:
        del contract, source_bindings
        return cls.require_callable(cls.input_topology(module).value)

    @classmethod
    def finalize_module_blocks_for_invocation(cls, blocks, *, invocation, step_context) -> tuple[ModuleBlock, ...]:
        blocks = super().finalize_module_blocks_for_invocation(
            blocks, invocation=invocation,
            step_context=step_context,
        )
        for block in blocks:
            topology = cls.input_topology(block)
            if invocation.contract.function_name != topology.value:
                raise ValueError(
                    f"{cls.module_name} callable {invocation.contract.function_name!r} "
                    f"does not match declared input topology {topology.name}."
                )
        return blocks


__all__ = public_names_from_objects(
    CentrosomeNumpyMorphologyBackendStrategy,
    CellProfilerDeclumpMethod,
    CentroidStats,
    CombineObjectsStats,
    CombineObjectsStrategy,
    DeclumpingMaximaGeometry,
    ExpandDefinedPixelsStrategy,
    ExpandInfiniteStrategy,
    ExpandShrinkOperationStrategy,
    FillHolesOption,
    FillMode,
    HolePredicate,
    HoleRemovalDiameterPolicy,
    MaskChoice,
    MaskObjectsOutputLabels,
    MaskObjectsPlaneOperation,
    MaskObjectsPlaneResult,
    MaskObjectsStats,
    MorphOperation,
    MorphOperationRequest,
    MorphOperationStrategy,
    MorphologyBackendStrategy,
    NumbaNumpyMorphologyBackendStrategy,
    NumpyMorphologyBackendStrategy,
    RepeatMode,
    RepeatModeStrategy,
    ResizeObjectsMethod,
    ResizeObjects3DStats,
    ResizeObjectsStats,
    DilationStats,
    DilationStats3D,
    DistanceSplitOrMergeMergeMethodStrategy,
    ErosionStats,
    MergeObjectsStrategy,
    MidpointPreservationPolicy,
    ParentSplitOrMergeMergeMethodStrategy,
    SplitObjectsStrategy,
    SplitOrMergeConvexHull,
    SplitOrMergeGuideImageFilter,
    SplitOrMergeIntensityMethod,
    SplitOrMergeMergeMethod,
    SplitOrMergeMergeMethodStrategy,
    SplitOrMergeOperation,
    SplitOrMergeOperationStrategy,
    SplitOrMergeOutputObjectType,
    SplitOrMergeRequest,
    SplitOrMergeStats,
    SimpleDiskMidpointPreservationPolicy,
    apply_morph_operation,
    closing,
    combineobjects,
    dense_label_area_statistics,
    dilate_image,
    dilate_objects,
    dilate_objects_3d,
    erode_image,
    erode_objects,
    expand_or_shrink_objects,
    fill_objects,
    filter_border_objects,
    filter_labels_above_maximum_diameter,
    filter_labels_below_minimum_diameter,
    filter_labels_by_area_numba,
    filter_labels_by_diameter_range,
    filter_labels_by_diameter_range_numba,
    filter_physical_border_objects_numba,
    manual_declumping_size,
    mask_objects,
    morph,
    morphological_skeleton_2d,
    morphological_skeleton_3d,
    morphologicalskeleton,
    opening,
    positive_dense_label_count,
    prepare_expand_or_shrink_objects,
    resize_objects,
    resize_objects_3d,
    resize_objects_target_shape,
    resize_objects_zoom_factors,
    remove_holes,
    remove_holes_3d,
    shrink_to_object_centers,
    shrink_to_object_centers_3d,
    split_or_merge_objects,
    split_or_merge_objects_per_parent,
    split_or_merge_objects_with_guide_image,
)
