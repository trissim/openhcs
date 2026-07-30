"""Grid-label backends for CellProfiler-compatible processing."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import VariableComponents
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.artifacts import (
    ArtifactSpecCollection,
    ArtifactSpecRelation,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    SourceStackLineageSourceRelation,
    SpatialGridArtifactType,
)
from openhcs.core.memory.decorators import numpy
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
)
from openhcs.core.pipeline.function_contracts import special_inputs
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.core.runtime_object_label_building import (
    SourceImageObjectLabelBuildRequest,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
)
from openhcs.core.runtime_spatial_grid import (
    SpatialGridOrdering,
    SpatialGridOrigin,
)
from openhcs.core.runtime_spatial_grid import (
    SpatialGrid,
)
from openhcs.interop.cellprofiler.cellprofiler_literals import (
    cellprofiler_enum_from_literal,
)
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
    ObjectArtifactOutputModule,
    PlaneRuntimeArtifactModule,
    )
from openhcs.interop.cellprofiler.parser import ModuleBlock
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    MeasurementFeatureRecord,
    NoObjectNameMeasurementRecordMixin,
)
from openhcs.core.steps.function_runtime import RuntimeCallableArgument, RuntimeCallableKwargs
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    optional_setting_value,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingsBinder,
    SettingToKeywordBinding,
    coerce_cellprofiler_enum,
    parse_cellprofiler_bool,
    parse_cellprofiler_int,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

if TYPE_CHECKING:
    from openhcs.core.artifacts import ArtifactSpec
    from openhcs.core.function_patterns import FunctionInvocationKey
    from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
    from openhcs.core.callable_contract import CallableContract
    from openhcs.core.source_bindings import StepSourceBindingsConfig


class DefineGridCycleScope(str, Enum):
    """Closed DefineGrid execution scopes from CellProfiler."""

    EACH_CYCLE = "each_cycle"
    ONCE = "once"

    @classmethod
    def from_setting(
        cls, value: object, *, default: "DefineGridCycleScope" = EACH_CYCLE
    ) -> "DefineGridCycleScope":
        if value is None:
            return default
        if isinstance(value, cls):
            return value
        if isinstance(value, Enum):
            value = value.value
        normalized = re.sub("[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
        if normalized in {"each", "each_cycle"}:
            return cls.EACH_CYCLE
        if normalized == "once":
            return cls.ONCE
        return cls(normalized)


class DefineGridVariant(str, Enum):
    """Absorbed DefineGrid function variants."""

    MANUAL = "define_grid_manual"
    AUTOMATIC = "define_grid_automatic"

    @property
    def requires_object_input(self) -> bool:
        """Return whether this variant derives the grid from object labels."""
        return self is DefineGridVariant.AUTOMATIC

    @classmethod
    def from_module(cls, module: ModuleBlock) -> "DefineGridVariant":
        method = DefineGridManualModule.DefinitionMethod.from_literal(
            _setting_value(
                module,
                DefineGridManualModule.definition_method_setting,
                default=DefineGridManualModule.definition_method_default.value,
            )
        )
        return (
            cls.AUTOMATIC
            if method is DefineGridManualModule.DefinitionMethod.automatic
            else cls.MANUAL
        )


def define_grid_bound_kwargs(
    module: ModuleBlock, binder: SettingsBinder
) -> dict[str, Any]:
    """Return kwargs for the absorbed DefineGrid variant."""
    kwargs = {
        "grid_rows": TypedGridSetting(
            module, binder, DefineGridManualModule.rows_setting, default="8"
        ).value,
        "grid_columns": TypedGridSetting(
            module, binder, DefineGridManualModule.columns_setting, default="12"
        ).value,
        "origin": _grid_origin(
            _setting_value(
                module,
                DefineGridManualModule.origin_setting,
                default="Top left",
            )
        ),
        "ordering": _grid_ordering(
            _setting_value(
                module,
                DefineGridManualModule.ordering_setting,
                default="Rows",
            )
        ),
        "cycle_scope": _grid_cycle_scope(
            _setting_value(
                module,
                DefineGridManualModule.cycle_scope_setting,
                default="Each cycle",
            )
        ),
    }
    if DefineGridVariant.from_module(module) is DefineGridVariant.MANUAL:
        first_x, first_y = _coordinate_pair(
            _setting_value(
                module,
                DefineGridManualModule.first_coordinates_setting,
                default="100,100",
            )
        )
        second_x, second_y = _coordinate_pair(
            _setting_value(
                module,
                DefineGridManualModule.second_coordinates_setting,
                default="200,200",
            )
        )
        kwargs.update(
            {
                "first_spot_x": first_x,
                "first_spot_y": first_y,
                "first_spot_row": TypedGridSetting(
                    module,
                    binder,
                    DefineGridManualModule.first_row_setting,
                    default="1",
                ).value,
                "first_spot_col": TypedGridSetting(
                    module,
                    binder,
                    DefineGridManualModule.first_column_setting,
                    default="1",
                ).value,
                "second_spot_x": second_x,
                "second_spot_y": second_y,
                "second_spot_row": TypedGridSetting(
                    module,
                    binder,
                    DefineGridManualModule.second_row_setting,
                    default="8",
                ).value,
                "second_spot_col": TypedGridSetting(
                    module,
                    binder,
                    DefineGridManualModule.second_column_setting,
                    default="12",
                ).value,
            }
        )
    return kwargs


@dataclass(frozen=True, slots=True)
class TypedGridSetting:
    """Nominal parser request for one typed grid setting."""

    module: ModuleBlock
    binder: SettingsBinder
    setting_name: str
    default: str

    @property
    def value(self) -> Any:
        return self.binder.parse_value(
            self.setting_name,
            _setting_value(self.module, self.setting_name, default=self.default),
        )


@dataclass(frozen=True, slots=True)
class FragmentMatchedLiteral:
    """Nominal owner for CP grid literal matching by normalized word fragments."""

    value: str
    fragments_to_literal: dict[tuple[str, ...], str]

    @property
    def literal(self) -> str:
        normalized = self.value.strip().lower()
        for fragments, literal in self.fragments_to_literal.items():
            if all((fragment in normalized for fragment in fragments)):
                return literal
        raise ValueError(f"Unsupported grid setting value: {self.value!r}.")


def _setting_value(
    module: ModuleBlock,
    setting_name: str | SettingNameFamily,
    *,
    default: str,
) -> str:
    return optional_setting_value(module, setting_name) or default


def _coordinate_pair(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Grid coordinate must be x,y, got {value!r}.")
    return (int(float(parts[0])), int(float(parts[1])))


def _grid_origin(value: str) -> str:
    return FragmentMatchedLiteral(
        value=value,
        fragments_to_literal={
            ("top", "left"): "top_left",
            ("bottom", "left"): "bottom_left",
            ("top", "right"): "top_right",
            ("bottom", "right"): "bottom_right",
        },
    ).literal


def _grid_ordering(value: str) -> str:
    return FragmentMatchedLiteral(
        value=value, fragments_to_literal={("row",): "rows", ("column",): "columns"}
    ).literal


def _grid_cycle_scope(value: str) -> DefineGridCycleScope:
    return DefineGridCycleScope(
        FragmentMatchedLiteral(
            value=value,
            fragments_to_literal={("once",): "once", ("each",): "each_cycle"},
        ).literal
    )


def _shape_choice(value: str) -> str:
    return FragmentMatchedLiteral(
        value=value,
        fragments_to_literal={
            ("rectangle",): "rectangle_forced_location",
            ("circle", "forced"): "circle_forced_location",
            ("circle", "natural"): "circle_natural_location",
            ("natural",): "natural_shape_and_location",
        },
    ).literal


def _diameter_choice(value: str) -> str:
    normalized = value.strip().lower()
    if "automatic" in normalized or normalized in {"yes", "true"}:
        return "automatic"
    if "manual" in normalized or normalized in {"no", "false"}:
        return "manual"
    raise ValueError(f"Unsupported grid diameter choice: {value!r}.")


class GridCycleScopeExecutionModePolicy:
    """Honor DefineGrid's per-cycle versus once-only grid definition scope."""

    @classmethod
    def execution_mode(
        cls,
        default: ImagePayloadExecutionMode,
        *,
        image: RuntimeCallableArgument,
        kwargs: RuntimeCallableKwargs,
        variable_components: tuple[VariableComponents, ...],
    ) -> ImagePayloadExecutionMode:
        del cls, image, variable_components
        parameter_name = (
            DefineGridManualModule.cycle_scope_binding.require_parameter_name()
        )
        cycle_scope = DefineGridCycleScope.from_setting(
            kwargs[parameter_name] if parameter_name in kwargs else None
        )
        if cycle_scope is DefineGridCycleScope.ONCE:
            return ImagePayloadExecutionMode.FULL_STACK
        return default


class DefineGridManualModule(
    GridCycleScopeExecutionModePolicy,
    ObjectArtifactInputModule,
    CellProfilerModule,
):
    module_name = "DefineGridManual"
    function_name = "define_grid_manual"
    validated = True
    aliases = ("DefineGrid",)
    function_variants = ("define_grid_automatic",)
    confidence = 0.0
    definition_method_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Select the method to define the grid"
    )
    rows_setting: ClassVar[str] = "Number of rows"
    columns_setting: ClassVar[str] = "Number of columns"
    origin_setting: ClassVar[str] = "Location of the first spot"
    ordering_setting: ClassVar[str] = "Order of the spots"
    first_coordinates_setting: ClassVar[str] = "Coordinates of the first cell"
    first_row_setting: ClassVar[str] = "Row number of the first cell"
    first_column_setting: ClassVar[str] = "Column number of the first cell"
    second_coordinates_setting: ClassVar[str] = "Coordinates of the second cell"
    second_row_setting: ClassVar[str] = "Row number of the second cell"
    second_column_setting: ClassVar[str] = "Column number of the second cell"
    cycle_scope_setting: ClassVar[str] = "Define a grid for which cycle?"
    manual_definition_method_setting: ClassVar[str] = (
        "Select the method to define the grid manually"
    )
    reuse_previous_grid_setting: ClassVar[str] = (
        "Use a previous grid if gridding fails?"
    )
    display_grid_image_setting: ClassVar[str] = (
        "Select the image on which to display the grid"
    )
    drawing_image_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Select the image to display when drawing",
        aliases=("Select the image to display",),
    )
    previous_objects_setting: ClassVar[str] = "Select the previously identified objects"
    retain_grid_image_setting: ClassVar[str] = "Retain an image of the grid?"
    grid_image_output_setting: ClassVar[str] = "Name the output image"
    grid_output_setting: ClassVar[str] = "Name the grid"
    display_grid_image_binding = SettingToKeywordBinding.input(
        display_grid_image_setting, ImageArtifactType
    )
    previous_objects_binding = SettingToKeywordBinding.input(
        previous_objects_setting, ObjectLabelsArtifactType, runtime_parameter_name="labels"
    )
    grid_image_output_binding = SettingToKeywordBinding.output(
        grid_image_output_setting, ImageArtifactType
    )
    grid_output_binding = SettingToKeywordBinding.output(
        grid_output_setting, SpatialGridArtifactType
    )
    ignored_settings = (drawing_image_setting,)
    _bound_kwargs = staticmethod(define_grid_bound_kwargs)

    class DefinitionMethod(str, Enum):
        manual = "Manual"
        automatic = "Automatic"

        @classmethod
        def from_literal(
            cls, value: "DefineGridManualModule.DefinitionMethod | str"
        ) -> "DefineGridManualModule.DefinitionMethod":
            return cellprofiler_enum_from_literal(cls, value)

    definition_method_default = DefinitionMethod.manual
    cycle_scope_binding = SettingToKeywordBinding(
        cycle_scope_setting,
        "cycle_scope",
        _grid_cycle_scope,
    )
    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        display_grid_image_binding,
        previous_objects_binding,
        grid_image_output_binding,
        grid_output_binding,
        SettingToKeywordBinding(rows_setting, "grid_rows", parse_cellprofiler_int),
        SettingToKeywordBinding(
            columns_setting,
            "grid_columns",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(origin_setting, "origin", _grid_origin),
        SettingToKeywordBinding(ordering_setting, "ordering", _grid_ordering),
        SettingToKeywordBinding(
            first_row_setting,
            "first_spot_row",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            first_column_setting,
            "first_spot_col",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            second_row_setting,
            "second_spot_row",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            second_column_setting,
            "second_spot_col",
            parse_cellprofiler_int,
        ),
        cycle_scope_binding,
        SettingToKeywordBinding(retain_grid_image_setting),
    )

    @classmethod
    def bind_settings(cls, module, *, binder):
        """Bind scalar rows and the two coordinate-pair compound rows."""

        variant = DefineGridVariant.from_module(module)
        if not variant.requires_object_input:
            manual_method = optional_setting_value(
                module, cls.manual_definition_method_setting
            )
            if (
                manual_method is not None
                and "coordinate" not in manual_method.casefold()
            ):
                raise NotImplementedError(
                    "Headless DefineGrid supports coordinate-based manual grids only."
                )
        reuse_previous = optional_setting_value(module, cls.reuse_previous_grid_setting)
        if reuse_previous is not None and parse_cellprofiler_bool(reuse_previous):
            raise NotImplementedError(
                "Headless DefineGrid does not support previous-grid fallback."
            )
        bound = cls._bind_declared_settings(module, binder=binder)
        bound = bound.with_kwargs(cls._bound_kwargs(module, binder))
        bound = bound.with_consumed_settings(
            cls.definition_method_setting,
            cls.manual_definition_method_setting,
            cls.reuse_previous_grid_setting,
            cls.first_coordinates_setting,
            cls.second_coordinates_setting,
        )
        return cls._finalize_bound_settings(module, binder=binder, bound=bound)

    @classmethod
    def resolve_function(
        cls,
        module: "ModuleBlock",
        *,
        contract: "CallableContract",
        source_bindings: "StepSourceBindingsConfig",
    ) -> Callable[..., object]:
        del contract, source_bindings
        method = cls.DefinitionMethod.from_literal(
            cls.setting_value(module, cls.definition_method_setting)
            or cls.definition_method_default.value
        )
        function_name = (
            cls.function_variants[0]
            if method is cls.DefinitionMethod.automatic
            else str(cls.function_name)
        )
        return cls.require_callable(function_name)

    @classmethod
    def active_artifact_bindings(
        cls,
        module: "ModuleBlock | None",
        *,
        invocation_key: "FunctionInvocationKey | None" = None,
    ) -> tuple[SettingToKeywordBinding, ...]:
        """Expose the selected variant input and optional rendered grid."""

        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        if invocation_key is None:
            raise TypeError("DefineGrid requires its exact invocation key.")
        requires_objects = DefineGridVariant(
            invocation_key.function_name
        ).requires_object_input
        retain_image = optional_setting_value(module, cls.retain_grid_image_setting)
        retain_image = retain_image is not None and parse_cellprofiler_bool(
            retain_image
        )
        return tuple(
            binding
            for binding in bindings
            if requires_objects or binding is not cls.previous_objects_binding
            if retain_image or binding is not cls.grid_image_output_binding
        )

    @classmethod
    def artifact_output_relations(
        cls,
        module: "ModuleBlock",
        *,
        binding: SettingToKeywordBinding,
        name: str,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        artifact_inputs: ArtifactSpecCollection,
        output_position: int,
    ) -> tuple[ArtifactSpecRelation, ...]:
        """Bind grid geometry to the image or objects that define its domain."""

        if binding is not cls.grid_output_binding:
            return super().artifact_output_relations(
                module,
                binding=binding,
                name=name,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
                output_position=output_position,
            )
        del module, name, step_context, output_position
        variant = DefineGridVariant(invocation_key.function_name)
        source_type = (
            ObjectLabelsArtifactType
            if variant.requires_object_input
            else ImageArtifactType
        )
        sources = artifact_inputs.of_artifact_type(source_type)
        if len(sources) != 1:
            raise ValueError(
                f"{variant.value} requires exactly one {source_type.value} "
                f"grid-domain source, got {tuple(spec.ref() for spec in sources)!r}."
            )
        return (SourceStackLineageSourceRelation(source=sources[0].ref()),)


class IdentifyObjectsInGridModule(
    PlaneRuntimeArtifactModule,
    NoObjectNameMeasurementRecordMixin,
    ObjectArtifactInputModule,
    MeasurementArtifactOutputModule,
    ObjectArtifactOutputModule,
    CellProfilerModule,
):
    module_name = "IdentifyObjectsInGrid"
    function_name = "identify_objects_in_grid"
    validated = True
    confidence = 1.0
    grid_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Select the defined grid"
    )
    output_objects_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Name the objects to be identified"
    )
    guiding_objects_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Select the guiding objects"
    )
    shape_choice_setting: ClassVar[str] = "Select object shapes and locations"
    diameter_choice_setting: ClassVar[str] = (
        "Specify the circle diameter automatically?"
    )
    circle_diameter_setting: ClassVar[str] = "Circle diameter"
    guiding_objects_default: ClassVar[str] = "None"
    grid_binding = SettingToKeywordBinding.input(
        grid_setting, SpatialGridArtifactType, runtime_parameter_name="topology_inputs"
    )
    guiding_objects_binding = SettingToKeywordBinding.input(
        guiding_objects_setting,
        ObjectLabelsArtifactType,
        runtime_parameter_name="topology_inputs",
    )
    output_objects_binding = SettingToKeywordBinding.output(
        output_objects_setting,
        ObjectLabelsArtifactType,
    )
    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        grid_binding,
        guiding_objects_binding,
        output_objects_binding,
        SettingToKeywordBinding(shape_choice_setting, "shape_choice", _shape_choice),
        SettingToKeywordBinding(
            diameter_choice_setting,
            "diameter_choice",
            _diameter_choice,
        ),
        SettingToKeywordBinding(
            circle_diameter_setting,
            "circle_diameter",
            parse_cellprofiler_int,
        ),
    )

    @classmethod
    def active_artifact_bindings(
        cls,
        module: "ModuleBlock | None" = None,
        *,
        invocation_key: "FunctionInvocationKey | None" = None,
    ) -> tuple[SettingToKeywordBinding, ...]:
        """Declare the guide role required by the module's explicit shape."""

        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        shape_choice = _shape_choice(
            cls.setting_value(module, cls.shape_choice_setting)
            or ShapeChoice.RECTANGLE.value
        )
        if not GridShapeStrategy.for_enum_member(shape_choice).requires_guides:
            return tuple(
                binding
                for binding in bindings
                if binding is not cls.guiding_objects_binding
            )
        return bindings

    @classmethod
    def artifact_contract_inputs(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
    ) -> tuple[ArtifactSpec, ...]:
        """Anchor unguided grid labels in the current main-flow image domain."""

        declared_inputs = super().artifact_contract_inputs(
            module,
            invocation_key=invocation_key,
            step_context=step_context,
        )
        shape_choice = _shape_choice(
            cls.setting_value(module, cls.shape_choice_setting)
            or ShapeChoice.RECTANGLE.value
        )
        if GridShapeStrategy.for_enum_member(shape_choice).requires_guides:
            return declared_inputs
        spatial_grids = ArtifactSpecCollection(declared_inputs).of_artifact_type(
            SpatialGridArtifactType
        )
        if len(spatial_grids) != 1:
            raise ValueError(
                f"IdentifyObjectsInGrid shape {shape_choice!r} requires exactly "
                "one spatial-grid input."
            )
        declared_grid = spatial_grids[0]
        produced_grid = step_context.available_artifacts.by_name_and_artifact_type(
            declared_grid.name,
            SpatialGridArtifactType,
        )
        if produced_grid is None:
            raise ValueError(
                f"IdentifyObjectsInGrid cannot resolve spatial grid "
                f"{declared_grid.name!r} from available artifact producers."
            )
        lineage_refs = produced_grid.group_scope_sources()
        if len(lineage_refs) != 1:
            raise ValueError(
                f"IdentifyObjectsInGrid shape {shape_choice!r} requires its "
                "spatial grid to declare exactly one source domain, got "
                f"{lineage_refs!r}."
            )
        lineage_ref = lineage_refs[0]
        lineage_source = step_context.main_flow_artifacts.by_ref(lineage_ref)
        if lineage_source is None:
            raise ValueError(
                f"IdentifyObjectsInGrid grid lineage {lineage_ref!r} is not in "
                "the current main-flow image domain."
            )
        return (*declared_inputs, lineage_source)


GridInfo = SpatialGrid


def label_centroid_extremes(
    labels: np.ndarray,
) -> tuple[int, float, float, float, float]:
    """Return object count and min/max centroid coordinates for dense labels."""
    label_array = np.ascontiguousarray(labels, dtype=np.int32)
    if label_array.ndim != 2:
        raise ValueError(f"Automatic grid labels must be 2D, got {label_array.ndim}D.")
    return _label_centroid_extremes_numba(label_array)


@njit(cache=True)
def _label_centroid_extremes_numba(
    labels: np.ndarray,
) -> tuple[int, float, float, float, float]:
    max_label = 0
    height, width = labels.shape
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label > max_label:
                max_label = label
    if max_label <= 0:
        return (0, 0.0, 0.0, 0.0, 0.0)
    counts = np.zeros(max_label + 1, dtype=np.int64)
    y_sums = np.zeros(max_label + 1, dtype=np.float64)
    x_sums = np.zeros(max_label + 1, dtype=np.float64)
    for y in range(height):
        for x in range(width):
            label = int(labels[y, x])
            if label <= 0:
                continue
            counts[label] += 1
            y_sums[label] += float(y)
            x_sums[label] += float(x)
    object_count = 0
    first_y = np.inf
    first_x = np.inf
    second_y = -np.inf
    second_x = -np.inf
    for label in range(1, max_label + 1):
        count = counts[label]
        if count == 0:
            continue
        object_count += 1
        centroid_y = y_sums[label] / count
        centroid_x = x_sums[label] / count
        if centroid_y < first_y:
            first_y = centroid_y
        if centroid_y > second_y:
            second_y = centroid_y
        if centroid_x < first_x:
            first_x = centroid_x
        if centroid_x > second_x:
            second_x = centroid_x
    if object_count == 0:
        return (0, 0.0, 0.0, 0.0, 0.0)
    return (object_count, first_y, first_x, second_y, second_x)


class ShapeChoice(Enum):
    RECTANGLE = "rectangle_forced_location"
    CIRCLE_FORCED = "circle_forced_location"
    CIRCLE_NATURAL = "circle_natural_location"
    NATURAL = "natural_shape_and_location"


class DiameterChoice(Enum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class GridSpotReference:
    """One user-selected spot in CellProfiler DefineGrid coordinates."""

    x: float
    y: float
    row: int
    column: int


@dataclass(frozen=True, slots=True)
class SpatialGridDefinitionBase(ABC, metaclass=AutoRegisterMeta):
    """Shared CellProfiler DefineGrid coordinate policy."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True
    registry_key: ClassVar[str | None] = None
    rows: int
    columns: int
    origin: SpatialGridOrigin
    ordering: SpatialGridOrdering
    image_shape_yx: tuple[int, int]

    @classmethod
    def registered_definition_types(
        cls,
    ) -> tuple[type["SpatialGridDefinitionBase"], ...]:
        return tuple(cls.__registry__.values())

    def canonical_row_index(self, row: int) -> int:
        if self.origin.reverses_rows:
            return self.rows - row
        return row - 1

    def canonical_column_index(self, column: int) -> int:
        if self.origin.reverses_columns:
            return self.columns - column
        return column - 1

    def canonical_row_col(self, row: int, column: int) -> tuple[int, int]:
        return (self.canonical_row_index(row), self.canonical_column_index(column))


@dataclass(frozen=True, slots=True)
class SpatialGridManualDefinition(SpatialGridDefinitionBase):
    """Manual two-spot CellProfiler DefineGrid geometry policy."""

    registry_key = "manual"
    first_spot: GridSpotReference
    second_spot: GridSpotReference

    def spatial_grid(self) -> SpatialGrid:
        first_row, first_column = self.canonical_row_col(
            self.first_spot.row, self.first_spot.column
        )
        second_row, second_column = self.canonical_row_col(
            self.second_spot.row, self.second_spot.column
        )
        x_spacing = (
            1.0
            if first_column == second_column
            else float(self.first_spot.x - self.second_spot.x)
            / float(first_column - second_column)
        )
        y_spacing = (
            1.0
            if first_row == second_row
            else float(self.first_spot.y - self.second_spot.y)
            / float(first_row - second_row)
        )
        x_origin = int(self.first_spot.x - first_column * x_spacing)
        y_origin = int(self.first_spot.y - first_row * y_spacing)
        return spatial_grid_from_spacing(
            rows=self.rows,
            columns=self.columns,
            x_spacing=abs(x_spacing),
            y_spacing=abs(y_spacing),
            x_origin=x_origin,
            y_origin=y_origin,
            origin=self.origin,
            ordering=self.ordering,
            image_shape_yx=self.image_shape_yx,
        )


@dataclass(frozen=True, slots=True)
class SpatialGridAutomaticDefinition(SpatialGridDefinitionBase):
    """Automatic CellProfiler DefineGrid geometry policy from object extrema."""

    registry_key = "automatic"
    labels: np.ndarray

    def spatial_grid(self) -> SpatialGrid:
        object_count, first_y, first_x, second_y, second_x = label_centroid_extremes(
            object_label_dense_array(self.labels, dtype=np.int32)
        )
        if object_count < 2:
            raise ValueError("Need at least 2 objects to define grid automatically.")
        first_row, second_row = (
            (self.rows, 1) if self.origin.reverses_rows else (1, self.rows)
        )
        first_column, second_column = (
            (self.columns, 1) if self.origin.reverses_columns else (1, self.columns)
        )
        manual_definition = SpatialGridManualDefinition(
            rows=self.rows,
            columns=self.columns,
            first_spot=GridSpotReference(first_x, first_y, first_row, first_column),
            second_spot=GridSpotReference(
                second_x, second_y, second_row, second_column
            ),
            origin=self.origin,
            ordering=self.ordering,
            image_shape_yx=self.image_shape_yx,
        )
        first_row_c, first_col_c = manual_definition.canonical_row_col(
            first_row, first_column
        )
        second_row_c, second_col_c = manual_definition.canonical_row_col(
            second_row, second_column
        )
        if first_col_c != second_col_c:
            x_spacing = float(first_x - second_x) / float(first_col_c - second_col_c)
        else:
            x_spacing = (second_x - first_x) / max(self.columns - 1, 1)
        if first_row_c != second_row_c:
            y_spacing = float(first_y - second_y) / float(first_row_c - second_row_c)
        else:
            y_spacing = (second_y - first_y) / max(self.rows - 1, 1)
        return spatial_grid_from_spacing(
            rows=self.rows,
            columns=self.columns,
            x_spacing=abs(x_spacing),
            y_spacing=abs(y_spacing),
            x_origin=int(np.floor(first_x - first_col_c * x_spacing)),
            y_origin=int(np.floor(first_y - first_row_c * y_spacing)),
            origin=self.origin,
            ordering=self.ordering,
            image_shape_yx=self.image_shape_yx,
        )


def spatial_grid_from_spacing(
    *,
    rows: int,
    columns: int,
    x_spacing: float,
    y_spacing: float,
    x_origin: float,
    y_origin: float,
    origin: SpatialGridOrigin,
    ordering: SpatialGridOrdering,
    image_shape_yx: tuple[int, int],
) -> SpatialGrid:
    """Build an OpenHCS SpatialGrid from CP DefineGrid spacing fields."""
    total_width = int(abs(x_spacing) * columns)
    total_height = int(abs(y_spacing) * rows)
    return SpatialGrid(
        name="grid_info",
        rows=rows,
        columns=columns,
        x_spacing=abs(x_spacing),
        y_spacing=abs(y_spacing),
        x_origin=int(x_origin),
        y_origin=int(y_origin),
        total_width=total_width,
        total_height=total_height,
        origin=origin,
        ordering=ordering,
        x_locations=tuple(
            (float(int(x_origin + index * abs(x_spacing))) for index in range(columns))
        ),
        y_locations=tuple(
            (float(int(y_origin + index * abs(y_spacing))) for index in range(rows))
        ),
        source_spatial_shape_yx=image_shape_yx,
    )


@dataclass(frozen=True, slots=True)
class GridRuntimeDefinitionRequest:
    """Runtime grid geometry fields shared by grid-definition builders."""

    image_shape: tuple[int, int]
    grid: SpatialGrid | None
    grid_rows: int
    grid_columns: int
    x_spacing: float
    y_spacing: float
    x_origin: float
    y_origin: float
    ordering: SpatialGridOrdering = SpatialGridOrdering.BY_ROWS

    def spatial_grid(self) -> SpatialGrid:
        if self.grid is not None:
            return self.grid
        return SpatialGrid(
            name="grid",
            rows=self.grid_rows,
            columns=self.grid_columns,
            x_spacing=self.x_spacing,
            y_spacing=self.y_spacing,
            x_origin=self.x_origin,
            y_origin=self.y_origin,
            ordering=self.ordering,
            source_spatial_shape_yx=tuple((int(value) for value in self.image_shape)),
        )


@dataclass
class GridDefinition:
    """Executable grid geometry derived from OpenHCS spatial-grid artifacts."""

    rows: int
    columns: int
    x_spacing: float
    y_spacing: float
    x_location_of_lowest_x_spot: float
    y_location_of_lowest_y_spot: float
    x_locations: np.ndarray
    y_locations: np.ndarray
    spot_table: np.ndarray
    image_height: int
    image_width: int
    ordering: SpatialGridOrdering = SpatialGridOrdering.BY_ROWS

    @classmethod
    def from_runtime(
        cls,
        *,
        image_shape: tuple[int, int],
        grid: SpatialGrid | None,
        grid_rows: int,
        grid_columns: int,
        x_spacing: float,
        y_spacing: float,
        x_origin: float,
        y_origin: float,
        ordering: SpatialGridOrdering = SpatialGridOrdering.BY_ROWS,
    ) -> "GridDefinition":
        """Build executable grid geometry from a runtime grid or direct kwargs."""
        return cls.from_runtime_request(
            GridRuntimeDefinitionRequest(
                image_shape=image_shape,
                grid=grid,
                grid_rows=grid_rows,
                grid_columns=grid_columns,
                x_spacing=x_spacing,
                y_spacing=y_spacing,
                x_origin=x_origin,
                y_origin=y_origin,
                ordering=ordering,
            )
        )

    @classmethod
    def from_runtime_request(
        cls, request: GridRuntimeDefinitionRequest
    ) -> "GridDefinition":
        """Build executable grid geometry from one runtime request record."""
        spatial_grid = request.spatial_grid()
        height, width = (
            spatial_grid.source_spatial_shape_yx
            if spatial_grid.source_spatial_shape_yx is not None
            else request.image_shape
        )
        return cls(
            rows=spatial_grid.rows,
            columns=spatial_grid.columns,
            x_spacing=spatial_grid.x_spacing,
            y_spacing=spatial_grid.y_spacing,
            x_location_of_lowest_x_spot=spatial_grid.x_origin,
            y_location_of_lowest_y_spot=spatial_grid.y_origin,
            x_locations=np.asarray(spatial_grid.x_locations, dtype=np.float64),
            y_locations=np.asarray(spatial_grid.y_locations, dtype=np.float64),
            spot_table=np.asarray(spatial_grid.spot_table, dtype=np.int32),
            image_height=height,
            image_width=width,
            ordering=spatial_grid.ordering,
        )

    def filled_labels(self) -> np.ndarray:
        """Fill a labels matrix by labeling each rectangle in the grid."""
        i_min = int(self.y_location_of_lowest_y_spot - self.y_spacing / 2)
        j_min = int(self.x_location_of_lowest_x_spot - self.x_spacing / 2)
        return _fill_grid_numba(
            int(self.image_height),
            int(self.image_width),
            float(self.y_spacing),
            float(self.x_spacing),
            i_min,
            j_min,
            np.asarray(self.spot_table, dtype=np.int32),
        )

    def labels_for_shape(self, shape: tuple[int, int]) -> np.ndarray:
        """Return grid labels aligned to the requested output shape."""
        labels = self.filled_labels()
        if labels.shape == shape:
            return labels
        result = np.zeros(
            [max(labels.shape[i], shape[i]) for i in range(2)], dtype=np.int32
        )
        result[0 : labels.shape[0], 0 : labels.shape[1]] = labels
        return result

    def circle_labels(
        self,
        *,
        center_i: np.ndarray,
        center_j: np.ndarray,
        radius: float,
        guiding_labels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return labels constrained to circles centered on grid spot IDs."""
        labels = (
            self.labels_for_shape(guiding_labels.shape)
            if guiding_labels is not None
            else self.filled_labels()
        )
        center_i_by_label, center_j_by_label = _spot_center_lookup_numba(
            np.asarray(self.spot_table, dtype=np.int32),
            np.asarray(center_i, dtype=np.float64),
            np.asarray(center_j, dtype=np.float64),
            int(self.spot_table.max()),
        )
        return _apply_circle_mask_numba(
            np.asarray(labels, dtype=np.int32),
            center_i_by_label,
            center_j_by_label,
            float(radius),
        )

    def forced_circle_labels(self, radius: float) -> np.ndarray:
        """Return circular labels centered in each grid cell."""
        row_indices, col_indices = np.mgrid[0 : self.rows, 0 : self.columns]
        return self.circle_labels(
            center_i=(
                self.y_locations[row_indices, col_indices]
                if self.y_locations.ndim == 2
                else self.y_locations[row_indices]
            ),
            center_j=(
                self.x_locations[row_indices, col_indices]
                if self.x_locations.ndim == 2
                else self.x_locations[col_indices]
            ),
            radius=radius,
        )

    def guide_label_center_grid_ids(
        self, guide_labels: np.ndarray, *, grid_labels: np.ndarray | None = None
    ) -> np.ndarray:
        """Map each guide label ID to the grid object ID containing its center."""
        labels = self.filled_labels() if grid_labels is None else grid_labels
        return self._guide_label_grid_ids_for_labels(
            guide_labels, grid_labels=self.boundary_masked_grid_labels(labels)
        )

    def boundary_masked_grid_labels(self, labels: np.ndarray) -> np.ndarray:
        """Return grid labels masked at CP guide-acceptance cell boundaries."""
        masked_labels = labels.copy()
        y_border = int(np.ceil(self.y_spacing / 10))
        x_border = int(np.ceil(self.x_spacing / 10))
        if y_border > 0:
            ymask = labels[y_border:, :] != labels[:-y_border, :]
            masked_labels[y_border:, :][ymask] = 0
            masked_labels[:-y_border, :][ymask] = 0
        if x_border > 0:
            xmask = labels[:, x_border:] != labels[:, :-x_border]
            masked_labels[:, x_border:][xmask] = 0
            masked_labels[:, :-x_border][xmask] = 0
        return masked_labels

    @staticmethod
    def _guide_label_grid_ids_for_labels(
        guide_labels: np.ndarray, *, grid_labels: np.ndarray
    ) -> np.ndarray:
        """Map each guide label center into the supplied grid-label plane."""
        labels = grid_labels
        max_guide = int(np.max(guide_labels))
        label_center_grid_ids = np.zeros(max_guide + 1, dtype=np.int32)
        if max_guide == 0:
            return label_center_grid_ids
        centers = np.zeros((2, max_guide + 1), dtype=np.float64)
        centers_i, centers_j = _centers_of_labels_numba(
            np.asarray(guide_labels, dtype=np.int32), max_guide
        )
        centers[0, 1:] = centers_i
        centers[1, 1:] = centers_j
        bad_centers = (
            ~np.isfinite(centers[0, :])
            | ~np.isfinite(centers[1, :])
            | (centers[0, :] >= labels.shape[0])
            | (centers[1, :] >= labels.shape[1])
        )
        rounded_centers = np.zeros_like(centers, dtype=int)
        valid_centers = ~bad_centers
        rounded_centers[:, valid_centers] = np.round(centers[:, valid_centers]).astype(
            int
        )
        label_center_grid_ids = labels[rounded_centers[0, :], rounded_centers[1, :]]
        label_center_grid_ids[bad_centers] = 0
        return np.asarray(label_center_grid_ids, dtype=np.int32)

    def filtered_guides(self, guide_labels: np.ndarray) -> np.ndarray:
        """Return accepted guide-label pixels after CP grid-edge filtering."""
        grid_labels = self.filled_labels()
        return _filter_labels_by_grid_numba(
            np.asarray(guide_labels, dtype=np.int32),
            self.guide_label_center_grid_ids(guide_labels, grid_labels=grid_labels),
            grid_labels,
        )

    def labels_from_filtered_guides(self, filtered_guides: np.ndarray) -> np.ndarray:
        """Return grid-object IDs masked by accepted guide-object pixels."""
        return _mask_grid_labels_by_filtered_guides_numba(
            self.labels_for_shape(filtered_guides.shape),
            np.asarray(filtered_guides, dtype=np.int32),
        )

    def labels_as_grid_ids(self, labels: np.ndarray) -> np.ndarray:
        """Project any positive grid-object mask onto authoritative grid IDs."""
        label_array = np.asarray(labels, dtype=np.int32)
        grid_labels = self.labels_for_shape(label_array.shape)
        return _mask_grid_labels_by_filtered_guides_numba(grid_labels, label_array)


@dataclass
class GridObjectStats(MeasurementFeatureRecord):
    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
    object_count: int
    grid_rows: int
    grid_columns: int
    shape_type: str


@dataclass(frozen=True, slots=True, kw_only=True)
class GridShapeContext(ABC, metaclass=AutoRegisterMeta):
    """Shared grid-shape execution state carried through nominal requests."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True
    registry_key: ClassVar[str | None] = None
    grid: GridDefinition
    guiding_labels: np.ndarray | None = None
    diameter_choice: DiameterChoice = DiameterChoice.MANUAL
    circle_diameter: int = 20

    @classmethod
    def registered_context_types(cls) -> tuple[type["GridShapeContext"], ...]:
        return tuple(cls.__registry__.values())


@dataclass(frozen=True, slots=True, kw_only=True)
class GridShapeRequest(GridShapeContext):
    """Inputs needed to materialize one grid object shape strategy."""

    registry_key = "shape_request"
    filtered_guides: np.ndarray | None = None

    def labels(self, shape_choice: ShapeChoice) -> np.ndarray:
        """Materialize labels through the registered strategy family."""
        strategy = GridShapeStrategy.for_enum_member(shape_choice)
        return strategy.labels(self)

    @property
    def required_guiding_labels(self) -> np.ndarray:
        if self.guiding_labels is None:
            raise ValueError("Grid shape strategy requires guiding labels.")
        return self.guiding_labels

    @property
    def required_filtered_guides(self) -> np.ndarray:
        if self.filtered_guides is None:
            raise ValueError("Grid shape strategy requires filtered guiding labels.")
        return self.filtered_guides

    def circle_radius(self) -> float:
        """Return manual or area-derived circle radius for grid object modes."""
        if self.diameter_choice is DiameterChoice.MANUAL:
            return self.circle_diameter / 2.0
        filtered_guides = self.required_filtered_guides
        areas = np.bincount(filtered_guides[filtered_guides != 0].flatten())
        if len(areas) > 0 and np.any(areas != 0):
            median_area = np.median(areas[areas != 0])
            return max(1, np.sqrt(median_area / np.pi))
        return self.circle_diameter / 2.0


@dataclass(frozen=True, slots=True, kw_only=True)
class IdentifyObjectsInGridRequest(GridShapeContext):
    """Executable request for CellProfiler IdentifyObjectsInGrid semantics."""

    registry_key = "identify_objects"
    image: np.ndarray
    shape_choice: ShapeChoice

    @classmethod
    def from_runtime(
        cls,
        *,
        image: np.ndarray,
        grid_definition: GridRuntimeDefinitionRequest,
        shape_choice: ShapeChoice,
        diameter_choice: DiameterChoice,
        circle_diameter: int,
        guiding_labels: np.ndarray | None = None,
    ) -> "IdentifyObjectsInGridRequest":
        """Bind CP/runtime inputs into one nominal executable request."""
        return cls(
            image=image,
            grid=GridDefinition.from_runtime_request(grid_definition),
            shape_choice=shape_choice,
            diameter_choice=diameter_choice,
            circle_diameter=circle_diameter,
            guiding_labels=guiding_labels,
        )

    @property
    def object_count(self) -> int:
        return self.grid.rows * self.grid.columns

    @property
    def filtered_guides(self) -> np.ndarray | None:
        if self.guiding_labels is None:
            return None
        return self.grid.filtered_guides(self.guiding_labels)

    def stats(self) -> GridObjectStats:
        return GridObjectStats(
            slice_index=0,
            object_count=self.object_count,
            grid_rows=self.grid.rows,
            grid_columns=self.grid.columns,
            shape_type=self.shape_choice.value,
        )

    def execute(
        self,
    ) -> tuple[np.ndarray, DataclassMeasurementColumnarRows, ObjectLabelPayload]:
        labels = GridShapeRequest(
            grid=self.grid,
            guiding_labels=self.guiding_labels,
            filtered_guides=self.filtered_guides,
            diameter_choice=self.diameter_choice,
            circle_diameter=self.circle_diameter,
        ).labels(self.shape_choice)
        if self.shape_choice is not ShapeChoice.NATURAL:
            labels = self.grid.labels_as_grid_ids(labels)
        declared_object_extent = max(
            self.object_count, int(labels.max()) if labels.size else 0
        )
        return (
            self.image,
            DataclassMeasurementColumnarRows(
                (self.stats(),),
                row_type=GridObjectStats,
            ),
            SourceImageObjectLabelBuildRequest(
                image=self.image,
                labels=labels.astype(np.int32, copy=False),
                declared_object_count=declared_object_extent,
                declared_object_ids=tuple(range(1, declared_object_extent + 1)),
            ).payload(),
        )


@numpy(contract=ProcessingContract.PURE_2D)
def define_grid_manual(
    image: np.ndarray,
    grid_rows: int = 8,
    grid_columns: int = 12,
    first_spot_x: int = 100,
    first_spot_y: int = 100,
    first_spot_row: int = 1,
    first_spot_col: int = 1,
    second_spot_x: int = 200,
    second_spot_y: int = 200,
    second_spot_row: int = 8,
    second_spot_col: int = 12,
    origin: SpatialGridOrigin = SpatialGridOrigin.TOP_LEFT,
    ordering: SpatialGridOrdering = SpatialGridOrdering.BY_ROWS,
    cycle_scope: DefineGridCycleScope = DefineGridCycleScope.EACH_CYCLE,
) -> tuple[np.ndarray, SpatialGrid]:
    """Define a CellProfiler grid manually from two spot references.

    Args:
        first_spot_x: Horizontal pixel coordinate of the first reference spot.
        first_spot_y: Vertical pixel coordinate of the first reference spot.
        second_spot_x: Horizontal pixel coordinate of the second reference spot.
        second_spot_y: Vertical pixel coordinate of the second reference spot.
    """
    if not isinstance(cycle_scope, DefineGridCycleScope):
        raise TypeError(
            "define_grid_manual cycle_scope must be DefineGridCycleScope, "
            f"got {type(cycle_scope).__name__}."
        )
    grid = SpatialGridManualDefinition(
        rows=grid_rows,
        columns=grid_columns,
        first_spot=GridSpotReference(
            first_spot_x, first_spot_y, first_spot_row, first_spot_col
        ),
        second_spot=GridSpotReference(
            second_spot_x, second_spot_y, second_spot_row, second_spot_col
        ),
        origin=origin,
        ordering=ordering,
        image_shape_yx=tuple((int(value) for value in image.shape[-2:])),
    ).spatial_grid()
    return (image, grid)


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
def define_grid_automatic(
    image: np.ndarray,
    labels: ObjectLabelValue,
    grid_rows: int = 8,
    grid_columns: int = 12,
    origin: SpatialGridOrigin = SpatialGridOrigin.TOP_LEFT,
    ordering: SpatialGridOrdering = SpatialGridOrdering.BY_ROWS,
    cycle_scope: DefineGridCycleScope = DefineGridCycleScope.EACH_CYCLE,
) -> tuple[np.ndarray, SpatialGrid]:
    """Define a CellProfiler grid from object-label centroid extrema.

    Args:
        labels: Object-label plane whose centroid extrema establish the grid;
            at least two labeled objects are required.
    """
    if not isinstance(cycle_scope, DefineGridCycleScope):
        raise TypeError(
            "define_grid_automatic cycle_scope must be DefineGridCycleScope, "
            f"got {type(cycle_scope).__name__}."
        )
    grid = SpatialGridAutomaticDefinition(
        rows=grid_rows,
        columns=grid_columns,
        labels=object_label_dense_array(labels, dtype=np.int32),
        origin=origin,
        ordering=ordering,
        image_shape_yx=tuple((int(value) for value in image.shape[-2:])),
    ).spatial_grid()
    return (image, grid)


@numpy(contract=ProcessingContract.PURE_2D)
def draw_grid_overlay(
    image: np.ndarray,
    grid_rows: int = 8,
    grid_columns: int = 12,
    x_spacing: float = 50.0,
    y_spacing: float = 50.0,
    x_origin: float = 25.0,
    y_origin: float = 25.0,
    line_width: int = 1,
) -> np.ndarray:
    """Draw grid lines on an image plane."""
    result = image.copy().astype(np.float32)
    height, width = result.shape
    if result.max() > 1.0:
        result = result / result.max()
    line_left_x = int(x_origin - x_spacing / 2)
    line_top_y = int(y_origin - y_spacing / 2)
    for index in range(grid_columns + 1):
        x = int(line_left_x + index * x_spacing)
        if 0 <= x < width:
            y_start = max(0, line_top_y)
            y_end = min(height, int(line_top_y + grid_rows * y_spacing))
            for dx in range(-line_width // 2, line_width // 2 + 1):
                if 0 <= x + dx < width:
                    result[y_start:y_end, x + dx] = 1.0
    for index in range(grid_rows + 1):
        y = int(line_top_y + index * y_spacing)
        if 0 <= y < height:
            x_start = max(0, line_left_x)
            x_end = min(width, int(line_left_x + grid_columns * x_spacing))
            for dy in range(-line_width // 2, line_width // 2 + 1):
                if 0 <= y + dy < height:
                    result[y + dy, x_start:x_end] = 1.0
    return result


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("topology_inputs")
def identify_objects_in_grid(
    image: np.ndarray,
    topology_inputs: tuple[SpatialGrid | ObjectLabelValue, ...],
    grid_rows: int = 8,
    grid_columns: int = 12,
    x_spacing: float = 100.0,
    y_spacing: float = 100.0,
    x_origin: float = 50.0,
    y_origin: float = 50.0,
    shape_choice: ShapeChoice = ShapeChoice.RECTANGLE,
    diameter_choice: DiameterChoice = DiameterChoice.MANUAL,
    circle_diameter: int = 20,
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows, ObjectLabelPayload]:
    """Identify objects within each section of a grid pattern.

    Args:
        topology_inputs: The authoritative ``SpatialGrid`` followed, for natural
            shapes only, by the guiding object-label value.
        grid_rows: Compatibility row-count fallback; the supplied ``SpatialGrid``
            takes precedence.
        grid_columns: Compatibility column-count fallback; the supplied
            ``SpatialGrid`` takes precedence.
        x_spacing: Compatibility horizontal spot spacing in pixels; the supplied
            ``SpatialGrid`` takes precedence.
        y_spacing: Compatibility vertical spot spacing in pixels; the supplied
            ``SpatialGrid`` takes precedence.
        x_origin: Compatibility horizontal center coordinate of the first spot;
            the supplied ``SpatialGrid`` takes precedence.
        y_origin: Compatibility vertical center coordinate of the first spot;
            the supplied ``SpatialGrid`` takes precedence.
    """
    shape_strategy = GridShapeStrategy.for_enum_member(shape_choice)
    expected_input_count = 2 if shape_strategy.requires_guides else 1
    if len(topology_inputs) != expected_input_count:
        required_inputs = (
            "a spatial grid and one guiding object-label value"
            if shape_strategy.requires_guides
            else "a spatial grid"
        )
        raise ValueError(
            f"IdentifyObjectsInGrid shape {shape_choice!r} requires exactly "
            f"{expected_input_count} topology input(s): {required_inputs}; got "
            f"{len(topology_inputs)}."
        )
    grid = topology_inputs[0]
    if not isinstance(grid, SpatialGrid):
        raise TypeError(
            "IdentifyObjectsInGrid topology input 0 must be SpatialGrid, "
            f"got {type(grid).__name__}."
        )
    guiding_labels = (
        object_label_dense_array(topology_inputs[1], dtype=np.int32)
        if shape_strategy.requires_guides
        else None
    )
    return IdentifyObjectsInGridRequest.from_runtime(
        image=image,
        grid_definition=GridRuntimeDefinitionRequest(
            image_shape=image.shape,
            grid=grid,
            grid_rows=grid_rows,
            grid_columns=grid_columns,
            x_spacing=x_spacing,
            y_spacing=y_spacing,
            x_origin=x_origin,
            y_origin=y_origin,
        ),
        shape_choice=shape_choice,
        diameter_choice=diameter_choice,
        circle_diameter=circle_diameter,
        guiding_labels=guiding_labels,
    ).execute()


def prepare_identify_objects_in_grid() -> None:
    """Compile grid-label kernels before timed execution."""
    image = np.zeros((64, 64), dtype=np.float32)
    grid = SpatialGrid(
        name="Grid",
        rows=4,
        columns=4,
        x_spacing=16.0,
        y_spacing=16.0,
        x_origin=8.0,
        y_origin=8.0,
    )
    guide_labels = np.zeros((64, 64), dtype=np.int32)
    guide_labels[8:18, 8:18] = 1
    guide_labels[24:34, 24:34] = 2
    guide_payload = SourceImageObjectLabelBuildRequest(
        image=image,
        labels=guide_labels,
        declared_object_count=2,
        declared_object_ids=(1, 2),
    ).payload()
    identify_objects_in_grid.__wrapped__(
        image,
        topology_inputs=(grid,),
        shape_choice=ShapeChoice.RECTANGLE,
    )
    identify_objects_in_grid.__wrapped__(
        image,
        topology_inputs=(grid, guide_payload),
        shape_choice=ShapeChoice.NATURAL,
    )


identify_objects_in_grid.__openhcs_prepare__ = prepare_identify_objects_in_grid


@njit(cache=True)
def _fill_grid_numba(
    image_height: int,
    image_width: int,
    y_spacing: float,
    x_spacing: float,
    row_origin: int,
    col_origin: int,
    spot_table: np.ndarray,
) -> np.ndarray:
    labels = np.zeros((image_height, image_width), dtype=np.int32)
    grid_rows, grid_columns = spot_table.shape
    for grid_row in range(grid_rows):
        row_start = int(np.ceil(row_origin + grid_row * y_spacing))
        row_stop = int(np.ceil(row_origin + (grid_row + 1) * y_spacing))
        if row_start < 0:
            row_start = 0
        if row_stop > image_height:
            row_stop = image_height
        if row_start >= row_stop:
            continue
        for grid_col in range(grid_columns):
            col_start = int(np.ceil(col_origin + grid_col * x_spacing))
            col_stop = int(np.ceil(col_origin + (grid_col + 1) * x_spacing))
            if col_start < 0:
                col_start = 0
            if col_stop > image_width:
                col_stop = image_width
            if col_start >= col_stop:
                continue
            label_id = int(spot_table[grid_row, grid_col])
            for row in range(row_start, row_stop):
                for col in range(col_start, col_stop):
                    labels[row, col] = label_id
    return labels


def centers_of_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate centers of mass for each label."""
    max_label = int(labels.max())
    if max_label == 0:
        return (np.array([]), np.array([]))
    centers_i, centers_j = _centers_of_labels_numba(
        np.asarray(labels, dtype=np.int32), max_label
    )
    return (centers_i, centers_j)


@njit(cache=True)
def _centers_of_labels_numba(
    labels: np.ndarray, max_label: int
) -> tuple[np.ndarray, np.ndarray]:
    sums_i = np.zeros(max_label + 1, dtype=np.float64)
    sums_j = np.zeros(max_label + 1, dtype=np.float64)
    counts = np.zeros(max_label + 1, dtype=np.int64)
    height, width = labels.shape
    for row in range(height):
        for col in range(width):
            label_id = int(labels[row, col])
            if label_id > 0 and label_id <= max_label:
                sums_i[label_id] += row
                sums_j[label_id] += col
                counts[label_id] += 1
    centers_i = np.empty(max_label, dtype=np.float64)
    centers_j = np.empty(max_label, dtype=np.float64)
    for label_id in range(1, max_label + 1):
        count = counts[label_id]
        if count == 0:
            centers_i[label_id - 1] = np.nan
            centers_j[label_id - 1] = np.nan
        else:
            centers_i[label_id - 1] = sums_i[label_id] / count
            centers_j[label_id - 1] = sums_j[label_id] / count
    return (centers_i, centers_j)


@njit(cache=True)
def _spot_center_lookup_numba(
    spot_table: np.ndarray,
    spot_center_i: np.ndarray,
    spot_center_j: np.ndarray,
    max_label: int,
) -> tuple[np.ndarray, np.ndarray]:
    center_i_by_label = np.empty(max_label + 1, dtype=np.float64)
    center_j_by_label = np.empty(max_label + 1, dtype=np.float64)
    for label_id in range(max_label + 1):
        center_i_by_label[label_id] = np.nan
        center_j_by_label[label_id] = np.nan
    rows, columns = spot_table.shape
    for row in range(rows):
        for col in range(columns):
            label_id = int(spot_table[row, col])
            if label_id <= 0 or label_id > max_label:
                continue
            center_i_by_label[label_id] = float(spot_center_i[row, col])
            center_j_by_label[label_id] = float(spot_center_j[row, col])
    return (center_i_by_label, center_j_by_label)


@njit(cache=True)
def _apply_circle_mask_numba(
    labels: np.ndarray,
    center_i_by_label: np.ndarray,
    center_j_by_label: np.ndarray,
    radius: float,
) -> np.ndarray:
    radius2 = (radius + 0.5) * (radius + 0.5)
    height, width = labels.shape
    max_label = len(center_i_by_label) - 1
    for row in range(height):
        for col in range(width):
            label_id = int(labels[row, col])
            if label_id <= 0 or label_id > max_label:
                labels[row, col] = 0
                continue
            center_i = center_i_by_label[label_id]
            center_j = center_j_by_label[label_id]
            if np.isnan(center_i) or np.isnan(center_j):
                labels[row, col] = 0
                continue
            delta_i = row - center_i
            delta_j = col - center_j
            if delta_i * delta_i + delta_j * delta_j > radius2:
                labels[row, col] = 0
    return labels


@njit(cache=True)
def _filter_labels_by_grid_numba(
    guide_labels: np.ndarray, label_center_grid_ids: np.ndarray, grid_labels: np.ndarray
) -> np.ndarray:
    filtered = np.zeros_like(guide_labels)
    guide_height, guide_width = guide_labels.shape
    grid_height, grid_width = grid_labels.shape
    for row in range(guide_height):
        for col in range(guide_width):
            guide_id = int(guide_labels[row, col])
            if guide_id <= 0 or guide_id >= len(label_center_grid_ids):
                continue
            if row >= grid_height or col >= grid_width:
                continue
            center_grid_id = int(label_center_grid_ids[guide_id])
            if center_grid_id > 0 and int(grid_labels[row, col]) == center_grid_id:
                filtered[row, col] = guide_id
    return filtered


@njit(cache=True)
def _mask_grid_labels_by_filtered_guides_numba(
    grid_labels: np.ndarray, filtered_guides: np.ndarray
) -> np.ndarray:
    labels = grid_labels.copy()
    height, width = labels.shape
    guide_height, guide_width = filtered_guides.shape
    for row in range(height):
        for col in range(width):
            if (
                row >= guide_height
                or col >= guide_width
                or filtered_guides[row, col] == 0
            ):
                labels[row, col] = 0
    return labels


class GridShapeStrategy(
    EnumKeyedStrategyMixin[ShapeChoice],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal strategy for materializing grid object labels."""

    __enum_member_attr__ = "shape_choice"
    shape_choice: ClassVar[ShapeChoice | None] = None
    requires_guides: ClassVar[bool] = False

    @abstractmethod
    def labels(self, request: GridShapeRequest) -> np.ndarray:
        """Return dense labels for one grid shape mode."""


class RectangleGridShapeStrategy(GridShapeStrategy):
    """Fill each grid rectangle with its object label."""

    shape_choice = ShapeChoice.RECTANGLE

    def labels(self, request: GridShapeRequest) -> np.ndarray:
        return request.grid.filled_labels()


class ForcedCircleGridShapeStrategy(GridShapeStrategy):
    """Draw fixed-diameter circles at grid centers."""

    shape_choice = ShapeChoice.CIRCLE_FORCED

    def labels(self, request: GridShapeRequest) -> np.ndarray:
        return request.grid.forced_circle_labels(request.circle_diameter / 2.0)


class NaturalCircleGridShapeStrategy(GridShapeStrategy):
    """Draw automatic circles using accepted guide objects for centers/area."""

    shape_choice = ShapeChoice.CIRCLE_NATURAL
    requires_guides = True

    def labels(self, request: GridShapeRequest) -> np.ndarray:
        guiding_labels = request.required_guiding_labels
        filtered_guides = request.required_filtered_guides
        labels = request.grid.filled_labels()
        labels[filtered_guides[0 : labels.shape[0], 0 : labels.shape[1]] == 0] = 0
        centers_i, centers_j = centers_of_labels(labels)
        nmissing = np.max(request.grid.spot_table) - len(centers_i)
        if nmissing > 0:
            centers_i = np.hstack((centers_i, [np.nan] * nmissing))
            centers_j = np.hstack((centers_j, [np.nan] * nmissing))
        spot_centers_i = centers_i[request.grid.spot_table - 1]
        spot_centers_j = centers_j[request.grid.spot_table - 1]
        return request.grid.circle_labels(
            center_i=spot_centers_i,
            center_j=spot_centers_j,
            radius=request.circle_radius(),
            guiding_labels=guiding_labels,
        )


class NaturalGridShapeStrategy(GridShapeStrategy):
    """Preserve accepted guide-object shapes and relabel by center grid cell."""

    shape_choice = ShapeChoice.NATURAL
    requires_guides = True

    def labels(self, request: GridShapeRequest) -> np.ndarray:
        return request.grid.labels_from_filtered_guides(
            request.required_filtered_guides
        )


__all__ = public_names_from_objects(
    DiameterChoice,
    GridDefinition,
    GridInfo,
    GridObjectStats,
    GridRuntimeDefinitionRequest,
    GridShapeContext,
    GridShapeRequest,
    GridShapeStrategy,
    GridSpotReference,
    IdentifyObjectsInGridRequest,
    ShapeChoice,
    SpatialGridAutomaticDefinition,
    SpatialGridManualDefinition,
    centers_of_labels,
    define_grid_automatic,
    define_grid_manual,
    draw_grid_overlay,
    identify_objects_in_grid,
    label_centroid_extremes,
    prepare_identify_objects_in_grid,
    spatial_grid_from_spacing,
)
