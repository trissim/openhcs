"""Object outline backends for CellProfiler-compatible processing."""

from __future__ import annotations
from openhcs.interop.cellprofiler.setting_names import (
    required_setting_value,
    setting_values,
    SettingNameFamily,
    block_setting_value,
    normalized_symbol_name,
    repeating_setting_blocks,
    setting_name_matches,
    setting_names,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    parse_cellprofiler_bool,
    parse_cellprofiler_float,
)
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Any
import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit
from openhcs.constants.constants import MemoryType, VariableComponents
from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactType,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    SourceStackLineageSourceRelation,
)
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.steps.function_runtime import RuntimeCallableKwargs
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelInputExecutionMode,
    object_label_input_execution_mode,
    special_inputs,
)
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.core.runtime_image_values import (
    image_payload_data,
    image_payload_metadata,
    with_image_payload_data,
)
from openhcs.core.runtime_object_label_domains import ObjectLabelDomainScope
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.cellprofiler._backend import (
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    CellProfilerBackendAuthority,
)
from openhcs.processing.backends.cellprofiler.color import coerce_rgb_color
from openhcs.interop.cellprofiler.module_settings import (
    BoundModuleSettings,
)
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    ObjectArtifactInputModule,
    PlaneRuntimeArtifactModule,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    ObjectLabelsInputBindingMixin,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting

if TYPE_CHECKING:
    from openhcs.core.function_patterns import (
        FunctionInvocationKey,
        NormalizedFunctionItem,
    )
    from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext


class OutlineSourceKind(str, Enum):
    """Nominal source kind for one OverlayOutlines row."""

    IMAGE = "image"
    OBJECTS = "objects"

    @property
    def artifact_type(self) -> type[ArtifactType]:
        if self is type(self).IMAGE:
            return ImageArtifactType
        return ObjectLabelsArtifactType


class OverlayObjectsModule(
    PlaneRuntimeArtifactModule,
    ObjectArtifactInputModule,
    CellProfilerModule,
):
    module_name = "OverlayObjects"
    function_name = "overlay_objects"
    validated = True
    confidence = 1.0
    input_image_setting = SettingNameFamily(
        "Select the input image", aliases=("Input",)
    )
    input_objects_setting = SettingNameFamily(
        "Select objects to display", aliases=("Objects",)
    )
    output_image_setting = SettingNameFamily("Name the output image")
    input_objects_binding = SettingToKeywordBinding.input(
        input_objects_setting,
        ObjectLabelsArtifactType,
        runtime_parameter_name="labels",
    )
    setting_bindings = (
        SettingToKeywordBinding.input(input_image_setting, ImageArtifactType),
        input_objects_binding,
        SettingToKeywordBinding("Opacity", "opacity", parse_cellprofiler_float),
    )
    ignored_settings = (output_image_setting,)

    @classmethod
    def execution_mode(
        cls,
        default: ImagePayloadExecutionMode,
        *,
        image: "RuntimeArrayData",
        kwargs: "RuntimeCallableKwargs",
        variable_components: tuple[VariableComponents, ...],
    ) -> ImagePayloadExecutionMode:
        """Preserve one payload-scoped object volume as one invocation."""

        del image, variable_components
        labels = kwargs[
            cls.input_objects_binding.require_runtime_parameter_name()
        ]
        if (
            isinstance(labels, ObjectLabelValue)
            and labels.object_label_domain().scope is ObjectLabelDomainScope.PAYLOAD
        ):
            return ImagePayloadExecutionMode.FULL_STACK
        return default


from openhcs.processing.backends.cellprofiler.image_geometry import (
    CellProfilerPlaneGeometry,
    align_binary_mask_to_shape,
    align_label_plane_to_shape,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


class OverlayOutlinesModule(
    ObjectLabelsInputBindingMixin,
    ObjectArtifactInputModule,
    ):
    module_name = "OverlayOutlines"
    function_name = "overlay_outlines"
    validated = True
    confidence = 1.0
    blank_image_setting = "Display outlines on a blank image?"
    base_image_setting = "Select image on which to display outlines"
    output_image_setting = "Name the output image"
    display_mode_setting = SettingNameFamily(
        "Outline display mode", aliases=("Select outline display mode",)
    )
    max_type_setting = "Select method to determine brightness of outlines"
    line_mode_setting = "How to outline"
    outline_image_setting = SettingNameFamily(
        "Select outlines to display", aliases=("Select outline to display",)
    )
    objects_setting = SettingNameFamily(
        "Select objects to display", aliases=("Select object to display",)
    )
    source_kind_setting = "Load outlines from an image or objects?"
    color_setting = "Select outline color"
    base_image_binding = SettingToKeywordBinding.input(
        base_image_setting, ImageArtifactType
    )
    output_image_binding = SettingToKeywordBinding.output(
        output_image_setting, ImageArtifactType
    )
    source_kind_binding = SettingToKeywordBinding(
        source_kind_setting,
        "outline_source_kinds",
    )
    color_binding = SettingToKeywordBinding(color_setting, "outline_colors")
    outline_image_binding = SettingToKeywordBinding.input(
        outline_image_setting, ImageArtifactType, repeated=True
    )
    objects_binding = SettingToKeywordBinding.input(
        objects_setting,
        ObjectLabelsArtifactType,
        runtime_parameter_name="object_labels",
        repeated=True,
    )
    blank_image_binding = SettingToKeywordBinding(
        blank_image_setting,
        "blank_image",
        parse_cellprofiler_bool,
    )
    display_mode_binding = SettingToKeywordBinding(
        display_mode_setting,
        "display_mode",
    )
    max_type_binding = SettingToKeywordBinding(max_type_setting, "max_type")
    line_mode_binding = SettingToKeywordBinding(line_mode_setting, "line_mode")
    setting_bindings = (base_image_binding, outline_image_binding, objects_binding,output_image_binding,blank_image_binding,
        display_mode_binding,
        max_type_binding,
        line_mode_binding,
        source_kind_binding,
        color_binding,)

    @dataclass(frozen=True, slots=True)
    class OutlineRow:
        source_kind: OutlineSourceKind
        image_name: str | None
        objects_name: str | None
        color: str

        @property
        def input_name(self) -> str:
            if self.source_kind is OutlineSourceKind.IMAGE:
                if self.image_name is None:
                    raise RuntimeError("Image outline row has no image input.")
                return self.image_name
            if self.objects_name is None:
                raise RuntimeError("Object outline row has no object input.")
            return self.objects_name

        @property
        def input_is_image(self) -> bool:
            return self.source_kind is OutlineSourceKind.IMAGE

        @property
        def artifact_type(self) -> type[ArtifactType]:
            return self.source_kind.artifact_type

    @classmethod
    def postprocess_bound_settings(
        cls,
        module: "ModuleBlock",
        bound: BoundModuleSettings,
    ) -> BoundModuleSettings:
        rows = cls.outline_rows(module)
        kwargs: dict[str, object] = {
            cls.blank_image_binding.require_parameter_name(): cls.uses_blank_image(
                module
            ),
            cls.display_mode_binding.require_parameter_name(): required_setting_value(
                module, cls.display_mode_binding.setting_name
            ),
            cls.line_mode_binding.require_parameter_name(): required_setting_value(
                module, cls.line_mode_binding.setting_name
            ),
            cls.max_type_binding.require_parameter_name(): required_setting_value(
                module, cls.max_type_binding.setting_name
            ),
            cls.source_kind_binding.require_parameter_name(): tuple(
                row.source_kind.value for row in rows
            ),
            cls.color_binding.require_parameter_name(): tuple(
                row.color for row in rows
            ),
        }
        image_names = tuple(
            row.image_name for row in rows if row.image_name is not None
        )
        object_names = tuple(
            row.objects_name for row in rows if row.objects_name is not None
        )
        if image_names:
            kwargs[cls.outline_image_binding.require_parameter_name()] = (
                image_names[0] if len(image_names) == 1 else image_names
            )
        if object_names:
            kwargs[cls.objects_binding.require_parameter_name()] = (
                object_names[0] if len(object_names) == 1 else object_names
            )
        return bound.with_kwargs(kwargs)

    @classmethod
    def _source_kinds(
        cls,
        module: ModuleBlock,
    ) -> tuple[OutlineSourceKind, ...]:
        source_kinds = tuple(
            coerce_cellprofiler_enum(OutlineSourceKind, value)
            for value in setting_values(module, cls.source_kind_setting)
        )
        if not source_kinds:
            raise ValueError("OverlayOutlines requires at least one outline row.")
        return source_kinds

    @classmethod
    def active_artifact_bindings(
        cls,
        module: ModuleBlock | None = None,
        *,
        invocation_key: FunctionInvocationKey | None = None,
    ) -> tuple[SettingToKeywordBinding, ...]:
        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        source_kinds = (
            cls._source_kinds(module)
            if setting_values(module, cls.source_kind_setting)
            else (OutlineSourceKind.OBJECTS,)
        )
        has_image_outlines = OutlineSourceKind.IMAGE in source_kinds
        has_object_outlines = OutlineSourceKind.OBJECTS in source_kinds
        return tuple(
            binding
            for binding in bindings
            if not cls.uses_blank_image(module) or binding is not cls.base_image_binding
            if has_image_outlines or binding is not cls.outline_image_binding
            if has_object_outlines or binding is not cls.objects_binding
        )

    @classmethod
    def finalize_module_blocks_for_invocation(
        cls,
        blocks, *,
        invocation: NormalizedFunctionItem,
        step_context: ArtifactDeclarationStepContext,
    ) -> tuple[ModuleBlock, ...]:
        """Reconstruct every ordered outline row on the nominal declaration."""

        blocks = super().finalize_module_blocks_for_invocation(
            blocks, invocation=invocation,
            step_context=step_context,
        )
        reconstructed_blocks = tuple(
            reconstructed
            for block in blocks
            if (reconstructed := cls._block_with_outline_rows(block)) is not None
        )
        return reconstructed_blocks

    @classmethod
    def _block_with_outline_rows(cls, block: ModuleBlock) -> ModuleBlock | None:
        source_kinds = cls._source_kinds(block)
        if (
            len(cls._outline_names(block, setting=cls.outline_image_setting))
            != sum(kind is OutlineSourceKind.IMAGE for kind in source_kinds)
            or len(cls._outline_names(block, setting=cls.objects_setting))
            != sum(kind is OutlineSourceKind.OBJECTS for kind in source_kinds)
        ):
            return None
        rows = cls._outline_rows_from_columns(block)
        records = [
            record
            for record in block.iter_settings()
            if not (
                setting_name_matches(
                    record.name,
                    cls.source_kind_binding.setting_name,
                )
                or setting_name_matches(
                    record.name,
                    cls.color_binding.setting_name,
                )
                or setting_name_matches(
                    record.name,
                    cls.outline_image_binding.setting_name,
                )
                or setting_name_matches(
                    record.name,
                    cls.objects_binding.setting_name,
                )
            )
        ]
        for row in rows:
            records.extend(
                (
                    ModuleSetting(
                        setting_names(cls.outline_image_setting)[0],
                        row.image_name or "None",
                    ),
                    ModuleSetting(
                        setting_names(cls.objects_setting)[0],
                        row.objects_name or "None",
                    ),
                    ModuleSetting(
                        cls.source_kind_setting,
                        row.source_kind.value,
                    ),
                    ModuleSetting(cls.color_setting, row.color),
                )
            )
        return replace(
            block,
            setting_records=records,
        )

    @classmethod
    def _outline_colors(cls, block: ModuleBlock, row_count: int) -> tuple[str, ...]:
        colors = setting_values(block, cls.color_setting)
        if len(colors) == 1:
            return colors * row_count
        if len(colors) != row_count:
            raise ValueError(
                "OverlayOutlines colors must contain one shared value or one "
                f"value for each of {row_count} rows, got {colors!r}."
            )
        return colors

    @classmethod
    def _required_outline_names(
        cls,
        block: ModuleBlock,
        *,
        setting: str | SettingNameFamily,
        count: int,
    ) -> tuple[str, ...]:
        names = cls._outline_names(block, setting=setting)
        if len(names) != count:
            raise ValueError(
                "OverlayOutlines row reconstruction requires "
                f"{count} value(s) for {setting_names(setting)[0]!r}, got "
                f"{names!r}."
            )
        return names

    @classmethod
    def _outline_names(
        cls,
        block: ModuleBlock,
        *,
        setting: str | SettingNameFamily,
    ) -> tuple[str, ...]:
        return tuple(
            name
            for value in setting_values(block, setting)
            if (name := normalized_symbol_name(value)) is not None
        )

    @classmethod
    def uses_blank_image(cls, module: "ModuleBlock") -> bool:
        return parse_cellprofiler_bool(
            required_setting_value(module, cls.blank_image_binding.setting_name)
        )

    @classmethod
    def base_image_name(cls, module: "ModuleBlock") -> str | None:
        if cls.uses_blank_image(module):
            return None
        return required_setting_value(module, cls.base_image_setting)

    @classmethod
    def artifact_output_relations(
        cls,
        module: ModuleBlock,
        *,
        invocation_key,
        step_context,
        binding,
        name,
        artifact_inputs: ArtifactSpecCollection,
        output_position: int,
    ):
        """Anchor output provenance to the declared canvas source."""

        del invocation_key, step_context, binding, name, output_position
        base_image_name = cls.base_image_name(module)
        if base_image_name is None:
            first_row = cls.outline_rows(module)[0]
            source_name = first_row.input_name
            source_type = first_row.artifact_type
        else:
            source_name = base_image_name
            source_type = ImageArtifactType
        source = artifact_inputs.require_by_name_and_artifact_type(
            source_name,
            source_type,
        )
        return (SourceStackLineageSourceRelation(source=source.ref()),)

    @classmethod
    def outline_rows(
        cls, module: "ModuleBlock"
    ) -> tuple["OverlayOutlinesModule.OutlineRow", ...]:
        rows = (
            cls._ordered_outline_rows(module)
            if setting_values(module, cls.source_kind_setting)
            else cls._object_outline_rows(module)
        )
        if not rows:
            raise ValueError(
                f"Module {module.name}({module.module_num}) declares no OverlayOutlines rows."
            )
        return rows

    @classmethod
    def _object_outline_rows(
        cls,
        module: "ModuleBlock",
    ) -> tuple["OverlayOutlinesModule.OutlineRow", ...]:
        """Return the CP3 object/color row schema in declared column order."""

        object_names = setting_values(module, cls.objects_setting)
        colors = setting_values(module, cls.color_setting)
        if len(object_names) != len(colors):
            raise ValueError(
                "OverlayOutlines object rows require one color per object: "
                f"objects={object_names!r}, colors={colors!r}."
            )
        return tuple(
            cls._outline_row_from_fields(
                module,
                image_name=None,
                objects_name=normalized_symbol_name(object_name),
                source_kind=OutlineSourceKind.OBJECTS,
                color=color,
            )
            for object_name, color in zip(object_names, colors, strict=True)
        )

    @classmethod
    def _ordered_outline_rows(
        cls, module: "ModuleBlock"
    ) -> tuple["OverlayOutlinesModule.OutlineRow", ...]:
        image_blocks = repeating_setting_blocks(
            module.iter_settings(), start_name=cls.outline_image_setting
        )
        if image_blocks and all(
            block_setting_value(block, cls.source_kind_setting, default="").strip()
            for block in image_blocks
        ):
            return tuple(
                (cls._outline_row_from_block(module, block) for block in image_blocks)
            )
        return cls._outline_rows_from_columns(module)

    @classmethod
    def _outline_rows_from_columns(
        cls,
        module: "ModuleBlock",
    ) -> tuple["OverlayOutlinesModule.OutlineRow", ...]:
        source_kinds = cls._source_kinds(module)
        colors = cls._outline_colors(module, len(source_kinds))
        image_names = iter(
            cls._required_outline_names(
                module,
                setting=cls.outline_image_setting,
                count=sum(kind is OutlineSourceKind.IMAGE for kind in source_kinds),
            )
        )
        object_names = iter(
            cls._required_outline_names(
                module,
                setting=cls.objects_setting,
                count=sum(kind is OutlineSourceKind.OBJECTS for kind in source_kinds),
            )
        )
        return tuple(
            cls._outline_row_from_fields(
                module,
                image_name=(
                    next(image_names)
                    if source_kind is OutlineSourceKind.IMAGE
                    else None
                ),
                objects_name=(
                    next(object_names)
                    if source_kind is OutlineSourceKind.OBJECTS
                    else None
                ),
                source_kind=source_kind,
                color=color,
            )
            for source_kind, color in zip(source_kinds, colors, strict=True)
        )

    @classmethod
    def _outline_row_from_block(
        cls, module: "ModuleBlock", block: Sequence["ModuleSetting"]
    ) -> "OverlayOutlinesModule.OutlineRow":
        return cls._outline_row_from_fields(
            module,
            image_name=normalized_symbol_name(
                block_setting_value(block, cls.outline_image_setting)
            ),
            objects_name=normalized_symbol_name(
                block_setting_value(block, cls.objects_setting)
            ),
            source_kind=coerce_cellprofiler_enum(
                OutlineSourceKind,
                block_setting_value(block, cls.source_kind_setting),
            ),
            color=block_setting_value(block, cls.color_setting, default="Red"),
        )

    @classmethod
    def _outline_row_from_fields(
        cls,
        module: "ModuleBlock",
        *,
        image_name: str | None,
        objects_name: str | None,
        source_kind: OutlineSourceKind,
        color: str,
    ) -> "OverlayOutlinesModule.OutlineRow":
        row = cls.OutlineRow(
            source_kind=source_kind,
            image_name=image_name,
            objects_name=objects_name,
            color=color,
        )
        cls._validate_outline_row(module, row)
        return row

    @classmethod
    def _validate_outline_row(
        cls, module: "ModuleBlock", row: "OverlayOutlinesModule.OutlineRow"
    ) -> None:
        if row.input_is_image:
            if row.image_name is None:
                raise ValueError(
                    f"Module {module.name}({module.module_num}) has an image outline row with no outline image input."
                )
            return
        if row.objects_name is None:
            raise ValueError(
                f"Module {module.name}({module.module_num}) has an object outline row with no object input."
            )

    @classmethod
    def artifact_contract_inputs(
        cls,
        module: ModuleBlock,
        *,
        invocation_key: FunctionInvocationKey,
        step_context: ArtifactDeclarationStepContext,
    ) -> tuple[ArtifactSpec, ...]:
        """Preserve base-image then mixed outline-row input order."""

        inputs: list[ArtifactSpec] = []
        base_image_name = cls.base_image_name(module)
        if base_image_name is not None:
            inputs.append(
                cls.require_available_artifact_input(
                    module,
                    binding=cls.base_image_binding,
                    name=base_image_name,
                    invocation_key=invocation_key,
                    step_context=step_context,
                )
            )
        for row in cls.outline_rows(module):
            inputs.append(
                cls.require_available_artifact_input(
                    module,
                    binding=(
                        cls.outline_image_binding
                        if row.source_kind is OutlineSourceKind.IMAGE
                        else cls.objects_binding
                    ),
                    name=row.input_name,
                    invocation_key=invocation_key,
                    step_context=step_context,
                )
            )
        return tuple(inputs)


class ObjectOutlineBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Object outline operations keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def outline(self, labels: np.ndarray) -> np.ndarray:
        """Return a labeled inner outline image."""


class NumbaNumpyObjectOutlineBackendStrategy(ObjectOutlineBackendStrategy):
    """Numba-accelerated NumPy object outline primitives."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        labels = np.array([[0, 1, 1], [0, 1, 0], [2, 2, 0]], dtype=np.int32)
        self.outline(labels)

    def outline(self, labels: np.ndarray) -> np.ndarray:
        label_array = np.asarray(labels, dtype=np.int32)
        if label_array.ndim > 2:
            label_array = np.max(label_array, axis=tuple(range(label_array.ndim - 2)))
        if label_array.ndim != 2:
            raise NotImplementedError("Object outlines currently support 2-D labels.")
        return _outline_numba(np.ascontiguousarray(label_array))


class CentrosomeNumpyObjectOutlineBackendStrategy(ObjectOutlineBackendStrategy):
    """Explicit centrosome provider for NumPy object outlines."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.CENTROSOME
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.CENTROSOME
    is_default_backend = False

    def outline(self, labels: np.ndarray) -> np.ndarray:
        from centrosome.outline import outline

        return outline(labels)


@njit(cache=True)
def _outline_numba(labels: np.ndarray) -> np.ndarray:
    height, width = labels.shape
    output = np.zeros((height, width), dtype=labels.dtype)
    for y in range(height):
        for x in range(width):
            center = labels[y, x]
            if center <= 0:
                continue
            min_label = center
            max_label = center
            for dy in range(-1, 2):
                ny = y + dy
                for dx in range(-1, 2):
                    nx = x + dx
                    if ny < 0 or ny >= height or nx < 0 or (nx >= width):
                        value = 0
                    else:
                        value = labels[ny, nx]
                    if value < min_label:
                        min_label = value
                    if value > max_label:
                        max_label = value
            if max_label != min_label:
                output[y, x] = center
    return output


class LineMode(Enum):
    """Closed CellProfiler outline boundary modes."""

    INNER = ("inner", "Inner")
    OUTER = ("outer", "Outer")
    THICK = ("thick", "Thick")

    @property
    def skimage_mode(self) -> str:
        return self.value[0]


class OutlineDisplayMode(Enum):
    """Closed CellProfiler outline display modes."""

    COLOR = ("color", "Color")
    GRAYSCALE = ("grayscale", "Grayscale")


class MaxType(Enum):
    """Closed CellProfiler grayscale outline intensity modes."""

    MAX_IMAGE = ("max_image", "Max of image")
    MAX_POSSIBLE = ("max_possible", "Max possible")


@dataclass(frozen=True, slots=True)
class OverlayOutlineRuntimeRow:
    """One runtime OverlayOutlines row after compiler lowering."""

    source_kind: OutlineSourceKind
    color: tuple[float, float, float]

    @classmethod
    def from_literals(
        cls, source_kind: OutlineSourceKind | str, color: str | Sequence[float]
    ) -> "OverlayOutlineRuntimeRow":
        return cls(
            source_kind=coerce_cellprofiler_enum(OutlineSourceKind, source_kind),
            color=coerce_rgb_color(color),
        )


@dataclass(frozen=True, slots=True)
class OverlayOutlineExecutionContext:
    """Runtime OverlayOutlines plan shared by plane and single-slice execution."""

    rows: tuple[OverlayOutlineRuntimeRow, ...]
    object_labels: tuple[ObjectLabelValue, ...]
    blank_image: bool
    display_mode: OutlineDisplayMode
    line_mode: LineMode
    max_type: MaxType

    def __post_init__(self) -> None:
        if len(self.object_labels) != self.object_row_count:
            raise ValueError(
                "OverlayOutlines object_labels count must match object rows."
            )
        if not all(
            isinstance(labels, ObjectLabelValue) for labels in self.object_labels
        ):
            raise TypeError(
                "OverlayOutlines requires runtime-projected ObjectLabelValue inputs."
            )

    @property
    def image_row_count(self) -> int:
        return sum((row.source_kind is OutlineSourceKind.IMAGE for row in self.rows))

    @property
    def object_row_count(self) -> int:
        return sum((row.source_kind is OutlineSourceKind.OBJECTS for row in self.rows))

    @property
    def first_outline_image_index(self) -> int:
        return 0 if self.blank_image else 1

    def render(self, image_sources: tuple[np.ndarray, ...]) -> np.ndarray:
        return self.render_single_plane(image_sources)

    def render_single_plane(self, image_sources: tuple[np.ndarray, ...]) -> np.ndarray:
        import skimage.color

        output = _base_image(
            image_sources=image_sources,
            object_labels=self.object_labels,
            blank_image=self.blank_image,
            display_mode=self.display_mode,
        )
        outline_intensity = _outline_intensity(output, self.blank_image, self.max_type)
        image_index = self.first_outline_image_index
        object_index = 0
        for row in self.rows:
            if row.source_kind is OutlineSourceKind.IMAGE:
                output = _draw_outline_image(
                    output,
                    image_sources[image_index],
                    row.color,
                    outline_intensity=outline_intensity,
                    display_mode=self.display_mode,
                )
                image_index += 1
                continue
            output = _draw_object_labels(
                output,
                self.object_labels[object_index],
                row.color,
                outline_intensity=outline_intensity,
                display_mode=self.display_mode,
                line_mode=self.line_mode,
            )
            object_index += 1
        if self.display_mode is OutlineDisplayMode.GRAYSCALE and output.ndim == 3:
            return skimage.color.rgb2gray(output).astype(np.float32)
        return output.astype(np.float32)


@numpy(contract=ProcessingContract.FLEXIBLE)
@object_label_input_execution_mode(ObjectLabelInputExecutionMode.MATCH_IMAGE_STACK)
@special_inputs("object_labels")
def overlay_outlines(
    image: np.ndarray,
    *,
    blank_image: bool = False,
    display_mode: OutlineDisplayMode = OutlineDisplayMode.COLOR,
    line_mode: LineMode = LineMode.INNER,
    max_type: MaxType = MaxType.MAX_IMAGE,
    outline_source_kinds: Sequence[OutlineSourceKind] = (
        OutlineSourceKind.OBJECTS,
    ),
    outline_colors: Sequence[str | Sequence[float]] = ("Red",),
    object_labels: Sequence[ObjectLabelValue] = (),
) -> np.ndarray:
    """Overlay object-derived or image-derived outlines onto one output image.

    Args:
        object_labels: Object-label inputs consumed by ``objects`` rows in
            ``outline_source_kinds``, in matching row order.
    """
    context = OverlayOutlineExecutionContext(
        rows=_runtime_rows(outline_source_kinds, outline_colors),
        object_labels=tuple(object_labels),
        blank_image=blank_image,
        display_mode=display_mode,
        line_mode=line_mode,
        max_type=max_type,
    )
    image_sources = _image_sources_from_payload(
        image, blank_image=context.blank_image, image_row_count=context.image_row_count
    )
    output = context.render(image_sources)
    return with_image_payload_data(
        image,
        output,
        metadata=replace(
            image_payload_metadata(image),
            source_channel_axis=(
                -1 if context.display_mode is OutlineDisplayMode.COLOR else None
            ),
        ),
    )


@numpy(contract=ProcessingContract.FLEXIBLE)
@object_label_input_execution_mode(ObjectLabelInputExecutionMode.MATCH_IMAGE_STACK)
@special_inputs("labels")
def overlay_objects(
    image: np.ndarray,
    labels: ObjectLabelValue,
    opacity: float = 0.3,
    max_label: int | None = None,
    seed: int | None = None,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlay object labels onto a 2-D image or one 3-D image volume.

    Args:
        labels: Two-dimensional label image or three-dimensional label volume whose
            regions receive colored overlays.
        max_label: Upper label ID used to normalize colormap sampling; omit to use
            the largest observed positive label.
        seed: Optional random seed for reproducible stochastic colormap behavior;
            omit for deterministic named colormaps.
        colormap: Matplotlib colormap name used to assign RGB colors across positive
            object labels.
    """
    if not isinstance(labels, ObjectLabelValue):
        raise TypeError("OverlayObjects requires a runtime-projected ObjectLabelValue.")
    label_data = object_label_dense_array(labels, dtype=np.int32)
    if label_data.ndim == 2:
        overlay = _overlay_objects_array(
            image,
            label_data,
            opacity=opacity,
            max_label=max_label,
            seed=seed,
            colormap=colormap,
        )
    elif label_data.ndim == 3:
        image_data = np.asarray(image_payload_data(image))
        channel_axis = image_payload_metadata(image).normalized_source_channel_axis(
            image
        )
        image_volume = (
            np.mean(image_data, axis=channel_axis)
            if channel_axis is not None
            else image_data
        )
        if image_volume.shape != label_data.shape:
            raise ValueError(
                "OverlayObjects image and label volumes must have matching shapes; "
                f"got {image_volume.shape!r} and {label_data.shape!r}."
            )
        volume_max_label = (
            int(label_data.max()) if max_label is None else int(max_label)
        )
        overlay = np.stack(
            tuple(
                _overlay_objects_array(
                    image_plane,
                    label_plane,
                    opacity=opacity,
                    max_label=volume_max_label,
                    seed=seed,
                    colormap=colormap,
                )
                for image_plane, label_plane in zip(
                    image_volume, label_data, strict=True
                )
            ),
            axis=0,
        )
    else:
        raise ValueError(
            "OverlayObjects requires 2-D or 3-D object labels, got "
            f"shape {label_data.shape!r}."
        )
    return with_image_payload_data(
        image,
        overlay,
        metadata=replace(image_payload_metadata(image), source_channel_axis=-1),
    )


def _overlay_objects_array(
    image: np.ndarray,
    labels: np.ndarray,
    *,
    opacity: float,
    max_label: int | None,
    seed: int | None,
    colormap: str,
) -> np.ndarray:
    """Return OverlayObjects pixels for one image/label plane."""
    image_data = np.asarray(image_payload_data(image))
    channel_axis = image_payload_metadata(image).normalized_source_channel_axis(image)
    image_plane = (
        np.mean(image_data, axis=channel_axis)
        if channel_axis is not None
        else image_data.copy()
    )
    if image_plane.ndim != 2:
        raise ValueError(
            "OverlayObjects requires one projected XY image plane; "
            f"got shape {image_data.shape!r}."
        )
    if image_plane.max() > 1.0:
        image_plane = image_plane / image_plane.max()
    label_plane = CellProfilerPlaneGeometry.from_image_plane(image_plane).label_plane(
        labels
    )
    if max_label is None:
        max_label = int(label_plane.max())
    if seed is not None:
        np.random.seed(seed)
    label_count = max_label + 1
    colors = _overlay_objects_color_table(colormap, label_count)
    if colors.size == 0:
        overlay = np.stack([image_plane, image_plane, image_plane], axis=-1)
    else:
        overlay = np.stack([image_plane, image_plane, image_plane], axis=-1).astype(
            np.float32,
            copy=False,
        )
        foreground = label_plane > 0
        if np.any(foreground):
            foreground_colors = colors[(label_plane[foreground] - 1) % colors.shape[0]]
            overlay[foreground] = (1.0 - opacity) * overlay[
                foreground
            ] + opacity * foreground_colors
    return np.clip(overlay, 0, 1).astype(np.float32)


@lru_cache(maxsize=256)
def _overlay_objects_color_table(colormap: str, label_count: int) -> np.ndarray:
    """Return CellProfiler OverlayObjects RGB colors for one label domain."""
    from matplotlib import colormaps

    colormap_object = colormaps.get_cmap(colormap)
    return np.asarray(
        [
            colormap_object(index / max(label_count - 1, 1))[:3]
            for index in range(1, label_count)
        ],
        dtype=np.float32,
    )


def _runtime_rows(
    source_kinds: Sequence[OutlineSourceKind],
    colors: Sequence[str | Sequence[float]],
) -> tuple[OverlayOutlineRuntimeRow, ...]:
    if not source_kinds:
        raise ValueError("OverlayOutlines requires at least one outline row.")
    return tuple(
        (
            OverlayOutlineRuntimeRow.from_literals(
                source_kind, _indexed_value(colors, index, default="Red")
            )
            for index, source_kind in enumerate(source_kinds)
        )
    )


def _image_sources_from_payload(
    image: np.ndarray, *, blank_image: bool, image_row_count: int
) -> tuple[np.ndarray, ...]:
    expected_count = image_row_count if blank_image else image_row_count + 1
    if expected_count == 0:
        return ()
    if expected_count == 1:
        return (image,)
    if image.ndim < 3 or image.shape[0] != expected_count:
        raise ValueError(
            f"OverlayOutlines expected a stack whose first axis contains the base image plus outline images; expected {expected_count} planes, got shape {getattr(image, 'shape', None)}."
        )
    return tuple((image[index] for index in range(expected_count)))


def _base_image(
    *,
    image_sources: tuple[np.ndarray, ...],
    object_labels: Sequence[ObjectLabelValue],
    blank_image: bool,
    display_mode: OutlineDisplayMode,
) -> np.ndarray:
    import skimage.color
    from skimage import img_as_float

    if blank_image:
        shape = _blank_shape(image_sources, object_labels)
        if display_mode is OutlineDisplayMode.COLOR:
            return np.zeros((*shape, 3), dtype=np.float32)
        return np.zeros(shape, dtype=np.float32)
    if not image_sources:
        raise ValueError("OverlayOutlines requires a base image outside blank mode.")
    base = img_as_float(image_sources[0])
    if display_mode is OutlineDisplayMode.COLOR:
        if base.ndim == 2:
            return skimage.color.gray2rgb(base).astype(np.float32)
        return base.astype(np.float32)
    if base.ndim == 3:
        return skimage.color.rgb2gray(base).astype(np.float32)
    return base.astype(np.float32)


def _blank_shape(
    image_sources: tuple[np.ndarray, ...], object_labels: Sequence[ObjectLabelValue]
) -> tuple[int, ...]:
    if object_labels:
        labels = object_label_dense_array(next(iter(object_labels)))
        if labels.ndim != 2:
            raise ValueError(
                "OverlayOutlines requires runtime-projected 2-D object labels."
            )
        return tuple(labels.shape)
    if image_sources:
        return tuple(image_sources[0].shape[:2])
    raise ValueError("OverlayOutlines blank mode requires an outline source.")


def _outline_intensity(
    output: np.ndarray, blank_image: bool, max_type: MaxType
) -> float:
    if blank_image or max_type is MaxType.MAX_POSSIBLE:
        return 1.0
    return float(np.max(output))


def _draw_object_labels(
    output: np.ndarray,
    labels: ObjectLabelValue,
    color: tuple[float, float, float],
    *,
    outline_intensity: float,
    display_mode: OutlineDisplayMode,
    line_mode: LineMode,
) -> np.ndarray:
    import skimage.color
    import skimage.segmentation

    label_plane = object_label_dense_array(labels, dtype=np.int32)
    if label_plane.ndim != 2:
        raise ValueError(
            "OverlayOutlines requires runtime-projected 2-D object labels."
        )
    labels_2d = align_label_plane_to_shape(label_plane, output.shape[:2])
    outline_color: tuple[float, float, float] | float
    if display_mode is OutlineDisplayMode.COLOR:
        if output.ndim == 2:
            output = skimage.color.gray2rgb(output)
        outline_color = color
    else:
        outline_color = outline_intensity
    boundaries = skimage.segmentation.find_boundaries(
        labels_2d, mode=line_mode.skimage_mode
    )
    if not np.any(boundaries):
        return output
    return skimage.segmentation.mark_boundaries(
        output, labels_2d, color=outline_color, mode=line_mode.skimage_mode
    )


def _draw_outline_image(
    output: np.ndarray,
    outline_image: np.ndarray,
    color: tuple[float, float, float],
    *,
    outline_intensity: float,
    display_mode: OutlineDisplayMode,
) -> np.ndarray:
    import skimage.color

    mask = _outline_image_mask(outline_image)
    mask = align_binary_mask_to_shape(mask, output.shape[:2])
    if display_mode is OutlineDisplayMode.COLOR:
        if output.ndim == 2:
            output = skimage.color.gray2rgb(output)
        output[mask] = color
        return output
    output[mask] = outline_intensity
    return output


def _outline_image_mask(outline_image: np.ndarray) -> np.ndarray:
    mask = np.asarray(image_payload_data(outline_image)) > 0
    channel_axis = image_payload_metadata(outline_image).normalized_source_channel_axis(
        outline_image
    )
    if channel_axis is not None:
        return np.any(mask, axis=channel_axis)
    return mask


def _indexed_value(values: Sequence[Any], index: int, *, default: Any) -> Any:
    if not values:
        return default
    if index < len(values):
        return values[index]
    return values[-1]


__all__ = public_names_from_objects(
    CentrosomeNumpyObjectOutlineBackendStrategy,
    LineMode,
    MaxType,
    NumbaNumpyObjectOutlineBackendStrategy,
    ObjectOutlineBackendStrategy,
    OutlineDisplayMode,
    OutlineSourceKind,
    OverlayOutlineExecutionContext,
    OverlayOutlineRuntimeRow,
    overlay_objects,
    overlay_outlines,
)
