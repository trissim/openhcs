"""Object outline backends for CellProfiler-compatible processing."""

from __future__ import annotations
from openhcs.interop.cellprofiler.setting_names import (
    required_setting_value,
    optional_setting_value,
    setting_values,
    RepeatedSettingSequence,
    SettingNameFamily,
    block_setting_value,
    normalized_symbol_name,
    repeating_setting_blocks,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    parse_cellprofiler_float,
)
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, ClassVar
import numpy as np
import skimage.color
import skimage.segmentation
from metaclass_registry import AutoRegisterMeta
from numba import njit
from skimage import img_as_float
from openhcs.constants.constants import MemoryType
from openhcs.core.image_shapes import is_color_image_slice
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.runtime_batch_contracts import (
    RuntimePure2DSliceBatchRequest,
    pure_2d_batch_executor,
)
from openhcs.core.runtime_values import (
    image_payload_data,
    object_label_dense_array,
    with_image_payload_data,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.cellprofiler._backend import (
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    CellProfilerBackendAuthority,
)
from openhcs.processing.backends.cellprofiler.color import coerce_rgb_color
from openhcs.interop.cellprofiler.module_declarations import (
    ProcessingContract,
    BinderSettingsSourceModule,
    BoundModuleSettings,
    CellProfilerModule,
    CellProfilerArtifactCapability,
    ImageArtifactInputCapability,
    ImageArtifactInputModule,
    ImageArtifactOutputCapability,
    ImageArtifactOutputModule,
    ModuleSettingsSourceModule,
    ObjectArtifactInputModule,
    ObjectLabelArtifactInputCapability,
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
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    OverlayOutlinesInputPolicy,
)


class OverlayObjectsModule(
    ObjectArtifactInputModule,
    ImageArtifactInputModule,
    ImageArtifactOutputModule,
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
    image_input_settings = (input_image_setting,)
    object_input_settings = (input_objects_setting,)
    image_output_settings = (output_image_setting,)
    setting_bindings = (
        SettingToKeywordBinding("Opacity", "opacity", parse_cellprofiler_float),
    )

    @classmethod
    def ignored_settings_for(
        cls, module: "ModuleBlock"
    ) -> tuple[str | SettingNameFamily, ...]:
        del module
        return (
            cls.input_image_setting,
            cls.input_objects_setting,
            cls.output_image_setting,
        )


from openhcs.processing.backends.cellprofiler.image_geometry import (
    CellProfilerPlaneGeometry,
    align_binary_mask_to_shape,
    align_label_plane_to_shape,
    collapse_singleton_plane_stack,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


class OverlayOutlinesModule(
    OverlayOutlinesInputPolicy,
    ImageArtifactInputModule,
    ObjectArtifactInputModule,
    ImageArtifactOutputModule,
    ModuleSettingsSourceModule,
):
    module_name = "OverlayOutlines"
    function_name = "overlay_outlines"
    validated = True
    contract = ProcessingContract.FLEXIBLE
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

    class SourceKind(str, Enum):
        IMAGE = "image"
        OBJECTS = "objects"

    @dataclass(frozen=True, slots=True)
    class OutlineRow:
        source_kind: "OverlayOutlinesModule.SourceKind"
        image_name: str | None
        objects_name: str | None
        color: str

        @property
        def input_name(self) -> str:
            if self.source_kind.value == "image":
                if self.image_name is None:
                    raise RuntimeError("Image outline row has no image input.")
                return self.image_name
            if self.objects_name is None:
                raise RuntimeError("Object outline row has no object input.")
            return self.objects_name

        @property
        def input_is_image(self) -> bool:
            return self.source_kind.value == "image"

        @property
        def input_capability(self) -> type[CellProfilerArtifactCapability]:
            if self.source_kind is OverlayOutlinesModule.SourceKind.IMAGE:
                return ImageArtifactInputCapability
            return ObjectLabelArtifactInputCapability

    @classmethod
    def settings_source(cls, module: "ModuleBlock") -> "CellProfilerKwargs":
        rows = cls.outline_rows(module)
        return {
            "blank_image": cls.uses_blank_image(module),
            "display_mode": optional_setting_value(module, cls.display_mode_setting)
            or "Color",
            "line_mode": optional_setting_value(module, cls.line_mode_setting)
            or "Inner",
            "max_type": optional_setting_value(module, cls.max_type_setting)
            or "Max of image",
            "outline_source_kinds": tuple((row.source_kind.value for row in rows)),
            "outline_colors": tuple((row.color for row in rows)),
        }

    @classmethod
    def compile_time_public_setting_names(cls):
        return (
            *super().compile_time_public_setting_names(),
            cls.base_image_setting,
            cls.outline_image_setting,
            cls.objects_setting,
            cls.output_image_setting,
        )

    @classmethod
    def compile_time_public_setting_records(cls, module, source_schema=None):
        del source_schema
        from openhcs.interop.cellprofiler.parser import ModuleSetting
        from openhcs.interop.cellprofiler.setting_names import (
            setting_name_matches,
            setting_names,
        )

        preserved_settings = (
            cls.base_image_setting,
            cls.outline_image_setting,
            cls.objects_setting,
            cls.output_image_setting,
        )
        setting_records = module.iter_settings()
        if setting_records:
            return tuple(
                setting
                for setting in setting_records
                if any(
                    setting_name_matches(setting.name, preserved)
                    for preserved in preserved_settings
                )
            )
        return tuple(
            ModuleSetting(setting_names(setting_name)[0], value)
            for setting_name in preserved_settings
            for value in setting_values(module, setting_name)
        )

    @classmethod
    def uses_blank_image(cls, module: "ModuleBlock") -> bool:
        value = optional_setting_value(module, cls.blank_image_setting)
        return value is not None and value.strip().lower() == "yes"

    @classmethod
    def base_image_name(cls, module: "ModuleBlock") -> str | None:
        if cls.uses_blank_image(module):
            return None
        return required_setting_value(module, cls.base_image_setting)

    @classmethod
    def output_image_name(cls, module: "ModuleBlock") -> str:
        return required_setting_value(module, cls.output_image_setting)

    @classmethod
    def outline_rows(
        cls, module: "ModuleBlock"
    ) -> tuple["OverlayOutlinesModule.OutlineRow", ...]:
        if module.iter_settings():
            rows = cls._ordered_outline_rows(module)
        else:
            rows = cls._outline_rows_from_mapping(module)
        if not rows:
            raise ValueError(
                f"Module {module.name}({module.module_num}) declares no OverlayOutlines rows."
            )
        return rows

    @classmethod
    def _ordered_outline_rows(
        cls, module: "ModuleBlock"
    ) -> tuple["OverlayOutlinesModule.OutlineRow", ...]:
        image_blocks = repeating_setting_blocks(
            module.iter_settings(), start_name=cls.outline_image_setting
        )
        if image_blocks:
            return tuple(
                (cls._outline_row_from_block(module, block) for block in image_blocks)
            )
        object_blocks = repeating_setting_blocks(
            module.iter_settings(), start_name=cls.objects_setting
        )
        if object_blocks:
            return cls._outline_rows_from_mapping(module)
        return ()

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
            source_kind_literal=block_setting_value(block, cls.source_kind_setting),
            color=block_setting_value(block, cls.color_setting, default="Red"),
        )

    @classmethod
    def _outline_rows_from_mapping(
        cls, module: "ModuleBlock"
    ) -> tuple["OverlayOutlinesModule.OutlineRow", ...]:
        image_names = setting_values(module, cls.outline_image_setting)
        object_names = setting_values(module, cls.objects_setting)
        source_kind_values = setting_values(module, cls.source_kind_setting)
        colors = setting_values(module, cls.color_setting)
        row_count = max(
            len(image_names),
            len(object_names),
            len(source_kind_values),
            1 if object_names or image_names else 0,
        )
        return tuple(
            (
                cls._outline_row_from_fields(
                    module,
                    image_name=normalized_symbol_name(
                        RepeatedSettingSequence(image_names).at(index)
                    ),
                    objects_name=normalized_symbol_name(
                        RepeatedSettingSequence(object_names).at(index)
                    ),
                    source_kind_literal=RepeatedSettingSequence(source_kind_values).at(
                        index
                    ),
                    color=RepeatedSettingSequence(colors, default="Red").at(index),
                )
                for index in range(row_count)
            )
        )

    @classmethod
    def _outline_row_from_fields(
        cls,
        module: "ModuleBlock",
        *,
        image_name: str | None,
        objects_name: str | None,
        source_kind_literal: str,
        color: str,
    ) -> "OverlayOutlinesModule.OutlineRow":
        source_kind = cls._source_kind_from_fields(
            source_kind_literal, image_name=image_name, objects_name=objects_name
        )
        row = cls.OutlineRow(
            source_kind=source_kind,
            image_name=image_name,
            objects_name=objects_name,
            color=color,
        )
        cls._validate_outline_row(module, row)
        return row

    @classmethod
    def _source_kind_from_fields(
        cls, value: str, *, image_name: str | None, objects_name: str | None
    ) -> "OverlayOutlinesModule.SourceKind":
        normalized = value.strip().lower()
        if normalized.startswith("image"):
            return cls.SourceKind.IMAGE
        if normalized.startswith("object"):
            return cls.SourceKind.OBJECTS
        if image_name is not None and objects_name is None:
            return cls.SourceKind.IMAGE
        return cls.SourceKind.OBJECTS

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
    def artifact_contract(cls, assembler, builder, module):
        inputs = []
        base_image_name = cls.base_image_name(module)
        if base_image_name is not None:
            inputs.append(
                ImageArtifactInputCapability.bind_artifact(cls, builder, module, ImageArtifactInputCapability.spec(base_image_name))
            )
        for row in cls.outline_rows(module):
            inputs.append(
                row.input_capability.bind_artifact(cls, builder, module, row.input_capability.spec(row.input_name))
            )
        output = cls.image_output_artifact(
            builder,
            module,
            cls.output_image_name(module),
            setting=cls.output_image_setting,
        )
        return assembler.assemble_contract(
            module, builder, inputs=inputs, outputs=[output]
        )


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


class OutlineSourceKind(str, Enum):
    """Runtime source kind for one OverlayOutlines row."""

    IMAGE = "image"
    OBJECTS = "objects"


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
    object_labels: tuple[np.ndarray, ...]
    blank_image: bool
    display_mode: OutlineDisplayMode
    line_mode: LineMode
    max_type: MaxType

    def __post_init__(self) -> None:
        if len(self.object_labels) != self.object_row_count:
            raise ValueError(
                "OverlayOutlines object_labels count must match object rows."
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

    def plane(self, slice_index: int) -> "OverlayOutlineExecutionContext":
        return type(self)(
            rows=self.rows,
            object_labels=tuple(
                (
                    _plane_payload_slice(labels, slice_index)
                    for labels in self.object_labels
                )
            ),
            blank_image=self.blank_image,
            display_mode=self.display_mode,
            line_mode=self.line_mode,
            max_type=self.max_type,
        )

    def render(self, image_sources: tuple[np.ndarray, ...]) -> np.ndarray:
        if _requires_plane_stack_execution(image_sources, self.object_labels):
            return self.render_plane_stack(image_sources)
        return self.render_single_plane(image_sources)

    def render_plane_stack(self, image_sources: tuple[np.ndarray, ...]) -> np.ndarray:
        slice_count = _aligned_plane_slice_count((*image_sources, *self.object_labels))
        return np.stack(
            tuple(
                (
                    self.plane(slice_index).render_single_plane(
                        tuple(
                            (
                                _plane_payload_slice(source, slice_index)
                                for source in image_sources
                            )
                        )
                    )
                    for slice_index in range(slice_count)
                )
            )
        ).astype(np.float32)

    def render_single_plane(self, image_sources: tuple[np.ndarray, ...]) -> np.ndarray:
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
                collapse_singleton_plane_stack(self.object_labels[object_index]),
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
def overlay_outlines(
    image: np.ndarray,
    *,
    blank_image: bool = False,
    display_mode: OutlineDisplayMode | str = OutlineDisplayMode.COLOR,
    line_mode: LineMode | str = LineMode.INNER,
    max_type: MaxType | str = MaxType.MAX_IMAGE,
    outline_source_kinds: Sequence[OutlineSourceKind | str] = (
        OutlineSourceKind.OBJECTS,
    ),
    outline_colors: Sequence[str | Sequence[float]] = ("Red",),
    object_labels: Sequence[np.ndarray] = (),
    dtype_config: Any | None = None,
) -> np.ndarray:
    """Overlay object-derived or image-derived outlines onto one output image."""
    del dtype_config
    context = OverlayOutlineExecutionContext(
        rows=_runtime_rows(outline_source_kinds, outline_colors),
        object_labels=tuple(
            (object_label_dense_array(labels) for labels in object_labels)
        ),
        blank_image=blank_image,
        display_mode=coerce_cellprofiler_enum(OutlineDisplayMode, display_mode),
        line_mode=coerce_cellprofiler_enum(LineMode, line_mode),
        max_type=coerce_cellprofiler_enum(MaxType, max_type),
    )
    image_sources = _image_sources_from_payload(
        image, blank_image=context.blank_image, image_row_count=context.image_row_count
    )
    return context.render(image_sources)


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
def overlay_objects(
    image: np.ndarray,
    labels: np.ndarray,
    opacity: float = 0.3,
    max_label: int | None = None,
    seed: int | None = None,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlay object labels onto an image plane using CellProfiler geometry."""
    overlay = _overlay_objects_array(
        image_payload_data(image),
        labels,
        opacity=opacity,
        max_label=max_label,
        seed=seed,
        colormap=colormap,
    )
    return with_image_payload_data(image, overlay)


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
    if image.ndim == 3:
        image_plane = np.mean(image, axis=-1)
    else:
        image_plane = image.copy()
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
            overlay[foreground] = (
                (1.0 - opacity) * overlay[foreground]
                + opacity * foreground_colors
            )
    return np.clip(overlay, 0, 1).astype(np.float32)


def overlay_objects_batch(
    request: RuntimePure2DSliceBatchRequest,
) -> list[np.ndarray]:
    """Batch OverlayObjects over equivalent pure-2D slice invocations."""
    kwargs = request.kwargs
    labels = kwargs.get("labels")
    if labels is None:
        return [request.execute_one(index) for index in range(request.slice_count)]
    opacity = float(kwargs.get("opacity", 0.3))
    max_label = kwargs.get("max_label", None)
    seed = kwargs.get("seed", None)
    colormap = str(kwargs.get("colormap", "jet"))
    outputs: list[np.ndarray] = []
    for index, image_slice in enumerate(request.slices_2d):
        label_slice = labels[index] if getattr(labels, "ndim", 0) >= 3 else labels
        overlay = _overlay_objects_array(
            image_payload_data(image_slice),
            label_slice,
            opacity=opacity,
            max_label=max_label,
            seed=seed,
            colormap=colormap,
        )
        outputs.append(with_image_payload_data(image_slice, overlay))
    return outputs


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
    source_kinds: Sequence[OutlineSourceKind | str],
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


def _requires_plane_stack_execution(
    image_sources: tuple[np.ndarray, ...], object_labels: Sequence[np.ndarray]
) -> bool:
    return any(
        (
            _is_plane_stack_payload(payload)
            for payload in (*image_sources, *object_labels)
        )
    )


def _aligned_plane_slice_count(payloads: Sequence[np.ndarray]) -> int:
    slice_counts = frozenset(
        (
            _plane_slice_count(payload)
            for payload in payloads
            if _is_plane_stack_payload(payload)
        )
    )
    if not slice_counts:
        return 1
    if len(slice_counts) != 1:
        raise ValueError(
            f"OverlayOutlines plane-stack inputs must have aligned slice counts; got {sorted(slice_counts)!r}."
        )
    return next(iter(slice_counts))


def _plane_payload_slice(payload: np.ndarray, slice_index: int) -> np.ndarray:
    if _is_plane_stack_payload(payload):
        return payload[slice_index]
    return payload


def _plane_slice_count(payload: np.ndarray) -> int:
    return int(payload.shape[0])


def _is_plane_stack_payload(payload: np.ndarray) -> bool:
    return payload.ndim == 3 and (not is_color_image_slice(payload))


def _base_image(
    *,
    image_sources: tuple[np.ndarray, ...],
    object_labels: Sequence[np.ndarray],
    blank_image: bool,
    display_mode: OutlineDisplayMode,
) -> np.ndarray:
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
    image_sources: tuple[np.ndarray, ...], object_labels: Sequence[np.ndarray]
) -> tuple[int, ...]:
    if object_labels:
        return tuple(collapse_singleton_plane_stack(object_labels[0]).shape)
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
    labels: np.ndarray,
    color: tuple[float, float, float],
    *,
    outline_intensity: float,
    display_mode: OutlineDisplayMode,
    line_mode: LineMode,
) -> np.ndarray:
    labels_2d = align_label_plane_to_shape(
        object_label_dense_array(labels, dtype=np.int32), output.shape[:2]
    )
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
    mask = np.asarray(outline_image) > 0
    if is_color_image_slice(mask):
        return np.any(mask, axis=-1)
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


pure_2d_batch_executor(overlay_objects_batch)(overlay_objects)
