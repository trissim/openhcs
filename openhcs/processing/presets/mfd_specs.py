"""Typed builders for MFD preset pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from metaclass_registry import AutoRegisterMeta
from arraybridge.decorators import DtypeConversion

from openhcs.constants.constants import VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.config import LazyDtypeConfig, LazyProcessingConfig
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.analysis.cell_counting_cpu import (
    DetectionMethod,
    count_cells_single_channel,
)
from openhcs.processing.backends.analysis.multi_template_matching import (
    OpenCVTemplateMatchMethod,
    multi_template_crop_reference_channel,
)
from openhcs.processing.backends.analysis.skan_axon_analysis import (
    AnalysisDimension,
    OutputMode,
    skan_axon_skeletonize_and_analyze,
)
from openhcs.processing.backends.assemblers.assemble_stack_cupy import (
    assemble_stack_cupy,
)
from openhcs.processing.backends.pos_gen.ashlar_main_cpu import (
    ashlar_compute_tile_positions_cpu,
)
from openhcs.processing.backends.pos_gen.ashlar_main_gpu import (
    ashlar_compute_tile_positions_gpu,
)
from openhcs.processing.backends.processors.cupy_processor import (
    create_composite,
    crop,
    sobel,
    stack_percentile_normalize,
    tophat,
)

MFD_WHOLE_DEVICE_TEMPLATE_PATH = (
    "/home/ts/nvme_usb/configs/templates/mfd_96_sobel_10x_whole_device.tif"
)
MFD_COMPARTMENT_WIDTH = 5046
MFD_COMPARTMENT_HEIGHT = 3694
MFD_AXON_START_X = 5253


@dataclass(frozen=True, slots=True, kw_only=True)
class PresetStepBinding:
    """Shared declaration fields for MFD preset step rows."""

    name: str
    variable_components: tuple[Any, ...] = ()
    input_source: InputSource | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class PresetStepSpec:
    """Declarative FunctionStep materialization spec."""

    binding: PresetStepBinding
    func_factory: Callable[[], Any]

    def materialize(self) -> FunctionStep:
        kwargs: dict[str, Any] = {
            "func": self.func_factory(),
            "name": self.binding.name,
        }
        if self.binding.variable_components or self.binding.input_source is not None:
            kwargs["processing_config"] = LazyProcessingConfig(
                variable_components=(
                    list(self.binding.variable_components)
                    if self.binding.variable_components
                    else None
                ),
                input_source=self.binding.input_source,
            )
        return FunctionStep(**kwargs)


@dataclass(frozen=True, slots=True)
class PresetPipelineSpec:
    """Declarative preset pipeline spec with fresh step materialization."""

    name: str
    steps: tuple[PresetStepSpec, ...]

    def materialize(self) -> list[FunctionStep]:
        return [step.materialize() for step in self.steps]


class MfdPresetKey(str, Enum):
    """Named MFD preset variants materialized by this spec authority."""

    CROP_ANALYZE = "10x_mfd_crop_analyze"
    CROP_ANALYZE_DAPI_FITC_CY5 = "10x_mfd_crop_analyze_dapi_fitc_cy5"
    STITCH_ASHLAR_CPU = "10x_mfd_stitch_ashlar_cpu"
    STITCH_GPU = "10x_mfd_stitch_gpu"


class MfdPresetFamily(str, Enum):
    """High-level MFD preset template families."""

    CROP_ANALYZE = "crop_analyze"
    STITCH = "stitch"


@dataclass(frozen=True, slots=True)
class MfdPresetDefinition:
    """Variant row for MFD preset materialization."""

    key: MfdPresetKey
    family: MfdPresetFamily
    include_cy5_cell_count: bool = False
    stitch_func: Callable[..., Any] | None = None

    def required_stitch_func(self) -> Callable[..., Any]:
        if self.stitch_func is None:
            raise ValueError(f"MFD stitch preset {self.key.value} is missing stitch_func")
        return self.stitch_func


def _normalize(percentile_low: float = 0.1, percentile_high: float = 99.9) -> tuple[Any, dict[str, float]]:
    return (
        stack_percentile_normalize,
        {
            "low_percentile": percentile_low,
            "high_percentile": percentile_high,
        },
    )


def _sobel_slice_by_slice() -> tuple[Any, dict[str, bool]]:
    return (sobel, {"slice_by_slice": True})


def _template_crop_func() -> tuple[Any, dict[str, Any]]:
    return (
        multi_template_crop_reference_channel,
        {
            "score_threshold": 0.1,
            "method": OpenCVTemplateMatchMethod.SQDIFF_NORMED,
            "template_path": MFD_WHOLE_DEVICE_TEMPLATE_PATH,
            "rotate_result": False,
        },
    )


def _compartment_crop(channel: str) -> tuple[Any, dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "width": MFD_COMPARTMENT_WIDTH,
        "height": MFD_COMPARTMENT_HEIGHT,
    }
    if channel == "3":
        kwargs["start_x"] = MFD_AXON_START_X
        kwargs["dtype_config"] = LazyDtypeConfig(
            default_dtype_conversion=DtypeConversion.UINT16
        )
    return (crop, kwargs)


def _crop_compartments_func() -> dict[str, Any]:
    return {
        "1": _compartment_crop("1"),
        "2": _compartment_crop("2"),
        "3": _compartment_crop("3"),
        "4": [_compartment_crop("4"), tophat],
    }


def _cell_count_func(
    min_cell_area: int,
    max_cell_area: int,
) -> tuple[Any, dict[str, Any]]:
    return (
        count_cells_single_channel,
        {
            "min_cell_area": min_cell_area,
            "max_cell_area": max_cell_area,
            "enable_preprocessing": False,
            "detection_method": DetectionMethod.WATERSHED,
            "dtype_config": LazyDtypeConfig(
                default_dtype_conversion=DtypeConversion.UINT8
            ),
        },
    )


def _axon_analysis_func() -> tuple[Any, dict[str, Any]]:
    return (
        skan_axon_skeletonize_and_analyze,
        {
            "analysis_dimension": AnalysisDimension.TWO_D,
            "return_skeleton_visualizations": True,
            "skeleton_visualization_mode": OutputMode.SKELETON,
            "min_branch_length": 20.0,
            "dtype_config": LazyDtypeConfig(
                default_dtype_conversion=DtypeConversion.UINT8
            ),
        },
    )


def _analysis_func(*, include_cy5_cell_count: bool) -> dict[str, Any]:
    channel_4: Any = []
    if include_cy5_cell_count:
        channel_4 = _cell_count_func(min_cell_area=100, max_cell_area=1000)
    return {
        "1": [],
        "2": _cell_count_func(min_cell_area=40, max_cell_area=200),
        "3": _axon_analysis_func(),
        "4": channel_4,
    }


def _channel_preprocess_func(include_channel_4: bool) -> dict[str, Any]:
    channel_preprocess = {
        "1": [_normalize(), _sobel_slice_by_slice(), _normalize()],
        "2": [_normalize(), tophat, _normalize()],
        "3": [_normalize(), tophat, _normalize()],
    }
    if include_channel_4:
        channel_preprocess["4"] = [_normalize(), tophat, _normalize()]
    return channel_preprocess


@dataclass(frozen=True, slots=True, kw_only=True)
class PresetStepTemplate:
    """Template row for repeated MFD step declarations."""

    binding: PresetStepBinding
    func_factory_factory: Callable[[MfdPresetDefinition], Callable[[], Any]]

    def materialize_spec(self, definition: MfdPresetDefinition) -> PresetStepSpec:
        return PresetStepSpec(
            binding=self.binding,
            func_factory=self.func_factory_factory(definition),
        )


STITCH_STEP_TEMPLATES: tuple[PresetStepTemplate, ...] = (
    PresetStepTemplate(
        binding=PresetStepBinding(name="process"),
        func_factory_factory=lambda definition: (
            lambda: _channel_preprocess_func(include_channel_4=False)
        ),
    ),
    PresetStepTemplate(
        binding=PresetStepBinding(
            name="composite",
            variable_components=(VariableComponents.CHANNEL,),
        ),
        func_factory_factory=lambda definition: lambda: create_composite,
    ),
    PresetStepTemplate(
        binding=PresetStepBinding(name="gpu_stitch"),
        func_factory_factory=lambda definition: (
            lambda: (definition.required_stitch_func(), {"stitch_alpha": 0.2})
        ),
    ),
    PresetStepTemplate(
        binding=PresetStepBinding(
            name="process_2",
            input_source=InputSource.PIPELINE_START,
        ),
        func_factory_factory=lambda definition: (
            lambda: _channel_preprocess_func(include_channel_4=True)
        ),
    ),
    PresetStepTemplate(
        binding=PresetStepBinding(name="assemble"),
        func_factory_factory=lambda definition: lambda: assemble_stack_cupy,
    ),
)


MFD_PRESET_DEFINITIONS: dict[MfdPresetKey, MfdPresetDefinition] = {
    MfdPresetKey.CROP_ANALYZE: MfdPresetDefinition(
        key=MfdPresetKey.CROP_ANALYZE,
        family=MfdPresetFamily.CROP_ANALYZE,
    ),
    MfdPresetKey.CROP_ANALYZE_DAPI_FITC_CY5: MfdPresetDefinition(
        key=MfdPresetKey.CROP_ANALYZE_DAPI_FITC_CY5,
        family=MfdPresetFamily.CROP_ANALYZE,
        include_cy5_cell_count=True,
    ),
    MfdPresetKey.STITCH_ASHLAR_CPU: MfdPresetDefinition(
        key=MfdPresetKey.STITCH_ASHLAR_CPU,
        family=MfdPresetFamily.STITCH,
        stitch_func=ashlar_compute_tile_positions_cpu,
    ),
    MfdPresetKey.STITCH_GPU: MfdPresetDefinition(
        key=MfdPresetKey.STITCH_GPU,
        family=MfdPresetFamily.STITCH,
        stitch_func=ashlar_compute_tile_positions_gpu,
    ),
}


class MfdPresetMaterializer(ABC, metaclass=AutoRegisterMeta):
    """Nominal strategy family for materializing MFD preset families."""

    __registry_key__ = "family"
    __skip_if_no_key__ = True
    __registry__: dict[MfdPresetFamily, type["MfdPresetMaterializer"]] = {}

    family: MfdPresetFamily | None = None

    @classmethod
    def for_family(cls, family: MfdPresetFamily) -> "MfdPresetMaterializer":
        return cls.__registry__[family]()

    @abstractmethod
    def materialize(self, definition: MfdPresetDefinition) -> list[FunctionStep]:
        """Materialize the preset definition into FunctionStep objects."""


class CropAnalyzeMfdPresetMaterializer(MfdPresetMaterializer):
    """Materialize MFD crop/analyze preset variants."""

    family = MfdPresetFamily.CROP_ANALYZE

    def materialize(self, definition: MfdPresetDefinition) -> list[FunctionStep]:
        return PresetPipelineSpec(
            name=definition.key.value,
            steps=(
                PresetStepSpec(
                    binding=PresetStepBinding(
                        name="crop_device",
                        variable_components=(VariableComponents.CHANNEL,),
                    ),
                    func_factory=_template_crop_func,
                ),
                PresetStepSpec(
                    binding=PresetStepBinding(name="crop_compartments"),
                    func_factory=_crop_compartments_func,
                ),
                PresetStepSpec(
                    binding=PresetStepBinding(name="analysis"),
                    func_factory=lambda: _analysis_func(
                        include_cy5_cell_count=definition.include_cy5_cell_count
                    ),
                ),
            ),
        ).materialize()


class StitchMfdPresetMaterializer(MfdPresetMaterializer):
    """Materialize MFD stitch preset variants."""

    family = MfdPresetFamily.STITCH

    def materialize(self, definition: MfdPresetDefinition) -> list[FunctionStep]:
        return PresetPipelineSpec(
            name=definition.key.value,
            steps=tuple(
                template.materialize_spec(definition)
                for template in STITCH_STEP_TEMPLATES
            ),
        ).materialize()


def build_mfd_preset(key: MfdPresetKey | str) -> list[FunctionStep]:
    """Materialize an MFD preset by typed key."""
    preset_key = MfdPresetKey(key)
    definition = MFD_PRESET_DEFINITIONS[preset_key]
    return MfdPresetMaterializer.for_family(definition.family).materialize(definition)
