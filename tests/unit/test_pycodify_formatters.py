"""Tests for OpenHCS pycodify formatter extensions."""

import ast
import inspect

from pycodify import Assignment, generate_python_source

import openhcs.serialization.pycodify_formatters  # noqa: F401
from openhcs.config_framework.object_state import ObjectState
from openhcs.constants import InputSource
from openhcs.constants.constants import GroupBy
from openhcs.core.config import (
    DtypeConfig,
    LazyNapariStreamingConfig,
    LazyProcessingConfig,
    LazyWellFilterConfig,
    PipelineConfig,
)
from openhcs.core.memory import DtypeConversion
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.cellprofiler.colocalization import (
    measure_colocalization_objects,
)
from openhcs.processing.backends.analysis.count_cells_simple import count_cells_simple


def configurable_test_function(image, threshold: int = 3, enabled: bool = True):
    return image


def _source(value, *, clean_mode: bool = True) -> str:
    return generate_python_source(
        Assignment("config", value),
        clean_mode=clean_mode,
    )


def test_clean_pipeline_config_omits_empty_inherited_lazy_config_groups():
    source = _source(PipelineConfig())

    assert "config = PipelineConfig()" in source
    assert "step_source_bindings_config=" not in source
    assert "LazyWellFilterConfig" not in source
    assert "napari_display_config=" not in source
    assert "napari_streaming_config=" not in source


def test_clean_pipeline_config_keeps_explicit_nested_lazy_config_values():
    source = _source(
        PipelineConfig(
            well_filter_config=LazyWellFilterConfig(well_filter=1),
            napari_streaming_config=LazyNapariStreamingConfig(enabled=True),
        )
    )

    assert "well_filter_config=LazyWellFilterConfig(" in source
    assert "well_filter=1" in source
    assert "napari_streaming_config=LazyNapariStreamingConfig(" in source
    assert "enabled=True" in source
    assert "napari_display_config=" not in source
    assert "fiji_streaming_config=" not in source


def test_clean_pipeline_config_keeps_explicit_empty_nested_lazy_config():
    source = _source(PipelineConfig(well_filter_config=LazyWellFilterConfig()))

    assert "well_filter_config=LazyWellFilterConfig()" in source


def test_dtype_conversion_serializes_from_owning_enum_module():
    source = _source(DtypeConfig(default_dtype_conversion=DtypeConversion.UINT8))

    assert "from arraybridge.decorators import DtypeConversion" in source
    assert "default_dtype_conversion=DtypeConversion.UINT8" in source


def test_full_pipeline_config_still_emits_empty_lazy_config_groups():
    source = _source(PipelineConfig(), clean_mode=False)

    assert "napari_display_config=LazyNapariDisplayConfig(" in source
    assert "well_filter_config=LazyWellFilterConfig(" in source
    assert "well_filter=None" in source


def test_clean_function_step_omits_objectstate_reconstructed_empty_lazy_configs():
    step = ObjectState(
        FunctionStep(func=count_cells_simple, name="count_cells")
    ).to_object()

    source = generate_python_source(
        Assignment("pipeline_steps", [step]),
        clean_mode=True,
    )

    assert "func=count_cells_simple" in source
    assert "name='count_cells'" in source
    assert "dtype_config=" not in source
    assert "processing_config=" not in source
    assert "napari_streaming_config=" not in source
    assert "source_bindings=" not in source


def test_clean_function_step_keeps_nondefault_lazy_config_values():
    step = FunctionStep(
        func=count_cells_simple,
        name="count_cells",
        processing_config=LazyProcessingConfig(group_by=GroupBy.CHANNEL),
    )

    source = generate_python_source(
        Assignment("pipeline_steps", [step]),
        clean_mode=True,
    )

    assert "processing_config=LazyProcessingConfig(" in source
    assert "group_by=GroupBy.CHANNEL" in source
    assert "dtype_config=" not in source


def test_clean_lazy_dataclass_omits_explicit_none_leaf_fields():
    step = FunctionStep(
        func=count_cells_simple,
        name="count_cells",
        processing_config=LazyProcessingConfig(
            variable_components=[],
            group_by=None,
            input_source=None,
        ),
    )

    source = generate_python_source(
        Assignment("pipeline_steps", [step]),
        clean_mode=True,
    )

    assert "processing_config=LazyProcessingConfig(" in source
    assert "variable_components=[]" in source
    assert "group_by=None" not in source
    assert "input_source=None" not in source


def test_full_lazy_dataclass_keeps_none_leaf_fields():
    source = _source(
        LazyProcessingConfig(input_source=InputSource.PREVIOUS_STEP),
        clean_mode=False,
    )

    assert "variable_components=None" in source
    assert "group_by=None" in source
    assert "input_source=InputSource.PREVIOUS_STEP" in source


def test_function_pattern_clean_mode_elides_signature_default_kwargs():
    source = generate_python_source(
        Assignment(
            "pattern",
            (
                configurable_test_function,
                {"threshold": 9, "enabled": True},
            ),
        ),
        clean_mode=True,
    )

    ast.parse(source)
    assert "'threshold': 9" in source
    assert "enabled" not in source


def test_full_function_pattern_source_preserves_signature_default_kwargs():
    source = generate_python_source(
        Assignment(
            "pattern",
            (
                configurable_test_function,
                {"threshold": 9, "enabled": True},
            ),
        ),
        clean_mode=False,
    )

    ast.parse(source)
    assert "'threshold': 9" in source
    assert "'enabled': True" in source


def test_function_pattern_source_does_not_emit_declared_hidden_parameters():
    rank_provider_default = (
        inspect.signature(measure_colocalization_objects)
        .parameters["rank_provider"]
        .default
    )
    step = FunctionStep(
        func=(
            measure_colocalization_objects,
            {
                "labels": None,
                "rank_provider": rank_provider_default,
            },
        ),
        name="MeasureColocalization",
    )

    source = generate_python_source(
        Assignment("pipeline_steps", [step]),
        clean_mode=False,
    )

    ast.parse(source)
    assert "'labels': None" in source
    assert "rank_provider" not in source
    assert "DirectObjectColocalizationRankProvider" not in source


def test_cellprofiler_public_callable_source_keeps_hidden_parameters_hidden():
    rank_provider_default = (
        inspect.signature(measure_colocalization_objects)
        .parameters["rank_provider"]
        .default
    )
    step = FunctionStep(
        func=(
            measure_colocalization_objects,
            {
                "labels": None,
                "rank_provider": rank_provider_default,
            },
        ),
        name="MeasureColocalization",
    )

    source = generate_python_source(
        Assignment("pipeline_steps", [step]),
        clean_mode=False,
    )

    ast.parse(source)
    assert "CellProfilerModuleExecutor" not in source
    assert "ModuleArtifactContract(" not in source
    assert "MeasureColocalization_14_measurements" not in source
    assert "measure_colocalization_objects" in source
    assert "rank_provider" not in source
    assert "DirectObjectColocalizationRankProvider" not in source
