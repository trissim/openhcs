import zipfile
from multiprocessing import SimpleQueue
from pathlib import Path

import numpy as np
import tifffile
from objectstate import ObjectStateRegistry
from openhcs.config_framework.lazy_factory import ensure_global_config_context

from openhcs.constants import AllComponents, GroupBy, Microscope, VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.callable_contract import CallableContract
from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.function_patterns import get_core_callable
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.progress import set_progress_queue
from openhcs.processing.backends.analysis.neurite_outgrowth import (
    MetaXpressCellBodySettings,
    MetaXpressNuclearSettings,
    MetaXpressOutgrowthSettings,
    neurite_outgrowth_metaxpress,
)
from openhcs.processing.backends.processors.numpy_processor import percentile_normalize
from openhcs.processing.presets.demo_contribution import PipelineDemoContribution
from openhcs.processing.presets.pipelines.neuroncyto_ii_crossover_neurite_outgrowth import (  # noqa: E501
    NeuronCytoIICrossoverInputs,
    build_neuroncyto_ii_crossover_demo,
    enhance_neurite_channel_gamma,
    neuroncyto_ii_crossover_demo_contribution,
)


def _inputs(plate_path: Path, output_root: Path) -> NeuronCytoIICrossoverInputs:
    return NeuronCytoIICrossoverInputs(
        plate_path=plate_path,
        output_root=output_root,
        image_id="Image15",
        neurite_filename="15_w1.tif",
        soma_nuclei_filename="15_w2.tif",
        viewer_port=5999,
    )


def test_neuroncyto_demo_declares_exact_crossover_channel_semantics(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path / "CrossOvers_Images", tmp_path / "output")
    pipeline_config, steps = build_neuroncyto_ii_crossover_demo(inputs)

    assert pipeline_config.microscope is Microscope.SOURCE_BINDINGS
    assert pipeline_config.well_filter_config.well_filter == "Image15"
    assert pipeline_config.path_planning_config.well_filter == 0
    assert pipeline_config.materialize_runtime_artifacts is True
    assert pipeline_config.materialization_results_path == (
        inputs.output_root.resolve() / "results"
    )

    bindings = pipeline_config.source_bindings_config.bindings
    assert [binding.alias for binding in bindings] == [
        "NeuriteCellSignal",
        "SomaNucleiSignal",
    ]
    assert [binding.selector.filters[0].value for binding in bindings] == [
        "15_w1.tif",
        "15_w2.tif",
    ]
    assert [
        next(
            selector.value
            for selector in binding.component_identity
            if selector.component is AllComponents.CHANNEL
        )
        for binding in bindings
    ] == ["1", "2"]

    assert len(steps) == 1
    step = steps[0]
    assert get_core_callable(step.func) is neurite_outgrowth_metaxpress
    assert step.processing_config.variable_components == [VariableComponents.CHANNEL]
    assert step.processing_config.group_by is GroupBy.NONE
    assert step.processing_config.input_source is InputSource.PIPELINE_START
    assert step.napari_streaming_config.enabled is True
    assert step.napari_streaming_config.persistent is True
    assert step.napari_streaming_config.port == 5999

    kwargs = step.func[1]
    assert kwargs["neurite_channel_index"] == 0
    assert kwargs["cell_body"] == MetaXpressCellBodySettings(
        approximate_max_width=30.0,
        minimum_area=20.0,
        intensity_above_local_background=20.0,
        channel_index=1,
    )
    assert kwargs["outgrowth"] == MetaXpressOutgrowthSettings(
        maximum_width=3.0,
        intensity_above_local_background=2.0,
        minimum_cell_growth_to_log_as_significant=10.0,
    )
    assert kwargs["use_nuclear_stain"] is True
    assert kwargs["nuclear_stain"] == MetaXpressNuclearSettings(
        channel_index=1,
        approx_min_width=3.0,
        approx_max_width=30.0,
        intensity_above_local_background=20.0,
    )
    assert CallableContract.from_callable(
        neurite_outgrowth_metaxpress
    ).artifact_outputs.names() == (
        "neurite_outgrowth_summary",
        "neurite_outgrowth_cells",
        "cell_bodies",
        "neurite_outgrowth",
        "neurons",
        "nuclei",
        "neurite_morphology",
    )


def test_neuroncyto_demo_compiles_exact_loose_tiff_pair(tmp_path: Path) -> None:
    plate_path = tmp_path / "CrossOvers_Images"
    plate_path.mkdir()
    inputs = _inputs(plate_path, tmp_path / "output")
    tifffile.imwrite(plate_path / inputs.neurite_filename, np.ones((32, 32), np.uint8))
    tifffile.imwrite(
        plate_path / inputs.soma_nuclei_filename,
        np.ones((32, 32), np.uint8),
    )

    pipeline_config, steps = build_neuroncyto_ii_crossover_demo(inputs)
    ObjectStateRegistry.clear()
    set_progress_queue(SimpleQueue())
    try:
        ensure_global_config_context(
            GlobalPipelineConfig,
            GlobalPipelineConfig(num_workers=1),
        )
        orchestrator = PipelineOrchestrator(
            plate_path,
            pipeline_config=pipeline_config,
        ).initialize()
        compilation = orchestrator.compile_pipelines(
            pipeline_definition=steps,
            well_filter=[inputs.image_id],
            is_zmq_execution=True,
        )
    finally:
        set_progress_queue(None)

    context = compilation["execution_bundle"].runtime_contexts[inputs.image_id]
    plan = context.step_plans[0]
    assert plan.step_name == "NeuronCyto II Crossover Neurite Outgrowth"
    assert tuple(plan.variable_components) == (VariableComponents.CHANNEL,)
    assert plan.compiled_function_pattern is not None


def test_neuroncyto_demo_contributor_prepares_only_declared_pair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    for filename in ("15_w1.tif", "15_w2.tif"):
        tifffile.imwrite(source_root / filename, np.ones((8, 8), np.uint8))
    archive_path = tmp_path / "Testing image.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for filename in ("15_w1.tif", "15_w2.tif"):
            archive.write(
                source_root / filename,
                f"Testing image/CrossOvers_Images/{filename}",
            )
        archive.writestr("Testing image/CrossOvers_Images/35_w1.tif", b"unused")
    monkeypatch.setenv("OPENHCS_NEURONCYTO_II_TEST_ARCHIVE", str(archive_path))

    contribution = neuroncyto_ii_crossover_demo_contribution(
        session_root=tmp_path / "session"
    )
    assert isinstance(contribution, PipelineDemoContribution)
    assert contribution.demo_id == "neuroncyto_ii_crossover_neurite_outgrowth"
    assert contribution.title == "NeuronCyto II crossover neurite morphology"
    assert contribution.plate_path.name == (
        "NeuronCyto II crossover neurite morphology"
    )
    assert len(contribution.pipeline_steps) == 3
    contrast_step, enhancement_step, analysis_step = contribution.pipeline_steps
    assert get_core_callable(contrast_step.func) is percentile_normalize
    assert contrast_step.processing_config.input_source is InputSource.PIPELINE_START
    assert get_core_callable(enhancement_step.func) is enhance_neurite_channel_gamma
    assert enhancement_step.processing_config.input_source is InputSource.PREVIOUS_STEP
    assert enhancement_step.func[1] == {"channel_index": 0, "gamma": 0.6}
    assert get_core_callable(analysis_step.func) is neurite_outgrowth_metaxpress
    assert analysis_step.processing_config.input_source is InputSource.PREVIOUS_STEP
    assert contribution.presentation_identity.output_key == "neurite_morphology"
    assert contribution.presentation_identity.artifact_kind == "spatial_graph"
    assert contribution.presentation_identity.step_name == analysis_step.name
    assert len(contribution.supporting_presentation_identities) == 1
    supporting = contribution.supporting_presentation_identities[0]
    assert supporting.output_kind == "main"
    assert supporting.output_key == "main"
    assert supporting.step_name == enhancement_step.name
    contrast_callable, contrast_kwargs = contrast_step.func
    assert contrast_callable.__name__ == "percentile_normalize"
    assert contrast_kwargs == {
        "low_percentile": 1.0,
        "high_percentile": 99.8,
        "target_max": 255.0,
    }
    assert contribution.prepare is not None

    contribution.prepare()

    assert sorted(path.name for path in contribution.plate_path.iterdir()) == [
        "15_w1.tif",
        "15_w2.tif",
    ]

    ObjectStateRegistry.clear()
    set_progress_queue(SimpleQueue())
    try:
        ensure_global_config_context(
            GlobalPipelineConfig,
            GlobalPipelineConfig(num_workers=1),
        )
        orchestrator = PipelineOrchestrator(
            contribution.plate_path,
            pipeline_config=contribution.pipeline_config,
        ).initialize()
        compilation = orchestrator.compile_pipelines(
            pipeline_definition=list(contribution.pipeline_steps),
            well_filter=["Image15"],
            is_zmq_execution=True,
        )
    finally:
        set_progress_queue(None)

    context = compilation["execution_bundle"].runtime_contexts["Image15"]
    assert [
        context.step_plans[index].step_name for index in range(len(context.step_plans))
    ] == [
        "Percentile-normalized raw neuron signals",
        "Neurite-channel gamma enhancement",
        "NeuronCyto II Crossover Neurite Outgrowth",
    ]


def test_neuroncyto_gamma_enhancement_changes_only_declared_neurite_channel() -> None:
    image = np.array(
        [
            [[0, 16], [64, 255]],
            [[0, 16], [64, 255]],
        ],
        dtype=np.uint8,
    )

    enhanced = enhance_neurite_channel_gamma(image, channel_index=0, gamma=0.6)

    assert enhanced.dtype == image.dtype
    assert np.array_equal(enhanced[1], image[1])
    assert np.all(enhanced[0] >= image[0])
    assert np.any(enhanced[0] > image[0])
