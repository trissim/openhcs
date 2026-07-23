"""NeuronCyto II crossover-image neurite-outgrowth demonstration.

The public NeuronCyto II testing archive contains paired, loose TIFF files
without microscope plate metadata.  In ``CrossOvers_Images`` the ``*_w1.tif``
plane is the neuronal cell/neurite signal and ``*_w2.tif`` is the soma/nucleus
signal.  The separate non-crossover folder uses a different plane convention,
so this preset deliberately accepts both exact filenames instead of inferring
biological meaning from ``w1`` or ``w2`` globally.

The compact analysis streams the final source planes, cell bodies, owned
neurites, unified neurons, nuclei, morphology graph, and live measurement
tables to Napari.  Typed measurements, ROI labels, graph ROI paths, and SWC are
also materialized.  The top-level well filter bounds loading to the selected
field; the path-planning filter suppresses an otherwise redundant final image
copy while runtime artifacts remain materialized.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from polystore.streaming.identity import StreamProducerIdentity
from skimage.exposure import adjust_gamma

from openhcs.constants import AllComponents, GroupBy, Microscope, VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.aligned_image_payload import AlignedImageSliceContext
from openhcs.core.config import (
    LazyNapariStreamingConfig,
    LazyPathPlanningConfig,
    LazyProcessingConfig,
    LazyWellFilterConfig,
    NapariColormap,
    PipelineConfig,
)
from openhcs.core.memory import numpy as numpy_func
from openhcs.core.source_bindings import (
    ComponentSelector,
    NamedSourceBinding,
    SourceBindingsConfig,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceSelector,
)
from openhcs.core.steps.function_output_manifest import (
    FunctionStepOutputProducerIdentityRequest,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.analysis.neurite_outgrowth import (
    NEURITE_MORPHOLOGY_OUTPUT,
    MetaXpressCellBodySettings,
    MetaXpressNuclearSettings,
    MetaXpressOutgrowthSettings,
    NeuriteIllumination,
    neurite_outgrowth_metaxpress,
)
from openhcs.processing.backends.processors.numpy_processor import percentile_normalize
from openhcs.processing.presets.demo_contribution import PipelineDemoContribution


@dataclass(frozen=True, slots=True)
class NeuronCytoIICrossoverInputs:
    """Exact local boundary for one field from ``CrossOvers_Images``."""

    plate_path: Path
    output_root: Path
    image_id: str
    neurite_filename: str
    soma_nuclei_filename: str
    viewer_port: int = 5888


@numpy_func
def enhance_neurite_channel_gamma(
    image: np.ndarray,
    channel_index: int = 0,
    gamma: float = 0.6,
) -> np.ndarray:
    """Reveal dim processes on one declared neurite plane only."""

    image_array = np.asarray(image)
    if image_array.ndim != 3:
        raise ValueError(
            f"Expected a channel stack with shape (C, Y, X), got {image_array.shape}"
        )
    if not 0 <= channel_index < image_array.shape[0]:
        raise ValueError("channel_index is outside the input stack")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and > 0")
    enhanced = image_array.copy()
    enhanced[channel_index] = adjust_gamma(
        image_array[channel_index],
        gamma=float(gamma),
    )
    return enhanced


def _exact_source_binding(
    inputs: NeuronCytoIICrossoverInputs,
    *,
    alias: str,
    filename: str,
    channel: str,
) -> NamedSourceBinding:
    """Bind one exact loose TIFF to its declared field/channel identity."""

    return NamedSourceBinding(
        alias=alias,
        selector=SourceSelector(
            filters=(
                SourceFilterClause(
                    subject=SourceFilterSubject.FILE,
                    match_type=SourceFilterMatchType.EQUALS,
                    value=filename,
                ),
            ),
        ),
        component_identity=(
            ComponentSelector(AllComponents.WELL, inputs.image_id),
            ComponentSelector(AllComponents.SITE, "1"),
            ComponentSelector(AllComponents.CHANNEL, channel),
            ComponentSelector(AllComponents.Z_INDEX, "1"),
            ComponentSelector(AllComponents.TIMEPOINT, "1"),
        ),
    )


def build_neuroncyto_ii_crossover_demo(
    inputs: NeuronCytoIICrossoverInputs,
) -> tuple[PipelineConfig, list[FunctionStep]]:
    """Build the compact two-channel crossover analysis and viewer demo."""

    output_root = inputs.output_root.expanduser().resolve()
    source_bindings = (
        _exact_source_binding(
            inputs,
            alias="NeuriteCellSignal",
            filename=inputs.neurite_filename,
            channel="1",
        ),
        _exact_source_binding(
            inputs,
            alias="SomaNucleiSignal",
            filename=inputs.soma_nuclei_filename,
            channel="2",
        ),
    )
    pipeline_config = PipelineConfig(
        microscope=Microscope.SOURCE_BINDINGS,
        well_filter_config=LazyWellFilterConfig(well_filter=inputs.image_id),
        path_planning_config=LazyPathPlanningConfig(
            well_filter=0,
            global_output_folder=output_root,
        ),
        materialization_results_path=output_root / "results",
        materialize_runtime_artifacts=True,
        source_bindings_config=SourceBindingsConfig(bindings=source_bindings),
    )
    step = FunctionStep(
        name="NeuronCyto II Crossover Neurite Outgrowth",
        func=(
            neurite_outgrowth_metaxpress,
            {
                "neurite_channel_index": 0,
                "illumination": NeuriteIllumination.FLUORESCENCE,
                "cell_body": MetaXpressCellBodySettings(
                    approximate_max_width=30.0,
                    minimum_area=20.0,
                    intensity_above_local_background=20.0,
                    channel_index=1,
                ),
                "outgrowth": MetaXpressOutgrowthSettings(
                    maximum_width=3.0,
                    intensity_above_local_background=2.0,
                    minimum_cell_growth_to_log_as_significant=10.0,
                ),
                "use_nuclear_stain": True,
                "nuclear_stain": MetaXpressNuclearSettings(
                    channel_index=1,
                    approx_min_width=3.0,
                    approx_max_width=30.0,
                    intensity_above_local_background=20.0,
                ),
            },
        ),
        processing_config=LazyProcessingConfig(
            variable_components=[VariableComponents.CHANNEL],
            group_by=GroupBy.NONE,
            input_source=InputSource.PIPELINE_START,
        ),
        napari_streaming_config=LazyNapariStreamingConfig(
            enabled=True,
            persistent=True,
            port=inputs.viewer_port,
            colormap=NapariColormap.MAGMA,
        ),
    )
    return pipeline_config, [step]


def neuroncyto_ii_crossover_demo_contribution(
    *,
    session_root: Path,
) -> PipelineDemoContribution:
    """Contribute field 15 to an explicitly requested showcase session.

    Set ``OPENHCS_NEURONCYTO_II_TEST_ARCHIVE`` to override the default
    OpenHCS dataset-cache location.  The returned preparation hook extracts
    only the two declared field planes into the caller-owned session directory;
    it never runs analysis or mutates the source archive.
    """

    demo_id = "neuroncyto_ii_crossover_neurite_outgrowth"
    default_archive = (
        Path.home()
        / ".cache"
        / "openhcs"
        / "datasets"
        / "neuroncyto_ii"
        / "Testing image.zip"
    )
    archive_path = Path(
        os.environ.get("OPENHCS_NEURONCYTO_II_TEST_ARCHIVE", default_archive)
    ).expanduser()
    if not archive_path.is_file():
        raise FileNotFoundError(
            "NeuronCyto II test archive not found at "
            f"{archive_path}. Set OPENHCS_NEURONCYTO_II_TEST_ARCHIVE to the "
            "official 'Testing image.zip' download."
        )

    resolved_session_root = session_root.expanduser().resolve()
    plate_path = (
        resolved_session_root
        / "plates"
        / ("NeuronCyto II crossover neurite morphology")
    )
    output_root = resolved_session_root / "outputs" / demo_id
    archive_members = {
        "15_w1.tif": "Testing image/CrossOvers_Images/15_w1.tif",
        "15_w2.tif": "Testing image/CrossOvers_Images/15_w2.tif",
    }

    def prepare() -> None:
        plate_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            missing = tuple(
                member
                for member in archive_members.values()
                if member not in archive.namelist()
            )
            if missing:
                raise FileNotFoundError(
                    f"NeuronCyto II archive {archive_path} is missing declared "
                    f"members: {missing}"
                )
            for filename, member in archive_members.items():
                with (
                    archive.open(member) as source,
                    (plate_path / filename).open("wb") as destination,
                ):
                    shutil.copyfileobj(source, destination)

    inputs = NeuronCytoIICrossoverInputs(
        plate_path=plate_path,
        output_root=output_root,
        image_id="Image15",
        neurite_filename="15_w1.tif",
        soma_nuclei_filename="15_w2.tif",
        viewer_port=5888,
    )
    pipeline_config, compact_steps = build_neuroncyto_ii_crossover_demo(inputs)
    contrast_step_name = "Percentile-normalized raw neuron signals"
    contrast_step = FunctionStep(
        name=contrast_step_name,
        func=(
            percentile_normalize,
            {
                "low_percentile": 1.0,
                "high_percentile": 99.8,
                "target_max": 255.0,
            },
        ),
        processing_config=LazyProcessingConfig(
            variable_components=[VariableComponents.CHANNEL],
            group_by=GroupBy.NONE,
            input_source=InputSource.PIPELINE_START,
        ),
        napari_streaming_config=LazyNapariStreamingConfig(
            enabled=True,
            persistent=True,
            port=inputs.viewer_port,
            colormap=NapariColormap.MAGMA,
        ),
    )
    enhancement_step = FunctionStep(
        name="Neurite-channel gamma enhancement",
        func=(
            enhance_neurite_channel_gamma,
            {"channel_index": 0, "gamma": 0.6},
        ),
        processing_config=LazyProcessingConfig(
            variable_components=[VariableComponents.CHANNEL],
            group_by=GroupBy.NONE,
            input_source=InputSource.PREVIOUS_STEP,
        ),
        napari_streaming_config=LazyNapariStreamingConfig(
            enabled=True,
            persistent=True,
            port=inputs.viewer_port,
            colormap=NapariColormap.MAGMA,
        ),
    )
    compact_step = compact_steps[0]
    analysis_step = FunctionStep(
        name=compact_step.name,
        func=compact_step.func,
        processing_config=LazyProcessingConfig(
            variable_components=[VariableComponents.CHANNEL],
            group_by=GroupBy.NONE,
            input_source=InputSource.PREVIOUS_STEP,
        ),
        napari_streaming_config=compact_step.napari_streaming_config,
    )
    pipeline_steps = [contrast_step, enhancement_step, analysis_step]
    return PipelineDemoContribution(
        demo_id=demo_id,
        title="NeuronCyto II crossover neurite morphology",
        plate_path=plate_path,
        pipeline_config=pipeline_config,
        pipeline_steps=tuple(pipeline_steps),
        presentation_identity=StreamProducerIdentity.pipeline_output(
            output_kind=(
                FunctionStepOutputProducerIdentityRequest.ARTIFACT_OUTPUT_KIND
            ),
            output_key=NEURITE_MORPHOLOGY_OUTPUT.name,
            projection_key=NEURITE_MORPHOLOGY_OUTPUT.name,
            step_name=analysis_step.name,
            pipeline_position=None,
            artifact_kind=NEURITE_MORPHOLOGY_OUTPUT.artifact_type.require_value(),
        ),
        supporting_presentation_identities=(
            StreamProducerIdentity.pipeline_output(
                output_kind=AlignedImageSliceContext.MAIN_FLOW_OUTPUT_KIND,
                output_key=AlignedImageSliceContext.ANONYMOUS_MAIN_FLOW_OUTPUT_KEY,
                projection_key=AlignedImageSliceContext.MAIN_FLOW_OUTPUT_KIND,
                step_name=enhancement_step.name,
                pipeline_position=None,
            ),
        ),
        prepare=prepare,
    )


# Edit this single boundary after extracting the public ``Testing image.zip``.
example_inputs = NeuronCytoIICrossoverInputs(
    plate_path=Path("path/to/Testing image/CrossOvers_Images"),
    output_root=Path("openhcs_neuroncyto_ii_output"),
    image_id="Image15",
    neurite_filename="15_w1.tif",
    soma_nuclei_filename="15_w2.tif",
    viewer_port=5888,
)

plate_path = example_inputs.plate_path.expanduser().resolve()
pipeline_config, pipeline_steps = build_neuroncyto_ii_crossover_demo(example_inputs)
