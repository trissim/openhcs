"""Bounded axon and neuronal-cell-body demo for the repository CZI.

The source container is about 19.5 GB, so this contribution never calls the
whole-plane ``load`` path.  The registered Bio-Formats source-plane authority
first declares exact scene/channel identities, and the preparation hook samples
only one full-resolution 768 x 768 field from three declared scalar planes.
That small crop is projected as a manifest-backed Bio-Formats plate while
preserving the source pixel calibration and channel labels.

The neuronal-cell-body analysis uses Rhodamine cytoplasm candidates confirmed
by DAPI nuclei.  A separate Skan analysis exposes all signal-supported Cy5 axon
branches in the tissue field.  The two result identities deliberately remain
independent because processes in a tissue section need not belong to a soma in
the crop.  Both analyses stream their meaningful outputs to Napari and
materialize their typed artifacts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from polystore.base import (
    ImageSamplingRequest,
    ensure_storage_registry,
    storage_registry,
)
from polystore.filemanager import FileManager
from polystore.streaming.identity import StreamProducerIdentity

from openhcs.constants import AllComponents, GroupBy, Microscope, VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.artifacts import ArtifactSpec
from openhcs.core.config import (
    LazyNapariStreamingConfig,
    LazyPathPlanningConfig,
    LazyProcessingConfig,
    LazySourceBindingsConfig,
    LazyStepSourceBindingsConfig,
    LazyWellFilterConfig,
    PipelineConfig,
)
from openhcs.core.function_patterns import get_core_callable
from openhcs.core.image_file_serialization import ImageFileFormat
from openhcs.core.source_bindings import (
    ComponentSelector,
    NamedSourceBinding,
    SourceSelector,
)
from openhcs.core.source_projection import SourceCandidate, SourcePlaneDataset
from openhcs.core.steps.function_output_manifest import (
    FunctionStepOutputProducerIdentityRequest,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.microscopes.bioformats import BioFormatsHandler, BioFormatsMetadataHandler
from openhcs.microscopes.bioformats_adapter import BIOFORMATS_MANIFEST_FILENAME
from openhcs.processing.backends.analysis.neurite_outgrowth import (
    NEURONAL_CELL_BODIES_OUTPUT,
    MetaXpressCellBodySettings,
    MetaXpressNuclearSettings,
    NeuriteIllumination,
    count_neuronal_cell_bodies_metaxpress,
)
from openhcs.processing.backends.analysis.skan_axon_analysis import (
    SKELETON_VISUALIZATIONS_OUTPUT,
    AnalysisDimension,
    ThresholdMethod,
    skan_axon_skeletonize_and_analyze,
)
from openhcs.processing.backends.processors.numpy_processor import tophat
from openhcs.processing.presets.demo_contribution import PipelineDemoContribution


@dataclass(frozen=True, slots=True)
class CziBrainCropSelection:
    """Exact source-declared field and calibrated crop used by the demo."""

    site_label: str = "ScanRegion0"
    origin_yx: tuple[int, int] = (19_000, 30_000)
    shape_yx: tuple[int, int] = (768, 768)
    pixel_size: float = 0.325
    axon_channel_label: str = "Cy5 (Red Laser)"
    soma_channel_label: str = "Rhodamine (Green Laser)"
    nuclei_channel_label: str = "DAPI (UV Laser)"

    @property
    def ordered_channel_labels(self) -> tuple[str, str, str]:
        """Return the owned stack order consumed by the two assay callables."""

        return (
            self.axon_channel_label,
            self.soma_channel_label,
            self.nuclei_channel_label,
        )


@dataclass(frozen=True, slots=True)
class CziBrainAxonCellBodyInputs:
    """Prepared bounded plate and output boundary for the CZI assay."""

    plate_path: Path
    output_root: Path
    viewer_port: int = 6008
    pixel_size: float = 0.325


def _analysis_stream(
    inputs: CziBrainAxonCellBodyInputs,
    colormap: str,
) -> LazyNapariStreamingConfig:
    return LazyNapariStreamingConfig(
        enabled=True,
        persistent=True,
        port=inputs.viewer_port,
        colormap=colormap,
    )


def build_czi_brain_axon_cellbody_demo(
    inputs: CziBrainAxonCellBodyInputs,
) -> tuple[PipelineConfig, list[FunctionStep]]:
    """Build independent neuronal-soma and full-field axon analyses."""

    output_root = inputs.output_root.expanduser().resolve()
    pipeline_config = PipelineConfig(
        microscope=Microscope.BIOFORMATS,
        source_bindings_config=LazySourceBindingsConfig(
            bindings=(
                NamedSourceBinding(
                    alias="Axon",
                    selector=SourceSelector(
                        components=(ComponentSelector(AllComponents.CHANNEL, "1"),),
                    ),
                ),
                NamedSourceBinding(
                    alias="NeuronalSoma",
                    selector=SourceSelector(
                        components=(ComponentSelector(AllComponents.CHANNEL, "2"),),
                    ),
                ),
                NamedSourceBinding(
                    alias="Nuclei",
                    selector=SourceSelector(
                        components=(ComponentSelector(AllComponents.CHANNEL, "3"),),
                    ),
                ),
            ),
        ),
        well_filter_config=LazyWellFilterConfig(well_filter="A01"),
        path_planning_config=LazyPathPlanningConfig(
            well_filter=0,
            global_output_folder=output_root,
        ),
        materialization_results_path=output_root / "results",
        materialize_runtime_artifacts=True,
    )
    steps = [
        FunctionStep(
            name="CZI Tissue Background Correction",
            func=(
                tophat,
                {
                    "selem_radius": 50,
                    "downsample_factor": 2,
                    "downsample_anti_aliasing": True,
                    "upsample_order": 0,
                },
            ),
            processing_config=LazyProcessingConfig(
                variable_components=[VariableComponents.CHANNEL],
                group_by=GroupBy.NONE,
                input_source=InputSource.PIPELINE_START,
            ),
        ),
        FunctionStep(
            name="CZI Neuronal Cell-Body Count",
            func=(
                count_neuronal_cell_bodies_metaxpress,
                {
                    "illumination": NeuriteIllumination.FLUORESCENCE,
                    "cell_body": MetaXpressCellBodySettings(
                        approximate_max_width=25.0,
                        minimum_area=40.0,
                        intensity_above_local_background=800.0,
                        channel_index=1,
                    ),
                    "nuclear_stain": MetaXpressNuclearSettings(
                        channel_index=2,
                        approx_min_width=4.0,
                        approx_max_width=16.0,
                        intensity_above_local_background=500.0,
                    ),
                },
            ),
            processing_config=LazyProcessingConfig(
                variable_components=[VariableComponents.CHANNEL],
                group_by=GroupBy.NONE,
                input_source=InputSource.PREVIOUS_STEP,
            ),
            napari_streaming_config=_analysis_stream(
                inputs,
                "magma",
            ),
        ),
        FunctionStep(
            name="CZI Axon Background Correction",
            func=(
                tophat,
                {
                    "selem_radius": 50,
                    "downsample_factor": 2,
                    "downsample_anti_aliasing": True,
                    "upsample_order": 0,
                },
            ),
            processing_config=LazyProcessingConfig(
                variable_components=[VariableComponents.CHANNEL],
                group_by=GroupBy.NONE,
                input_source=InputSource.PIPELINE_START,
            ),
            source_bindings=LazyStepSourceBindingsConfig(
                enabled=True,
                bindings=(NamedSourceBinding(alias="Axon"),),
            ),
        ),
        FunctionStep(
            name="CZI Full-Field Axon Network",
            func=(
                skan_axon_skeletonize_and_analyze,
                {
                    "voxel_spacing": (
                        1.0,
                        inputs.pixel_size,
                        inputs.pixel_size,
                    ),
                    "threshold_method": ThresholdMethod.MANUAL,
                    "threshold_value": 6000.0,
                    "min_object_size": 16,
                    "min_branch_length": 3.0,
                    "return_skeleton_visualizations": True,
                    "analysis_dimension": AnalysisDimension.TWO_D,
                },
            ),
            processing_config=LazyProcessingConfig(
                variable_components=[VariableComponents.CHANNEL],
                group_by=GroupBy.NONE,
                input_source=InputSource.PREVIOUS_STEP,
            ),
            napari_streaming_config=_analysis_stream(
                inputs,
                "gray",
            ),
        ),
    ]
    return pipeline_config, steps


def _source_filemanager(source_root: Path, workspace_root: Path) -> FileManager:
    ensure_storage_registry()
    filemanager = FileManager(dict(storage_registry))
    workspace_root.mkdir(parents=True, exist_ok=True)
    source_entry = workspace_root / source_root.name
    if source_entry.exists():
        if source_entry.is_symlink() or not source_entry.samefile(source_root):
            raise FileExistsError(
                f"Source workspace path {source_entry} must be a hard link to "
                f"{source_root}."
            )
    else:
        source_entry.hardlink_to(source_root)
    BioFormatsHandler(filemanager).initialize_workspace(workspace_root, filemanager)
    return filemanager


def _declared_crop_candidates(
    dataset: SourcePlaneDataset,
    selection: CziBrainCropSelection,
) -> tuple[SourceCandidate, ...]:
    """Resolve one exact plane per role from store-owned labels and addresses."""

    selected: list[SourceCandidate] = []
    for channel_label in selection.ordered_channel_labels:
        matches = tuple(
            candidate
            for candidate in dataset.candidates
            if candidate.component_labels.get(AllComponents.SITE.value)
            == selection.site_label
            and candidate.component_labels.get(AllComponents.CHANNEL.value)
            == channel_label
            and candidate.declared_address is not None
            and candidate.declared_address.z_index == "1"
            and candidate.declared_address.timepoint == "1"
        )
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {channel_label!r} plane in "
                f"{selection.site_label!r} at z=1, t=1; found {len(matches)}."
            )
        selected.append(matches[0])
    return tuple(selected)


def _prepared_manifest(
    *,
    title: str,
    source_path: Path,
    selection: CziBrainCropSelection,
    pixel_size: float,
) -> dict[str, object]:
    planes = [
        {"c": index, "z": 1, "t": 1, "index": index - 1}
        for index in range(1, len(selection.ordered_channel_labels) + 1)
    ]
    return {
        "plates": [
            {
                "plate_id": "Plate:czi-brain-axon-cellbody",
                "name": title,
                "wells": [
                    {
                        "well_id": "Well:0:0",
                        "row": 0,
                        "column": 0,
                        "samples": [
                            {
                                "sample_id": "WellSample:0:0:0",
                                "image_id": "Image:czi-brain-crop",
                                "index": 0,
                            }
                        ],
                    }
                ],
            }
        ],
        "images": [
            {
                "image_id": "Image:czi-brain-crop",
                "image_name": (
                    f"{source_path.name}: {selection.site_label} "
                    f"y{selection.origin_yx[0]} x{selection.origin_yx[1]} "
                    f"{selection.shape_yx[0]}x{selection.shape_yx[1]}"
                ),
                "source_path": "stack.npy",
                "series_index": 0,
                "reader": "npy",
                "channel_names": list(selection.ordered_channel_labels),
                "pixel_size": pixel_size,
                "pixels": {
                    "size_c": len(selection.ordered_channel_labels),
                    "size_z": 1,
                    "size_t": 1,
                    "planes": planes,
                },
            }
        ],
    }


def czi_brain_axon_cellbody_demo_contribution(
    *,
    session_root: Path,
    source_path: Path | None = None,
    selection: CziBrainCropSelection = CziBrainCropSelection(),
) -> PipelineDemoContribution:
    """Contribute one bounded field from the repository's large CZI."""

    demo_id = "czi_brain_axon_cellbody"
    title = "CZI brain axons and neuronal cell bodies"
    default_source = (
        Path(__file__).resolve().parents[4] / "58-36 GFP-NeuN-CS 7.2.24.czi"
    )
    resolved_source = (
        Path(
            source_path
            if source_path is not None
            else os.environ.get("OPENHCS_CZI_AXON_SOURCE", default_source)
        )
        .expanduser()
        .resolve()
    )
    if not resolved_source.is_file():
        raise FileNotFoundError(
            f"CZI axon demo source not found at {resolved_source}. Set "
            "OPENHCS_CZI_AXON_SOURCE to the intended container."
        )

    resolved_session_root = session_root.expanduser().resolve()
    plate_path = resolved_session_root / "plates" / title
    output_root = resolved_session_root / "outputs" / demo_id
    source_workspace = resolved_session_root / "source_workspaces" / demo_id

    def prepare() -> None:
        dataset = BioFormatsMetadataHandler().source_dataset(resolved_source)
        if not np.isclose(dataset.pixel_size, selection.pixel_size):
            raise ValueError(
                f"Source pixel size {dataset.pixel_size} conflicts with the "
                f"declared demo calibration {selection.pixel_size}."
            )
        candidates = _declared_crop_candidates(dataset, selection)
        request = ImageSamplingRequest(
            origin_yx=selection.origin_yx,
            shape_yx=selection.shape_yx,
            resolution_index=0,
        )
        filemanager = _source_filemanager(resolved_source, source_workspace)
        sampled_planes = []
        sampled_results = []
        for candidate in candidates:
            sampled = filemanager.sample(
                candidate.source_ref.backend_address,
                candidate.source_ref.backend,
                request,
            )
            plane = np.asarray(sampled.data)
            if plane.shape != selection.shape_yx:
                raise ValueError(
                    f"Bounded sample returned {plane.shape}, expected "
                    f"{selection.shape_yx}."
                )
            sampled_planes.append(plane)
            sampled_results.append(sampled)

        selected_resolution_indexes = {
            int(sampled.selected_resolution_index) for sampled in sampled_results
        }
        if selected_resolution_indexes != {request.resolution_index}:
            raise ValueError(
                "Prepared CZI planes did not all use the requested native "
                f"resolution level {request.resolution_index}: "
                f"{sorted(selected_resolution_indexes)}."
            )
        plate_path.mkdir(parents=True, exist_ok=True)
        stack = np.stack(sampled_planes, axis=0)[None, None]
        stack_path = plate_path / "stack.npy"
        ImageFileFormat.require_path(stack_path).write(stack_path, stack)
        manifest = _prepared_manifest(
            title=title,
            source_path=resolved_source,
            selection=selection,
            pixel_size=dataset.pixel_size,
        )
        (plate_path / BIOFORMATS_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    inputs = CziBrainAxonCellBodyInputs(
        plate_path=plate_path,
        output_root=output_root,
        viewer_port=6008,
        pixel_size=selection.pixel_size,
    )
    pipeline_config, pipeline_steps = build_czi_brain_axon_cellbody_demo(inputs)
    soma_matches = tuple(
        step
        for step in pipeline_steps
        if get_core_callable(step.func) is count_neuronal_cell_bodies_metaxpress
    )
    axon_matches = tuple(
        step
        for step in pipeline_steps
        if get_core_callable(step.func) is skan_axon_skeletonize_and_analyze
    )
    if len(soma_matches) != 1 or len(axon_matches) != 1:
        raise ValueError(
            "CZI brain demo requires exactly one neuronal-cell-body analysis "
            "and one full-field axon analysis."
        )
    soma_step = soma_matches[0]
    axon_step = axon_matches[0]

    def artifact_presentation(
        step: FunctionStep,
        output_spec: ArtifactSpec,
    ) -> StreamProducerIdentity:
        return StreamProducerIdentity.pipeline_output(
            output_kind=(
                FunctionStepOutputProducerIdentityRequest.ARTIFACT_OUTPUT_KIND
            ),
            output_key=output_spec.name,
            projection_key=output_spec.name,
            step_name=step.name,
            pipeline_position=None,
            artifact_kind=output_spec.artifact_type.require_value(),
        )

    return PipelineDemoContribution(
        demo_id=demo_id,
        title=title,
        plate_path=plate_path,
        pipeline_config=pipeline_config,
        pipeline_steps=tuple(pipeline_steps),
        presentation_identity=artifact_presentation(
            axon_step,
            SKELETON_VISUALIZATIONS_OUTPUT,
        ),
        supporting_presentation_identities=(
            artifact_presentation(
                soma_step,
                NEURONAL_CELL_BODIES_OUTPUT,
            ),
        ),
        prepare=prepare,
        biological_question=(
            "Where do neuronal cell bodies and full-field axon branches occupy "
            "this bounded brain-tissue field?"
        ),
    )


example_inputs = CziBrainAxonCellBodyInputs(
    plate_path=Path("path/to/prepared_czi_brain_crop"),
    output_root=Path("openhcs_czi_brain_axon_output"),
    viewer_port=6008,
)

plate_path = example_inputs.plate_path.expanduser().resolve()
pipeline_config, pipeline_steps = build_czi_brain_axon_cellbody_demo(example_inputs)
