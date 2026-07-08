from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openhcs.constants.constants import AllComponents, GroupBy, VariableComponents
from openhcs.formats.pattern.pattern_discovery import PatternDiscoveryEngine
from openhcs.core.function_patterns import compile_function_pattern
from openhcs.core.steps.function_output_manifest import StepOutputManifestStore
from openhcs.core.steps.function_execution import (
    FunctionStepExecutor,
    PatternGroups,
    StepAnchorPatternFilter,
)
from openhcs.core.aligned_image_payload import AlignedImageSliceContext
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSidecarRole,
    ArtifactSpec,
    ImageArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.module_artifact_contract import (
    DeclaredArtifactOutputPartition,
    ModuleArtifactContract,
    RecordedArtifactOutputPartition,
    module_artifact_contract,
)
from openhcs.core.runtime_adapters import runtime_adapter
from openhcs.core.steps.function_output_identity import (
    FunctionOutputIdentity,
    FunctionOutputIdentityAuthority,
    FunctionOutputPathAuthority,
    FunctionOutputPathRequest,
)
from openhcs.core.steps.function_output_manifest import ProducedOutputSemantics
from openhcs.core.source_workspace_projection import (
    VirtualWorkspacePathLookup,
    VirtualWorkspaceSourceProjection,
    VirtualWorkspaceSourceProjectionCache,
)
from openhcs.core.aligned_image_payload import (
    ImagePayloadBundleContext,
    payload_slices_for_alignment,
    stack_image_payload_context,
)
from openhcs.core.pipeline_image_schema import (
    ColorImageTypeSourceRole,
    GrayscaleImageTypeSourceRole,
    SOURCE_IMAGE_TYPE_METADATA_FIELD,
)
from openhcs.core.source_image_semantics import source_image_payload_role
from openhcs.core.step_dependencies import StepInputDependency
from openhcs.core.source_bindings import (
    ComponentSelector,
    CompiledSourceBindingPlan,
    NamedSourceBinding,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingOrigin,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceSelector,
)
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    RuntimeImagePayloadContext,
    SourceImageProvenancePlanes,
    image_payload_data,
    image_payload_metadata,
)
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser


def test_pattern_discovery_uses_authoritative_virtual_source_files(
    tmp_path: Path,
) -> None:
    plate_path = tmp_path / "plate"
    source_files = [
        plate_path / "A01_s001_w1_z001_t001.tif",
        plate_path / "A01_s002_w1_z001_t001.tif",
        plate_path / "B01_s001_w1_z001_t001.tif",
    ]
    engine = PatternDiscoveryEngine(
        SourceSchemaFilenameParser(),
        SimpleNamespace(),
    )

    patterns = engine.auto_detect_patterns_from_files(
        source_files,
        variable_components=[VariableComponents.SITE.value],
        well_filter=["A01"],
    )

    assert patterns == {"A01": ["A01_s{iii}_w1_z001_t001.tif"]}


def test_pattern_discovery_accepts_axis_scoped_source_projection_files(
    tmp_path: Path,
) -> None:
    plate_path = tmp_path / "plate"
    source_files = [
        plate_path / "A01_s001_w1_z001_t001.tif",
    ]
    engine = PatternDiscoveryEngine(
        SourceSchemaFilenameParser(),
        SimpleNamespace(),
    )

    patterns = engine.auto_detect_patterns_from_axis_files(
        source_files,
        axis_id="source_projection_axis",
        variable_components=[],
    )

    assert patterns == {
        "source_projection_axis": ["A01_s001_w1_z001_t001.tif"]
    }


def test_virtual_workspace_pipeline_start_files_preserve_virtual_identity(tmp_path: Path) -> None:
    """Pipeline-start source resolution must not collapse per-well virtual files."""

    plate_path = tmp_path / "plate"
    real_path = tmp_path / "source" / "image.png"
    virtual_a = plate_path / "W001_s001_w1_z001_t001.png"
    virtual_b = plate_path / "W002_s001_w1_z001_t001.png"
    projection = VirtualWorkspaceSourceProjection(
        source_paths_by_virtual_path={
            virtual_a.name: str(real_path),
            str(virtual_a): str(real_path),
            virtual_b.name: str(real_path),
            str(virtual_b): str(real_path),
        },
        source_metadata_by_path={
            virtual_a.name: {"well": "W001"},
            virtual_b.name: {"well": "W002"},
        },
        workspace_root=str(plate_path),
    )

    assert projection.pipeline_start_files() == (str(virtual_a), str(virtual_b))
    assert projection.pipeline_start_files(axis_id="W001") == (str(virtual_a),)
    assert projection.pipeline_start_files(axis_id="W002") == (str(virtual_b),)
    assert projection.source_metadata_for(
        VirtualWorkspacePathLookup.from_paths(virtual_a.name, str(virtual_a))
    ) == {"well": "W001"}
    assert projection.source_metadata_for(
        VirtualWorkspacePathLookup.from_paths(virtual_b.name, str(virtual_b))
    ) == {"well": "W002"}


def test_virtual_workspace_axis_filter_uses_virtual_filename_metadata_when_missing(
    tmp_path: Path,
) -> None:
    plate_path = tmp_path / "plate"
    real_path = tmp_path / "source" / "image.png"
    virtual_a = plate_path / "A01_s001_w1_z001_t001.tif"
    virtual_b = plate_path / "A02_s001_w1_z001_t001.tif"
    projection = VirtualWorkspaceSourceProjection(
        source_paths_by_virtual_path={
            virtual_a.name: str(real_path),
            virtual_b.name: str(real_path),
        },
        source_metadata_by_path={},
        workspace_root=str(plate_path),
    )

    assert projection.pipeline_start_files(axis_id="A01") == (str(virtual_a),)
    assert projection.pipeline_start_files(axis_id="A02") == (str(virtual_b),)
    assert projection.source_metadata_for(
        VirtualWorkspacePathLookup.from_paths(virtual_a.name, str(virtual_a))
    ) == {
        "well": "A01",
        "site": 1,
        "channel": 1,
        "z_index": 1,
        "timepoint": 1,
        "extension": ".tif",
    }


def test_virtual_workspace_runtime_metadata_projection_validates_explicit_metadata(
    tmp_path: Path,
) -> None:
    plate_path = tmp_path / "plate"
    real_path = tmp_path / "source" / "image.tif"
    virtual_path = plate_path / "A01_s001_w1_z001_t001.tif"
    projection = VirtualWorkspaceSourceProjection(
        source_paths_by_virtual_path={
            virtual_path.name: str(real_path),
            str(virtual_path): str(real_path),
        },
        source_metadata_by_path={
            virtual_path.name: {"OpenHCSSourceVoxelSpacingZYX": "2,1,1"},
            str(virtual_path): {"OpenHCSSourceVoxelSpacingZYX": "2,1,1"},
        },
        workspace_root=str(plate_path),
    )

    projection.validate_runtime_metadata_projection()


def test_virtual_workspace_runtime_metadata_projection_rejects_path_spelling_drift(
    tmp_path: Path,
) -> None:
    plate_path = tmp_path / "plate"
    real_path = tmp_path / "source" / "image.tif"
    virtual_path = plate_path / "A01_s001_w1_z001_t001.tif"
    projection = VirtualWorkspaceSourceProjection(
        source_paths_by_virtual_path={
            virtual_path.name: str(real_path),
        },
        source_metadata_by_path={
            virtual_path.name: {"OpenHCSSourceVoxelSpacingZYX": "2,1,1"},
        },
        workspace_root=str(plate_path),
    )

    with pytest.raises(ValueError, match="OpenHCSSourceVoxelSpacingZYX"):
        projection.validate_runtime_metadata_projection()


def test_stack_payload_context_promotes_single_channel_slice_metadata() -> None:
    first = RuntimeImagePayloadContext(
        np.zeros((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes = SourceImageProvenancePlanes.from_components(paths = ("/input/A01_s001_w1_z001_t001.tif",), component_metadata = (
                {"well": "A01", "site": 1, "channel": 1},
            ))),
    mask = None).payload()
    second = RuntimeImagePayloadContext(
        np.ones((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes = SourceImageProvenancePlanes.from_components(paths = ("/input/A01_s002_w1_z001_t001.tif",), component_metadata = (
                {"well": "A01", "site": 2, "channel": 1},
            ))),
    mask = None).payload()
    stack = np.stack(
        (
            image_payload_data(first),
            image_payload_data(second),
        )
    )

    payload = stack_image_payload_context((first, second), stack)
    metadata = image_payload_metadata(payload)

    assert metadata.source_image_provenance_planes.paths == (
        "/input/A01_s001_w1_z001_t001.tif",
        "/input/A01_s002_w1_z001_t001.tif",
    )
    assert tuple(dict(item) for item in metadata.source_image_provenance_planes.component_metadata) == (
        {"well": "A01", "site": 1, "channel": 1},
        {"well": "A01", "site": 2, "channel": 1},
    )


def test_bundle_payload_context_preserves_source_binding_plane_metadata() -> None:
    first = RuntimeImagePayloadContext(
        np.zeros((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/input/A01_s001_w1_z001_t001.tif",),
                component_metadata=(
                    {"well": "A01", "site": 1, "channel": 1},
                ),
            )
        ),
        mask=None,
    ).payload()
    second = RuntimeImagePayloadContext(
        np.ones((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/input/A01_s001_w2_z001_t001.tif",),
                component_metadata=(
                    {"well": "A01", "site": 1, "channel": 2},
                ),
            )
        ),
        mask=None,
    ).payload()

    bundle = ImagePayloadBundleContext.from_payloads((first, second)).compose()
    metadata = image_payload_metadata(bundle)

    assert metadata.source_image_provenance_planes.paths == (
        "/input/A01_s001_w1_z001_t001.tif",
        "/input/A01_s001_w2_z001_t001.tif",
    )
    assert tuple(
        dict(item)
        for item in metadata.source_image_provenance_planes.component_metadata
    ) == (
        {"well": "A01", "site": 1, "channel": 1},
        {"well": "A01", "site": 1, "channel": 2},
    )
    assert dict(metadata.source_component_metadata) == {
        "well": "A01",
        "site": 1,
        "extension": ".tif",
    }


def test_mixed_source_role_payload_slices_by_source_plane() -> None:
    metadata = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("grayscale.tif", "color.tif"),
            component_metadata=(
                {
                    SOURCE_IMAGE_TYPE_METADATA_FIELD: (
                        GrayscaleImageTypeSourceRole.image_type()
                    )
                },
                {
                    SOURCE_IMAGE_TYPE_METADATA_FIELD: (
                        ColorImageTypeSourceRole.image_type()
                    )
                },
            ),
        )
    )
    payload = metadata.payload_with(np.zeros((2, 4, 5), dtype=np.float32))

    slices = payload_slices_for_alignment(payload)

    assert len(slices) == 2
    assert type(source_image_payload_role(slices[0])) is GrayscaleImageTypeSourceRole
    assert type(source_image_payload_role(slices[1])) is ColorImageTypeSourceRole
    assert image_payload_metadata(slices[0]).source_path == "grayscale.tif"
    assert image_payload_metadata(slices[1]).source_path == "color.tif"


def test_step_output_manifest_scopes_previous_step_inputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "images"
    producer = SimpleNamespace(
        step_scope_id="enhance",
        step_name="Enhance",
        pipeline_position=1,
        axis_id="A14",
        output_dir=output_dir,
    )
    consumer = SimpleNamespace(
        axis_id="A14",
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=3,
            source_step_scope_id="enhance",
        ),
        source_binding_plan=CompiledSourceBindingPlan.empty(),
        artifact_inputs={},
    )
    store = StepOutputManifestStore()

    store.begin_step(producer)
    store.record_outputs(
        producer,
        [
            ProducedOutputSemantics.from_output(
                producer,
                output_dir / "A14_s001_w1_z001_t001.tif",
                FunctionOutputIdentity(
                    component_values={
                        "well": "A14",
                        "site": 1,
                        "channel": 1,
                        "z_index": 1,
                        "timepoint": 1,
                    },
                    extension=".tif",
                    source="test",
                ),
            )
        ],
    )
    store.begin_step(producer)
    store.record_outputs(
        producer,
        [
            ProducedOutputSemantics.from_output(
                producer,
                output_dir / "A14_s001_w3_z001_t001.tif",
                FunctionOutputIdentity(
                    component_values={
                        "well": "A14",
                        "site": 1,
                        "channel": 3,
                        "z_index": 1,
                        "timepoint": 1,
                    },
                    extension=".tif",
                    source="test",
                ),
            ),
            ProducedOutputSemantics.from_output(
                producer,
                output_dir / "A14_s002_w3_z001_t001.tif",
                FunctionOutputIdentity(
                    component_values={
                        "well": "A14",
                        "site": 2,
                        "channel": 3,
                        "z_index": 1,
                        "timepoint": 1,
                    },
                    extension=".tif",
                    source="test",
                ),
            ),
        ],
    )

    assert store.producer_paths_for(consumer) == (
        "A14_s001_w3_z001_t001.tif",
        "A14_s002_w3_z001_t001.tif",
    )


def test_step_output_manifest_pattern_lookup_returns_producer_memory_paths(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "images"
    producer = SimpleNamespace(
        step_scope_id="mask_image",
        step_name="MaskImage",
        pipeline_position=4,
        axis_id="A14",
        output_dir=output_dir,
    )
    consumer = SimpleNamespace(
        axis_id="A14",
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=4,
            source_step_scope_id="mask_image",
        ),
        source_binding_plan=CompiledSourceBindingPlan.empty(),
        artifact_inputs={},
    )
    store = StepOutputManifestStore()

    store.begin_step(producer)
    store.record_outputs(
        producer,
        (
            ProducedOutputSemantics.from_output(
                producer,
                output_dir / "A14_s001_w3_z001_t001.tif",
                FunctionOutputIdentity(
                    component_values={
                        "well": "A14",
                        "site": 1,
                        "channel": 3,
                        "z_index": 1,
                        "timepoint": 1,
                    },
                    extension=".tif",
                    source="test",
                ),
            ),
            ProducedOutputSemantics.from_output(
                producer,
                output_dir / "A14_s002_w3_z001_t001.tif",
                FunctionOutputIdentity(
                    component_values={
                        "well": "A14",
                        "site": 2,
                        "channel": 3,
                        "z_index": 1,
                        "timepoint": 1,
                    },
                    extension=".tif",
                    source="test",
                ),
            ),
        ),
    )

    assert store.producer_paths_matching_pattern(
        consumer,
        "A14_s{iii}_w3_z001_t001.tif",
        SourceSchemaFilenameParser(),
    ) == [
        str(output_dir / "A14_s001_w3_z001_t001.tif"),
        str(output_dir / "A14_s002_w3_z001_t001.tif"),
    ]


def test_step_output_manifest_deduplicates_multi_artifact_anchor_paths(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "images"
    cells_producer = SimpleNamespace(
        step_scope_id="identify_cells",
        step_name="IdentifySecondaryObjects",
        pipeline_position=6,
        axis_id="A14",
        output_dir=output_dir,
    )
    nuclei_producer = SimpleNamespace(
        step_scope_id="identify_nuclei",
        step_name="IdentifyPrimaryObjects",
        pipeline_position=5,
        axis_id="A14",
        output_dir=output_dir,
    )
    consumer = SimpleNamespace(
        axis_id="A14",
        main_input_dependency=StepInputDependency.pipeline_start(),
        source_binding_plan=CompiledSourceBindingPlan.empty(),
        artifact_inputs={
            "Cells": ArtifactInputPlan(
                name="Cells",
                path="Cells",
                artifact_type=ObjectLabelsArtifactType,
                source_step_id=6,
                source_step_scope_id="identify_cells",
            ),
            "Nuclei": ArtifactInputPlan(
                name="Nuclei",
                path="Nuclei",
                artifact_type=ObjectLabelsArtifactType,
                source_step_id=5,
                source_step_scope_id="identify_nuclei",
            ),
        },
    )
    store = StepOutputManifestStore()

    for producer, output_key in (
        (cells_producer, "Cells"),
        (nuclei_producer, "Nuclei"),
    ):
        store.begin_step(producer)
        store.record_outputs(
            producer,
            (
                ProducedOutputSemantics.from_output(
                    producer,
                    output_dir / "A14_s001_w1_z001_t001.tif",
                    FunctionOutputIdentity(
                        component_values={
                            "well": "A14",
                            "site": 1,
                            "channel": 1,
                            "z_index": 1,
                            "timepoint": 1,
                        },
                        extension=".tif",
                        source="test",
                    ),
                    output_context=AlignedImageSliceContext.main_flow(
                        output_key=output_key,
                        artifact_kind=ObjectLabelsArtifactType.value,
                    ),
                ),
                ProducedOutputSemantics.from_output(
                    producer,
                    output_dir / "A14_s002_w1_z001_t001.tif",
                    FunctionOutputIdentity(
                        component_values={
                            "well": "A14",
                            "site": 2,
                            "channel": 1,
                            "z_index": 1,
                            "timepoint": 1,
                        },
                        extension=".tif",
                        source="test",
                    ),
                    output_context=AlignedImageSliceContext.main_flow(
                        output_key=output_key,
                        artifact_kind=ObjectLabelsArtifactType.value,
                    ),
                ),
            ),
        )

    assert store.producer_paths_matching_pattern(
        consumer,
        "A14_s{iii}_w1_z001_t001.tif",
        SourceSchemaFilenameParser(),
    ) == [
        str(output_dir / "A14_s001_w1_z001_t001.tif"),
        str(output_dir / "A14_s002_w1_z001_t001.tif"),
    ]


def test_step_output_manifest_uses_declared_artifact_producer_scope(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "images"
    requested_producer = SimpleNamespace(
        step_scope_id="crop_blue",
        step_name="CropBlue",
        pipeline_position=2,
        axis_id="A01",
        output_dir=output_dir,
    )
    previous_producer = SimpleNamespace(
        step_scope_id="crop_red",
        step_name="CropRed",
        pipeline_position=3,
        axis_id="A01",
        output_dir=output_dir,
    )
    consumer = SimpleNamespace(
        axis_id="A01",
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=3,
            source_step_scope_id="crop_red",
        ),
        source_binding_plan=CompiledSourceBindingPlan.empty(),
        artifact_inputs={
            "CropBlue": ArtifactInputPlan(
                name="CropBlue",
                path="CropBlue",
                artifact_type=ImageArtifactType,
                source_step_id=2,
                source_step_scope_id="crop_blue",
            )
        },
    )
    store = StepOutputManifestStore()

    store.begin_step(requested_producer)
    store.record_outputs(
        requested_producer,
        (
            ProducedOutputSemantics.from_output(
                requested_producer,
                output_dir / "A01_s001_w1_z001_t001.tif",
                FunctionOutputIdentity(
                    component_values={
                        "well": "A01",
                        "site": 1,
                        "channel": 1,
                        "z_index": 1,
                        "timepoint": 1,
                    },
                    extension=".tif",
                    source="test",
                ),
                output_context=AlignedImageSliceContext.main_flow(
                    output_key="CropBlue",
                    artifact_kind=ImageArtifactType.value,
                ),
            ),
        ),
    )
    store.begin_step(previous_producer)
    store.record_outputs(
        previous_producer,
        (
            ProducedOutputSemantics.from_output(
                previous_producer,
                output_dir / "A01_s001_w2_z001_t001.tif",
                FunctionOutputIdentity(
                    component_values={
                        "well": "A01",
                        "site": 1,
                        "channel": 2,
                        "z_index": 1,
                        "timepoint": 1,
                    },
                    extension=".tif",
                    source="test",
                ),
                output_context=AlignedImageSliceContext.main_flow(
                    output_key="CropRed",
                    artifact_kind=ImageArtifactType.value,
                ),
            ),
        ),
    )

    assert store.filter_to_producer_paths(
        consumer,
        [
            "A01_s001_w1_z001_t001.tif",
            "A01_s001_w2_z001_t001.tif",
        ],
        SourceSchemaFilenameParser(),
    ) == ["A01_s001_w2_z001_t001.tif"]


def test_step_output_manifest_ignores_sidecar_artifact_for_anchor_filtering() -> None:
    consumer = SimpleNamespace(
        axis_id="A01",
        main_input_dependency=StepInputDependency.pipeline_start(),
        source_binding_plan=CompiledSourceBindingPlan.empty(),
        artifact_inputs={
            "CropBlue__crop_mask": ArtifactInputPlan(
                name="CropBlue__crop_mask",
                path="CropBlue__crop_mask",
                artifact_type=ImageArtifactType,
                sidecar_role=ArtifactSidecarRole.CROP_MASK,
                source_step_id=0,
                source_step_scope_id="crop_mask",
            )
        },
    )

    assert StepOutputManifestStore().filter_to_producer_paths(
        consumer,
        ["A01_s{iii}_w2_z001_t001.tif"],
        SourceSchemaFilenameParser(),
    ) == ["A01_s{iii}_w2_z001_t001.tif"]


def test_step_output_manifest_preserves_pipeline_start_source_bound_anchor() -> None:
    consumer = SimpleNamespace(
        axis_id="A14",
        main_input_dependency=StepInputDependency.pipeline_start(),
        source_binding_plan=CompiledSourceBindingPlan(
            bindings=(NamedSourceBinding(alias="OrigActin_Golgi_Membrane"),)
        ),
        artifact_inputs={
            "Nuclei": ArtifactInputPlan(
                name="Nuclei",
                path="Nuclei",
                artifact_type=ObjectLabelsArtifactType,
                source_step_id=0,
                source_step_scope_id="identify_primary",
            )
        },
    )

    assert StepOutputManifestStore().filter_to_producer_paths(
        consumer,
        ["A14_s{iii}_w4_z001_t001.tif"],
        SourceSchemaFilenameParser(),
    ) == ["A14_s{iii}_w4_z001_t001.tif"]


def test_step_output_anchor_filter_skips_source_binding_filter() -> None:
    plan = SimpleNamespace(
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=4,
            source_step_scope_id="mask_image",
        ),
        source_binding_plan=CompiledSourceBindingPlan(
            bindings=(NamedSourceBinding(alias="OrigDNA"),)
        ),
    )
    grouped_patterns = PatternGroups({None: ("A14_s{iii}_w1_z001_t001.tif",)})
    pattern_filter = StepAnchorPatternFilter(
        plan=plan,
        parser=None,
        output_manifest=None,
        source_workspace_authority=None,
        source_workspace_projection_cache=VirtualWorkspaceSourceProjectionCache(),
    )

    assert pattern_filter.source_bound_anchor_patterns(grouped_patterns) is grouped_patterns


def test_source_bound_anchor_filter_scopes_bindings_to_execution_component() -> None:
    source_binding_plan = CompiledSourceBindingPlan(
        bindings=(
            NamedSourceBinding(
                alias="OrigStain1",
                selector=SourceSelector(
                    components=(ComponentSelector(AllComponents.CHANNEL, "1"),),
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
                component_identity=(
                    ComponentSelector(AllComponents.CHANNEL, "1"),
                ),
            ),
            NamedSourceBinding(
                alias="OrigStain2",
                selector=SourceSelector(
                    components=(ComponentSelector(AllComponents.CHANNEL, "2"),),
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
                component_identity=(
                    ComponentSelector(AllComponents.CHANNEL, "2"),
                ),
            ),
        ),
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )
    plan = SimpleNamespace(
        axis_id="A01",
        main_input_dependency=StepInputDependency.pipeline_start(),
        source_binding_plan=source_binding_plan,
        execution_group_component=AllComponents.CHANNEL,
        compiled_function_pattern=compile_function_pattern(lambda image: image, {}, {}),
    )
    pattern_filter = StepAnchorPatternFilter(
        plan=plan,
        parser=SourceSchemaFilenameParser(),
        output_manifest=None,
        source_workspace_authority=SimpleNamespace(
            projection_or_empty=lambda: VirtualWorkspaceSourceProjection.empty()
        ),
        source_workspace_projection_cache=VirtualWorkspaceSourceProjectionCache(),
    )

    filtered = pattern_filter.source_bound_anchor_patterns(
        PatternGroups(
            {
                "1": ("A01_s001_w1_z001_t001.tif",),
                "2": ("A01_s001_w2_z001_t001.tif",),
            }
        )
    )

    assert filtered.groups == {
        "1": ("A01_s001_w1_z001_t001.tif",),
        "2": ("A01_s001_w2_z001_t001.tif",),
    }


def test_source_bound_artifact_managed_step_keeps_source_anchors() -> None:
    contract = ModuleArtifactContract(
        module_name="SourceBoundArtifactManaged",
        items=(
            *ModuleArtifactContract.items_for_partition(
                RecordedArtifactOutputPartition,
                (ArtifactSpec.output("IllumStain1", ImageArtifactType),),
            ),
            *ModuleArtifactContract.items_for_partition(
                DeclaredArtifactOutputPartition,
                (ArtifactSpec.output("IllumStain1", ImageArtifactType),),
            ),
        ),
    )

    @module_artifact_contract(contract)
    @runtime_adapter("runtime", lambda _request: object())
    def source_bound_module(image, *, runtime):
        return image

    source_binding_plan = CompiledSourceBindingPlan(
        bindings=(
            NamedSourceBinding(
                alias="OrigStain1",
                selector=SourceSelector(
                    components=(ComponentSelector(AllComponents.CHANNEL, "1"),),
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
                component_identity=(
                    ComponentSelector(AllComponents.CHANNEL, "1"),
                ),
            ),
        ),
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )
    compiled_pattern = compile_function_pattern(
        source_bound_module,
        {},
        {
            "IllumStain1": ArtifactOutputPlan(
                name="IllumStain1",
                path="/memory/IllumStain1.pkl",
                artifact_type=ImageArtifactType,
            ),
        },
    )
    plan = SimpleNamespace(
        axis_id="A01",
        main_input_dependency=StepInputDependency.pipeline_start(),
        source_binding_plan=source_binding_plan,
        execution_group_component=None,
        compiled_function_pattern=compiled_pattern,
    )
    pattern_filter = StepAnchorPatternFilter(
        plan=plan,
        parser=SourceSchemaFilenameParser(),
        output_manifest=None,
        source_workspace_authority=SimpleNamespace(
            projection_or_empty=lambda: VirtualWorkspaceSourceProjection.empty()
        ),
        source_workspace_projection_cache=VirtualWorkspaceSourceProjectionCache(),
    )
    grouped_patterns = PatternGroups(
        {
            None: (
                "A01_s001_w1_z001_t001.tif",
                "A01_s002_w1_z001_t001.tif",
            )
        }
    )

    assert (
        pattern_filter.artifact_driven_anchor_patterns(grouped_patterns).groups
        == grouped_patterns.groups
    )


def test_step_output_load_filter_skips_source_binding_filter() -> None:
    from openhcs.core.steps.function_runtime import PatternGroupRuntime

    plan = SimpleNamespace(
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=4,
            source_step_scope_id="mask_image",
        ),
        source_binding_plan=CompiledSourceBindingPlan(
            bindings=(NamedSourceBinding(alias="OrigDNA"),)
        ),
    )
    runtime = PatternGroupRuntime.__new__(PatternGroupRuntime)
    runtime.request = SimpleNamespace(execution_plan=plan)
    matching_files = ["/tmp/outputs/A14_s001_w3_z001_t001.tif"]

    assert (
        runtime._filter_matching_files_for_source_bindings(matching_files)
        is matching_files
    )


def test_ungrouped_runtime_scope_omits_axis_component() -> None:
    from openhcs.core.steps.function_runtime import PatternGroupExecutionScope

    scope = PatternGroupExecutionScope(
        context=SimpleNamespace(),
        execution_plan=SimpleNamespace(
            axis_id="A01",
            group_by_value=GroupBy.CHANNEL.value,
            execution_group_value=None,
        ),
        compiled_group=SimpleNamespace(),
        component_value=None,
    )

    assert scope.axis_component is None
    assert scope.axis_component_value is None
    assert scope.axis_scope.component is None
    assert scope.axis_scope.value is None


def test_step_output_manifest_filters_declared_artifact_output_identity(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "images"
    producer = SimpleNamespace(
        step_scope_id="correct_illumination",
        step_name="CorrectIlluminationApply",
        pipeline_position=1,
        axis_id="A01",
        output_dir=output_dir,
    )
    consumer = SimpleNamespace(
        axis_id="A01",
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=1,
            source_step_scope_id="correct_illumination",
        ),
        source_binding_plan=CompiledSourceBindingPlan.empty(),
        artifact_inputs={
            "CorrDNA": ArtifactInputPlan(
                name="CorrDNA",
                path="CorrDNA",
                artifact_type=ImageArtifactType,
                source_step_id=1,
            )
        },
    )
    store = StepOutputManifestStore()

    store.begin_step(producer)
    store.record_outputs(
        producer,
        (
            ProducedOutputSemantics.from_output(
                producer,
                output_dir / "A01_s001_w1_z001_t001.tif",
                FunctionOutputIdentity(
                    component_values={
                        "well": "A01",
                        "site": 1,
                        "channel": 1,
                    },
                    extension=".tif",
                    source="test",
                ),
                output_context=AlignedImageSliceContext.main_flow(
                    output_key="CorrProtein",
                    artifact_kind=ImageArtifactType.value,
                ),
            ),
            ProducedOutputSemantics.from_output(
                producer,
                output_dir / "A01_s001_w2_z001_t001.tif",
                FunctionOutputIdentity(
                    component_values={
                        "well": "A01",
                        "site": 1,
                        "channel": 2,
                    },
                    extension=".tif",
                    source="test",
                ),
                output_context=AlignedImageSliceContext.main_flow(
                    output_key="CorrDNA",
                    artifact_kind=ImageArtifactType.value,
                ),
            ),
        ),
    )

    assert store.filter_to_producer_paths(
        consumer,
        [
            "A01_s001_w1_z001_t001.tif",
            "A01_s001_w2_z001_t001.tif",
        ],
        SourceSchemaFilenameParser(),
    ) == ["A01_s001_w2_z001_t001.tif"]


def test_step_output_manifest_preserves_anonymous_side_effect_main_flow(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "images"
    producer = SimpleNamespace(
        step_scope_id="identify_primary",
        step_name="IdentifyPrimaryObjects",
        pipeline_position=2,
        axis_id="A01",
        output_dir=output_dir,
    )
    consumer = SimpleNamespace(
        axis_id="A01",
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=2,
            source_step_scope_id="identify_primary",
        ),
        source_binding_plan=CompiledSourceBindingPlan.empty(),
        artifact_inputs={
            "Nuclei": ArtifactInputPlan(
                name="Nuclei",
                path="Nuclei",
                artifact_type=ObjectLabelsArtifactType,
                source_step_id=2,
            )
        },
    )
    store = StepOutputManifestStore()

    store.begin_step(producer)
    store.record_outputs(
        producer,
        (
            ProducedOutputSemantics.from_output(
                producer,
                output_dir / "A01_s001_w2_z001_t001.tif",
                FunctionOutputIdentity(
                    component_values={
                        "well": "A01",
                        "site": 1,
                        "channel": 2,
                    },
                    extension=".tif",
                    source="test",
                ),
            ),
        ),
    )

    assert store.filter_to_producer_paths(
        consumer,
        ["A01_s001_w2_z001_t001.tif"],
        SourceSchemaFilenameParser(),
    ) == ["A01_s001_w2_z001_t001.tif"]


def test_step_output_manifest_accepts_source_anchor_for_qualified_output(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "images"
    producer = SimpleNamespace(
        step_scope_id="correct_illumination_apply",
        step_name="CorrectIlluminationApply",
        pipeline_position=4,
        axis_id="A01",
        output_dir=output_dir,
    )
    consumer = SimpleNamespace(
        axis_id="A01",
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=4,
            source_step_scope_id="correct_illumination_apply",
        ),
        source_binding_plan=CompiledSourceBindingPlan.empty(),
        artifact_inputs={
            "CorrBlue": ArtifactInputPlan(
                name="CorrBlue",
                path="CorrBlue",
                artifact_type=ImageArtifactType,
                source_step_id=4,
            )
        },
    )
    store = StepOutputManifestStore()
    identity = FunctionOutputIdentity(
        component_values={
            "well": "A01",
            "site": 1,
            "channel": 1,
            "z_index": 1,
            "timepoint": 1,
        },
        extension=".jpg",
        source="test",
    ).with_filename_qualifier("CorrBlue")

    store.begin_step(producer)
    store.record_outputs(
        producer,
        (
            ProducedOutputSemantics.from_output(
                producer,
                output_dir / "A01_s001_w1_z001_t001_CorrBlue.jpg",
                identity,
                output_context=AlignedImageSliceContext.main_flow(
                    output_key="CorrBlue",
                    artifact_kind=ImageArtifactType.value,
                ),
            ),
        ),
    )

    assert store.filter_to_producer_paths(
        consumer,
        ["A01_s001_w1_z001_t001.jpg"],
        SourceSchemaFilenameParser(),
    ) == ["A01_s001_w1_z001_t001.jpg"]


def test_function_output_path_uses_payload_identity_over_input_carrier(
    tmp_path: Path,
) -> None:
    payload = RuntimeImagePayloadContext(
        np.zeros((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/source/plate1_A14_site2_Ch3.tif",
            source_component_metadata={
                "well": "A14",
                "site": "2",
                "channel": "3",
            },
        ),
        mask=None,
    ).payload()

    output_path = FunctionOutputPathAuthority.output_path(
        FunctionOutputPathRequest(
            parser=SourceSchemaFilenameParser(),
            output_dir=tmp_path,
            output_payload=payload,
            input_path="A14_s001_w1_z001_t001.tif",
        )
    )

    assert output_path.name == "A14_s002_w3_z001_t001.tif"


def test_function_output_path_uses_payload_identity_without_input_path(
    tmp_path: Path,
) -> None:
    payload = RuntimeImagePayloadContext(
        np.zeros((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/source/plate1_A14_site1_Ch5.tif",
            source_component_metadata={
                "well": "A14",
                "site": "1",
                "channel": "5",
                "z_index": "1",
                "timepoint": "1",
            },
        ),
        mask=None,
    ).payload()

    output_path = FunctionOutputPathAuthority.output_path(
        FunctionOutputPathRequest(
            parser=SourceSchemaFilenameParser(),
            output_dir=tmp_path,
            output_payload=payload,
            input_path=None,
        )
    )

    assert output_path.name == "A14_s001_w5_z001_t001.tif"


def test_function_output_identity_completes_partial_payload_metadata_from_fallback_path() -> None:
    parser = SourceSchemaFilenameParser()
    metadata = ImagePayloadMetadata(
        source_component_metadata={
            "well": "Sequence1",
            "site": 1,
            "z_index": 1,
            "channel": 1,
            "extension": ".tif",
        },
    )

    identity = FunctionOutputIdentityAuthority.identity_from_metadata(
        parser,
        metadata,
        fallback_identity_path="Sequence1_s001_w1_z001_t000.tif",
    )

    assert identity is not None
    assert (
        FunctionOutputPathAuthority.filename_for_identity(parser, identity)
        == "Sequence1_s001_w1_z001_t000.tif"
    )


def test_function_output_identity_uses_fallback_path_extension_for_payload_identity() -> None:
    parser = SourceSchemaFilenameParser()
    metadata = ImagePayloadMetadata(
        source_component_metadata={
            "well": "A01",
            "site": 1,
            "z_index": 1,
            "timepoint": 1,
            "channel": 2,
        },
    )

    identity = FunctionOutputIdentityAuthority.identity_from_metadata(
        parser,
        metadata,
        fallback_identity_path="A01_s001_w1_z001_t001.png",
    )

    assert identity is not None
    assert (
        FunctionOutputPathAuthority.filename_for_identity(parser, identity)
        == "A01_s001_w2_z001_t001.png"
    )


def test_function_output_path_uses_input_identity_for_multi_plane_carrier(
    tmp_path: Path,
) -> None:
    payload = RuntimeImagePayloadContext(
        np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_component_metadata={
                "well": "A14",
                "site": "1",
                "channel": "1",
                "timepoint": "1",
                "extension": ".tif",
            },
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/source/A14_s001_w1_z001_t001.tif",
                    "/source/A14_s001_w2_z001_t001.tif",
                ),
                component_metadata=(
                    {"well": "A14", "site": "1", "channel": "1"},
                    {"well": "A14", "site": "1", "channel": "2"},
                ),
            ),
        ),
        mask=None,
    ).payload()

    output_path = FunctionOutputPathAuthority.output_path(
        FunctionOutputPathRequest(
            parser=SourceSchemaFilenameParser(),
            output_dir=tmp_path,
            output_payload=payload,
            input_path="A14_s001_w1_z001_t001.tif",
        )
    )

    assert output_path.name == "A14_s001_w1_z001_t001.tif"


def test_function_output_path_rejects_multi_plane_carrier_without_input_path(
    tmp_path: Path,
) -> None:
    payload = RuntimeImagePayloadContext(
        np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/source/A14_s001_w1_z001_t001.tif",
                    "/source/A14_s001_w2_z001_t001.tif",
                ),
            ),
        ),
        mask=None,
    ).payload()

    with pytest.raises(ValueError, match="multi-plane source provenance"):
        FunctionOutputPathAuthority.output_path(
            FunctionOutputPathRequest(
                parser=SourceSchemaFilenameParser(),
                output_dir=tmp_path,
                output_payload=payload,
                input_path=None,
            )
        )


def test_function_output_path_uses_variable_component_identity(
    tmp_path: Path,
) -> None:
    payload = RuntimeImagePayloadContext(
        np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/source/A14_s001_w1_z001_t001.tif",
                    "/source/A14_s001_w1_z002_t001.tif",
                ),
                component_metadata=(
                    {
                        "well": "A14",
                        "site": "1",
                        "channel": "1",
                        "z_index": "1",
                        "timepoint": "1",
                    },
                    {
                        "well": "A14",
                        "site": "1",
                        "channel": "1",
                        "z_index": "2",
                        "timepoint": "1",
                    },
                ),
            ),
        ),
        mask=None,
    ).payload()
    request = FunctionOutputPathRequest(
        parser=SourceSchemaFilenameParser(),
        output_dir=tmp_path,
        output_payload=payload,
        input_path=None,
        variable_components=(VariableComponents.Z_INDEX,),
    )

    identity = FunctionOutputIdentityAuthority.identity(request)
    output_path = FunctionOutputPathAuthority.output_path_for_identity(
        request,
        identity,
    )

    assert output_path.name == "A14_s001_w1_z001_t001.tif"
    assert identity.component_values == {
        "well": "A14",
        "site": 1,
        "channel": 1,
        "timepoint": 1,
    }
    assert identity.filename_component_values is not None
    assert identity.filename_component_values["z_index"] == 1


def test_variable_component_identity_uses_fallback_path_extension(
    tmp_path: Path,
) -> None:
    payload = RuntimeImagePayloadContext(
        np.zeros((3, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/source/A01_s001_w1_z001_t001.png",
                    "/source/A01_s001_w2_z001_t001.jpg",
                    "/source/A01_s001_w3_z001_t001.jpg",
                ),
            ),
        ),
        mask=None,
    ).payload()
    request = FunctionOutputPathRequest(
        parser=SourceSchemaFilenameParser(),
        output_dir=tmp_path,
        output_payload=payload,
        input_path="A01_s001_w2_z001_t001.png",
        variable_components=(VariableComponents.CHANNEL,),
    )

    identity = FunctionOutputIdentityAuthority.identity(request)
    output_path = FunctionOutputPathAuthority.output_path_for_identity(
        request,
        identity,
    )

    assert output_path.name == "A01_s001_w1_z001_t001.png"
    assert identity.extension == ".png"
    assert identity.component_values == {
        "well": "A01",
        "site": 1,
        "z_index": 1,
        "timepoint": 1,
    }


def test_declared_main_flow_output_context_qualifies_output_filename(
    tmp_path: Path,
) -> None:
    payload = RuntimeImagePayloadContext(
        np.zeros((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/source/A01_s001_w1_z001_t001.jpg",
        ),
        mask=None,
    ).payload()
    request = FunctionOutputPathRequest(
        parser=SourceSchemaFilenameParser(),
        output_dir=tmp_path,
        output_payload=payload,
        input_path="A01_s001_w1_z001_t001.jpg",
    )
    identity = FunctionOutputIdentityAuthority.identity(request)

    red_path = FunctionOutputPathAuthority.output_path_for_identity(
        request,
        identity.with_filename_qualifier("CorrRed"),
    )
    green_path = FunctionOutputPathAuthority.output_path_for_identity(
        request,
        identity.with_filename_qualifier("CorrGreen"),
    )

    assert red_path.name == "A01_s001_w1_z001_t001_CorrRed.jpg"
    assert green_path.name == "A01_s001_w1_z001_t001_CorrGreen.jpg"
    assert SourceSchemaFilenameParser().parse_filename(red_path.name) == {
        "well": "A01",
        "site": 1,
        "channel": 1,
        "z_index": 1,
        "timepoint": 1,
        "extension": ".jpg",
    }


def test_function_output_path_rejects_variation_outside_identity_components(
    tmp_path: Path,
) -> None:
    payload = RuntimeImagePayloadContext(
        np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/source/A14_s001_w1_z001_t001.tif",
                    "/source/A14_s001_w2_z002_t001.tif",
                ),
            ),
        ),
        mask=None,
    ).payload()

    with pytest.raises(ValueError, match="varies outside identity components"):
        FunctionOutputIdentityAuthority.identity(
            FunctionOutputPathRequest(
                parser=SourceSchemaFilenameParser(),
                output_dir=tmp_path,
                output_payload=payload,
                input_path=None,
                variable_components=(VariableComponents.Z_INDEX,),
            )
        )


def test_function_output_path_rejects_group_by_component_stack_variation(
    tmp_path: Path,
) -> None:
    payload = RuntimeImagePayloadContext(
        np.zeros((3, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/source/A01_s001_w1_z001_t001.tif",
                    "/source/A01_s001_w3_z001_t001.tif",
                    "/source/A01_s001_w2_z001_t001.tif",
                ),
            ),
        ),
        mask=None,
    ).payload()
    with pytest.raises(ValueError, match="varies outside identity components"):
        FunctionOutputIdentityAuthority.identity(
            FunctionOutputPathRequest(
                parser=SourceSchemaFilenameParser(),
                output_dir=tmp_path,
                output_payload=payload,
                input_path=None,
                variable_components=(VariableComponents.SITE,),
            )
        )


def test_input_aligned_stack_output_uses_input_filename_identity(
    tmp_path: Path,
) -> None:
    payload = RuntimeImagePayloadContext(
        np.zeros((2, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/source/A01_s001_w1_z001_t001.png",
                    "/source/A01_s002_w1_z001_t001.png",
                ),
            ),
        ),
        mask=None,
    ).payload()
    request = FunctionOutputPathRequest(
        parser=SourceSchemaFilenameParser(),
        output_dir=tmp_path,
        output_payload=payload,
        input_path="A01_s002_w1_z001_t001.png",
        variable_components=(VariableComponents.SITE,),
        input_aligned_output=True,
    )

    identity = FunctionOutputIdentityAuthority.identity(request)
    output_path = FunctionOutputPathAuthority.output_path_for_identity(
        request,
        identity,
    )

    assert output_path.name == "A01_s002_w1_z001_t001.png"
    assert identity.component_values["site"] == 2
    assert identity.filename_component_values is not None
    assert identity.filename_component_values["site"] == 2


def test_function_output_path_uses_input_split_axis_for_payload_filename(
    tmp_path: Path,
) -> None:
    payload = RuntimeImagePayloadContext(
        np.zeros((4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/source/A01_s001_w1_z001_t001.tif",
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "1",
                "z_index": "1",
                "timepoint": "1",
            },
        ),
        mask=None,
    ).payload()
    request = FunctionOutputPathRequest(
        parser=SourceSchemaFilenameParser(),
        output_dir=tmp_path,
        output_payload=payload,
        input_path="A01_s003_w1_z001_t001.tif",
        variable_components=(VariableComponents.SITE,),
    )

    identity = FunctionOutputIdentityAuthority.identity(request)
    output_path = FunctionOutputPathAuthority.output_path_for_identity(
        request,
        identity,
    )

    assert output_path.name == "A01_s003_w1_z001_t001.tif"
    assert identity.component_values["site"] == 3
    assert identity.filename_component_values is not None
    assert identity.filename_component_values["site"] == 3
