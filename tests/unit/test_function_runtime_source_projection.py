from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from polystore.virtual_workspace import SourcePixelRef

from openhcs.constants.constants import AllComponents, GroupBy, VariableComponents
from openhcs.formats.pattern.pattern_discovery import PatternDiscoveryEngine
from openhcs.core.function_patterns import (
    InvocationArtifactInputEdgePlan,
    InvocationArtifactInputProjectionKey,
    RuntimeInvocationDomain,
    compile_function_pattern,
)
from openhcs.core.pipeline.function_contracts import (
    artifact_inputs,
    artifact_outputs,
    composed_image_payload,
    special_inputs,
)
from openhcs.core.component_group_scope import (
    ComponentGroupScope,
)
from openhcs.core.component_set import ComponentSet
from openhcs.core.pipeline.path_planner import PathPlanner, PathPlannerArtifactStage
from openhcs.core.steps.function_output_manifest import (
    NoStepOutputManifestMatch,
    StepOutputManifestStore,
)
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
    GroupLineageSourceRelation,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.runtime_adapters import runtime_adapter
from openhcs.core.steps.function_output_identity import (
    FunctionOutputIdentity,
    FunctionOutputIdentityCache,
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
from openhcs.core.source_projection import (
    OpenHCSPlaneAddress,
    SourceArtifactProjection,
    SourcePlaneProjection,
    SourceProjectionSet,
)
from openhcs.core.source_binding_selection import SourcePatternResolutionContext
from openhcs.core.source_metadata import SourceFilterPathMetadata
from openhcs.core.aligned_image_payload import (
    ImagePayloadBundleContext,
    payload_slices_for_alignment,
    stack_image_payload_context,
)
from openhcs.core.step_dependencies import StepInputDependency
from openhcs.core.source_bindings import (
    SOURCE_BINDING_ALIAS_METADATA_FIELD,
    ComponentSelector,
    CompiledSourceBindingPlan,
    NamedSourceBinding,
    SourceBindingRuntimeContext,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingOrigin,
    SourceProjectionRole,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceSelector,
)
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    ImagePayloadMetadataCompositionMode,
    image_payload_data,
    image_payload_metadata,
)
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis
from openhcs.core.source_image_provenance import (
    SourceImageProvenancePlanes,
)
from openhcs.core.source_matching import SourceImageSetIdentityPolicy
from openhcs.core.runtime_stack_cache import RuntimeImageStackCache
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser


@composed_image_payload
def _compose_image_domain(image: object) -> object:
    return image


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

    assert patterns == {"source_projection_axis": ["A01_s001_w1_z001_t001.tif"]}


def test_virtual_workspace_pipeline_start_files_preserve_virtual_identity(
    tmp_path: Path,
) -> None:
    """Pipeline-start source resolution must not collapse per-well virtual files."""

    plate_path = tmp_path / "plate"
    real_path = tmp_path / "source" / "image.png"
    virtual_a = plate_path / "W001_s001_w1_z001_t001.png"
    virtual_b = plate_path / "W002_s001_w1_z001_t001.png"
    projection = VirtualWorkspaceSourceProjection(
        source_refs_by_virtual_path={
            virtual_a.name: SourcePixelRef("disk", str(real_path)),
            str(virtual_a): SourcePixelRef("disk", str(real_path)),
            virtual_b.name: SourcePixelRef("disk", str(real_path)),
            str(virtual_b): SourcePixelRef("disk", str(real_path)),
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


def test_workspace_source_files_select_exact_projection_roles(
    tmp_path: Path,
) -> None:
    plate_path = tmp_path / "plate"
    canonical_path = "A01_s001_w1_z001_t001.tif"
    artifact_path = f"_source/Illumination/{canonical_path}"
    address = OpenHCSPlaneAddress(
        well="A01",
        site="1",
        channel="1",
        z_index="1",
        timepoint="1",
    )
    plane_projection = SourcePlaneProjection(
        address=address,
        ref=SourcePixelRef("disk", "/source/image.tif"),
        source_alias="Original",
    )
    artifact_projection = SourceArtifactProjection(
        address=address,
        ref=SourcePixelRef("disk", "/source/illumination.npy"),
        source_alias="Illumination",
        artifact_kind=ImageArtifactType,
    )
    projection = VirtualWorkspaceSourceProjection(
        source_refs_by_virtual_path={
            canonical_path: plane_projection.ref,
            artifact_path: artifact_projection.ref,
        },
        source_metadata_by_path={},
        source_projections_by_virtual_path={
            canonical_path: plane_projection,
            artifact_path: artifact_projection,
        },
        workspace_root=str(plate_path),
    )

    assert projection.files_for_projection_role(
        SourceProjectionRole.PRIMARY_PLANE
    ) == (
        str(plate_path / canonical_path),
    )
    assert projection.files_for_projection_role(
        SourceProjectionRole.SOURCE_ARTIFACT
    ) == (str(plate_path / artifact_path),)


def test_workspace_source_projection_carries_exact_aliases_into_stack_provenance(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    source_paths = (
        tmp_path / "source" / "raw.tif",
        tmp_path / "source" / "illum.mat",
    )
    projection_set = SourceProjectionSet(
        tuple(
            SourcePlaneProjection(
                address=OpenHCSPlaneAddress(
                    well="A01",
                    site="1",
                    channel=str(channel),
                    z_index="1",
                    timepoint="1",
                ),
                ref=SourcePixelRef("disk", str(source_path)),
                source_alias=alias,
            )
            for channel, alias, source_path in zip(
                (1, 2),
                ("Raw", "Illum"),
                source_paths,
                strict=True,
            )
        )
    )
    subdirectory = projection_set.metadata_dict(
        parser=SourceSchemaFilenameParser(),
        microscope_handler_name="source_bindings",
        source_filename_parser_name="SourceSchemaFilenameParser",
        grid_dimensions=[1, 1],
        pixel_size=1.0,
    )
    projection = VirtualWorkspaceSourceProjection.from_openhcs_metadata(
        workspace_root,
        {"subdirectories": {".": subdirectory}},
    )

    lookups = tuple(
        VirtualWorkspacePathLookup.from_paths(
            virtual_path,
            str(workspace_root / virtual_path),
        )
        for virtual_path in subdirectory["image_files"]
    )
    projected_payloads = tuple(
        projection.project_payload(
            lookup,
            ImagePayloadMetadata.for_array_payload(
                np.full((4, 5), channel, dtype=np.float32),
                source_path=str(source_path),
            ).payload_with(np.full((4, 5), channel, dtype=np.float32)),
        )
        for channel, source_path, lookup in zip(
            (1, 2),
            source_paths,
            lookups,
            strict=True,
        )
    )

    assert tuple(
        projection.require_source_projection_for(
            VirtualWorkspacePathLookup.from_paths(
                virtual_path,
                str(workspace_root / virtual_path),
            )
        ).source_alias
        for virtual_path in subdirectory["image_files"]
    ) == ("Raw", "Illum")
    assert tuple(
        image_payload_metadata(payload).source_image_names
        for payload in projected_payloads
    ) == (("Raw",), ("Illum",))
    assert all(
        "source_alias"
        not in (image_payload_metadata(payload).source_component_metadata or {})
        for payload in projected_payloads
    )

    stack = stack_image_payload_context(
        projected_payloads,
        np.stack(tuple(image_payload_data(payload) for payload in projected_payloads)),
        metadata_mode=projection.payload_composition_mode(lookups),
    )

    assert image_payload_metadata(stack).source_image_names == ("Raw", "Illum")
    assert image_payload_metadata(stack).plane_axis is RuntimePlaneAxis.SOURCE_BINDING


@pytest.mark.parametrize(
    "source_binding_plan",
    (
        CompiledSourceBindingPlan.empty(),
        CompiledSourceBindingPlan(bindings=(NamedSourceBinding(alias="OrigBlue"),)),
    ),
)
def test_workspace_source_loading_preserves_declared_tiff_intensity_scale(
    tmp_path: Path,
    source_binding_plan: CompiledSourceBindingPlan,
) -> None:
    import tifffile
    from openhcs.core.steps.function_runtime import PatternGroupRuntime

    source_path = tmp_path / "source.tif"
    source_pixels = np.array([[0, 4095]], dtype=np.uint16)
    tifffile.imwrite(
        source_path,
        source_pixels,
        extratags=((281, "H", 1, 4095, False),),
    )
    virtual_path = "A01_s001_w1_z001_t001.tif"
    source_plane = SourcePlaneProjection(
        address=OpenHCSPlaneAddress(
            well="A01",
            site="1",
            channel="1",
            z_index="1",
            timepoint="1",
        ),
        ref=SourcePixelRef("disk", str(source_path)),
        source_alias="OrigBlue",
    )
    projection = VirtualWorkspaceSourceProjection(
        source_refs_by_virtual_path={virtual_path: source_plane.ref},
        source_metadata_by_path={
            virtual_path: {SOURCE_BINDING_ALIAS_METADATA_FIELD: "OrigBlue"}
        },
        source_projections_by_virtual_path={virtual_path: source_plane},
        workspace_root=str(tmp_path),
    )

    class SourceFileManager:
        @staticmethod
        def resolve_address(backend_address, backend, *, base_path):
            assert backend == "disk"
            assert Path(backend_address) == source_path
            assert Path(base_path) == source_path.parent
            return backend_address

    runtime = PatternGroupRuntime.__new__(PatternGroupRuntime)
    runtime.request = SimpleNamespace(
        source_binding_plan=source_binding_plan,
        context=SimpleNamespace(filemanager=SourceFileManager()),
    )

    payload = runtime._apply_workspace_source_binding_payload(
        source_pixels,
        source_projection=projection,
        lookup=VirtualWorkspacePathLookup.from_paths(
            virtual_path,
            str(tmp_path / virtual_path),
        ),
    )

    metadata = image_payload_metadata(payload)
    assert metadata.intensity_scale == 4095.0
    assert metadata.source_dtype == "uint16"
    assert metadata.source_image_names == ("OrigBlue",)


def test_virtual_workspace_source_filters_use_persisted_candidate_identity(
    tmp_path: Path,
) -> None:
    plate_path = tmp_path / "plate"
    virtual_paths = (
        plate_path / "A01_s001_w1_z001_t001.tif",
        plate_path / "A01_s002_w1_z001_t001.tif",
    )
    source_filter_paths = (
        tmp_path / "source" / "0_1_N_R.png",
        tmp_path / "source" / "0_2_N_R.png",
    )
    source_metadata_by_path: dict[str, dict[str, object]] = {}
    for site, (virtual_path, source_filter_path) in enumerate(
        zip(virtual_paths, source_filter_paths, strict=True),
        start=1,
    ):
        filter_metadata: dict[str, object] = {"site": str(site)}
        SourceFilterPathMetadata.from_paths((str(source_filter_path),)).merge_into(
            filter_metadata,
            path=virtual_path.name,
        )
        source_metadata_by_path[virtual_path.name] = filter_metadata
        source_metadata_by_path[str(virtual_path)] = filter_metadata
    source_ref = SourcePixelRef(
        "opaque_backend",
        '{"opaque":"address-without-source-name"}',
    )
    projection = VirtualWorkspaceSourceProjection(
        source_refs_by_virtual_path={
            path_key: source_ref
            for virtual_path in virtual_paths
            for path_key in (virtual_path.name, str(virtual_path))
        },
        source_metadata_by_path=source_metadata_by_path,
        workspace_root=str(plate_path),
    )
    context = SourcePatternResolutionContext.from_projection(
        parser=SourceSchemaFilenameParser(),
        projection=projection,
    )

    assert context.candidate_filter_paths(str(virtual_paths[0])) == (
        str(source_filter_paths[0]),
    )
    assert context.candidate_filter_paths("A01_s{iii}_w1_z001_t001.tif") == (
        str(source_filter_paths[0]),
        str(source_filter_paths[1]),
    )


def test_virtual_workspace_axis_filter_uses_virtual_filename_metadata_when_missing(
    tmp_path: Path,
) -> None:
    plate_path = tmp_path / "plate"
    real_path = tmp_path / "source" / "image.png"
    virtual_a = plate_path / "A01_s001_w1_z001_t001.tif"
    virtual_b = plate_path / "A02_s001_w1_z001_t001.tif"
    projection = VirtualWorkspaceSourceProjection(
        source_refs_by_virtual_path={
            virtual_a.name: SourcePixelRef("disk", str(real_path)),
            virtual_b.name: SourcePixelRef("disk", str(real_path)),
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
        source_refs_by_virtual_path={
            virtual_path.name: SourcePixelRef("disk", str(real_path)),
            str(virtual_path): SourcePixelRef("disk", str(real_path)),
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
        source_refs_by_virtual_path={
            virtual_path.name: SourcePixelRef("disk", str(real_path)),
        },
        source_metadata_by_path={
            virtual_path.name: {"OpenHCSSourceVoxelSpacingZYX": "2,1,1"},
        },
        workspace_root=str(plate_path),
    )

    with pytest.raises(ValueError, match="OpenHCSSourceVoxelSpacingZYX"):
        projection.validate_runtime_metadata_projection()


def test_stack_payload_context_promotes_single_channel_slice_metadata() -> None:
    first = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s001_w1_z001_t001.tif",),
            component_metadata=({"well": "A01", "site": 1, "channel": 1},),
        )
    ).payload_with(np.zeros((4, 5), dtype=np.float32), None)
    second = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s002_w1_z001_t001.tif",),
            component_metadata=({"well": "A01", "site": 2, "channel": 1},),
        )
    ).payload_with(np.ones((4, 5), dtype=np.float32), None)
    stack = np.stack(
        (
            image_payload_data(first),
            image_payload_data(second),
        )
    )

    payload = stack_image_payload_context(
        (first, second),
        stack,
        metadata_mode=ImagePayloadMetadataCompositionMode.STACK,
    )
    metadata = image_payload_metadata(payload)

    assert metadata.source_image_provenance_planes.paths == (
        "/input/A01_s001_w1_z001_t001.tif",
        "/input/A01_s002_w1_z001_t001.tif",
    )
    assert tuple(
        dict(item)
        for item in metadata.source_image_provenance_planes.component_metadata
    ) == (
        {"well": "A01", "site": 1, "channel": 1},
        {"well": "A01", "site": 2, "channel": 1},
    )


def test_bundle_payload_context_preserves_source_binding_plane_metadata() -> None:
    first = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s001_w1_z001_t001.tif",),
            component_metadata=({"well": "A01", "site": 1, "channel": 1},),
        )
    ).payload_with(np.zeros((4, 5), dtype=np.float32), None)
    second = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s001_w2_z001_t001.tif",),
            component_metadata=({"well": "A01", "site": 1, "channel": 2},),
        )
    ).payload_with(np.ones((4, 5), dtype=np.float32), None)

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


def test_payload_slices_do_not_infer_alignment_from_source_provenance() -> None:
    metadata = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("grayscale.tif", "color.tif"),
            component_metadata=(
                {"site": "1"},
                {"site": "2"},
            ),
        )
    )
    payload = metadata.payload_with(np.zeros((2, 4, 5), dtype=np.float32))

    slices = payload_slices_for_alignment(payload)

    assert len(slices) == 1
    assert slices[0] is payload


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
        compiled_function_pattern=compile_function_pattern(lambda image: image, {}, {}),
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
        compiled_function_pattern=compile_function_pattern(lambda image: image, {}, {}),
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


def test_step_output_manifest_does_not_treat_artifact_inputs_as_main_flow(
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
            plan.ref(): plan
            for plan in (
                ArtifactInputPlan(
                    name="Cells",
                    path="Cells",
                    artifact_type=ObjectLabelsArtifactType,
                    source_step_id=6,
                    source_step_scope_id="identify_cells",
                ),
                ArtifactInputPlan(
                    name="Nuclei",
                    path="Nuclei",
                    artifact_type=ObjectLabelsArtifactType,
                    source_step_id=5,
                    source_step_scope_id="identify_nuclei",
                ),
            )
        },
        compiled_function_pattern=compile_function_pattern(lambda image: image, {}, {}),
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

    assert (
        store.producer_paths_matching_pattern(
            consumer,
            "A14_s{iii}_w1_z001_t001.tif",
            SourceSchemaFilenameParser(),
        )
        == []
    )


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
            plan.ref(): plan
            for plan in (
                ArtifactInputPlan(
                    name="CropBlue",
                    path="CropBlue",
                    artifact_type=ImageArtifactType,
                    source_step_id=2,
                    source_step_scope_id="crop_blue",
                ),
            )
        },
        compiled_function_pattern=compile_function_pattern(lambda image: image, {}, {}),
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


def test_special_artifact_input_does_not_narrow_step_output_main_flow(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "images"
    producer = SimpleNamespace(
        step_scope_id="image_set",
        step_name="ImageSet",
        pipeline_position=2,
        axis_id="A01",
        output_dir=output_dir,
    )
    consumer = SimpleNamespace(
        axis_id="A01",
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=2,
            source_step_scope_id="image_set",
        ),
        source_binding_plan=CompiledSourceBindingPlan.empty(),
        artifact_inputs={
            plan.ref(): plan
            for plan in (
                ArtifactInputPlan(
                    name="Objects",
                    path="Objects",
                    artifact_type=ObjectLabelsArtifactType,
                    source_step_id=2,
                    source_step_scope_id="image_set",
                ),
            )
        },
        compiled_function_pattern=compile_function_pattern(lambda image: image, {}, {}),
    )
    store = StepOutputManifestStore()
    store.begin_step(producer)
    store.record_outputs(
        producer,
        tuple(
            ProducedOutputSemantics.from_output(
                producer,
                output_dir / f"A01_s001_w{channel}_z001_t001.tif",
                FunctionOutputIdentity(
                    component_values={
                        "well": "A01",
                        "site": 1,
                        "channel": channel,
                    },
                    extension=".tif",
                    source="test",
                ),
                output_context=AlignedImageSliceContext.main_flow(
                    output_key=output_key,
                    artifact_kind=ImageArtifactType.value,
                ),
            )
            for channel, output_key in ((1, "Image1"), (2, "Image2"))
        ),
    )

    assert store.filter_to_producer_paths(
        consumer,
        [
            "A01_s001_w1_z001_t001.tif",
            "A01_s001_w2_z001_t001.tif",
        ],
        SourceSchemaFilenameParser(),
    ) == [
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w2_z001_t001.tif",
    ]


def test_step_output_manifest_ignores_sidecar_artifact_for_anchor_filtering() -> None:
    consumer = SimpleNamespace(
        axis_id="A01",
        main_input_dependency=StepInputDependency.pipeline_start(),
        source_binding_plan=CompiledSourceBindingPlan.empty(),
        artifact_inputs={
            plan.ref(): plan
            for plan in (
                ArtifactInputPlan(
                    name="CropBlue__crop_mask",
                    path="CropBlue__crop_mask",
                    artifact_type=ImageArtifactType,
                    sidecar_role=ArtifactSidecarRole.CROP_MASK,
                    source_step_id=0,
                    source_step_scope_id="crop_mask",
                ),
            )
        },
        compiled_function_pattern=compile_function_pattern(lambda image: image, {}, {}),
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
            plan.ref(): plan
            for plan in (
                ArtifactInputPlan(
                    name="Nuclei",
                    path="Nuclei",
                    artifact_type=ObjectLabelsArtifactType,
                    source_step_id=0,
                    source_step_scope_id="identify_primary",
                ),
            )
        },
        compiled_function_pattern=compile_function_pattern(lambda image: image, {}, {}),
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

    assert (
        pattern_filter.source_bound_anchor_patterns(grouped_patterns)
        is grouped_patterns
    )


def test_step_output_anchor_uses_compiler_owned_component_scope() -> None:
    plan = SimpleNamespace(
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=4,
            source_step_scope_id="object_to_image",
        ),
        execution_group_scope=ComponentGroupScope.from_raw(
            ("0",),
            component=AllComponents.CHANNEL,
        ),
        compiled_function_pattern=compile_function_pattern(lambda image: image, {}, {}),
    )
    grouped_patterns = PatternGroups({"2": ("A01_s{iii}_w2_z001_t001.tif",)})
    pattern_filter = StepAnchorPatternFilter(
        plan=plan,
        parser=None,
        output_manifest=None,
        source_workspace_authority=None,
        source_workspace_projection_cache=VirtualWorkspaceSourceProjectionCache(),
    )

    assert pattern_filter.execution_group_anchor_patterns(grouped_patterns).groups == {
        "0": ("A01_s{iii}_w2_z001_t001.tif",),
    }


def test_step_output_dispatch_projects_producer_group_before_pattern_selection(
    monkeypatch,
) -> None:
    def identify_secondary(image):
        return image

    compiled_pattern = compile_function_pattern(
        {"2": identify_secondary},
        {},
        {},
    )
    plan = SimpleNamespace(
        axis_id="A01",
        step_index=4,
        step_name="IdentifySecondaryObjects",
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=3,
            source_step_scope_id="identify_primary",
        ),
        source_binding_plan=CompiledSourceBindingPlan.empty(),
        execution_group_value="channel",
        execution_group_scope=ComponentGroupScope.from_raw(
            ("2",),
            component=AllComponents.CHANNEL,
        ),
        compiled_function_pattern=compiled_pattern,
    )
    pattern_filter = StepAnchorPatternFilter(
        plan=plan,
        parser=SourceSchemaFilenameParser(),
        output_manifest=SimpleNamespace(
            filter_to_producer_paths=lambda _plan, paths, _parser: tuple(paths)
        ),
        source_workspace_authority=None,
        source_workspace_projection_cache=VirtualWorkspaceSourceProjectionCache(),
    )
    monkeypatch.setattr(
        StepAnchorPatternFilter,
        "from_context",
        classmethod(lambda cls, context, plan: pattern_filter),
    )
    executor = object.__new__(FunctionStepExecutor)
    executor.context = SimpleNamespace()
    executor.plan = plan

    grouped = executor._prepare_groups({"A01": {"1": ("A01_s{iii}_w1_z001_t001.tif",)}})

    assert grouped.groups == {
        "2": ("A01_s{iii}_w1_z001_t001.tif",),
    }


def test_artifact_managed_dispatch_validates_producer_before_group_projection() -> None:
    runtime_image = ArtifactSpec.input("CropBlue", ImageArtifactType)

    @artifact_inputs(runtime_image)
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_inputs=True,
    )
    def identify_primary_objects(image, *, runtime):
        del runtime
        return image

    runtime_plan = ArtifactInputPlan(
        name=runtime_image.name,
        path="/memory/CropBlue.pkl",
        artifact_type=runtime_image.artifact_type,
    )
    compiled_pattern = compile_function_pattern(
        identify_primary_objects,
        {plan.ref(): plan for plan in (runtime_plan,)},
        {},
    )
    compiled_pattern = PathPlannerArtifactStage(
        PathPlanner.__new__(PathPlanner)
    ).compile_invocation_input_edges(
        compiled_pattern,
        artifact_inputs={runtime_plan.ref(): runtime_plan},
        relation_source_scopes={
            runtime_image.ref(): runtime_plan.producer_group_scope(),
        },
        execution_group_scope=ComponentGroupScope.from_raw(
            ("1",),
            component=AllComponents.CHANNEL,
        ),
        consumer_variable_components=ComponentSet(),
    )
    invocation = next(compiled_pattern.iter_invocations())
    assert tuple(edge.spec for edge in invocation.artifact_input_edges) == (
        runtime_image,
    )
    plan = SimpleNamespace(
        axis_id="A01",
        step_index=2,
        step_name="IdentifyPrimaryObjects",
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=1,
            source_step_scope_id="crop_green_red",
        ),
        source_binding_plan=CompiledSourceBindingPlan.empty(),
        execution_group_value="channel",
        execution_group_scope=ComponentGroupScope.from_raw(
            ("1",),
            component=AllComponents.CHANNEL,
        ),
        compiled_function_pattern=compiled_pattern,
    )

    def filter_to_producer_paths(_plan, paths, _parser):
        return tuple(path for path in paths if "_w2_" in path)

    pattern_filter = StepAnchorPatternFilter(
        plan=plan,
        parser=SourceSchemaFilenameParser(),
        output_manifest=SimpleNamespace(
            filter_to_producer_paths=filter_to_producer_paths,
        ),
        source_workspace_authority=None,
        source_workspace_projection_cache=VirtualWorkspaceSourceProjectionCache(),
    )

    filtered = pattern_filter.filtered(
        PatternGroups(
            {
                "1": ("A01_s{iii}_w1_z001_t001.tif",),
                "2": ("A01_s{iii}_w2_z001_t001.tif",),
            }
        )
    )

    assert filtered.groups == {
        "1": ("A01_s{iii}_w2_z001_t001.tif",),
    }


def test_artifact_managed_missing_output_context_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openhcs.core.steps import function_runtime

    class MissingProducerManifest:
        def producer_output_contexts_for_paths(self, *_args):
            raise NoStepOutputManifestMatch

    monkeypatch.setattr(
        function_runtime,
        "step_output_manifest",
        lambda _context: MissingProducerManifest(),
    )
    runtime = function_runtime.PatternGroupRuntime(
        SimpleNamespace(
            context=SimpleNamespace(
                microscope_handler=SimpleNamespace(parser=SourceSchemaFilenameParser())
            ),
            execution_plan=SimpleNamespace(),
            compiled_group=SimpleNamespace(
                runtime_domain=RuntimeInvocationDomain.ARTIFACT_MANAGED,
            ),
            pattern_group_info="A01_s001_w2_z001_t001.tif",
        )
    )

    with pytest.raises(NoStepOutputManifestMatch):
        runtime._producer_output_contexts(("A01_s001_w2_z001_t001.tif",))


def test_step_output_anchor_resolves_dynamic_component_scope_from_patterns() -> None:
    plan = SimpleNamespace(
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=4,
            source_step_scope_id="crop",
        ),
        execution_group_scope=ComponentGroupScope.dynamic(AllComponents.SITE),
        compiled_function_pattern=compile_function_pattern(
            lambda image: image,
            {},
            {},
        ),
    )
    grouped_patterns = PatternGroups(
        {
            "1": ("A01_s001_w{iii}_z001_t001.tif",),
            "2": ("A01_s002_w{iii}_z001_t001.tif",),
        }
    )
    pattern_filter = StepAnchorPatternFilter(
        plan=plan,
        parser=None,
        output_manifest=None,
        source_workspace_authority=None,
        source_workspace_projection_cache=VirtualWorkspaceSourceProjectionCache(),
    )

    assert (
        pattern_filter.execution_group_anchor_patterns(grouped_patterns).groups
        == grouped_patterns.groups
    )


def test_source_anchor_uses_compiler_owned_static_component_scope() -> None:
    plan = SimpleNamespace(
        main_input_dependency=StepInputDependency.pipeline_start(),
        execution_group_scope=ComponentGroupScope.from_raw(
            ("1",),
            component=AllComponents.CHANNEL,
        ),
        compiled_function_pattern=compile_function_pattern(
            lambda image: image,
            {},
            {},
        ),
    )
    grouped_patterns = PatternGroups(
        {
            "1": ("A01_s{iii}_w1_z001_t001.tif",),
            "2": ("A01_s{iii}_w2_z001_t001.tif",),
            "3": ("A01_s{iii}_w3_z001_t001.tif",),
        }
    )
    pattern_filter = StepAnchorPatternFilter(
        plan=plan,
        parser=None,
        output_manifest=None,
        source_workspace_authority=None,
        source_workspace_projection_cache=VirtualWorkspaceSourceProjectionCache(),
    )

    assert pattern_filter.execution_group_anchor_patterns(grouped_patterns).groups == {
        "1": ("A01_s{iii}_w1_z001_t001.tif",),
    }


def test_source_bound_anchor_filter_combines_ordered_non_grouped_source_sets() -> None:
    bindings = (
        NamedSourceBinding(
            alias="OrigStain1",
            selector=SourceSelector(
                components=(ComponentSelector(AllComponents.CHANNEL, "1"),),
            ),
            origin=SourceBindingOrigin.PIPELINE_START,
            component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
        ),
        NamedSourceBinding(
            alias="OrigStain2",
            selector=SourceSelector(
                components=(ComponentSelector(AllComponents.CHANNEL, "2"),),
            ),
            origin=SourceBindingOrigin.PIPELINE_START,
            component_identity=(ComponentSelector(AllComponents.CHANNEL, "2"),),
        ),
    )
    source_binding_plan = CompiledSourceBindingPlan(
        bindings=bindings,
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )

    source_set_measurements = ArtifactSpec.output(
        "SourceSetMeasurements",
        MeasurementsArtifactType,
        relations=tuple(
            GroupLineageSourceRelation(source=binding.input_spec().ref())
            for binding in bindings
        ),
    )

    @artifact_inputs(*(binding.input_spec() for binding in bindings))
    @artifact_outputs(source_set_measurements)
    @composed_image_payload
    def measure_source_set(image):
        return image

    measurement_plan = ArtifactOutputPlan(
        name="SourceSetMeasurements",
        path="/memory/SourceSetMeasurements.pkl",
        artifact_type=MeasurementsArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.CHANNEL,
        relations=source_set_measurements.relations,
    )

    plan = SimpleNamespace(
        axis_id="A01",
        step_index=0,
        step_name="SourceBoundAnchorFilter",
        main_input_dependency=StepInputDependency.pipeline_start(),
        source_binding_plan=source_binding_plan,
        execution_group_scope=ComponentGroupScope.from_raw(
            ("1", "2"),
            component=AllComponents.CHANNEL,
        ),
        compiled_function_pattern=compile_function_pattern(
            measure_source_set,
            {},
            {plan.ref(): plan for plan in (measurement_plan,)},
        ),
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
                "1": (
                    "A01_s001_w1_z001_t001.tif",
                    "A01_s002_w1_z001_t001.tif",
                ),
                "2": (
                    "A01_s001_w2_z001_t001.tif",
                    "A01_s002_w2_z001_t001.tif",
                ),
                "3": ("A01_s001_w3_z001_t001.tif",),
            }
        )
    )

    assert filtered.groups == {
        "1": (
            "A01_s001_w1_z001_t001.tif",
            "A01_s002_w1_z001_t001.tif",
        ),
        "2": (),
        "3": (),
    }


def test_callable_contract_source_inputs_project_bindings_through_exact_ref_authority() -> (
    None
):
    source_spec = ArtifactSpec.input("OrigBlue", ImageArtifactType)

    @artifact_inputs(source_spec)
    @runtime_adapter("runtime", lambda _request: object())
    def exact_source_input(image, *, runtime):
        del runtime
        return image

    compiled_group = compile_function_pattern(
        exact_source_input,
        {},
        {},
    ).default_group
    source_binding_plan = CompiledSourceBindingPlan(
        bindings=(
            NamedSourceBinding(alias="OrigBlue"),
            NamedSourceBinding(alias="OrigGreen"),
        )
    )

    invocation = compiled_group.invocations[0]
    source_refs = tuple(spec.ref() for spec in invocation.contract.artifact_inputs)
    assert source_refs == (source_spec.ref(),)
    assert tuple(
        binding.alias
        for binding in source_binding_plan.for_artifact_refs(
            source_refs
        ).binding_declarations
    ) == ("OrigBlue",)
    assert tuple(
        binding.alias for binding in source_binding_plan.binding_declarations
    ) == ("OrigBlue", "OrigGreen")


def test_execution_scope_excludes_cross_component_source_anchor_before_remap() -> None:
    source = ArtifactSpec.input("OrigBlue", ImageArtifactType)
    runtime_image = ArtifactSpec.input(
        "RGBImage",
        ImageArtifactType,
        parameter_name="image_to_save",
    )
    output = ArtifactSpec.output(
        "SavedRGBImage",
        ImageArtifactType,
        relations=(GroupLineageSourceRelation(source=runtime_image.ref()),),
    )

    @artifact_inputs(source, runtime_image)
    @artifact_outputs(output)
    @special_inputs("image_to_save")
    def save_image(image, *, image_to_save: np.ndarray):
        del image_to_save
        return image

    runtime_plan = ArtifactInputPlan(
        name=runtime_image.name,
        path="/memory/RGBImage.pkl",
        artifact_type=runtime_image.artifact_type,
    )
    output_plan = ArtifactOutputPlan(
        name=output.name,
        path="/memory/SavedRGBImage.pkl",
        artifact_type=output.artifact_type,
        relations=output.relations,
    )
    compiled_pattern = compile_function_pattern(
        {"3": save_image},
        {plan.ref(): plan for plan in (runtime_plan,)},
        {plan.ref(): plan for plan in (output_plan,)},
    )
    compiled_pattern = PathPlannerArtifactStage(
        PathPlanner.__new__(PathPlanner)
    ).compile_invocation_input_edges(
        compiled_pattern,
        artifact_inputs={runtime_plan.ref(): runtime_plan},
        relation_source_scopes={
            runtime_image.ref(): runtime_plan.producer_group_scope(),
        },
        execution_group_scope=ComponentGroupScope.from_raw(
            ("3",),
            component=AllComponents.CHANNEL,
        ),
        consumer_variable_components=ComponentSet(),
    )
    invocation = next(compiled_pattern.iter_invocations())
    assert tuple(edge.spec for edge in invocation.artifact_input_edges) == (
        source,
        runtime_image,
    )
    assert tuple(
        edge.spec for edge in invocation.artifact_input_edges
        if edge.storage_plan is not None
    ) == (runtime_image,)
    assert invocation.artifact_output_plans == (output_plan,)
    source_binding_plan = CompiledSourceBindingPlan(
        bindings=(
            NamedSourceBinding(
                alias="OrigBlue",
                selector=SourceSelector(
                    components=(ComponentSelector(AllComponents.CHANNEL, "1"),),
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
            ),
            NamedSourceBinding(
                alias="OrigRed",
                selector=SourceSelector(
                    components=(ComponentSelector(AllComponents.CHANNEL, "3"),),
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "3"),),
            ),
        ),
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )
    plan = SimpleNamespace(
        axis_id="A01",
        step_index=16,
        step_name="SaveImages",
        main_input_dependency=StepInputDependency.pipeline_start(),
        source_binding_plan=source_binding_plan,
        execution_group_scope=ComponentGroupScope.from_raw(
            ("3",),
            component=AllComponents.CHANNEL,
        ),
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
            "1": ("A01_s001_w1_z001_t001.tif",),
            "3": ("A01_s001_w3_z001_t001.tif",),
        }
    )

    source_anchors = pattern_filter.source_bound_anchor_patterns(grouped_patterns)
    assert source_anchors.groups == {
        "1": (),
        "3": (),
    }
    assert pattern_filter.execution_group_anchor_patterns(source_anchors).groups == {
        "3": (),
    }


def test_source_anchored_dict_pattern_excludes_out_of_scope_source_group() -> None:
    source = ArtifactSpec.input("OrigBlue", ImageArtifactType)

    @artifact_inputs(source)
    def exact_source_input(image):
        return image

    compiled_pattern = compile_function_pattern(
        {"1": exact_source_input},
        {},
        {},
    )
    source_binding_plan = CompiledSourceBindingPlan(
        bindings=(
            NamedSourceBinding(
                alias="OrigBlue",
                selector=SourceSelector(
                    components=(ComponentSelector(AllComponents.CHANNEL, "1"),),
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
            ),
        ),
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )
    plan = SimpleNamespace(
        axis_id="A01",
        step_index=0,
        step_name="ExactSourceInput",
        main_input_dependency=StepInputDependency.pipeline_start(),
        source_binding_plan=source_binding_plan,
        execution_group_scope=ComponentGroupScope.from_raw(
            ("1",),
            component=AllComponents.CHANNEL,
        ),
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
            "1": ("A01_s001_w1_z001_t001.tif",),
            "2": ("A01_s001_w2_z001_t001.tif",),
        }
    )

    source_anchors = pattern_filter.source_bound_anchor_patterns(grouped_patterns)

    assert source_anchors.groups == {
        "1": ("A01_s001_w1_z001_t001.tif",),
        "2": (),
    }
    assert pattern_filter.execution_group_anchor_patterns(source_anchors).groups == {
        "1": ("A01_s001_w1_z001_t001.tif",),
    }


def test_exact_source_artifact_filters_undeclared_detected_component_groups() -> None:
    source = ArtifactSpec.input("OrigBlue", ImageArtifactType)

    @artifact_inputs(source)
    def exact_source_input(image):
        return image

    source_binding_plan = CompiledSourceBindingPlan(
        bindings=(
            NamedSourceBinding(
                alias="OrigBlue",
                selector=SourceSelector(
                    components=(ComponentSelector(AllComponents.CHANNEL, "1"),),
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
            ),
        ),
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )
    plan = SimpleNamespace(
        axis_id="A01",
        step_index=0,
        step_name="ExactSourceInput",
        main_input_dependency=StepInputDependency.pipeline_start(),
        source_binding_plan=source_binding_plan,
        execution_group_scope=ComponentGroupScope.from_raw(
            ("1",),
            component=AllComponents.CHANNEL,
        ),
        compiled_function_pattern=compile_function_pattern(exact_source_input, {}, {}),
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
                "3": ("A01_s001_w3_z001_t001.tif",),
            }
        )
    )

    assert filtered.groups == {
        "1": ("A01_s001_w1_z001_t001.tif",),
        "2": (),
        "3": (),
    }


def test_source_bound_artifact_managed_step_keeps_source_anchors() -> None:
    output = ArtifactSpec.output("IllumStain1", ImageArtifactType)

    @artifact_outputs(output)
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
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
            ),
        ),
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )
    compiled_pattern = compile_function_pattern(
        source_bound_module,
        {},
        {plan.ref(): plan for plan in (ArtifactOutputPlan(
                name=output.name,
                path="/memory/IllumStain1.pkl",
                artifact_type=output.artifact_type,
            ),)},
    )
    plan = SimpleNamespace(
        axis_id="A01",
        main_input_dependency=StepInputDependency.pipeline_start(),
        source_binding_plan=source_binding_plan,
        execution_group_value=None,
        execution_group_scope=ComponentGroupScope.ungrouped(),
        compiled_function_pattern=compiled_pattern,
    )
    pattern_filter = StepAnchorPatternFilter(
        plan=plan,
        parser=SourceSchemaFilenameParser(),
        output_manifest=SimpleNamespace(
            filter_to_producer_paths=lambda _plan, paths, _parser: paths,
        ),
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

    assert pattern_filter.filtered(grouped_patterns).groups == grouped_patterns.groups


def test_default_callable_runtime_scope_projects_bindings_to_selected_group() -> None:
    from openhcs.core.steps.function_runtime import PatternGroupExecutionScope

    source_binding_plan = CompiledSourceBindingPlan(
        bindings=(
            NamedSourceBinding(
                alias="OrigStain1",
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
            ),
            NamedSourceBinding(
                alias="OrigStain2",
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "2"),),
            ),
        )
    )
    scope = PatternGroupExecutionScope(
        context=SimpleNamespace(),
        execution_plan=SimpleNamespace(
            axis_id="A01",
            execution_group_scope=ComponentGroupScope.dynamic(AllComponents.CHANNEL),
            source_binding_plan=source_binding_plan,
            compiled_function_pattern=compile_function_pattern(
                lambda image: image,
                {},
                {},
            ),
        ),
        compiled_group=compile_function_pattern(
            lambda image: image,
            {},
            {},
        ).default_group,
        component_value="1",
    )

    assert tuple(
        binding.alias for binding in scope.source_binding_plan.binding_declarations
    ) == ("OrigStain1",)


def test_dict_callable_runtime_scope_projects_bindings_to_selected_group() -> None:
    from openhcs.core.steps.function_runtime import PatternGroupExecutionScope

    source_binding_plan = CompiledSourceBindingPlan(
        bindings=(
            NamedSourceBinding(
                alias="OrigStain1",
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
            ),
            NamedSourceBinding(
                alias="OrigStain2",
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "2"),),
            ),
        )
    )
    compiled_pattern = compile_function_pattern(
        {"1": lambda image: image},
        {},
        {},
    )
    scope = PatternGroupExecutionScope(
        context=SimpleNamespace(),
        execution_plan=SimpleNamespace(
            axis_id="A01",
            execution_group_scope=ComponentGroupScope.dynamic(AllComponents.CHANNEL),
            source_binding_plan=source_binding_plan,
            compiled_function_pattern=compiled_pattern,
        ),
        compiled_group=compiled_pattern.require_group("1"),
        component_value="1",
    )

    assert tuple(
        binding.alias for binding in scope.source_binding_plan.binding_declarations
    ) == ("OrigStain1",)


def test_grouped_runtime_adapter_receives_component_selected_source_bindings() -> None:
    from openhcs.core.runtime_adapters import (
        RuntimeAdapterRequest,
        RuntimePlaneProjection,
    )
    from openhcs.core.source_load_plan import SourceLoadPlan
    from openhcs.core.steps.function_runtime import (
        ComponentArtifactPlans,
        FunctionRuntimeScope,
    )

    source_binding_plan = CompiledSourceBindingPlan(
        bindings=(
            NamedSourceBinding(
                alias="OrigStain1",
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
            ),
            NamedSourceBinding(
                alias="OrigStain2",
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "2"),),
            ),
        )
    )
    compiled_pattern = compile_function_pattern(
        lambda image: image,
        {},
        {},
    )
    execution_plan = SimpleNamespace(
        axis_id="A01",
        execution_group_scope=ComponentGroupScope.dynamic(AllComponents.CHANNEL),
        source_binding_plan=source_binding_plan,
        compiled_function_pattern=compiled_pattern,
        variable_components=(VariableComponents.SITE,),
        source_load_plan=SourceLoadPlan(),
    )
    scope = FunctionRuntimeScope(
        context=SimpleNamespace(),
        execution_plan=execution_plan,
        compiled_group=compiled_pattern.default_group,
        component_value="1",
        artifacts=ComponentArtifactPlans(inputs={}, outputs={}),
        source_binding_context=SourceBindingRuntimeContext.empty(),
        runtime_plane_index=0,
        runtime_plane_count=2,
    )

    request = RuntimeAdapterRequest.from_runtime_scope(
        runtime_scope=scope,
        artifact_inputs={},
        artifact_outputs={},
        group_key="1",
        plane_projection=RuntimePlaneProjection.stack(),
        source_payload=np.zeros((2, 3, 4), dtype=np.uint16),
    )

    assert tuple(
        binding.alias for binding in scope.source_binding_plan.binding_declarations
    ) == ("OrigStain1",)
    assert tuple(
        binding.alias for binding in request.source_binding_plan.binding_declarations
    ) == ("OrigStain1",)
    assert request.source_binding_context is scope.source_binding_context

    from openhcs.interop.cellprofiler.runtime.module_execution import (
        cellprofiler_runtime_adapter_factory,
    )

    adapter = cellprofiler_runtime_adapter_factory(request)
    assert adapter.request is request
    assert not hasattr(adapter, "artifact_inputs")


def test_runtime_invocation_uses_only_active_source_bound_main_flow_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openhcs.core.artifacts import NoMainFlowOutput
    from openhcs.core.steps import function_runtime
    from openhcs.core.source_load_plan import SourceLoadPlan
    from openhcs.core.steps.function_runtime import (
        ComponentArtifactPlans,
        FunctionRuntimeScope,
    )

    source_binding_plan = CompiledSourceBindingPlan(
        bindings=tuple(
            NamedSourceBinding(
                alias=alias,
                component_identity=(
                    ComponentSelector(AllComponents.CHANNEL, channel),
                ),
            )
            for alias, channel in (("OrigStain1", "1"), ("OrigStain2", "2"))
        )
    )
    source_specs = tuple(
        binding.input_spec()
        for binding in source_binding_plan.binding_declarations
    )
    output_specs = tuple(
        ArtifactSpec.output(
            f"IllumStain{channel}",
            ImageArtifactType,
            relations=(GroupLineageSourceRelation(source=source.ref()),),
        )
        for channel, source in enumerate(source_specs, start=1)
    )
    output_plans = {
        spec.ref(): ArtifactOutputPlan(
            name=spec.name,
            path=f"/memory/{spec.name}.pkl",
            artifact_type=spec.artifact_type,
            group_keys=(channel,),
            group_component=AllComponents.CHANNEL,
            paths_by_group={channel: f"/memory/{spec.name}__{channel}.pkl"},
            relations=spec.relations,
        )
        for channel, spec in zip(("1", "2"), output_specs, strict=True)
    }

    @artifact_inputs(*source_specs)
    @artifact_outputs(*output_specs)
    def consume_channel_source(image):
        return image

    compiled_pattern = compile_function_pattern(
        consume_channel_source,
        {},
        output_plans,
    )
    invocation = compiled_pattern.default_group.invocations[0]
    invocation = invocation.with_artifact_input_edges(
        tuple(
            InvocationArtifactInputEdgePlan(
                key=edge_key,
                spec=spec,
                storage_plan=None,
                projection=None,
                consumes_main_flow=True,
            )
            for edge_key, spec in zip(
                InvocationArtifactInputProjectionKey.for_input_count(
                    invocation.key,
                    len(source_specs),
                ),
                source_specs,
                strict=True,
            )
        )
    )
    compiled_group = replace(
        compiled_pattern.default_group,
        invocations=(invocation,),
    )
    execution_plan = SimpleNamespace(
        axis_id="A01",
        execution_group_scope=ComponentGroupScope.dynamic(AllComponents.CHANNEL),
        source_binding_plan=source_binding_plan,
        input_memory_type="numpy",
        variable_components=(VariableComponents.SITE,),
        source_load_plan=SourceLoadPlan(),
        artifact_inputs={},
        artifact_outputs=output_plans,
    )
    scope = FunctionRuntimeScope(
        context=SimpleNamespace(),
        execution_plan=execution_plan,
        compiled_group=compiled_group,
        component_value="1",
        artifacts=ComponentArtifactPlans.from_step_component(execution_plan, "1"),
        source_binding_context=SourceBindingRuntimeContext.empty(),
        runtime_plane_index=0,
        runtime_plane_count=1,
    )
    captured_executor_kwargs = []
    core_executor_type = function_runtime.FunctionCoreExecutor

    class CapturingExecutor:
        def __init__(self, **kwargs):
            captured_executor_kwargs.append(kwargs)

        def execute(self, *, debug_sink=None):
            return NoMainFlowOutput()

    monkeypatch.setattr(function_runtime, "FunctionCoreExecutor", CapturingExecutor)
    monkeypatch.setattr(
        function_runtime,
        "debug_event_sink_from_context",
        lambda context: SimpleNamespace(captures_invocation_events=lambda: False),
    )

    result = scope.execute_chain(np.zeros((1, 3, 4), dtype=np.uint16))

    assert isinstance(result, NoMainFlowOutput)
    assert tuple(
        edge.spec.name
        for edge in captured_executor_kwargs[0]["invocation"].artifact_input_edges
    ) == ("OrigStain1", "OrigStain2")
    selected_artifacts = captured_executor_kwargs[0]["artifacts"]
    assert tuple(edge.spec.name for edge in selected_artifacts.inputs.values()) == (
        "OrigStain1",
    )
    assert tuple(selected_artifacts.outputs) == (output_specs[0].ref(),)

    request = core_executor_type(**captured_executor_kwargs[0]).runtime_adapter_request(
        np.zeros((1, 3, 4), dtype=np.uint16)
    )

    assert tuple(edge.spec.name for edge in request.artifact_inputs.values()) == (
        "OrigStain1",
    )
    assert request.selected_artifact_input_specs().names() == ("OrigStain1",)


def test_invocation_source_artifact_owns_cross_component_runtime_binding_scope() -> (
    None
):
    from openhcs.core.steps.function_runtime import PatternGroupExecutionScope

    source_binding_plan = CompiledSourceBindingPlan(
        bindings=(
            NamedSourceBinding(
                alias="OrigStain1",
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
            ),
            NamedSourceBinding(
                alias="OrigStain2",
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "2"),),
            ),
        )
    )
    source_spec = source_binding_plan.binding_declarations[0].input_spec()

    @artifact_inputs(source_spec)
    def exact_source_input(image):
        return image

    compiled_pattern = compile_function_pattern(exact_source_input, {}, {})
    scope = PatternGroupExecutionScope(
        context=SimpleNamespace(),
        execution_plan=SimpleNamespace(
            axis_id="A01",
            execution_group_scope=ComponentGroupScope.dynamic(AllComponents.SITE),
            source_binding_plan=source_binding_plan,
            compiled_function_pattern=compiled_pattern,
        ),
        compiled_group=compiled_pattern.default_group,
        component_value="1",
    )

    assert tuple(
        binding.alias for binding in scope.source_binding_plan.binding_declarations
    ) == ("OrigStain1",)


def test_main_flow_input_owns_runtime_binding_scope_with_auxiliary_source() -> None:
    from openhcs.core.steps.function_runtime import PatternGroupExecutionScope

    source_binding_plan = CompiledSourceBindingPlan(
        bindings=(
            NamedSourceBinding(
                alias="OrigStain1",
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
            ),
            NamedSourceBinding(
                alias="OrigStain2",
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "2"),),
            ),
        )
    )
    source_spec = source_binding_plan.binding_declarations[0].input_spec()
    main_flow_spec = source_binding_plan.binding_declarations[1].input_spec()

    @artifact_inputs(main_flow_spec, source_spec)
    def main_flow_with_auxiliary_source(image):
        return image

    compiled_pattern = compile_function_pattern(
        main_flow_with_auxiliary_source,
        {},
        {},
    )
    invocation = compiled_pattern.default_group.invocations[0]
    invocation = invocation.with_artifact_input_edges(
        tuple(
            InvocationArtifactInputEdgePlan(
                key=edge_key,
                spec=spec,
                storage_plan=None,
                projection=None,
                consumes_main_flow=input_index == 0,
            )
            for input_index, (edge_key, spec) in enumerate(
                zip(
                    InvocationArtifactInputProjectionKey.for_input_count(
                        invocation.key,
                        2,
                    ),
                    (main_flow_spec, source_spec),
                    strict=True,
                )
            )
        )
    )
    compiled_pattern = replace(
        compiled_pattern,
        groups=(
            replace(
                compiled_pattern.default_group,
                invocations=(invocation,),
            ),
        ),
    )
    scope = PatternGroupExecutionScope(
        context=SimpleNamespace(),
        execution_plan=SimpleNamespace(
            axis_id="A01",
            execution_group_scope=ComponentGroupScope.dynamic(AllComponents.CHANNEL),
            source_binding_plan=source_binding_plan,
            compiled_function_pattern=compiled_pattern,
        ),
        compiled_group=compiled_pattern.default_group,
        component_value="2",
    )

    assert tuple(
        binding.alias for binding in scope.source_binding_plan.binding_declarations
    ) == ("OrigStain1", "OrigStain2")
    assert tuple(
        binding.alias
        for binding in scope.main_flow_source_binding_plan.binding_declarations
    ) == ("OrigStain2",)


def test_main_flow_source_scope_intersects_cross_component_invocation_inputs() -> None:
    from openhcs.core.steps.function_runtime import PatternGroupExecutionScope

    source_binding_plan = CompiledSourceBindingPlan(
        bindings=tuple(
            NamedSourceBinding(
                alias=alias,
                component_identity=(
                    ComponentSelector(AllComponents.CHANNEL, channel),
                ),
            )
            for alias, channel in (
                ("Worms", "1"),
                ("GFP", "2"),
                ("mCherry", "3"),
            )
        )
    )
    selected_specs = tuple(
        binding.input_spec()
        for binding in source_binding_plan.binding_declarations[1:]
    )

    @artifact_inputs(*selected_specs)
    def consume_selected_channels(image):
        return image

    compiled_pattern = compile_function_pattern(consume_selected_channels, {}, {})
    invocation = compiled_pattern.default_group.invocations[0]
    invocation = invocation.with_artifact_input_edges(
        tuple(
            InvocationArtifactInputEdgePlan(
                key=edge_key,
                spec=spec,
                storage_plan=None,
                projection=None,
                consumes_main_flow=True,
            )
            for edge_key, spec in zip(
                InvocationArtifactInputProjectionKey.for_input_count(
                    invocation.key,
                    len(selected_specs),
                ),
                selected_specs,
                strict=True,
            )
        )
    )
    compiled_pattern = replace(
        compiled_pattern,
        groups=(
            replace(
                compiled_pattern.default_group,
                invocations=(invocation,),
            ),
        ),
    )
    scope = PatternGroupExecutionScope(
        context=SimpleNamespace(),
        execution_plan=SimpleNamespace(
            axis_id="A01",
            execution_group_scope=ComponentGroupScope.dynamic(AllComponents.SITE),
            source_binding_plan=source_binding_plan,
            compiled_function_pattern=compiled_pattern,
        ),
        compiled_group=compiled_pattern.default_group,
        component_value="1",
    )

    assert tuple(
        binding.alias
        for binding in scope.main_flow_source_binding_plan.binding_declarations
    ) == ("GFP", "mCherry")
    assert tuple(
        binding.alias for binding in scope.source_binding_plan.binding_declarations
    ) == ("GFP", "mCherry")


def test_auxiliary_cross_group_source_does_not_anchor_main_flow() -> None:
    from openhcs.core.steps.function_runtime import PatternGroupExecutionScope

    source_binding_plan = CompiledSourceBindingPlan(
        bindings=(
            NamedSourceBinding(
                alias="ColorFluor",
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
            ),
        )
    )
    auxiliary_source = source_binding_plan.binding_declarations[0].input_spec()
    producer_image = ArtifactSpec.input("TumorOutline", ImageArtifactType)

    @artifact_inputs(auxiliary_source, producer_image)
    def save_cross_group_image(image):
        return image

    compiled_pattern = compile_function_pattern(save_cross_group_image, {}, {})
    invocation = compiled_pattern.default_group.invocations[0]
    invocation = invocation.with_artifact_input_edges(
        tuple(
            InvocationArtifactInputEdgePlan(
                key=edge_key,
                spec=spec,
                storage_plan=None,
                projection=None,
                consumes_main_flow=False,
            )
            for edge_key, spec in zip(
                InvocationArtifactInputProjectionKey.for_input_count(
                    invocation.key,
                    2,
                ),
                (auxiliary_source, producer_image),
                strict=True,
            )
        )
    )
    compiled_pattern = replace(
        compiled_pattern,
        groups=(
            replace(
                compiled_pattern.default_group,
                invocations=(invocation,),
            ),
        ),
    )
    scope = PatternGroupExecutionScope(
        context=SimpleNamespace(),
        execution_plan=SimpleNamespace(
            axis_id="A01",
            execution_group_scope=ComponentGroupScope.dynamic(AllComponents.CHANNEL),
            source_binding_plan=source_binding_plan,
            compiled_function_pattern=compiled_pattern,
        ),
        compiled_group=compiled_pattern.default_group,
        component_value="2",
    )

    assert not scope.main_flow_source_binding_plan.binding_declarations
    assert tuple(
        binding.alias for binding in scope.source_binding_plan.binding_declarations
    ) == ("ColorFluor",)


def test_runtime_plane_count_comes_from_loaded_slices_not_dispatch_groups() -> None:
    from openhcs.core.steps.function_runtime import (
        FunctionRuntimeScope,
        PatternGroupData,
        PatternGroupExecutionRequest,
    )

    plan = SimpleNamespace(
        axis_id="A01",
        execution_group_scope=ComponentGroupScope.dynamic(AllComponents.CHANNEL),
        source_binding_plan=CompiledSourceBindingPlan(),
        artifact_inputs={},
        artifact_outputs={},
        variable_components=(VariableComponents.SITE,),
    )
    request = PatternGroupExecutionRequest(
        context=SimpleNamespace(),
        execution_plan=plan,
        compiled_group=compile_function_pattern(
            lambda image: image,
            {},
            {},
        ).default_group,
        component_value="1",
        pattern_group_info="A01_s{iii}_w1_z001_t001.tif",
        component_index=0,
        component_count=1,
    )
    loaded = PatternGroupData(
        matching_files=[
            "A01_s001_w1_z001_t001.tif",
            "A01_s002_w1_z001_t001.tif",
        ],
        main_data_stack=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "A01_s001_w1_z003_t002.tif",
                    "A01_s002_w1_z003_t002.tif",
                ),
                component_metadata=(
                    {
                        "well": "A01",
                        "site": "1",
                        "channel": "1",
                        "z_index": "3",
                        "timepoint": "2",
                        "extension": ".tif",
                    },
                    {
                        "well": "A01",
                        "site": "2",
                        "channel": "1",
                        "z_index": "3",
                        "timepoint": "2",
                        "extension": ".tif",
                    },
                ),
            ),
        ).payload_with(np.zeros((2, 4, 5), dtype=np.float32)),
    )

    scope = FunctionRuntimeScope.from_pattern_group(request, loaded)

    assert scope.runtime_plane_count == 2
    assert scope.axis_scope.fixed_component_values == (
        (AllComponents.Z_INDEX, "3"),
        (AllComponents.TIMEPOINT, "2"),
    )


def test_grouped_main_flow_context_uses_component_selected_output_plan() -> None:
    from openhcs.core.steps.function_runtime import (
        PatternGroupExecutionRequest,
        PatternGroupRuntime,
    )

    corrected_stain_1_spec = ArtifactSpec.output(
        "CorrectedStain1",
        ImageArtifactType,
    )
    corrected_stain_2_spec = ArtifactSpec.output(
        "CorrectedStain2",
        ImageArtifactType,
    )
    corrected_stain_1 = ArtifactOutputPlan(
        name=corrected_stain_1_spec.name,
        path="/memory/CorrectedStain1.pkl",
        artifact_type=corrected_stain_1_spec.artifact_type,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": "/memory/CorrectedStain1_1.pkl"},
    )
    corrected_stain_2 = ArtifactOutputPlan(
        name=corrected_stain_2_spec.name,
        path="/memory/CorrectedStain2.pkl",
        artifact_type=corrected_stain_2_spec.artifact_type,
        group_keys=("2",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"2": "/memory/CorrectedStain2_2.pkl"},
    )
    plan = SimpleNamespace(
        artifact_inputs={},
        artifact_outputs={
            plan.ref(): plan
            for plan in (corrected_stain_1, corrected_stain_2)
        },
        execution_group_scope=ComponentGroupScope.dynamic(AllComponents.CHANNEL),
        source_binding_plan=CompiledSourceBindingPlan(),
    )
    @artifact_outputs(
        corrected_stain_1_spec,
        corrected_stain_2_spec,
    )
    def correct_illumination(image):
        return image

    compiled_group = compile_function_pattern(
        correct_illumination,
        {},
        {plan.ref(): plan for plan in (corrected_stain_1, corrected_stain_2)},
    ).default_group
    runtime = PatternGroupRuntime(
        PatternGroupExecutionRequest(
            context=SimpleNamespace(),
            execution_plan=plan,
            compiled_group=compiled_group,
            component_value="1",
            pattern_group_info="A01_s{iii}_w1_z001_t001.tif",
            component_index=0,
            component_count=2,
        )
    )

    context = runtime._unwrapped_main_flow_output_context()

    assert context is not None
    assert context.output_key == "CorrectedStain1"
    assert context.artifact_kind == ImageArtifactType.value


def test_adapter_recorded_outputs_use_compiled_canonical_context() -> None:
    from openhcs.core.steps.function_runtime import (
        PatternGroupExecutionRequest,
        PatternGroupRuntime,
    )

    outline_spec = ArtifactSpec.output("outline", ImageArtifactType)
    first_labels_spec = ArtifactSpec.output("first_labels", ObjectLabelsArtifactType)
    second_labels_spec = ArtifactSpec.output("second_labels", ObjectLabelsArtifactType)
    output_plans = {
        spec.ref(): ArtifactOutputPlan(
            name=spec.name,
            path=f"/memory/{spec.name}.pkl",
            artifact_type=spec.artifact_type,
        )
        for spec in (outline_spec, first_labels_spec, second_labels_spec)
    }

    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_outputs=True,
    )
    @artifact_outputs(outline_spec, first_labels_spec, second_labels_spec)
    def record_mixed_outputs(image, *, runtime):
        del runtime
        return image

    compiled_group = compile_function_pattern(
        record_mixed_outputs,
        {},
        output_plans,
    ).default_group
    runtime = PatternGroupRuntime(
        PatternGroupExecutionRequest(
            context=SimpleNamespace(),
            execution_plan=SimpleNamespace(
                artifact_inputs={},
                artifact_outputs=output_plans,
                execution_group_scope=ComponentGroupScope.ungrouped(),
                source_binding_plan=CompiledSourceBindingPlan.empty(),
            ),
            compiled_group=compiled_group,
            component_value="default",
            pattern_group_info="A01_s001_w1_z001_t001.tif",
            component_index=0,
            component_count=1,
        )
    )

    context = runtime._unwrapped_main_flow_output_context()

    assert context is not None
    assert context.output_key == "outline"
    assert context.artifact_kind == ImageArtifactType.value


def test_component_output_selection_keeps_distinct_axes_with_equal_keys() -> None:
    from openhcs.core.steps.function_runtime import ComponentArtifactPlans

    channel_1 = ArtifactOutputPlan(
        name="Stain1",
        path="/memory/Stain1.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": "/memory/Stain1_1.pkl"},
    )
    channel_2 = ArtifactOutputPlan(
        name="Stain2",
        path="/memory/Stain2.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("2",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"2": "/memory/Stain2_2.pkl"},
    )
    plan = SimpleNamespace(
        artifact_inputs={},
        artifact_outputs={
            plan.ref(): plan for plan in (channel_1, channel_2)
        },
        execution_group_scope=ComponentGroupScope.dynamic(AllComponents.SITE),
    )

    selected = ComponentArtifactPlans.from_step_component(plan, "1")

    assert tuple(selected.outputs) == (channel_1.ref(), channel_2.ref())
    assert selected.outputs[channel_1.ref()].path == "/memory/Stain1_1.pkl"
    assert selected.outputs[channel_2.ref()].path == "/memory/Stain2_2.pkl"


def test_component_artifact_plans_reject_malformed_exact_plan_maps() -> None:
    from openhcs.core.steps.function_runtime import ComponentArtifactPlans

    input_plan = ArtifactInputPlan(
        name="InputImage",
        path="/memory/InputImage.pkl",
        artifact_type=ImageArtifactType,
    )
    output_plan = ArtifactOutputPlan(
        name="OutputImage",
        path="/memory/OutputImage.pkl",
        artifact_type=ImageArtifactType,
    )
    invalid_maps = (
        (
            {input_plan.name: input_plan},
            {},
            TypeError,
            "input maps require ArtifactSpecRef keys",
        ),
        (
            {input_plan.ref(): output_plan},
            {},
            TypeError,
            "input maps require ArtifactInputPlan values",
        ),
        (
            {
                ArtifactSpec.input("OtherInput", ImageArtifactType).ref(): (
                    input_plan
                )
            },
            {},
            ValueError,
            "input key .* conflicts with plan ref",
        ),
        (
            {},
            {output_plan.name: output_plan},
            TypeError,
            "output maps require ArtifactSpecRef keys",
        ),
        (
            {},
            {output_plan.ref(): input_plan},
            TypeError,
            "output maps require ArtifactOutputPlan values",
        ),
        (
            {},
            {
                ArtifactSpec.output("OtherOutput", ImageArtifactType).ref(): (
                    output_plan
                )
            },
            ValueError,
            "output key .* conflicts with plan ref",
        ),
    )

    for invalid_inputs, invalid_outputs, error_type, message in invalid_maps:
        step_plan = SimpleNamespace(
            artifact_inputs=invalid_inputs,
            artifact_outputs=invalid_outputs,
            execution_group_scope=ComponentGroupScope.ungrouped(),
        )
        with pytest.raises(error_type, match=message):
            ComponentArtifactPlans.from_step_component(step_plan, None)


def test_grouped_runtime_scope_preserves_empty_source_binding_plan() -> None:
    from openhcs.core.steps.function_runtime import PatternGroupExecutionScope

    source_binding_plan = CompiledSourceBindingPlan.empty()
    compiled_pattern = compile_function_pattern(lambda image: image, {}, {})
    scope = PatternGroupExecutionScope(
        context=SimpleNamespace(),
        execution_plan=SimpleNamespace(
            axis_id="A01",
            execution_group_scope=ComponentGroupScope.dynamic(AllComponents.SITE),
            source_binding_plan=source_binding_plan,
        ),
        compiled_group=compiled_pattern.default_group,
        component_value="1",
    )

    assert scope.source_binding_plan is source_binding_plan


def test_grouped_runtime_source_expansion_uses_scoped_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openhcs.core.steps import function_runtime

    source_binding_plan = CompiledSourceBindingPlan(
        bindings=(
            NamedSourceBinding(
                alias="OrigStain1",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "N_R",
                        ),
                    )
                ),
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
            ),
            NamedSourceBinding(
                alias="OrigStain2",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "N_G",
                        ),
                    )
                ),
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "2"),),
            ),
        )
    )
    compiled_pattern = compile_function_pattern(
        {"1": lambda image: image},
        {},
        {},
    )
    runtime = function_runtime.PatternGroupRuntime(
        function_runtime.PatternGroupExecutionRequest(
            context=SimpleNamespace(
                source_image_set_identity_policy=SourceImageSetIdentityPolicy(
                    frozenset((AllComponents.SITE,))
                )
            ),
            execution_plan=SimpleNamespace(
                axis_id="A01",
                execution_group_scope=ComponentGroupScope.dynamic(
                    AllComponents.CHANNEL
                ),
                main_input_dependency=StepInputDependency.pipeline_start(),
                source_binding_plan=source_binding_plan,
                compiled_function_pattern=compiled_pattern,
                variable_component_values=(VariableComponents.SITE.value,),
            ),
            compiled_group=compiled_pattern.require_group("1"),
            component_value="1",
            pattern_group_info="A01_s{iii}_w1_z001_t001.png",
            component_index=0,
            component_count=2,
        )
    )
    captured_aliases: tuple[str, ...] = ()

    class MatchedImageSet:
        def expand(self, matching_files, *, source_universe):
            del source_universe
            return tuple(matching_files)

    def matched_image_set_from_plan(*, bindings, **_kwargs):
        nonlocal captured_aliases
        captured_aliases = tuple(binding.alias for binding in bindings)
        return MatchedImageSet()

    monkeypatch.setattr(
        function_runtime.SourceBindingMatchedImageSet,
        "from_plan",
        matched_image_set_from_plan,
    )
    monkeypatch.setattr(
        runtime,
        "_source_binding_candidate_context",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        runtime,
        "_source_binding_load_universe",
        lambda: (),
    )

    matching_files = ["/input/A01_s001_N_R.png"]
    assert runtime._filter_matching_files_for_source_bindings(matching_files) == (
        matching_files
    )
    assert captured_aliases == ("OrigStain1",)


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


def test_disabled_source_binding_plan_does_not_filter_runtime_inputs() -> None:
    from openhcs.core.steps.function_runtime import (
        PatternGroupExecutionRequest,
        PatternGroupRuntime,
    )

    compiled_pattern = compile_function_pattern(lambda image: image, {}, {})
    plan = SimpleNamespace(
        axis_id="A01",
        execution_group_scope=ComponentGroupScope.ungrouped(),
        main_input_dependency=StepInputDependency.pipeline_start(),
        source_binding_plan=CompiledSourceBindingPlan(
            bindings=(
                NamedSourceBinding(
                    alias="InheritedDNA",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.CONTAINS,
                                "DNA",
                            ),
                        )
                    ),
                ),
            ),
            enabled=False,
        ),
    )
    runtime = PatternGroupRuntime.__new__(PatternGroupRuntime)
    runtime.request = PatternGroupExecutionRequest(
        context=SimpleNamespace(),
        execution_plan=plan,
        compiled_group=compiled_pattern.default_group,
        component_value=None,
        pattern_group_info="A14_s001_w3_z001_t001.tif",
        component_index=0,
        component_count=1,
    )
    matching_files = ["/tmp/outputs/A14_s001_w3_z001_t001.tif"]

    assert (
        runtime._filter_matching_files_for_source_bindings(matching_files)
        is matching_files
    )


def test_producer_anchored_pipeline_start_paths_use_exact_source_projection_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Producer bookkeeping must not override exact workspace source ownership."""

    from openhcs.core.steps import function_runtime

    virtual_paths = (
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w2_z001_t001.tif",
    )
    aliases = ("OrigColor", "PlateTemplate")
    source_paths = (
        tmp_path / "source" / "color.tif",
        tmp_path / "source" / "template.tif",
    )
    source_planes = tuple(
        SourcePlaneProjection(
            address=OpenHCSPlaneAddress(
                well="A01",
                site="1",
                channel=str(channel),
                z_index="1",
                timepoint="1",
            ),
            ref=SourcePixelRef("disk", str(source_path)),
            source_alias=alias,
        )
        for channel, alias, source_path in zip(
            (1, 2),
            aliases,
            source_paths,
            strict=True,
        )
    )
    projection = VirtualWorkspaceSourceProjection(
        source_refs_by_virtual_path={
            virtual_path: source_plane.ref
            for virtual_path, source_plane in zip(
                virtual_paths,
                source_planes,
                strict=True,
            )
        },
        source_metadata_by_path={
            virtual_paths[0]: {
                SOURCE_BINDING_ALIAS_METADATA_FIELD: aliases[0],
                "specimen": "sample",
            },
            virtual_paths[1]: {
                SOURCE_BINDING_ALIAS_METADATA_FIELD: aliases[1],
            },
        },
        source_projections_by_virtual_path={
            virtual_path: source_plane
            for virtual_path, source_plane in zip(
                virtual_paths,
                source_planes,
                strict=True,
            )
        },
        workspace_root=str(tmp_path),
    )
    source_binding_plan = CompiledSourceBindingPlan(
        bindings=(
            NamedSourceBinding(
                alias="OrigColor",
                source_channel_axis=-1,
                source_channel_counts=frozenset({3}),
            ),
            NamedSourceBinding(alias="PlateTemplate"),
        )
    )

    class SourceManifest:
        @staticmethod
        def producer_paths_matching_pattern(*_args):
            return list(virtual_paths)

        @staticmethod
        def filter_to_producer_paths(_plan, paths, _parser):
            return list(paths)

    class SourceFileManager:
        @staticmethod
        def load_batch(paths, backend):
            assert backend == "memory"
            payloads = {
                virtual_paths[0]: np.arange(60, dtype=np.float32).reshape(4, 5, 3),
                virtual_paths[1]: np.full((4, 5), 7, dtype=np.float32),
            }
            return [payloads[Path(path).name] for path in paths]

        @staticmethod
        def resolve_address(backend_address, backend, *, base_path):
            del base_path
            assert backend == "disk"
            return backend_address

    monkeypatch.setattr(
        function_runtime,
        "step_output_manifest",
        lambda _context: SourceManifest(),
    )
    monkeypatch.setattr(
        function_runtime.SourceBindingRuntimeContextRequest,
        "from_context",
        classmethod(
            lambda cls, **_kwargs: SimpleNamespace(
                runtime_context=SourceBindingRuntimeContext.empty
            )
        ),
    )
    monkeypatch.setattr(
        function_runtime.PatternGroupRuntime,
        "source_workspace_projection_authority",
        lambda _self: SimpleNamespace(projection_if_available=lambda: projection),
    )

    compiled_pattern = compile_function_pattern(lambda image: image, {}, {})
    plan = SimpleNamespace(
        axis_id="A01",
        input_dir=tmp_path,
        input_memory_type="numpy",
        variable_components=(),
        variable_component_values=(),
        device_id=0,
        step_index=0,
        step_name="PipelineStart",
        main_input_dependency=StepInputDependency.pipeline_start(),
        execution_group_value=None,
        source_binding_plan=source_binding_plan,
    )
    context = SimpleNamespace(
        microscope_handler=SimpleNamespace(
            parser=SourceSchemaFilenameParser(),
            path_list_from_pattern=lambda *_args: list(virtual_paths),
        ),
        filemanager=SourceFileManager(),
        runtime_image_stack_cache=RuntimeImageStackCache(),
    )
    runtime = function_runtime.PatternGroupRuntime(
        function_runtime.PatternGroupExecutionRequest(
            context=context,
            execution_plan=plan,
            compiled_group=compiled_pattern.default_group,
            pattern_group_info="A01_s001_w{iii}_z001_t001.tif",
            component_index=0,
            component_count=1,
        )
    )

    loaded = runtime._load_input_stack()

    data = image_payload_data(loaded.main_data_stack)
    metadata = image_payload_metadata(loaded.main_data_stack)
    assert data.shape == (2, 4, 5, 3)
    np.testing.assert_array_equal(data[1], np.full((4, 5, 3), 7, dtype=np.float32))
    assert metadata.source_image_names == aliases
    assert metadata.plane_axis is RuntimePlaneAxis.SOURCE_BINDING
    assert metadata.source_channel_axis == 3
    assert metadata.source_component_metadata["specimen"] == "sample"


def test_step_output_load_preserves_producer_stack_plane_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Derived stacks must not be rebound as pipeline-source images on load."""

    from openhcs.core.steps import function_runtime

    output_path = tmp_path / "A01_s001_w2_z001_t001_RescaledDNA.tif"
    producer_payload = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(str(output_path), str(output_path)),
            component_metadata=(
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "2",
                    "z_index": "1",
                    "timepoint": "1",
                },
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "2",
                    "z_index": "2",
                    "timepoint": "1",
                },
            ),
        ),
    ).payload_with(np.zeros((2, 4, 5), dtype=np.float32), None)

    class ProducerManifest:
        def producer_paths_matching_pattern(self, *_args):
            return [str(output_path)]

        def filter_to_producer_paths(self, _plan, paths, _parser):
            return list(paths)

    class MemoryFileManager:
        def load_batch(self, paths, backend):
            assert paths == [str(output_path)]
            assert backend == "memory"
            return [producer_payload]

    source_context = SourceBindingRuntimeContext()
    monkeypatch.setattr(
        function_runtime,
        "step_output_manifest",
        lambda _context: ProducerManifest(),
    )
    monkeypatch.setattr(
        function_runtime.SourceBindingRuntimeContextRequest,
        "from_context",
        classmethod(
            lambda cls, **_kwargs: SimpleNamespace(
                runtime_context=lambda: source_context
            )
        ),
    )
    monkeypatch.setattr(
        function_runtime.PatternGroupRuntime,
        "source_workspace_projection_authority",
        lambda _self: SimpleNamespace(
            projection_if_available=lambda: VirtualWorkspaceSourceProjection.empty()
        ),
    )
    monkeypatch.setattr(
        function_runtime.PatternGroupRuntime,
        "_filter_matching_files_for_group",
        lambda _self, paths: paths,
    )
    monkeypatch.setattr(
        function_runtime.PatternGroupRuntime,
        "_filter_matching_files_for_source_bindings",
        lambda _self, paths: paths,
    )
    plan = SimpleNamespace(
        input_dir=tmp_path,
        input_memory_type="numpy",
        variable_components=(VariableComponents.Z_INDEX,),
        device_id=0,
        step_index=1,
        step_name="Resize",
    )
    context = SimpleNamespace(
        microscope_handler=SimpleNamespace(parser=SourceSchemaFilenameParser()),
        filemanager=MemoryFileManager(),
        runtime_image_stack_cache=RuntimeImageStackCache(),
    )
    runtime = function_runtime.PatternGroupRuntime(
        SimpleNamespace(
            context=context,
            execution_plan=plan,
            compiled_group=SimpleNamespace(
                runtime_domain=RuntimeInvocationDomain.SOURCE_ANCHORED,
            ),
            pattern_group_info="A01_s001_w2_z{iii}_t001.tif",
        )
    )

    loaded = runtime._load_input_stack()

    provenance_planes = image_payload_metadata(
        loaded.main_data_stack
    ).source_image_provenance_planes
    assert provenance_planes.count == 2
    assert provenance_planes.contributor_count == 0
    plane_metadata = provenance_planes.component_metadata
    assert tuple(metadata["z_index"] for metadata in plane_metadata) == ("1", "2")


def test_artifact_managed_group_uses_compiler_group_without_filtering_anchor_files() -> (
    None
):
    from openhcs.core.steps.function_runtime import PatternGroupRuntime

    runtime = PatternGroupRuntime.__new__(PatternGroupRuntime)
    runtime.request = SimpleNamespace(
        compiled_group=SimpleNamespace(
            runtime_domain=RuntimeInvocationDomain.ARTIFACT_MANAGED,
        ),
        execution_plan=SimpleNamespace(
            execution_group_value="channel",
            main_input_dependency=StepInputDependency.pipeline_start(),
        ),
        component_value="1",
    )
    matching_files = ["/tmp/outputs/A01_s001_w2_z001_t001.tif"]

    assert runtime._filter_matching_files_for_group(matching_files) is matching_files


def test_step_output_group_does_not_reinterpret_producer_path_component() -> None:
    from openhcs.core.steps.function_runtime import PatternGroupRuntime

    runtime = PatternGroupRuntime.__new__(PatternGroupRuntime)
    runtime.request = SimpleNamespace(
        compiled_group=SimpleNamespace(
            runtime_domain=RuntimeInvocationDomain.SOURCE_ANCHORED,
        ),
        execution_plan=SimpleNamespace(
            execution_group_value="channel",
            main_input_dependency=StepInputDependency.step_output(
                source_step_index=4,
                source_step_scope_id="object_to_image",
            ),
        ),
        component_value="0",
    )
    matching_files = ["/tmp/outputs/A01_s001_w2_z001_t001.tif"]

    assert runtime._filter_matching_files_for_group(matching_files) is matching_files


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


def test_step_output_manifest_does_not_filter_main_flow_by_artifact_input(
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
            plan.ref(): plan
            for plan in (
                ArtifactInputPlan(
                    name="CorrDNA",
                    path="CorrDNA",
                    artifact_type=ImageArtifactType,
                    source_step_id=1,
                ),
            )
        },
        compiled_function_pattern=compile_function_pattern(lambda image: image, {}, {}),
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
    ) == [
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w2_z001_t001.tif",
    ]


def test_step_output_manifest_filters_declared_main_flow_contract_identity(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "images"
    producer = SimpleNamespace(
        step_scope_id="align",
        step_name="Align",
        pipeline_position=2,
        axis_id="A01",
        output_dir=output_dir,
    )
    input_spec = ArtifactSpec.input("Stain1", ImageArtifactType)

    @artifact_inputs(input_spec)
    def identify_stain_1(image):
        return image

    compiled_pattern = compile_function_pattern(identify_stain_1, {}, {})
    invocation = compiled_pattern.default_group.invocations[0]
    invocation = invocation.with_artifact_input_edges(
        (
            InvocationArtifactInputEdgePlan(
                key=InvocationArtifactInputProjectionKey(
                    invocation_key=invocation.key,
                    input_index=0,
                ),
                spec=input_spec,
                storage_plan=None,
                projection=None,
                consumes_main_flow=True,
            ),
        )
    )
    compiled_pattern = replace(
        compiled_pattern,
        groups=(
            replace(
                compiled_pattern.default_group,
                invocations=(invocation,),
            ),
        ),
    )
    consumer = SimpleNamespace(
        axis_id="A01",
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=2,
            source_step_scope_id="align",
        ),
        source_binding_plan=CompiledSourceBindingPlan.empty(),
        artifact_inputs={},
        compiled_function_pattern=compiled_pattern,
    )
    store = StepOutputManifestStore()
    store.begin_step(producer)
    store.record_outputs(
        producer,
        tuple(
            ProducedOutputSemantics.from_output(
                producer,
                output_dir / f"A01_s001_w{channel}_z001_t001.tif",
                FunctionOutputIdentity(
                    component_values={
                        "well": "A01",
                        "site": 1,
                        "channel": channel,
                    },
                    extension=".tif",
                    source="test",
                ),
                output_context=AlignedImageSliceContext.main_flow(
                    output_key=output_key,
                    artifact_kind=ImageArtifactType.value,
                ),
            )
            for channel, output_key in ((1, "Stain1"), (2, "Stain2"))
        ),
    )

    assert store.filter_to_producer_paths(
        consumer,
        [
            "A01_s001_w1_z001_t001.tif",
            "A01_s001_w2_z001_t001.tif",
        ],
        SourceSchemaFilenameParser(),
    ) == ["A01_s001_w1_z001_t001.tif"]
    contexts = store.producer_output_contexts_for_paths(
        consumer,
        ("A01_s001_w1_z001_t001.tif",),
        SourceSchemaFilenameParser(),
    )
    assert tuple(context.output_key for context in contexts) == ("Stain1",)


def test_step_output_manifest_updates_selected_slot_and_preserves_other_components(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "images"
    producer = SimpleNamespace(
        step_scope_id="producer",
        step_name="Producer",
        pipeline_position=1,
        axis_id="A01",
        output_dir=output_dir,
    )
    update = SimpleNamespace(
        step_scope_id="update",
        step_name="Update",
        pipeline_position=2,
        axis_id="A01",
        output_dir=output_dir,
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=1,
            source_step_scope_id="producer",
        ),
        source_binding_plan=CompiledSourceBindingPlan.empty(),
        artifact_inputs={},
        compiled_function_pattern=compile_function_pattern(lambda image: image, {}, {}),
    )
    store = StepOutputManifestStore()
    store.begin_step(producer)
    store.record_outputs(
        producer,
        tuple(
            ProducedOutputSemantics.from_output(
                producer,
                output_dir / f"A01_s001_w{channel}_z001_t001.tif",
                FunctionOutputIdentity(
                    component_values={
                        "well": "A01",
                        "site": 1,
                        "channel": channel,
                    },
                    extension=".tif",
                    source="test",
                ),
                output_context=AlignedImageSliceContext.main_flow(
                    output_key=output_key,
                    artifact_kind=ImageArtifactType.value,
                ),
            )
            for channel, output_key in ((1, "Image1"), (2, "Image2"))
        ),
    )

    store.begin_step(update, store.producer_records_for(update) or ())
    store.record_outputs(
        update,
        (
            ProducedOutputSemantics.from_output(
                update,
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
                    output_key="Image1",
                    artifact_kind=ImageArtifactType.value,
                ),
            ),
        ),
    )

    records = store.produced_records_for(update)
    assert tuple(record.output_context.output_key for record in records) == (
        "Image1",
        "Image2",
    )
    assert records[0].producer_identity.step_scope_id == "update"
    assert records[1].producer_identity.step_scope_id == "producer"


@pytest.mark.parametrize(
    ("func", "collapsed_input_domain"),
    (
        pytest.param(lambda image: image, True, id="runtime-cardinality"),
        pytest.param(_compose_image_domain, False, id="callable-contract"),
    ),
)
def test_step_output_manifest_collapsed_domain_replaces_inherited_components(
    tmp_path: Path,
    func: Callable,
    collapsed_input_domain: bool,
) -> None:
    output_dir = tmp_path / "images"
    producer = SimpleNamespace(
        step_scope_id="producer",
        step_name="Producer",
        pipeline_position=1,
        axis_id="A01",
        output_dir=output_dir,
    )
    collapse = SimpleNamespace(
        step_scope_id="collapse",
        step_name="Collapse",
        pipeline_position=2,
        axis_id="A01",
        output_dir=output_dir,
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=1,
            source_step_scope_id="producer",
        ),
        source_binding_plan=CompiledSourceBindingPlan.empty(),
        artifact_inputs={},
        compiled_function_pattern=compile_function_pattern(func, {}, {}),
    )
    store = StepOutputManifestStore()
    store.begin_step(producer)
    store.record_outputs(
        producer,
        tuple(
            ProducedOutputSemantics.from_output(
                producer,
                output_dir / f"A01_s001_w{channel}_z001_t001.tif",
                FunctionOutputIdentity(
                    component_values={
                        "well": "A01",
                        "site": 1,
                        "channel": channel,
                    },
                    extension=".tif",
                    source="test",
                ),
            )
            for channel in (1, 2, 3)
        ),
    )

    store.begin_step(collapse, store.producer_records_for(collapse) or ())
    store.record_outputs(
        collapse,
        (
            ProducedOutputSemantics.from_output(
                collapse,
                output_dir / "A01_s001_w1_z001_t001.tif",
                FunctionOutputIdentity(
                    component_values={"well": "A01", "site": 1, "channel": 1},
                    extension=".tif",
                    source="test",
                ),
            ),
        ),
        collapsed_input_domain=collapsed_input_domain,
    )

    records = store.produced_records_for(collapse)
    assert len(records) == 1
    assert records[0].producer_identity.step_scope_id == "collapse"


def test_step_output_manifest_new_output_address_replaces_inherited_components(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "images"
    producer = SimpleNamespace(
        step_scope_id="producer",
        step_name="Producer",
        pipeline_position=1,
        axis_id="A01",
        output_dir=output_dir,
    )
    replacement = SimpleNamespace(
        step_scope_id="replacement",
        step_name="Replacement",
        pipeline_position=2,
        axis_id="A01",
        output_dir=output_dir,
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=1,
            source_step_scope_id="producer",
        ),
        source_binding_plan=CompiledSourceBindingPlan.empty(),
        artifact_inputs={},
        compiled_function_pattern=compile_function_pattern(lambda image: image, {}, {}),
    )
    store = StepOutputManifestStore()
    store.begin_step(producer)
    store.record_outputs(
        producer,
        tuple(
            ProducedOutputSemantics.from_output(
                producer,
                output_dir / f"A01_s001_w{channel}_z001_t001.tif",
                FunctionOutputIdentity(
                    component_values={
                        "well": "A01",
                        "site": 1,
                        "channel": channel,
                    },
                    extension=".tif",
                    source="test",
                ),
                output_context=AlignedImageSliceContext.main_flow(
                    output_key=output_key,
                    artifact_kind=ImageArtifactType.value,
                ),
            )
            for channel, output_key in ((1, "Image1"), (2, "Image2"))
        ),
    )

    store.begin_step(replacement, store.producer_records_for(replacement) or ())
    store.record_outputs(
        replacement,
        (
            ProducedOutputSemantics.from_output(
                replacement,
                output_dir / "A01_s001_w1_z001_t001_Projected.tif",
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
                    output_key="Projected",
                    artifact_kind=ImageArtifactType.value,
                ),
            ),
        ),
    )

    records = store.produced_records_for(replacement)
    assert tuple(record.output_context.output_key for record in records) == (
        "Projected",
    )
    assert records[0].producer_identity.step_scope_id == "replacement"


def test_step_output_manifest_grouped_subset_replaces_inherited_components(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "images"
    producer = SimpleNamespace(
        step_scope_id="producer",
        step_name="Producer",
        pipeline_position=1,
        axis_id="A01",
        output_dir=output_dir,
    )
    subset = SimpleNamespace(
        step_scope_id="subset",
        step_name="Subset",
        pipeline_position=2,
        axis_id="A01",
        output_dir=output_dir,
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=1,
            source_step_scope_id="producer",
        ),
        source_binding_plan=CompiledSourceBindingPlan.empty(),
        artifact_inputs={},
        compiled_function_pattern=compile_function_pattern(
            {"2": lambda image: image},
            {},
            {},
        ),
    )
    store = StepOutputManifestStore()
    store.begin_step(producer)
    store.record_outputs(
        producer,
        tuple(
            ProducedOutputSemantics.from_output(
                producer,
                output_dir / f"A01_s001_w{channel}_z001_t001.tif",
                FunctionOutputIdentity(
                    component_values={
                        "well": "A01",
                        "site": 1,
                        "channel": channel,
                    },
                    extension=".tif",
                    source="test",
                ),
            )
            for channel in (1, 2)
        ),
    )

    store.begin_step(subset, store.producer_records_for(subset) or ())
    store.record_outputs(
        subset,
        (
            ProducedOutputSemantics.from_output(
                subset,
                output_dir / "A01_s001_w2_z001_t001.tif",
                FunctionOutputIdentity(
                    component_values={"well": "A01", "site": 1, "channel": 2},
                    extension=".tif",
                    source="test",
                ),
            ),
        ),
    )

    records = store.produced_records_for(subset)
    assert tuple(record.component_values["channel"] for record in records) == (2,)


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
            plan.ref(): plan
            for plan in (
                ArtifactInputPlan(
                    name="Nuclei",
                    path="Nuclei",
                    artifact_type=ObjectLabelsArtifactType,
                    source_step_id=2,
                ),
            )
        },
        compiled_function_pattern=compile_function_pattern(lambda image: image, {}, {}),
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
            plan.ref(): plan
            for plan in (
                ArtifactInputPlan(
                    name="CorrBlue",
                    path="CorrBlue",
                    artifact_type=ImageArtifactType,
                    source_step_id=4,
                ),
            )
        },
        compiled_function_pattern=compile_function_pattern(lambda image: image, {}, {}),
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
    payload = ImagePayloadMetadata(
        source_path="/source/plate1_A14_site2_Ch3.tif",
        source_component_metadata={
            "well": "A14",
            "site": "2",
            "channel": "3",
        },
    ).payload_with(np.zeros((4, 5), dtype=np.float32), None)

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
    payload = ImagePayloadMetadata(
        source_path="/source/plate1_A14_site1_Ch5.tif",
        source_component_metadata={
            "well": "A14",
            "site": "1",
            "channel": "5",
            "z_index": "1",
            "timepoint": "1",
        },
    ).payload_with(np.zeros((4, 5), dtype=np.float32), None)

    output_path = FunctionOutputPathAuthority.output_path(
        FunctionOutputPathRequest(
            parser=SourceSchemaFilenameParser(),
            output_dir=tmp_path,
            output_payload=payload,
            input_path=None,
        )
    )

    assert output_path.name == "A14_s001_w5_z001_t001.tif"


def test_function_output_identity_completes_partial_payload_metadata_from_fallback_path() -> (
    None
):
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


def test_function_output_identity_uses_fallback_path_extension_for_payload_identity() -> (
    None
):
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
    payload = ImagePayloadMetadata(
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
    ).payload_with(np.zeros((2, 4, 5), dtype=np.float32), None)

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
    payload = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/source/A14_s001_w1_z001_t001.tif",
                "/source/A14_s001_w2_z001_t001.tif",
            ),
        ),
    ).payload_with(np.zeros((2, 4, 5), dtype=np.float32), None)

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
    payload = ImagePayloadMetadata(
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
    ).payload_with(np.zeros((2, 4, 5), dtype=np.float32), None)
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


def test_collapsed_output_identity_uses_retained_source_contributors(
    tmp_path: Path,
) -> None:
    stack_metadata = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/source/A14_s001_w1_z001_t001.tif",
                "/source/A14_s002_w1_z001_t001.tif",
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
                    "site": "2",
                    "channel": "1",
                    "z_index": "1",
                    "timepoint": "1",
                },
            ),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    collapsed_metadata = stack_metadata.collapse_leading_plane_axis()
    payload = collapsed_metadata.payload_with(
        np.zeros((4, 5), dtype=np.float32),
        None,
    )
    request = FunctionOutputPathRequest(
        parser=SourceSchemaFilenameParser(),
        output_dir=tmp_path,
        output_payload=payload,
        input_path=None,
        variable_components=(VariableComponents.SITE,),
    )

    identity = FunctionOutputIdentityAuthority.identity(request)
    filename_identity = FunctionOutputIdentityAuthority.filename_identity_from_metadata(
        request.parser,
        collapsed_metadata,
    )
    output_path = FunctionOutputPathAuthority.output_path_for_identity(
        request,
        identity,
    )

    assert collapsed_metadata.source_image_provenance_planes.count == 0
    assert collapsed_metadata.source_image_provenance_planes.contributor_count == 2
    assert output_path.name == "A14_s001_w1_z001_t001.tif"
    assert identity.component_values == {
        "well": "A14",
        "channel": 1,
        "z_index": 1,
        "timepoint": 1,
    }
    assert identity.filename_component_values is not None
    assert identity.filename_component_values["site"] == 1
    assert filename_identity is not None
    assert (
        FunctionOutputPathAuthority.filename_for_identity(
            request.parser,
            filename_identity,
        )
        == "A14_s001_w1_z001_t001.tif"
    )


def test_composite_then_z_collapse_uses_current_scalar_identity(
    tmp_path: Path,
) -> None:
    composite_payloads = []
    for z_index in (1, 2, 3):
        channel_stack_metadata = ImagePayloadMetadata(
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(
                    paths=tuple(
                        f"/source/A01_s001_w{channel}_z{z_index:03d}_t001.tif"
                        for channel in (1, 2)
                    ),
                    component_metadata=tuple(
                        {
                            "well": "A01",
                            "site": "1",
                            "channel": str(channel),
                            "z_index": str(z_index),
                            "timepoint": "1",
                        }
                        for channel in (1, 2)
                    ),
                )
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        )
        composite_payloads.append(
            channel_stack_metadata.collapse_leading_plane_axis().payload_with(
                np.zeros((4, 5), dtype=np.float32),
                None,
            )
        )

    z_stack_metadata = ImagePayloadMetadata.compose(composite_payloads)
    projected_metadata = z_stack_metadata.collapse_leading_plane_axis()
    payload = projected_metadata.payload_with(
        np.zeros((4, 5), dtype=np.float32),
        None,
    )
    request = FunctionOutputPathRequest(
        parser=SourceSchemaFilenameParser(),
        output_dir=tmp_path,
        output_payload=payload,
        input_path="A01_s001_w1_z001_t001.tif",
        variable_components=(VariableComponents.Z_INDEX,),
    )

    identity = FunctionOutputIdentityAuthority.identity(request)
    output_path = FunctionOutputPathAuthority.output_path_for_identity(
        request,
        identity,
    )

    assert z_stack_metadata.source_provenance.source_plane_count == 3
    assert projected_metadata.source_provenance.source_plane_count == 0
    assert projected_metadata.source_image_provenance_planes.contributor_count == 6
    assert identity.component_values == {
        "well": "A01",
        "site": 1,
        "timepoint": 1,
    }
    assert identity.filename_component_values is not None
    assert identity.filename_component_values["channel"] == 1
    assert identity.filename_component_values["z_index"] == 1
    assert output_path.name == "A01_s001_w1_z001_t001.tif"


def test_variable_component_identity_uses_fallback_path_extension(
    tmp_path: Path,
) -> None:
    payload = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/source/A01_s001_w1_z001_t001.png",
                "/source/A01_s001_w2_z001_t001.jpg",
                "/source/A01_s001_w3_z001_t001.jpg",
            ),
        ),
    ).payload_with(np.zeros((3, 4, 5), dtype=np.float32), None)
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
    payload = ImagePayloadMetadata(
        source_path="/source/A01_s001_w1_z001_t001.jpg",
    ).payload_with(np.zeros((4, 5), dtype=np.float32), None)
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
    payload = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/source/A14_s001_w1_z001_t001.tif",
                "/source/A14_s001_w2_z002_t001.tif",
            ),
        ),
    ).payload_with(np.zeros((2, 4, 5), dtype=np.float32), None)

    assert image_payload_metadata(payload).source_provenance.source_plane_count == 2
    assert (
        image_payload_metadata(payload).source_image_provenance_planes.contributor_count
        == 0
    )

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
    payload = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/source/A01_s001_w1_z001_t001.tif",
                "/source/A01_s001_w3_z001_t001.tif",
                "/source/A01_s001_w2_z001_t001.tif",
            ),
        ),
    ).payload_with(np.zeros((3, 4, 5), dtype=np.float32), None)
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
    payload = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/source/A01_s001_w1_z001_t001.png",
                "/source/A01_s002_w1_z001_t001.png",
            ),
        ),
    ).payload_with(np.zeros((2, 4, 5), dtype=np.float32), None)
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


def test_function_output_path_keeps_payload_split_axis_over_input_alignment(
    tmp_path: Path,
) -> None:
    payload = ImagePayloadMetadata(
        source_path="/source/A01_s001_w1_z001_t001.tif",
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "1",
            "z_index": "1",
            "timepoint": "1",
        },
    ).payload_with(np.zeros((4, 5), dtype=np.float32), None)
    request = FunctionOutputPathRequest(
        parser=SourceSchemaFilenameParser(),
        output_dir=tmp_path,
        output_payload=payload,
        input_path="A01_s003_w1_z001_t001.tif",
        variable_components=(VariableComponents.SITE,),
        input_aligned_output=True,
    )

    identity = FunctionOutputIdentityAuthority.identity(request)
    output_path = FunctionOutputPathAuthority.output_path_for_identity(
        request,
        identity,
    )

    assert output_path.name == "A01_s001_w1_z001_t001.tif"
    assert identity.component_values["site"] == 1
    assert identity.filename_component_values is not None
    assert identity.filename_component_values["site"] == 1


def test_save_outputs_positional_lowering_preserves_explicit_payload_identity(
    tmp_path: Path,
) -> None:
    from openhcs.core.steps.function_runtime import (
        PatternGroupOutputData,
        PatternGroupRuntime,
    )

    class OutputFileManager:
        saved_payloads: list[object] = []
        saved_paths: list[str] = []

        @staticmethod
        def exists(_path: str, _backend: str) -> bool:
            return False

        @staticmethod
        def ensure_directory(_path: str, _backend: str) -> None:
            return None

        def save_batch(
            self,
            payloads: list[object],
            paths: list[str],
            _backend: str,
        ) -> None:
            self.saved_payloads = payloads
            self.saved_paths = paths

    filemanager = OutputFileManager()
    runtime = PatternGroupRuntime(
        SimpleNamespace(
            context=SimpleNamespace(
                filemanager=filemanager,
                microscope_handler=SimpleNamespace(
                    parser=SourceSchemaFilenameParser(),
                ),
                runtime_function_output_identity_cache=FunctionOutputIdentityCache(),
            ),
            execution_plan=SimpleNamespace(
                output_dir=tmp_path,
                variable_components=(VariableComponents.SITE,),
                step_name="ExplicitIdentity",
                pipeline_position=0,
                step_scope_id="explicit-identity",
            ),
            pattern_group_info="A01_s{iii}_w1_z001_t001.tif",
        )
    )
    payload = ImagePayloadMetadata(
        source_path="/source/A01_s001_w1_z001_t001.tif",
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "1",
            "z_index": "1",
            "timepoint": "1",
        },
    ).payload_with(np.zeros((4, 5), dtype=np.float32), None)

    records = runtime._save_outputs(
        PatternGroupOutputData(slices=[payload]),
        ["A01_s003_w1_z001_t001.tif"],
    )

    assert Path(filemanager.saved_paths[0]).name == "A01_s001_w1_z001_t001.tif"
    assert (
        image_payload_metadata(filemanager.saved_payloads[0]).source_component_metadata[
            "site"
        ]
        == "1"
    )
    assert records[0].component_values["site"] == 1
