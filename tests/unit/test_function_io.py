import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from polystore.base import ensure_storage_registry, storage_registry
from polystore.filemanager import FileManager
from scipy.io import savemat

from openhcs.constants.constants import Backend
from openhcs.core.artifacts import ObjectLabelsArtifactType
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.components.parser_metaprogramming import FilenameParseResult
from openhcs.core.source_binding_selection import (
    PipelineStartSourceUniverseRequest,
    SourceFileUniverse,
    SourceUniverseRuntimeState,
)
from openhcs.core.source_binding_workspace import SourceBindingWorkspaceProjector
from openhcs.core.source_bindings import (
    CompiledSourceUniversePlan,
    NamedSourceBinding,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingsConfig,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceProjectionRole,
    SourceSelector,
)
from openhcs.core.source_workspace_projection import VirtualWorkspaceSourceProjection
from openhcs.core.steps.function_io import (
    bulk_preload_step_images,
    get_all_image_paths,
    zarr_batch_layout,
    zarr_output_batch_layout,
)
from openhcs.core.steps.function_output_identity import FunctionOutputIdentity
from openhcs.formats.pattern.pattern_discovery import PatternDiscoveryEngine
from openhcs.microscopes.source_bindings_handler import SourceBindingsHandler
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser


def _filemanager() -> FileManager:
    ensure_storage_registry()
    return FileManager(dict(storage_registry))


def test_get_all_image_paths_contextualizes_relative_backend_listings() -> None:
    class ListedPathFileManager:
        def __init__(self):
            self.resolution_calls = []

        def list_image_files(self, *_args, **_kwargs):
            return (
                "nested/A01_s001_w1_z001_t001.tif",
                "/already/absolute/A01_s002_w1_z001_t001.tif",
                "nested/B01_s001_w1_z001_t001.tif",
            )

        def resolve_listed_address(
            self,
            listed_address,
            backend,
            *,
            directory,
        ):
            self.resolution_calls.append((listed_address, backend, directory))
            if listed_address.startswith("/"):
                return listed_address
            return f"{directory}/{listed_address}"

    class WellParser:
        @staticmethod
        def parse_filename(filename: str):
            return FilenameParseResult({"well": filename[:3]})

    filemanager = ListedPathFileManager()
    paths = get_all_image_paths(
        "/virtual/plate_1",
        "relative_backend",
        "A01",
        filemanager,
        SimpleNamespace(parser=WellParser()),
    )

    assert paths == [
        "/already/absolute/A01_s002_w1_z001_t001.tif",
        "/virtual/plate_1/nested/A01_s001_w1_z001_t001.tif",
    ]
    assert filemanager.resolution_calls == [
        (
            "nested/A01_s001_w1_z001_t001.tif",
            "relative_backend",
            "/virtual/plate_1",
        ),
        (
            "/already/absolute/A01_s002_w1_z001_t001.tif",
            "relative_backend",
            "/virtual/plate_1",
        ),
    ]


def test_runtime_axis_matching_preserves_numeric_filename_components() -> None:
    class ListedPathFileManager:
        @staticmethod
        def list_image_files(*_args, **_kwargs):
            return (
                "/virtual/1_s001_w1_z001_t001.tif",
                "/virtual/2_s001_w1_z001_t001.tif",
            )

        @staticmethod
        def resolve_listed_address(listed_address, _backend, *, directory):
            del directory
            return listed_address

    parser = SourceSchemaFilenameParser()
    filemanager = ListedPathFileManager()

    assert get_all_image_paths(
        "/virtual",
        "virtual_workspace",
        "1",
        filemanager,
        SimpleNamespace(parser=parser),
    ) == ["/virtual/1_s001_w1_z001_t001.tif"]

    patterns = PatternDiscoveryEngine(
        parser, filemanager
    ).auto_detect_patterns_from_files(
        [
            "/virtual/1_s001_w1_z001_t001.tif",
            "/virtual/2_s001_w1_z001_t001.tif",
        ],
        variable_components=["channel"],
        well_filter=["1"],
    )
    assert tuple(patterns) == ("1",)


def test_zarr_batch_layout_projects_every_declared_image_axis() -> None:
    handler = SimpleNamespace(parser=SourceSchemaFilenameParser())
    paths = (
        "/images/A01_s003_w2_z001_t002.tif",
        "/images/A01_s003_w1_z001_t001.tif",
        "/images/A01_s003_w2_z001_t001.tif",
        "/images/A01_s003_w1_z001_t002.tif",
    )

    layout = zarr_batch_layout(paths, handler)

    assert tuple(axis.name for axis in layout.axes) == ("t", "field", "c", "z")
    assert layout.axes[0].values == ("2", "1")
    assert layout.axes[1].values == ("3",)
    assert layout.axes[2].values == ("2", "1")
    assert layout.axes[3].values == ("1",)
    assert layout.item_coordinates == (
        (0, 0, 0, 0),
        (1, 0, 1, 0),
        (1, 0, 0, 0),
        (0, 0, 1, 0),
    )


def test_zarr_output_layout_uses_declared_surface_to_disambiguate_channels() -> None:
    components = {
        "timepoint": 1,
        "site": 1,
        "channel": 1,
        "z_index": 1,
    }
    identities = (
        FunctionOutputIdentity(
            component_values=components,
            extension=".pkl",
            source="test",
        ).with_filename_qualifier("IllumDAPI"),
        FunctionOutputIdentity(
            component_values=components,
            extension=".pkl",
            source="test",
        ).with_filename_qualifier("IllumDAPIAvg"),
    )

    layout = zarr_output_batch_layout(identities)

    assert layout.axes[2].values == ("1:IllumDAPI", "1:IllumDAPIAvg")
    assert layout.item_coordinates == ((0, 0, 0, 0), (0, 0, 1, 0))


def test_zarr_output_layout_omits_components_collapsed_by_output_declaration() -> None:
    identities = tuple(
        FunctionOutputIdentity(
            component_values={
                "timepoint": 1,
                "site": 1,
                "channel": channel,
            },
            extension=".tif",
            source="projection",
        )
        for channel in (1, 2)
    )

    layout = zarr_output_batch_layout(identities)

    assert tuple(axis.name for axis in layout.axes) == ("t", "field", "c")
    assert layout.axes[2].values == ("1", "2")
    assert layout.item_coordinates == ((0, 0, 0), (0, 0, 1))


def test_zarr_output_layout_rejects_partial_component_presence() -> None:
    identities = (
        FunctionOutputIdentity(
            component_values={
                "timepoint": 1,
                "site": 1,
                "channel": 1,
                "z_index": 1,
            },
            extension=".tif",
            source="projection",
        ),
        FunctionOutputIdentity(
            component_values={
                "timepoint": 1,
                "site": 1,
                "channel": 2,
            },
            extension=".tif",
            source="projection",
        ),
    )

    with pytest.raises(ValueError, match="disagree on component 'z_index'"):
        zarr_output_batch_layout(identities)


def test_bulk_preload_preserves_nested_virtual_workspace_paths(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    workspace_root = tmp_path / "workspace"
    source_root.mkdir()
    sources = {
        "PrimaryImage": np.full((4, 4), 1, dtype=np.uint16),
        "ObjectLabels": np.full((4, 4), 2, dtype=np.uint16),
    }
    source_paths = []
    for alias, labels in sources.items():
        path = source_root / f"{alias}.npy"
        np.save(path, labels)
        source_paths.append(path)

    source_bindings = SourceBindingsConfig(
        bindings=(
            NamedSourceBinding(
                alias="PrimaryImage",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.EQUALS,
                            source_paths[0].name,
                        ),
                    ),
                ),
            ),
            NamedSourceBinding(
                alias="ObjectLabels",
                artifact_kind=ObjectLabelsArtifactType,
                projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.EQUALS,
                            source_paths[1].name,
                        ),
                    ),
                ),
            ),
        )
    )
    filemanager = _filemanager()
    SourceBindingWorkspaceProjector(
        source_bindings,
        parser=SourceSchemaFilenameParser(),
    ).materialize(
        source_root,
        workspace_root,
        filemanager=filemanager,
        source_backend=Backend.DISK,
        workspace_backend=Backend.DISK,
        source_files=source_paths,
    )
    handler = SourceBindingsHandler(filemanager, source_bindings)
    handler.initialize_workspace(workspace_root, filemanager)

    virtual_paths = get_all_image_paths(
        workspace_root,
        Backend.VIRTUAL_WORKSPACE.value,
        "A01",
        filemanager,
        handler,
    )
    expected_paths = [
        str(workspace_root / "A01_s001_w1_z001_t001.tif"),
        str(workspace_root / "_source" / "ObjectLabels" / "A01_s001_w1_z001_t001.tif"),
    ]
    assert virtual_paths == expected_paths

    bulk_preload_step_images(
        workspace_root,
        "A01",
        Backend.VIRTUAL_WORKSPACE.value,
        filemanager,
        handler,
    )

    loaded = filemanager.load_batch(expected_paths, Backend.MEMORY.value)
    for payload, labels in zip(loaded, sources.values(), strict=True):
        np.testing.assert_array_equal(payload.data, labels)


def test_object_only_source_anchors_match_compiled_pattern_after_preload(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    workspace_root = tmp_path / "workspace"
    source_root.mkdir()
    source_paths = tuple(
        source_root / filename for filename in ("First.npy", "Second.npy")
    )
    for object_id, source_path in enumerate(source_paths, start=1):
        np.save(source_path, np.full((4, 4), object_id, dtype=np.uint16))

    source_bindings = SourceBindingsConfig(
        bindings=tuple(
            NamedSourceBinding(
                alias=alias,
                artifact_kind=ObjectLabelsArtifactType,
                projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.EQUALS,
                            source_path.name,
                        ),
                    ),
                ),
            )
            for alias, source_path in zip(
                ("FirstObjects", "SecondObjects"),
                source_paths,
                strict=True,
            )
        ),
        match_plan=SourceBindingMatchPlan(SourceBindingMatchMethod.ORDER),
    )
    filemanager = _filemanager()
    materialization = SourceBindingWorkspaceProjector(
        source_bindings,
        parser=SourceSchemaFilenameParser(),
    ).materialize(
        source_root,
        workspace_root,
        filemanager=filemanager,
        source_backend=Backend.DISK,
        workspace_backend=Backend.DISK,
        source_files=source_paths,
    )
    handler = SourceBindingsHandler(filemanager, source_bindings)
    handler.initialize_workspace(workspace_root, filemanager)

    assert tuple(materialization.artifact_mappings) == (
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w2_z001_t001.tif",
    )
    bulk_preload_step_images(
        workspace_root,
        "A01",
        Backend.VIRTUAL_WORKSPACE.value,
        filemanager,
        handler,
    )

    assert handler.path_list_from_pattern(
        workspace_root,
        "A01_s{iii}_w1_z001_t001.tif",
        filemanager,
        Backend.MEMORY.value,
        ["site"],
    ) == ["A01_s001_w1_z001_t001.tif"]


def test_virtual_pipeline_source_universe_does_not_mix_physical_paths(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    workspace_root = tmp_path / "workspace"
    source_root.mkdir()
    source_paths = tuple(
        source_root / filename for filename in ("First.npy", "Second.npy")
    )
    for source_path in source_paths:
        np.save(source_path, np.ones((4, 4), dtype=np.uint16))

    source_bindings = SourceBindingsConfig(
        bindings=tuple(
            NamedSourceBinding(
                alias=alias,
                artifact_kind=ObjectLabelsArtifactType,
                projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.EQUALS,
                            source_path.name,
                        ),
                    )
                ),
            )
            for alias, source_path in zip(
                ("FirstObjects", "SecondObjects"),
                source_paths,
                strict=True,
            )
        ),
        match_plan=SourceBindingMatchPlan(SourceBindingMatchMethod.ORDER),
    )
    filemanager = _filemanager()
    materialization = SourceBindingWorkspaceProjector(
        source_bindings,
        parser=SourceSchemaFilenameParser(),
    ).materialize(
        source_root,
        workspace_root,
        filemanager=filemanager,
        source_backend=Backend.DISK,
        workspace_backend=Backend.DISK,
        source_files=source_paths,
    )
    projection = VirtualWorkspaceSourceProjection.from_openhcs_metadata(
        workspace_root,
        json.loads(materialization.metadata_path.read_text()),
    )
    universe = SourceFileUniverse(
        projection.pipeline_start_files(),
        Backend.VIRTUAL_WORKSPACE,
    )
    request = PipelineStartSourceUniverseRequest(
        context=SimpleNamespace(input_dir=source_root, filemanager=filemanager),
        plan=CompiledStepPlan(
            step_index=0,
            step_name="ObjectSource",
            step_type="FunctionStep",
            axis_id="A01",
            source_universe_plan=CompiledSourceUniversePlan(
                requires_full_pipeline_source_universe=False,
                uses_pipeline_start_binding_origin=True,
            ),
        ),
        matching_files=universe.files,
        source_backend=Backend.VIRTUAL_WORKSPACE,
        source_projection=projection,
    )

    state = request.contribute_runtime_state(
        SourceUniverseRuntimeState(),
        universe,
    )

    assert state.pipeline_source_candidate_files == universe.files
    assert not set(state.pipeline_source_candidate_files).intersection(
        str(source_path) for source_path in source_paths
    )


def test_bulk_preload_loads_matlab_pixels_through_declared_vfs_backend(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    workspace_root = tmp_path / "workspace"
    source_root.mkdir()
    raw = np.full((4, 4), 8, dtype=np.uint16)
    illumination = np.full((4, 4), 2.0, dtype=np.float32)
    raw_path = source_root / "raw.npy"
    illumination_path = source_root / "illumination.mat"
    np.save(raw_path, raw)
    savemat(illumination_path, {"Image": illumination})

    source_bindings = SourceBindingsConfig(
        bindings=tuple(
            NamedSourceBinding(
                alias=alias,
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.EQUALS,
                            path.name,
                        ),
                    ),
                ),
            )
            for alias, path in (
                ("Raw", raw_path),
                ("Illum", illumination_path),
            )
        )
    )
    filemanager = _filemanager()
    materialization = SourceBindingWorkspaceProjector(
        source_bindings,
        parser=SourceSchemaFilenameParser(),
    ).materialize(
        source_root,
        workspace_root,
        filemanager=filemanager,
        source_backend=Backend.DISK,
        workspace_backend=Backend.DISK,
        source_files=(raw_path, illumination_path),
    )
    SourceBindingsHandler(filemanager, source_bindings).initialize_workspace(
        workspace_root,
        filemanager,
    )

    assert all(
        set(mapping) == {"backend", "backend_address", "source_axis_indices"}
        for mapping in materialization.plane_mappings.values()
    )

    virtual_paths = tuple(
        str(workspace_root / virtual_path)
        for virtual_path in materialization.plane_mappings
    )
    bulk_preload_step_images(
        workspace_root,
        "A01",
        Backend.VIRTUAL_WORKSPACE.value,
        filemanager,
        SourceBindingsHandler(filemanager, source_bindings),
    )

    loaded = filemanager.load_batch(
        list(virtual_paths),
        Backend.MEMORY.value,
    )
    np.testing.assert_array_equal(loaded[0].data, raw)
    np.testing.assert_array_equal(loaded[1].data, illumination)
    assert loaded[1].metadata.source_path == virtual_paths[1]
