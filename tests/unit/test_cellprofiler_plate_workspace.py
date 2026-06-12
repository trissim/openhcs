from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import tifffile

from openhcs.core.artifacts import ArtifactKind
from openhcs.interop.cellprofiler.plate_workspace import (
    CellProfilerPlateWorkspacePreparer,
    CellProfilerPlateWorkspaceRequest,
)


def test_prepare_cellprofiler_plate_workspace_materializes_metadata(
    tmp_path: Path,
) -> None:
    fixture = CellProfilerPlateWorkspaceFixture(tmp_path / "ExampleFly")
    fixture.create_source_file("A01_s1_D.TIF")
    fixture.write_names_and_types_cppipe("ExampleFly")

    result = CellProfilerPlateWorkspacePreparer(
        CellProfilerPlateWorkspaceRequest(fixture.plate_root)
    ).prepare()

    assert result.materialized is True
    assert result.cppipe_path == fixture.plate_root / "ExampleFly.cppipe"
    assert (fixture.plate_root / "openhcs_metadata.json").exists()
    generated_pipeline = (
        fixture.plate_root / ".openhcs_cellprofiler" / "ExampleFly_openhcs.py"
    )
    assert generated_pipeline.exists()


def test_prepare_cellprofiler_plate_workspace_prepares_pipeline_when_metadata_exists(
    tmp_path: Path,
) -> None:
    fixture = CellProfilerPlateWorkspaceFixture(tmp_path / "ExampleFly")
    fixture.create_source_file("A01_s1_D.TIF")
    fixture.write_names_and_types_cppipe("ExampleFly")
    (fixture.plate_root / "openhcs_metadata.json").write_text("{}", encoding="utf-8")

    result = CellProfilerPlateWorkspacePreparer(
        CellProfilerPlateWorkspaceRequest(fixture.plate_root)
    ).prepare()

    assert result.materialized is False
    assert result.ingestion is not None
    assert result.ingestion.source_workspace_path is None
    assert result.ingestion.prepared_pipeline.pipeline.steps


def test_prepare_cellprofiler_plate_workspace_ignores_non_cellprofiler_plate(
    tmp_path: Path,
) -> None:
    plate_root = tmp_path / "native"
    plate_root.mkdir()

    result = CellProfilerPlateWorkspacePreparer(
        CellProfilerPlateWorkspaceRequest(plate_root)
    ).prepare()

    assert result.cppipe_path is None
    assert result.ingestion is None


def test_prepare_cellprofiler_plate_workspace_requires_unambiguous_cppipe(
    tmp_path: Path,
) -> None:
    plate_root = tmp_path / "ambiguous"
    plate_root.mkdir()
    (plate_root / "first.cppipe").write_text("Version:5", encoding="utf-8")
    (plate_root / "second.cppipe").write_text("Version:5", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="contains multiple \\.cppipe files",
    ):
        CellProfilerPlateWorkspacePreparer(
            CellProfilerPlateWorkspaceRequest(plate_root)
        ).prepare()


def test_cellprofiler_plate_workspace_orders_tutorial_start_before_final(
    tmp_path: Path,
) -> None:
    plate_root = tmp_path / "AdvancedSegmentation"
    plate_root.mkdir()
    final_cppipe = plate_root / "BBBC022_Analysis_Final.cppipe"
    start_cppipe = plate_root / "BBBC022_Analysis_Start.cppipe"
    final_cppipe.write_text("Version:5", encoding="utf-8")
    start_cppipe.write_text("Version:5", encoding="utf-8")

    paths = CellProfilerPlateWorkspacePreparer(
        CellProfilerPlateWorkspaceRequest(plate_root)
    ).cppipe_paths()

    assert paths == (start_cppipe, final_cppipe)


def test_cellprofiler_plate_workspace_ignores_appledouble_cppipe_sidecars(
    tmp_path: Path,
) -> None:
    plate_root = tmp_path / "segmentation"
    plate_root.mkdir()
    sidecar = plate_root / "._segmentation_start.cppipe"
    start_cppipe = plate_root / "segmentation_start.cppipe"
    sidecar.write_text("not a CellProfiler pipeline", encoding="utf-8")
    start_cppipe.write_text("Version:5", encoding="utf-8")

    paths = CellProfilerPlateWorkspacePreparer(
        CellProfilerPlateWorkspaceRequest(plate_root)
    ).cppipe_paths()

    assert paths == (start_cppipe,)


def test_prepare_cellprofiler_plate_workspace_accepts_explicit_cppipe(
    tmp_path: Path,
) -> None:
    fixture = CellProfilerPlateWorkspaceFixture(tmp_path / "multi")
    fixture.create_source_file("A01_s1_D.TIF")
    fixture.write_names_and_types_cppipe("first")
    selected_cppipe = fixture.write_names_and_types_cppipe("second")

    result = CellProfilerPlateWorkspacePreparer(
        CellProfilerPlateWorkspaceRequest(
            fixture.plate_root,
            cppipe_path=selected_cppipe,
        )
    ).prepare()

    assert result.cppipe_path == selected_cppipe
    assert (
        fixture.plate_root / ".openhcs_cellprofiler" / "second_openhcs.py"
    ).exists()


def test_prepare_cellprofiler_input_workspace_preserves_external_object_inputs(
    tmp_path: Path,
) -> None:
    fixture = CellProfilerPlateWorkspaceFixture(tmp_path / "ExampleFly")
    fixture.create_source_file("A01_s1_D.TIF")
    fixture.write_incomplete_processing_cppipe("ExampleFly")

    result = CellProfilerPlateWorkspacePreparer(
        CellProfilerPlateWorkspaceRequest(fixture.plate_root)
    ).prepare_input_workspace()

    assert result.execution_plate_path == fixture.plate_root
    assert result.materialization is not None
    assert result.source_schema is not None
    assert result.pipeline_import_error is None
    assert result.prepared_pipeline is not None

    bindings = tuple(
        binding
        for step in result.prepared_pipeline.pipeline.steps
        for group in step.source_bindings.groups
        for binding in group.bindings
    )
    assert any(
        binding.alias == "Nuclei"
        and binding.artifact_kind is ArtifactKind.OBJECT_LABELS
        for binding in bindings
    )


@dataclass(frozen=True, slots=True)
class CellProfilerPlateWorkspaceFixture:
    """Test authority for a minimal CellProfiler source folder."""

    plate_root: Path

    def create_source_file(self, filename: str) -> None:
        self.plate_root.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(
            self.plate_root / filename,
            np.full((4, 4), 1, dtype=np.uint16),
        )

    def write_names_and_types_cppipe(self, stem: str) -> Path:
        cppipe_path = self.plate_root / f"{stem}.cppipe"
        cppipe_path.write_text(
            "\n".join(
                (
                    "CellProfiler Pipeline: http://www.cellprofiler.org",
                    "Version:5",
                    "DateRevision:500",
                    "GitHash:",
                    "ModuleCount:3",
                    "HasImagePlaneDetails:False",
                    "",
                    "Images:[module_num:1|enabled:True]",
                    "    Filter images?:Images only",
                    "    Select the rule criteria:and (extension does isimage)",
                    "",
                    "NamesAndTypes:[module_num:2|enabled:True]",
                    "    Assign a name to:Images matching rules",
                    "    Select the image type:Grayscale image",
                    "    Name to assign these images:OrigBlue",
                    "    Image set matching method:Order",
                    "    Assignments count:1",
                    "    Single images count:0",
                    "    Select the rule criteria:and (file does contain \"D.TIF\")",
                    "    Name to assign these images:OrigBlue",
                    "",
                    "IdentifyPrimaryObjects:[module_num:3|enabled:True]",
                    "    Select the input image:OrigBlue",
                    "    Name the primary objects to be identified:Nuclei",
                    "",
                )
            ),
            encoding="utf-8",
        )
        return cppipe_path

    def write_incomplete_processing_cppipe(self, stem: str) -> Path:
        cppipe_path = self.plate_root / f"{stem}.cppipe"
        cppipe_path.write_text(
            "\n".join(
                (
                    "CellProfiler Pipeline: http://www.cellprofiler.org",
                    "Version:5",
                    "DateRevision:500",
                    "GitHash:",
                    "ModuleCount:3",
                    "HasImagePlaneDetails:False",
                    "",
                    "Images:[module_num:1|enabled:True]",
                    "    Filter images?:Images only",
                    "    Select the rule criteria:and (extension does isimage)",
                    "",
                    "NamesAndTypes:[module_num:2|enabled:True]",
                    "    Assign a name to:Images matching rules",
                    "    Select the image type:Grayscale image",
                    "    Name to assign these images:OrigBlue",
                    "    Image set matching method:Order",
                    "    Assignments count:1",
                    "    Single images count:0",
                    "    Select the rule criteria:and (file does contain \"D.TIF\")",
                    "    Name to assign these images:OrigBlue",
                    "",
                    "MaskImage:[module_num:3|enabled:True]",
                    "    Select the input image:OrigBlue",
                    "    Name the output image:MaskedBlue",
                    "    Use objects or an image as a mask?:Objects",
                    "    Select object for mask:Nuclei",
                    "",
                )
            ),
            encoding="utf-8",
        )
        return cppipe_path
