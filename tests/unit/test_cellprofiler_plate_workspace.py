from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

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


@dataclass(frozen=True, slots=True)
class CellProfilerPlateWorkspaceFixture:
    """Test authority for a minimal CellProfiler source folder."""

    plate_root: Path

    def create_source_file(self, filename: str) -> None:
        self.plate_root.mkdir(parents=True, exist_ok=True)
        (self.plate_root / filename).write_bytes(b"")

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
