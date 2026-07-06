from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from benchmark.adapters.cellprofiler import (
    CELLPROFILER_FIRST_IMAGE_SET_PARAM,
    CELLPROFILER_LAST_IMAGE_SET_PARAM,
    DETERMINISTIC_PYTHONHASHSEED,
    HeadlessCellProfilerPipelinePolicy,
    HeadlessCellProfilerPipelinePatch,
    NativeCellProfilerImportedMetadataPlacementPlan,
    NativeCellProfilerImportedMetadataPipelinePatch,
    NativeCellProfilerInputDomainStrategyKey,
    NativeCellProfilerProvenanceField,
    PYTHONHASHSEED_ENV,
    CellProfilerAdapter,
    native_cellprofiler_reference_is_complete,
)
from benchmark.adapters.cellprofiler_installation import (
    CELLPROFILER_EXECUTABLE_ENV,
    CellProfilerExecutableResolver,
    CellProfilerExecutableSource,
    OPENHCS_BENCHMARK_TOOL_ROOTS_ENV,
)
from benchmark.contracts.tool_adapter import ToolNotInstalledError
from openhcs.core.pipeline_image_schema import ImportedMetadataTable
from openhcs.core.source_bindings import SourceBindingRuntimeContext
from openhcs.core.source_schema_workspace import SourceSchemaImageSetSelection
from openhcs.interop.cellprofiler.runtime.source_candidates import (
    SourceCandidatePathProjection,
)


def test_cellprofiler_adapter_requires_executable(monkeypatch) -> None:
    monkeypatch.setattr(
        "benchmark.adapters.cellprofiler_installation.shutil.which",
        lambda _name: None,
    )
    monkeypatch.setattr(
        "benchmark.adapters.cellprofiler_installation.sys.executable",
        "/missing/python",
    )
    monkeypatch.setattr(
        CellProfilerExecutableResolver,
        "_current_environment_candidates",
        lambda _self: (),
    )
    monkeypatch.setattr(
        CellProfilerExecutableResolver,
        "_declared_tool_root_candidates",
        lambda _self: (),
    )
    monkeypatch.setattr(
        CellProfilerExecutableResolver,
        "_local_workspace_tool_root_candidates",
        lambda _self: (),
    )

    with pytest.raises(ToolNotInstalledError, match="CellProfiler executable"):
        CellProfilerAdapter().validate_installation()


def test_native_cellprofiler_imported_metadata_places_files_by_path_columns(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    image_dir = source_root / "20585"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "IXMtest_A01_s1_w1.tif"
    image_path.write_bytes(b"image")
    csv_path = source_root / "20585_AE.csv"
    csv_path.write_text(
        "Image_FileName_OrigHoechst,Image_PathName_OrigHoechst\n"
        "IXMtest_A01_s1_w1.tif,20585/\n",
        encoding="utf-8",
    )

    placements = NativeCellProfilerImportedMetadataPlacementPlan(
        source_root,
        (ImportedMetadataTable(location="20585_AE.csv"),),
        (image_path, csv_path),
    ).placements()

    assert {
        placement.source_path.name: placement.relative_path.as_posix()
        for placement in placements
    } == {
        "IXMtest_A01_s1_w1.tif": "20585/IXMtest_A01_s1_w1.tif",
        "20585_AE.csv": "20585_AE.csv",
    }


def test_native_cellprofiler_imported_metadata_uses_shared_stale_path_resolution(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "AdvancedSegmentation" / "BBBC022_20585_AE"
    source_root.mkdir(parents=True)
    metadata_path = source_root.parent / "20585_AE.csv"
    metadata_path.write_text("Metadata_Plate\n20585\n", encoding="utf-8")

    resolved = NativeCellProfilerImportedMetadataPlacementPlan(
        source_root,
        (ImportedMetadataTable(location="/Users/pryder/Documents/tutorials/AdvancedSegmentation/20585_AE.csv"),),
        (metadata_path,),
    ).imported_metadata_path(
        ImportedMetadataTable(
            location="/Users/pryder/Documents/tutorials/AdvancedSegmentation/20585_AE.csv"
        )
    )

    assert resolved == metadata_path


def test_native_cellprofiler_imported_metadata_pipeline_patch_targets_staged_input(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "source" / "20585_AE.csv"
    metadata_path.parent.mkdir()
    metadata_path.write_text("Metadata_Plate\n20585\n", encoding="utf-8")
    source_text = "\n".join(
        (
            "CellProfiler Pipeline: http://www.cellprofiler.org",
            "Metadata:[module_num:2|enabled:True]",
            "    Metadata extraction method:Extract from file/folder names",
            "    Metadata file location:Elsewhere...|",
            "    Metadata file name:",
            "    Metadata extraction method:Import from file",
            "    Metadata file location:Default Input Folder|/Users/pryder/Documents/tutorials/AdvancedSegmentation",
            "    Metadata file name:20585_AE.csv",
            "",
        )
    )

    patched = NativeCellProfilerImportedMetadataPipelinePatch(
        (metadata_path,)
    ).patch_text(source_text)

    assert (
        "    Metadata file location:Default Input Folder|\n"
        "    Metadata file name:20585_AE.csv"
    ) in patched
    assert "/Users/pryder" not in patched


def test_source_candidate_paths_preserve_virtual_well_identity_for_shared_source(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "Channel 1-01-A-01-00.tif"
    source_path.write_bytes(b"image")
    context = SourceBindingRuntimeContext(
        step_input_dir=str(tmp_path),
        step_input_source_paths={
            "W001_s001_w1_z001_t001.tif": str(source_path),
            "W002_s001_w1_z001_t001.tif": str(source_path),
        },
    )
    adapter = SimpleNamespace(source_binding_context=context)

    assert tuple(
        (path.path, path.resolved_path)
        for path in SourceCandidatePathProjection(str(source_path), adapter).paths()
    ) == (
        ("W001_s001_w1_z001_t001.tif", str(source_path)),
        ("W002_s001_w1_z001_t001.tif", str(source_path)),
    )


def test_cellprofiler_adapter_accepts_executable_env(monkeypatch) -> None:
    commands: list[tuple[str, ...]] = []
    monkeypatch.setenv(CELLPROFILER_EXECUTABLE_ENV, "/opt/cellprofiler/bin/cellprofiler")

    def _run(
        command,
        *,
        capture_output: bool,
        cwd: Path | None = None,
        env=None,
        text: bool,
        timeout: float | None,
        check: bool,
    ):
        commands.append(tuple(command))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="CellProfiler 4.2.8.1\n",
            stderr="",
        )

    monkeypatch.setattr("benchmark.adapters.cellprofiler.subprocess.run", _run)

    adapter = CellProfilerAdapter()
    adapter.validate_installation()

    assert adapter.version == "CellProfiler 4.2.8.1"
    assert commands == [("/opt/cellprofiler/bin/cellprofiler", "--version")]


def test_cellprofiler_resolver_discovers_local_workspace_tool_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "benchmark.adapters.cellprofiler_installation.shutil.which",
        lambda _name: None,
    )
    repo_root = tmp_path / "openhcs-benchmark-platform"
    executable = (
        tmp_path
        / "openhcs"
        / ".venv-cellprofiler39"
        / "bin"
        / "cellprofiler"
    )
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")

    resolver = CellProfilerExecutableResolver(
        environment={},
        repo_root=repo_root,
        python_executable=tmp_path / "benchmark-venv" / "bin" / "python",
    )

    assert resolver.resolve() == executable
    assert (
        CellProfilerExecutableSource.LOCAL_WORKSPACE_TOOL_ROOT
        in {candidate.source for candidate in resolver.candidates()}
    )


def test_cellprofiler_resolver_discovers_declared_tool_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "benchmark.adapters.cellprofiler_installation.shutil.which",
        lambda _name: None,
    )
    tool_root = tmp_path / "cellprofiler-tools"
    executable = tool_root / ".venv-cellprofiler39" / "bin" / "cellprofiler"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")

    resolver = CellProfilerExecutableResolver(
        environment={OPENHCS_BENCHMARK_TOOL_ROOTS_ENV: str(tool_root)},
        repo_root=tmp_path / "openhcs-benchmark-platform",
        python_executable=tmp_path / "benchmark-venv" / "bin" / "python",
    )

    assert resolver.resolve() == executable
    assert (
        CellProfilerExecutableSource.DECLARED_TOOL_ROOT
        in {candidate.source for candidate in resolver.candidates()}
    )


def test_cellprofiler_adapter_runs_cppipe_headless(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = tmp_path / "plate"
    dataset_path.mkdir()
    cppipe_path = tmp_path / "pipeline.cppipe"
    cppipe_path.write_text("CellProfiler Pipeline: http://www.cellprofiler.org\n")
    commands: list[tuple[str, ...]] = []

    def _run(
        command,
        *,
        capture_output: bool,
        cwd: Path | None = None,
        env=None,
        text: bool,
        timeout: float | None,
        check: bool,
    ):
        assert capture_output is True
        assert text is True
        assert check is False
        command = tuple(command)
        commands.append(command)
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="CellProfiler 4.2.6\n",
                stderr="",
            )
        assert env[PYTHONHASHSEED_ENV] == DETERMINISTIC_PYTHONHASHSEED
        assert cwd is not None
        if "--file-list" in command:
            execution_cppipe_path = Path(command[command.index("-p") + 1])
            assert "Filter images?:No filtering" in execution_cppipe_path.read_text(
                encoding="utf-8"
            )
        output_root = Path(command[command.index("-o") + 1])
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "Image.csv").write_text("ImageNumber,Count\n1,2\n")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("benchmark.adapters.cellprofiler.subprocess.run", _run)

    adapter = CellProfilerAdapter(
        executable="/usr/bin/cellprofiler",
        source_schema_image_set_selection=SourceSchemaImageSetSelection(
            max_image_set_count=1,
        ),
    )
    adapter.validate_installation()
    result = adapter.run(
        dataset_path=dataset_path,
        pipeline_name="native_reference",
        pipeline_params={
            "dataset_id": "synthetic",
            "cppipe_path": str(cppipe_path),
            "cellprofiler_timeout_seconds": 12,
        },
        metrics=[],
        output_dir=tmp_path / "outputs",
    )

    assert result.success is True
    assert result.provenance["cellprofiler_version"] == "CellProfiler 4.2.6"
    assert result.provenance["pipeline_source"] == "native_cppipe"
    assert result.provenance["csv_output_count"] == 1
    assert result.provenance["pythonhashseed"] == DETERMINISTIC_PYTHONHASHSEED
    assert {
        record["phase"] for record in result.provenance["phase_timing_records"]
    } == {"RESOLVE_SOURCE", "EXECUTE_NATIVE_CP", "SNAPSHOT_OUTPUTS"}
    assert commands[1] == (
        "/usr/bin/cellprofiler",
        "-c",
        "-r",
        "-p",
        str(cppipe_path),
        "-i",
        str(dataset_path),
        "-o",
        str(result.output_path),
    )


def test_cellprofiler_adapter_runs_bounded_image_set_range(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = tmp_path / "plate"
    dataset_path.mkdir()
    cppipe_path = tmp_path / "pipeline.cppipe"
    cppipe_path.write_text("CellProfiler Pipeline: http://www.cellprofiler.org\n")
    commands: list[tuple[str, ...]] = []

    def _run(
        command,
        *,
        capture_output: bool,
        cwd: Path | None = None,
        env=None,
        text: bool,
        timeout: float | None,
        check: bool,
    ):
        command = tuple(command)
        commands.append(command)
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, stdout="CellProfiler\n")
        output_root = Path(command[command.index("-o") + 1])
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "Image.csv").write_text("ImageNumber,Count\n1,2\n")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("benchmark.adapters.cellprofiler.subprocess.run", _run)

    adapter = CellProfilerAdapter(
        executable="/usr/bin/cellprofiler",
        source_schema_image_set_selection=SourceSchemaImageSetSelection(
            max_image_set_count=1,
        ),
    )
    adapter.validate_installation()
    result = adapter.run(
        dataset_path=dataset_path,
        pipeline_name="native_reference",
        pipeline_params={
            "dataset_id": "synthetic",
            "cppipe_path": str(cppipe_path),
            CELLPROFILER_FIRST_IMAGE_SET_PARAM: 1,
            CELLPROFILER_LAST_IMAGE_SET_PARAM: 1,
        },
        metrics=[],
        output_dir=tmp_path / "outputs",
    )

    assert commands[1][-4:] == ("--first-image-set", "1", "--last-image-set", "1")
    assert result.provenance[CELLPROFILER_FIRST_IMAGE_SET_PARAM] == 1
    assert result.provenance[CELLPROFILER_LAST_IMAGE_SET_PARAM] == 1


def test_cellprofiler_adapter_file_list_limits_selected_image_sets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = tmp_path / "plate"
    dataset_path.mkdir()
    for well in ("A01", "A02"):
        for site in ("1", "2"):
            for channel, suffix in (("DAPI", "dapi"), ("GFP", "gfp")):
                image = np.full((8, 8), int(site), dtype=np.uint16)
                Image.fromarray(image).save(
                    dataset_path / f"Plate_{well}_s{site}_{suffix}.tif"
                )
    cppipe_path = tmp_path / "pipeline.cppipe"
    cppipe_path.write_text(
        "\n".join(
            [
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "HasImagePlaneDetails:False",
                "",
                "Images:[module_num:1|enabled:True]",
                "    Filter images?:Images only",
                "    Select the rule criteria:and (extension does isimage)",
                "",
                "Metadata:[module_num:2|enabled:True]",
                "    Extract metadata?:Yes",
                "    Metadata extraction method:Extract from file/folder names",
                "    Metadata source:File name",
                "    Regular expression to extract from file name:^Plate_(?P<Well>[A-Z][0-9]{2})_s(?P<Site>[0-9]+)_(?P<ChannelName>.*)",
                "",
                "NamesAndTypes:[module_num:3|enabled:True]",
                "    Assign a name to:Images matching rules",
                "    Select the image type:Grayscale image",
                "    Name to assign these images:DNA",
                "    Match metadata:[{\"DNA\":\"Well\",\"Protein\":\"Well\"},{\"DNA\":\"Site\",\"Protein\":\"Site\"}]",
                "    Image set matching method:Metadata",
                "    Set intensity range from:Image metadata",
                "    Assignments count:2",
                "    Select the rule criteria:and (metadata does Well) (file does contain \"dapi\")",
                "    Name to assign these images:DNA",
                "    Name to assign these objects:Cell",
                "    Select the image type:Grayscale image",
                "    Set intensity range from:Image metadata",
                "    Select the rule criteria:and (metadata does Well) (file does contain \"gfp\")",
                "    Name to assign these images:Protein",
                "    Name to assign these objects:Cell",
                "",
                "Groups:[module_num:4|enabled:True]",
                "    Do you want to group your images?:Yes",
                "    grouping metadata count:1",
                "    Metadata category:Well",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    commands: list[tuple[str, ...]] = []

    def _run(
        command,
        *,
        capture_output: bool,
        cwd: Path | None = None,
        env=None,
        text: bool,
        timeout: float | None,
        check: bool,
    ):
        command = tuple(command)
        commands.append(command)
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, stdout="CellProfiler\n")
        output_root = Path(command[command.index("-o") + 1])
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "Image.csv").write_text("ImageNumber,Count\n1,2\n")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("benchmark.adapters.cellprofiler.subprocess.run", _run)

    adapter = CellProfilerAdapter(
        executable="/usr/bin/cellprofiler",
        source_schema_image_set_selection=SourceSchemaImageSetSelection(
            max_image_set_count=1
        ),
    )
    adapter.validate_installation()
    result = adapter.run(
        dataset_path=dataset_path,
        pipeline_name="native_reference",
        pipeline_params={
            "dataset_id": "synthetic",
            "cppipe_path": str(cppipe_path),
        },
        metrics=[],
        output_dir=tmp_path / "outputs",
    )

    command = commands[1]
    file_list_path = Path(command[command.index("--file-list") + 1])
    file_list = file_list_path.read_text(encoding="utf-8").splitlines()
    assert len(file_list) == 2
    assert all("Plate_A01_" in entry for entry in file_list)
    assert any("_s1_dapi" in entry for entry in file_list)
    assert any("_s1_gfp" in entry for entry in file_list)
    assert not any("_s2_" in entry for entry in file_list)
    assert result.provenance[NativeCellProfilerProvenanceField.SELECTED_WELLS] == (
        "A01",
    )
    assert (
        result.provenance[
            NativeCellProfilerProvenanceField.SELECTED_SOURCE_FILE_COUNT
        ]
        == 2
    )


def test_cellprofiler_adapter_isolates_embedded_image_plane_input_domain(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = tmp_path / "plate"
    dataset_path.mkdir()
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(dataset_path / "url_D.TIF")
    Image.fromarray(np.ones((2, 2), dtype=np.uint8)).save(dataset_path / "url_F.TIF")
    cppipe_path = tmp_path / "url_planes.cppipe"
    cppipe_path.write_text(
        "\n".join(
            [
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "HasImagePlaneDetails:True",
                "",
                "Images:[module_num:1|enabled:True]",
                "    Filter images?:Images only",
                "    Select the rule criteria:and (extension does isimage)",
                "",
                "NamesAndTypes:[module_num:2|enabled:True]",
                "    Assign a name to:Images matching rules",
                "    Select the image type:Grayscale image",
                "    Name to assign these images:OrigBlue",
                "    Match metadata:[]",
                "    Image set matching method:Order",
                "    Set intensity range from:Image metadata",
                "    Assignments count:2",
                "    Select the rule criteria:and (file does contain \"D.TIF\")",
                "    Name to assign these images:OrigBlue",
                "    Name to assign these objects:Cell",
                "    Select the image type:Grayscale image",
                "    Set intensity range from:Image metadata",
                "    Maximum intensity:255.0",
                "    Select the rule criteria:and (file does contain \"F.TIF\")",
                "    Name to assign these images:OrigGreen",
                "    Name to assign these objects:Cell",
                "    Select the image type:Grayscale image",
                "    Set intensity range from:Image metadata",
                "    Maximum intensity:255.0",
                "",
                '"Version":"1","PlaneCount":"2"',
                '"URL","Series","Index","Channel"',
                '"https://example.invalid/data/url_D.TIF",,,',
                '"https://example.invalid/data/url_F.TIF",,,',
            ]
        ),
        encoding="utf-8",
    )
    commands: list[tuple[str, ...]] = []

    def _run(
        command,
        *,
        capture_output: bool,
        cwd: Path | None = None,
        env=None,
        text: bool,
        timeout: float | None,
        check: bool,
    ):
        command = tuple(command)
        commands.append(command)
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="CellProfiler 4.2.8.1\n",
                stderr="",
            )
        output_root = Path(command[command.index("-o") + 1])
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "Image.csv").write_text("ImageNumber,Count\n1,2\n")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("benchmark.adapters.cellprofiler.subprocess.run", _run)

    adapter = CellProfilerAdapter(executable="/usr/bin/cellprofiler")
    adapter.validate_installation()
    result = adapter.run(
        dataset_path=dataset_path,
        pipeline_name="native_reference",
        pipeline_params={
            "dataset_id": "synthetic",
            "cppipe_path": str(cppipe_path),
        },
        metrics=[],
        output_dir=tmp_path / "outputs",
    )

    native_command = commands[1]
    execution_cppipe = Path(native_command[native_command.index("-p") + 1])
    input_dir = Path(native_command[native_command.index("-i") + 1])
    patched_text = execution_cppipe.read_text(encoding="utf-8")

    assert input_dir.name == "native_cellprofiler_empty_input"
    assert not tuple(input_dir.iterdir())
    assert "https://example.invalid" not in patched_text
    assert '"file://' in patched_text
    assert "url_D.TIF" in patched_text
    assert "url_F.TIF" in patched_text
    assert (
        result.provenance[NativeCellProfilerProvenanceField.INPUT_DOMAIN_STRATEGY]
        is NativeCellProfilerInputDomainStrategyKey.EMBEDDED_IMAGE_PLANES
    )
    assert result.provenance[NativeCellProfilerProvenanceField.SOURCE_PLANE_COUNT] == 2


def test_native_reference_completeness_rejects_stale_embedded_plane_domain(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "url_planes.cppipe"
    cppipe_path.write_text(
        "\n".join(
            [
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "HasImagePlaneDetails:True",
                "",
                "Images:[module_num:1|enabled:True]",
                "    Filter images?:Images only",
                "",
                '"Version":"1","PlaneCount":"1"',
                '"URL","Series","Index","Channel"',
                '"https://example.invalid/data/url_D.TIF",,,',
            ]
        ),
        encoding="utf-8",
    )
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    (reference_dir / ".cellprofiler_benchmark_reference.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provenance": {
                    "cppipe_path": str(cppipe_path),
                    NativeCellProfilerProvenanceField.INPUT_DOMAIN_STRATEGY: (
                        NativeCellProfilerInputDomainStrategyKey.DATASET_FOLDER
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    assert native_cellprofiler_reference_is_complete(reference_dir) is False


def test_native_reference_completeness_accepts_selected_source_schema_domain(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "url_planes.cppipe"
    cppipe_path.write_text(
        "\n".join(
            [
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "HasImagePlaneDetails:True",
                "",
                "Images:[module_num:1|enabled:True]",
                "    Filter images?:Images only",
                "",
                '"Version":"1","PlaneCount":"1"',
                '"URL","Series","Index","Channel"',
                '"https://example.invalid/data/url_D.TIF",,,',
            ]
        ),
        encoding="utf-8",
    )
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    (reference_dir / ".cellprofiler_benchmark_reference.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provenance": {
                    "cppipe_path": str(cppipe_path),
                    NativeCellProfilerProvenanceField.INPUT_DOMAIN_STRATEGY: (
                        NativeCellProfilerInputDomainStrategyKey.SELECTED_SOURCE_SCHEMA_WELLS
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    assert native_cellprofiler_reference_is_complete(reference_dir) is True


def test_headless_cellprofiler_cppipe_enables_saveimages_overwrite(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "pipeline.cppipe"
    cppipe_path.write_text(
        "CellProfiler Pipeline: http://www.cellprofiler.org\n"
        "SaveImages:[module_num:1|enabled:True]\n"
        "    Overwrite existing files without warning?:No\n",
        encoding="utf-8",
    )

    execution_path = HeadlessCellProfilerPipelinePolicy.execution_path(
        cppipe_path,
        tmp_path / "outputs",
    )

    assert execution_path != cppipe_path
    assert "Overwrite existing files without warning?:Yes" in execution_path.read_text(
        encoding="utf-8"
    )
    assert "Overwrite existing files without warning?:No" in cppipe_path.read_text(
        encoding="utf-8"
    )


def test_headless_cellprofiler_cppipe_trusts_selected_source_universe(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "pipeline.cppipe"
    cppipe_path.write_text(
        "CellProfiler Pipeline: http://www.cellprofiler.org\n"
        "Images:[module_num:1|enabled:True]\n"
        "    Filter images?:Images only\n",
        encoding="utf-8",
    )

    execution_path = HeadlessCellProfilerPipelinePolicy.execution_path(
        cppipe_path,
        tmp_path / "outputs",
        patches=(HeadlessCellProfilerPipelinePatch.TRUST_SELECTED_SOURCE_UNIVERSE,),
    )

    assert execution_path != cppipe_path
    assert "Filter images?:No filtering" in execution_path.read_text(
        encoding="utf-8"
    )
    assert "Filter images?:Images only" in cppipe_path.read_text(encoding="utf-8")
