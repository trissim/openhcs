from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from benchmark.adapters.cellprofiler import (
    CELLPROFILER_FIRST_IMAGE_SET_PARAM,
    CELLPROFILER_LAST_IMAGE_SET_PARAM,
    DETERMINISTIC_NUMPY_DISABLED_CPU_FEATURES,
    DETERMINISTIC_PYTHONHASHSEED,
    HeadlessCellProfilerPipelinePolicy,
    HeadlessCellProfilerPipelinePatch,
    NativeCellProfilerImportedMetadataPlacementPlan,
    NativeCellProfilerImportedMetadataPipelinePatch,
    NativeCellProfilerInputDomainStrategyKey,
    NativeCellProfilerProvenanceField,
    NUMPY_DISABLED_CPU_FEATURES_ENV,
    PYTHONHASHSEED_ENV,
    CellProfilerAdapter,
)
from benchmark.adapters.cellprofiler_installation import (
    CELLPROFILER_EXECUTABLE_ENV,
    CellProfilerExecutableResolver,
    CellProfilerExecutableSource,
    OPENHCS_BENCHMARK_TOOL_ROOTS_ENV,
)
from benchmark.contracts.tool_adapter import ToolExecutionError, ToolNotInstalledError
from openhcs.core.config import GlobalPipelineConfig, WellFilterConfig
from openhcs.core.component_group_scope import RuntimeExecutionAxisScope
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.source_bindings import (
    ImportedMetadataTable,
    SourceBindingRuntimeContext,
)
from tests.unit.cellprofiler_runtime_test_support import (
    cellprofiler_runtime_adapter_for_test,
)


def _minimal_images_cppipe() -> str:
    return "\n".join(
        (
            "CellProfiler Pipeline: http://www.cellprofiler.org",
            "Version:5",
            "ModuleCount:1",
            "HasImagePlaneDetails:False",
            "",
            "Images:[module_num:1|enabled:True]",
            "    Filter images?:No filtering",
            "",
        )
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

    metadata_table = ImportedMetadataTable(location="20585_AE.csv").resolved(
        source_root
    )
    placements = NativeCellProfilerImportedMetadataPlacementPlan(
        source_root,
        (metadata_table,),
        (image_path, csv_path),
    ).placements()

    assert {
        placement.source_path.name: placement.relative_path.as_posix()
        for placement in placements
    } == {
        "IXMtest_A01_s1_w1.tif": "20585/IXMtest_A01_s1_w1.tif",
        "20585_AE.csv": "20585_AE.csv",
    }


def test_native_cellprofiler_imported_metadata_rejects_unresolved_foreign_path(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "AdvancedSegmentation" / "BBBC022_20585_AE"
    source_root.mkdir(parents=True)
    metadata_path = source_root.parent / "20585_AE.csv"
    metadata_path.write_text("Metadata_Plate\n20585\n", encoding="utf-8")

    table = ImportedMetadataTable(
        location="/Users/pryder/Documents/tutorials/AdvancedSegmentation/20585_AE.csv"
    )
    placement_plan = NativeCellProfilerImportedMetadataPlacementPlan(
        source_root,
        (table,),
        (metadata_path,),
    )

    with pytest.raises(
        ToolExecutionError,
        match="Resolved imported metadata table does not exist",
    ):
        placement_plan.imported_metadata_path(table)


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
            "    Extract metadata from:All images",
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


def test_cellprofiler_adapter_accepts_executable_env(monkeypatch) -> None:
    commands: list[tuple[str, ...]] = []
    monkeypatch.setenv(
        CELLPROFILER_EXECUTABLE_ENV, "/opt/cellprofiler/bin/cellprofiler"
    )

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
    executable = tmp_path / "openhcs" / ".venv-cellprofiler39" / "bin" / "cellprofiler"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")

    resolver = CellProfilerExecutableResolver(
        environment={},
        repo_root=repo_root,
        python_executable=tmp_path / "benchmark-venv" / "bin" / "python",
    )

    assert resolver.resolve() == executable
    assert CellProfilerExecutableSource.LOCAL_WORKSPACE_TOOL_ROOT in {
        candidate.source for candidate in resolver.candidates()
    }


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
    assert CellProfilerExecutableSource.DECLARED_TOOL_ROOT in {
        candidate.source for candidate in resolver.candidates()
    }


def test_cellprofiler_adapter_runs_cppipe_headless(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = tmp_path / "plate"
    dataset_path.mkdir()
    cppipe_path = tmp_path / "pipeline.cppipe"
    cppipe_path.write_text(_minimal_images_cppipe(), encoding="utf-8")
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
        assert (
            env[NUMPY_DISABLED_CPU_FEATURES_ENV]
            == DETERMINISTIC_NUMPY_DISABLED_CPU_FEATURES
        )
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

    adapter = CellProfilerAdapter(executable="/usr/bin/cellprofiler")
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
    assert (
        result.provenance[
            NativeCellProfilerProvenanceField.NUMPY_DISABLED_CPU_FEATURES
        ]
        == DETERMINISTIC_NUMPY_DISABLED_CPU_FEATURES
    )
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
    cppipe_path.write_text(_minimal_images_cppipe(), encoding="utf-8")
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

    adapter = CellProfilerAdapter(executable="/usr/bin/cellprofiler")
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


def test_cellprofiler_adapter_file_list_applies_public_well_filter(
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
                "    Extract metadata from:All images",
                "    Metadata source:File name",
                "    Regular expression to extract from file name:^Plate_(?P<Well>[A-Z][0-9]{2})_s(?P<Site>[0-9]+)_(?P<ChannelName>.*)",
                "",
                "NamesAndTypes:[module_num:3|enabled:True]",
                "    Assign a name to:Images matching rules",
                "    Select the image type:Grayscale image",
                "    Name to assign these images:DNA",
                '    Match metadata:[{"DNA":"Well","Protein":"Well"},{"DNA":"Site","Protein":"Site"}]',
                "    Image set matching method:Metadata",
                "    Set intensity range from:Image metadata",
                "    Assignments count:2",
                '    Select the rule criteria:and (metadata does Well) (file does contain "dapi")',
                "    Name to assign these images:DNA",
                "    Name to assign these objects:Cell",
                "    Select the image type:Grayscale image",
                "    Set intensity range from:Image metadata",
                '    Select the rule criteria:and (metadata does Well) (file does contain "gfp")',
                "    Name to assign these images:Protein",
                "    Name to assign these objects:Cell",
                "    Select the image type:Grayscale image",
                "    Set intensity range from:Image metadata",
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
        global_config=GlobalPipelineConfig(
            well_filter_config=WellFilterConfig(well_filter=1),
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
    assert len(file_list) == 4
    assert all("Plate_A01_" in entry for entry in file_list)
    assert any("_s1_dapi" in entry for entry in file_list)
    assert any("_s1_gfp" in entry for entry in file_list)
    assert any("_s2_dapi" in entry for entry in file_list)
    assert any("_s2_gfp" in entry for entry in file_list)
    assert not any("Plate_A02_" in entry for entry in file_list)
    assert result.provenance[NativeCellProfilerProvenanceField.SELECTED_WELLS] == (
        "A01",
    )
    assert (
        result.provenance[NativeCellProfilerProvenanceField.SELECTED_SOURCE_FILE_COUNT]
        == 4
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
                '    Select the rule criteria:and (file does contain "D.TIF")',
                "    Name to assign these images:OrigBlue",
                "    Name to assign these objects:Cell",
                "    Select the image type:Grayscale image",
                "    Set intensity range from:Image metadata",
                "    Maximum intensity:255.0",
                '    Select the rule criteria:and (file does contain "F.TIF")',
                "    Name to assign these images:OrigGreen",
                "    Name to assign these objects:Cell",
                "    Select the image type:Grayscale image",
                "    Set intensity range from:Image metadata",
                "    Maximum intensity:255.0",
                "",
                '"Version":"1","PlaneCount":"2"',
                '"URL","Series","Index","Channel"',
                f'"{(dataset_path / "url_D.TIF").as_uri()}",,,',
                f'"{(dataset_path / "url_F.TIF").as_uri()}",,,',
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
    assert "Filter images?:No filtering" in execution_path.read_text(encoding="utf-8")
    assert "Filter images?:Images only" in cppipe_path.read_text(encoding="utf-8")
