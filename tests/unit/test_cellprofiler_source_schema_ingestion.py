from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import tifffile

from openhcs.core.pipeline_image_schema import ImagesRule, PipelineImageSchema
from openhcs.core.source_bindings import (
    MetadataExtractionRule,
    MetadataSource,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
)
from openhcs.interop.cellprofiler.source_schema_ingestion import (
    CellProfilerPipelinePreparationError,
    CellProfilerSourceRootResolver,
    CellProfilerSourceSchemaWorkspace,
    CellProfilerSourceSchemaWorkspaceRequest,
    CellProfilerSourceWorkspaceMaterializationError,
    prepare_cellprofiler_source_schema_only_workspace,
    prepare_cellprofiler_source_schema_workspace,
)


def test_prepare_cellprofiler_source_schema_workspace_materializes_openhcs_metadata(
    tmp_path: Path,
) -> None:
    fixture = SourceSchemaIngestionFixture(tmp_path)
    source_root = tmp_path / "source"
    source_root.mkdir()
    tifffile.imwrite(
        source_root / "A01_s1_D.TIF",
        np.full((4, 4), 1, dtype=np.uint16),
    )
    cppipe_path = fixture.write_names_and_types_cppipe("source_schema")

    result = fixture.prepare(
        source_root=source_root,
        cppipe_path=cppipe_path,
    )

    assert result.materialization is not None
    assert result.execution_plate_path == result.materialization.workspace_root
    assert result.source_workspace_path == result.materialization.workspace_root
    assert result.prepared_pipeline.import_result.source_schema.is_empty is False
    assert result.materialization.metadata_path.exists()
    assert set(result.materialization.primary_mappings) == {
        "A01_s001_w1_z001_t001.TIF",
    }


def test_prepare_cellprofiler_source_schema_workspace_reports_missing_sources(
    tmp_path: Path,
) -> None:
    fixture = SourceSchemaIngestionFixture(tmp_path)
    source_root = tmp_path / "empty_source"
    source_root.mkdir()
    cppipe_path = fixture.write_names_and_types_cppipe("source_schema")

    with pytest.raises(
        CellProfilerSourceWorkspaceMaterializationError,
        match="Source schema image alias 'OrigBlue' matched no image files",
    ):
        fixture.prepare(
            source_root=source_root,
            cppipe_path=cppipe_path,
        )


def test_prepare_cellprofiler_source_schema_workspace_reports_prepare_errors(
    tmp_path: Path,
) -> None:
    fixture = SourceSchemaIngestionFixture(tmp_path)
    bad_cppipe = tmp_path / "bad.cppipe"
    bad_cppipe.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "",
                "DefinitelyMissing:[module_num:1|enabled:True]",
                "",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        CellProfilerPipelinePreparationError,
        match="Failed to prepare converted .cppipe pipeline bad.cppipe",
    ):
        fixture.prepare(
            source_root=tmp_path,
            cppipe_path=bad_cppipe,
        )


def test_prepare_cellprofiler_source_schema_only_workspace_allows_processing_errors(
    tmp_path: Path,
) -> None:
    fixture = SourceSchemaIngestionFixture(tmp_path)
    source_root = tmp_path / "source"
    source_root.mkdir()
    tifffile.imwrite(
        source_root / "A01_s1_D.TIF",
        np.full((4, 4), 1, dtype=np.uint16),
    )
    cppipe_path = fixture.write_incomplete_processing_cppipe("incomplete")

    result = prepare_cellprofiler_source_schema_only_workspace(
        CellProfilerSourceSchemaWorkspaceRequest(
            source_root=source_root,
            cppipe_path=cppipe_path,
            workspace_root=tmp_path / "workspace",
            generated_pipeline_path=tmp_path / "incomplete_openhcs.py",
        )
    )

    assert result.source_schema.is_empty is False
    assert result.materialization is not None
    assert result.materialization.metadata_path.exists()


def test_cellprofiler_source_root_resolver_ignores_nested_pipeline_copies(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "3DNoiseNuclei"
    input_dir = source_root / "Input3DNuclei"
    archive_input_dir = source_root / "Archive_PT" / "Input3DNuclei"
    input_dir.mkdir(parents=True)
    archive_input_dir.mkdir(parents=True)
    (source_root / "Archive_PT" / "3DNucleiPipelineComputeConsumingFinal.cppipe").write_text(
        "",
        encoding="utf-8",
    )
    (input_dir / "nuclei1_out_c00_dr90_image.tif").write_bytes(b"")
    (archive_input_dir / "nuclei1_out_c00_dr90_image.tif").write_bytes(b"")
    (source_root / "__MACOSX").mkdir()
    (source_root / "__MACOSX" / "._nuclei1_out_c00_dr90_image.tif").write_bytes(b"")

    resolved = CellProfilerSourceRootResolver(
        source_root,
        PipelineImageSchema(
            images_rule=ImagesRule(
                filters=(
                    SourceFilterClause(
                        SourceFilterSubject.EXTENSION,
                        SourceFilterMatchType.IS_IMAGE,
                    ),
                ),
            ),
        ),
    ).source_root()

    assert resolved == input_dir


def test_cellprofiler_source_root_resolver_uses_single_child_with_folder_metadata(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "AdvancedSegmentation"
    image_dir = source_root / "BBBC022_20585_AE"
    image_dir.mkdir(parents=True)
    (image_dir / "A01_s1_w1.tif").write_bytes(b"")

    resolved = CellProfilerSourceRootResolver(
        source_root,
        PipelineImageSchema(
            images_rule=ImagesRule(
                filters=(
                    SourceFilterClause(
                        SourceFilterSubject.EXTENSION,
                        SourceFilterMatchType.IS_IMAGE,
                    ),
                ),
            ),
            metadata_rules=(
                MetadataExtractionRule(
                    source=MetadataSource.FOLDER_NAME,
                    pattern=r"(?P<Plate>[0-9]{5})",
                ),
            ),
        ),
    ).source_root()

    assert resolved == image_dir


@dataclass(frozen=True, slots=True)
class SourceSchemaIngestionFixture:
    """Test authority for constructing source-schema ingestion requests."""

    tmp_path: Path

    def prepare(
        self,
        *,
        source_root: Path,
        cppipe_path: Path,
    ) -> CellProfilerSourceSchemaWorkspace:
        return prepare_cellprofiler_source_schema_workspace(
            CellProfilerSourceSchemaWorkspaceRequest(
                source_root=source_root,
                cppipe_path=cppipe_path,
                workspace_root=self.tmp_path / "workspace",
                generated_pipeline_path=(
                    self.tmp_path / f"{cppipe_path.stem}_openhcs.py"
                ),
            )
        )

    def write_names_and_types_cppipe(self, stem: str) -> Path:
        cppipe_path = self.tmp_path / f"{stem}.cppipe"
        cppipe_path.write_text(
            "\n".join(
                (
                    "CellProfiler Pipeline: http://www.cellprofiler.org",
                    "Version:5",
                    "DateRevision:500",
                    "GitHash:",
                    "ModuleCount:4",
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
                    "ExportToSpreadsheet:[module_num:4|enabled:True]",
                    "    Select measurements to export:No",
                    "",
                )
            ),
            encoding="utf-8",
        )
        return cppipe_path

    def write_incomplete_processing_cppipe(self, stem: str) -> Path:
        cppipe_path = self.tmp_path / f"{stem}.cppipe"
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
