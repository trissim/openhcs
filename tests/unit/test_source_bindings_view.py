from polystore.virtual_workspace import SourcePixelRef

from openhcs.constants.constants import AllComponents, Backend
from openhcs.core.artifacts import ObjectLabelsArtifactType
from openhcs.core.source_bindings import (
    ComponentSelector,
    ImagePlaneSource,
    ImportedMetadataJoin,
    ImportedMetadataTable,
    MetadataExtractionRule,
    MetadataSource,
    MetadataSelector,
    NamedSourceBinding,
    SourceBindingMatchDimension,
    SourceBindingMatchField,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingOrigin,
    SourceProjectionRole,
    SourceBindingsConfig,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceSelector,
    StepSourceBindingsConfig,
)
from openhcs.core.source_bindings_view import (
    SourceBindingDiagnosticSeverity,
    SourceBindingsPreview,
    SourceBindingsViewModel,
    SourceInventory,
)


class FileManagerInventoryStub:
    """Minimal FileManagerLike implementation for VFS inventory tests."""

    def __init__(self, files: tuple[str, ...]) -> None:
        self.files = files
        self.calls: list[tuple[str, str, bool]] = []

    def list_files(
        self,
        directory: str,
        backend: str,
        *,
        recursive: bool = False,
    ) -> list[str]:
        self.calls.append((str(directory), backend, recursive))
        return list(self.files)


def test_source_bindings_view_model_projects_pipeline_and_step_bindings():
    source_bindings = SourceBindingsConfig(
        source_filters=(
            SourceFilterClause(
                SourceFilterSubject.EXTENSION,
                SourceFilterMatchType.IS_TIF,
            ),
        ),
        image_plane_sources=(ImagePlaneSource(uri="file:///tmp/A01_w1.tif"),),
        imported_metadata_tables=(
            ImportedMetadataTable(
                location="metadata.csv",
                joins=(ImportedMetadataJoin("Well", "well_id"),),
            ),
        ),
        bindings=(
            NamedSourceBinding(
                alias="DNA",
                selector=SourceSelector(
                    components=(ComponentSelector(AllComponents.CHANNEL, "1"),),
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
            NamedSourceBinding(
                alias="Nuclei",
                artifact_kind=ObjectLabelsArtifactType,
                projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
                selector=SourceSelector(
                    metadata=(MetadataSelector("object_type", "nuclei"),),
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
        ),
        metadata_rules=(
            MetadataExtractionRule(
                source=MetadataSource.FILE_NAME,
                pattern=r"(?P<well>[A-H]\d{2})_(?P<site>s\d+)\.tif",
            ),
        ),
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        grouping_metadata_fields=("Well",),
    )
    step_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            NamedSourceBinding(
                alias="LocalDNA",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "DNA",
                        ),
                    ),
                ),
            ),
        ),
        metadata_rules=(
            MetadataExtractionRule(
                source=MetadataSource.FOLDER_NAME,
                pattern=r"Plate_(?P<plate>\d+)",
            ),
        ),
    )

    view = SourceBindingsViewModel.from_config_and_step_bindings(
        source_bindings=source_bindings,
        step_bindings=step_bindings,
    )

    assert view.pipeline_sources.image_plane_source_count == 1
    assert view.pipeline_sources.filters[0].match_type == "is_tif"
    assert view.pipeline_sources.imported_metadata_tables[0].joins == (
        ("Well", "well_id"),
    )
    assert [(row.alias, row.artifact_kind) for row in view.pipeline_bindings] == [
        ("DNA", "image"),
        ("Nuclei", "object_labels"),
    ]
    assert view.pipeline_bindings[0].selector.components == (("channel", "1"),)
    assert view.pipeline_bindings[1].projection_role == "source_artifact"
    assert view.step_bindings[0].selector.filters[0].value == "DNA"
    assert view.metadata_rules[0].extracted_fields == ("well", "site")
    assert view.metadata_rules[1].declaration_scope == "step"
    assert view.match_plans[0].method == "order"
    assert view.grouping is not None
    assert view.grouping.metadata_fields == ("Well",)
    assert view.artifact_kinds == ("image", "object_labels")


def test_source_bindings_view_model_exposes_metadata_match_dimensions():
    step_bindings = StepSourceBindingsConfig(
        enabled=True,
        match_plan=SourceBindingMatchPlan(
            method=SourceBindingMatchMethod.METADATA,
            dimensions=(
                SourceBindingMatchDimension(
                    fields=(
                        SourceBindingMatchField("DNA", "well"),
                        SourceBindingMatchField("GFP", "well"),
                    ),
                ),
            ),
        ),
    )

    view = SourceBindingsViewModel.from_config_and_step_bindings(
        source_bindings=SourceBindingsConfig(),
        step_bindings=step_bindings,
    )

    assert view.pipeline_bindings == ()
    assert view.match_plans[0].declaration_scope == "step"
    assert view.match_plans[0].dimensions[0].fields == (
        ("DNA", "well"),
        ("GFP", "well"),
    )


def test_source_bindings_view_model_accepts_empty_resolved_step_override():
    view = SourceBindingsViewModel.from_config_and_step_bindings(
        source_bindings=SourceBindingsConfig(),
        step_bindings=StepSourceBindingsConfig(enabled=True),
    )

    assert view.step_bindings == ()
    assert view.metadata_rules == ()
    assert view.match_plans == ()


def test_source_bindings_preview_reuses_typed_filter_and_order_matching(tmp_path):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    for name in (
        "A01_s1_DNA.tif",
        "A01_s1_GFP.tif",
        "A02_s1_DNA.tif",
        "A02_s1_GFP.tif",
        "notes.txt",
    ):
        (source_root / name).write_text("placeholder", encoding="utf-8")
    source_bindings = SourceBindingsConfig(
        source_filters=(
            SourceFilterClause(
                SourceFilterSubject.EXTENSION,
                SourceFilterMatchType.IS_TIF,
            ),
        ),
        bindings=(
            NamedSourceBinding(
                alias="DNA",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "DNA",
                        ),
                    ),
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
            NamedSourceBinding(
                alias="GFP",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "GFP",
                        ),
                    ),
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
        ),
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )
    inventory = SourceInventory.from_paths(
        tuple(sorted(source_root.iterdir())),
        source_root=source_root,
        source_backend=Backend.DISK,
        source_bindings=source_bindings,
    )

    preview = SourceBindingsPreview.from_config_and_step_bindings(
        source_bindings=source_bindings,
        step_bindings=StepSourceBindingsConfig(),
        inventory=inventory,
        sample_limit=1,
    )

    counts_by_alias = {
        row.alias: (row.matched_source_count, row.sample_paths)
        for row in preview.binding_rows
    }
    assert counts_by_alias["DNA"] == (2, ("A01_s1_DNA.tif",))
    assert counts_by_alias["GFP"] == (2, ("A01_s1_GFP.tif",))
    assert len(preview.source_set_rows) == 2
    assert preview.source_set_rows[0].paths_by_alias == (
        ("DNA", "A01_s1_DNA.tif"),
        ("GFP", "A01_s1_GFP.tif"),
    )


def test_source_inventory_uses_declared_image_plane_sources(tmp_path):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source_path = source_root / "A01_s1_DNA.tif"
    source_path.write_text("placeholder", encoding="utf-8")
    source_bindings = SourceBindingsConfig(
        image_plane_sources=(ImagePlaneSource(uri=str(source_path)),),
        metadata_rules=(
            MetadataExtractionRule(
                source=MetadataSource.FILE_NAME,
                pattern=r"(?P<well>A\d{2})_(?P<site>s\d+)_(?P<channel>DNA)\.tif",
            ),
        ),
        bindings=(
            NamedSourceBinding(
                alias="DNA",
                selector=SourceSelector(
                    metadata=(MetadataSelector("channel", "DNA"),),
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
        ),
    )

    inventory = SourceInventory.from_paths(
        (),
        source_root=source_root,
        source_backend=Backend.DISK,
        source_bindings=source_bindings,
    )
    preview = SourceBindingsPreview.from_config_and_step_bindings(
        source_bindings=source_bindings,
        step_bindings=StepSourceBindingsConfig(),
        inventory=inventory,
    )

    assert inventory.candidates[0].source_ref == SourcePixelRef(
        backend=Backend.DISK.value,
        backend_address=source_path.name,
    )
    assert inventory.candidates[0].relative_path == "A01_s1_DNA.tif"
    assert inventory.candidates[0].metadata["channel"] == "DNA"
    assert preview.binding_rows[0].matched_source_count == 1


def test_source_inventory_from_paths_applies_source_filters(tmp_path):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    for name in ("A01_DNA.tif", "A01_GFP.tif", "notes.txt"):
        (source_root / name).write_text("placeholder", encoding="utf-8")
    source_bindings = SourceBindingsConfig(
        source_filters=(
            SourceFilterClause(
                SourceFilterSubject.EXTENSION,
                SourceFilterMatchType.IS_TIF,
            ),
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.CONTAINS,
                "DNA",
            ),
        ),
    )

    inventory = SourceInventory.from_paths(
        tuple(sorted(source_root.iterdir())),
        source_root=source_root,
        source_backend=Backend.DISK,
        source_bindings=source_bindings,
    )

    assert tuple(candidate.relative_path for candidate in inventory.candidates) == (
        "A01_DNA.tif",
    )


def test_source_inventory_and_preview_use_resolved_step_override(tmp_path):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    for name in ("A01_DNA.tif", "A01_GFP.tif"):
        (source_root / name).write_text("placeholder", encoding="utf-8")
    source_bindings = SourceBindingsConfig(
        bindings=(
            NamedSourceBinding(
                alias="DNA",
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
        ),
    )
    step_bindings = StepSourceBindingsConfig(
        enabled=True,
        source_filters=(
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.CONTAINS,
                "GFP",
            ),
        ),
        bindings=(
            NamedSourceBinding(
                alias="GFP",
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
        ),
    )

    inventory = SourceInventory.from_paths(
        tuple(sorted(source_root.iterdir())),
        source_root=source_root,
        source_backend=Backend.DISK,
        source_bindings=source_bindings,
        step_bindings=step_bindings,
    )
    preview = SourceBindingsPreview.from_config_and_step_bindings(
        source_bindings=source_bindings,
        step_bindings=step_bindings,
        inventory=inventory,
    )

    assert tuple(candidate.relative_path for candidate in inventory.candidates) == (
        "A01_GFP.tif",
    )
    assert [(row.alias, row.declaration_scope) for row in preview.binding_rows] == [
        ("GFP", "step"),
    ]
    assert preview.binding_rows[0].matched_source_count == 1


def test_source_binding_preview_reports_required_alias_without_matches():
    source_bindings = SourceBindingsConfig(
        bindings=(
            NamedSourceBinding(
                alias="DNA",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "DNA",
                        ),
                    ),
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
        ),
    )

    preview = SourceBindingsPreview.from_config_and_step_bindings(
        source_bindings=source_bindings,
        step_bindings=StepSourceBindingsConfig(),
        inventory=SourceInventory(candidates=()),
    )

    assert preview.diagnostics[0].severity is SourceBindingDiagnosticSeverity.ERROR
    assert preview.diagnostics[0].code == "source_binding.no_match"
    assert preview.diagnostics[0].alias == "DNA"


def test_source_inventory_from_filemanager_uses_vfs_file_listing():
    filemanager = FileManagerInventoryStub(
        files=(
            "/vfs/plate/A01_DNA.tif",
            "/vfs/plate/A01_GFP.tif",
            "/vfs/plate/notes.txt",
        )
    )
    source_bindings = SourceBindingsConfig(
        source_filters=(
            SourceFilterClause(
                SourceFilterSubject.FILE,
                SourceFilterMatchType.CONTAINS,
                "DNA",
            ),
        ),
    )

    inventory = SourceInventory.from_filemanager(
        filemanager=filemanager,
        source_root="/vfs/plate",
        backend=Backend.ZARR.value,
        source_bindings=source_bindings,
    )

    assert filemanager.calls == [("/vfs/plate", "zarr", True)]
    assert tuple(candidate.relative_path for candidate in inventory.candidates) == (
        "A01_DNA.tif",
    )
