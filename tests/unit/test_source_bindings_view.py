from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import ArtifactKind
from openhcs.core.pipeline_image_schema import (
    GroupingPlan,
    ImageAssignment,
    ImportedMetadataJoin,
    ImportedMetadataTable,
    ImagePlaneSource,
    ImagesRule,
    PipelineImageSchema,
    SourceArtifactAssignment,
)
from openhcs.core.source_bindings import (
    ComponentSelector,
    GroupedSourceBindings,
    MetadataExtractionRule,
    MetadataSource,
    MetadataSelector,
    NamedSourceBinding,
    SourceBindingMatchDimension,
    SourceBindingMatchField,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingOrigin,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceSelector,
    StepSourceBindingsConfig,
)
from openhcs.core.source_bindings_view import (
    SourceBindingsPreview,
    SourceBindingsViewModel,
    SourceInventory,
)


class PipelineImageAssignmentFactory:
    """Test factory for canonical pipeline-start image assignments."""

    @staticmethod
    def build(alias: str, selector: SourceSelector) -> ImageAssignment:
        return ImageAssignment(
            alias=alias,
            image_type="Grayscale image",
            selector=selector,
            origin=SourceBindingOrigin.PIPELINE_START,
        )


class FileManagerInventoryStub:
    """Minimal FileManagerLike implementation for VFS inventory tests."""

    def __init__(self, files: tuple[str, ...]) -> None:
        self.files = files
        self.calls: list[tuple[str, str, bool]] = []

    def list_files(self, directory, backend, **kwargs):
        self.calls.append((str(directory), str(backend), bool(kwargs.get("recursive"))))
        return list(self.files)


def test_source_bindings_view_model_projects_pipeline_and_step_bindings():
    schema = PipelineImageSchema(
        images_rule=ImagesRule(
            filters=(
                SourceFilterClause(
                    SourceFilterSubject.EXTENSION,
                    SourceFilterMatchType.IS_TIF,
                ),
            )
        ),
        image_plane_sources=(ImagePlaneSource(uri="file:///tmp/A01_w1.tif"),),
        imported_metadata_tables=(
            ImportedMetadataTable(
                location="metadata.csv",
                joins=(ImportedMetadataJoin("Well", "well_id"),),
            ),
        ),
        assignments_by_alias={
            "DNA": PipelineImageAssignmentFactory.build(
                "DNA",
                SourceSelector(
                    components=(ComponentSelector(AllComponents.CHANNEL, "1"),),
                ),
            ),
        },
        source_artifacts_by_alias={
            "Nuclei": SourceArtifactAssignment(
                alias="Nuclei",
                artifact_kind=ArtifactKind.OBJECT_LABELS,
                payload_type="Objects",
                selector=SourceSelector(
                    metadata=(MetadataSelector("object_type", "nuclei"),),
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
        },
        metadata_rules=(
            MetadataExtractionRule(
                source=MetadataSource.FILE_NAME,
                pattern=r"(?P<well>[A-H]\d{2})_(?P<site>s\d+)\.tif",
            ),
        ),
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        grouping=GroupingPlan(metadata_fields=("Well",)),
    )
    bindings = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                group_key="segment",
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
            ),
        ),
        metadata_rules=(
            MetadataExtractionRule(
                source=MetadataSource.FOLDER_NAME,
                pattern=r"Plate_(?P<plate>\d+)",
            ),
        ),
    )

    view = SourceBindingsViewModel.from_schema_and_bindings(
        schema=schema,
        bindings=bindings,
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
    assert view.pipeline_bindings[1].payload_type == "Objects"
    assert view.step_binding_groups[0].group_key == "segment"
    assert view.step_binding_groups[0].bindings[0].selector.filters[0].value == "DNA"
    assert view.metadata_rules[0].extracted_fields == ("well", "site")
    assert view.metadata_rules[1].declaration_scope == "step"
    assert view.match_plans[0].method == "order"
    assert view.grouping is not None
    assert view.grouping.metadata_fields == ("Well",)
    assert view.artifact_kinds == ("image", "object_labels")


def test_source_bindings_view_model_exposes_metadata_match_dimensions():
    schema = PipelineImageSchema.empty()
    bindings = StepSourceBindingsConfig(
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

    view = SourceBindingsViewModel.from_schema_and_bindings(
        schema=schema,
        bindings=bindings,
    )

    assert view.pipeline_bindings == ()
    assert view.match_plans[0].declaration_scope == "step"
    assert view.match_plans[0].dimensions[0].fields == (
        ("DNA", "well"),
        ("GFP", "well"),
    )


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
    schema = PipelineImageSchema(
        assignments_by_alias={
            "DNA": PipelineImageAssignmentFactory.build(
                "DNA",
                SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "DNA",
                        ),
                    ),
                ),
            ),
            "GFP": PipelineImageAssignmentFactory.build(
                "GFP",
                SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "GFP",
                        ),
                    ),
                ),
            ),
        },
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )
    inventory = SourceInventory.from_paths(
        tuple(sorted(source_root.iterdir())),
        schema=schema,
        source_root=source_root,
    )

    preview = SourceBindingsPreview.from_schema_and_bindings(
        schema=schema,
        bindings=StepSourceBindingsConfig(),
        inventory=inventory,
        sample_limit=1,
    )

    counts_by_alias = {
        row.alias: (row.matched_source_count, row.sample_paths)
        for row in preview.binding_rows
    }
    assert counts_by_alias["DNA"] == (2, ("A01_s1_DNA.tif",))
    assert counts_by_alias["GFP"] == (2, ("A01_s1_GFP.tif",))
    assert len(preview.image_set_rows) == 2
    assert preview.image_set_rows[0].paths_by_alias == (
        ("DNA", "A01_s1_DNA.tif"),
        ("GFP", "A01_s1_GFP.tif"),
    )


def test_source_inventory_from_schema_sources_decodes_file_uri_and_metadata(tmp_path):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source_path = source_root / "A01_s1_DNA.tif"
    source_path.write_text("placeholder", encoding="utf-8")
    schema = PipelineImageSchema(
        image_plane_sources=(ImagePlaneSource(uri=source_path.as_uri()),),
        metadata_rules=(
            MetadataExtractionRule(
                source=MetadataSource.FILE_NAME,
                pattern=r"(?P<well>A\d{2})_(?P<site>s\d+)_(?P<channel>DNA)\.tif",
            ),
        ),
        assignments_by_alias={
            "DNA": PipelineImageAssignmentFactory.build(
                "DNA",
                SourceSelector(
                    metadata=(MetadataSelector("channel", "DNA"),),
                ),
            ),
        },
    )

    inventory = SourceInventory.from_schema_sources(
        schema,
        source_root=source_root,
    )
    preview = SourceBindingsPreview.from_schema_and_bindings(
        schema=schema,
        bindings=StepSourceBindingsConfig(),
        inventory=inventory,
    )

    assert inventory.candidates[0].path == source_path
    assert inventory.candidates[0].relative_path == "A01_s1_DNA.tif"
    assert inventory.candidates[0].metadata["channel"] == "DNA"
    assert preview.binding_rows[0].matched_source_count == 1


def test_source_inventory_from_directory_applies_images_rule(tmp_path):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    for name in ("A01_DNA.tif", "A01_GFP.tif", "notes.txt"):
        (source_root / name).write_text("placeholder", encoding="utf-8")
    schema = PipelineImageSchema(
        images_rule=ImagesRule(
            filters=(
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
        ),
    )

    inventory = SourceInventory.from_directory(source_root, schema=schema)

    assert tuple(candidate.relative_path for candidate in inventory.candidates) == (
        "A01_DNA.tif",
    )


def test_source_inventory_from_schema_context_uses_directory_without_embedded_sources(
    tmp_path,
):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source_path = source_root / "A01_DNA.tif"
    source_path.write_text("placeholder", encoding="utf-8")
    schema = PipelineImageSchema(
        assignments_by_alias={
            "DNA": PipelineImageAssignmentFactory.build(
                "DNA",
                SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "DNA",
                        ),
                    ),
                ),
            ),
        },
    )

    inventory = SourceInventory.from_schema_context(
        schema,
        source_root=source_root,
    )
    preview = SourceBindingsPreview.from_schema_and_bindings(
        schema=schema,
        bindings=StepSourceBindingsConfig(),
        inventory=inventory,
    )

    assert inventory.candidates[0].path == source_path
    assert preview.binding_rows[0].matched_source_count == 1


def test_source_inventory_from_filemanager_uses_vfs_file_listing():
    filemanager = FileManagerInventoryStub(
        files=(
            "/vfs/plate/A01_DNA.tif",
            "/vfs/plate/A01_GFP.tif",
            "/vfs/plate/notes.txt",
        )
    )
    schema = PipelineImageSchema(
        images_rule=ImagesRule(
            filters=(
                SourceFilterClause(
                    SourceFilterSubject.FILE,
                    SourceFilterMatchType.CONTAINS,
                    "DNA",
                ),
            ),
        ),
    )

    inventory = SourceInventory.from_filemanager(
        filemanager=filemanager,
        source_root="/vfs/plate",
        backend="zarr",
        schema=schema,
    )

    assert filemanager.calls == [("/vfs/plate", "zarr", True)]
    assert tuple(candidate.relative_path for candidate in inventory.candidates) == (
        "A01_DNA.tif",
    )
