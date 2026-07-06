from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from openhcs.core import source_bindings as source_bindings_module
from openhcs.constants import AllComponents
from openhcs.core.artifacts import ImageArtifactType, ObjectLabelsArtifactType
from openhcs.core.pipeline_image_schema import (
    GroupingPlan,
    ImageAssignment,
    ImagePlaneSource,
    ImagesRule,
    ImportedMetadataJoin,
    ImportedMetadataTable,
    PipelineImageSchema,
    SOURCE_IMAGE_TYPE_METADATA_FIELD,
    SourceArtifactAssignment,
)
from openhcs.core.source_bindings import (
    ComponentSelector,
    CompiledSourceBindingPlan,
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
)
from openhcs.core.source_schema_workspace import (
    ImageSetRecord,
    SOURCE_SCHEMA_WORKSPACE_SOURCE_DIR,
    SourceSchemaImageSetSelection,
    SourceSchemaCandidateDiscovery,
    SourceSchemaCandidateDiscoveryRequest,
    expand_source_schema_workspace_wells as expand_A01_schema_workspace_wells,
    materialize_source_schema_workspace as materialize_A01_schema_workspace,
    source_schema_metadata_with_virtual_components,
)
from openhcs.core.source_binding_selection import (
    SourceAnchorSelectionStatus,
    SourceBindingCandidateMatcher,
    SourceBindingMatchedImageSet,
)
from openhcs.core.source_matching import (
    ORIGINAL_SOURCE_METADATA_FIELD,
    metadata_from_rules,
    source_component_metadata_values,
    source_metadata_value,
)
from openhcs.core.steps.function_execution import (
    SourceBoundAnchorPatternPolicy,
    SourcePatternResolutionContext,
)
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser


def _source_pattern_context() -> SourcePatternResolutionContext:
    return SourcePatternResolutionContext(
        parser=SourceSchemaFilenameParser(),
        source_paths_by_virtual_path={},
    )


def test_folder_metadata_rules_support_slash_qualified_folder_patterns() -> None:
    path = "/source/Sequence1/DrosophilaEmbryo_GFPHistone_0000.tif"

    basename_metadata = metadata_from_rules(
        path,
        (
            MetadataExtractionRule(
                source=MetadataSource.FOLDER_NAME,
                pattern=r"(?P<Run>.*)$",
            ),
        ),
    )
    slash_qualified_metadata = metadata_from_rules(
        path,
        (
            MetadataExtractionRule(
                source=MetadataSource.FOLDER_NAME,
                pattern=r".*[\\/](?P<Run>.*)$",
            ),
        ),
    )

    assert basename_metadata["Run"] == "Sequence1"
    assert slash_qualified_metadata["Run"] == "Sequence1"


def _write_tiff_stack(path: Path, values: tuple[int, ...]) -> None:
    stack = np.stack(
        [np.full((4, 4), value, dtype=np.uint16) for value in values],
        axis=0,
    )
    tifffile.imwrite(path, stack, metadata={"axes": "ZYX"})


def test_A01_schema_filename_parser_handles_artifact_suffixes() -> None:
    parser = SourceSchemaFilenameParser()

    expected = {
        "well": "A01",
        "site": 1,
        "channel": 1,
        "z_index": 1,
        "timepoint": 1,
    }

    assert parser.parse_filename(
        "A01_s001_w1_z001_t001_CorrectIlluminationCalculate_7_measurements_step2.csv"
    ) == {**expected, "extension": ".csv"}
    assert parser.parse_filename(
        "A01_s001_w1_z001_t001_Nuclei_step3_rois.roi.zip"
    ) == {**expected, "extension": ".roi.zip"}


def test_source_schema_metadata_with_virtual_components_overlays_canonical_axes() -> None:
    metadata = source_schema_metadata_with_virtual_components(
        "A01_s002_w3_z001_t001.TIF",
        {
            "OpenHCSImageType": "Grayscale image",
            "Site": "POS002",
            "ChannelNumber": "2",
            "ChannelName": "DNA",
        },
    )

    assert metadata == {
        "OpenHCSImageType": "Grayscale image",
        ORIGINAL_SOURCE_METADATA_FIELD: {
            "ChannelNumber": "2",
            "Site": "POS002",
        },
        "ChannelName": "DNA",
        "well": "A01",
        "site": "2",
        "channel": "3",
        "z_index": "1",
        "timepoint": "1",
        "extension": ".TIF",
    }
    assert source_metadata_value(metadata, "ChannelNumber") == "2"
    assert source_metadata_value(metadata, "channel") == "3"
    assert source_metadata_value(metadata, "Site") == "POS002"
    assert source_metadata_value(metadata, "site") == "2"
    assert source_component_metadata_values(metadata, AllComponents.CHANNEL) == ("3",)


def test_source_schema_candidate_discovery_uses_openhcs_workspace_metadata(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "openhcs_workspace"
    source_root.mkdir()
    (source_root / "raw").mkdir()
    _write_image(source_root / "raw" / "source.ome.tiff", value=1)
    _write_image(source_root / "raw_source_that_should_not_win.tif", value=2)
    virtual_path = "A01_s001_w1_z001_t001.tif"
    (source_root / "openhcs_metadata.json").write_text(
        json.dumps(
            {
                "subdirectories": {
                    "default": {
                        "workspace_mapping": {
                            virtual_path: "raw/source.ome.tiff",
                        },
                        "source_metadata": {
                            virtual_path: {
                                "plate": "Plate1",
                                ORIGINAL_SOURCE_METADATA_FIELD: {
                                    "ChannelNumber": "2",
                                },
                            },
                        },
                        "channels": {
                            "1": "DAPI",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    schema = PipelineImageSchema(
        assignments_by_alias={
            "DNA": ImageAssignment(
                alias="DNA",
                image_type="Grayscale image",
                origin=SourceBindingOrigin.PIPELINE_START,
                selector=SourceSelector(
                    components=(ComponentSelector(AllComponents.CHANNEL, "1"),),
                ),
            ),
        },
    )

    candidates = SourceSchemaCandidateDiscovery(
        SourceSchemaCandidateDiscoveryRequest(
            source_root,
            source_files=(source_root / "raw_source_that_should_not_win.tif",),
            schema=schema,
        )
    ).candidates()

    assert tuple(candidate.relative_path for candidate in candidates) == (
        virtual_path,
    )
    assert tuple(candidate.source_filter_paths for candidate in candidates) == (
        ("raw/source.ome.tiff",),
    )
    assert candidates[0].path == source_root / "raw/source.ome.tiff"
    assert candidates[0].metadata["channel_name"] == "DAPI"
    assert candidates[0].metadata["plate"] == "Plate1"
    assert source_metadata_value(candidates[0].metadata, "ChannelNumber") == "2"


def test_source_schema_candidate_discovery_folder_metadata_uses_folder_basename_for_imported_join(
    tmp_path: Path,
) -> None:
    source_parent = tmp_path / "visible_sources_20260627"
    source_root = source_parent / "BBBC022_20585_AE"
    source_root.mkdir(parents=True)
    image_path = source_root / "IXMtest_A01_s1_w164FBEEF7-F77C-4892-86F5-72D0160D4FB2.tif"
    _write_image(image_path, value=1)
    (source_parent / "20585_AE.csv").write_text(
        "Image_Metadata_PlateID,Image_Metadata_CPD_WELL_POSITION,Compound\n"
        "20585,A01,DMSO\n",
        encoding="utf-8",
    )
    schema = PipelineImageSchema(
        metadata_rules=(
            MetadataExtractionRule(
                source=MetadataSource.FILE_NAME,
                pattern=(
                    r"IXMtest_(?P<Well>[A-P][0-9]{2})_s(?P<Site>[0-9])_w"
                    r"(?P<ChannelNumber>[0-9])"
                ),
            ),
            MetadataExtractionRule(
                source=MetadataSource.FOLDER_NAME,
                pattern=r"(?P<Plate>[0-9]{5})",
            ),
        ),
        imported_metadata_tables=(
            ImportedMetadataTable(
                location="20585_AE.csv",
                joins=(
                    ImportedMetadataJoin("Plate", "Image_Metadata_PlateID"),
                    ImportedMetadataJoin(
                        "Well",
                        "Image_Metadata_CPD_WELL_POSITION",
                    ),
                ),
            ),
        ),
    )

    candidates = SourceSchemaCandidateDiscovery(
        SourceSchemaCandidateDiscoveryRequest(
            source_root,
            source_files=(image_path,),
            schema=schema,
        )
    ).candidates()

    assert len(candidates) == 1
    assert candidates[0].metadata["Plate"] == "20585"
    assert candidates[0].metadata["Compound"] == "DMSO"


def test_source_schema_candidate_discovery_auto_skips_incompatible_metadata_provider(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "mixed_source"
    source_root.mkdir()
    local_image = source_root / "Channel1.tif"
    _write_image(local_image, value=1)
    (source_root / "openhcs_metadata.json").write_text(
        json.dumps(
            {
                "subdirectories": {
                    "default": {
                        "workspace_mapping": {
                            "A01_s001_w9_z001_t001.tif": "raw/source.ome.tiff",
                        },
                        "source_metadata": {},
                        "channels": {
                            "9": "Other",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    schema = PipelineImageSchema(
        assignments_by_alias={
            "DNA": ImageAssignment(
                alias="DNA",
                image_type="Grayscale image",
                origin=SourceBindingOrigin.PIPELINE_START,
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            subject=SourceFilterSubject.FILE,
                            match_type=SourceFilterMatchType.CONTAINS,
                            value="Channel1",
                        ),
                    ),
                ),
            ),
        },
    )

    candidates = SourceSchemaCandidateDiscovery(
        SourceSchemaCandidateDiscoveryRequest(
            source_root,
            source_files=(local_image,),
            schema=schema,
        )
    ).candidates()

    assert tuple(candidate.relative_path for candidate in candidates) == (
        "Channel1.tif",
    )


def test_materialize_openhcs_workspace_preserves_mapped_source_selector_semantics(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "openhcs_workspace"
    workspace_root = tmp_path / "materialized"
    source_root.mkdir()
    virtual_dna_path = "A01_s001_w1_z001_t001.tif"
    virtual_gfp_path = "A01_s001_w2_z001_t001.tif"
    dna_source_path = "raw/BBBC013_A01_s1_w2.tif"
    gfp_source_path = "raw/BBBC013_A01_s1_w1.tif"
    (source_root / "openhcs_metadata.json").write_text(
        json.dumps(
            {
                "subdirectories": {
                    "default": {
                        "workspace_mapping": {
                            virtual_dna_path: dna_source_path,
                            virtual_gfp_path: gfp_source_path,
                        },
                        "source_metadata": {
                            virtual_dna_path: {
                                "Plate": "BBBC013",
                                "well": "A01",
                                "site": "1",
                                "z_index": "1",
                                "timepoint": "1",
                                "channel": "1",
                            },
                            virtual_gfp_path: {
                                "Plate": "BBBC013",
                                "well": "A01",
                                "site": "1",
                                "z_index": "1",
                                "timepoint": "1",
                                "channel": "2",
                            },
                        },
                        "channels": {
                            "1": "rawDNA",
                            "2": "rawGFP",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    schema = PipelineImageSchema(
        assignments_by_alias={
            "rawDNA": ImageAssignment(
                alias="rawDNA",
                image_type="Grayscale image",
                origin=SourceBindingOrigin.PIPELINE_START,
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            subject=SourceFilterSubject.FILE,
                            match_type=SourceFilterMatchType.CONTAINS,
                            value="w2",
                        ),
                    ),
                ),
            ),
            "rawGFP": ImageAssignment(
                alias="rawGFP",
                image_type="Grayscale image",
                origin=SourceBindingOrigin.PIPELINE_START,
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            subject=SourceFilterSubject.FILE,
                            match_type=SourceFilterMatchType.CONTAINS,
                            value="w1",
                        ),
                    ),
                ),
            ),
        },
    )

    materialize_A01_schema_workspace(source_root, workspace_root, schema)

    materialized = json.loads(
        (workspace_root / "openhcs_metadata.json").read_text(encoding="utf-8")
    )["subdirectories"]["."]
    assert materialized["workspace_mapping"][virtual_dna_path].endswith(
        "/raw/BBBC013_A01_s1_w2.tif"
    )
    assert materialized["workspace_mapping"][virtual_gfp_path].endswith(
        "/raw/BBBC013_A01_s1_w1.tif"
    )
    assert materialized["source_metadata"][virtual_dna_path]["channel_label"] == "rawDNA"
    assert materialized["source_metadata"][virtual_gfp_path]["channel_label"] == "rawGFP"


def test_source_binding_plan_views_are_registered_nominal_family() -> None:
    assert set(CompiledSourceBindingPlan.registered_plan_types()) == {
        source_bindings_module.SourceBindingsConfig,
        source_bindings_module.StepSourceBindingsConfig,
        CompiledSourceBindingPlan,
    }


def test_source_bound_anchor_filter_uses_declared_A01_selectors() -> None:
    patterns = [
        "A01_s001_w1_z001_t001.png",
        "A01_s001_w1_z001_t001.tiff",
        "A01_s002_w1_z001_t001.png",
    ]
    binding = NamedSourceBinding(
        alias="phase",
        artifact_kind=ImageArtifactType,
        selector=SourceSelector(
            components=(ComponentSelector(AllComponents.SITE, "1"),),
            filters=(
                SourceFilterClause(
                    subject=SourceFilterSubject.EXTENSION,
                    match_type=SourceFilterMatchType.EQUALS,
                    value=".png",
                ),
            ),
        ),
        origin=SourceBindingOrigin.PIPELINE_START,
    )

    plan = CompiledSourceBindingPlan(bindings=(binding,),
        match_plan=None,
    )
    filtered = SourceBoundAnchorPatternPolicy.for_plan(plan).select(
        patterns,
        bindings=plan.bindings,
        source_context=_source_pattern_context(),
    )

    assert filtered == ["A01_s001_w1_z001_t001.png"]


def test_source_bound_anchor_filter_resolves_template_workspace_sources() -> None:
    binding = NamedSourceBinding(
        alias="OrigBlue",
        selector=SourceSelector(
            filters=(
                SourceFilterClause(
                    subject=SourceFilterSubject.FILE,
                    match_type=SourceFilterMatchType.CONTAINS,
                    value="hoe",
                ),
            ),
        ),
        origin=SourceBindingOrigin.PIPELINE_START,
    )
    plan = CompiledSourceBindingPlan(bindings=(binding,),
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )
    source_context = SourcePatternResolutionContext(
        parser=SourceSchemaFilenameParser(),
        source_paths_by_virtual_path={
            "A01_s001_w1_z001_t001.tif": "/source/1-162hrhoe2.tif",
            "A01_s001_w2_z001_t001.tif": "/source/1-162hrh2ax2.tif",
        },
    )

    filtered = SourceBoundAnchorPatternPolicy.for_plan(plan).select(
        [
            "A01_s{iii}_w1_z001_t001.tif",
            "A01_s{iii}_w2_z001_t001.tif",
        ],
        bindings=plan.bindings,
        source_context=source_context,
    )

    assert filtered == ["A01_s{iii}_w1_z001_t001.tif"]


def test_order_matched_source_bound_anchor_filter_uses_one_anchor_per_image_set() -> None:
    patterns = [
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w2_z001_t001.tif",
        "A01_s001_w3_z001_t001.tif",
        "A01_s002_w1_z001_t001.tif",
        "A01_s002_w2_z001_t001.tif",
        "A01_s002_w3_z001_t001.tif",
    ]
    bindings = (
        NamedSourceBinding(
            alias="origDNA",
            selector=SourceSelector(
                components=(ComponentSelector(AllComponents.CHANNEL, "1"),)
            ),
            origin=SourceBindingOrigin.PIPELINE_START,
        ),
        NamedSourceBinding(
            alias="origMemb",
            selector=SourceSelector(
                components=(ComponentSelector(AllComponents.CHANNEL, "2"),)
            ),
            origin=SourceBindingOrigin.PIPELINE_START,
        ),
        NamedSourceBinding(
            alias="origMito",
            selector=SourceSelector(
                components=(ComponentSelector(AllComponents.CHANNEL, "3"),)
            ),
            origin=SourceBindingOrigin.PIPELINE_START,
        ),
    )
    plan = CompiledSourceBindingPlan(bindings=bindings,
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )

    filtered = SourceBoundAnchorPatternPolicy.for_plan(plan).select(
        patterns,
        bindings=plan.bindings,
        source_context=_source_pattern_context(),
    )

    assert filtered == [
        "A01_s001_w1_z001_t001.tif",
        "A01_s002_w1_z001_t001.tif",
    ]


def test_order_matched_source_bound_anchor_filter_rejects_incomplete_image_sets() -> None:
    bindings = (
        NamedSourceBinding(
            alias="origDNA",
            selector=SourceSelector(
                components=(ComponentSelector(AllComponents.CHANNEL, "1"),)
            ),
            origin=SourceBindingOrigin.PIPELINE_START,
        ),
        NamedSourceBinding(
            alias="origMemb",
            selector=SourceSelector(
                components=(ComponentSelector(AllComponents.CHANNEL, "2"),)
            ),
            origin=SourceBindingOrigin.PIPELINE_START,
        ),
    )
    plan = CompiledSourceBindingPlan(bindings=bindings,
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )

    with pytest.raises(ValueError, match="incomplete image set"):
        SourceBoundAnchorPatternPolicy.for_plan(plan).select(
            ["A01_s001_w1_z001_t001.tif"],
            bindings=plan.bindings,
            source_context=_source_pattern_context(),
        )


def test_order_matched_source_workspace_anchor_filter_accepts_component_narrowing() -> None:
    bindings = (
        NamedSourceBinding(
            alias="OrigStain1",
            selector=SourceSelector(
                filters=(
                    SourceFilterClause(
                        SourceFilterSubject.FILE,
                        SourceFilterMatchType.CONTAINS,
                        "N_R",
                    ),
                ),
            ),
            origin=SourceBindingOrigin.PIPELINE_START,
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
                ),
            ),
            origin=SourceBindingOrigin.PIPELINE_START,
        ),
    )
    plan = CompiledSourceBindingPlan(bindings=bindings,
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )
    source_context = SourcePatternResolutionContext(
        parser=SourceSchemaFilenameParser(),
        source_paths_by_virtual_path={
            "A01_s001_w1_z001_t001.tif": "/source/Image_N_R.tif",
            "A01_s001_w2_z001_t001.tif": "/source/Image_N_G.tif",
        },
    )

    filtered = SourceBoundAnchorPatternPolicy.for_plan(plan).select(
        ["A01_s{iii}_w1_z001_t001.tif"],
        bindings=plan.bindings,
        source_context=source_context,
    )

    assert filtered == ["A01_s{iii}_w1_z001_t001.tif"]


def test_metadata_matched_source_workspace_template_uses_image_set_anchor() -> None:
    bindings = (
        NamedSourceBinding(
            alias="OrigProtein",
            selector=SourceSelector(
                filters=(
                    SourceFilterClause(
                        SourceFilterSubject.FILE,
                        SourceFilterMatchType.CONTAINS,
                        "Channel 1",
                    ),
                ),
            ),
            origin=SourceBindingOrigin.PIPELINE_START,
        ),
        NamedSourceBinding(
            alias="OrigDNA",
            selector=SourceSelector(
                filters=(
                    SourceFilterClause(
                        SourceFilterSubject.FILE,
                        SourceFilterMatchType.CONTAINS,
                        "Channel 2",
                    ),
                ),
            ),
            origin=SourceBindingOrigin.PIPELINE_START,
        ),
    )
    plan = CompiledSourceBindingPlan(bindings=bindings,
        match_plan=SourceBindingMatchPlan(
            method=SourceBindingMatchMethod.METADATA,
            dimensions=(
                SourceBindingMatchDimension(
                    fields=(
                        SourceBindingMatchField(
                            alias="OrigProtein",
                            metadata_field="Folder",
                        ),
                        SourceBindingMatchField(
                            alias="OrigDNA",
                            metadata_field="Folder",
                        ),
                    )
                ),
            ),
        ),
    )
    source_context = SourcePatternResolutionContext(
        parser=SourceSchemaFilenameParser(),
        source_paths_by_virtual_path={
            "A01_s001_w1_z001_t001.tif": "/source/Channel 1-A01.tif",
            "A01_s001_w2_z001_t001.tif": "/source/Channel 2-A01.tif",
        },
        source_metadata_by_path={
            "A01_s001_w1_z001_t001.tif": {"Folder": "/source", "well": "A01"},
            "A01_s001_w2_z001_t001.tif": {"Folder": "/source", "well": "A01"},
        },
    )

    filtered = SourceBoundAnchorPatternPolicy.for_plan(plan).select(
        ["A01_s001_w{iii}_z001_t001.tif"],
        bindings=plan.bindings,
        source_context=source_context,
    )

    assert filtered == ["A01_s001_w{iii}_z001_t001.tif"]


def test_source_bound_anchor_filter_defers_unavailable_metadata_selector() -> None:
    binding = NamedSourceBinding(
        alias="origDNA",
        selector=SourceSelector(
            metadata=(MetadataSelector("ChannelNumber", "2"),),
        ),
        origin=SourceBindingOrigin.PIPELINE_START,
    )
    plan = CompiledSourceBindingPlan(bindings=(binding,),
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )
    source_context = SourcePatternResolutionContext(
        parser=SourceSchemaFilenameParser(),
        source_paths_by_virtual_path={
            "A01_s001_w1_z001_t001.tif": "/source/3d_monolayer_xy1_ch2.tif",
            "A01_s001_w2_z001_t001.tif": "/source/3d_monolayer_xy1_ch1.tif",
        },
        source_metadata_by_path={
            "A01_s001_w1_z001_t001.tif": {"channel": "1", "well": "A01"},
            "A01_s001_w2_z001_t001.tif": {"channel": "2", "well": "A01"},
        },
    )

    filtered = SourceBoundAnchorPatternPolicy.for_plan(plan).select(
        ["A01_s001_w{iii}_z001_t001.tif"],
        bindings=plan.bindings,
        source_context=source_context,
    )

    assert filtered == ["A01_s001_w{iii}_z001_t001.tif"]


def test_source_bound_anchor_filter_reports_runtime_defer_authority() -> None:
    binding = NamedSourceBinding(
        alias="origDNA",
        selector=SourceSelector(
            metadata=(MetadataSelector("ChannelNumber", "2"),),
        ),
        origin=SourceBindingOrigin.PIPELINE_START,
    )
    plan = CompiledSourceBindingPlan(bindings=(binding,),
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )
    source_context = SourcePatternResolutionContext(
        parser=SourceSchemaFilenameParser(),
        source_paths_by_virtual_path={
            "A01_s001_w1_z001_t001.tif": "/source/3d_monolayer_xy1_ch2.tif",
        },
        source_metadata_by_path={
            "A01_s001_w1_z001_t001.tif": {"channel": "1", "well": "A01"},
        },
    )

    selection = SourceBoundAnchorPatternPolicy.for_plan(
        plan
    )._source_compatible_anchor_selection(
        ["A01_s001_w{iii}_z001_t001.tif"],
        bindings=plan.bindings,
        source_context=source_context,
    )

    assert selection.status is SourceAnchorSelectionStatus.DEFERRED_TO_RUNTIME
    assert selection.patterns == ("A01_s001_w{iii}_z001_t001.tif",)


def test_source_bound_anchor_filter_does_not_defer_available_metadata_mismatch() -> None:
    binding = NamedSourceBinding(
        alias="origDNA",
        selector=SourceSelector(
            metadata=(MetadataSelector("ChannelNumber", "2"),),
        ),
        origin=SourceBindingOrigin.PIPELINE_START,
    )
    plan = CompiledSourceBindingPlan(bindings=(binding,),
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )
    source_context = SourcePatternResolutionContext(
        parser=SourceSchemaFilenameParser(),
        source_paths_by_virtual_path={
            "A01_s001_w1_z001_t001.tif": "/source/3d_monolayer_xy1_ch1.tif",
        },
        source_metadata_by_path={
            "A01_s001_w1_z001_t001.tif": {
                "ChannelNumber": "1",
                "well": "A01",
            },
        },
    )

    selection = SourceBoundAnchorPatternPolicy.for_plan(
        plan
    )._source_compatible_anchor_selection(
        ["A01_s001_w{iii}_z001_t001.tif"],
        bindings=plan.bindings,
        source_context=source_context,
    )

    assert selection.status is SourceAnchorSelectionStatus.SELECTED
    assert selection.patterns == ()


def test_runtime_source_binding_metadata_selector_uses_literal_metadata_fields() -> None:
    binding = NamedSourceBinding(
        alias="origDNA",
        selector=SourceSelector(
            metadata=(MetadataSelector("ChannelNumber", "2"),),
        ),
        origin=SourceBindingOrigin.PIPELINE_START,
    )
    source_context = SourcePatternResolutionContext(
        parser=SourceSchemaFilenameParser(),
        source_paths_by_virtual_path={
            "A01_s001_w1_z001_t001.tif": "/source/3d_monolayer_xy1_ch1.tif",
            "A01_s001_w2_z001_t001.tif": "/source/3d_monolayer_xy1_ch2.tif",
        },
        source_metadata_by_path={
            "A01_s001_w1_z001_t001.tif": {
                "ChannelNumber": "1",
                "channel": "1",
                "well": "A01",
            },
            "A01_s001_w2_z001_t001.tif": {
                "ChannelNumber": "2",
                "channel": "2",
                "well": "A01",
            },
        },
    )

    assert source_context.has_metadata_field(
        ("A01_s001_w1_z001_t001.tif",),
        "ChannelNumber",
    )
    assert SourceBindingCandidateMatcher.compatible_candidates(
        (
            "A01_s001_w1_z001_t001.tif",
            "A01_s001_w2_z001_t001.tif",
        ),
        bindings=(binding,),
        source_context=source_context,
    ) == ("A01_s001_w2_z001_t001.tif",)


def test_runtime_source_binding_component_selector_uses_component_metadata() -> None:
    binding = NamedSourceBinding(
        alias="origDNA",
        selector=SourceSelector(
            components=(ComponentSelector(AllComponents.CHANNEL, "2"),),
        ),
        origin=SourceBindingOrigin.PIPELINE_START,
    )
    source_context = SourcePatternResolutionContext(
        parser=SourceSchemaFilenameParser(),
        source_paths_by_virtual_path={
            "A01_s001_w1_z001_t001.tif": "/source/3d_monolayer_xy1_ch1.tif",
            "A01_s001_w2_z001_t001.tif": "/source/3d_monolayer_xy1_ch2.tif",
        },
        source_metadata_by_path={
            "A01_s001_w1_z001_t001.tif": {"channel": "1", "well": "A01"},
            "A01_s001_w2_z001_t001.tif": {"channel": "2", "well": "A01"},
        },
    )

    assert SourceBindingCandidateMatcher.compatible_candidates(
        (
            "A01_s001_w1_z001_t001.tif",
            "A01_s001_w2_z001_t001.tif",
        ),
        bindings=(binding,),
        source_context=source_context,
    ) == ("A01_s001_w2_z001_t001.tif",)


def test_matched_image_set_rebases_single_alias_source_stack_from_source_universe() -> None:
    binding = NamedSourceBinding(
        alias="origDNA",
        selector=SourceSelector(
            metadata=(MetadataSelector("ChannelNumber", "2"),),
        ),
        origin=SourceBindingOrigin.PIPELINE_START,
    )
    source_context = SourcePatternResolutionContext(
        parser=SourceSchemaFilenameParser(),
        source_paths_by_virtual_path={
            "A01_s001_w1_z001_t001.tif": "/source/3d_monolayer_xy1_ch2.tif",
            "A01_s001_w2_z001_t001.tif": "/source/3d_monolayer_xy1_ch1.tif",
            "A01_s001_w1_z002_t001.tif": "/source/3d_monolayer_xy1_ch2.tif",
            "A01_s001_w2_z002_t001.tif": "/source/3d_monolayer_xy1_ch1.tif",
        },
        source_metadata_by_path={
            "A01_s001_w1_z001_t001.tif": {
                "well": "A01",
                "site": "1",
                "timepoint": "1",
                "z_index": "1",
                "channel": "1",
                ORIGINAL_SOURCE_METADATA_FIELD: {"ChannelNumber": "2"},
            },
            "A01_s001_w2_z001_t001.tif": {
                "well": "A01",
                "site": "1",
                "timepoint": "1",
                "z_index": "1",
                "channel": "2",
                ORIGINAL_SOURCE_METADATA_FIELD: {"ChannelNumber": "1"},
            },
            "A01_s001_w1_z002_t001.tif": {
                "well": "A01",
                "site": "1",
                "timepoint": "1",
                "z_index": "2",
                "channel": "1",
                ORIGINAL_SOURCE_METADATA_FIELD: {"ChannelNumber": "2"},
            },
            "A01_s001_w2_z002_t001.tif": {
                "well": "A01",
                "site": "1",
                "timepoint": "1",
                "z_index": "2",
                "channel": "2",
                ORIGINAL_SOURCE_METADATA_FIELD: {"ChannelNumber": "1"},
            },
        },
    )

    assert SourceBindingMatchedImageSet.from_plan(
        bindings=(binding,),
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        source_context=source_context,
        plane_member_fields=frozenset((AllComponents.Z_INDEX.value,)),
    ).expand(
        (
            "A01_s001_w2_z001_t001.tif",
            "A01_s001_w2_z002_t001.tif",
        ),
        source_universe=(
            "A01_s001_w1_z001_t001.tif",
            "A01_s001_w2_z001_t001.tif",
            "A01_s001_w1_z002_t001.tif",
            "A01_s001_w2_z002_t001.tif",
        ),
    ) == (
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w1_z002_t001.tif",
    )


def test_matched_image_set_prefers_complete_alias_set_from_mapped_sources() -> None:
    hoechst = NamedSourceBinding(
        alias="OrigHoechst",
        selector=SourceSelector(
            filters=(
                SourceFilterClause(
                    subject=SourceFilterSubject.FILE,
                    match_type=SourceFilterMatchType.CONTAINS,
                    value="_w1",
                ),
            ),
        ),
        origin=SourceBindingOrigin.PIPELINE_START,
    )
    ph_golgi = NamedSourceBinding(
        alias="OrigPh_golgi",
        selector=SourceSelector(
            filters=(
                SourceFilterClause(
                    subject=SourceFilterSubject.FILE,
                    match_type=SourceFilterMatchType.CONTAINS,
                    value="_w4",
                ),
            ),
        ),
        origin=SourceBindingOrigin.PIPELINE_START,
    )
    source_context = SourcePatternResolutionContext(
        parser=SourceSchemaFilenameParser(),
        source_paths_by_virtual_path={
            "A01_s001_w1_z001_t001.tif": "/source/IXMtest_A01_s1_w2.tif",
            "A01_s001_w2_z001_t001.tif": "/source/IXMtest_A01_s1_w1.tif",
            "A01_s001_w4_z001_t001.tif": "/source/IXMtest_A01_s1_w4.tif",
        },
        source_metadata_by_path={
            "A01_s001_w1_z001_t001.tif": {"Plate": "20585", "well": "A01", "site": "1"},
            "A01_s001_w2_z001_t001.tif": {"Plate": "20585", "well": "A01", "site": "1"},
            "A01_s001_w4_z001_t001.tif": {"Plate": "20585", "well": "A01", "site": "1"},
        },
    )
    match_plan = SourceBindingMatchPlan(
        method=SourceBindingMatchMethod.METADATA,
        dimensions=(
            SourceBindingMatchDimension(
                fields=(
                    SourceBindingMatchField(alias="OrigHoechst", metadata_field="Plate"),
                    SourceBindingMatchField(alias="OrigPh_golgi", metadata_field="Plate"),
                )
            ),
            SourceBindingMatchDimension(
                fields=(
                    SourceBindingMatchField(alias="OrigHoechst", metadata_field="well"),
                    SourceBindingMatchField(alias="OrigPh_golgi", metadata_field="well"),
                )
            ),
            SourceBindingMatchDimension(
                fields=(
                    SourceBindingMatchField(alias="OrigHoechst", metadata_field="site"),
                    SourceBindingMatchField(alias="OrigPh_golgi", metadata_field="site"),
                )
            ),
        ),
    )

    assert SourceBindingMatchedImageSet.from_plan(
        bindings=(hoechst, ph_golgi),
        match_plan=match_plan,
        source_context=source_context,
    ).expand(
        (
            "A01_s001_w1_z001_t001.tif",
            "A01_s001_w2_z001_t001.tif",
            "A01_s001_w4_z001_t001.tif",
        ),
        source_universe=(
            "A01_s001_w1_z001_t001.tif",
            "A01_s001_w2_z001_t001.tif",
            "A01_s001_w4_z001_t001.tif",
        ),
    ) == (
        "A01_s001_w2_z001_t001.tif",
        "A01_s001_w4_z001_t001.tif",
    )


def test_source_bound_anchor_filter_uses_metadata_rules_for_mapped_source_paths() -> None:
    binding = NamedSourceBinding(
        alias="origMemb",
        selector=SourceSelector(
            metadata=(MetadataSelector("ChannelNumber", "0"),),
        ),
        origin=SourceBindingOrigin.PIPELINE_START,
    )
    metadata_rules = (
        MetadataExtractionRule(
            source=MetadataSource.FILE_NAME,
            pattern=r"^(?P<Plate>.*)_xy(?P<Site>[0-9])_ch(?P<ChannelNumber>[0-9])",
        ),
    )
    plan = CompiledSourceBindingPlan(bindings=(binding,),
        metadata_rules=metadata_rules,
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )
    source_context = SourcePatternResolutionContext(
        parser=SourceSchemaFilenameParser(),
        source_paths_by_virtual_path={
            "A01_s001_w1_z001_t001.tif": "/source/3d_monolayer_xy1_ch2.tif",
            "A01_s001_w2_z001_t001.tif": "/source/3d_monolayer_xy1_ch1.tif",
            "A01_s001_w3_z001_t001.tif": "/source/3d_monolayer_xy1_ch0.tif",
        },
        source_metadata_by_path={
            "A01_s001_w1_z001_t001.tif": {"channel": "1", "well": "A01"},
            "A01_s001_w2_z001_t001.tif": {"channel": "2", "well": "A01"},
            "A01_s001_w3_z001_t001.tif": {"channel": "3", "well": "A01"},
        },
        metadata_rules=metadata_rules,
    )

    assert SourceBindingCandidateMatcher.compatible_candidates(
        (
            "A01_s001_w1_z001_t001.tif",
            "A01_s001_w2_z001_t001.tif",
            "A01_s001_w3_z001_t001.tif",
        ),
        bindings=(binding,),
        source_context=source_context,
    ) == ("A01_s001_w3_z001_t001.tif",)
    assert SourceBoundAnchorPatternPolicy.for_plan(plan).select(
        ["A01_s001_w{iii}_z001_t001.tif"],
        bindings=plan.bindings,
        source_context=source_context,
    ) == ["A01_s001_w{iii}_z001_t001.tif"]


def test_source_bound_anchor_metadata_rules_do_not_override_workspace_metadata() -> None:
    metadata_rules = (
        MetadataExtractionRule(
            source=MetadataSource.FOLDER_NAME,
            pattern=r"(?P<Folder>.*)$",
        ),
    )
    source_context = SourcePatternResolutionContext(
        parser=SourceSchemaFilenameParser(),
        source_paths_by_virtual_path={
            "A01_s001_w1_z001_t001.tif": "/source/Channel 1-01-A-01.tif",
        },
        source_metadata_by_path={
            "A01_s001_w1_z001_t001.tif": {
                "Folder": "/source",
                "well": "A01",
            },
        },
        metadata_rules=metadata_rules,
    )

    assert source_context.candidate_metadata(
        "A01_s001_w1_z001_t001.tif"
    ).first_required("A01_s001_w1_z001_t001.tif")["Folder"] == "/source"


def test_runtime_source_binding_file_filters_use_mapped_source_paths() -> None:
    binding = NamedSourceBinding(
        alias="rawGFP",
        selector=SourceSelector(
            filters=(
                SourceFilterClause(
                    subject=SourceFilterSubject.FILE,
                    match_type=SourceFilterMatchType.CONTAINS,
                    value="w1",
                ),
            ),
        ),
        origin=SourceBindingOrigin.PIPELINE_START,
    )
    source_context = SourcePatternResolutionContext(
        parser=SourceSchemaFilenameParser(),
        source_paths_by_virtual_path={
            "A01_s001_w1_z001_t001.tif": "/source/BBBC013_A01_s1_w2.tif",
            "A01_s001_w2_z001_t001.tif": "/source/BBBC013_A01_s1_w1.tif",
            "A12_s001_w1_z001_t001.tif": "/source/BBBC013_A12_s1_w2.tif",
            "A12_s001_w2_z001_t001.tif": "/source/BBBC013_A12_s1_w1.tif",
        },
        source_metadata_by_path={
            "A01_s001_w1_z001_t001.tif": {"channel": "1", "channel_label": "rawDNA"},
            "A01_s001_w2_z001_t001.tif": {"channel": "2", "channel_label": "rawGFP"},
            "A12_s001_w1_z001_t001.tif": {"channel": "1", "channel_label": "rawDNA"},
            "A12_s001_w2_z001_t001.tif": {"channel": "2", "channel_label": "rawGFP"},
        },
    )

    assert SourceBindingCandidateMatcher.compatible_candidates(
        (
            "A01_s001_w1_z001_t001.tif",
            "A01_s001_w2_z001_t001.tif",
            "A12_s001_w1_z001_t001.tif",
            "A12_s001_w2_z001_t001.tif",
        ),
        bindings=(binding,),
        source_context=source_context,
    ) == (
        "A01_s001_w2_z001_t001.tif",
        "A12_s001_w2_z001_t001.tif",
    )


def test_metadata_matched_source_workspace_defers_when_template_matches_no_alias() -> None:
    binding = NamedSourceBinding(
        alias="OrigDNA",
        selector=SourceSelector(
            metadata=(MetadataSelector("ChannelNumber", "2"),),
        ),
        origin=SourceBindingOrigin.PIPELINE_START,
    )
    plan = CompiledSourceBindingPlan(bindings=(binding,),
        match_plan=SourceBindingMatchPlan(
            method=SourceBindingMatchMethod.METADATA,
            dimensions=(
                SourceBindingMatchDimension(
                    fields=(
                        SourceBindingMatchField(
                            alias="OrigDNA",
                            metadata_field="Plate",
                        ),
                    )
                ),
            ),
        ),
    )
    source_context = SourcePatternResolutionContext(
        parser=SourceSchemaFilenameParser(),
        source_paths_by_virtual_path={
            "A01_s001_w1_z001_t001.tif": "/source/3d_monolayer_xy1_ch2.tif",
        },
        source_metadata_by_path={
            "A01_s001_w1_z001_t001.tif": {"Plate": "3d_monolayer"},
        },
    )

    filtered = SourceBoundAnchorPatternPolicy.for_plan(plan).select(
        ["A01_s001_w{iii}_z001_t001.tif"],
        bindings=plan.bindings,
        source_context=source_context,
    )

    assert filtered == ["A01_s001_w{iii}_z001_t001.tif"]


def test_metadata_matched_source_workspace_ignores_other_step_aliases() -> None:
    binding = NamedSourceBinding(
        alias="OrigDNA",
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
    )
    plan = CompiledSourceBindingPlan(bindings=(binding,),
        match_plan=SourceBindingMatchPlan(
            method=SourceBindingMatchMethod.METADATA,
            dimensions=(
                SourceBindingMatchDimension(
                    fields=(
                        SourceBindingMatchField(
                            alias="OrigDNA",
                            metadata_field="Well",
                        ),
                        SourceBindingMatchField(
                            alias="OrigER",
                            metadata_field="Well",
                        ),
                    )
                ),
            ),
        ),
    )
    source_context = SourcePatternResolutionContext(
        parser=SourceSchemaFilenameParser(),
        source_paths_by_virtual_path={
            "A01_s001_w1_z001_t001.tif": "/source/OrigDNA_A01.tif",
        },
        source_metadata_by_path={
            "A01_s001_w1_z001_t001.tif": {"Well": "A01"},
        },
    )

    filtered = SourceBoundAnchorPatternPolicy.for_plan(plan).select(
        ["A01_s001_w1_z001_t001.tif"],
        bindings=plan.bindings,
        source_context=source_context,
    )

    assert filtered == ["A01_s001_w1_z001_t001.tif"]


def test_order_matched_source_artifact_bindings_do_not_add_execution_anchors() -> None:
    bindings = (
        NamedSourceBinding(
            alias="A",
            artifact_kind=ObjectLabelsArtifactType,
            selector=SourceSelector(
                filters=(
                    SourceFilterClause(
                        SourceFilterSubject.FILE,
                        SourceFilterMatchType.CONTAINS,
                        "A",
                    ),
                ),
            ),
            origin=SourceBindingOrigin.PIPELINE_START,
        ),
        NamedSourceBinding(
            alias="B",
            artifact_kind=ObjectLabelsArtifactType,
            selector=SourceSelector(
                filters=(
                    SourceFilterClause(
                        SourceFilterSubject.FILE,
                        SourceFilterMatchType.CONTAINS,
                        "B",
                    ),
                ),
            ),
            origin=SourceBindingOrigin.PIPELINE_START,
        ),
    )
    plan = CompiledSourceBindingPlan(bindings=bindings,
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )

    filtered = SourceBoundAnchorPatternPolicy.for_plan(plan).select(
        ["A01_s001_w1_z001_t001_A.png"],
        bindings=plan.bindings,
        source_context=_source_pattern_context(),
    )

    assert filtered == ["A01_s001_w1_z001_t001_A.png"]


def test_order_matched_source_anchor_filter_ignores_non_stack_image_operands() -> None:
    bindings = (
        NamedSourceBinding(
            alias="origDNA",
            selector=SourceSelector(
                metadata=(MetadataSelector("ChannelNumber", "2"),),
            ),
            origin=SourceBindingOrigin.PIPELINE_START,
        ),
        NamedSourceBinding(
            alias="origMemb",
            selector=SourceSelector(
                metadata=(MetadataSelector("ChannelNumber", "0"),),
            ),
            origin=SourceBindingOrigin.PIPELINE_START,
            participates_in_image_stack=False,
        ),
        NamedSourceBinding(
            alias="origMito",
            selector=SourceSelector(
                metadata=(MetadataSelector("ChannelNumber", "1"),),
            ),
            origin=SourceBindingOrigin.PIPELINE_START,
            participates_in_image_stack=False,
        ),
    )
    plan = CompiledSourceBindingPlan(bindings=bindings,
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )
    source_context = SourcePatternResolutionContext(
        parser=SourceSchemaFilenameParser(),
        source_paths_by_virtual_path={},
        source_metadata_by_path={
            "A01_s001_w1_z001_t001.tif": {"ChannelNumber": "0"},
            "A01_s001_w2_z001_t001.tif": {"ChannelNumber": "1"},
            "A01_s001_w3_z001_t001.tif": {"ChannelNumber": "2"},
        },
    )

    filtered = SourceBoundAnchorPatternPolicy.for_plan(plan).select(
        [
            "A01_s001_w1_z001_t001.tif",
            "A01_s001_w2_z001_t001.tif",
            "A01_s001_w3_z001_t001.tif",
        ],
        bindings=plan.bindings,
        source_context=source_context,
    )

    assert filtered == ["A01_s001_w3_z001_t001.tif"]


def test_expand_A01_schema_workspace_wells_preserves_disambiguating_suffixes(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "openhcs_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "subdirectories": {
                    ".": {
                        "workspace_mapping": {
                            "A01_s001_w1_z001_t001.tif": "Sequence1/image.tif",
                            "A01_s001_w1_z001_t001_002.tif": "Sequence2/image.tif",
                        },
                        "source_metadata": {
                            "A01_s001_w1_z001_t001.tif": {"sequence": "1"},
                            "A01_s001_w1_z001_t001_002.tif": {"sequence": "2"},
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    expand_A01_schema_workspace_wells(metadata_path, ("W001", "W002"))

    metadata = json.loads(metadata_path.read_text())
    mapping = metadata["subdirectories"]["."]["workspace_mapping"]
    assert mapping == {
        "W001_s001_w1_z001_t001.tif": "Sequence1/image.tif",
        "W001_s001_w1_z001_t001_002.tif": "Sequence2/image.tif",
        "W002_s001_w1_z001_t001.tif": "Sequence1/image.tif",
        "W002_s001_w1_z001_t001_002.tif": "Sequence2/image.tif",
    }


def test_expand_A01_schema_workspace_wells_preserves_original_well_dimension(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "openhcs_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "subdirectories": {
                    ".": {
                        "workspace_mapping": {
                            "Sequence1_s001_w1_z001_t001.tif": "Sequence1/frame0.tif",
                            "Sequence2_s001_w1_z001_t001.tif": "Sequence2/frame0.tif",
                        },
                        "source_metadata": {},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    expand_A01_schema_workspace_wells(metadata_path, ("W001",))

    metadata = json.loads(metadata_path.read_text())
    mapping = metadata["subdirectories"]["."]["workspace_mapping"]
    assert mapping == {
        "W001_s001_w1_z001_t001.tif": "Sequence1/frame0.tif",
        "W001_s002_w1_z001_t001.tif": "Sequence2/frame0.tif",
    }


def test_expand_A01_schema_workspace_wells_replaces_source_well_metadata(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "openhcs_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "subdirectories": {
                    ".": {
                        "workspace_mapping": {
                            "A01_s001_w1_z001_t001.tif": "Sequence1/image.tif",
                        },
                        "source_metadata": {
                            "A01_s001_w1_z001_t001.tif": {
                                "Well": "A01",
                                "Metadata_Well": "A01",
                                "ChannelNumber": "1",
                            },
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    expand_A01_schema_workspace_wells(metadata_path, ("W001", "W002"))

    metadata = json.loads(metadata_path.read_text())
    source_metadata = metadata["subdirectories"]["."]["source_metadata"]
    assert source_metadata == {
        "W001_s001_w1_z001_t001.tif": {
            "well": "W001",
            "site": "1",
            "channel": "1",
            "z_index": "1",
            "timepoint": "1",
            "extension": ".tif",
            ORIGINAL_SOURCE_METADATA_FIELD: {"ChannelNumber": "1"},
        },
        "W002_s001_w1_z001_t001.tif": {
            "well": "W002",
            "site": "1",
            "channel": "1",
            "z_index": "1",
            "timepoint": "1",
            "extension": ".tif",
            ORIGINAL_SOURCE_METADATA_FIELD: {"ChannelNumber": "1"},
        },
    }


def test_materialize_A01_schema_workspace_projects_cellprofiler_sources(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "cellprofiler_source"
    source_root.mkdir()
    _write_image(source_root / "Channel1-01-A-01.tif", value=1)
    _write_image(source_root / "Channel2-01-A-01.tif", value=2)
    (source_root / "Channel2ILLUM.mat").write_bytes(b"mat payload")
    workspace_root = tmp_path / "openhcs_workspace"

    result = materialize_A01_schema_workspace(
        source_root,
        workspace_root,
        _example_sbs_A01_schema(),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]
    auxiliary = metadata["subdirectories"][SOURCE_SCHEMA_WORKSPACE_SOURCE_DIR]

    assert result.workspace_root == workspace_root
    assert set(primary["workspace_mapping"]) == {
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w2_z001_t001.tif",
    }
    assert primary["channels"] == {"1": "rawGFP", "2": "rawDNA"}
    assert primary["wells"] == {"A01": None}
    assert primary["sites"] == {"1": None}
    assert primary["source_filename_parser_name"] == "SourceSchemaFilenameParser"
    assert primary["available_backends"]["virtual_workspace"] is True
    assert set(auxiliary["workspace_mapping"]) == {
        f"{SOURCE_SCHEMA_WORKSPACE_SOURCE_DIR}/IllumGFP/001_Channel2ILLUM.mat"
    }
    assert not (source_root / "openhcs_metadata.json").exists()
    assert result.primary_wells() == ("A01",)
    source_universe = {
        path.name
        for path in result.source_paths_for_primary_wells(result.primary_wells())
    }
    assert source_universe == {
        "Channel1-01-A-01.tif",
        "Channel2-01-A-01.tif",
        "Channel2ILLUM.mat",
    }


def test_materialize_A01_schema_workspace_selects_sample_before_projection(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "cellprofiler_source"
    source_root.mkdir()
    _write_image(source_root / "Channel1-01-A-01.tif", value=1)
    _write_image(source_root / "Channel2-01-A-01.tif", value=2)
    _write_image(source_root / "Channel1-01-B-01.tif", value=3)
    _write_image(source_root / "Channel2-01-B-01.tif", value=4)

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "openhcs_workspace",
        _example_sbs_A01_schema(),
        image_set_selection=SourceSchemaImageSetSelection(well_filter=("B01",)),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]

    assert set(primary["workspace_mapping"]) == {
        "B01_s001_w1_z001_t001.tif",
        "B01_s001_w2_z001_t001.tif",
    }
    assert primary["wells"] == {"B01": None}
    assert set(result.primary_mappings) == set(primary["workspace_mapping"])


def test_A01_schema_image_set_selection_limits_image_sets() -> None:
    schema = PipelineImageSchema()
    image_sets = (
        ImageSetRecord(0, {}, {"ImageNumber": "1"}),
        ImageSetRecord(1, {}, {"ImageNumber": "2"}),
        ImageSetRecord(2, {}, {"ImageNumber": "3"}),
    )

    selected = SourceSchemaImageSetSelection(max_image_set_count=1).apply(
        schema,
        image_sets,
    )

    assert selected == image_sets[:1]


def test_materialize_A01_schema_workspace_derives_well_match_field(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "cellprofiler_vitra_source"
    source_root.mkdir()
    _write_image(source_root / "Channel 1-01-A-01-00.tif", value=1)
    _write_image(source_root / "Channel 2-01-A-01-00.tif", value=2)
    workspace_root = tmp_path / "openhcs_workspace"

    result = materialize_A01_schema_workspace(
        source_root,
        workspace_root,
        _well_row_column_match_A01_schema(),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]

    assert set(primary["workspace_mapping"]) == {
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w2_z001_t001.tif",
    }
    assert primary["wells"] == {"A01": None}


def test_materialize_A01_schema_workspace_applies_images_rule(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "Channel1-A01.tif", value=1)
    _write_image(source_root / "Channel2-A01.tif", value=2)
    _write_image(source_root / "Channel1-B01.tif", value=3)
    _write_image(source_root / "Channel2-B01.tif", value=4)

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        _filtered_A01_schema(),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]

    assert set(primary["workspace_mapping"]) == {
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w2_z001_t001.tif",
    }
    assert primary["wells"] == {"A01": None}


def test_materialize_A01_schema_workspace_uses_single_default_well_for_ordered_sets(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "field-001.tif", value=1)
    _write_image(source_root / "field-002.tif", value=2)

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            assignments_by_alias={
                "OrigGreen": ImageAssignment(
                    alias="OrigGreen",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.CONTAINS,
                                "field-",
                            ),
                        )
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]

    assert set(primary["workspace_mapping"]) == {
        "A01_s001_w1_z001_t001.tif",
        "A01_s002_w1_z001_t001.tif",
    }
    assert primary["wells"] == {"A01": None}
    assert primary["sites"] == {"1": None, "2": None}
    for path in primary["workspace_mapping"]:
        assert (
            primary["source_metadata"][path][SOURCE_IMAGE_TYPE_METADATA_FIELD]
            == "Grayscale image"
        )
        assert primary["source_metadata"][path]["well"] == "A01"
        assert primary["source_metadata"][path]["channel"] == "1"
    assert primary["source_metadata"]["A01_s001_w1_z001_t001.tif"]["site"] == "1"
    assert primary["source_metadata"]["A01_s002_w1_z001_t001.tif"]["site"] == "2"


def test_order_matched_source_workspace_applies_bounded_selection_before_incomplete_alias_rejection(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "cho01.png", value=1)
    _write_image(source_root / "cho02.png", value=2)
    _write_image(source_root / "cho01_Probabilities.tiff", value=3)

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            assignments_by_alias={
                "phase": ImageAssignment(
                    alias="phase",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.EXTENSION,
                                SourceFilterMatchType.EQUALS,
                                ".png",
                            ),
                        ),
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
                "cho": ImageAssignment(
                    alias="cho",
                    image_type="Color image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.EXTENSION,
                                SourceFilterMatchType.IS_TIF,
                            ),
                        ),
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
        image_set_selection=SourceSchemaImageSetSelection(max_image_set_count=1),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]

    assert primary["workspace_mapping"] == {
        "A01_s001_w1_z001_t001.png": "../source/cho01.png",
        "A01_s001_w2_z001_t001.tiff": "../source/cho01_Probabilities.tiff",
    }
    assert primary["channels"] == {"1": "phase", "2": "cho"}


def test_materialize_A01_schema_workspace_projects_frame_metadata_to_timepoints(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "Embryo_GFPHistone_0000.tif", value=1)
    _write_image(source_root / "Embryo_GFPHistone_0001.tif", value=2)

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            metadata_rules=(
                MetadataExtractionRule(
                    source=MetadataSource.FILE_NAME,
                    pattern=r"^(?P<Specimen>.*)_(?P<Stain>.*)_(?P<FrameNumber>[0-9]*)",
                ),
            ),
            assignments_by_alias={
                "OrigColor": ImageAssignment(
                    alias="OrigColor",
                    image_type="Color image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.CONTAINS,
                                "GFPHistone",
                            ),
                        )
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
    )

    primary = json.loads(result.metadata_path.read_text())["subdirectories"]["."]

    assert set(primary["workspace_mapping"]) == {
        "A01_s001_w1_z001_t000.tif",
        "A01_s001_w1_z001_t001.tif",
    }
    assert primary["sites"] == {"1": None}
    assert primary["timepoints"] == {"0": None, "1": None}


def test_materialize_A01_schema_workspace_projects_z_metadata_to_z_axis(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "nuclei_z001.tif", value=1)
    _write_image(source_root / "nuclei_z002.tif", value=2)

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            metadata_rules=(
                MetadataExtractionRule(
                    source=MetadataSource.FILE_NAME,
                    pattern=r"^nuclei_z(?P<ZIndex>[0-9]+)",
                ),
            ),
            assignments_by_alias={
                "Nuclei": ImageAssignment(
                    alias="Nuclei",
                    image_type="Grayscale image",
                    selector=SourceSelector(),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
    )

    primary = json.loads(result.metadata_path.read_text())["subdirectories"]["."]

    assert set(primary["workspace_mapping"]) == {
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w1_z002_t001.tif",
    }
    assert primary["sites"] == {"1": None}
    assert primary["z_indexes"] == {"1": None, "2": None}


def test_materialize_A01_schema_workspace_projects_tiff_stack_pages_to_z_axis(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_tiff_stack(source_root / "nuclei_stack.tif", (10, 20, 30))

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            assignments_by_alias={
                "Nuclei": ImageAssignment(
                    alias="Nuclei",
                    image_type="Grayscale image",
                    selector=SourceSelector(),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
    )

    primary = json.loads(result.metadata_path.read_text())["subdirectories"]["."]

    assert set(primary["workspace_mapping"]) == {
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w1_z002_t001.tif",
        "A01_s001_w1_z003_t001.tif",
    }
    assert primary["sites"] == {"1": None}
    assert primary["z_indexes"] == {"1": None, "2": None, "3": None}
    assert primary["workspace_mapping"]["A01_s001_w1_z002_t001.tif"] == {
        "backend": "disk",
        "source_path": "../source/nuclei_stack.tif",
        "plane_index": 1,
        "z": 2,
    }
    assert (
        "source_plane_group_key"
        not in primary["source_metadata"]["A01_s001_w1_z002_t001.tif"]
    )


def test_source_schema_image_set_selection_preserves_tiff_stack_planes(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_tiff_stack(source_root / "stack_ch1.tif", (10, 20, 30))
    _write_tiff_stack(source_root / "stack_ch2.tif", (40, 50, 60))

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            assignments_by_alias={
                "DNA": ImageAssignment(
                    alias="DNA",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.CONTAINS,
                                "ch1",
                            ),
                        ),
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
                "Actin": ImageAssignment(
                    alias="Actin",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.CONTAINS,
                                "ch2",
                            ),
                        ),
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
        image_set_selection=SourceSchemaImageSetSelection(max_image_set_count=1),
    )

    primary = json.loads(result.metadata_path.read_text())["subdirectories"]["."]

    assert set(primary["workspace_mapping"]) == {
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w1_z002_t001.tif",
        "A01_s001_w1_z003_t001.tif",
        "A01_s001_w2_z001_t001.tif",
        "A01_s001_w2_z002_t001.tif",
        "A01_s001_w2_z003_t001.tif",
    }
    assert primary["sites"] == {"1": None}
    assert primary["z_indexes"] == {"1": None, "2": None, "3": None}
    assert primary["workspace_mapping"]["A01_s001_w2_z003_t001.tif"] == {
        "backend": "disk",
        "source_path": "../source/stack_ch2.tif",
        "plane_index": 2,
        "z": 3,
    }


def test_virtual_workspace_structured_disk_ref_loads_tiff_plane(
    tmp_path: Path,
) -> None:
    from polystore.virtual_workspace import VirtualWorkspaceBackend

    plate_root = tmp_path / "workspace"
    source_root = tmp_path / "source"
    plate_root.mkdir()
    source_root.mkdir()
    _write_tiff_stack(source_root / "stack.tif", (10, 20, 30))
    (plate_root / "openhcs_metadata.json").write_text(
        json.dumps(
            {
                "subdirectories": {
                    ".": {
                        "workspace_mapping": {
                            "A01_s001_w1_z002_t001.tif": {
                                "backend": "disk",
                                "source_path": "../source/stack.tif",
                                "plane_index": 1,
                                "z": 2,
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = VirtualWorkspaceBackend(plate_root).load(
        plate_root / "A01_s001_w1_z002_t001.tif"
    )

    assert loaded.shape == (4, 4)
    assert np.all(loaded == 20)


def test_expand_A01_schema_workspace_wells_preserves_structured_tiff_refs(
    tmp_path: Path,
) -> None:
    from polystore.virtual_workspace import VirtualWorkspaceBackend

    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_tiff_stack(source_root / "nuclei_stack.tif", (10, 20, 30))

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            assignments_by_alias={
                "Nuclei": ImageAssignment(
                    alias="Nuclei",
                    image_type="Grayscale image",
                    selector=SourceSelector(),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
    )

    expand_A01_schema_workspace_wells(result.metadata_path, ("B01",))
    primary = json.loads(result.metadata_path.read_text())["subdirectories"]["."]
    source_ref = primary["workspace_mapping"]["B01_s001_w1_z002_t001.tif"]

    assert source_ref == {
        "backend": "disk",
        "source_path": "../source/nuclei_stack.tif",
        "plane_index": 1,
        "z": 2,
    }
    loaded = VirtualWorkspaceBackend(result.workspace_root).load(
        result.workspace_root / "B01_s001_w1_z002_t001.tif"
    )
    assert loaded.shape == (4, 4)
    assert np.all(loaded == 20)


def test_materialize_A01_schema_workspace_rejects_unsupported_tiff_axes(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    tifffile.imwrite(
        source_root / "channel_stack.tif",
        np.stack(
            [np.full((4, 4), value, dtype=np.uint16) for value in (10, 20)],
            axis=0,
        ),
        metadata={"axes": "CYX"},
    )

    with pytest.raises(ValueError, match="supports only YX/YXS images and Z-first Z-stacks"):
        materialize_A01_schema_workspace(
            source_root,
            tmp_path / "workspace",
            PipelineImageSchema(
                assignments_by_alias={
                    "Nuclei": ImageAssignment(
                        alias="Nuclei",
                        image_type="Grayscale image",
                        selector=SourceSelector(),
                        origin=SourceBindingOrigin.PIPELINE_START,
                    ),
                },
                match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
            ),
        )


def test_materialize_A01_schema_workspace_rejects_unreadable_tiff_axes(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "broken.tif").write_bytes(b"not a tiff")

    with pytest.raises(ValueError, match="Could not inspect TIFF source-plane axes"):
        materialize_A01_schema_workspace(
            source_root,
            tmp_path / "workspace",
            PipelineImageSchema(
                assignments_by_alias={
                    "Nuclei": ImageAssignment(
                        alias="Nuclei",
                        image_type="Grayscale image",
                        selector=SourceSelector(),
                        origin=SourceBindingOrigin.PIPELINE_START,
                    ),
                },
                match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
            ),
        )


def test_materialize_A01_schema_workspace_uses_complete_ordered_image_sets(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "field-001.png", value=1)
    _write_image(source_root / "field-002.png", value=2)
    _write_image(source_root / "probabilities-001.tiff", value=3)
    _write_image(source_root / "probabilities-002.tiff", value=4)

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            assignments_by_alias={
                "phase": ImageAssignment(
                    alias="phase",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.EXTENSION,
                                SourceFilterMatchType.EQUALS,
                                ".png",
                            ),
                        )
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
                "probability": ImageAssignment(
                    alias="probability",
                    image_type="Color image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.EXTENSION,
                                SourceFilterMatchType.EQUALS,
                                ".tiff",
                            ),
                        )
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
    )

    metadata = json.loads(result.metadata_path.read_text())
    mapping = metadata["subdirectories"]["."]["workspace_mapping"]

    assert set(mapping) == {
        "A01_s001_w1_z001_t001.png",
        "A01_s001_w2_z001_t001.tiff",
        "A01_s002_w1_z001_t001.png",
        "A01_s002_w2_z001_t001.tiff",
    }
    assert mapping["A01_s001_w2_z001_t001.tiff"].endswith(
        "source/probabilities-001.tiff"
    )
    assert mapping["A01_s002_w2_z001_t001.tiff"].endswith(
        "source/probabilities-002.tiff"
    )


def test_materialize_A01_schema_workspace_allows_multichannel_image_set_metadata(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "A01_s001_w1_z001_t001.tif", value=1)
    _write_image(source_root / "A01_s001_w2_z001_t001.tif", value=2)

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            assignments_by_alias={
                "rawDNA": ImageAssignment(
                    alias="rawDNA",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.CONTAINS,
                                "w1",
                            ),
                        )
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
                "rawGFP": ImageAssignment(
                    alias="rawGFP",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.CONTAINS,
                                "w2",
                            ),
                        )
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
    )

    primary = json.loads(result.metadata_path.read_text())["subdirectories"]["."]

    assert primary["channels"] == {"1": "rawDNA", "2": "rawGFP"}
    assert primary["source_metadata"]["A01_s001_w1_z001_t001.tif"]["channel"] == "1"
    assert primary["source_metadata"]["A01_s001_w2_z001_t001.tif"]["channel"] == "2"


def test_materialize_A01_schema_workspace_projects_ordered_channel_site_sets(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "imaging_flow_source"
    source_root.mkdir()
    for channel in ("Ch1", "Ch6", "Ch7"):
        _write_image(source_root / f"{channel}_1.tif", value=1)
        _write_image(source_root / f"{channel}_2.tif", value=2)

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            assignments_by_alias={
                "BF_image": ImageAssignment(
                    alias="BF_image",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.CONTAINS,
                                "Ch1",
                            ),
                        )
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
                "DF_image": ImageAssignment(
                    alias="DF_image",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.CONTAINS,
                                "Ch6",
                            ),
                        )
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
                "Marker_image": ImageAssignment(
                    alias="Marker_image",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.CONTAINS,
                                "Ch7",
                            ),
                        )
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
    )

    primary = json.loads(result.metadata_path.read_text())["subdirectories"]["."]

    assert set(primary["workspace_mapping"]) == {
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w2_z001_t001.tif",
        "A01_s001_w3_z001_t001.tif",
        "A01_s002_w1_z001_t001.tif",
        "A01_s002_w2_z001_t001.tif",
        "A01_s002_w3_z001_t001.tif",
    }
    assert primary["wells"] == {"A01": None}
    assert primary["sites"] == {"1": None, "2": None}


def test_materialize_A01_schema_workspace_disambiguates_duplicate_site_metadata(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "Plate_A01_s1_w1_GUID1.tif", value=1)
    _write_image(source_root / "Plate_A01_s1_w1_GUID2.tif", value=2)

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            metadata_rules=(
                MetadataExtractionRule(
                    source=MetadataSource.FILE_NAME,
                    pattern=r"^Plate_(?P<Well>[A-Z][0-9]{2})_s(?P<Site>[0-9]+)_w(?P<ChannelNumber>[0-9]+)",
                ),
            ),
            assignments_by_alias={
                "DNA": ImageAssignment(
                    alias="DNA",
                    image_type="Grayscale image",
                    selector=SourceSelector(),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]

    assert set(primary["workspace_mapping"]) == {
        "A01_s001_w1_z001_t001.tif",
        "A01_s002_w1_z001_t001.tif",
    }
    assert primary["source_metadata"]["A01_s002_w1_z001_t001.tif"]["site"] == "2"
    assert "Site" not in primary["source_metadata"]["A01_s002_w1_z001_t001.tif"]


def test_materialize_A01_schema_workspace_matches_numeric_component_values(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "Sample_ch00.tif", value=1)

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            metadata_rules=(
                MetadataExtractionRule(
                    source=MetadataSource.FILE_NAME,
                    pattern=r"^Sample_ch(?P<ChannelNumber>[0-9]+)",
                ),
            ),
            assignments_by_alias={
                "typeI": ImageAssignment(
                    alias="typeI",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        components=(ComponentSelector(AllComponents.CHANNEL, "0"),),
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]

    assert primary["workspace_mapping"]["A01_s001_w1_z001_t001.tif"].endswith(
        "source/Sample_ch00.tif"
    )


def test_materialize_A01_schema_workspace_preserves_raw_metadata_selectors(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "Plate_xy1_ch2.tif", value=1)
    _write_image(source_root / "Plate_xy1_ch1.tif", value=2)
    _write_image(source_root / "Plate_xy1_ch0.tif", value=3)

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            metadata_rules=(
                MetadataExtractionRule(
                    source=MetadataSource.FILE_NAME,
                    pattern=r"^Plate_xy(?P<Site>[0-9]+)_ch(?P<ChannelNumber>[0-9]+)",
                ),
            ),
            assignments_by_alias={
                "origDNA": ImageAssignment(
                    alias="origDNA",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        metadata=(MetadataSelector("ChannelNumber", "2"),),
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
                "origMito": ImageAssignment(
                    alias="origMito",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        metadata=(MetadataSelector("ChannelNumber", "1"),),
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
                "origMemb": ImageAssignment(
                    alias="origMemb",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        metadata=(MetadataSelector("ChannelNumber", "0"),),
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    primary = metadata["subdirectories"]["."]
    dna_metadata = primary["source_metadata"]["A01_s001_w1_z001_t001.tif"]

    assert dna_metadata["channel"] == "1"
    assert dna_metadata[ORIGINAL_SOURCE_METADATA_FIELD]["ChannelNumber"] == "2"
    assert source_metadata_value(dna_metadata, "ChannelNumber") == "2"
    assert source_component_metadata_values(dna_metadata, AllComponents.CHANNEL) == (
        "1",
    )


def test_materialize_A01_schema_workspace_keeps_default_input_folder_files(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "Sample_D.TIF", value=1)

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            images_rule=ImagesRule(
                filters=(
                    SourceFilterClause(
                        subject=SourceFilterSubject.DIRECTORY,
                        match_type=SourceFilterMatchType.DOES_NOT_START_WITH,
                        value=".",
                    ),
                ),
            ),
            assignments_by_alias={
                "OrigBlue": ImageAssignment(
                    alias="OrigBlue",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                subject=SourceFilterSubject.FILE,
                                match_type=SourceFilterMatchType.CONTAINS,
                                value="D.TIF",
                            ),
                        )
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]
    assert set(primary["workspace_mapping"]) == {"A01_s001_w1_z001_t001.TIF"}


def test_materialize_A01_schema_workspace_uses_filemanager_for_vfs_operations(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "Sample_ch00.tif", value=1)
    workspace_root = tmp_path / "workspace"

    class RecordingFileManager:
        def __init__(self) -> None:
            self.listed: list[tuple[str, str]] = []
            self.saved: list[tuple[str, str]] = []
            self.saved_payloads: list[object] = []

        def list_files(self, directory: str, backend: str, **kwargs: object) -> list[str]:
            self.listed.append((directory, backend))
            assert kwargs["recursive"] is True
            return [str(source_root / "Sample_ch00.tif")]

        def exists(self, path: str, backend: str) -> bool:
            return Path(path).exists()

        def is_dir(self, path: str, backend: str) -> bool:
            return Path(path).is_dir()

        def ensure_directory(self, directory: str, backend: str) -> str:
            Path(directory).mkdir(parents=True, exist_ok=True)
            return directory

        def save(self, data: object, output_path: str, backend: str) -> None:
            self.saved.append((output_path, backend))
            self.saved_payloads.append(data)
            Path(output_path).write_text(json.dumps(data), encoding="utf-8")

    filemanager = RecordingFileManager()
    result = materialize_A01_schema_workspace(
        source_root,
        workspace_root,
        PipelineImageSchema(
            metadata_rules=(
                MetadataExtractionRule(
                    source=MetadataSource.FILE_NAME,
                    pattern=r"^Sample_ch(?P<ChannelNumber>[0-9]+)",
                ),
            ),
            assignments_by_alias={
                "typeI": ImageAssignment(
                    alias="typeI",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        components=(ComponentSelector(AllComponents.CHANNEL, "00"),),
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
        filemanager=filemanager,
    )

    assert filemanager.listed == [(str(source_root), "disk")]
    assert filemanager.saved == [(str(result.metadata_path), "disk")]
    assert isinstance(filemanager.saved_payloads[0], dict)


def test_materialize_A01_schema_workspace_applies_source_filters_relative_to_root(
    tmp_path: Path,
) -> None:
    hidden_parent = tmp_path / ".cache" / "dataset"
    hidden_parent.mkdir(parents=True)
    _write_image(hidden_parent / "Sample_ch00.tif", value=1)

    result = materialize_A01_schema_workspace(
        hidden_parent,
        tmp_path / "workspace",
        PipelineImageSchema(
            images_rule=ImagesRule(
                filters=(
                    SourceFilterClause(
                        subject=SourceFilterSubject.EXTENSION,
                        match_type=SourceFilterMatchType.IS_IMAGE,
                    ),
                    SourceFilterClause(
                        subject=SourceFilterSubject.DIRECTORY,
                        match_type=SourceFilterMatchType.DOES_NOT_CONTAIN_REGEX,
                        value=r"[\\/]\.",
                    ),
                ),
            ),
            metadata_rules=(
                MetadataExtractionRule(
                    source=MetadataSource.FILE_NAME,
                    pattern=r"^Sample_ch(?P<ChannelNumber>[0-9]+)",
                ),
            ),
            assignments_by_alias={
                "typeI": ImageAssignment(
                    alias="typeI",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        components=(ComponentSelector(AllComponents.CHANNEL, "00"),),
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]

    assert primary["workspace_mapping"]["A01_s001_w1_z001_t001.tif"].endswith(
        ".cache/dataset/Sample_ch00.tif"
    )


def test_materialize_A01_schema_workspace_includes_embedded_image_planes(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    external_root = tmp_path / "external"
    external_root.mkdir()
    _write_image(source_root / "local_D.TIF", value=1)
    _write_image(source_root / "local_F.TIF", value=2)
    _write_image(external_root / "url_D.TIF", value=3)
    _write_image(external_root / "url_F.TIF", value=4)

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            image_plane_sources=(
                ImagePlaneSource(uri=(external_root / "url_D.TIF").as_uri()),
                ImagePlaneSource(uri=(external_root / "url_F.TIF").as_uri()),
            ),
            assignments_by_alias={
                "OrigBlue": ImageAssignment(
                    alias="OrigBlue",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.CONTAINS,
                                "D.TIF",
                            ),
                        )
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
                "OrigGreen": ImageAssignment(
                    alias="OrigGreen",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.CONTAINS,
                                "F.TIF",
                            ),
                        )
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]

    assert set(primary["workspace_mapping"]) == {
        "A01_s001_w1_z001_t001.TIF",
        "A01_s001_w2_z001_t001.TIF",
        "A01_s002_w1_z001_t001.TIF",
        "A01_s002_w2_z001_t001.TIF",
    }
    assert primary["sites"] == {"1": None, "2": None}
    assert primary["workspace_mapping"]["A01_s002_w1_z001_t001.TIF"].endswith(
        "external/url_D.TIF"
    )


def test_materialize_A01_schema_workspace_resolves_embedded_urls_to_local_sources(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "url_D.TIF", value=1)
    _write_image(source_root / "url_F.TIF", value=2)

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            image_plane_sources=(
                ImagePlaneSource(uri="https://example.invalid/data/url_D.TIF"),
                ImagePlaneSource(uri="https://example.invalid/data/url_F.TIF"),
            ),
            assignments_by_alias={
                "OrigBlue": ImageAssignment(
                    alias="OrigBlue",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.CONTAINS,
                                "D.TIF",
                            ),
                        )
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
                "OrigGreen": ImageAssignment(
                    alias="OrigGreen",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.CONTAINS,
                                "F.TIF",
                            ),
                        )
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]

    assert set(primary["workspace_mapping"]) == {
        "A01_s001_w1_z001_t001.TIF",
        "A01_s001_w2_z001_t001.TIF",
    }
    assert primary["workspace_mapping"]["A01_s001_w1_z001_t001.TIF"].endswith(
        "source/url_D.TIF"
    )


def test_materialize_A01_schema_workspace_projects_groups_to_well_axis(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    (source_root / "Sequence1").mkdir(parents=True)
    (source_root / "Sequence2").mkdir()
    _write_image(source_root / "Sequence1" / "Embryo_GFP_0000.tif", value=1)
    _write_image(source_root / "Sequence1" / "Embryo_GFP_0001.tif", value=2)
    _write_image(source_root / "Sequence2" / "Embryo_GFP_0000.tif", value=3)
    _write_image(source_root / "Sequence2" / "Embryo_GFP_0001.tif", value=4)

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        _grouped_order_A01_schema(),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]

    assert set(primary["workspace_mapping"]) == {
        "Sequence1_s001_w1_z001_t000.tif",
        "Sequence1_s001_w1_z001_t001.tif",
        "Sequence2_s001_w1_z001_t000.tif",
        "Sequence2_s001_w1_z001_t001.tif",
    }
    assert primary["wells"] == {"Sequence1": None, "Sequence2": None}
    assert primary["sites"] == {"1": None}
    assert primary["timepoints"] == {"0": None, "1": None}
    assert (
        primary["source_metadata"]["Sequence1_s001_w1_z001_t000.tif"]["timepoint"]
        == "0"
    )
    assert (
        primary["source_metadata"]["Sequence1_s001_w1_z001_t000.tif"]["Run"]
        == "Sequence1"
    )


def test_materialize_A01_schema_workspace_does_not_infer_axes_from_source_names(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "AS_09047_050428030001_O01f00d2.TIF", value=1)
    _write_image(source_root / "AS_09047_050428030001_O01f01d2.TIF", value=2)
    _write_image(source_root / "AS_09047_050428030001_O02f00d2.TIF", value=3)
    _write_image(source_root / "AS_09047_050428030001_O02f01d2.TIF", value=4)

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            assignments_by_alias={
                "OrigGreen": ImageAssignment(
                    alias="OrigGreen",
                    image_type="Color image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.CONTAINS,
                                "AS_09047_",
                            ),
                        )
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        ),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]

    assert primary["wells"] == {"A01": None}
    assert primary["sites"] == {"1": None, "2": None, "3": None, "4": None}
    assert set(primary["workspace_mapping"]) == {
        "A01_s001_w1_z001_t001.TIF",
        "A01_s002_w1_z001_t001.TIF",
        "A01_s003_w1_z001_t001.TIF",
        "A01_s004_w1_z001_t001.TIF",
    }


def test_materialize_A01_schema_workspace_joins_imported_metadata(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "Channel1-A-01.tif", value=1)
    _write_image(source_root / "Channel2-A-01.tif", value=2)
    (source_root / "metadata.csv").write_text(
        "Row,Compound\nA,DMSO\n",
        encoding="utf-8",
    )

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        _imported_metadata_A01_schema(),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]

    assert set(primary["workspace_mapping"]) == {
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w2_z001_t001.tif",
    }
    assert primary["wells"] == {"A01": None}
    assert primary["source_metadata"]["A01_s001_w1_z001_t001.tif"]["Compound"] == (
        "DMSO"
    )
    assert result.source_metadata["A01_s001_w2_z001_t001.tif"]["Compound"] == "DMSO"


def test_materialize_A01_schema_workspace_resolves_stale_imported_metadata_location(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source" / "images"
    source_root.mkdir(parents=True)
    _write_image(source_root / "Channel1-A-01.tif", value=1)
    _write_image(source_root / "Channel2-A-01.tif", value=2)
    (source_root.parent / "metadata.csv").write_text(
        "Row,Compound\nA,DMSO\n",
        encoding="utf-8",
    )

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        _imported_metadata_A01_schema_with_location(
            "/old/default/input/folder/metadata.csv"
        ),
    )

    assert result.source_metadata["A01_s001_w1_z001_t001.tif"]["Compound"] == "DMSO"


def test_materialize_A01_schema_workspace_resolves_imported_metadata_through_visible_symlink(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / ".cache" / "dataset" / "images"
    source_root.mkdir(parents=True)
    _write_image(source_root / "Channel1-A-01.tif", value=1)
    _write_image(source_root / "Channel2-A-01.tif", value=2)
    (source_root.parent / "metadata.csv").write_text(
        "Row,Compound\nA,DMSO\n",
        encoding="utf-8",
    )
    visible_root = tmp_path / "visible" / "images_alias"
    visible_root.parent.mkdir()
    visible_root.symlink_to(source_root, target_is_directory=True)

    result = materialize_A01_schema_workspace(
        visible_root,
        tmp_path / "workspace",
        _imported_metadata_A01_schema_with_location(
            "/old/default/input/folder/metadata.csv"
        ),
    )

    assert result.source_metadata["A01_s001_w1_z001_t001.tif"]["Compound"] == "DMSO"


def test_materialize_A01_schema_workspace_skips_imported_metadata_partial_join(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "Channel1-A-01.tif", value=1)
    np.save(source_root / "IllumChannel1.npy", np.ones((2, 2), dtype=np.uint8))
    (source_root / "metadata.csv").write_text(
        "WellRow,Plate,Compound\nA,Illum,DMSO\n",
        encoding="utf-8",
    )

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            metadata_rules=(
                MetadataExtractionRule(
                    source=MetadataSource.FILE_NAME,
                    pattern=r"^(?P<Plate>Illum).*",
                ),
            ),
            imported_metadata_tables=(
                ImportedMetadataTable(
                    location="metadata.csv",
                    joins=(
                        ImportedMetadataJoin("WellRow", "WellRow"),
                        ImportedMetadataJoin("Plate", "Plate"),
                    ),
                ),
            ),
            assignments_by_alias={
                "raw": ImageAssignment(
                    alias="raw",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.CONTAINS,
                                "Channel1",
                            ),
                        ),
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
            source_artifacts_by_alias={
                    "Illum": SourceArtifactAssignment(
                        alias="Illum",
                        artifact_kind=ImageArtifactType,
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.CONTAINS,
                                "IllumChannel1",
                            ),
                        ),
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
        ),
    )

    metadata = result.source_metadata[
        f"{SOURCE_SCHEMA_WORKSPACE_SOURCE_DIR}/Illum/001_IllumChannel1.npy"
    ]
    assert metadata["Plate"] == "Illum"
    assert "Compound" not in metadata


def test_source_paths_for_primary_wells_includes_imported_metadata_tables(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "Channel1-A-01.tif", value=1)
    (source_root / "metadata.csv").write_text("WellRow\nA\n", encoding="utf-8")

    metadata_table = ImportedMetadataTable(
        location="metadata.csv",
        joins=(ImportedMetadataJoin("WellRow", "WellRow"),),
    )
    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            imported_metadata_tables=(metadata_table,),
            assignments_by_alias={
                "raw": ImageAssignment(
                    alias="raw",
                    image_type="Grayscale image",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.CONTAINS,
                                "Channel1",
                            ),
                        ),
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
        ),
    )

    source_universe = {
        path.name
        for path in result.source_paths_for_primary_wells(
            result.primary_wells(),
            imported_metadata_tables=(metadata_table,),
        )
    }

    assert source_universe == {"Channel1-A-01.tif", "metadata.csv"}


def test_materialize_A01_schema_workspace_merges_duplicate_imported_metadata_consensus(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "Channel1-A-01.tif", value=1)
    _write_image(source_root / "Channel2-A-01.tif", value=2)
    (source_root / "metadata.csv").write_text(
        "Row,Compound,Replicate\nA,DMSO,1\nA,DMSO,2\n",
        encoding="utf-8",
    )

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        _imported_metadata_A01_schema(),
    )

    metadata = result.source_metadata["A01_s001_w1_z001_t001.tif"]
    assert metadata["Compound"] == "DMSO"
    assert "Replicate" not in metadata


def test_materialize_A01_schema_workspace_supports_source_artifact_only_schema(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "A.png", value=1)
    _write_image(source_root / "B.png", value=2)

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            source_artifacts_by_alias={
                "A": SourceArtifactAssignment(
                    alias="A",
                    artifact_kind=ObjectLabelsArtifactType,
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.CONTAINS,
                                "A",
                            ),
                        ),
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                    payload_type="Objects",
                ),
                "B": SourceArtifactAssignment(
                    alias="B",
                    artifact_kind=ObjectLabelsArtifactType,
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.CONTAINS,
                                "B",
                            ),
                        ),
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                    payload_type="Objects",
                ),
            },
        ),
    )

    metadata = json.loads(result.metadata_path.read_text())
    primary = metadata["subdirectories"]["."]
    auxiliary = metadata["subdirectories"][SOURCE_SCHEMA_WORKSPACE_SOURCE_DIR]
    assert primary["workspace_mapping"] == {
        "A01_s001_w1_z001_t001.png": "../source/A.png"
    }
    assert primary["wells"] == {"A01": None}
    assert primary["channels"] == {"1": "A"}
    assert set(auxiliary["workspace_mapping"]) == {
        f"{SOURCE_SCHEMA_WORKSPACE_SOURCE_DIR}/A/001_A.png",
        f"{SOURCE_SCHEMA_WORKSPACE_SOURCE_DIR}/B/001_B.png",
    }
    assert result.source_metadata[
        f"{SOURCE_SCHEMA_WORKSPACE_SOURCE_DIR}/A/001_A.png"
    ][SOURCE_IMAGE_TYPE_METADATA_FIELD] == "Objects"


def test_materialize_A01_schema_workspace_keeps_extracted_metadata_on_import_conflict(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "plate_file-A-01.tif", value=1)
    (source_root / "metadata.csv").write_text(
        "Row,Plate,Compound\nA,plate_csv,DMSO\n",
        encoding="utf-8",
    )

    result = materialize_A01_schema_workspace(
        source_root,
        tmp_path / "workspace",
        PipelineImageSchema(
            metadata_rules=(
                MetadataExtractionRule(
                    source=MetadataSource.FILE_NAME,
                    pattern=r"^(?P<Plate>plate_file)-(?P<WellRow>[A-Z])-(?P<WellColumn>[0-9]{2})",
                ),
            ),
            imported_metadata_tables=(
                ImportedMetadataTable(
                    location="metadata.csv",
                    joins=(ImportedMetadataJoin("WellRow", "Row"),),
                ),
            ),
            assignments_by_alias={
                "raw": ImageAssignment(
                    alias="raw",
                    image_type="Grayscale image",
                    selector=SourceSelector(),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            },
        ),
    )

    metadata = result.source_metadata["A01_s001_w1_z001_t001.tif"]
    assert metadata["Plate"] == "plate_file"
    assert metadata["Compound"] == "DMSO"


def _example_sbs_A01_schema() -> PipelineImageSchema:
    metadata_rule = MetadataExtractionRule(
        source=MetadataSource.FILE_NAME,
        pattern=r".*-(?P<ImageNumber>\d*)-(?P<WellRow>.*)-(?P<WellColumn>\d*)",
    )
    return PipelineImageSchema(
        metadata_rules=(metadata_rule,),
        assignments_by_alias={
            "rawGFP": ImageAssignment(
                alias="rawGFP",
                image_type="Grayscale image",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "Channel1-",
                        ),
                    )
                ),
                origin=SourceBindingOrigin.STEP_INPUT,
            ),
            "rawDNA": ImageAssignment(
                alias="rawDNA",
                image_type="Grayscale image",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "Channel2-",
                        ),
                    )
                ),
                origin=SourceBindingOrigin.STEP_INPUT,
            ),
            "IllumGFP": ImageAssignment(
                alias="IllumGFP",
                image_type="Illumination function",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.ENDS_WITH,
                            ".mat",
                        ),
                    )
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
        },
        match_plan=SourceBindingMatchPlan(
            method=SourceBindingMatchMethod.METADATA,
            dimensions=(
                SourceBindingMatchDimension(
                    fields=(
                        SourceBindingMatchField("rawGFP", "WellRow"),
                        SourceBindingMatchField("rawDNA", "WellRow"),
                    )
                ),
                SourceBindingMatchDimension(
                    fields=(
                        SourceBindingMatchField("rawGFP", "WellColumn"),
                        SourceBindingMatchField("rawDNA", "WellColumn"),
                    )
                ),
            ),
        ),
    )


def _filtered_A01_schema() -> PipelineImageSchema:
    return PipelineImageSchema(
        images_rule=ImagesRule(
            filters=(
                SourceFilterClause(
                    SourceFilterSubject.FILE,
                    SourceFilterMatchType.CONTAINS,
                    "A01",
                ),
            ),
        ),
        assignments_by_alias={
            "rawGFP": ImageAssignment(
                alias="rawGFP",
                image_type="Grayscale image",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "Channel1",
                        ),
                    )
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
            "rawDNA": ImageAssignment(
                alias="rawDNA",
                image_type="Grayscale image",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "Channel2",
                        ),
                    )
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
        },
    )


def _imported_metadata_A01_schema() -> PipelineImageSchema:
    return _imported_metadata_A01_schema_with_location("metadata.csv")


def _imported_metadata_A01_schema_with_location(
    location: str,
) -> PipelineImageSchema:
    return PipelineImageSchema(
        metadata_rules=(
            MetadataExtractionRule(
                source=MetadataSource.FILE_NAME,
                pattern=r"^Channel(?P<ChannelNumber>[0-9])-(?P<WellRow>[A-Z])-(?P<WellColumn>[0-9]{2})",
            ),
        ),
        imported_metadata_tables=(
            ImportedMetadataTable(
                location=location,
                joins=(
                    ImportedMetadataJoin(
                        image_metadata_field="WellRow",
                        imported_metadata_field="Row",
                    ),
                ),
            ),
        ),
        assignments_by_alias={
            "rawGFP": ImageAssignment(
                alias="rawGFP",
                image_type="Grayscale image",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "Channel1",
                        ),
                    ),
                    metadata=(MetadataSelector("Compound", "DMSO"),),
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
            "rawDNA": ImageAssignment(
                alias="rawDNA",
                image_type="Grayscale image",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "Channel2",
                        ),
                    ),
                    metadata=(MetadataSelector("Compound", "DMSO"),),
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
        },
        match_plan=SourceBindingMatchPlan(
            method=SourceBindingMatchMethod.METADATA,
            dimensions=(
                SourceBindingMatchDimension(
                    fields=(
                        SourceBindingMatchField("rawGFP", "Compound"),
                        SourceBindingMatchField("rawDNA", "Compound"),
                    )
                ),
            ),
        ),
    )


def _grouped_order_A01_schema() -> PipelineImageSchema:
    return PipelineImageSchema(
        metadata_rules=(
            MetadataExtractionRule(
                source=MetadataSource.FILE_NAME,
                pattern=r"^(?P<Specimen>.*)_(?P<Stain>.*)_(?P<FrameNumber>[0-9]*)",
            ),
            MetadataExtractionRule(
                source=MetadataSource.FOLDER_NAME,
                pattern=r"(?P<Run>.*)$",
            ),
        ),
        assignments_by_alias={
            "OrigColor": ImageAssignment(
                alias="OrigColor",
                image_type="Grayscale image",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "GFP",
                        ),
                    )
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
        },
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        grouping=GroupingPlan(metadata_fields=("Run",)),
    )


def _well_row_column_match_A01_schema() -> PipelineImageSchema:
    metadata_rule = MetadataExtractionRule(
        source=MetadataSource.FILE_NAME,
        pattern=(
            r"^Channel (?P<ChannelNumber>[0-9])-[0-9]{2}-"
            r"(?P<WellRow>[A-P])-(?P<WellCol>[0-9]{2})"
        ),
    )
    return PipelineImageSchema(
        metadata_rules=(metadata_rule,),
        assignments_by_alias={
            "OrigProtein": ImageAssignment(
                alias="OrigProtein",
                image_type="Grayscale image",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "Channel 1",
                        ),
                    )
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
            "OrigDNA": ImageAssignment(
                alias="OrigDNA",
                image_type="Color image",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.CONTAINS,
                            "Channel 2",
                        ),
                    )
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
        },
        match_plan=SourceBindingMatchPlan(
            method=SourceBindingMatchMethod.METADATA,
            dimensions=(
                SourceBindingMatchDimension(
                    fields=(
                        SourceBindingMatchField("OrigProtein", "Well"),
                        SourceBindingMatchField("OrigDNA", "Well"),
                    )
                ),
            ),
        ),
    )


def _write_image(path: Path, *, value: int) -> None:
    image = np.full((8, 8), value, dtype=np.uint16)
    Image.fromarray(image).save(path)
