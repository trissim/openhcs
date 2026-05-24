from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from openhcs.constants import AllComponents
from openhcs.core.artifacts import ArtifactKind
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
    StepSourceBindingsConfig,
)
from openhcs.core.source_schema_workspace import (
    ImageSetRecord,
    SOURCE_SCHEMA_WORKSPACE_SOURCE_DIR,
    SourceSchemaImageSetSelection,
    expand_source_schema_workspace_wells as expand_A01_schema_workspace_wells,
    materialize_source_schema_workspace as materialize_A01_schema_workspace,
    source_schema_metadata_with_virtual_components,
)
from openhcs.core.steps.function_execution import (
    SourceBoundAnchorPatternPolicy,
    SourceAnchorSelectionStatus,
    SourcePatternResolutionContext,
)
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser


def _source_pattern_context() -> SourcePatternResolutionContext:
    return SourcePatternResolutionContext(
        parser=SourceSchemaFilenameParser(),
        source_paths_by_virtual_path={},
    )


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
            "ChannelName": "DNA",
        },
    )

    assert metadata == {
        "OpenHCSImageType": "Grayscale image",
        "ChannelName": "DNA",
        "well": "A01",
        "site": "2",
        "channel": "3",
        "z_index": "1",
        "timepoint": "1",
        "extension": ".TIF",
    }


def test_source_binding_plan_views_are_registered_nominal_family() -> None:
    assert set(CompiledSourceBindingPlan.registered_plan_types()) == {
        StepSourceBindingsConfig,
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
        artifact_kind=ArtifactKind.IMAGE,
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

    plan = CompiledSourceBindingPlan(
        bindings_by_group={None: (binding,)},
        match_plan=None,
    )
    filtered = SourceBoundAnchorPatternPolicy.for_plan(plan).select(
        patterns,
        bindings=plan.bindings_for_group(None),
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
    plan = CompiledSourceBindingPlan(
        bindings_by_group={None: (binding,)},
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
        bindings=plan.bindings_for_group(None),
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
    plan = CompiledSourceBindingPlan(
        bindings_by_group={None: bindings},
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )

    filtered = SourceBoundAnchorPatternPolicy.for_plan(plan).select(
        patterns,
        bindings=plan.bindings_for_group(None),
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
    plan = CompiledSourceBindingPlan(
        bindings_by_group={None: bindings},
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )

    with pytest.raises(ValueError, match="incomplete image set"):
        SourceBoundAnchorPatternPolicy.for_plan(plan).select(
            ["A01_s001_w1_z001_t001.tif"],
            bindings=plan.bindings_for_group(None),
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
    plan = CompiledSourceBindingPlan(
        bindings_by_group={None: bindings},
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
        bindings=plan.bindings_for_group(None),
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
    plan = CompiledSourceBindingPlan(
        bindings_by_group={None: bindings},
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
        bindings=plan.bindings_for_group(None),
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
    plan = CompiledSourceBindingPlan(
        bindings_by_group={None: (binding,)},
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
        bindings=plan.bindings_for_group(None),
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
    plan = CompiledSourceBindingPlan(
        bindings_by_group={None: (binding,)},
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
        bindings=plan.bindings_for_group(None),
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
    plan = CompiledSourceBindingPlan(
        bindings_by_group={None: (binding,)},
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
        bindings=plan.bindings_for_group(None),
        source_context=source_context,
    )

    assert selection.status is SourceAnchorSelectionStatus.SELECTED
    assert selection.patterns == ()


def test_metadata_matched_source_workspace_defers_when_template_matches_no_alias() -> None:
    binding = NamedSourceBinding(
        alias="OrigDNA",
        selector=SourceSelector(
            metadata=(MetadataSelector("ChannelNumber", "2"),),
        ),
        origin=SourceBindingOrigin.PIPELINE_START,
    )
    plan = CompiledSourceBindingPlan(
        bindings_by_group={None: (binding,)},
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
        bindings=plan.bindings_for_group(None),
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
    plan = CompiledSourceBindingPlan(
        bindings_by_group={None: (binding,)},
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
        bindings=plan.bindings_for_group(None),
        source_context=source_context,
    )

    assert filtered == ["A01_s001_w1_z001_t001.tif"]


def test_order_matched_source_artifact_bindings_do_not_add_execution_anchors() -> None:
    bindings = (
        NamedSourceBinding(
            alias="A",
            artifact_kind=ArtifactKind.OBJECT_LABELS,
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
            artifact_kind=ArtifactKind.OBJECT_LABELS,
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
    plan = CompiledSourceBindingPlan(
        bindings_by_group={None: bindings},
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
    )

    filtered = SourceBoundAnchorPatternPolicy.for_plan(plan).select(
        ["A01_s001_w1_z001_t001_A.png"],
        bindings=plan.bindings_for_group(None),
        source_context=_source_pattern_context(),
    )

    assert filtered == ["A01_s001_w1_z001_t001_A.png"]


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
        },
        "W002_s001_w1_z001_t001.tif": {
            "well": "W002",
            "site": "1",
            "channel": "1",
            "z_index": "1",
            "timepoint": "1",
            "extension": ".tif",
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


def test_A01_schema_image_set_selection_keeps_all_sites_for_selected_sample() -> None:
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

    assert selected == image_sets


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


def test_materialize_A01_schema_workspace_uses_complete_ordered_image_sets(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write_image(source_root / "field-001.png", value=1)
    _write_image(source_root / "field-002.png", value=2)
    _write_image(source_root / "shared-probabilities.tiff", value=3)

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
    }
    assert mapping["A01_s001_w2_z001_t001.tiff"].endswith(
        "source/shared-probabilities.tiff"
    )


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


def test_materialize_A01_schema_workspace_recovers_plate_well_tokens_without_metadata_rules(
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

    assert primary["wells"] == {"O01": None, "O02": None}
    assert primary["sites"] == {"1": None, "2": None}
    assert set(primary["workspace_mapping"]) == {
        "O01_s001_w1_z001_t001.TIF",
        "O01_s002_w1_z001_t001.TIF",
        "O02_s001_w1_z001_t001.TIF",
        "O02_s002_w1_z001_t001.TIF",
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
                        artifact_kind=ArtifactKind.IMAGE,
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
                    artifact_kind=ArtifactKind.OBJECT_LABELS,
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
                    artifact_kind=ArtifactKind.OBJECT_LABELS,
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
                pattern=r".*[\\/](?P<Run>.*)$",
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
