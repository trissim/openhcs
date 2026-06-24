import pytest

from openhcs.constants.constants import AllComponents
from openhcs.core.source_bindings import (
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
)
from openhcs.core.source_matching import (
    ORIGINAL_SOURCE_METADATA_FIELD,
    source_component_metadata_values,
    merge_source_metadata,
    source_component_metadata_value,
    source_filters_match,
    source_metadata_component,
    source_metadata_value,
    source_metadata_values_equal,
    with_source_component_metadata,
    with_original_source_metadata,
)


@pytest.mark.parametrize(
    ("match_type", "path", "expected"),
    (
        (
            SourceFilterMatchType.EQUALS,
            "/plate/VitraChannel1ILLUM.npy",
            True,
        ),
        (
            SourceFilterMatchType.EQUALS,
            "/plate/VitraChannel2ILLUM.npy",
            False,
        ),
        (
            SourceFilterMatchType.DOES_NOT_EQUAL,
            "/plate/VitraChannel1ILLUM.npy",
            False,
        ),
        (
            SourceFilterMatchType.DOES_NOT_EQUAL,
            "/plate/VitraChannel2ILLUM.npy",
            True,
        ),
    ),
)
def test_file_source_filters_match_exact_file_names(
    match_type: SourceFilterMatchType,
    path: str,
    expected: bool,
):
    assert (
        source_filters_match(
            path,
            (
                SourceFilterClause(
                    subject=SourceFilterSubject.FILE,
                    match_type=match_type,
                    value="VitraChannel1ILLUM.npy",
                ),
            ),
        )
        is expected
    )


@pytest.mark.parametrize(
    ("field", "component"),
    (
        ("well", AllComponents.WELL),
        ("Metadata_Site", AllComponents.SITE),
        ("ChannelNumber", AllComponents.CHANNEL),
    ),
)
def test_source_metadata_component_matches_semantic_component_names(
    field: str,
    component: AllComponents,
):
    assert source_metadata_component(field) is component


def test_source_component_metadata_value_matches_alias_fields():
    metadata = {"ChannelNumber": "01", "Metadata_Site": "A"}

    assert source_component_metadata_value(metadata, AllComponents.CHANNEL) == "01"
    assert source_component_metadata_value(metadata, AllComponents.SITE) == "A"


def test_source_component_metadata_values_include_native_and_alias_fields():
    metadata = {"channel": "1", "ChannelNumber": "00"}

    assert source_component_metadata_values(metadata, AllComponents.CHANNEL) == (
        "1",
        "00",
    )


def test_original_source_metadata_preserves_literal_selectors_without_axis_pollution():
    metadata = with_original_source_metadata(
        {"channel": "1"},
        {"ChannelNumber": "2"},
        path="A01_s001_w1_z001_t001.tif",
    )

    assert source_metadata_value(metadata, "ChannelNumber") == "2"
    assert source_metadata_value(metadata, "channel") == "1"
    assert source_component_metadata_values(metadata, AllComponents.CHANNEL) == ("1",)


def test_with_source_component_metadata_replaces_alias_fields():
    metadata = {
        "Well": "A01",
        ORIGINAL_SOURCE_METADATA_FIELD: {"Well": "A01"},
        "Metadata_Well": "A01",
        "ChannelNumber": "1",
    }

    updated = with_source_component_metadata(
        metadata,
        AllComponents.WELL,
        "W001",
    )

    assert updated == {
        ORIGINAL_SOURCE_METADATA_FIELD: {"Well": "A01"},
        "ChannelNumber": "1",
        "well": "W001",
    }
    assert source_component_metadata_values(updated, AllComponents.WELL) == ("W001",)


def test_source_metadata_values_equal_normalizes_numeric_padding():
    assert source_metadata_values_equal("01", "1")
    assert source_metadata_values_equal(" 02 ", "2")
    assert not source_metadata_values_equal("ch01", "1")


def test_merge_source_metadata_accepts_equivalent_absolute_paths(
    tmp_path,
):
    root = tmp_path / "root"
    leaf = root / "plate"
    leaf.mkdir(parents=True)
    equivalent = root / "other" / ".." / "plate"
    metadata = {"Folder": str(equivalent)}

    merge_source_metadata(metadata, {"Folder": str(leaf)}, path="image.tif")

    assert metadata["Folder"] == str(leaf.resolve())


def test_merge_source_metadata_rejects_distinct_values():
    metadata = {"Well": "A01"}

    with pytest.raises(RuntimeError):
        merge_source_metadata(metadata, {"Well": "B01"}, path="image.tif")
