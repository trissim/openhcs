import pytest

from openhcs.constants.constants import AllComponents
from openhcs.core.source_bindings import (
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
)
from openhcs.core.source_matching import (
    source_filters_match,
    source_metadata_component,
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
