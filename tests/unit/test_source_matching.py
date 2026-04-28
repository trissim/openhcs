import pytest

from openhcs.core.source_bindings import (
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
)
from openhcs.core.source_matching import source_filters_match


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
