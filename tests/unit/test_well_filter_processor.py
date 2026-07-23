from openhcs.core.config import WellFilterMode
from openhcs.core.utils import WellFilterProcessor


def test_well_filter_preserves_available_identity_across_case_variants() -> None:
    available = ["r04c09", "r04c10"]

    assert WellFilterProcessor.resolve_filter_with_mode(
        "R04C09",
        WellFilterMode.INCLUDE,
        available,
    ) == ["r04c09"]
    assert WellFilterProcessor.resolve_filter_with_mode(
        ["R04C10"],
        WellFilterMode.INCLUDE,
        available,
    ) == ["r04c10"]
    assert WellFilterProcessor.resolve_filter_with_mode(
        "R04C09",
        WellFilterMode.EXCLUDE,
        available,
    ) == ["r04c10"]
