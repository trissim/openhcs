from __future__ import annotations

from openhcs.formats.experimental_layout_rows import ExperimentalLayoutRowRole


def test_experimental_layout_row_role_classifies_replicate_count_rows() -> None:
    assert ExperimentalLayoutRowRole("N").is_replicate_count
    assert ExperimentalLayoutRowRole("replicates").is_replicate_count
    assert not ExperimentalLayoutRowRole("condition").is_replicate_count


def test_experimental_layout_row_role_classifies_well_rows() -> None:
    assert ExperimentalLayoutRowRole("well").is_well_all_replicates
    assert ExperimentalLayoutRowRole("wells").is_well_all_replicates
    assert not ExperimentalLayoutRowRole("well1").is_well_all_replicates
    assert ExperimentalLayoutRowRole("well1").is_well_specific_replicate
    assert not ExperimentalLayoutRowRole("well").is_well_specific_replicate
