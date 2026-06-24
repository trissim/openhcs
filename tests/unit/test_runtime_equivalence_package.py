"""Runtime equivalence package boundary tests."""

from collections import Counter

import numpy as np

from openhcs.core import runtime_equivalence
from openhcs.core.equivalence import (
    RuntimeCellSignature,
    RuntimeCellValueKind,
    RuntimeEquivalenceDifference,
    RuntimeEquivalenceDifferenceKind,
    RuntimeEquivalencePolicy,
    RuntimeEquivalenceReport,
    RuntimeImageSnapshot,
    RuntimeMeasurementDialect,
    RuntimeMeasurementFeatureKey,
    RuntimeMeasurementFeatureNumericTolerance,
    RuntimeMeasurementSubjectKey,
    RuntimeOutputSnapshot,
    RuntimeTableSnapshot,
    finite_signature_number,
    normalize_runtime_identifier,
    normalize_runtime_source_name,
    runtime_table_differences,
    runtime_cell_signature,
    runtime_cell_signature_counters_equivalent,
)
from openhcs.core.equivalence.arrays import semantic_array_payload
from openhcs.core.equivalence.cells import measurement_table_cell_payload
from openhcs.core.runtime_semantics import MeasurementScope


def test_runtime_equivalence_report_types_have_package_owner() -> None:
    assert runtime_equivalence.RuntimeEquivalenceDifference is (
        RuntimeEquivalenceDifference
    )
    assert runtime_equivalence.RuntimeEquivalenceDifferenceKind is (
        RuntimeEquivalenceDifferenceKind
    )
    assert runtime_equivalence.RuntimeEquivalenceReport is RuntimeEquivalenceReport


def test_runtime_equivalence_policy_types_have_package_owner() -> None:
    policy = RuntimeEquivalencePolicy(measurement_dialect=RuntimeMeasurementDialect())

    assert runtime_equivalence.RuntimeEquivalencePolicy is RuntimeEquivalencePolicy
    assert policy.measurement_dialect is not None
    assert normalize_runtime_identifier("MeanIntensity_OrigBlue") == (
        "mean_intensity_orig_blue"
    )
    assert normalize_runtime_source_name("OrigBlue__CorrGray") == (
        "orig_blue__corr_gray"
    )


def test_runtime_equivalence_policy_non_negative_fields_are_annotation_driven() -> None:
    try:
        RuntimeEquivalencePolicy(image_abs_tolerance=-0.1)
    except ValueError as exc:
        assert str(exc) == (
            "RuntimeEquivalencePolicy.image_abs_tolerance cannot be negative."
        )
    else:
        raise AssertionError("negative image_abs_tolerance should fail")

    try:
        RuntimeMeasurementFeatureNumericTolerance(
            feature_names=frozenset({"area_shape"}),
            numeric_rel_tolerance=-0.1,
        )
    except ValueError as exc:
        assert str(exc) == (
            "RuntimeMeasurementFeatureNumericTolerance.numeric_rel_tolerance "
            "cannot be negative."
        )
    else:
        raise AssertionError("negative numeric_rel_tolerance should fail")


def test_runtime_equivalence_measurement_keys_have_package_owner() -> None:
    subject = RuntimeMeasurementSubjectKey(MeasurementScope.IMAGE, "OrigBlue")
    feature = RuntimeMeasurementFeatureKey(
        subject,
        "mean_intensity",
        source_name="OrigBlue__CorrGray",
    )

    assert runtime_equivalence.RuntimeMeasurementSubjectKey is (
        RuntimeMeasurementSubjectKey
    )
    assert runtime_equivalence.RuntimeMeasurementFeatureKey is (
        RuntimeMeasurementFeatureKey
    )
    assert feature.subject.name == "orig_blue"
    assert feature.source_name == "orig_blue__corr_gray"


def test_runtime_equivalence_cell_signatures_have_package_owner() -> None:
    signature = RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "1.0")

    assert runtime_equivalence.RuntimeCellSignature is RuntimeCellSignature
    assert runtime_equivalence.RuntimeCellValueKind is RuntimeCellValueKind
    assert signature.to_cache_payload() == ("number", "1.0")
    assert runtime_cell_signature("1.23456", RuntimeEquivalencePolicy()).kind is (
        RuntimeCellValueKind.NUMBER
    )
    assert finite_signature_number(signature) == 1.0


def test_runtime_equivalence_cell_signatures_canonicalize_negative_zero() -> None:
    policy = RuntimeEquivalencePolicy(numeric_decimal_places=8)

    assert runtime_cell_signature("-0.0", policy) == runtime_cell_signature(
        "0.0",
        policy,
    )
    assert runtime_cell_signature("-0.0000000001", policy) == runtime_cell_signature(
        "0.0",
        policy,
    )


def test_runtime_equivalence_cell_counter_comparison_has_package_owner() -> None:
    signature = RuntimeCellSignature(RuntimeCellValueKind.NUMBER, "1.0")

    assert runtime_cell_signature_counters_equivalent(
        Counter((signature,)),
        Counter((signature,)),
        RuntimeEquivalencePolicy(),
    )


def test_runtime_equivalence_image_snapshot_has_package_owner() -> None:
    assert runtime_equivalence.RuntimeImageSnapshot is RuntimeImageSnapshot


def test_runtime_equivalence_array_payloads_use_canonical_equivalence_surface() -> None:
    array = np.array([[1, 2], [3, 4]], dtype=np.int16)

    assert semantic_array_payload(array)[:3] == ("array", "int16", (2, 2))
    assert measurement_table_cell_payload(np.float64(1.5)) == ("float", "1.5")
    assert measurement_table_cell_payload(array)[:3] == ("array", "int16", (2, 2))


def test_runtime_equivalence_table_snapshot_has_package_owner() -> None:
    table = RuntimeTableSnapshot(
        path="measurements.csv",
        header=("ImageNumber", "MeanIntensity"),
        rows=(("1", "2.0"),),
    )

    assert runtime_equivalence.RuntimeTableSnapshot is RuntimeTableSnapshot
    assert table.schema_key == ("ImageNumber", "MeanIntensity")


def test_runtime_equivalence_output_snapshot_has_package_owner() -> None:
    snapshot = RuntimeOutputSnapshot(tables=(), images=())

    assert runtime_equivalence.RuntimeOutputSnapshot is RuntimeOutputSnapshot
    assert snapshot.tables == ()


def test_runtime_equivalence_table_comparison_has_package_owner() -> None:
    assert runtime_table_differences((), (), RuntimeEquivalencePolicy()) == ()
