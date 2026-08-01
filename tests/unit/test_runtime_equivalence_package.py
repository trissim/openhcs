"""Runtime equivalence package boundary tests."""

import ast
from collections import Counter
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

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
from openhcs.core.equivalence.cells import (
    measurement_table_cell_payload,
    runtime_measurement_cell_is_present,
    runtime_measurement_cell_signature,
    runtime_measurement_cell_signature_if_present,
)
from openhcs.core.equivalence.keys import RuntimeMeasurementSourcePair
from openhcs.core.runtime_measurements import MeasurementScope


PROJECT_ROOT = Path(__file__).parents[2]


def test_runtime_equivalence_imports_owned_package_in_a_cold_process() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import openhcs.core; "
                "assert 'openhcs.core.equivalence' not in sys.modules; "
                "import openhcs.core.runtime_equivalence"
            ),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_runtime_equivalence_does_not_depend_on_deep_alias_import_side_effects() -> (
    None
):
    source_path = PROJECT_ROOT / "openhcs/core/runtime_equivalence.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

    assert not [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name.startswith("openhcs.core.equivalence.")
    ]


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


def test_default_measurement_dialect_renders_backend_neutral_spatial_grid_names() -> (
    None
):
    dialect = RuntimeMeasurementDialect()

    assert dialect.spatial_grid_measurement_feature_name("Grid", "x_origin") == (
        "spatial_grid_grid_x_origin"
    )


def test_generic_equivalence_sources_do_not_own_cellprofiler_leaf_semantics() -> None:
    source_paths = (
        *(PROJECT_ROOT / "openhcs/core/equivalence").glob("*.py"),
        PROJECT_ROOT / "openhcs/core/runtime_equivalence.py",
    )
    forbidden_literals = {
        "cellprofiler",
        "defined_grid",
        "distance_centroid",
        "distance_minimum",
        "x_location_of_lowest_x_spot",
        "y_location_of_lowest_y_spot",
    }

    for path in source_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_modules = tuple(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        literals = {
            node.value.lower()
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }

        assert not any(
            module.startswith("openhcs.interop.cellprofiler")
            or module.startswith("openhcs.processing.backends.cellprofiler")
            for module in imported_modules
        ), path
        assert forbidden_literals.isdisjoint(literals), path


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


def test_runtime_measurement_source_pair_requires_canonical_separator() -> None:
    assert RuntimeMeasurementSourcePair.from_source_name("orig_blue_corr_gray") is None
    assert RuntimeMeasurementSourcePair.from_source_name(
        "orig_blue__corr_gray"
    ) == RuntimeMeasurementSourcePair("orig_blue", "corr_gray")


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


def test_runtime_measurement_scalar_cache_and_signatures_are_exact() -> None:
    policy = RuntimeEquivalencePolicy()
    cases = (
        (None, None, None, False),
        ("", ("str", ""), None, False),
        ("value", ("str", "value"), ("text", "value"), True),
        (False, ("bool", False), ("text", "False"), True),
        (7, ("int", 7), ("number", "7.0"), True),
        (1.5, ("float", "1.5"), ("number", "1.5"), True),
        (float("nan"), ("float", "nan"), None, False),
        (float("inf"), ("float", "inf"), ("number", "inf"), True),
        (float("-inf"), ("float", "-inf"), ("number", "-inf"), True),
        (np.str_(""), ("str", ""), None, False),
        (np.bool_(False), ("bool", False), ("text", "False"), True),
        (np.int64(7), ("int", 7), ("number", "7.0"), True),
        (np.float64(1.5), ("float", "1.5"), ("number", "1.5"), True),
        (np.float64("nan"), ("float", "nan"), None, False),
        (np.float64("inf"), ("float", "inf"), ("number", "inf"), True),
        (np.float64("-inf"), ("float", "-inf"), ("number", "-inf"), True),
    )

    with (
        patch(
            "openhcs.core.equivalence.cells.semantic_array_payload",
            side_effect=AssertionError("scalar payload reached array classification"),
        ),
        patch(
            "openhcs.core.equivalence.cells.semantic_array_shape",
            side_effect=AssertionError("scalar presence reached array classification"),
        ),
    ):
        for value, exact_payload, signature_payload, is_present in cases:
            assert measurement_table_cell_payload(value) == exact_payload
            present_signature = runtime_measurement_cell_signature_if_present(
                value,
                policy,
            )
            assert (
                None
                if present_signature is None
                else present_signature.to_cache_payload()
            ) == signature_payload
            assert runtime_measurement_cell_is_present(value) is is_present


def test_runtime_measurement_array_and_container_cache_and_signatures_are_exact() -> (
    None
):
    policy = RuntimeEquivalencePolicy()
    array_cases = (
        (
            np.array(7, dtype=np.uint8),
            (
                "array",
                "uint8",
                (1,),
                "ca358758f6d27e6cf45272937977a748fd88391db679ceda7dc7bf1f005ee879",
            ),
            True,
        ),
        (
            np.array([], dtype=np.uint8),
            (
                "array",
                "uint8",
                (0,),
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            False,
        ),
    )
    container_cases = (
        (
            {},
            ("mapping", ()),
            False,
        ),
        (
            {"a": 1},
            ("mapping", ((("str", "a"), ("int", 1)),)),
            True,
        ),
        ([], ("list", ()), False),
        ([1, "a"], ("list", (("int", 1), ("str", "a"))), True),
        ((), ("tuple", ()), False),
        ((1, "a"), ("tuple", (("int", 1), ("str", "a"))), True),
    )

    for value, exact_payload, is_present in (*array_cases, *container_cases):
        assert measurement_table_cell_payload(value) == exact_payload
        signature = runtime_measurement_cell_signature(value, policy)
        present_signature = runtime_measurement_cell_signature_if_present(value, policy)
        assert present_signature == (signature if is_present else None)
        assert runtime_measurement_cell_is_present(value) is is_present


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
