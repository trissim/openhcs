import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import skimage.measure

from openhcs.processing.backends.analysis import region_properties
from openhcs.processing.backends.analysis.region_properties import (
    NumbaNumpyLabelRegionPropertiesBackendStrategy,
    label_area_and_rounded_perimeter_2d,
)


def _skimage_label_area_and_rounded_perimeter(
    labels: np.ndarray,
) -> tuple[float, float]:
    label_array = np.asarray(labels)
    perimeter = 0.0
    for label_id in np.unique(label_array):
        if label_id > 0:
            perimeter += float(
                np.round(
                    skimage.measure.perimeter(
                        label_array == label_id,
                        neighborhood=4,
                    )
                )
            )
    return float(np.count_nonzero(label_array > 0)), perimeter


def _edge_case_parameters() -> tuple[object, ...]:
    single = np.zeros((5, 6), dtype=np.int32)
    single[2, 3] = 1

    multiple = np.zeros((10, 12), dtype=np.int32)
    multiple[1:4, 1:5] = 1
    multiple[6:9, 7:11] = 2

    touching = np.zeros((9, 11), dtype=np.int32)
    touching[1:8, 1:5] = 2
    touching[2:7, 5:10] = 11

    sparse_ids = np.zeros((12, 14), dtype=np.int32)
    sparse_ids[1:5, 1:6] = 3
    sparse_ids[6:11, 7:13] = 4093
    sparse_ids[8, 9] = 0

    border = np.zeros((8, 9), dtype=np.int32)
    border[:4, :3] = 7
    border[5:, 5:] = 13

    hole = np.zeros((13, 15), dtype=np.int32)
    hole[1:12, 2:14] = 4
    hole[4:9, 5:11] = 0

    storage = np.zeros((25, 31), dtype=np.int32)
    noncontiguous = storage[1::2, 2::2]
    noncontiguous[1:5, 1:6] = 5
    noncontiguous[6:10, 8:13] = 101
    assert not noncontiguous.flags.c_contiguous

    return (
        pytest.param(np.zeros((0, 0), dtype=np.int32), id="empty"),
        pytest.param(np.zeros((7, 11), dtype=np.int32), id="background"),
        pytest.param(single, id="single"),
        pytest.param(multiple, id="multiple"),
        pytest.param(touching, id="touching"),
        pytest.param(sparse_ids, id="sparse-ids"),
        pytest.param(border, id="border"),
        pytest.param(hole, id="hole"),
        pytest.param(noncontiguous, id="noncontiguous"),
    )


@pytest.mark.parametrize("labels", _edge_case_parameters())
def test_label_area_and_perimeter_matches_skimage_edge_cases(
    labels: np.ndarray,
) -> None:
    assert label_area_and_rounded_perimeter_2d(
        labels
    ) == _skimage_label_area_and_rounded_perimeter(labels)


def test_label_perimeter_rounds_each_label_before_summing() -> None:
    labels = np.zeros((5, 11), dtype=np.int32)
    labels[1, 1:3] = 3
    labels[2, 1] = 3
    labels[1, 7:9] = 11
    labels[2, 7] = 11

    raw_perimeters = tuple(
        skimage.measure.perimeter(labels == label_id, neighborhood=4)
        for label_id in (3, 11)
    )

    assert raw_perimeters == (3.414213562373095, 3.414213562373095)
    assert float(np.round(sum(raw_perimeters))) == 7.0
    assert label_area_and_rounded_perimeter_2d(labels) == (6.0, 6.0)


def test_label_area_and_perimeter_matches_seeded_randomized_skimage() -> None:
    rng = np.random.default_rng(0xC0FFEE)
    for case_index in range(256):
        height = int(rng.integers(1, 42))
        width = int(rng.integers(1, 47))
        dense_count = int(rng.integers(1, 9))
        dense = rng.integers(
            -1,
            dense_count + 1,
            size=(height, width),
            dtype=np.int32,
        )
        sparse_ids = np.zeros(dense_count + 1, dtype=np.int32)
        sparse_ids[1:] = np.sort(
            rng.choice(
                np.arange(1, 4096, dtype=np.int32),
                size=dense_count,
                replace=False,
            )
        )
        labels = np.where(
            dense > 0,
            sparse_ids[np.maximum(dense, 0)],
            dense,
        )
        if case_index % 3 == 0:
            storage = np.zeros(
                (height * 2 + 1, width * 2 + 1),
                dtype=np.int32,
            )
            storage[1::2, 1::2] = labels
            labels = storage[1::2, 1::2]
            assert not labels.flags.c_contiguous

        assert label_area_and_rounded_perimeter_2d(
            labels
        ) == _skimage_label_area_and_rounded_perimeter(labels), case_index


def test_dense_region_perimeters_reuse_exact_label_border_classification() -> None:
    labels = np.zeros((14, 17), dtype=np.int32)
    labels[1:12, 2:8] = 3
    labels[4:9, 4:6] = 0
    labels[2:13, 8:16] = 19
    properties = NumbaNumpyLabelRegionPropertiesBackendStrategy().measure_2d(labels)

    expected = np.asarray(
        [
            skimage.measure.perimeter(labels == label_id, neighborhood=4)
            for label_id in properties.label
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(properties.perimeter, expected, rtol=0.0, atol=1e-12)


def test_dense_region_numba_cache_is_reusable_across_processes(tmp_path: Path) -> None:
    source = """\
import numpy as np

from openhcs.processing.backends.analysis.region_properties import (
    NumbaNumpyLabelRegionPropertiesBackendStrategy,
)

labels = np.zeros((12, 12), dtype=np.int32)
labels[2:10, 3:9] = 1
properties = NumbaNumpyLabelRegionPropertiesBackendStrategy().measure_2d(labels)
assert properties.label.tolist() == [1]
"""
    environment = {
        **os.environ,
        "NUMBA_CACHE_DIR": str(tmp_path / "numba-cache"),
    }

    for process_index in range(2):
        completed = subprocess.run(
            (sys.executable, "-c", source),
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            env=environment,
        )
        assert completed.returncode == 0, (
            process_index,
            completed.stdout,
            completed.stderr,
        )


def test_label_perimeter_ast_deletes_repeated_border_predicate_lattice() -> None:
    source_path = Path(region_properties.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "_label_border_pixel_4" not in functions
    for function_name in (
        "_dense_label_region_properties_2d_numba",
        "_label_area_rounded_perimeter_2d_numba",
    ):
        called_names = [
            node.func.id
            for node in ast.walk(functions[function_name])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        assert called_names.count("_label_border_pixels_4") == 1
        assert called_names.count("_label_perimeter_from_border_pixels_2d") == 1

    perimeter_calls = [
        node.func.id
        for node in ast.walk(functions["_label_perimeter_from_border_pixels_2d"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert perimeter_calls.count("_perimeter_weight_for_config_numba") == 1
    assert "_label_pixel_at" not in perimeter_calls
