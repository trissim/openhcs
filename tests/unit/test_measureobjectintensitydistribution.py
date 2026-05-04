import numpy as np
import pytest

from benchmark.cellprofiler_library.functions import measureobjectintensitydistribution as mid
from openhcs.core.config import DtypeConfig


@pytest.fixture(autouse=True)
def clear_radial_label_geometry_cache():
    mid._RADIAL_LABEL_GEOMETRY_CACHE.clear()
    yield
    mid._RADIAL_LABEL_GEOMETRY_CACHE.clear()


def test_radial_distribution_excludes_pixels_without_propagated_center(monkeypatch):
    image = np.ones((2, 2), dtype=np.float32)
    labels = np.ones((2, 2), dtype=np.int32)

    monkeypatch.setattr(
        mid,
        "_distance_to_edge",
        lambda labels: np.zeros(labels.shape, dtype=np.float64),
    )
    monkeypatch.setattr(
        mid,
        "_find_object_centers",
        lambda labels, d_to_edge, nobjects: (
            np.array([0.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        mid,
        "_compute_distance_from_centers",
        lambda labels, centers_i, centers_j, nobjects: (
            np.full(labels.shape, -1000.0, dtype=np.float64),
            np.zeros(labels.shape, dtype=np.int32),
        ),
    )

    result, measurements = mid.measure_object_intensity_distribution(
        image,
        labels,
        bin_count=4,
        dtype_config=DtypeConfig(),
    )

    assert result is image
    assert len(measurements) == 4
    assert all(np.isnan(measurement.frac_at_d) for measurement in measurements)
    assert all(np.isnan(measurement.mean_frac) for measurement in measurements)
    assert all(measurement.radial_cv == 0.0 for measurement in measurements)


def test_radial_label_geometry_cache_reuses_equal_label_values():
    labels = np.array(
        [
            [1, 1, 0, 2],
            [1, 0, 0, 2],
            [0, 0, 2, 2],
        ],
        dtype=np.int32,
    )

    first = mid._radial_label_geometry(labels, 2)
    second = mid._radial_label_geometry(labels.copy(), 2)

    assert second is first
    assert len(mid._RADIAL_LABEL_GEOMETRY_CACHE) == 1


def test_radial_distribution_marks_missing_dense_label_fraction_fields_nan():
    image = np.ones((3, 3), dtype=np.float32)
    labels = np.array(
        [
            [1, 0, 3],
            [1, 0, 3],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )

    _result, measurements = mid.measure_object_intensity_distribution(
        image,
        labels,
        bin_count=4,
        dtype_config=DtypeConfig(),
    )

    missing_label_measurements = [
        measurement for measurement in measurements
        if measurement.object_label == 2
    ]

    assert len(missing_label_measurements) == 4
    assert all(np.isnan(measurement.frac_at_d) for measurement in missing_label_measurements)
    assert all(np.isnan(measurement.mean_frac) for measurement in missing_label_measurements)
    assert all(measurement.radial_cv == 0.0 for measurement in missing_label_measurements)


def test_radial_cv_ignores_empty_angular_wedges(monkeypatch):
    image = np.ones((3, 3), dtype=np.float32)
    labels = np.ones((3, 3), dtype=np.int32)

    monkeypatch.setattr(
        mid,
        "_distance_to_edge",
        lambda labels: np.zeros(labels.shape, dtype=np.float64),
    )
    monkeypatch.setattr(
        mid,
        "_find_object_centers",
        lambda labels, d_to_edge, nobjects: (
            np.array([1.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        mid,
        "_compute_distance_from_centers",
        lambda labels, centers_i, centers_j, nobjects: (
            np.zeros(labels.shape, dtype=np.float64),
            np.ones(labels.shape, dtype=np.int32),
        ),
    )

    _result, measurements = mid.measure_object_intensity_distribution(
        image,
        labels,
        bin_count=4,
        dtype_config=DtypeConfig(),
    )

    assert measurements[0].radial_cv == 0.0


def test_distance_from_centers_uses_label_own_center_without_propagation():
    labels = np.array(
        [
            [1, 1, 0, 2],
            [1, 0, 0, 2],
            [0, 0, 2, 2],
        ],
        dtype=np.int32,
    )

    distances, center_labels = mid._compute_distance_from_centers(
        labels,
        np.array([0.0, 2.0], dtype=np.float64),
        np.array([0.0, 2.0], dtype=np.float64),
        2,
    )

    assert center_labels.tolist() == labels.tolist()
    assert distances[0, 0] == 0.0
    assert distances[0, 1] == 1.0
    assert distances[1, 0] == 1.0
    assert distances[2, 2] == 0.0
    assert distances[2, 3] == 1.0
