"""Exactness contracts for CellProfiler primary-object performance kernels."""

import numpy as np
import pytest

from openhcs.core.config import DtypeConfig
from openhcs.processing.backends.cellprofiler import morphology as morphology_backend
from openhcs.processing.backends.cellprofiler import thresholding as thresholding_backend
from openhcs.processing.backends.cellprofiler.primary_objects import (
    _remap_object_label_variant_after_final_relabel,
    identify_primary_objects,
)
from openhcs.processing.backends.cellprofiler.thresholding import (
    NumbaNumpyThresholdDiagnosticsBackendStrategy,
    NumpyThresholdDiagnosticsBackendStrategy,
    ThresholdApplicationSmoothing,
)
from openhcs.processing.backends.cellprofiler.thresholding_threshold_numba_otsu import (
    _finite_flat_float32,
    _finite_flat_float64,
)


def _exhaustive_local_maxima_by_label(
    image: np.ndarray,
    labels: np.ndarray,
    footprint: np.ndarray,
) -> np.ndarray:
    offsets = (
        np.argwhere(footprint)
        - np.asarray(footprint.shape, dtype=np.int64) // 2
    )
    maxima = np.zeros(image.shape, dtype=bool)
    for y, x in np.argwhere(labels > 0):
        label = labels[y, x]
        max_value = -np.inf
        for offset_y, offset_x in offsets:
            neighbor_y = y + offset_y
            neighbor_x = x + offset_x
            if not (
                0 <= neighbor_y < image.shape[0]
                and 0 <= neighbor_x < image.shape[1]
            ):
                continue
            if labels[neighbor_y, neighbor_x] != label:
                continue
            value = image[neighbor_y, neighbor_x]
            if value > max_value:
                max_value = value
        maxima[y, x] = image[y, x] == max_value
    return maxima


@pytest.mark.parametrize("image_dtype", (np.float32, np.float64))
@pytest.mark.parametrize("include_center", (False, True))
def test_numba_local_maxima_exact_rejection_matches_exhaustive_semantics(
    image_dtype: type[np.floating],
    include_center: bool,
) -> None:
    rng = np.random.default_rng(31)
    backend = morphology_backend.NumbaNumpyMorphologyBackendStrategy()
    for _ in range(40):
        image = rng.standard_normal((7, 9)).astype(image_dtype)
        labels = rng.choice(
            np.array([0, 0, 0, 7, 7, 203], dtype=np.int32),
            size=image.shape,
        )
        footprint = rng.random((5, 7)) > 0.55
        footprint[2, 3] = include_center
        image[0, 0] = np.nan
        image[1, 2] = np.inf
        image[5, 7] = -np.inf

        expected = _exhaustive_local_maxima_by_label(image, labels, footprint)
        actual = backend.local_maxima_by_label(image, labels, footprint)

        np.testing.assert_array_equal(actual, expected)


def test_numba_local_maxima_preserves_inputs_and_orders_nearest_offsets_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_storage = np.arange(90, dtype=np.float32).reshape((9, 10))
    label_storage = np.ones((9, 10), dtype=np.int32)
    image = image_storage[:, ::-1]
    labels = label_storage[:, ::-1]
    footprint = np.ones((7, 9), dtype=bool)
    captured: dict[str, np.ndarray] = {}

    def capture_kernel(
        image_array: np.ndarray,
        labels_array: np.ndarray,
        offset_y: np.ndarray,
        offset_x: np.ndarray,
    ) -> np.ndarray:
        captured.update(
            image=image_array,
            labels=labels_array,
            offset_y=offset_y,
            offset_x=offset_x,
        )
        return np.zeros(image_array.shape, dtype=bool)

    monkeypatch.setattr(
        morphology_backend,
        "_local_maxima_by_label_numba",
        capture_kernel,
    )

    morphology_backend.NumbaNumpyMorphologyBackendStrategy().local_maxima_by_label(
        image,
        labels,
        footprint,
    )

    assert captured["image"].dtype == image.dtype
    assert captured["labels"].dtype == labels.dtype
    assert captured["image"].flags.c_contiguous
    assert captured["labels"].flags.c_contiguous
    distances = captured["offset_y"] ** 2 + captured["offset_x"] ** 2
    assert np.all(distances[:-1] <= distances[1:])


@pytest.mark.parametrize("use_partial_mask", (False, True))
def test_planar_threshold_diagnostics_match_numpy_without_entropy_fallback(
    monkeypatch: pytest.MonkeyPatch,
    use_partial_mask: bool,
) -> None:
    rng = np.random.default_rng(7)
    image = rng.random((384, 512), dtype=np.float32)
    binary = image > np.float32(0.45)
    mask = None
    reference_mask = np.ones(image.shape, dtype=bool)
    if use_partial_mask:
        reference_mask = rng.random(image.shape) > 0.2
        mask = reference_mask

    reference = NumpyThresholdDiagnosticsBackendStrategy()
    expected = (
        reference.weighted_variance(image, reference_mask, binary),
        reference.sum_of_entropies(image, reference_mask, binary),
    )

    def reject_numpy_entropy_fallback(*_args: object, **_kwargs: object) -> float:
        raise AssertionError("planar diagnostics must use the combined backend kernel")

    monkeypatch.setattr(
        thresholding_backend,
        "_numpy_threshold_sum_of_entropies",
        reject_numpy_entropy_fallback,
    )
    actual = NumbaNumpyThresholdDiagnosticsBackendStrategy().diagnostics(
        image,
        mask,
        binary,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize(
    ("dtype", "finite_flat"),
    (
        (np.float32, _finite_flat_float32),
        (np.float64, _finite_flat_float64),
    ),
)
def test_finite_flat_threshold_values_retain_contiguous_storage(
    dtype: type[np.floating],
    finite_flat,
) -> None:
    values = np.linspace(0.0, 1.0, 64, dtype=dtype).reshape((8, 8))

    flattened = finite_flat(values)

    assert flattened.flags.c_contiguous
    assert np.shares_memory(flattened, values)
    np.testing.assert_array_equal(flattened, values.ravel())


@pytest.mark.parametrize(
    ("dtype", "finite_flat"),
    (
        (np.float32, _finite_flat_float32),
        (np.float64, _finite_flat_float64),
    ),
)
def test_finite_flat_threshold_values_filter_nonfinite_inputs(
    dtype: type[np.floating],
    finite_flat,
) -> None:
    values = np.array([0.0, np.nan, 0.5, np.inf, 1.0], dtype=dtype)

    flattened = finite_flat(values)

    np.testing.assert_array_equal(flattened, np.array([0.0, 0.5, 1.0], dtype=dtype))


def test_primary_object_variant_relabel_reuses_identical_accepted_domain() -> None:
    accepted = np.array([[0, 4, 4], [9, 9, 0]], dtype=np.int32)
    final = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.int32)

    remapped = _remap_object_label_variant_after_final_relabel(
        accepted,
        accepted,
        final,
        object_count=2,
    )

    assert remapped is final


def test_primary_object_border_filter_preserves_unedited_variant() -> None:
    image = np.zeros((7, 7), dtype=np.float32)
    image[0, :2] = 1.0
    image[3:5, 3:5] = 1.0

    _image, _measurements, labels = identify_primary_objects(
        image,
        min_diameter=1,
        max_diameter=20,
        exclude_size=False,
        exclude_border_objects=True,
        unclump_method="None",
        watershed_method="None",
        fill_holes="Never",
        threshold_method="Manual",
        threshold_smoothing_scale=0.0,
        manual_threshold=0.5,
        dtype_config=DtypeConfig(),
    )

    assert labels.labels[0, 0] == 0
    assert labels.unedited_labels is not None
    assert labels.unedited_labels[0, 0] > int(labels.labels.max())


def test_primary_object_relabel_preserves_integer_input_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = np.array([[0, 3, 3], [7, 7, 0]], dtype=np.int32)
    captured: dict[str, np.dtype] = {}

    def fake_relabel(values: np.ndarray) -> tuple[np.ndarray, int]:
        captured["dtype"] = values.dtype
        return np.array([[0, 1, 1], [2, 2, 0]], dtype=np.int32), 2

    monkeypatch.setattr(
        morphology_backend,
        "_relabel_sequential_numba",
        fake_relabel,
    )
    relabeled, count = (
        morphology_backend.NumbaNumpyMorphologyBackendStrategy().relabel_sequential(
            labels
        )
    )

    assert captured["dtype"] == np.dtype(np.int32)
    assert count == 2
    np.testing.assert_array_equal(
        relabeled,
        np.array([[0, 1, 1], [2, 2, 0]], dtype=np.int32),
    )


@pytest.mark.parametrize("shape", ((17, 19), (3, 9, 11)))
@pytest.mark.parametrize("use_partial_mask", (False, True))
def test_threshold_application_smoothing_promotes_at_filter_output_exactly(
    shape: tuple[int, ...],
    use_partial_mask: bool,
) -> None:
    from scipy import ndimage as ndi

    rng = np.random.default_rng(23)
    image = rng.random(shape, dtype=np.float32)
    mask = rng.random(shape) > 0.2 if use_partial_mask else None
    policy = ThresholdApplicationSmoothing(1.3488)

    image64 = image.astype(np.float64)
    masked_image = image64 if mask is None else np.where(mask, image64, 0.0)
    expected_numerator = ndi.gaussian_filter(
        masked_image,
        sigma=policy.sigma,
        mode="constant",
        cval=0,
        truncate=4.0,
    )
    weight_source = np.ones(shape, dtype=np.float64) if mask is None else mask.astype(float)
    weight = ndi.gaussian_filter(
        weight_source,
        sigma=policy.sigma,
        mode="constant",
        cval=0,
        truncate=4.0,
    )
    denominator = weight + np.finfo(float).eps
    if mask is None:
        expected = expected_numerator / denominator
    else:
        expected = np.zeros(shape, dtype=np.float64)
        valid = weight != 0
        expected[valid] = expected_numerator[valid] / denominator[valid]

    actual, sigma = policy.smooth(image, mask)

    assert sigma == pytest.approx(1.0)
    np.testing.assert_array_equal(actual, expected)


def test_threshold_application_smoothing_retains_float32_input_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.linspace(0.0, 1.0, 63, dtype=np.float32).reshape((7, 9))
    filtered_inputs: list[np.ndarray] = []

    def record_filter(
        _policy: ThresholdApplicationSmoothing,
        values: np.ndarray,
    ) -> np.ndarray:
        filtered_inputs.append(values)
        return np.asarray(values, dtype=np.float64)

    monkeypatch.setattr(ThresholdApplicationSmoothing, "gaussian_filter", record_filter)

    ThresholdApplicationSmoothing(1.3488).smooth(image, None)

    assert len(filtered_inputs) == 1
    assert filtered_inputs[0].dtype == np.dtype(np.float32)
    assert np.shares_memory(filtered_inputs[0], image)
