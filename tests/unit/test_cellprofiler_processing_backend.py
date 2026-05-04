from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from openhcs.constants.constants import MemoryType
from openhcs.core.callable_contract import PROCESSING_CONTRACT_ATTR
from openhcs.processing.backends.lib_registry.openhcs_registry import OpenHCSRegistry
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


def test_cellprofiler_processing_backend_exports_absorbed_function() -> None:
    from openhcs.processing.backends import cellprofiler

    function = cellprofiler.identify_primary_objects

    assert function.__module__ == "openhcs.processing.backends.cellprofiler"
    assert function.input_memory_type == MemoryType.NUMPY.value
    assert function.output_memory_type == MemoryType.NUMPY.value
    assert getattr(function, PROCESSING_CONTRACT_ATTR) is ProcessingContract.PURE_2D
    assert (
        function
        is cellprofiler.get_cellprofiler_function("identify_primary_objects")
    )
    assert (
        function
        is cellprofiler.require_cellprofiler_function(
            "IdentifyPrimaryObjects",
            function_name="identify_primary_objects",
        )
    )


def test_cellprofiler_processing_backend_exports_module_function_variants() -> None:
    from openhcs.processing.backends import cellprofiler

    function = cellprofiler.require_cellprofiler_function(
        "IdentifyObjectsInGrid",
        function_name="identify_objects_in_grid_with_guides",
    )

    assert function is cellprofiler.identify_objects_in_grid_with_guides
    assert function.input_memory_type == MemoryType.NUMPY.value
    assert function.output_memory_type == MemoryType.NUMPY.value
    assert getattr(function, PROCESSING_CONTRACT_ATTR) is ProcessingContract.PURE_2D


def test_cellprofiler_processing_backend_submodule_import_is_lazy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import benchmark.cellprofiler_library.functions."
                "measureobjectsizeshape as module; "
                "assert callable(module.measure_object_size_shape)"
            ),
        ],
        check=False,
    )

    assert result.returncode == 0


def test_openhcs_registry_discovers_cellprofiler_backend_contracts() -> None:
    registry = OpenHCSRegistry()
    registry.MODULES_TO_SCAN = ["openhcs.processing.backends.cellprofiler"]

    functions = registry.discover_functions()

    metadata = functions["cellprofiler_identify_primary_objects"]
    assert metadata.contract is ProcessingContract.PURE_2D
    assert metadata.func.input_memory_type == MemoryType.NUMPY.value
    assert "cellprofiler" in metadata.tags
    assert "cellprofiler_identify_objects_in_grid_with_guides" in functions


def test_openhcs_registry_cache_invalidates_when_scanned_modules_change(tmp_path) -> None:
    registry = OpenHCSRegistry()
    registry._cache_path = tmp_path / "openhcs_function_metadata.json"
    registry.MODULES_TO_SCAN = []

    assert registry.load_or_discover_functions() == {}

    registry.MODULES_TO_SCAN = ["openhcs.processing.backends.cellprofiler"]
    functions = registry.load_or_discover_functions()

    assert "cellprofiler_identify_primary_objects" in functions


def test_cellprofiler_threshold_diagnostics_backend_resolves_numpy() -> None:
    from openhcs.processing.backends.cellprofiler.thresholding import (
        CentrosomeNumpyThresholdPrimitiveBackendStrategy,
        NumbaNumpyThresholdPrimitiveBackendStrategy,
        NumbaNumpyThresholdDiagnosticsBackendStrategy,
        NumbaNumpyThresholdSmoothingBackendStrategy,
        NumpyThresholdDiagnosticsBackendStrategy,
        ThresholdDiagnosticsBackendStrategy,
        ThresholdPrimitiveBackendStrategy,
        ThresholdSmoothingBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.shape import (
        CentrosomeNumpyShapeMeasurementBackendStrategy,
        LegacyFastNumpyShapeMeasurementBackendStrategy,
        NumbaNumpyShapeMeasurementBackendStrategy,
        ShapeMeasurementBackendStrategy,
    )

    strategy = ThresholdDiagnosticsBackendStrategy.for_memory_type(MemoryType.NUMPY)
    image = np.array([[0.1, 0.2], [0.8, 0.9]], dtype=np.float32)
    mask = np.ones(image.shape, dtype=bool)
    binary = image > 0.5

    weighted_variance, sum_of_entropies = strategy.diagnostics(image, mask, binary)

    assert type(strategy) is NumbaNumpyThresholdDiagnosticsBackendStrategy
    assert weighted_variance >= 0.0
    assert sum_of_entropies <= 0.0


def test_cellprofiler_backend_selection_is_memory_provider_keyed() -> None:
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
        cellprofiler_backend_key,
    )
    from openhcs.processing.backends.cellprofiler.morphology import (
        CentrosomeNumpyMorphologyBackendStrategy,
        MorphologyBackendStrategy,
        NumbaNumpyMorphologyBackendStrategy,
        NumpyMorphologyBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.intensity import (
        NumbaNumpyObjectIntensityBackendStrategy,
        ObjectIntensityBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.intensity_distribution import (
        NumbaNumpyRadialDistributionBackendStrategy,
        RadialDistributionBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.neighbors import (
        NeighborTopologyBackendStrategy,
        NumbaNumpyNeighborTopologyBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.shape import (
        CentrosomeNumpyShapeMeasurementBackendStrategy,
        LegacyFastNumpyShapeMeasurementBackendStrategy,
        NumbaNumpyShapeMeasurementBackendStrategy,
        ShapeMeasurementBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.thresholding import (
        CentrosomeNumpyThresholdPrimitiveBackendStrategy,
        NumbaNumpyThresholdPrimitiveBackendStrategy,
        NumbaNumpyThresholdDiagnosticsBackendStrategy,
        NumbaNumpyThresholdSmoothingBackendStrategy,
        NumpyThresholdDiagnosticsBackendStrategy,
        ThresholdDiagnosticsBackendStrategy,
        ThresholdPrimitiveBackendStrategy,
        ThresholdSmoothingBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.watershed import (
        LegacyWatershedBackendStrategy,
        NumbaNumpyLegacyWatershedBackendStrategy,
        NumpyLegacyWatershedBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.zernike import (
        CentrosomeNumpyShapeZernikeBackendStrategy,
        LegacyFastNumpyShapeZernikeBackendStrategy,
        ShapeZernikeBackendStrategy,
    )

    assert cellprofiler_backend_key(MemoryType.NUMPY) == (
        f"{MemoryType.NUMPY.value}:"
        f"{CellProfilerBackendProvider.NATIVE.value}"
    )
    assert type(MorphologyBackendStrategy.for_memory_type(MemoryType.NUMPY)) is (
        NumbaNumpyMorphologyBackendStrategy
    )
    assert type(
        MorphologyBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.NATIVE,
        )
    ) is NumpyMorphologyBackendStrategy
    assert type(
        MorphologyBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CENTROSOME,
        )
    ) is CentrosomeNumpyMorphologyBackendStrategy
    assert type(
        MorphologyBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.NUMBA,
        )
    ) is NumbaNumpyMorphologyBackendStrategy
    assert type(ThresholdDiagnosticsBackendStrategy.for_memory_type(MemoryType.NUMPY)) is (
        NumbaNumpyThresholdDiagnosticsBackendStrategy
    )
    assert type(ThresholdSmoothingBackendStrategy.for_memory_type(MemoryType.NUMPY)) is (
        NumbaNumpyThresholdSmoothingBackendStrategy
    )
    assert type(
        ThresholdSmoothingBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.NUMBA,
        )
    ) is NumbaNumpyThresholdSmoothingBackendStrategy
    assert type(ThresholdPrimitiveBackendStrategy.for_memory_type(MemoryType.NUMPY)) is (
        NumbaNumpyThresholdPrimitiveBackendStrategy
    )
    assert type(
        ThresholdPrimitiveBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.NUMBA,
        )
    ) is (
        NumbaNumpyThresholdPrimitiveBackendStrategy
    )
    assert type(
        ThresholdPrimitiveBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CENTROSOME,
        )
    ) is CentrosomeNumpyThresholdPrimitiveBackendStrategy
    assert type(LegacyWatershedBackendStrategy.for_memory_type(MemoryType.NUMPY)) is (
        NumbaNumpyLegacyWatershedBackendStrategy
    )
    assert type(
        LegacyWatershedBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.NATIVE,
        )
    ) is NumpyLegacyWatershedBackendStrategy
    assert type(
        LegacyWatershedBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.NUMBA,
        )
    ) is (
        NumbaNumpyLegacyWatershedBackendStrategy
    )
    assert type(ObjectIntensityBackendStrategy.for_memory_type(MemoryType.NUMPY)) is (
        NumbaNumpyObjectIntensityBackendStrategy
    )
    assert type(
        ObjectIntensityBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.NUMBA,
        )
    ) is NumbaNumpyObjectIntensityBackendStrategy
    assert type(RadialDistributionBackendStrategy.for_memory_type(MemoryType.NUMPY)) is (
        NumbaNumpyRadialDistributionBackendStrategy
    )
    assert type(
        RadialDistributionBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.NUMBA,
        )
    ) is NumbaNumpyRadialDistributionBackendStrategy
    assert type(NeighborTopologyBackendStrategy.for_memory_type(MemoryType.NUMPY)) is (
        NumbaNumpyNeighborTopologyBackendStrategy
    )
    assert type(
        NeighborTopologyBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.NUMBA,
        )
    ) is NumbaNumpyNeighborTopologyBackendStrategy
    assert type(ShapeMeasurementBackendStrategy.for_memory_type(MemoryType.NUMPY)) is (
        LegacyFastNumpyShapeMeasurementBackendStrategy
    )
    assert type(
        ShapeMeasurementBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.NUMBA,
        )
    ) is (
        NumbaNumpyShapeMeasurementBackendStrategy
    )
    assert type(
        ShapeMeasurementBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CENTROSOME,
        )
    ) is CentrosomeNumpyShapeMeasurementBackendStrategy
    assert type(ShapeZernikeBackendStrategy.for_memory_type(MemoryType.NUMPY)) is (
        LegacyFastNumpyShapeZernikeBackendStrategy
    )
    assert type(
        ShapeZernikeBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CENTROSOME,
        )
    ) is CentrosomeNumpyShapeZernikeBackendStrategy


def test_cellprofiler_backend_provider_rejects_raw_strings() -> None:
    import pytest

    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.morphology import (
        MorphologyBackendStrategy,
    )

    raw_provider = CellProfilerBackendProvider.NUMBA.value

    with pytest.raises(TypeError, match="CellProfilerBackendProvider"):
        MorphologyBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=raw_provider,
        )


def test_numba_threshold_smoothing_kernel_matches_centrosome_public_primitive() -> None:
    import centrosome.smooth
    import numpy as np

    from openhcs.processing.backends.cellprofiler.thresholding import (
        _threshold_smoothing_kernel,
    )

    sigma, kernel = _threshold_smoothing_kernel(1.3488, None)
    expected = centrosome.smooth.circular_gaussian_kernel(
        sigma,
        int(np.ceil(sigma * 4.0)),
    )

    np.testing.assert_allclose(kernel, expected, rtol=0.0, atol=1e-15)


def test_threshold_application_smoothing_is_mask_normalized() -> None:
    import centrosome.smooth
    import scipy.ndimage as ndi

    from benchmark.cellprofiler_library.functions.thresholding import (
        _threshold_application_smoothed_image,
    )

    image = np.linspace(0.0, 1.0, 49, dtype=np.float64).reshape(7, 7)
    mask = np.ones((7, 7), dtype=bool)
    mask[:2, :] = False
    mask[:, :1] = False

    smoothed, sigma = _threshold_application_smoothed_image(image, mask, 1.3488)
    expected = centrosome.smooth.smooth_with_function_and_mask(
        image,
        lambda array: ndi.gaussian_filter(
            array,
            sigma=1.0,
            mode="constant",
            cval=0,
            truncate=4.0,
        ),
        mask,
    )

    assert sigma == pytest.approx(1.0)
    np.testing.assert_allclose(smoothed, expected, rtol=0.0, atol=5e-15)


def test_numba_threshold_primitives_are_default_and_roundtrip_log_transform() -> None:
    from openhcs.processing.backends.cellprofiler.thresholding import (
        NumbaNumpyThresholdPrimitiveBackendStrategy,
        ThresholdPrimitiveBackendStrategy,
    )

    strategy = ThresholdPrimitiveBackendStrategy.for_memory_type(MemoryType.NUMPY)
    image = np.array([[0.0, 0.2], [0.5, 1.0]], dtype=np.float32)

    transformed, conversion = strategy.log_transform(image)
    restored = strategy.inverse_log_transform(transformed, conversion)

    assert type(strategy) is NumbaNumpyThresholdPrimitiveBackendStrategy
    assert transformed.dtype == np.float64
    np.testing.assert_allclose(
        restored[image > 0],
        image[image > 0],
        rtol=0.0,
        atol=2e-7,
    )


def test_numba_log_transform_uses_dynamic_range_floor() -> None:
    from openhcs.processing.backends.cellprofiler.thresholding import (
        ThresholdPrimitiveBackendStrategy,
    )

    strategy = ThresholdPrimitiveBackendStrategy.for_memory_type(MemoryType.NUMPY)
    image = np.array([[0.0, 1.0 / 255.0], [0.5, 1.0]], dtype=np.float32)

    transformed, conversion = strategy.log_transform(image)
    restored_cut = strategy.inverse_log_transform(100.5 / 128.0, conversion)

    assert transformed[0, 1] > 0.0
    assert restored_cut == pytest.approx(0.3038138449192047, abs=1e-12)


def test_numba_threshold_primitives_cover_otsu_and_multiotsu() -> None:
    from openhcs.processing.backends.cellprofiler.thresholding import (
        ThresholdPrimitiveBackendStrategy,
    )

    values = np.concatenate(
        (
            np.linspace(0.05, 0.20, 30),
            np.linspace(0.45, 0.60, 30),
            np.linspace(0.80, 0.95, 30),
        )
    ).astype(np.float64)
    strategy = ThresholdPrimitiveBackendStrategy.for_memory_type(MemoryType.NUMPY)

    otsu = strategy.otsu_threshold(values)
    multiotsu = strategy.multiotsu_thresholds(values, nbins=128)
    mce = strategy.minimum_cross_entropy_threshold(values.reshape(9, 10))

    assert 0.19 < otsu < 0.60
    assert multiotsu.shape == (2,)
    assert 0.20 < multiotsu[0] < 0.60
    assert 0.60 < multiotsu[1] < 0.90
    assert 0.19 < mce < 0.60


def test_numba_multiotsu_matches_public_tie_behavior() -> None:
    from skimage.filters import threshold_multiotsu

    from openhcs.processing.backends.cellprofiler.thresholding import (
        ThresholdPrimitiveBackendStrategy,
    )

    strategy = ThresholdPrimitiveBackendStrategy.for_memory_type(MemoryType.NUMPY)
    cases = (
        (np.repeat([0, 1, 2, 3], [10, 10, 10, 10]).astype(np.float64), 128),
        (np.repeat([0, 1, 2, 3, 4], [8, 8, 8, 8, 8]).astype(np.float64), 128),
        (np.repeat([0, 0.1, 0.2, 0.5, 0.9], [10, 20, 30, 20, 10]).astype(np.float64), 5),
    )

    for values, nbins in cases:
        observed = strategy.multiotsu_thresholds(values, nbins=nbins)
        expected = threshold_multiotsu(values, classes=3, nbins=nbins)

        assert observed == pytest.approx(expected)


def test_numba_threshold_primitives_cover_histogram_methods() -> None:
    from openhcs.processing.backends.cellprofiler.thresholding import (
        ThresholdPrimitiveBackendStrategy,
    )

    strategy = ThresholdPrimitiveBackendStrategy.for_memory_type(MemoryType.NUMPY)
    values = np.concatenate(
        (
            np.full(100, 0.25, dtype=np.float64),
            np.linspace(0.30, 0.35, 50, dtype=np.float64),
            np.linspace(0.70, 0.75, 50, dtype=np.float64),
            np.full(100, 0.80, dtype=np.float64),
        )
    )

    li = strategy.li_threshold(values)
    triangle = strategy.triangle_threshold(values)
    isodata = strategy.isodata_threshold(values)
    mean = strategy.mean_threshold(values)
    yen = strategy.yen_threshold(values)
    minimum = strategy.minimum_threshold(
        np.concatenate(
            (
                np.full(100, 0.30, dtype=np.float64),
                np.full(100, 0.70, dtype=np.float64),
            )
        )
    )
    sauvola = strategy.sauvola_threshold_image(values.reshape(20, 15), window_size=5)

    assert 0.25 <= li <= 0.80
    assert 0.25 <= triangle <= 0.80
    assert 0.25 <= isodata <= 0.80
    assert mean == pytest.approx(np.mean(values))
    assert 0.25 <= yen <= 0.80
    assert 0.25 <= minimum <= 0.80
    assert sauvola.shape == (20, 15)
    assert np.all(np.isfinite(sauvola))


def test_numba_minimum_cross_entropy_uses_li_positive_shift_semantics() -> None:
    from skimage.filters import threshold_li

    from openhcs.processing.backends.cellprofiler.thresholding import (
        ThresholdPrimitiveBackendStrategy,
    )

    values = np.array(
        [0.10, 0.11, 0.12, 0.30, 0.45, 0.46, 0.47, 0.90],
        dtype=np.float64,
    )
    strategy = ThresholdPrimitiveBackendStrategy.for_memory_type(MemoryType.NUMPY)
    expected = threshold_li(values)

    assert strategy.li_threshold(values) == pytest.approx(expected)
    assert strategy.minimum_cross_entropy_threshold(values.reshape(2, 4)) == (
        pytest.approx(expected)
    )

    values32 = np.array(
        [
            2.8775458e-04,
            1.6496025e-04,
            1.5679000e-04,
            2.0272080e-04,
            5.3595490e-05,
            1.7344829e-04,
            1.8645125e-04,
            1.7703290e-04,
            5.2483805e-04,
            9.2633464e-04,
            7.6701742e-04,
            1.2272450e-03,
            7.2445755e-04,
            9.1529643e-04,
            9.3011564e-04,
            1.1702714e-03,
        ],
        dtype=np.float32,
    )
    expected32 = threshold_li(values32.copy())

    assert strategy.li_threshold(values32) == pytest.approx(float(expected32))
    assert strategy.minimum_cross_entropy_threshold(values32.reshape(4, 4)) == (
        pytest.approx(float(expected32))
    )


def test_cellprofiler_backend_selection_does_not_silently_fallback() -> None:
    import pytest

    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.morphology import (
        MorphologyBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.intensity import (
        ObjectIntensityBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.intensity_distribution import (
        RadialDistributionBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.neighbors import (
        NeighborTopologyBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.shape import (
        ShapeMeasurementBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.thresholding import (
        ThresholdPrimitiveBackendStrategy,
        ThresholdSmoothingBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.watershed import (
        LegacyWatershedBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler.zernike import (
        ShapeZernikeBackendStrategy,
    )

    with pytest.raises(NotImplementedError, match="provider 'cucim'"):
        MorphologyBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CUCIM,
        )
    with pytest.raises(NotImplementedError, match="provider 'cucim'"):
        ThresholdSmoothingBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CUCIM,
        )
    with pytest.raises(NotImplementedError, match="provider 'cucim'"):
        ThresholdPrimitiveBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CUCIM,
        )
    with pytest.raises(NotImplementedError, match="provider 'cucim'"):
        ShapeMeasurementBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CUCIM,
        )
    with pytest.raises(NotImplementedError, match="provider 'cucim'"):
        ObjectIntensityBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CUCIM,
        )
    with pytest.raises(NotImplementedError, match="provider 'cucim'"):
        RadialDistributionBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CUCIM,
        )
    with pytest.raises(NotImplementedError, match="provider 'cucim'"):
        NeighborTopologyBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CUCIM,
        )
    with pytest.raises(NotImplementedError, match="provider 'cucim'"):
        ShapeZernikeBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CUCIM,
        )
    with pytest.raises(NotImplementedError, match="provider 'cucim'"):
        LegacyWatershedBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=CellProfilerBackendProvider.CUCIM,
        )


def test_numba_declumping_smoothing_matches_native_provider() -> None:
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.morphology import (
        MorphologyBackendStrategy,
    )

    rng = np.random.default_rng(11)
    image = rng.random((21, 24), dtype=np.float32)
    mask = rng.random((21, 24)) > 0.2
    mask[:2, :3] = False
    filter_size = 6.5

    native = MorphologyBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=CellProfilerBackendProvider.NATIVE,
    ).smooth_image_for_declumping(image, mask, filter_size)
    numba = MorphologyBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=CellProfilerBackendProvider.NUMBA,
    ).smooth_image_for_declumping(image, mask, filter_size)

    np.testing.assert_allclose(numba, native, rtol=0.0, atol=1e-7)
    assert numba.dtype == native.dtype


def test_numba_seed_shrink_matches_centrosome_provider() -> None:
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.morphology import (
        MorphologyBackendStrategy,
    )

    mask = np.zeros((24, 28), dtype=bool)
    mask[2:8, 3:10] = True
    mask[10:16, 12:20] = np.eye(6, 8, dtype=bool)
    mask[17:20, 21:25] = True
    mask[19, 20] = True

    numba = MorphologyBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=CellProfilerBackendProvider.NUMBA,
    ).shrink_components_to_seed_points(mask)
    centrosome = MorphologyBackendStrategy.for_memory_type(
        MemoryType.NUMPY,
        backend_provider=CellProfilerBackendProvider.CENTROSOME,
    ).shrink_components_to_seed_points(mask)

    np.testing.assert_array_equal(numba, centrosome)


def test_legacy_fast_shape_zernike_backend_matches_centrosome_provider() -> None:
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.zernike import (
        shape_zernike_moments,
    )

    labels = np.zeros((16, 18), dtype=np.int32)
    labels[2:8, 3:10] = 1
    labels[9:14, 11:16] = 2
    labels[11, 13] = 0
    measured_labels = np.array([1, 2], dtype=np.int32)

    legacy_fast_indexes, legacy_fast_values = shape_zernike_moments(
        labels,
        measured_labels,
        max_order=5,
        backend_provider=CellProfilerBackendProvider.LEGACY_FAST,
    )
    centrosome_indexes, centrosome_values = shape_zernike_moments(
        labels,
        measured_labels,
        max_order=5,
        backend_provider=CellProfilerBackendProvider.CENTROSOME,
    )

    assert legacy_fast_indexes == centrosome_indexes
    np.testing.assert_allclose(
        legacy_fast_values,
        centrosome_values,
        atol=1e-12,
        rtol=1e-12,
    )


def test_legacy_fast_shape_zernike_backend_zeros_pixels_outside_unit_circle() -> None:
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.zernike import (
        shape_zernike_moments,
    )

    labels = np.zeros((12, 12), dtype=np.int32)
    labels[2:6, 2:6] = 1
    labels[2, 2] = 0
    measured_labels = np.array([1], dtype=np.int32)

    legacy_fast_indexes, legacy_fast_values = shape_zernike_moments(
        labels,
        measured_labels,
        max_order=5,
        backend_provider=CellProfilerBackendProvider.LEGACY_FAST,
    )
    centrosome_indexes, centrosome_values = shape_zernike_moments(
        labels,
        measured_labels,
        max_order=5,
        backend_provider=CellProfilerBackendProvider.CENTROSOME,
    )

    assert legacy_fast_indexes == centrosome_indexes
    np.testing.assert_allclose(
        legacy_fast_values,
        centrosome_values,
        atol=1e-12,
        rtol=1e-12,
    )


def test_legacy_fast_intensity_zernike_backend_matches_centrosome_provider() -> None:
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.zernike import (
        intensity_zernike_moments,
    )

    labels = np.zeros((16, 18), dtype=np.int32)
    labels[2:8, 3:10] = 1
    labels[9:14, 11:16] = 2
    labels[11, 13] = 0
    image = np.linspace(0.0, 1.0, labels.size, dtype=np.float64).reshape(labels.shape)
    measured_labels = np.array([1, 2], dtype=np.int32)

    legacy_fast_indexes, legacy_fast_magnitudes, legacy_fast_phases = (
        intensity_zernike_moments(
            image,
            labels,
            measured_labels,
            max_order=5,
            backend_provider=CellProfilerBackendProvider.LEGACY_FAST,
        )
    )
    (
        centrosome_indexes,
        centrosome_magnitudes,
        centrosome_phases,
    ) = intensity_zernike_moments(
        image,
        labels,
        measured_labels,
        max_order=5,
        backend_provider=CellProfilerBackendProvider.CENTROSOME,
    )

    assert legacy_fast_indexes == centrosome_indexes
    np.testing.assert_allclose(
        legacy_fast_magnitudes,
        centrosome_magnitudes,
        atol=1e-12,
        rtol=1e-12,
    )
    meaningful_phase = centrosome_magnitudes > 1e-12
    np.testing.assert_allclose(
        legacy_fast_phases[meaningful_phase],
        centrosome_phases[meaningful_phase],
        atol=1e-12,
        rtol=1e-12,
    )


def test_zernike_numba_provider_is_not_registered_until_pure() -> None:
    import pytest

    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.zernike import (
        ShapeZernikeBackendStrategy,
    )

    with pytest.raises(NotImplementedError, match="provider 'numba'"):
        ShapeZernikeBackendStrategy.for_memory_type(
            backend_provider=CellProfilerBackendProvider.NUMBA,
        )
