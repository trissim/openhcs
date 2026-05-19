from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from openhcs.constants.constants import MemoryType
from openhcs.core.callable_contract import PROCESSING_CONTRACT_ATTR
from openhcs.processing.backends.lib_registry.openhcs_registry import OpenHCSRegistry
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


def test_openhcs_product_code_does_not_import_benchmark_package() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    offenders: list[tuple[str, int, str]] = []
    for file_path in sorted((repo_root / "openhcs").rglob("*.py")):
        tree = ast.parse(file_path.read_text(), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "benchmark" or alias.name.startswith("benchmark."):
                        offenders.append(
                            (str(file_path.relative_to(repo_root)), node.lineno, alias.name)
                        )
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if module_name == "benchmark" or module_name.startswith("benchmark."):
                    offenders.append(
                        (str(file_path.relative_to(repo_root)), node.lineno, module_name)
                    )

    assert offenders == []


def test_generated_cellprofiler_pipelines_import_product_library() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    offenders: list[str] = []
    for file_path in sorted((repo_root / "benchmark" / "cellprofiler_pipelines").glob("*_openhcs.py")):
        if "benchmark.cellprofiler_library" in file_path.read_text():
            offenders.append(str(file_path.relative_to(repo_root)))

    assert offenders == []


def test_cellprofiler_converter_does_not_import_absorbed_runtime_packages() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    banned_modules = (
        "benchmark.cellprofiler_library.functions",
        "benchmark.cellprofiler_compat.measurement_lookup",
        "benchmark.cellprofiler_compat.perf_fixtures",
    )
    offenders: list[tuple[str, int, str]] = []
    for file_path in sorted((repo_root / "benchmark" / "converter").rglob("*.py")):
        tree = ast.parse(file_path.read_text(), filename=str(file_path))
        for node in ast.walk(tree):
            module_names: list[str] = []
            if isinstance(node, ast.Import):
                module_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module_names = [node.module or ""]
            for module_name in module_names:
                if any(
                    module_name == banned or module_name.startswith(f"{banned}.")
                    for banned in banned_modules
                ):
                    offenders.append(
                        (str(file_path.relative_to(repo_root)), node.lineno, module_name)
                    )

    assert offenders == []


def test_cellprofiler_runtime_bridge_uses_product_semantics_for_special_payloads() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    file_path = repo_root / "benchmark" / "cellprofiler_compat" / "module_execution.py"
    banned_modules = (
        "benchmark.cellprofiler_library.functions.relateobjects",
        "benchmark.cellprofiler_library.functions.structuring_elements",
        "benchmark.cellprofiler_library.functions.untangleworms",
        "benchmark.cellprofiler_library.functions.watershed",
    )
    tree = ast.parse(file_path.read_text(), filename=str(file_path))
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        module_names: list[str] = []
        if isinstance(node, ast.Import):
            module_names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module_names = [node.module or ""]
        for module_name in module_names:
            if module_name in banned_modules:
                offenders.append((node.lineno, module_name))

    assert offenders == []


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


def test_cellprofiler_processing_backend_exports_resize_volumetric_variant() -> None:
    from openhcs.processing.backends import cellprofiler

    function = cellprofiler.require_cellprofiler_function(
        "Resize",
        function_name="resize_volumetric",
    )

    assert function is cellprofiler.resize_volumetric
    assert function.input_memory_type == MemoryType.NUMPY.value
    assert function.output_memory_type == MemoryType.NUMPY.value
    assert getattr(function, PROCESSING_CONTRACT_ATTR) is ProcessingContract.PURE_3D


def test_cellprofiler_processing_backend_exports_resize_objects_volumetric_variant() -> None:
    from openhcs.processing.backends import cellprofiler

    function = cellprofiler.require_cellprofiler_function(
        "ResizeObjects",
        function_name="resize_objects_3d",
    )

    assert function is cellprofiler.resize_objects_3d
    assert function.input_memory_type == MemoryType.NUMPY.value
    assert function.output_memory_type == MemoryType.NUMPY.value
    assert getattr(function, PROCESSING_CONTRACT_ATTR) is ProcessingContract.PURE_3D


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


def test_cellprofiler_threshold_diagnostics_handles_nd_images_as_one_domain() -> None:
    from openhcs.processing.backends.cellprofiler.thresholding import (
        NumpyThresholdDiagnosticsBackendStrategy,
        ThresholdDiagnosticsBackendStrategy,
    )

    strategy = ThresholdDiagnosticsBackendStrategy.for_memory_type(MemoryType.NUMPY)
    image = np.array(
        [
            [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]],
            [[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]],
        ],
        dtype=np.float32,
    )
    binary = image > 0.5

    weighted_variance, sum_of_entropies = strategy.diagnostics(image, None, binary)
    reference_weighted_variance = (
        NumpyThresholdDiagnosticsBackendStrategy().weighted_variance(
            image,
            np.ones(image.shape, dtype=bool),
            binary,
        )
    )
    reference_sum_of_entropies = (
        NumpyThresholdDiagnosticsBackendStrategy().sum_of_entropies(
            image,
            np.ones(image.shape, dtype=bool),
            binary,
        )
    )

    assert isinstance(weighted_variance, float)
    assert isinstance(sum_of_entropies, float)
    np.testing.assert_allclose(weighted_variance, reference_weighted_variance)
    np.testing.assert_allclose(sum_of_entropies, reference_sum_of_entropies)


def test_cellprofiler_threshold_quantized_diagnostics_preserve_low_dynamic_range() -> None:
    from openhcs.processing.backends.cellprofiler.thresholding import (
        NumbaNumpyThresholdDiagnosticsBackendStrategy,
        NumpyThresholdDiagnosticsBackendStrategy,
    )

    image = np.array(
        [
            [0.0001, 0.0004, 0.0012, 0.0030],
            [0.0002, 0.0005, 0.0015, 0.0034],
        ],
        dtype=np.float32,
    )
    binary = image > 0.001
    mask = np.ones(image.shape, dtype=bool)

    weighted_variance, sum_of_entropies = (
        NumbaNumpyThresholdDiagnosticsBackendStrategy().diagnostics(
            image,
            mask,
            binary,
            proven_unit_interval_scale=65535,
        )
    )
    reference_weighted_variance = (
        NumpyThresholdDiagnosticsBackendStrategy().weighted_variance(
            np.rint(image * 65535).astype(np.int64) / 65535,
            mask,
            binary,
        )
    )

    np.testing.assert_allclose(weighted_variance, reference_weighted_variance)
    assert isinstance(sum_of_entropies, float)


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
        NativeNumpyRadialDistributionBackendStrategy,
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
            backend_provider=CellProfilerBackendProvider.NATIVE,
        )
    ) is NativeNumpyRadialDistributionBackendStrategy
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


def test_numba_neighbor_topology_outline_frontier_matches_dense_reference() -> None:
    from openhcs.processing.backends.cellprofiler.neighbors import (
        NumbaNumpyNeighborTopologyBackendStrategy,
    )

    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:8, 2:8] = 1
    labels[2:8, 11:17] = 2
    labels[12:18, 3:9] = 3
    perimeter = _labeled_perimeter(labels)
    footprint = _disk_footprint(4)
    touching_footprint = _disk_footprint(4.5)

    observed = NumbaNumpyNeighborTopologyBackendStrategy().measure_topology(
        labels,
        labels,
        perimeter,
        np.array([1, 2, 3], dtype=np.int32),
        distance=4,
        neighbors_are_same_objects=True,
        footprint=footprint,
        touching_footprint=touching_footprint,
        variant_object_count=3,
        variant_neighbor_count=3,
    )
    expected_neighbor_count, expected_touching = _dense_neighbor_topology_reference(
        labels,
        labels,
        perimeter,
        footprint,
        touching_footprint,
        neighbors_are_same_objects=True,
        variant_object_count=3,
        variant_neighbor_count=3,
    )

    np.testing.assert_array_equal(observed.neighbor_count, expected_neighbor_count)
    np.testing.assert_array_equal(observed.touching_pixel_count, expected_touching)


def _disk_footprint(radius: float) -> np.ndarray:
    limit = int(np.ceil(radius))
    yy, xx = np.ogrid[-limit : limit + 1, -limit : limit + 1]
    return (yy * yy + xx * xx) <= radius * radius


def _labeled_perimeter(labels: np.ndarray) -> np.ndarray:
    perimeter = np.zeros_like(labels)
    height, width = labels.shape
    for y in range(height):
        for x in range(width):
            object_number = labels[y, x]
            if object_number == 0:
                continue
            for offset_y, offset_x in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor_y = y + offset_y
                neighbor_x = x + offset_x
                if (
                    neighbor_y < 0
                    or neighbor_y >= height
                    or neighbor_x < 0
                    or neighbor_x >= width
                    or labels[neighbor_y, neighbor_x] != object_number
                ):
                    perimeter[y, x] = object_number
                    break
    return perimeter


def _dense_neighbor_topology_reference(
    working_labels: np.ndarray,
    neighbor_working_labels: np.ndarray,
    perimeter_outlines: np.ndarray,
    footprint: np.ndarray,
    touching_footprint: np.ndarray,
    *,
    neighbors_are_same_objects: bool,
    variant_object_count: int,
    variant_neighbor_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    adjacency = np.zeros((variant_object_count, variant_neighbor_count + 1), dtype=bool)
    touching_pixel_count = np.zeros(variant_object_count, dtype=float)
    height, width = working_labels.shape
    offsets = _footprint_offsets(footprint)
    touching_offsets = _footprint_offsets(touching_footprint)

    for y in range(height):
        for x in range(width):
            object_number = working_labels[y, x]
            if object_number <= 0 or object_number > variant_object_count:
                continue
            object_index = object_number - 1
            for offset_y, offset_x in offsets:
                neighbor_y = y + offset_y
                neighbor_x = x + offset_x
                if (
                    neighbor_y < 0
                    or neighbor_y >= height
                    or neighbor_x < 0
                    or neighbor_x >= width
                ):
                    continue
                neighbor_number = neighbor_working_labels[neighbor_y, neighbor_x]
                if neighbor_number <= 0 or neighbor_number > variant_neighbor_count:
                    continue
                if neighbors_are_same_objects and neighbor_number == object_number:
                    continue
                adjacency[object_index, neighbor_number] = True

            if perimeter_outlines[y, x] != object_number:
                continue
            for offset_y, offset_x in touching_offsets:
                neighbor_y = y + offset_y
                neighbor_x = x + offset_x
                if (
                    neighbor_y < 0
                    or neighbor_y >= height
                    or neighbor_x < 0
                    or neighbor_x >= width
                ):
                    continue
                if neighbors_are_same_objects:
                    touches = (
                        working_labels[neighbor_y, neighbor_x] != 0
                        and working_labels[neighbor_y, neighbor_x] != object_number
                    )
                else:
                    touches = neighbor_working_labels[neighbor_y, neighbor_x] != 0
                if touches:
                    touching_pixel_count[object_index] += 1.0
                    break

    return adjacency[:, 1:].sum(axis=1).astype(float), touching_pixel_count


def _footprint_offsets(footprint: np.ndarray) -> tuple[tuple[int, int], ...]:
    center_y = footprint.shape[0] // 2
    center_x = footprint.shape[1] // 2
    return tuple(
        (int(y - center_y), int(x - center_x))
        for y, x in np.argwhere(footprint)
    )


def test_shape_distance_to_edge_handles_stacked_planes_planewise() -> None:
    from openhcs.processing.backends.cellprofiler.shape import (
        ShapeMeasurementBackendStrategy,
    )

    backend = ShapeMeasurementBackendStrategy.for_memory_type(MemoryType.NUMPY)
    labels = np.zeros((2, 6, 6), dtype=np.int32)
    labels[0, 1:5, 1:5] = 1
    labels[1, 2:4, 2:4] = 2

    distances = backend.distance_to_edge(labels)

    np.testing.assert_allclose(distances[0], backend.distance_to_edge(labels[0]))
    np.testing.assert_allclose(distances[1], backend.distance_to_edge(labels[1]))


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

    from openhcs.processing.backends.cellprofiler.thresholding import (
        ThresholdApplicationSmoothing,
    )

    image = np.linspace(0.0, 1.0, 49, dtype=np.float64).reshape(7, 7)
    mask = np.ones((7, 7), dtype=bool)
    mask[:2, :] = False
    mask[:, :1] = False

    smoothed, sigma = ThresholdApplicationSmoothing(1.3488).smooth(image, mask)
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


def test_object_intensity_inner_edges_match_cellprofiler_boundary_semantics() -> None:
    from skimage.segmentation import find_boundaries

    from openhcs.processing.backends.cellprofiler.intensity import (
        object_intensity_backend,
    )

    image = np.arange(25, dtype=np.float64).reshape(5, 5)
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[:3, :3] = 1

    arrays = object_intensity_backend().measure(image, labels)
    edge_mask = find_boundaries(labels, mode="inner") & (labels == 1)

    assert arrays.integrated_intensity_edge[0] == pytest.approx(
        float(np.sum(image[edge_mask]))
    )
    assert arrays.mean_intensity_edge[0] == pytest.approx(
        float(np.mean(image[edge_mask]))
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


def test_shape_zernike_backend_uses_declared_measured_label_domain() -> None:
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.zernike import (
        shape_zernike_moments,
    )

    labels = np.zeros((16, 18), dtype=np.int32)
    labels[2:8, 3:10] = 1
    labels[9:14, 11:16] = 3
    measured_labels = np.array([1, 3], dtype=np.int32)

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
    assert legacy_fast_values.shape == centrosome_values.shape == (
        measured_labels.size,
        len(legacy_fast_indexes),
    )
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


def test_zernike_label_geometry_cache_reuses_equal_label_values() -> None:
    from openhcs.processing.backends.cellprofiler import zernike

    labels = np.zeros((12, 12), dtype=np.int32)
    labels[2:6, 2:6] = 1
    labels[7:10, 7:11] = 2
    object_ids = np.array([1, 2], dtype=np.int32)
    zernike._ZERNIKE_LABEL_GEOMETRY_CACHE.clear()
    strategy = zernike.LegacyFastNumpyShapeZernikeBackendStrategy()

    first = strategy.zernike_label_geometry(labels, object_ids)
    second = strategy.zernike_label_geometry(labels.copy(), object_ids.copy())

    assert second is first
    assert len(zernike._ZERNIKE_LABEL_GEOMETRY_CACHE) == 1
    zernike._ZERNIKE_LABEL_GEOMETRY_CACHE.clear()


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
