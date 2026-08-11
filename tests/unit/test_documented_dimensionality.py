"""Source-backed drift gates for published dimensionality claims."""

from pathlib import Path

from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.callable_contract import CallableContract
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelInputExecutionMode,
    object_label_input_execution_mode_from_callable,
)
from openhcs.processing.backends.cellprofiler.area_occupied import (
    measure_image_volume_occupied_binary,
    measure_image_volume_occupied_objects,
)
from openhcs.processing.backends.cellprofiler.intensity import (
    MeasureObjectIntensityModule,
)
from openhcs.processing.backends.cellprofiler.primary_objects import (
    identify_primary_objects,
)
from openhcs.processing.backends.cellprofiler.secondary import (
    identify_secondary_objects,
    identify_tertiary_objects,
)
from openhcs.processing.backends.cellprofiler.shape import (
    MeasureObjectSizeShapeModule,
    measure_object_size_shape,
)
from openhcs.processing.backends.cellprofiler.watershed import watershed_cellprofiler4
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = REPO_ROOT / "docs/source/reference/dimensionality_and_measurements.rst"


def test_published_dimensionality_boundary_matches_callable_declarations() -> None:
    reference = REFERENCE_PATH.read_text(encoding="utf-8")
    normalized_reference = reference.replace("_", "").casefold()
    website = (REPO_ROOT / "website/index.html").read_text(encoding="utf-8")

    for plane_local_callable in (
        identify_primary_objects,
        identify_secondary_objects,
        identify_tertiary_objects,
    ):
        assert (
            CallableContract.from_callable(
                plane_local_callable
            ).require_processing_contract()
            is ProcessingContract.PURE_2D
        )
        assert plane_local_callable.__name__.replace("_", "") in normalized_reference

    assert (
        CallableContract.from_callable(
            watershed_cellprofiler4
        ).runtime_image_execution_mode
        is ImagePayloadExecutionMode.FULL_STACK
    )
    assert (
        object_label_input_execution_mode_from_callable(measure_object_size_shape)
        is ObjectLabelInputExecutionMode.FULL_STACK
    )
    for volume_callable in (
        measure_image_volume_occupied_binary,
        measure_image_volume_occupied_objects,
    ):
        assert (
            CallableContract.from_callable(
                volume_callable
            ).require_processing_contract()
            is ProcessingContract.PURE_3D
        )

    shape_features = set(MeasureObjectSizeShapeModule.standard_3d_features)
    assert {
        MeasureObjectSizeShapeModule.MeasurementFeature.VOLUME,
        MeasureObjectSizeShapeModule.MeasurementFeature.SURFACE_AREA,
        MeasureObjectSizeShapeModule.MeasurementFeature.CENTER_Z,
        MeasureObjectSizeShapeModule.MeasurementFeature.BOUNDING_BOX_VOLUME,
    } <= shape_features
    intensity_features = set(MeasureObjectIntensityModule.MeasurementFeature)
    assert {
        MeasureObjectIntensityModule.MeasurementFeature.CENTER_MASS_INTENSITY_Z,
        MeasureObjectIntensityModule.MeasurementFeature.MAX_INTENSITY_Z,
    } <= intensity_features

    assert "Array dimensionality alone is not an execution contract" in reference
    assert "Plane-local labels are not stitched into volumetric objects" in reference
    assert "function-defined, not a global 2D/3D switch" in website
    assert "plane-local labels" in website
    assert "not silently stitched across Z" in website
