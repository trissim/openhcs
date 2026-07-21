import numpy as np

from openhcs.core.artifacts import ArtifactSpec, ObjectLabelsArtifactType
from openhcs.core.pipeline.function_contracts import (
    object_label_input_execution_mode_from_callable,
)
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_metadata,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelVariantData,
)
from openhcs.core.measurement_image_alignment import MeasurementImageReferenceDomain
from openhcs.core.runtime_object_label_domains import ObjectLabelDomain, ObjectLabelDomainScope
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis, RuntimePlaneAxisProjector, RuntimePlaneAxisValueProjection
from openhcs.interop.cellprofiler.runtime.invocation import (
    CellProfilerMeasurementImage,
)
from openhcs.interop.cellprofiler.runtime.measurement_execution_support import (
    object_measurement_runtime_inputs,
)
from openhcs.processing.backends.cellprofiler.intensity import (
    measure_object_intensity,
)


class _SingletonRuntimePlaneProjector(RuntimePlaneAxisProjector):
    def runtime_slice_plane_index(self) -> int | None:
        return None

    def runtime_slice_axis_size(self) -> int:
        return 1


def test_singleton_object_projection_consumes_measurement_image_plane_proof() -> None:
    image = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.ones((1, 4, 5), dtype=np.float32), None)
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.ones((1, 4, 5), dtype=np.int32),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((1,),),
        ),
    )
    source = CellProfilerMeasurementImage(
        source_image_name=None,
        payload=image,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=1,
        ),
        reference_domain=MeasurementImageReferenceDomain.OBJECT_LABELS,
    )

    (
        aligned_source,
        _labels,
        _completion,
        _mode,
        _events,
        _payload_time,
        _align_time,
    ) = object_measurement_runtime_inputs(
        object_label_execution=object_label_input_execution_mode_from_callable(
            measure_object_intensity
        ),
        measurement_image=source,
        object_spec=ArtifactSpec.input("Objects", ObjectLabelsArtifactType),
        label_payload=labels,
        adapter=_SingletonRuntimePlaneProjector(),
    )

    assert aligned_source.plane_projection is None
    assert image_payload_metadata(aligned_source.payload).plane_axis is None
    assert np.shape(aligned_source.payload) == (4, 5)
