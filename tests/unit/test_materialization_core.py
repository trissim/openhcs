import json

import numpy as np
import pytest

import openhcs  # noqa: F401
from polystore.filemanager import FileManager
from polystore.memory import MemoryStorageBackend
from polystore.streaming.identity import (
    FixedStreamProducerIdentityKind,
    StreamProducerIdentity,
)
from polystore.streaming.viewer_transport import (
    BatchViewerStreamSourceMetadata,
    ViewerDisplayConfigABC,
    ViewerFilenameParserABC,
    ViewerMetadataHandlerABC,
    ViewerMicroscopeHandlerABC,
    ViewerStreamProducer,
    ViewerStreamBackendKwargs,
    ViewerStreamRequest,
    ViewerStreamSource,
    ViewerStreamSourceIdentity,
)
from zmqruntime.viewer_protocol import ViewerTransportEndpoint

from openhcs.core.config import TransportMode
from openhcs.core.measurement_row_materialization import MeasurementProjectedColumnarRows
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser
from openhcs.processing.materialization import (
    JsonOptions,
    MaterializationSpec,
    ROIOptions,
    csv_only,
    json_materializer,
    json_only,
    materialize,
    tabular_field_names_from_materialization,
    tiff_stack,
)
from openhcs.processing.materialization.core import (
    MaterializationInputItem,
    ROIMaterializationArchiveIdentity,
    ROIMaterializationTarget,
    ROIMaterializationTargetCoalescer,
    ViewerStreamBackendCallKwargs,
)
from openhcs.core.runtime_values import (
    ImageMetadataPayload,
    ImagePayloadMetadata,
    ObjectLabelPayload,
    SourceImageProvenancePlanes,
)
from openhcs.core.source_image_provenance import SourceImageIdentity


def _memory_materialize(spec, data, path, filemanager):
    return materialize(
        spec,
        data=data,
        path=path,
        filemanager=filemanager,
        backends=["memory"],
        backend_kwargs={},
    )


class _TestViewerDisplayConfig(ViewerDisplayConfigABC):
    COMPONENT_ORDER = ("well", "site", "z_index", "channel")

    def component_modes(self):
        return {}


class _TestViewerFilenameParser(ViewerFilenameParserABC):
    def parse_filename(self, filename):
        return None


class _TestViewerMetadataHandler(ViewerMetadataHandlerABC):
    def get_component_values(self, plate_path, component_name):
        return {}


class _TestViewerMicroscopeHandler(ViewerMicroscopeHandlerABC):
    parser = _TestViewerFilenameParser()
    metadata_handler = _TestViewerMetadataHandler()


def _viewer_stream_backend_kwargs():
    request = ViewerStreamRequest(
        viewer_transport=ViewerTransportEndpoint(
            host="localhost",
            port=5555,
            transport_mode=TransportMode.IPC,
        ),
        display_config=_TestViewerDisplayConfig(),
        source=ViewerStreamSource(
            identity=ViewerStreamSourceIdentity(
                microscope_handler=_TestViewerMicroscopeHandler(),
                plate_path="/tmp/test_plate",
            ),
            metadata=BatchViewerStreamSourceMetadata(
                {"well": "A01", "site": 1, "channel": 1}
            ),
        ),
        producer=ViewerStreamProducer.from_identity(
            StreamProducerIdentity.fixed_output(
                FixedStreamProducerIdentityKind.DIRECT,
                "test",
            )
        ),
    )
    return ViewerStreamBackendCallKwargs(ViewerStreamBackendKwargs(request))


def _stream_component_metadata(saved_item):
    stream_request = saved_item[3]["stream_request"]
    return stream_request.source.metadata.component_metadata


def _stream_materialize(spec, data, path, filemanager, context=None):
    return materialize(
        spec,
        data=data,
        path=path,
        filemanager=filemanager,
        backends=["napari_stream"],
        backend_kwargs={"napari_stream": _viewer_stream_backend_kwargs()},
        context=context,
    )


def _two_plane_roi_labels():
    labels = np.zeros((2, 8, 8), dtype=np.int32)
    labels[0, 1:4, 1:4] = 1
    labels[1, 4:7, 4:7] = 2
    return labels


class _RecordingBackend:
    requires_filesystem_validation = False


class _RecordingFileManager:
    def __init__(self):
        self.saved = []

    def _get_backend(self, _backend):
        return _RecordingBackend()

    def save(self, content, path, backend, **kwargs):
        self.saved.append((content, path, backend, kwargs))


class _SourceSchemaMicroscopeHandler:
    parser = SourceSchemaFilenameParser()


class _SourceSchemaProcessingContext:
    microscope_handler = _SourceSchemaMicroscopeHandler()


def _addressable_roi_label_payload(first_path, second_path, first_metadata, second_metadata):
    return ObjectLabelPayload(
        labels=_two_plane_roi_labels(),
        source_image_provenance_planes = SourceImageProvenancePlanes.from_components(paths = (first_path, second_path), component_metadata = (first_metadata, second_metadata)))


def _unaddressed_roi_label_payload():
    return ObjectLabelPayload(
        labels=_two_plane_roi_labels(),
        source_image_provenance_planes = SourceImageProvenancePlanes.from_components(paths = (None, None), component_metadata = (None, None)))


def _roi_target(roi_path, labels, metadata):
    return ROIMaterializationTarget(
        archive=ROIMaterializationArchiveIdentity.from_metadata(
            path=roi_path,
            metadata=metadata,
        ),
        items=(
            MaterializationInputItem(
                value=ObjectLabelPayload(labels=labels),
                source_description="materialization payload",
            ),
        ),
    )


@pytest.mark.unit
def test_materialize_strips_roi_zip_compound_suffix_for_json() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})

    spec = MaterializationSpec(JsonOptions(filename_suffix=".json"))
    out = materialize(
        spec,
        data={"ok": True},
        path="/tmp/A01_test_output.roi.zip",
        filemanager=fm,
        backends=["memory"],
        backend_kwargs={},
    )

    assert out == "/tmp/A01_test_output.json"
    assert fm.load(out, "memory") == json.dumps({"ok": True}, indent=2, default=str)


@pytest.mark.unit
def test_csv_materialization_preserves_declared_fields_for_empty_rows() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})

    out = materialize(
        csv_only(fields=["object_label", "area"]),
        data=[],
        path="/tmp/A01_measurements",
        filemanager=fm,
        backends=["memory"],
        backend_kwargs={},
    )

    assert out == "/tmp/A01_measurements_details.csv"
    assert fm.load(out, "memory").splitlines()[0] == "object_label,area"


@pytest.mark.unit
def test_csv_materialization_preserves_declared_field_order_for_dict_rows() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})

    out = materialize(
        csv_only(fields=["object_label", "area"]),
        data=[{"area": 9, "object_label": 1, "ignored": 99}],
        path="/tmp/A01_measurements",
        filemanager=fm,
        backends=["memory"],
        backend_kwargs={},
    )

    assert out == "/tmp/A01_measurements_details.csv"
    assert fm.load(out, "memory").splitlines() == ["object_label,area", "1,9"]


@pytest.mark.unit
def test_csv_materialization_expands_columnar_rows() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})

    out = materialize(
        csv_only(fields=["object_label", "area"]),
        data=MeasurementProjectedColumnarRows(
            {
                "object_label": (1, 2),
                "area": (9.0, 10.5),
            }
        ),
        path="/tmp/A01_measurements",
        filemanager=fm,
        backends=["memory"],
        backend_kwargs={},
    )

    assert out == "/tmp/A01_measurements_details.csv"
    assert fm.load(out, "memory").splitlines() == [
        "object_label,area",
        "1,9.0",
        "2,10.5",
    ]


@pytest.mark.unit
def test_json_materialization_filters_to_declared_fields() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})

    out = materialize(
        json_materializer(fields=["object_label", "area"]),
        data=[{"object_label": 1, "area": 9, "ignored": 99}],
        path="/tmp/A01_measurements",
        filemanager=fm,
        backends=["memory"],
        backend_kwargs={},
    )

    assert out == "/tmp/A01_measurements.json"
    assert json.loads(fm.load(out, "memory")) == [{"object_label": 1, "area": 9}]


@pytest.mark.unit
def test_tabular_field_names_from_materialization_reads_csv_and_json_options() -> None:
    assert tabular_field_names_from_materialization(
        csv_only(fields=["object_label", "area"])
    ) == ("object_label", "area")
    assert tabular_field_names_from_materialization(
        json_only(fields=["object_label", "area"])
    ) == ("object_label", "area")


@pytest.mark.unit
def test_tiff_stack_preserves_channels_last_color_image_as_single_file() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})
    image = np.zeros((5, 7, 3), dtype=np.uint8)

    out = materialize(
        tiff_stack(),
        data=image,
        path="/tmp/A01_rgb",
        filemanager=fm,
        backends=["memory"],
        backend_kwargs={},
    )

    assert out == "/tmp/A01_rgb_slice_000.tif"
    assert fm.exists(out, "memory")
    assert not fm.exists("/tmp/A01_rgb_slice_001.tif", "memory")


@pytest.mark.unit
def test_tiff_stack_splits_scalar_3d_stack_by_plane() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})
    stack = np.zeros((3, 5, 7), dtype=np.uint8)

    out = materialize(
        tiff_stack(),
        data=stack,
        path="/tmp/A01_stack",
        filemanager=fm,
        backends=["memory"],
        backend_kwargs={},
    )

    assert out == "/tmp/A01_stack_slice_000.tif"
    assert fm.exists("/tmp/A01_stack_slice_000.tif", "memory")
    assert fm.exists("/tmp/A01_stack_slice_001.tif", "memory")
    assert fm.exists("/tmp/A01_stack_slice_002.tif", "memory")
    assert not fm.exists("/tmp/A01_stack_slice_003.tif", "memory")


@pytest.mark.unit
def test_tiff_stack_streaming_saves_per_slice_component_metadata() -> None:
    class _RecordingBackend:
        requires_filesystem_validation = False

    class _RecordingFileManager:
        def __init__(self):
            self.saved = []

        def _get_backend(self, _backend):
            return _RecordingBackend()

        def save(self, content, path, backend, **kwargs):
            self.saved.append((content, path, backend, kwargs))

    fm = _RecordingFileManager()
    payload = ObjectLabelPayload(
        labels=np.zeros((2, 5, 7), dtype=np.int32),
        source_image_provenance_planes = SourceImageProvenancePlanes.from_components(component_metadata = (
            {"well": "A01", "site": 1, "z_index": 1},
            {"well": "A01", "site": 2, "z_index": 1},
        )))

    materialize(
        tiff_stack(),
        data=payload,
        path="/tmp/A01_s001_w1_z001_t001_NucleiObjects3D_step9",
        filemanager=fm,
        backends=["napari_stream"],
        backend_kwargs={"napari_stream": _viewer_stream_backend_kwargs()},
    )

    saved_slices = [
        item
        for item in fm.saved
        if item[1].endswith(("_slice_000.tif", "_slice_001.tif"))
    ]
    assert [_stream_component_metadata(item) for item in saved_slices] == [
        {"well": "A01", "site": 1, "z_index": 1},
        {"well": "A01", "site": 2, "z_index": 1},
    ]


@pytest.mark.unit
def test_tiff_stack_streaming_uses_single_plane_scalar_source_identity() -> None:
    fm = _RecordingFileManager()
    payload = ImageMetadataPayload(
        data=np.zeros((5, 7, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                component_metadata=(
                    {"well": "A01", "site": 1, "channel": 3},
                )
            ),
        ),
    )

    materialize(
        tiff_stack(),
        data=payload,
        path="/tmp/A01_s001_w3_z001_t001_SanityCheck_step6",
        filemanager=fm,
        backends=["napari_stream"],
        backend_kwargs={"napari_stream": _viewer_stream_backend_kwargs()},
    )

    saved_images = [
        item
        for item in fm.saved
        if item[1].endswith(".tif")
    ]
    assert len(saved_images) == 1
    assert _stream_component_metadata(saved_images[0]) == {
        "well": "A01",
        "site": 1,
        "channel": 3,
    }


@pytest.mark.unit
def test_roi_materialization_offsets_object_label_payload_geometry() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[2:6, 3:7] = 1
    payload = ObjectLabelPayload(
        labels=labels,
        source_spatial_domain=SourceSpatialDomain(origin_yx=(10, 20)),
    )

    out = materialize(
        MaterializationSpec(ROIOptions(min_area=0)),
        data=payload,
        path="/tmp/A01_Nuclei_step3.roi.zip",
        filemanager=fm,
        backends=["memory"],
        backend_kwargs={},
    )

    rois = fm.load(out, "memory")
    assert out == "/tmp/A01_Nuclei_step3_rois.roi.zip"
    assert rois[0].metadata["bbox"] == (12, 23, 16, 27)
    assert rois[0].metadata["centroid"] == (13.5, 24.5)
    assert float(rois[0].shapes[0].coordinates[:, 0].min()) >= 11.5
    assert float(rois[0].shapes[0].coordinates[:, 1].min()) >= 22.5


@pytest.mark.unit
def test_roi_materialization_extracts_each_plane_from_object_label_stack() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})
    labels = np.zeros((2, 8, 8), dtype=np.int32)
    labels[0, 1:4, 1:4] = 1
    labels[1, 4:7, 4:7] = 2
    payload = ObjectLabelPayload(labels=labels)

    out = materialize(
        MaterializationSpec(ROIOptions(min_area=0)),
        data=payload,
        path="/tmp/A01_Nuclei_step3.roi.zip",
        filemanager=fm,
        backends=["memory"],
        backend_kwargs={},
    )

    rois = fm.load(out, "memory")
    assert out == "/tmp/A01_Nuclei_step3_rois.roi.zip"
    assert [roi.metadata["label"] for roi in rois] == [1, 2]
    assert [roi.metadata["plane_indices"] for roi in rois] == [(0,), (1,)]
    assert all(roi.metadata["plane_shape"] == (2,) for roi in rois)


@pytest.mark.unit
def test_roi_materialization_projects_singleton_object_label_stack() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})
    labels = np.zeros((1, 8, 8), dtype=np.int32)
    labels[0, 2:6, 3:7] = 1
    payload = ObjectLabelPayload(
        labels=labels,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            component_metadata=({"well": "A01", "site": 1, "channel": 1},),
        ),
    )

    out = materialize(
        MaterializationSpec(ROIOptions(min_area=0)),
        data=payload,
        path="/tmp/A01_Nuclei_step3.roi.zip",
        filemanager=fm,
        backends=["memory"],
        backend_kwargs={},
    )

    rois = fm.load(out, "memory")
    assert out == "/tmp/A01_Nuclei_step3_rois.roi.zip"
    assert [roi.metadata["label"] for roi in rois] == [1]


@pytest.mark.unit
def test_roi_streaming_preserves_singleton_projected_source_metadata() -> None:
    fm = _RecordingFileManager()
    labels = np.zeros((1, 8, 8), dtype=np.int32)
    labels[0, 2:6, 3:7] = 1
    payload = ObjectLabelPayload(
        labels=labels,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=("/input/A01_s001_w1_z001_t001.tif",),
            component_metadata=({"well": "A01", "site": 1, "channel": 1},),
        ),
    )

    out = _stream_materialize(
        MaterializationSpec(ROIOptions(min_area=0)),
        payload,
        "/tmp/A01_s001_w1_z001_t001_Nuclei_step3.roi.zip",
        fm,
    )

    roi_saves = [
        item
        for item in fm.saved
        if item[1].endswith(".roi.zip")
    ]
    assert out == "/tmp/A01_s001_w1_z001_t001_Nuclei_step3_rois.roi.zip"
    assert [item[1] for item in roi_saves] == [
        "/tmp/A01_s001_w1_z001_t001_Nuclei_step3_rois.roi.zip",
    ]
    assert [_stream_component_metadata(item) for item in roi_saves] == [
        {"well": "A01", "site": 1, "channel": 1},
    ]


@pytest.mark.unit
def test_roi_materialization_splits_addressable_label_planes_for_streaming() -> None:
    class _RecordingBackend:
        requires_filesystem_validation = False

    class _RecordingFileManager:
        def __init__(self):
            self.saved = []

        def _get_backend(self, _backend):
            return _RecordingBackend()

        def save(self, content, path, backend, **kwargs):
            self.saved.append((content, path, backend, kwargs))

    fm = _RecordingFileManager()
    labels = np.zeros((2, 8, 8), dtype=np.int32)
    labels[0, 1:4, 1:4] = 1
    labels[1, 4:7, 4:7] = 2
    payload = ObjectLabelPayload(
        labels=labels,
        source_image_provenance_planes = SourceImageProvenancePlanes.from_components(paths = (
            "/input/A01_s001_w1_z001_t001.tif",
            "/input/A01_s002_w1_z001_t001.tif",
        ), component_metadata = (
            {"well": "A01", "site": 1, "channel": 1},
            {"well": "A01", "site": 2, "channel": 1},
        )))

    out = materialize(
        MaterializationSpec(ROIOptions(min_area=0)),
        data=payload,
        path="/tmp/A01_s001_w1_z001_t001_Nuclei_step3.roi.zip",
        filemanager=fm,
        backends=["napari_stream"],
        backend_kwargs={"napari_stream": _viewer_stream_backend_kwargs()},
    )

    roi_saves = [
        item
        for item in fm.saved
        if item[1].endswith(".roi.zip")
    ]
    assert out == "/tmp/A01_s001_w1_z001_t001_Nuclei_step3_rois.roi.zip"
    assert [item[1] for item in roi_saves] == [
        "/tmp/A01_s001_w1_z001_t001_Nuclei_step3_rois.roi.zip",
        "/tmp/A01_s002_w1_z001_t001_Nuclei_step3_rois.roi.zip",
    ]
    assert [_stream_component_metadata(item) for item in roi_saves] == [
        {"well": "A01", "site": 1, "channel": 1},
        {"well": "A01", "site": 2, "channel": 1},
    ]
    assert [len(item[0]) for item in roi_saves] == [1, 1]


@pytest.mark.unit
def test_roi_materialization_replaces_parser_equivalent_reference_source_prefix() -> None:
    fm = _RecordingFileManager()
    labels = np.zeros((2, 8, 8), dtype=np.int32)
    labels[0, 1:4, 1:4] = 1
    labels[1, 4:7, 4:7] = 2
    payload = ObjectLabelPayload(
        labels=labels,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/images_Illum-corrected/plate1_A14_site1_Ch1.tif",
                "/input/images_Illum-corrected/plate1_A14_site2_Ch1.tif",
            ),
            component_metadata=(
                {
                    "well": "A14",
                    "site": 1,
                    "channel": 1,
                    "z_index": 1,
                    "timepoint": 1,
                    "extension": ".tif",
                },
                {
                    "well": "A14",
                    "site": 2,
                    "channel": 1,
                    "z_index": 1,
                    "timepoint": 1,
                    "extension": ".tif",
                },
            ),
        ),
    )

    out = _stream_materialize(
        MaterializationSpec(ROIOptions(min_area=0)),
        payload,
        "/tmp/A14_s001_w1_z001_t001_Nuclei_step0.roi.zip",
        fm,
        context=_SourceSchemaProcessingContext(),
    )

    roi_saves = [
        item
        for item in fm.saved
        if item[1].endswith(".roi.zip")
    ]
    assert out == "/tmp/plate1_A14_site1_Ch1_Nuclei_step0_rois.roi.zip"
    assert [item[1] for item in roi_saves] == [
        "/tmp/plate1_A14_site1_Ch1_Nuclei_step0_rois.roi.zip",
        "/tmp/plate1_A14_site2_Ch1_Nuclei_step0_rois.roi.zip",
    ]
    assert [_stream_component_metadata(item) for item in roi_saves] == [
        {
            "well": "A14",
            "site": 1,
            "channel": 1,
            "z_index": 1,
            "timepoint": 1,
            "extension": ".tif",
        },
        {
            "well": "A14",
            "site": 2,
            "channel": 1,
            "z_index": 1,
            "timepoint": 1,
            "extension": ".tif",
        },
    ]


@pytest.mark.unit
def test_roi_streaming_applies_target_metadata_without_scalar_stream_metadata() -> None:
    class _RecordingBackend:
        requires_filesystem_validation = False

    class _RecordingFileManager:
        def __init__(self):
            self.saved = []

        def _get_backend(self, _backend):
            return _RecordingBackend()

        def save(self, content, path, backend, **kwargs):
            self.saved.append((content, path, backend, kwargs))

    fm = _RecordingFileManager()
    labels = np.zeros((2, 8, 8), dtype=np.int32)
    labels[0, 1:4, 1:4] = 1
    labels[1, 4:7, 4:7] = 2
    payload = ObjectLabelPayload(
        labels=labels,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w1_z001_t001.tif",
                "/input/A01_s002_w1_z001_t001.tif",
            ),
            component_metadata=(
                {"well": "A01", "site": 1, "channel": 1},
                {"well": "A01", "site": 2, "channel": 1},
            ),
        ),
    )

    materialize(
        MaterializationSpec(ROIOptions(min_area=0)),
        data=payload,
        path="/tmp/A01_s001_w1_z001_t001_Nuclei_step3.roi.zip",
        filemanager=fm,
        backends=["napari_stream"],
        backend_kwargs={"napari_stream": _viewer_stream_backend_kwargs()},
    )

    roi_saves = [
        item
        for item in fm.saved
        if item[1].endswith(".roi.zip")
    ]
    assert [_stream_component_metadata(item) for item in roi_saves] == [
        {"well": "A01", "site": 1, "channel": 1},
        {"well": "A01", "site": 2, "channel": 1},
    ]


@pytest.mark.unit
def test_roi_materialization_coalesces_duplicate_stream_targets() -> None:
    class _RecordingBackend:
        requires_filesystem_validation = False

    class _RecordingFileManager:
        def __init__(self):
            self.saved = []

        def _get_backend(self, _backend):
            return _RecordingBackend()

        def save(self, content, path, backend, **kwargs):
            self.saved.append((content, path, backend, kwargs))

    fm = _RecordingFileManager()
    labels = np.zeros((2, 8, 8), dtype=np.int32)
    labels[0, 1:4, 1:4] = 1
    labels[1, 4:7, 4:7] = 2
    payload = ObjectLabelPayload(
        labels=labels,
        source_image_provenance_planes = SourceImageProvenancePlanes.from_components(paths = (
            "/input/A01_s001_w1_z001_t001.tif",
            "/input/A01_s001_w1_z001_t001.tif",
        ), component_metadata = (
            {"well": "A01", "site": 1, "channel": 1},
            {"well": "A01", "site": 1, "channel": 1},
        )))

    materialize(
        MaterializationSpec(ROIOptions(min_area=0)),
        data=payload,
        path="/tmp/A01_s001_w1_z001_t001_Nuclei_step3.roi.zip",
        filemanager=fm,
        backends=["napari_stream"],
        backend_kwargs={"napari_stream": _viewer_stream_backend_kwargs()},
    )

    roi_saves = [
        item
        for item in fm.saved
        if item[1].endswith(".roi.zip")
    ]
    assert [item[1] for item in roi_saves] == [
        "/tmp/A01_s001_w1_z001_t001_Nuclei_step3_rois.roi.zip",
    ]
    assert _stream_component_metadata(roi_saves[0]) == {
        "well": "A01",
        "site": 1,
        "channel": 1,
    }
    assert len(roi_saves[0][0]) == 2


@pytest.mark.unit
def test_roi_materialization_coalesces_source_stack_archive_with_artifact_identity() -> None:
    fm = _RecordingFileManager()
    labels = np.zeros((2, 8, 8), dtype=np.int32)
    labels[0, 1:4, 1:4] = 1
    labels[1, 4:7, 4:7] = 2
    payload = ObjectLabelPayload(
        labels=labels,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w1_z001_t001.tif",
                "/input/A01_s001_w1_z001_t001.tif",
            ),
            component_metadata=(
                {
                    "well": "A01",
                    "site": 1,
                    "channel": 1,
                    "z_index": 1,
                    "timepoint": 1,
                    "extension": ".tif",
                },
                {
                    "well": "A01",
                    "site": 1,
                    "channel": 1,
                    "z_index": 2,
                    "timepoint": 1,
                    "extension": ".tif",
                },
            ),
        ),
    )

    materialize(
        MaterializationSpec(ROIOptions(min_area=0)),
        data=payload,
        path="/tmp/A01_s001_w1_z001_t001_Nuclei_step3.roi.zip",
        filemanager=fm,
        backends=["napari_stream"],
        backend_kwargs={"napari_stream": _viewer_stream_backend_kwargs()},
        artifact_source_identity=SourceImageIdentity(
            component_metadata={
                "well": "A01",
                "site": 1,
                "channel": 1,
                "timepoint": 1,
                "extension": ".tif",
            },
        ),
    )

    roi_saves = [
        item
        for item in fm.saved
        if item[1].endswith(".roi.zip")
    ]
    assert [item[1] for item in roi_saves] == [
        "/tmp/A01_s001_w1_z001_t001_Nuclei_step3_rois.roi.zip",
    ]
    assert _stream_component_metadata(roi_saves[0]) == {
        "well": "A01",
        "site": 1,
        "channel": 1,
        "timepoint": 1,
        "extension": ".tif",
    }
    assert len(roi_saves[0][0]) == 2


@pytest.mark.unit
def test_roi_materialization_coalesces_same_stream_address_from_distinct_sources() -> None:
    roi_path = "/tmp/plate1_A14_site1_Ch1_Nuclei_step0_rois.roi.zip"
    coalesced = ROIMaterializationTargetCoalescer().coalesce(
        (
            _roi_target(
                roi_path,
                np.zeros((8, 8), dtype=np.int32),
                ImagePayloadMetadata(
                    source_path="/input/source_a.tif",
                    source_component_metadata={
                        "well": "A14",
                        "site": 1,
                        "channel": 1,
                    },
                ),
            ),
            _roi_target(
                roi_path,
                np.ones((8, 8), dtype=np.int32),
                ImagePayloadMetadata(
                    source_path="/input/source_b.tif",
                    source_component_metadata={
                        "well": "A14",
                        "site": 1,
                        "channel": 1,
                    },
                ),
            ),
        )
    )

    assert len(coalesced) == 1
    assert coalesced[0].archive.path == roi_path
    assert len(coalesced[0].items) == 2


@pytest.mark.unit
def test_roi_materialization_coalesces_localized_duplicate_source_stems() -> None:
    roi_path = "/tmp/plate1_A14_site1_Ch1_Nuclei_step0_rois.roi.zip"
    coalesced = ROIMaterializationTargetCoalescer().coalesce(
        (
            _roi_target(
                roi_path,
                np.zeros((8, 8), dtype=np.int32),
                ImagePayloadMetadata(
                    source_path="/tutorial/images_Illum-corrected/plate1_A14_site1_Ch1.tif",
                    source_component_metadata={
                        "well": "A14",
                        "site": 1,
                        "channel": 1,
                    },
                ),
            ),
            _roi_target(
                roi_path,
                np.ones((8, 8), dtype=np.int32),
                ImagePayloadMetadata(
                    source_path="/tutorial/Archive_EN/images_Illum-corrected/plate1_A14_site1_Ch1.tif",
                ),
            ),
        )
    )

    assert len(coalesced) == 1
    assert coalesced[0].archive.path == roi_path
    assert len(coalesced[0].items) == 2
    assert dict(coalesced[0].archive.source_component_metadata) == {
        "well": "A14",
        "site": 1,
        "channel": 1,
    }


@pytest.mark.unit
def test_roi_materialization_rejects_component_conflict_after_path_only_target() -> None:
    roi_path = "/tmp/plate1_A14_site1_Ch1_Nuclei_step0_rois.roi.zip"
    with pytest.raises(ValueError, match="conflicting stream identities"):
        ROIMaterializationTargetCoalescer().coalesce(
            (
                _roi_target(
                    roi_path,
                    np.zeros((8, 8), dtype=np.int32),
                    ImagePayloadMetadata(
                        source_path="/tutorial/Archive_EN/images_Illum-corrected/plate1_A14_site1_Ch1.tif",
                    ),
                ),
                _roi_target(
                    roi_path,
                    np.ones((8, 8), dtype=np.int32),
                    ImagePayloadMetadata(
                        source_path="/tutorial/images_Illum-corrected/plate1_A14_site1_Ch1.tif",
                        source_component_metadata={
                            "well": "A14",
                            "site": 1,
                            "channel": 1,
                        },
                    ),
                ),
                _roi_target(
                    roi_path,
                    np.full((8, 8), 2, dtype=np.int32),
                    ImagePayloadMetadata(
                        source_path="/tutorial/images_Illum-corrected/plate1_A14_site2_Ch1.tif",
                        source_component_metadata={
                            "well": "A14",
                            "site": 2,
                            "channel": 1,
                        },
                    ),
                ),
            )
        )


@pytest.mark.unit
def test_roi_materialization_rejects_conflicting_duplicate_archive_identity() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})
    payload = _addressable_roi_label_payload(
        "/input/A01_s001_w1_z001_t001.tif",
        "/input/A01_s001_w1_z001_t001.tif",
        {"well": "A01", "site": 1, "channel": 1},
        {"well": "A01", "site": 2, "channel": 1},
    )

    with pytest.raises(ValueError, match="conflicting stream identities"):
        _memory_materialize(
            MaterializationSpec(ROIOptions(min_area=0)),
            payload,
            "/tmp/A01_s001_w1_z001_t001_Nuclei_step3.roi.zip",
            fm,
        )


@pytest.mark.unit
def test_roi_materialization_uses_source_context_for_partial_label_stack() -> None:
    class _RecordingBackend:
        requires_filesystem_validation = False

    class _RecordingFileManager:
        def __init__(self):
            self.saved = []

        def _get_backend(self, _backend):
            return _RecordingBackend()

        def save(self, content, path, backend, **kwargs):
            self.saved.append((content, path, backend, kwargs))

    fm = _RecordingFileManager()
    source_image = ImageMetadataPayload(
        data=np.zeros((2, 8, 8), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes = SourceImageProvenancePlanes.from_components(paths = (
                "/input/A02_s001_w1_z001_t001.tif",
                "/input/A02_s002_w1_z001_t001.tif",
            ), component_metadata = (
                {"well": "A02", "site": 1, "channel": 1},
                {"well": "A02", "site": 2, "channel": 1},
            ))),
    )
    payload = _unaddressed_roi_label_payload().with_source_image_context(source_image)

    _stream_materialize(
        MaterializationSpec(ROIOptions(min_area=0)),
        payload,
        "/tmp/A02_s001_w1_z001_t001_Nuclei_step3.roi.zip",
        fm,
    )

    roi_saves = [
        item
        for item in fm.saved
        if item[1].endswith(".roi.zip")
    ]
    assert [item[1] for item in roi_saves] == [
        "/tmp/A02_s001_w1_z001_t001_Nuclei_step3_rois.roi.zip",
        "/tmp/A02_s002_w1_z001_t001_Nuclei_step3_rois.roi.zip",
    ]
    assert [_stream_component_metadata(item) for item in roi_saves] == [
        {"well": "A02", "site": 1, "channel": 1},
        {"well": "A02", "site": 2, "channel": 1},
    ]


@pytest.mark.unit
def test_roi_materialization_rejects_unaddressed_stack_provenance_slots() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})
    payload = _unaddressed_roi_label_payload()

    with pytest.raises(ValueError, match="per-plane source identity"):
        _memory_materialize(
            MaterializationSpec(ROIOptions(min_area=0)),
            payload,
            "/tmp/A01_s001_w1_z001_t001_Nuclei_step3.roi.zip",
            fm,
        )


@pytest.mark.unit
def test_roi_materialization_treats_non_spatial_label_payload_as_empty() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})
    payload = ObjectLabelPayload(labels=np.asarray(0, dtype=np.int32))

    out = _memory_materialize(
        MaterializationSpec(ROIOptions(min_area=0)),
        payload,
        "/tmp/A01_Worms_step3.roi.zip",
        fm,
    )

    assert out == "/tmp/A01_Worms_step3_segmentation_summary.txt"
    assert "No ROIs extracted" in fm.load(out, "memory")
