import json

import numpy as np
import pytest

import openhcs  # noqa: F401
from polystore.filemanager import FileManager
from polystore.memory import MemoryStorageBackend
from polystore.napari_stream import NapariStreamingBackend
from polystore.fiji_stream import FijiStreamingBackend
from polystore.streaming.identity import (
    FixedStreamProducerIdentityKind,
    StreamProducerIdentity,
)
from polystore.streaming import (
    StreamingBatchMessageBuilder,
    StreamingBatchMessageRequest,
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

from openhcs.constants.constants import AllComponents, VariableComponents
from openhcs.core.config import TransportMode
from openhcs.core.measurement_row_materialization import (
    MeasurementProjectedColumnarRows,
)
from openhcs.core.runtime_object_label_domains import ObjectLabelDomain, ObjectLabelDomainScope
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.core.runtime_slice_projection import RuntimeProjectionPlaneMetadata
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser
from openhcs.processing.materialization import (
    ImageFileOptions,
    JsonOptions,
    MaterializedFilenameIdentity,
    MaterializationSpec,
    ROIOptions,
    TiffStackOptions,
    csv_only,
    json_materializer,
    json_only,
    materialize,
    tabular_field_names_from_materialization,
    text_only,
    tiff_stack,
)
from openhcs.processing.materialization.core import (
    MaterializationInputItem,
    Output,
    ROIMaterializationArchiveIdentity,
    ROIMaterializationTarget,
    ROIMaterializationTargetCoalescer,
    RuntimePlaneStackAxesProjectionSelection,
    ViewerStreamBackendCallKwargs,
)
from openhcs.core.runtime_image_values import (
    ImageMetadataPayload,
    ImagePayloadMetadata,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelVariantData,
    ObjectLabelPayload,
)
from openhcs.core.source_image_provenance import (
    SourceImageProvenancePlanes,
)
from openhcs.core.source_metadata import (
    SOURCE_PLANE_COUNT_FIELD,
    SOURCE_PLANE_INDEX_FIELD,
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
    COMPONENT_ORDER = AllComponents.ordered_names()
    variable_size_handling = None
    auto_contrast = True

    def component_modes(self):
        return {}

    def get_colormap_name(self):
        return "gray"

    def get_lut_name(self):
        return "Grays"


class _TestViewerFilenameParser(ViewerFilenameParserABC):
    def parse_filename(self, filename):
        import re

        match = re.match(
            r"(?P<well>[A-Z]\d{2})_s(?P<site>\d+)_w(?P<channel>\d+)"
            r"(?:_z(?P<z_index>\d+))?(?:_t(?P<timepoint>\d+))?"
            r"(?P<extension>\.[^.]+)?$",
            filename,
        )
        if match is None:
            return None
        parsed = {
            key: value for key, value in match.groupdict().items() if value is not None
        }
        parsed.setdefault("z_index", "1")
        parsed.setdefault("timepoint", "1")
        return parsed


class _TestViewerMetadataHandler(ViewerMetadataHandlerABC):
    def get_component_values(self, plate_path, component_name):
        return {}


class _TestViewerMicroscopeHandler(ViewerMicroscopeHandlerABC):
    parser = _TestViewerFilenameParser()
    metadata_handler = _TestViewerMetadataHandler()


def _viewer_stream_backend_kwargs():
    display_config = _TestViewerDisplayConfig()
    request = ViewerStreamRequest(
        viewer_transport=ViewerTransportEndpoint(
            host="localhost",
            port=5555,
            transport_mode=TransportMode.IPC,
        ),
        display_config=display_config,
        source=ViewerStreamSource(
            identity=ViewerStreamSourceIdentity(
                microscope_handler=_TestViewerMicroscopeHandler(),
                plate_path="/tmp/test_plate",
            ),
            metadata=BatchViewerStreamSourceMetadata(
                {
                    "timepoint": 1,
                    "z_index": 1,
                    "site": 1,
                    "well": "A01",
                    "channel": 1,
                }
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


def _viewer_stream_backend_kwargs_for_display(display_config):
    request = ViewerStreamRequest(
        viewer_transport=ViewerTransportEndpoint(
            host="localhost",
            port=5555,
            transport_mode=TransportMode.IPC,
        ),
        display_config=display_config,
        source=ViewerStreamSource(
            identity=ViewerStreamSourceIdentity(
                microscope_handler=_TestViewerMicroscopeHandler(),
                plate_path="/tmp/test_plate",
            ),
            metadata=BatchViewerStreamSourceMetadata(
                {
                    "timepoint": 1,
                    "z_index": 1,
                    "site": 1,
                    "well": "A01",
                    "channel": 1,
                }
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
    return stream_request.source.metadata.component_metadata_for_item(
        saved_item[1],
        0,
    )


def test_viewer_backend_batches_distinct_timepoints_by_exact_output_path() -> None:
    backend_kwargs = _viewer_stream_backend_kwargs()
    outputs = tuple(
        Output(
            path=f"/tmp/DrosophilaEmbryo-{timepoint:04d}.png",
            content=np.zeros((1, 8, 8, 3), dtype=np.uint8),
            metadata=ImagePayloadMetadata(
                source_path=f"/input/Sequence1_t{timepoint:03d}.tif",
                source_component_metadata={
                    "well": "Sequence1",
                    "site": 1,
                    "channel": 1,
                    "z_index": 1,
                    "timepoint": timepoint,
                },
                source_spatial_domain=SourceSpatialDomain(source_shape_yx=(8, 8)),
            ),
        )
        for timepoint in range(3)
    )

    batches = backend_kwargs.filemanager_batches(outputs)

    assert len(batches) == 1
    batch_outputs, kwargs = batches[0]
    assert batch_outputs == outputs
    stream_request = kwargs["stream_request"]
    assert tuple(
        stream_request.source.metadata.component_metadata_for_item(
            output.path,
            index,
        )["timepoint"]
        for index, output in enumerate(outputs)
    ) == (0, 1, 2)


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


@pytest.mark.unit
def test_materialization_spec_candidate_paths_follow_registered_writers() -> None:
    assert csv_only(suffix=".csv").candidate_paths("/tmp/A01_measurements.pkl") == (
        "/tmp/A01_measurements.csv",
    )
    assert json_only().candidate_paths("/tmp/metadata.json") == ("/tmp/metadata.json",)
    assert text_only().candidate_paths("/tmp/report.txt") == ("/tmp/report.txt",)
    assert MaterializationSpec(ROIOptions()).candidate_paths(
        "/tmp/A01_s001_w1_z001_t001_Nuclei_step3.roi.zip"
    ) == (
        "/tmp/A01_s001_w1_z001_t001_Nuclei_step3_rois.roi.zip",
        "/tmp/A01_s001_w1_z001_t001_Nuclei_step3_segmentation_summary.txt",
    )
    assert tiff_stack().candidate_paths("/tmp/Overlay.pkl") == (
        "/tmp/Overlay_slice_000.tif",
        "/tmp/Overlay_summary.txt",
    )
    assert tiff_stack().candidate_paths("/tmp/A01_s001_w1_z001_t001.tif") == (
        "/tmp/A01_s001_w1_z001_t001.tif",
        "/tmp/A01_s001_w1_z001_t001_summary.txt",
    )


@pytest.mark.unit
def test_materialization_spec_declares_filename_identity() -> None:
    assert tiff_stack().uses_source_identity_filename()
    assert not tiff_stack(
        TiffStackOptions(filename_identity=MaterializedFilenameIdentity.ARTIFACT_NAME)
    ).uses_source_identity_filename()


@pytest.mark.parametrize(
    ("component", "component_values"),
    (
        (VariableComponents.TIMEPOINT, (0, 1)),
        (VariableComponents.Z_INDEX, (1, 2)),
    ),
)
def test_indexed_image_materialization_streams_each_declared_component_plane(
    component,
    component_values,
) -> None:
    data = np.zeros((2, 1, 8, 8, 3), dtype=np.uint8)
    payload = ImageMetadataPayload(
        data=data,
        metadata=ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            source_channel_axis=-1,
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(
                    component_metadata=tuple(
                        {
                            "well": "A01",
                            "site": 1,
                            "channel": 1,
                            "z_index": 1,
                            "timepoint": 1,
                            component.value: value,
                        }
                        for value in component_values
                    )
                )
            ),
        ),
    )
    spec = MaterializationSpec(
        ImageFileOptions(
            filename_identity=MaterializedFilenameIdentity.ARTIFACT_NAME,
            relative_path_template="frame-{index:04d}.png",
        )
    )
    assert spec.emits_variable_component_planes(payload)

    filemanager = _RecordingFileManager()
    materialize(
        spec,
        data=payload,
        path="/tmp/SavedImage",
        filemanager=filemanager,
        backends=["napari_stream"],
        backend_kwargs={"napari_stream": _viewer_stream_backend_kwargs()},
        variable_components=(component,),
    )

    assert len(filemanager.saved) == 2
    streamed_components = []
    for _content, path, _backend, kwargs in filemanager.saved:
        request = kwargs["stream_request"]
        streamed_components.append(
            dict(request.source.metadata.component_metadata_for_item(path, 0))[
                component.value
            ]
        )
    assert tuple(streamed_components) == component_values


def _two_plane_roi_labels():
    labels = np.zeros((2, 8, 8), dtype=np.int32)
    labels[0, 1:4, 1:4] = 1
    labels[1, 4:7, 4:7] = 2
    return labels


class _RecordingBackend:
    requires_filesystem_validation = False

    def supports_file_path(self, _path):
        return True


class _RecordingFileManager:
    def __init__(self):
        self.saved = []

    def _get_backend(self, _backend):
        return _RecordingBackend()

    def save(self, content, path, backend, **kwargs):
        self.saved.append((content, path, backend, kwargs))

    def save_batch(self, contents, paths, backend, **kwargs):
        self.saved.extend(
            (content, path, backend, kwargs)
            for content, path in zip(contents, paths, strict=True)
        )


class _SourceSchemaMicroscopeHandler:
    parser = SourceSchemaFilenameParser()


class _SourceSchemaProcessingContext:
    microscope_handler = _SourceSchemaMicroscopeHandler()


def _addressable_roi_label_payload(
    first_path, second_path, first_metadata, second_metadata
):
    return ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=_two_plane_roi_labels()),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(first_path, second_path),
            component_metadata=(first_metadata, second_metadata),
        ),
    )


def _unaddressed_roi_label_payload():
    return ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=_two_plane_roi_labels()),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(None, None), component_metadata=(None, None)
        ),
    )


def _roi_target(roi_path, labels, metadata):
    return ROIMaterializationTarget(
        archive=ROIMaterializationArchiveIdentity.from_metadata(
            path=roi_path,
            metadata=metadata,
        ),
        items=(
            MaterializationInputItem(
                value=ObjectLabelPayload(
                    variant_data=ObjectLabelVariantData(labels=labels)
                ),
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
            },
            fields=(
                FieldSpec("object_label", int),
                FieldSpec("area", float),
            ),
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
def test_csv_materialization_expands_mapping_columns() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})

    out = materialize(
        csv_only(
            fields=[
                "relationship_type",
                "parent_id",
                "child_id",
                "slice_index",
                "slice_count",
            ]
        ),
        data={
            "relationship_type": "parent_child",
            "parent_id": (10, 11),
            "child_id": (1, 2),
            "slice_count": 2,
            "source_component_metadata": {"channel": "1"},
        },
        path="/tmp/A01_relationships",
        filemanager=fm,
        backends=["memory"],
        backend_kwargs={},
    )

    assert out == "/tmp/A01_relationships_details.csv"
    assert fm.load(out, "memory").splitlines() == [
        "relationship_type,parent_id,child_id,slice_index,slice_count",
        "parent_child,10,1,0,2",
        "parent_child,11,2,1,2",
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
    stack = ImageMetadataPayload(
        data=np.zeros((3, 5, 7), dtype=np.uint8),
        metadata=ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ),
    )

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

        def supports_file_path(self, _path):
            return True

    class _RecordingFileManager:
        def __init__(self):
            self.saved = []

        def _get_backend(self, _backend):
            return _RecordingBackend()

        def save(self, content, path, backend, **kwargs):
            self.saved.append((content, path, backend, kwargs))

        def save_batch(self, contents, paths, backend, **kwargs):
            self.saved.extend(
                (content, path, backend, kwargs)
                for content, path in zip(contents, paths, strict=True)
            )

    fm = _RecordingFileManager()
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((2, 5, 7), dtype=np.int32)),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        domain=ObjectLabelDomain(
            scope=ObjectLabelDomainScope.PLANE,
            declared_object_id_domains=((), ()),
        ),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            component_metadata=(
                {"well": "A01", "site": 1, "z_index": 1, "channel": 1, "timepoint": 1},
                {"well": "A01", "site": 2, "z_index": 1, "channel": 1, "timepoint": 1},
            )
        ),
    )

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
        {"timepoint": 1, "z_index": 1, "site": 1, "well": "A01", "channel": 1},
        {"timepoint": 1, "z_index": 1, "site": 2, "well": "A01", "channel": 1},
    ]


@pytest.mark.unit
def test_tiff_stack_streaming_projects_scalar_metadata_over_declared_stack_axis() -> (
    None
):
    fm = _RecordingFileManager()
    payload = ImageMetadataPayload(
        data=np.zeros((2, 5, 7), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            source_component_metadata={
                "well": "A01",
                "site": 1,
                "z_index": 1,
                "channel": 3,
                "timepoint": 1,
            },
        ),
    )

    materialize(
        tiff_stack(),
        data=payload,
        path="/tmp/A01_s001_w3_z001_t001_DerivedStack_step6",
        filemanager=fm,
        backends=["napari_stream"],
        backend_kwargs={"napari_stream": _viewer_stream_backend_kwargs()},
        variable_components=(VariableComponents.Z_INDEX,),
    )

    saved_images = [
        item
        for item in fm.saved
        if item[1].endswith(("_slice_000.tif", "_slice_001.tif"))
    ]
    assert [_stream_component_metadata(item) for item in saved_images] == [
        {"timepoint": 1, "z_index": 1, "site": 1, "well": "A01", "channel": 3},
        {"timepoint": 1, "z_index": 2, "site": 1, "well": "A01", "channel": 3},
    ]


@pytest.mark.unit
def test_tiff_stack_projection_skips_axes_already_varying_in_slice_metadata() -> None:
    items = tuple(
        MaterializationInputItem(
            value=ImagePayloadMetadata(
                source_component_metadata={
                    AllComponents.WELL.value: "A01",
                    AllComponents.SITE.value: 1,
                    AllComponents.Z_INDEX.value: index + 1,
                    AllComponents.CHANNEL.value: 3,
                    AllComponents.TIMEPOINT.value: 1,
                },
            ).payload_with(np.zeros((5, 7), dtype=np.float32), None),
            source_description="materialization payload",
            runtime_plane_metadata=RuntimeProjectionPlaneMetadata(
                plane_indices=(index,),
                plane_shape=(3,),
            ),
        )
        for index in range(3)
    )

    axes = RuntimePlaneStackAxesProjectionSelection(
        (VariableComponents.Z_INDEX, VariableComponents.CHANNEL),
        items,
    ).axes()

    assert axes == frozenset()


@pytest.mark.unit
def test_tiff_stack_projection_rejects_ambiguous_scalar_declared_axes() -> None:
    items = tuple(
        MaterializationInputItem(
            value=ImagePayloadMetadata(
                source_component_metadata={
                    AllComponents.WELL.value: "A01",
                    AllComponents.SITE.value: 1,
                    AllComponents.CHANNEL.value: 3,
                },
            ).payload_with(np.zeros((5, 7), dtype=np.float32), None),
            source_description="materialization payload",
            runtime_plane_metadata=RuntimeProjectionPlaneMetadata(
                plane_indices=(index,),
                plane_shape=(3,),
            ),
        )
        for index in range(3)
    )

    with pytest.raises(ValueError, match="cannot map runtime plane metadata"):
        RuntimePlaneStackAxesProjectionSelection(
            (VariableComponents.Z_INDEX, VariableComponents.CHANNEL),
            items,
        ).axes()


@pytest.mark.unit
def test_tiff_stack_projection_uses_artifact_scalar_z_origin() -> None:
    items = tuple(
        MaterializationInputItem(
            value=np.zeros((5, 7), dtype=np.float32),
            source_description="materialization payload",
            runtime_plane_metadata=RuntimeProjectionPlaneMetadata(
                plane_indices=(index,),
                plane_shape=(3,),
            ),
        )
        for index in range(3)
    )

    axes = RuntimePlaneStackAxesProjectionSelection(
        (VariableComponents.Z_INDEX, VariableComponents.CHANNEL),
        items,
        SourceImageIdentity(
            component_metadata={
                AllComponents.WELL.value: "A01",
                AllComponents.SITE.value: 1,
                AllComponents.Z_INDEX.value: 1,
                AllComponents.CHANNEL.value: 3,
            },
        ),
    ).axes()

    assert axes == frozenset((AllComponents.Z_INDEX.value,))


@pytest.mark.unit
def test_tiff_stack_streaming_preserves_runtime_source_plane_metadata() -> None:
    fm = _RecordingFileManager()
    payload = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_path="/tmp/A01_s001_w3_z001_t001.tif",
        source_component_metadata={
            "well": "A01",
            "site": 1,
            "z_index": 1,
            "channel": 3,
            "timepoint": 1,
            SOURCE_PLANE_INDEX_FIELD: "0",
            SOURCE_PLANE_COUNT_FIELD: "2",
        },
    ).payload_with(np.zeros((2, 5, 7), dtype=np.float32), None)

    materialize(
        tiff_stack(),
        data=payload,
        path="/tmp/A01_s001_w3_z001_t001_DerivedStack_step6",
        filemanager=fm,
        backends=["napari_stream"],
        backend_kwargs={"napari_stream": _viewer_stream_backend_kwargs()},
        variable_components=(
            VariableComponents.Z_INDEX,
            VariableComponents.CHANNEL,
        ),
    )

    saved_images = [
        item
        for item in fm.saved
        if item[1].endswith(("_slice_000.tif", "_slice_001.tif"))
    ]
    assert [_stream_component_metadata(item) for item in saved_images] == [
        {"timepoint": 1, "z_index": 1, "site": 1, "well": "A01", "channel": 3},
        {"timepoint": 1, "z_index": 2, "site": 1, "well": "A01", "channel": 3},
    ]


@pytest.mark.unit
def test_tiff_stack_streaming_projects_artifact_identity_over_declared_stack_axis() -> (
    None
):
    fm = _RecordingFileManager()

    materialize(
        tiff_stack(),
        data=ImageMetadataPayload(
            data=np.zeros((2, 5, 7), dtype=np.float32),
            metadata=ImagePayloadMetadata(
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            ),
        ),
        path="/tmp/A01_s001_w3_z001_t001_DerivedStack_step6",
        filemanager=fm,
        backends=["napari_stream"],
        backend_kwargs={"napari_stream": _viewer_stream_backend_kwargs()},
        artifact_source_identity=SourceImageIdentity(
            component_metadata={
                "well": "A01",
                "site": 1,
                "z_index": 1,
                "channel": 3,
                "timepoint": 1,
            },
        ),
        variable_components=(VariableComponents.Z_INDEX,),
    )

    saved_images = [
        item
        for item in fm.saved
        if item[1].endswith(("_slice_000.tif", "_slice_001.tif"))
    ]
    assert [_stream_component_metadata(item) for item in saved_images] == [
        {"timepoint": 1, "z_index": 1, "site": 1, "well": "A01", "channel": 3},
        {"timepoint": 1, "z_index": 2, "site": 1, "well": "A01", "channel": 3},
    ]


@pytest.mark.unit
def test_tiff_stack_streaming_uses_single_plane_scalar_source_identity() -> None:
    fm = _RecordingFileManager()
    payload = ImageMetadataPayload(
        data=np.zeros((5, 7, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                component_metadata=(
                    {
                        "well": "A01",
                        "site": 1,
                        "z_index": 1,
                        "channel": 3,
                        "timepoint": 1,
                    },
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

    saved_images = [item for item in fm.saved if item[1].endswith(".tif")]
    assert len(saved_images) == 1
    assert _stream_component_metadata(saved_images[0]) == {
        "timepoint": 1,
        "z_index": 1,
        "site": 1,
        "well": "A01",
        "channel": 3,
    }


@pytest.mark.unit
def test_materialized_tiff_output_uses_artifact_source_identity_as_fallback() -> None:
    fm = _RecordingFileManager()
    payload = ImageMetadataPayload(
        data=np.zeros((5, 7, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_component_metadata={
                "well": "A01",
                "site": 1,
                "z_index": 1,
                "channel": 3,
            },
        ),
    )

    materialize(
        tiff_stack(),
        data=payload,
        path="/tmp/A01_s001_w3_z001_t020_AdjacentImage_step6",
        filemanager=fm,
        backends=["napari_stream"],
        backend_kwargs={
            "napari_stream": _viewer_stream_backend_kwargs_for_display(
                _TestViewerDisplayConfig()
            )
        },
        artifact_source_identity=SourceImageIdentity(
            component_metadata={
                "well": "A01",
                "site": 1,
                "z_index": 1,
                "channel": 1,
                "timepoint": 20,
            },
        ),
    )

    saved_images = [item for item in fm.saved if item[1].endswith(".tif")]
    assert len(saved_images) == 1
    assert _stream_component_metadata(saved_images[0]) == {
        "well": "A01",
        "site": 1,
        "z_index": 1,
        "channel": 3,
        "timepoint": 20,
    }


@pytest.mark.unit
def test_roi_materialization_offsets_object_label_payload_geometry() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[2:6, 3:7] = 1
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
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
def test_roi_viewer_stream_preserves_source_spatial_domain() -> None:
    fm = _RecordingFileManager()
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[2:6, 3:7] = 1
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        source_component_metadata={
            "well": "A01",
            "site": 1,
            "channel": 1,
            "z_index": 1,
            "timepoint": 1,
        },
        source_spatial_domain=SourceSpatialDomain(
            origin_yx=(10, 20),
            source_shape_yx=(100, 200),
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

    roi_outputs = [item for item in fm.saved if item[1].endswith(".roi.zip")]
    assert len(roi_outputs) == 1
    stream_source = roi_outputs[0][3]["stream_request"].source
    assert stream_source.item_fields == {
        "spatial_origin_yx": [10, 20],
        "source_spatial_shape_yx": [100, 200],
    }


@pytest.mark.unit
def test_roi_materialization_extracts_each_plane_from_object_label_stack() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})
    labels = np.zeros((2, 8, 8), dtype=np.int32)
    labels[0, 1:4, 1:4] = 1
    labels[1, 4:7, 4:7] = 2
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
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
    assert [roi.metadata["label"] for roi in rois] == [1, 2]
    assert [roi.metadata["plane_indices"] for roi in rois] == [(0,), (1,)]
    assert all(roi.metadata["plane_shape"] == (2,) for roi in rois)
    assert all(roi.metadata["spatial_origin_yx"] == (0, 0) for roi in rois)
    assert all(roi.metadata["source_spatial_shape_yx"] == (8, 8) for roi in rois)
    summary = fm.load(
        "/tmp/A01_Nuclei_step3_segmentation_summary.txt",
        "memory",
    )
    assert "Spatial dimensions: 2D" in summary
    assert "Projected source planes: 2" in summary
    assert "Z-planes" not in summary


@pytest.mark.unit
def test_roi_materialization_preserves_payload_scoped_volume_in_one_archive() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=_two_plane_roi_labels())
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
    assert payload.object_label_domain().scope is ObjectLabelDomainScope.PAYLOAD
    assert out == "/tmp/A01_Nuclei_step3_rois.roi.zip"
    assert [roi.metadata["label"] for roi in rois] == [1, 2]
    assert [roi.metadata["plane_indices"] for roi in rois] == [(0,), (1,)]
    assert all(roi.metadata["plane_shape"] == (2,) for roi in rois)


@pytest.mark.unit
def test_roi_streaming_maps_payload_scoped_volume_planes_from_provenance() -> None:
    fm = _RecordingFileManager()
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=_two_plane_roi_labels()),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w1_z001_t001.tif",
                "/input/A01_s001_w1_z002_t001.tif",
            ),
            component_metadata=(
                {
                    "well": "A01",
                    "site": 1,
                    "channel": 1,
                    "z_index": 1,
                    "timepoint": 1,
                },
                {
                    "well": "A01",
                    "site": 1,
                    "channel": 1,
                    "z_index": 2,
                    "timepoint": 1,
                },
            ),
        ),
    )

    materialize(
        MaterializationSpec(ROIOptions(min_area=0)),
        data=payload,
        path="/tmp/A01_Nuclei_step3.roi.zip",
        filemanager=fm,
        backends=["napari_stream"],
        backend_kwargs={"napari_stream": _viewer_stream_backend_kwargs()},
        variable_components=(VariableComponents.Z_INDEX,),
    )

    roi_saves = [item for item in fm.saved if item[1].endswith(".roi.zip")]
    assert len(roi_saves) == 1
    roi_content, roi_path, _backend, stream_kwargs = roi_saves[0]
    stream_request = stream_kwargs["stream_request"]
    item_fields = stream_request.source.item_fields
    assert item_fields["plane_axis"] == RuntimePlaneAxis.RUNTIME_SLICE.value
    assert item_fields["plane_component_values"] == {"z_index": ["1", "2"]}

    napari_backend = NapariStreamingBackend()
    streamed_item = StreamingBatchMessageBuilder.build(
        napari_backend,
        StreamingBatchMessageRequest(
            data_list=[roi_content],
            file_paths=[roi_path],
            stream_request=stream_request,
            component_names_request=napari_backend.component_names_request(
                stream_request
            ),
            display_payload_extra=napari_backend.display_payload_extra(stream_request),
        ),
    ).batch_images[0]
    assert streamed_item["plane_axis"] == RuntimePlaneAxis.RUNTIME_SLICE.value
    assert streamed_item["plane_component_values"] == {"z_index": ["1", "2"]}


@pytest.mark.unit
def test_roi_materialization_projects_singleton_object_label_stack() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})
    labels = np.zeros((1, 8, 8), dtype=np.int32)
    labels[0, 2:6, 3:7] = 1
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,),),
            scope=ObjectLabelDomainScope.PLANE,
        ),
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
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,),),
            scope=ObjectLabelDomainScope.PLANE,
        ),
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

    roi_saves = [item for item in fm.saved if item[1].endswith(".roi.zip")]
    assert out == "/tmp/A01_s001_w1_z001_t001_Nuclei_step3_rois.roi.zip"
    assert [item[1] for item in roi_saves] == [
        "/tmp/A01_s001_w1_z001_t001_Nuclei_step3_rois.roi.zip",
    ]
    assert [_stream_component_metadata(item) for item in roi_saves] == [
        {"timepoint": 1, "z_index": 1, "site": 1, "well": "A01", "channel": 1},
    ]


@pytest.mark.unit
def test_roi_streaming_maps_singleton_plane_from_exact_output_component() -> None:
    fm = _RecordingFileManager()
    labels = np.zeros((1, 8, 8), dtype=np.int32)
    labels[0, 2:6, 3:7] = 1
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        source_path="/input/A01_s001_w1_z001_t001.tif",
        source_component_metadata={
            "site": 1,
            "channel": 1,
            "z_index": 1,
            "timepoint": 1,
            "well": "A01",
        },
    )
    assert payload.object_label_domain().scope is ObjectLabelDomainScope.PAYLOAD

    materialize(
        MaterializationSpec(ROIOptions(min_area=0)),
        data=payload,
        path="/tmp/A01_s001_w1_z001_t001_Nuclei_step3.roi.zip",
        filemanager=fm,
        backends=["napari_stream"],
        backend_kwargs={"napari_stream": _viewer_stream_backend_kwargs()},
        variable_components=(VariableComponents.SITE,),
    )

    roi_saves = [item for item in fm.saved if item[1].endswith(".roi.zip")]
    assert len(roi_saves) == 1
    roi_content, roi_path, _backend, stream_kwargs = roi_saves[0]
    stream_request = stream_kwargs["stream_request"]
    assert stream_request.source.item_fields == {
        "plane_axis": RuntimePlaneAxis.RUNTIME_SLICE.value,
        "plane_component_values": {"site": ["1"]},
        "spatial_origin_yx": [0, 0],
        "source_spatial_shape_yx": [8, 8],
    }

    napari_backend = NapariStreamingBackend()
    streamed_item = StreamingBatchMessageBuilder.build(
        napari_backend,
        StreamingBatchMessageRequest(
            data_list=[roi_content],
            file_paths=[roi_path],
            stream_request=stream_request,
            component_names_request=napari_backend.component_names_request(
                stream_request
            ),
            display_payload_extra=napari_backend.display_payload_extra(stream_request),
        ),
    ).batch_images[0]
    assert streamed_item["plane_axis"] == RuntimePlaneAxis.RUNTIME_SLICE.value
    assert streamed_item["plane_component_values"] == {"site": ["1"]}


@pytest.mark.unit
def test_generic_object_labels_feed_napari_and_fiji_roi_transports() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[2:6, 3:7] = 1
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
    )
    roi_path = materialize(
        MaterializationSpec(ROIOptions(min_area=0)),
        data=payload,
        path="/tmp/A01_s001_w1_z001_t001_cells_step3.roi.zip",
        filemanager=fm,
        backends=["memory"],
        backend_kwargs={},
    )
    rois = fm.load(roi_path, "memory")
    stream_request = _viewer_stream_backend_kwargs().values.stream_request
    napari_backend = NapariStreamingBackend()
    napari_items = StreamingBatchMessageBuilder.build(
        napari_backend,
        StreamingBatchMessageRequest(
            data_list=[rois],
            file_paths=[roi_path],
            stream_request=stream_request,
            component_names_request=napari_backend.component_names_request(
                stream_request
            ),
            display_payload_extra=napari_backend.display_payload_extra(
                stream_request
            ),
        ),
    ).batch_images
    fiji_backend = FijiStreamingBackend()
    fiji_items = StreamingBatchMessageBuilder.build(
        fiji_backend,
        StreamingBatchMessageRequest(
            data_list=[rois],
            file_paths=[roi_path],
            stream_request=stream_request,
            component_names_request=fiji_backend.component_names_request(
                stream_request
            ),
            display_payload_extra=fiji_backend.display_payload_extra(
                stream_request
            ),
        ),
    )
    fiji_items = fiji_items.batch_images

    assert [item["data_type"] for item in napari_items] == ["shapes"]
    assert [item["data_type"] for item in fiji_items] == ["rois"]
    assert napari_items[0]["shapes"]
    assert fiji_items[0]["rois"]


@pytest.mark.unit
def test_roi_materialization_splits_addressable_label_planes_for_streaming() -> None:
    class _RecordingBackend:
        requires_filesystem_validation = False

        def supports_file_path(self, _path):
            return True

    class _RecordingFileManager:
        def __init__(self):
            self.saved = []

        def _get_backend(self, _backend):
            return _RecordingBackend()

        def save(self, content, path, backend, **kwargs):
            self.saved.append((content, path, backend, kwargs))

        def save_batch(self, contents, paths, backend, **kwargs):
            self.saved.extend(
                (content, path, backend, kwargs)
                for content, path in zip(contents, paths, strict=True)
            )

    fm = _RecordingFileManager()
    labels = np.zeros((2, 8, 8), dtype=np.int32)
    labels[0, 1:4, 1:4] = 1
    labels[1, 4:7, 4:7] = 2
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
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

    out = materialize(
        MaterializationSpec(ROIOptions(min_area=0)),
        data=payload,
        path="/tmp/A01_s001_w1_z001_t001_Nuclei_step3.roi.zip",
        filemanager=fm,
        backends=["napari_stream"],
        backend_kwargs={"napari_stream": _viewer_stream_backend_kwargs()},
    )

    roi_saves = [item for item in fm.saved if item[1].endswith(".roi.zip")]
    assert out == "/tmp/A01_s001_w1_z001_t001_Nuclei_step3_rois.roi.zip"
    assert [item[1] for item in roi_saves] == [
        "/tmp/A01_s001_w1_z001_t001_Nuclei_step3_rois.roi.zip",
        "/tmp/A01_s002_w1_z001_t001_Nuclei_step3_rois.roi.zip",
    ]
    assert [_stream_component_metadata(item) for item in roi_saves] == [
        {"timepoint": 1, "z_index": 1, "site": 1, "well": "A01", "channel": 1},
        {"timepoint": 1, "z_index": 1, "site": 2, "well": "A01", "channel": 1},
    ]
    assert [len(item[0]) for item in roi_saves] == [1, 1]
    assert all(
        "plane_indices" not in roi.metadata and "plane_shape" not in roi.metadata
        for content, _path, _backend, _kwargs in roi_saves
        for roi in content
    )


@pytest.mark.unit
def test_roi_materialization_replaces_parser_equivalent_reference_source_prefix() -> (
    None
):
    fm = _RecordingFileManager()
    labels = np.zeros((2, 8, 8), dtype=np.int32)
    labels[0, 1:4, 1:4] = 1
    labels[1, 4:7, 4:7] = 2
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
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

    roi_saves = [item for item in fm.saved if item[1].endswith(".roi.zip")]
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
        },
        {
            "well": "A14",
            "site": 2,
            "channel": 1,
            "z_index": 1,
            "timepoint": 1,
        },
    ]


@pytest.mark.unit
def test_roi_streaming_applies_target_metadata_without_scalar_stream_metadata() -> None:
    class _RecordingBackend:
        requires_filesystem_validation = False

        def supports_file_path(self, _path):
            return True

    class _RecordingFileManager:
        def __init__(self):
            self.saved = []

        def _get_backend(self, _backend):
            return _RecordingBackend()

        def save(self, content, path, backend, **kwargs):
            self.saved.append((content, path, backend, kwargs))

        def save_batch(self, contents, paths, backend, **kwargs):
            self.saved.extend(
                (content, path, backend, kwargs)
                for content, path in zip(contents, paths, strict=True)
            )

    fm = _RecordingFileManager()
    labels = np.zeros((2, 8, 8), dtype=np.int32)
    labels[0, 1:4, 1:4] = 1
    labels[1, 4:7, 4:7] = 2
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
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

    roi_saves = [item for item in fm.saved if item[1].endswith(".roi.zip")]
    assert [_stream_component_metadata(item) for item in roi_saves] == [
        {"timepoint": 1, "z_index": 1, "site": 1, "well": "A01", "channel": 1},
        {"timepoint": 1, "z_index": 1, "site": 2, "well": "A01", "channel": 1},
    ]


@pytest.mark.unit
def test_roi_materialization_coalesces_duplicate_stream_targets() -> None:
    class _RecordingBackend:
        requires_filesystem_validation = False

        def supports_file_path(self, _path):
            return True

    class _RecordingFileManager:
        def __init__(self):
            self.saved = []

        def _get_backend(self, _backend):
            return _RecordingBackend()

        def save(self, content, path, backend, **kwargs):
            self.saved.append((content, path, backend, kwargs))

        def save_batch(self, contents, paths, backend, **kwargs):
            self.saved.extend(
                (content, path, backend, kwargs)
                for content, path in zip(contents, paths, strict=True)
            )

    fm = _RecordingFileManager()
    labels = np.zeros((2, 8, 8), dtype=np.int32)
    labels[0, 1:4, 1:4] = 1
    labels[1, 4:7, 4:7] = 2
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w1_z001_t001.tif",
                "/input/A01_s001_w1_z001_t001.tif",
            ),
            component_metadata=(
                {"well": "A01", "site": 1, "channel": 1},
                {"well": "A01", "site": 1, "channel": 1},
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

    roi_saves = [item for item in fm.saved if item[1].endswith(".roi.zip")]
    assert [item[1] for item in roi_saves] == [
        "/tmp/A01_s001_w1_z001_t001_Nuclei_step3_rois.roi.zip",
    ]
    assert _stream_component_metadata(roi_saves[0]) == {
        "timepoint": 1,
        "z_index": 1,
        "site": 1,
        "well": "A01",
        "channel": 1,
    }
    assert len(roi_saves[0][0]) == 2


@pytest.mark.unit
def test_roi_materialization_coalesces_source_stack_archive_with_artifact_identity() -> (
    None
):
    fm = _RecordingFileManager()
    labels = np.zeros((2, 8, 8), dtype=np.int32)
    labels[0, 1:4, 1:4] = 1
    labels[1, 4:7, 4:7] = 2
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
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
                "z_index": 1,
                "timepoint": 1,
                "extension": ".tif",
            },
        ),
    )

    roi_saves = [item for item in fm.saved if item[1].endswith(".roi.zip")]
    assert [item[1] for item in roi_saves] == [
        "/tmp/A01_s001_w1_z001_t001_Nuclei_step3_rois.roi.zip",
    ]
    assert _stream_component_metadata(roi_saves[0]) == {
        "timepoint": 1,
        "z_index": 1,
        "site": 1,
        "well": "A01",
        "channel": 1,
    }
    assert len(roi_saves[0][0]) == 2


@pytest.mark.unit
def test_roi_materialization_coalesces_same_stream_address_from_distinct_sources() -> (
    None
):
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
def test_roi_materialization_rejects_component_conflict_after_path_only_target() -> (
    None
):
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

        def supports_file_path(self, _path):
            return True

    class _RecordingFileManager:
        def __init__(self):
            self.saved = []

        def _get_backend(self, _backend):
            return _RecordingBackend()

        def save(self, content, path, backend, **kwargs):
            self.saved.append((content, path, backend, kwargs))

        def save_batch(self, contents, paths, backend, **kwargs):
            self.saved.extend(
                (content, path, backend, kwargs)
                for content, path in zip(contents, paths, strict=True)
            )

    fm = _RecordingFileManager()
    source_image = ImageMetadataPayload(
        data=np.zeros((2, 8, 8), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A02_s001_w1_z001_t001.tif",
                    "/input/A02_s002_w1_z001_t001.tif",
                ),
                component_metadata=(
                    {"well": "A02", "site": 1, "channel": 1},
                    {"well": "A02", "site": 2, "channel": 1},
                ),
            )
        ),
    )
    payload = _unaddressed_roi_label_payload().with_source_image_context(source_image)

    _stream_materialize(
        MaterializationSpec(ROIOptions(min_area=0)),
        payload,
        "/tmp/A02_s001_w1_z001_t001_Nuclei_step3.roi.zip",
        fm,
    )

    roi_saves = [item for item in fm.saved if item[1].endswith(".roi.zip")]
    assert [item[1] for item in roi_saves] == [
        "/tmp/A02_s001_w1_z001_t001_Nuclei_step3_rois.roi.zip",
        "/tmp/A02_s002_w1_z001_t001_Nuclei_step3_rois.roi.zip",
    ]
    assert [_stream_component_metadata(item) for item in roi_saves] == [
        {"timepoint": 1, "z_index": 1, "site": 1, "well": "A02", "channel": 1},
        {"timepoint": 1, "z_index": 1, "site": 2, "well": "A02", "channel": 1},
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
def test_roi_materialization_spec_reports_projected_source_identities() -> None:
    labels = np.zeros((2, 8, 8), dtype=np.int32)
    labels[0, 1:4, 1:4] = 1
    labels[1, 4:7, 4:7] = 2
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w1_z001_t001.tif",
                "/input/A01_s001_w2_z001_t001.tif",
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
                    "channel": 2,
                    "z_index": 1,
                    "timepoint": 1,
                    "extension": ".tif",
                },
            ),
        ),
    )
    spec = MaterializationSpec(ROIOptions(min_area=0))

    assert spec.emits_variable_component_planes(payload) is True
    assert [
        identity.component_metadata
        for identity in spec.emitted_source_identities(payload)
    ] == [
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
            "channel": 2,
            "z_index": 1,
            "timepoint": 1,
            "extension": ".tif",
        },
    ]


@pytest.mark.unit
def test_roi_materialization_treats_non_spatial_label_payload_as_empty() -> None:
    fm = FileManager({"memory": MemoryStorageBackend()})
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.asarray(0, dtype=np.int32))
    )

    out = _memory_materialize(
        MaterializationSpec(ROIOptions(min_area=0)),
        payload,
        "/tmp/A01_Worms_step3.roi.zip",
        fm,
    )

    assert out == "/tmp/A01_Worms_step3_segmentation_summary.txt"
    assert "No ROIs extracted" in fm.load(out, "memory")
