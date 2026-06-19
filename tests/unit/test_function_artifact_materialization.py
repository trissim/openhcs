from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
from polystore.streaming.viewer_transport import ViewerMicroscopeHandlerABC
from polystore.streaming.viewer_transport import ViewerStreamKwarg
from polystore.streaming.viewer_transport import ViewerStreamSourceIdentity
from zmqruntime.viewer_protocol import ViewerTransportEndpoint

from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    ImageMetadataPayload,
    ObjectLabelPayload,
    RuntimeArrayPayload,
    image_payload_metadata,
    RuntimeImagePayloadContext,
    normalize_artifact_value,
SourceImageProvenancePlanes)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.steps.function_artifact_materialization import (
    PersistentArtifactMaterializationTargetPlan,
    StreamingOnlyArtifactMaterializationTargetPlan,
    materialize_artifact_outputs,
)
from openhcs.core.steps.function_runtime import (
    FunctionOutputContextRequest,
    FunctionOutputContextStrategy,
)
from openhcs.processing.materialization import CsvOptions, JsonOptions, ROIOptions, csv_only
from openhcs.utils.display_config_factory import ViewerDisplayConfigObject


class StreamingViewerSurfaceStub:
    def __init__(self, display_config, context):
        self.display_config = display_config
        self.runtime_config = SimpleNamespace(
            transport_endpoint=ViewerTransportEndpoint(
                host=display_config.host,
                port=display_config.port,
                transport_mode=display_config.transport_mode,
            )
        )
        self.source = ViewerStreamSourceIdentity(
            microscope_handler=context.microscope_handler,
            plate_path=context.plate_path,
        )


class StreamingConfigStub(ViewerDisplayConfigObject):
    backend = SimpleNamespace(value="napari_stream")
    COMPONENT_ORDER = ("well", "site", "channel", "z_index", "timepoint")
    host = "127.0.0.1"
    transport_mode = "tcp"

    def __init__(self, port):
        self.port = port

    def component_modes(self):
        return {component: "stack" for component in self.COMPONENT_ORDER}

    def streaming_viewer_surface(self, _context):
        return StreamingViewerSurfaceStub(self, _context)


def streaming_config_stub(port=5555):
    return StreamingConfigStub(port)


class MetadataHandlerStub:
    def __init__(self, values=None):
        self.values = values or {}

    def find_metadata_file(self, root):
        return Path(root) / "openhcs_metadata.json"

    def get_component_values(self, _root, component):
        return self.values.get(component, {})


class MicroscopeHandlerStub(ViewerMicroscopeHandlerABC):
    def __init__(self, parser, metadata_handler):
        self.parser = parser
        self.metadata_handler = metadata_handler
        self.microscope_type = "test"


class FileManagerStub:
    def __init__(self):
        self.memory = {}
        self.directories = set()

    def exists(self, path, backend):
        return path in self.memory

    def ensure_directory(self, path, backend):
        self.directories.add((str(path), backend))

    def load(self, path, backend):
        return self.memory[path]


class ArrayLike(RuntimeArrayPayload):
    shape = (2, 2)

    def array_payload_data(self):
        return np.zeros(self.shape, dtype=np.int32)

    def with_data(self, data):
        return data


def _plan(output_plan, *, streaming_configs=(), memory_paths=()):
    return SimpleNamespace(
        artifact_outputs={output_plan.name: output_plan},
        streaming_configs=streaming_configs,
        artifact_analysis_output_dir=Path("/analysis"),
        artifact_images_dir="/images",
        step_name="measure",
        axis_id="A01",
        pipeline_position=7,
        step_scope_id="measure-scope-7",
        get_paths_for_axis=lambda *_args: list(memory_paths),
        output_dir=Path("/tmp/output"),
        input_dir=Path("/tmp/input"),
        read_backend="memory",
        group_by_value=None,
    )


class ContextStub:
    pass


def _context(filemanager):
    context = ContextStub()
    context.filemanager = filemanager
    context.runtime_value_store = RuntimeValueStore()
    context.microscope_handler = MicroscopeHandlerStub(
        parser=SimpleNamespace(parse_filename=lambda _filename: None),
        metadata_handler=MetadataHandlerStub(
            {"channel": {"1": "OrigDNA", "2": "OrigER", "3": "OrigRNA"}}
        ),
    )
    context.plate_path = Path("/tmp/plate")
    context.input_dir = Path("/tmp/plate/images")
    context.owned_wells = ["A01"]
    return context


def test_slice_aligned_object_label_arrays_preserve_source_slice_metadata():
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        kind=ArtifactKind.OBJECT_LABELS,
    )
    source = RuntimeImagePayloadContext(
        np.zeros((2, 8, 8), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes = SourceImageProvenancePlanes.from_components(paths = (
                "/input/A02_s001_w1_z001_t001.tif",
                "/input/A02_s002_w1_z001_t001.tif",
            ), component_metadata = (
                {"well": "A02", "site": 1, "channel": 1},
                {"well": "A02", "site": 2, "channel": 1},
            ))),
    mask = None).payload()
    label_slices = RuntimeSliceAlignedValues(
        (
            np.array([[0, 1], [0, 0]], dtype=np.int32),
            np.array([[0, 2], [0, 0]], dtype=np.int32),
        )
    )

    request = FunctionOutputContextRequest(
        source_payload=source,
        output_value=label_slices,
        output_plan=output_plan,
    )
    contextualized = FunctionOutputContextStrategy.for_output(request).contextualize(
        request
    )
    runtime_value = normalize_artifact_value(
        output_plan,
        contextualized,
        axis_id="A02",
    )

    assert isinstance(runtime_value.data, ObjectLabelPayload)
    assert runtime_value.data.source_image_provenance_planes.paths == (
        "/input/A02_s001_w1_z001_t001.tif",
        "/input/A02_s002_w1_z001_t001.tif",
    )
    assert tuple(
        dict(item) for item in runtime_value.data.source_image_provenance_planes.component_metadata
    ) == (
        {"well": "A02", "site": 1, "channel": 1},
        {"well": "A02", "site": 2, "channel": 1},
    )


def test_image_outputs_merge_source_provenance_when_output_already_has_metadata():
    output_plan = ArtifactOutputPlan(
        name="Corrected",
        path="/memory/Corrected.pkl",
        kind=ArtifactKind.IMAGE,
    )
    source = RuntimeImagePayloadContext(
        np.zeros((2, 8, 8), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes = SourceImageProvenancePlanes.from_components(paths = (
                "/input/A02_s001_w1_z001_t001.tif",
                "/input/A02_s002_w1_z001_t001.tif",
            ), component_metadata = (
                {"well": "A02", "site": 1, "channel": 1},
                {"well": "A02", "site": 2, "channel": 1},
            ))),
    mask = None).payload()
    output = ImageMetadataPayload(
        data=np.ones((2, 8, 8), dtype=np.float32),
        metadata=ImagePayloadMetadata(source_dtype="float32"),
    )

    request = FunctionOutputContextRequest(
        source_payload=source,
        output_value=output,
        output_plan=output_plan,
    )
    contextualized = FunctionOutputContextStrategy.for_output(request).contextualize(
        request
    )

    metadata = image_payload_metadata(contextualized)
    assert metadata.source_dtype == "float32"
    assert metadata.source_image_provenance_planes.paths == (
        "/input/A02_s001_w1_z001_t001.tif",
        "/input/A02_s002_w1_z001_t001.tif",
    )
    assert tuple(dict(item) for item in metadata.source_image_provenance_planes.component_metadata) == (
        {"well": "A02", "site": 1, "channel": 1},
        {"well": "A02", "site": 2, "channel": 1},
    )


def test_object_label_payload_stack_preserves_source_slice_metadata():
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        kind=ArtifactKind.OBJECT_LABELS,
    )
    source = RuntimeImagePayloadContext(
        np.zeros((2, 8, 8), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes = SourceImageProvenancePlanes.from_components(paths = (
                "/input/A02_s001_w1_z001_t001.tif",
                "/input/A02_s002_w1_z001_t001.tif",
            ), component_metadata = (
                {"well": "A02", "site": 1, "channel": 1},
                {"well": "A02", "site": 2, "channel": 1},
            ))),
    mask = None).payload()
    labels = ObjectLabelPayload(
        labels=np.stack(
            (
                np.array([[0, 1], [0, 0]], dtype=np.int32),
                np.array([[0, 2], [0, 0]], dtype=np.int32),
            )
        ),
        declared_object_count=2,
    )

    request = FunctionOutputContextRequest(
        source_payload=source,
        output_value=labels,
        output_plan=output_plan,
    )
    contextualized = FunctionOutputContextStrategy.for_output(request).contextualize(
        request
    )
    runtime_value = normalize_artifact_value(
        output_plan,
        contextualized,
        axis_id="A02",
    )

    assert isinstance(runtime_value.data, ObjectLabelPayload)
    assert runtime_value.data.declared_object_count == 2
    assert runtime_value.data.source_image_provenance_planes.paths == (
        "/input/A02_s001_w1_z001_t001.tif",
        "/input/A02_s002_w1_z001_t001.tif",
    )
    assert tuple(
        dict(item) for item in runtime_value.data.source_image_provenance_planes.component_metadata
    ) == (
        {"well": "A02", "site": 1, "channel": 1},
        {"well": "A02", "site": 2, "channel": 1},
    )


def test_materialize_artifact_outputs_uses_runtime_store_payload(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="positions",
        path="/memory/positions.pkl",
        materialization=csv_only(),
    )
    filemanager = FileManagerStub()
    filemanager.memory[output_plan.path] = {"x": "from-vfs"}
    context = _context(filemanager)
    context.runtime_value_store.record(
        normalize_artifact_value(output_plan, {"x": "from-runtime"}, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(_spec, data, path, *_args, **_kwargs):
        materialized.append((data, path))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    assert materialized == [
        ({"x": "from-runtime"}, "/analysis/A01_positions_step7.roi.zip")
    ]


def test_materialize_artifact_outputs_requires_runtime_store_record():
    output_plan = ArtifactOutputPlan(
        name="positions",
        path="/memory/positions.pkl",
        materialization=object(),
    )
    filemanager = FileManagerStub()
    filemanager.memory[output_plan.path] = {"x": 1}
    context = _context(filemanager)

    with pytest.raises(RuntimeError, match="Missing RuntimeValueStore record"):
        materialize_artifact_outputs(
            filemanager,
            _plan(output_plan),
            PersistentArtifactMaterializationTargetPlan("disk"),
            context,
        )


def test_materialize_artifact_outputs_does_not_require_vfs_payload_for_store_record(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="positions",
        path="/memory/positions.pkl",
        materialization=csv_only(),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        normalize_artifact_value(output_plan, {"x": 1}, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(_spec, data, path, *_args, **_kwargs):
        materialized.append((data, path))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    assert materialized == [
        ({"x": 1}, "/analysis/A01_positions_step7.roi.zip")
    ]


def test_materialize_artifact_outputs_defaults_measurements_to_existing_csv_spec(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/measurements.pkl",
        kind=ArtifactKind.MEASUREMENTS,
    )
    filemanager = FileManagerStub()
    filemanager.memory[output_plan.path] = [{"object_id": 1, "area": 42}]
    context = _context(filemanager)
    context.runtime_value_store.record(
        normalize_artifact_value(
            output_plan,
            [{"object_id": 1, "area": 42}],
            axis_id="A01",
        ),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(spec, data, path, *_args, **_kwargs):
        materialized.append((spec, data, path))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    spec, data, path = materialized[0]
    assert isinstance(spec.outputs[0], CsvOptions)
    assert spec.outputs[0].filename_suffix == ".csv"
    assert data == [{"object_id": 1, "area": 42}]
    assert path == "/analysis/A01_measurements_step7.roi.zip"


def test_materialize_artifact_outputs_uses_actual_group_records(monkeypatch):
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/A01_measurements_step7.pkl",
        kind=ArtifactKind.MEASUREMENTS,
        group_keys=("1", "2"),
        paths_by_group={
            "1": "/memory/A01_w1_measurements_step7.pkl",
            "2": "/memory/A01_w2_measurements_step7.pkl",
        },
    )
    group_plan = output_plan.for_group("1")
    filemanager = FileManagerStub()
    filemanager.memory[group_plan.path] = [{"site": "1", "area": 42}]
    context = _context(filemanager)
    context.runtime_value_store.record(
        normalize_artifact_value(
            group_plan,
            [{"site": "1", "area": 42}],
            axis_id="A01",
        ),
        path=group_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(spec, data, path, *_args, **_kwargs):
        materialized.append((spec, data, path))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    assert len(materialized) == 1
    spec, data, path = materialized[0]
    assert isinstance(spec.outputs[0], CsvOptions)
    assert data == [{"site": "1", "area": 42}]
    assert path == "/analysis/A01_w1_measurements_step7.roi.zip"


def test_materialize_artifact_outputs_defaults_metadata_to_existing_json_spec(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="metadata",
        path="/memory/metadata.pkl",
        kind=ArtifactKind.METADATA,
    )
    filemanager = FileManagerStub()
    filemanager.memory[output_plan.path] = {"plate": "A"}
    context = _context(filemanager)
    context.runtime_value_store.record(
        normalize_artifact_value(output_plan, {"plate": "A"}, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(spec, data, path, *_args, **_kwargs):
        materialized.append((spec, data, path))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    spec, data, _path = materialized[0]
    assert isinstance(spec.outputs[0], JsonOptions)
    assert data == {"plate": "A"}


def test_materialize_artifact_outputs_skips_special_without_explicit_spec(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="positions",
        path="/memory/positions.pkl",
        kind=ArtifactKind.SPECIAL,
    )
    filemanager = FileManagerStub()
    filemanager.memory[output_plan.path] = {"x": 1}
    context = _context(filemanager)
    materialized = []

    def fake_materialize(*args, **kwargs):
        materialized.append((args, kwargs))

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    assert materialized == []


def test_materialize_artifact_outputs_defaults_object_labels_to_roi_spec(monkeypatch):
    output_plan = ArtifactOutputPlan(
        name="labels",
        path="/memory/labels.pkl",
        kind=ArtifactKind.OBJECT_LABELS,
    )
    array_like = ArrayLike()
    filemanager = FileManagerStub()
    filemanager.memory[output_plan.path] = array_like
    context = _context(filemanager)
    context.runtime_value_store.record(
        normalize_artifact_value(output_plan, array_like, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(spec, data, path, *_args, **_kwargs):
        materialized.append((spec, data, path))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    spec, data, path = materialized[0]
    assert isinstance(spec.outputs[0], ROIOptions)
    assert data is array_like
    assert path == "/analysis/A01_labels_step7.roi.zip"


def test_materialize_artifact_outputs_can_target_streaming_without_persistent_backend(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="labels",
        path="/memory/labels.pkl",
        kind=ArtifactKind.OBJECT_LABELS,
    )
    streaming_config = streaming_config_stub()
    labels = ObjectLabelPayload(
        labels=np.zeros((2, 2), dtype=np.int32),
        source_path="/input/A01_s001_w1.TIF",
        source_component_metadata={"well": "A01", "channel": 1},
        source_spatial_shape_yx=(100, 200),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        normalize_artifact_value(output_plan, labels, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(
        spec,
        data,
        path,
        _filemanager,
        backends,
        backend_kwargs,
        **_kwargs,
    ):
        materialized.append((spec, data, path, backends, backend_kwargs))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan, streaming_configs=(streaming_config,)),
        StreamingOnlyArtifactMaterializationTargetPlan(),
        context,
    )

    spec, data, path, backends, backend_kwargs = materialized[0]
    assert isinstance(spec.outputs[0], ROIOptions)
    assert data is labels
    assert path == "/analysis/A01_s001_w1_labels_step7.roi.zip"
    assert backends == ["napari_stream"]
    stream_request = backend_kwargs["napari_stream"][ViewerStreamKwarg.STREAM_REQUEST.value]
    assert stream_request.port == 5555
    assert stream_request.display_config is streaming_config
    assert stream_request.source.metadata.component_metadata_by_path == (
        {"well": "A01", "channel": 1},
    )
    assert stream_request.message_extra["component_value_domain"] == {
        "well": ["A01"],
        "channel": [1, 2, 3],
    }
    assert stream_request.producer_identity.to_payload() == {
        "origin": "pipeline",
        "output_kind": "artifact",
        "output_key": "labels",
        "step_name": "measure",
        "pipeline_position": 7,
        "step_scope_id": "measure-scope-7",
        "invocation_key": None,
        "artifact_kind": "object_labels",
    }


def test_materialize_artifact_outputs_uses_artifact_source_metadata_for_streaming(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="labels",
        path="/memory/labels.pkl",
        kind=ArtifactKind.OBJECT_LABELS,
    )
    streaming_config = streaming_config_stub()
    labels = ObjectLabelPayload(
        labels=np.zeros((2, 2), dtype=np.int32),
        source_path="/input/A01_s002_w3_z001_t001.TIF",
        source_spatial_shape_yx=(100, 200),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.microscope_handler = MicroscopeHandlerStub(
        parser=SimpleNamespace(
            parse_filename=lambda filename: (
                {"well": "A01", "site": 2, "channel": 3}
                if filename == "A01_s002_w3_z001_t001.TIF"
                else None
            )
        ),
        metadata_handler=MetadataHandlerStub(
            {"channel": {"1": "OrigDNA", "2": "OrigER", "3": "OrigRNA"}}
        ),
    )
    context.runtime_value_store.record(
        normalize_artifact_value(output_plan, labels, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(
        spec,
        data,
        path,
        _filemanager,
        backends,
        backend_kwargs,
        **_kwargs,
    ):
        materialized.append((spec, data, path, backends, backend_kwargs))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(
            output_plan,
            streaming_configs=(streaming_config,),
            memory_paths=("/memory/A01_s001_w1_z001_t001.TIF",),
        ),
        StreamingOnlyArtifactMaterializationTargetPlan(),
        context,
    )

    _spec, _data, path, _backends, backend_kwargs = materialized[0]
    assert path == "/analysis/A01_s002_w3_z001_t001_labels_step7.roi.zip"
    stream_request = backend_kwargs["napari_stream"][ViewerStreamKwarg.STREAM_REQUEST.value]
    assert stream_request.source.metadata.component_metadata_by_path == (
        {"well": "A01", "site": 2, "channel": 3},
    )
    assert stream_request.message_extra == {
        "component_value_domain": {
            "well": ["A01"],
            "site": [2],
            "channel": [1, 2, 3],
        },
        "component_names_metadata": {
            "channel": {"1": "OrigDNA", "2": "OrigER", "3": "OrigRNA"},
            "well": {"A01": None},
            "site": {"2": None},
        },
    }


def test_materialize_artifact_outputs_streams_payload_component_metadata(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        kind=ArtifactKind.OBJECT_LABELS,
    )
    streaming_config = streaming_config_stub()
    labels = ObjectLabelPayload(
        labels=np.zeros((2, 2), dtype=np.int32),
        source_path="/input/01_POS002_D.TIF",
        source_component_metadata={"well": "01", "site": "POS002", "channel": "D"},
        source_spatial_shape_yx=(100, 200),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        normalize_artifact_value(output_plan, labels, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(
        spec,
        data,
        path,
        _filemanager,
        backends,
        backend_kwargs,
        **_kwargs,
    ):
        materialized.append((spec, data, path, backends, backend_kwargs))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan, streaming_configs=(streaming_config,)),
        StreamingOnlyArtifactMaterializationTargetPlan(),
        context,
    )

    _spec, _data, path, _backends, backend_kwargs = materialized[0]
    assert path == "/analysis/01_POS002_D_Nuclei_step7.roi.zip"
    stream_request = backend_kwargs["napari_stream"][ViewerStreamKwarg.STREAM_REQUEST.value]
    assert stream_request.source.metadata.component_metadata_by_path == (
        {"well": "01", "site": "POS002", "channel": "D"},
    )


def test_materialize_artifact_outputs_merges_parser_axes_into_source_metadata(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        kind=ArtifactKind.OBJECT_LABELS,
    )
    streaming_config = streaming_config_stub()
    labels = ObjectLabelPayload(
        labels=np.zeros((2, 2), dtype=np.int32),
        source_path="/input/A01_s002_w3_z001_t001.TIF",
        source_component_metadata={"OpenHCSImageType": "Grayscale image"},
        source_spatial_shape_yx=(100, 200),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.microscope_handler = MicroscopeHandlerStub(
        parser=SimpleNamespace(
            parse_filename=lambda filename: (
                {"well": "A01", "site": 2, "channel": 3}
                if filename == "A01_s002_w3_z001_t001.TIF"
                else None
            )
        ),
        metadata_handler=MetadataHandlerStub(
            {"channel": {"1": "OrigDNA", "2": "OrigER", "3": "OrigRNA"}}
        ),
    )
    context.runtime_value_store.record(
        normalize_artifact_value(output_plan, labels, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    materialized = []

    def fake_materialize(
        spec,
        data,
        path,
        _filemanager,
        backends,
        backend_kwargs,
        **_kwargs,
    ):
        materialized.append((spec, data, path, backends, backend_kwargs))
        return path

    monkeypatch.setattr(
        "openhcs.processing.materialization.materialize",
        fake_materialize,
    )

    materialize_artifact_outputs(
        filemanager,
        _plan(output_plan, streaming_configs=(streaming_config,)),
        StreamingOnlyArtifactMaterializationTargetPlan(),
        context,
    )

    _spec, _data, path, _backends, backend_kwargs = materialized[0]
    assert path == "/analysis/A01_s002_w3_z001_t001_Nuclei_step7.roi.zip"
    stream_request = backend_kwargs["napari_stream"][ViewerStreamKwarg.STREAM_REQUEST.value]
    assert stream_request.source.metadata.component_metadata_by_path == (
        {
            "OpenHCSImageType": "Grayscale image",
            "well": "A01",
            "site": "2",
            "channel": "3",
        },
    )
