from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
from polystore.streaming.viewer_transport import ViewerMicroscopeHandlerABC
from polystore.streaming.viewer_transport import ViewerStreamKwarg
from zmqruntime.viewer_protocol import ViewerTransportEndpoint

from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.runtime_semantics import ObjectLabelDomain
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    ImageMetadataPayload,
    MeasurementTable,
    ObjectLabelPayload,
    RuntimeArrayPayload,
    RuntimeValue,
    RuntimeValueSchema,
    SourceImageProvenancePlanes,
    image_payload_metadata,
    RuntimeImagePayloadContext,
    normalize_artifact_value,
)
from openhcs.core.source_metadata import (
    SOURCE_PLANE_COUNT_FIELD,
    SOURCE_PLANE_INDEX_FIELD,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.steps.function_artifact_materialization import (
    AnalysisOutputDescriptorAuthority,
    PersistentArtifactMaterializationTargetPlan,
    StreamingOnlyArtifactMaterializationTargetPlan,
    materialize_artifact_outputs,
)
from openhcs.core.steps.function_runtime import FunctionOutputContextStrategy
from openhcs.core.streaming_config_factory import (
    StreamingViewerPresentation,
    StreamingViewerRuntimeConfig,
    StreamingViewerSurface,
)
from openhcs.processing.materialization import (
    CsvOptions,
    JsonOptions,
    ROIOptions,
    csv_only,
    tiff_stack,
)
from openhcs.utils.display_config_factory import ViewerDisplayConfigObject


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
        return StreamingViewerSurface(
            runtime_config=StreamingViewerRuntimeConfig(
                transport_endpoint=ViewerTransportEndpoint(
                    host=self.host,
                    port=self.port,
                    transport_mode=self.transport_mode,
                ),
                persistent=False,
                presentation=StreamingViewerPresentation(title="Napari"),
            ),
            display_config=self,
            source=self.viewer_source(_context),
        )

    @staticmethod
    def viewer_source(_context):
        from polystore.streaming.viewer_transport import ViewerStreamSourceIdentity

        return ViewerStreamSourceIdentity(
            microscope_handler=_context.microscope_handler,
            plate_path=_context.plate_path,
        )


def streaming_config_stub(port=5555):
    return StreamingConfigStub(port)


def stream_request_from_backend_kwargs(backend_kwargs):
    return backend_kwargs["napari_stream"].values.to_kwargs()[
        ViewerStreamKwarg.STREAM_REQUEST.value
    ]


class MetadataHandlerStub:
    def __init__(self, values=None):
        self.values = values or {}

    def find_metadata_file(self, root):
        return Path(root) / "openhcs_metadata.json"

    def get_component_values(self, _root, component):
        return self.values.get(component, {})


class FilenameParserStub:
    def parse_filename(self, filename):
        import re

        positional_match = re.match(
            r"(?P<well>\d+)_(?P<site>POS\d+)_(?P<channel>[A-Za-z])"
            r"(?P<extension>\.[^.]+)?$",
            Path(filename).name,
        )
        if positional_match is not None:
            return positional_match.groupdict()

        match = re.match(
            r"(?P<well>[A-Z]\d{2})_s(?P<site>\d+)_w(?P<channel>\d+)"
            r"(?:_z(?P<z_index>\d+))?(?:_t(?P<timepoint>\d+))?"
            r"(?P<extension>\.[^.]+)?$",
            Path(filename).name,
        )
        if match is None:
            return None
        parsed = {
            key: value
            for key, value in match.groupdict().items()
            if value is not None
        }
        parsed.setdefault("z_index", "1")
        parsed.setdefault("timepoint", "1")
        return parsed

    def construct_filename(
        self,
        *,
        well,
        site,
        channel,
        z_index=1,
        timepoint=1,
        extension=".tif",
    ):
        if str(site).startswith("POS"):
            return f"{well}_{site}_{channel}{extension}"
        return (
            f"{well}_s{int(site):03d}_w{int(channel)}"
            f"_z{int(z_index):03d}_t{int(timepoint):03d}{extension}"
        )


class MicroscopeHandlerStub(ViewerMicroscopeHandlerABC):
    def __init__(self, parser, metadata_handler):
        self.parser = parser
        self.metadata_handler = metadata_handler
        self.microscope_type = "test"


class FileManagerStub:
    def __init__(self):
        self.memory = {}
        self.directories = set()

    def _get_backend(self, backend):
        return BackendStub(backend)

    def exists(self, path, backend):
        return path in self.memory

    def ensure_directory(self, path, backend):
        self.directories.add((str(path), backend))

    def load(self, path, backend):
        return self.memory[path]


class BackendStub:
    requires_filesystem_validation = False

    def __init__(self, backend):
        self.backend = backend

    def supports_file_path(self, path):
        if self.backend != "napari_stream":
            return True
        name = Path(path).name.lower()
        return name.endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg", ".roi.zip"))


class ArrayLike(RuntimeArrayPayload):
    shape = (2, 2)

    def array_payload_data(self):
        return np.zeros(self.shape, dtype=np.int32)

    def with_data(self, data):
        return data


def _plan(
    output_plan,
    *,
    streaming_configs=(),
    memory_paths=(),
    source_identity_stack_axes=frozenset(),
    group_by_value=None,
):
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
        group_by_value=group_by_value,
        group_projects_runtime_plane=(
            group_by_value is not None
            and group_by_value in source_identity_stack_axes
        ),
        step_source_identity_stack_axes=source_identity_stack_axes,
        source_identity_stack_axes=source_identity_stack_axes,
    )


class ContextStub:
    pass


def _context(filemanager):
    context = ContextStub()
    context.filemanager = filemanager
    context.runtime_value_store = RuntimeValueStore()
    context.microscope_handler = MicroscopeHandlerStub(
        parser=FilenameParserStub(),
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

    contextualized = FunctionOutputContextStrategy.for_output_plan(
        output_plan
    ).contextualize(
        source,
        label_slices,
        output_plan,
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

    contextualized = FunctionOutputContextStrategy.for_output_plan(
        output_plan
    ).contextualize(
        source,
        output,
        output_plan,
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
        domain=ObjectLabelDomain(declared_object_count=2,
    ))

    contextualized = FunctionOutputContextStrategy.for_output_plan(
        output_plan
    ).contextualize(
        source,
        labels,
        output_plan,
    )
    runtime_value = normalize_artifact_value(
        output_plan,
        contextualized,
        axis_id="A02",
    )

    assert isinstance(runtime_value.data, ObjectLabelPayload)
    assert runtime_value.data.domain.declared_object_count == 2
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


def test_materialize_artifact_outputs_attaches_image_schema_provenance(monkeypatch):
    output_plan = ArtifactOutputPlan(
        name="converted_image",
        path="/memory/converted_image.pkl",
        kind=ArtifactKind.IMAGE,
        materialization=tiff_stack(),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        RuntimeValue.from_output_plan(
            output_plan,
            np.zeros((3, 5, 7), dtype=np.float32),
            axis_id="A01",
            schema=RuntimeValueSchema(
                kind=ArtifactKind.IMAGE,
                source_component_metadata={
                    "well": "A01",
                    "site": "1",
                    "channel": "3",
                    "z_index": "1",
                    SOURCE_PLANE_INDEX_FIELD: "0",
                    SOURCE_PLANE_COUNT_FIELD: "3",
                },
            ),
        ),
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
        _plan(output_plan, source_identity_stack_axes=frozenset({"z_index", "channel"})),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    assert len(materialized) == 1
    data, path = materialized[0]
    assert path == "/analysis/A01_s001_w3_z001_t001_converted_image_step7.roi.zip"
    assert isinstance(data, ImageMetadataPayload)
    assert dict(image_payload_metadata(data).source_component_metadata) == {
        "well": "A01",
        "site": "1",
        "channel": "3",
        "z_index": "1",
        SOURCE_PLANE_INDEX_FIELD: "0",
        SOURCE_PLANE_COUNT_FIELD: "3",
    }


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


def test_materialize_tabular_artifact_does_not_build_viewer_stream_kwargs(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/measurements.pkl",
        kind=ArtifactKind.MEASUREMENTS,
    )
    filemanager = FileManagerStub()
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
        _plan(output_plan, streaming_configs=(streaming_config_stub(),)),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    spec, data, path, backends, backend_kwargs = materialized[0]
    assert isinstance(spec.outputs[0], CsvOptions)
    assert data == [{"object_id": 1, "area": 42}]
    assert path == "/analysis/A01_measurements_step7.roi.zip"
    assert backends == ["disk"]
    assert "napari_stream" not in backend_kwargs


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


def test_materialize_artifact_outputs_uses_group_measurement_source_identity(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/A01_measurements_step7.pkl",
        kind=ArtifactKind.MEASUREMENTS,
        group_keys=("1", "2"),
        paths_by_group={
            "1": "/memory/A01_s001_measurements_step7.pkl",
            "2": "/memory/A01_s002_measurements_step7.pkl",
        },
    )
    group_one = output_plan.for_group("1")
    group_two = output_plan.for_group("2")
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        normalize_artifact_value(
            group_one,
            MeasurementTable(
                name="measurements",
                rows=[{"site": "1", "object_id": 1, "area": 42}],
                source_path="/input/A01_s001_w5_z001_t001.TIF",
                source_component_metadata={
                    "well": "A01",
                    "site": "1",
                    "channel": "5",
                },
            ),
            axis_id="A01",
        ),
        path=group_one.path,
        backend="memory",
    )
    context.runtime_value_store.record(
        normalize_artifact_value(
            group_two,
            MeasurementTable(
                name="measurements",
                rows=[{"site": "2", "object_id": 2, "area": 84}],
                source_path="/input/A01_s002_w5_z001_t001.TIF",
                source_component_metadata={
                    "well": "A01",
                    "site": "2",
                    "channel": "5",
                },
            ),
            axis_id="A01",
        ),
        path=group_two.path,
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

    assert [path for _spec, _data, path in materialized] == [
        "/analysis/A01_s001_w5_z001_t001_measurements_step7.roi.zip",
        "/analysis/A01_s002_w5_z001_t001_measurements_step7.roi.zip",
    ]
    assert [data for _spec, data, _path in materialized] == [
        [{"site": "1", "object_id": 1, "area": 42}],
        [{"site": "2", "object_id": 2, "area": 84}],
    ]


def test_materialize_artifact_outputs_uses_group_axis_for_partial_record_identity(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="measurements",
        path="/memory/measurements.pkl",
        kind=ArtifactKind.MEASUREMENTS,
        group_keys=("2",),
        paths_by_group={
            "2": "/memory/site2_measurements.pkl",
        },
    )
    group_plan = output_plan.for_group("2")
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        normalize_artifact_value(
            group_plan,
            MeasurementTable(
                name="measurements",
                rows=[{"site": "2", "object_id": 1, "area": 42}],
                source_component_metadata={"site": "2"},
            ),
            axis_id="A01",
        ),
        path=group_plan.path,
        backend="memory",
    )
    monkeypatch.setattr(
        AnalysisOutputDescriptorAuthority,
        "produced_memory_paths",
        classmethod(
            lambda cls, _context, _plan: [
                "/memory/A01_s001_w5_z001_t001.TIF",
                "/memory/A01_s002_w5_z001_t001.TIF",
            ]
        ),
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
        _plan(output_plan, group_by_value="site"),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    assert [path for _spec, _data, path in materialized] == [
        "/analysis/A01_s002_w5_z001_t001_measurements_step7.roi.zip",
    ]


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
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(100, 200)),
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
    assert path == "/analysis/A01_s001_w1_z001_t001_labels_step7.roi.zip"
    assert backends == ["napari_stream"]
    stream_request = stream_request_from_backend_kwargs(backend_kwargs)
    assert stream_request.port == 5555
    assert stream_request.display_config is streaming_config
    assert stream_request.source.metadata.metadata_by_index == (
        {
            "well": "A01",
            "site": "1",
            "channel": "1",
            "z_index": "1",
            "timepoint": "1",
        },
    )
    assert stream_request.message_extra["component_value_domain"] == {
        "well": ["A01"],
        "site": [1],
        "channel": [1, 2, 3],
        "z_index": [1],
        "timepoint": [1],
    }
    assert stream_request.producer.identity.to_payload() == {
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
        source_component_metadata={"channel": 3},
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(100, 200)),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.microscope_handler = MicroscopeHandlerStub(
        parser=FilenameParserStub(),
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
    stream_request = stream_request_from_backend_kwargs(backend_kwargs)
    assert stream_request.source.metadata.metadata_by_index == (
        {
            "well": "A01",
            "site": "2",
            "channel": "3",
            "z_index": "1",
            "timepoint": "1",
        },
    )
    assert stream_request.message_extra == {
        "component_value_domain": {
            "well": ["A01"],
            "site": [2],
            "channel": [1, 2, 3],
            "z_index": [1],
            "timepoint": [1],
        },
        "component_names_metadata": {
            "channel": {"1": "OrigDNA", "2": "OrigER", "3": "OrigRNA"},
            "well": {"A01": None},
            "site": {"2": None},
            "z_index": {"1": None},
            "timepoint": {"1": None},
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
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(100, 200)),
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
    stream_request = stream_request_from_backend_kwargs(backend_kwargs)
    assert stream_request.source.metadata.metadata_by_index == (
        {"well": "01", "site": "POS002", "channel": "D"},
    )


def test_materialize_artifact_outputs_uses_runtime_plane_group_identity(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="AdjacentImage",
        path="/memory/AdjacentImage.pkl",
        kind=ArtifactKind.IMAGE,
        materialization=csv_only(),
        group_keys=("11",),
        paths_by_group={
            "11": "/memory/A01_w11_AdjacentImage_step7.pkl",
        },
    )
    group_plan = output_plan.for_group("11")
    payload = ImageMetadataPayload(
        data=np.ones((8, 12, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "1",
                "z_index": "1",
                "extension": ".tif",
            },
            source_image_names=("OrigColor",),
        ),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        normalize_artifact_value(group_plan, payload, axis_id="A01"),
        path=group_plan.path,
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
        _plan(
            output_plan,
            group_by_value="timepoint",
            source_identity_stack_axes=frozenset({"timepoint"}),
        ),
        PersistentArtifactMaterializationTargetPlan("disk"),
        context,
    )

    assert materialized == [
        (
            payload,
            "/analysis/A01_s001_w1_z001_t011_AdjacentImage_step7.roi.zip",
        )
    ]


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
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(100, 200)),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.microscope_handler = MicroscopeHandlerStub(
        parser=FilenameParserStub(),
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
    stream_request = stream_request_from_backend_kwargs(backend_kwargs)
    assert stream_request.source.metadata.metadata_by_index == (
        {
            "well": "A01",
            "site": "2",
            "channel": "3",
            "z_index": "1",
            "timepoint": "1",
        },
    )


def test_materialize_artifact_outputs_uses_source_stack_identity_for_streaming(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="Nuclei",
        path="/memory/Nuclei.pkl",
        kind=ArtifactKind.OBJECT_LABELS,
    )
    streaming_config = streaming_config_stub()
    labels = ObjectLabelPayload(
        labels=np.zeros((2, 2, 2), dtype=np.int32),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w1_z001_t001.TIF",
                "/input/A01_s001_w1_z002_t001.TIF",
            ),
            component_metadata=(
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "1",
                    "z_index": "1",
                    "timepoint": "1",
                },
                {
                    "well": "A01",
                    "site": "1",
                    "channel": "1",
                    "z_index": "2",
                    "timepoint": "1",
                },
            ),
        ),
        source_spatial_domain=SourceSpatialDomain(source_shape_yx=(100, 200)),
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
        _plan(
            output_plan,
            streaming_configs=(streaming_config,),
            source_identity_stack_axes=frozenset({"z_index"}),
        ),
        StreamingOnlyArtifactMaterializationTargetPlan(),
        context,
    )

    _spec, _data, path, _backends, backend_kwargs = materialized[0]
    assert path == "/analysis/A01_s001_w1_z001_t001_Nuclei_step7.roi.zip"
    stream_request = stream_request_from_backend_kwargs(backend_kwargs)
    assert stream_request.source.metadata.metadata_by_index == (
        {
            "well": "A01",
            "site": "1",
            "channel": "1",
            "timepoint": "1",
        },
    )


def test_materialize_rgb_artifact_streams_filename_channel_identity(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="RGBImage",
        path="/memory/RGBImage.pkl",
        kind=ArtifactKind.IMAGE,
        materialization=tiff_stack(),
    )
    streaming_config = streaming_config_stub()
    rgb_payload = ImageMetadataPayload(
        data=np.ones((5, 7, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A01_s001_w3_z001_t001.TIF",
                    "/input/A01_s001_w2_z001_t001.TIF",
                    "/input/A01_s001_w1_z001_t001.TIF",
                ),
                component_metadata=(
                    {
                        "well": "A01",
                        "site": "1",
                        "channel": "3",
                        "z_index": "1",
                        "timepoint": "1",
                    },
                    {
                        "well": "A01",
                        "site": "1",
                        "channel": "2",
                        "z_index": "1",
                        "timepoint": "1",
                    },
                    {
                        "well": "A01",
                        "site": "1",
                        "channel": "1",
                        "z_index": "1",
                        "timepoint": "1",
                    },
                ),
            ),
        ),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        normalize_artifact_value(output_plan, rgb_payload, axis_id="A01"),
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
            source_identity_stack_axes=frozenset({"channel"}),
        ),
        StreamingOnlyArtifactMaterializationTargetPlan(),
        context,
    )

    _spec, _data, path, _backends, backend_kwargs = materialized[0]
    assert path == "/analysis/A01_s001_w3_z001_t001_RGBImage_step7.roi.zip"
    stream_request = stream_request_from_backend_kwargs(backend_kwargs)
    assert stream_request.source.metadata.metadata_by_index == (
        {
            "well": "A01",
            "site": "1",
            "channel": "3",
            "z_index": "1",
            "timepoint": "1",
        },
    )


def test_materialize_rgb_artifact_keeps_scalar_filename_identity_for_mixed_provenance(
    monkeypatch,
):
    output_plan = ArtifactOutputPlan(
        name="OrigOverlay",
        path="/memory/OrigOverlay.pkl",
        kind=ArtifactKind.IMAGE,
        materialization=tiff_stack(),
    )
    streaming_config = streaming_config_stub()
    rgb_payload = ImageMetadataPayload(
        data=np.ones((5, 7, 3), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            source_path="/input/A01_s001_w2_z001_t001.TIF",
            source_component_metadata={
                "well": "A01",
                "site": "1",
                "channel": "2",
                "z_index": "1",
                "timepoint": "1",
            },
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=(
                    "/input/A01_s001_w3_z001_t001.TIF",
                    "/input/A01_s001_w2_z001_t001.TIF",
                ),
                component_metadata=(
                    {
                        "well": "A01",
                        "site": "1",
                        "channel": "3",
                        "z_index": "1",
                        "timepoint": "1",
                    },
                    {
                        "well": "A01",
                        "site": "1",
                        "channel": "2",
                        "z_index": "1",
                        "timepoint": "1",
                    },
                ),
            ),
        ),
    )
    filemanager = FileManagerStub()
    context = _context(filemanager)
    context.runtime_value_store.record(
        normalize_artifact_value(output_plan, rgb_payload, axis_id="A01"),
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
            source_identity_stack_axes=frozenset({"site"}),
        ),
        StreamingOnlyArtifactMaterializationTargetPlan(),
        context,
    )

    _spec, _data, path, _backends, backend_kwargs = materialized[0]
    assert path == "/analysis/A01_s001_w2_z001_t001_OrigOverlay_step7.roi.zip"
    stream_request = stream_request_from_backend_kwargs(backend_kwargs)
    assert stream_request.source.metadata.metadata_by_index == (
        {
            "well": "A01",
            "site": "1",
            "channel": "2",
            "z_index": "1",
            "timepoint": "1",
        },
    )
