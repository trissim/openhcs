"""Reproducible phase benchmark for large native Napari ROI streams."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import tracemalloc
import uuid
from dataclasses import asdict, dataclass, replace

import numpy as np
import zmq
from napari.components import ViewerModel
from qtpy.QtWidgets import QApplication

from polystore.streaming.identity import StreamProducerIdentity
from polystore.streaming_constants import StreamingDataType

from openhcs.core.runtime_image_values import ImagePayloadMetadata
from openhcs.napari_roi_manager.widgets._roi_manager import QRoiManager
from openhcs.runtime.napari_streaming_handlers import (
    NapariShapeLayerPayload,
    NapariStreamLayerAddress,
    NapariStreamLayerItem,
    VisualMetadataField,
)
from openhcs.runtime.napari_viewer_server import NapariShapesLayerDisplayHandler
from openhcs.runtime.viewer_component_system import (
    ViewerComponentValueDomainPayload,
    ViewerLayerAxisProjection,
)


@dataclass(frozen=True, slots=True)
class NapariRoiStreamingBenchmarkSample:
    """One measured large-ROI stream application."""

    roi_count: int
    vertices_per_roi: int
    total_vertex_count: int
    wire_bytes: int
    serialization_seconds: float
    transport_seconds: float
    decoding_seconds: float
    feature_projection_seconds: float
    shapes_insertion_seconds: float
    roi_manager_construction_seconds: float
    roi_table_refresh_seconds: float
    selection_synchronization_seconds: float
    paint_settlement_seconds: float
    peak_python_allocation_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class NapariRoiStreamingBenchmarkSummary:
    """Median timings plus every raw sample for one payload scale."""

    roi_count: int
    vertices_per_roi: int
    median: NapariRoiStreamingBenchmarkSample
    samples: tuple[NapariRoiStreamingBenchmarkSample, ...]
    allocation_sample: NapariRoiStreamingBenchmarkSample | None

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def benchmark_napari_roi_streaming(
    *,
    roi_count: int,
    vertex_count: int = 12,
    repeats: int = 3,
    track_allocations: bool = True,
) -> NapariRoiStreamingBenchmarkSummary:
    """Measure warm large-ROI phases and one allocation-instrumented sample."""

    if roi_count <= 0 or vertex_count < 3 or repeats <= 0:
        raise ValueError(
            "ROI count/repeats must be positive and vertex count must be at least 3."
        )
    application = QApplication.instance() or QApplication([])
    shape_records = _shape_records(roi_count, vertex_count)
    _run_sample(application, shape_records, track_allocations=False)
    samples = tuple(
        _run_sample(application, shape_records, track_allocations=False)
        for _ in range(repeats)
    )
    allocation_sample = (
        _run_sample(
            application,
            shape_records,
            track_allocations=True,
        )
        if track_allocations
        else None
    )
    return NapariRoiStreamingBenchmarkSummary(
        roi_count=roi_count,
        vertices_per_roi=vertex_count,
        median=_median_sample(samples),
        samples=samples,
        allocation_sample=allocation_sample,
    )


def _run_sample(
    application: QApplication,
    shape_records: list[dict[str, object]],
    *,
    track_allocations: bool,
) -> NapariRoiStreamingBenchmarkSample:
    if track_allocations:
        tracemalloc.start()

    serialization_started = time.perf_counter()
    encoded = json.dumps({"shapes": shape_records}).encode()
    serialization_seconds = time.perf_counter() - serialization_started

    transmitted, transport_seconds = _transport_round_trip(encoded)
    decoding_started = time.perf_counter()
    decoded = json.loads(transmitted)
    decoding_seconds = time.perf_counter() - decoding_started

    projection_started = time.perf_counter()
    payload = NapariShapeLayerPayload.build(
        layer_items=[_stream_layer_item(decoded["shapes"])],
        axis_projection=_plane_projection(),
    )
    feature_projection_seconds = time.perf_counter() - projection_started

    insertion_started = time.perf_counter()
    viewer = ViewerModel()
    layer = _insert_native_shapes(viewer, payload)
    shapes_insertion_seconds = time.perf_counter() - insertion_started

    viewer.layers.selection.clear()
    manager_started = time.perf_counter()
    manager = QRoiManager(viewer)
    roi_manager_construction_seconds = time.perf_counter() - manager_started
    table_started = time.perf_counter()
    manager.connect_layer(layer)
    roi_table_refresh_seconds = time.perf_counter() - table_started

    selection_started = time.perf_counter()
    layer.selected_data = {len(payload.data) - 1}
    application.processEvents()
    selection_synchronization_seconds = time.perf_counter() - selection_started

    settlement_started = time.perf_counter()
    layer.visible = True
    application.processEvents()
    paint_settlement_seconds = time.perf_counter() - settlement_started

    peak_python_allocation_bytes = None
    if track_allocations:
        _, peak_python_allocation_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    manager.close()
    viewer.layers.clear()
    application.processEvents()
    return NapariRoiStreamingBenchmarkSample(
        roi_count=len(payload.data),
        vertices_per_roi=(
            len(payload.data[0]) if payload.data else 0
        ),
        total_vertex_count=sum(len(coordinates) for coordinates in payload.data),
        wire_bytes=len(encoded),
        serialization_seconds=serialization_seconds,
        transport_seconds=transport_seconds,
        decoding_seconds=decoding_seconds,
        feature_projection_seconds=feature_projection_seconds,
        shapes_insertion_seconds=shapes_insertion_seconds,
        roi_manager_construction_seconds=roi_manager_construction_seconds,
        roi_table_refresh_seconds=roi_table_refresh_seconds,
        selection_synchronization_seconds=selection_synchronization_seconds,
        paint_settlement_seconds=paint_settlement_seconds,
        peak_python_allocation_bytes=peak_python_allocation_bytes,
    )


def _transport_round_trip(payload: bytes) -> tuple[bytes, float]:
    context = zmq.Context.instance()
    sender = context.socket(zmq.PAIR)
    receiver = context.socket(zmq.PAIR)
    endpoint = f"inproc://openhcs-napari-roi-benchmark-{uuid.uuid4()}"
    sender.bind(endpoint)
    receiver.connect(endpoint)
    try:
        started = time.perf_counter()
        sender.send(payload)
        received = receiver.recv()
        return received, time.perf_counter() - started
    finally:
        sender.close()
        receiver.close()


def _insert_native_shapes(
    viewer: ViewerModel,
    payload: NapariShapeLayerPayload,
):
    chunks = payload.chunks(
        max_shape_count=NapariShapesLayerDisplayHandler.MAX_SHAPES_PER_WORK_UNIT,
        max_vertex_count=NapariShapesLayerDisplayHandler.MAX_VERTICES_PER_WORK_UNIT,
    )
    color_projection = payload.color_projection
    first = chunks[0]
    layer = viewer.add_shapes(
        first.data,
        shape_type=first.shape_types,
        features=first.features,
        edge_color=VisualMetadataField.LABEL.value,
        face_color=VisualMetadataField.LABEL.value,
        edge_color_cycle=color_projection.cycle,
        face_color_cycle=color_projection.cycle,
        opacity=0.7,
        ndim=payload.ndim,
        visible=False,
    )
    member_index = len(first.data)
    for chunk in chunks[1:]:
        next_member_index = member_index + len(chunk.data)
        colors = color_projection.member_colors[member_index:next_member_index]
        layer.add(
            chunk.data,
            shape_type=chunk.shape_types,
            edge_color=colors,
            face_color=colors,
        )
        member_index = next_member_index
    layer.features = payload.features
    layer.edge_color_cycle = color_projection.cycle
    layer.face_color_cycle = color_projection.cycle
    layer.edge_color_mode = "cycle"
    layer.face_color_mode = "cycle"
    return layer


def _stream_layer_item(
    shapes: list[dict[str, object]],
) -> NapariStreamLayerItem:
    return NapariStreamLayerItem(
        data=shapes,
        producer=StreamProducerIdentity.pipeline_output(
            output_kind="artifact",
            output_key="benchmark_rois",
            projection_key="benchmark_rois",
            step_name="Napari ROI benchmark",
            pipeline_position=0,
        ),
        address=NapariStreamLayerAddress(
            components={},
            path="benchmark.roi.zip",
            stream_layer_data_type=StreamingDataType.SHAPES,
        ),
        image_metadata=ImagePayloadMetadata(),
        plane_component_domain=ViewerComponentValueDomainPayload(()),
    )


def _plane_projection() -> ViewerLayerAxisProjection:
    return ViewerLayerAxisProjection(
        projected_axis_components=(),
        component_values={},
        routed_component_values={},
        axis_offsets=(),
        scalar_component_values={},
    )


def _shape_records(
    roi_count: int,
    vertex_count: int,
) -> list[dict[str, object]]:
    angles = np.linspace(0.0, 2.0 * math.pi, vertex_count, endpoint=False)
    unit_contour = np.stack((np.sin(angles), np.cos(angles)), axis=1)
    column_count = math.ceil(math.sqrt(roi_count))
    records: list[dict[str, object]] = []
    for index in range(roi_count):
        center = np.asarray(
            ((index // column_count) * 4.0, (index % column_count) * 4.0)
        )
        coordinates = center + unit_contour * (1.0 + (index % 5) * 0.08)
        records.append(
            {
                "type": "polygon",
                "coordinates": coordinates.tolist(),
                "metadata": {
                    "name": f"ROI-{index:05d}",
                    "label": index % 97 + 1,
                    "area": float(math.pi * (1.0 + (index % 5) * 0.08) ** 2),
                },
            }
        )
    return records


def _median_sample(
    samples: tuple[NapariRoiStreamingBenchmarkSample, ...],
) -> NapariRoiStreamingBenchmarkSample:
    first = samples[0]
    return replace(
        first,
        serialization_seconds=statistics.median(
            sample.serialization_seconds for sample in samples
        ),
        transport_seconds=statistics.median(
            sample.transport_seconds for sample in samples
        ),
        decoding_seconds=statistics.median(
            sample.decoding_seconds for sample in samples
        ),
        feature_projection_seconds=statistics.median(
            sample.feature_projection_seconds for sample in samples
        ),
        shapes_insertion_seconds=statistics.median(
            sample.shapes_insertion_seconds for sample in samples
        ),
        roi_manager_construction_seconds=statistics.median(
            sample.roi_manager_construction_seconds for sample in samples
        ),
        roi_table_refresh_seconds=statistics.median(
            sample.roi_table_refresh_seconds for sample in samples
        ),
        selection_synchronization_seconds=statistics.median(
            sample.selection_synchronization_seconds for sample in samples
        ),
        paint_settlement_seconds=statistics.median(
            sample.paint_settlement_seconds for sample in samples
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-count", type=int, default=4_097)
    parser.add_argument("--vertex-count", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--skip-allocation",
        action="store_true",
        help="Skip the slower tracemalloc-instrumented sample.",
    )
    arguments = parser.parse_args()
    summary = benchmark_napari_roi_streaming(
        roi_count=arguments.roi_count,
        vertex_count=arguments.vertex_count,
        repeats=arguments.repeats,
        track_allocations=not arguments.skip_allocation,
    )
    print(json.dumps(summary.to_json_dict(), indent=2))


if __name__ == "__main__":
    main()
