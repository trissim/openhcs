"""Code-mode source-store acceptance through the canonical ZMQ boundary."""

from __future__ import annotations
from openhcs.core.pipeline_document import PipelineDocumentAuthority

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr
from zmqruntime.execution.responses import (
    ExecutionSubmissionResponse,
    ExecutionWaitResult,
)
from zmqruntime.messages import MessageFields

from objectstate.lazy_factory import ensure_global_config_context
from openhcs.constants import Microscope
from openhcs.constants.constants import AllComponents, Backend
from openhcs.constants.input_source import InputSource
from openhcs.core.config import (
    AnalysisConsolidationConfig,
    GlobalPipelineConfig,
    LazyPathPlanningConfig,
    LazyProcessingConfig,
    LazyVFSConfig,
    MaterializationBackend,
    PipelineConfig,
)
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.image_file_serialization import ImageFileFormat
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.source_bindings import (
    LazySourceBindingsConfig,
    LazyStepSourceBindingsConfig,
    NamedSourceBinding,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceSelector,
)
from openhcs.core.source_metadata import SourceMetadataRoleView
from openhcs.core.steps.function_step import FunctionStep
from openhcs.microscopes.bioformats_adapter import (
    BioFormatsAdapterUnavailableError,
    SourcePlaneStoreAdapter,
)
from openhcs.processing.backends.processors.numpy_processor import (
    stack_percentile_normalize,
)
from openhcs.pyqt_gui.widgets.source_bindings_editor import SourceBindingsEditorValue
from openhcs.ui.shared.plate_manager_code_document import (
    PlateManagerCodeDocumentAuthority,
    PlateManagerOrchestratorCodePayload,
)
from openhcs.runtime.zmq_execution_client import (
    OpenHCSExecutionSubmission,
    ZMQExecutionClient,
)
from openhcs.runtime.zmq_execution_observation import (
    ZMQRuntimeExecutionObservationExport,
)
from polystore.bioformats_java import BioFormatsJavaContext


LIVE_ZMQ_ENV = "OPENHCS_RUN_SOURCE_STORE_ZMQ_ACCEPTANCE"
REQUIRE_FORMAT_FIXTURES_ENV = "OPENHCS_REQUIRE_SOURCE_STORE_FORMAT_FIXTURES"
DEFAULT_CZI_FIXTURE = Path(
    "/tmp/openhcs-czi-audit/fixture/Image_1_2023_08_18__14_32_31_964.czi"
)
DEFAULT_OME_TIFF_FIXTURE_ROOT = Path(
    "/tmp/openhcs-plane-store-fixtures/ome-tiff-companion"
)


class _NoJavaStores:
    """Keep generated ordinary/NGFF fixture gates independent of Java."""

    def declares_path(self, source_path: Path) -> bool:
        del source_path
        return False


def _binding(alias: str, file_name: str | None = None) -> NamedSourceBinding:
    selector = SourceSelector()
    if file_name is not None:
        selector = SourceSelector(
            filters=(
                SourceFilterClause(
                    subject=SourceFilterSubject.FILE,
                    match_type=SourceFilterMatchType.EQUALS,
                    value=file_name,
                ),
            ),
        )
    return NamedSourceBinding(alias=alias, selector=selector)


def _write_ngff_plate(path: Path, pixels: np.ndarray) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["plate"] = {
        "columns": [{"name": "01"}],
        "name": "Plate:code-mode-zmq",
        "rows": [{"name": "A"}],
        "version": "0.4",
        "wells": [{"columnIndex": 0, "path": "A/01", "rowIndex": 0}],
    }
    well = root.require_group("A/01")
    well.attrs["well"] = {"images": [{"path": "0"}], "version": "0.4"}
    image = well.require_group("0")
    image.attrs["multiscales"] = [
        {
            "axes": [
                {"name": "field", "type": "field"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [
                {
                    "coordinateTransformations": [
                        {"scale": [1.0] * 5, "type": "scale"}
                    ],
                    "path": "0",
                },
            ],
            "name": "Image:code-mode-zmq",
            "version": "0.4",
        },
    ]
    image.attrs["omero"] = {"channels": [{"label": "NGFF"}]}
    image.create_dataset("0", data=pixels[None, None, None])


def _write_mixed_stores(
    root: Path,
) -> dict[str, tuple[Path, np.ndarray]]:
    base = np.arange(12, dtype=np.uint16).reshape(3, 4)
    stores = {
        "NGFF": (root / "plate.zarr", base + 7),
        "TIFF": (root / "plain.tif", base + 17),
        "PNG": (root / "mask.png", base + 27),
    }
    _write_ngff_plate(*stores["NGFF"])
    for alias in ("TIFF", "PNG"):
        path, pixels = stores[alias]
        ImageFileFormat.require_path(path).write(path, pixels)
    return stores


def _global_config() -> GlobalPipelineConfig:
    return GlobalPipelineConfig(
        num_workers=1,
        use_threading=False,
        microscope=Microscope.AUTO,
        analysis_consolidation_config=AnalysisConsolidationConfig(enabled=False),
    )


def _source_step() -> FunctionStep:
    return FunctionStep(
        name="Normalize named source store",
        func=stack_percentile_normalize,
        processing_config=LazyProcessingConfig(
            input_source=InputSource.PIPELINE_START,
        ),
        source_bindings=LazyStepSourceBindingsConfig(enabled=True),
    )


def _code_mode_round_trip(
    *,
    plate_root: Path,
    bindings: tuple[NamedSourceBinding, ...],
    output_root: Path,
) -> tuple[str, PlateManagerOrchestratorCodePayload]:
    global_config = _global_config()
    pipeline_config = PipelineConfig(
        source_bindings_config=LazySourceBindingsConfig(bindings=bindings),
        path_planning_config=LazyPathPlanningConfig(
            global_output_folder=output_root,
        ),
        vfs_config=LazyVFSConfig(
            materialization_backend=MaterializationBackend.DISK,
        ),
    )
    payload = PlateManagerCodeDocumentAuthority.from_values(
        plate_paths=[plate_root],
        pipeline_data={plate_root: [_source_step()]},
        global_pipeline_config=global_config,
        per_plate_configs={plate_root: pipeline_config},
    )
    source = PlateManagerCodeDocumentAuthority.render(payload)
    restored = PlateManagerCodeDocumentAuthority.from_source(source)
    return source, restored


def _payload_values(
    payload: PlateManagerOrchestratorCodePayload,
    plate_root: Path,
) -> tuple[GlobalPipelineConfig, PipelineConfig, list[FunctionStep]]:
    scope_id = str(plate_root)
    if payload.global_pipeline_config is None:
        raise RuntimeError("Source-store acceptance requires a global config.")
    if payload.per_plate_configs is None:
        raise RuntimeError("Source-store acceptance requires a per-plate config.")
    return (
        payload.global_pipeline_config,
        payload.per_plate_configs[scope_id],
        payload.pipeline_data[scope_id],
    )


def _submission(
    *,
    payload: PlateManagerOrchestratorCodePayload,
    plate_root: Path,
    observation_path: Path,
    compile_artifact_id: str | None = None,
) -> OpenHCSExecutionSubmission:
    global_config, pipeline_config, pipeline_steps = _payload_values(
        payload,
        plate_root,
    )
    return OpenHCSExecutionSubmission(
        plate_id=plate_root,
        execution_plate_id=plate_root,
        pipeline_document=PipelineDocumentAuthority.from_values(
            pipeline_config=pipeline_config, pipeline_steps=pipeline_steps
        ),
        global_config=global_config,
        config_params={
            "runtime_observation_export_path": str(observation_path),
        },
        compile_artifact_id=compile_artifact_id,
    )


def test_code_mode_and_zmq_wire_preserve_mixed_store_sources(
    monkeypatch,
    tmp_path: Path,
) -> None:
    stores = _write_mixed_stores(tmp_path)
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(lambda cls: _NoJavaStores()),
    )
    bindings = tuple(
        _binding(alias, path.name) for alias, (path, _pixels) in stores.items()
    )
    source, payload = _code_mode_round_trip(
        plate_root=tmp_path,
        bindings=bindings,
        output_root=tmp_path.parent / f"{tmp_path.name}-outputs",
    )
    global_config, pipeline_config, pipeline_steps = _payload_values(payload, tmp_path)

    assert "plate_paths =" in source
    assert payload.plate_paths == (str(tmp_path),)
    assert isinstance(pipeline_config.source_bindings_config, LazySourceBindingsConfig)
    concrete_bindings = SourceBindingsEditorValue(
        pipeline_config.source_bindings_config
    ).concrete_view()
    assert [binding.alias for binding in concrete_bindings.bindings] == list(stores)
    pipeline_source = FunctionStepTransportAuthority.source_from_pipeline(
        pipeline_steps
    )
    reconstructed_steps = FunctionStepTransportAuthority.pipeline_steps_from_namespace(
        _exec_pipeline_source(pipeline_source)
    )
    assert (
        FunctionStepTransportAuthority.source_from_pipeline(reconstructed_steps)
        == pipeline_source
    )

    submission = _submission(
        payload=payload,
        plate_root=tmp_path,
        observation_path=tmp_path / "non_live_observation.pkl",
    )
    wire = ZMQExecutionClient().serialize_task(submission.compile_request())
    wire_pipeline = _exec_pipeline_source(wire[MessageFields.PIPELINE_CODE])
    wire_pipeline_document = PipelineDocumentAuthority.from_namespace(wire_pipeline)
    wire_global_config = _exec_config_source(wire[MessageFields.CONFIG_CODE])

    assert wire[MessageFields.COMPILE_ONLY] is True
    assert MessageFields.PIPELINE_CONFIG_CODE not in wire
    assert (
        FunctionStepTransportAuthority.source_from_pipeline(
            wire_pipeline["pipeline_steps"]
        )
        == pipeline_source
    )
    assert wire_global_config == global_config
    assert isinstance(
        wire_pipeline_document.pipeline_config.source_bindings_config,
        LazySourceBindingsConfig,
    )
    assert [
        binding.alias
        for binding in wire_pipeline_document.pipeline_config.source_bindings_config.bindings
    ] == list(stores)

    ensure_global_config_context(GlobalPipelineConfig, global_config)
    orchestrator = PipelineOrchestrator(
        plate_path=tmp_path,
        pipeline_config=pipeline_config,
    ).initialize()
    projection = orchestrator.source_workspace_projection()
    records_by_alias = {
        record.source_alias: (path, record)
        for path in projection.relative_virtual_paths()
        for record in (projection.source_projections_by_virtual_path[path],)
    }
    records = tuple(record for _path, record in records_by_alias.values())

    assert set(records_by_alias) == set(stores)
    assert {record.address.well for record in records} == {
        "A01",
        "mask.png",
        "plain.tif",
    }
    assert {
        (
            record.address.site,
            record.address.channel,
            record.address.z_index,
            record.address.timepoint,
        )
        for record in records
    } == {("1", "1", "1", "1")}
    assert {record.ref.backend for record in records} == {
        Backend.DISK.value,
        Backend.OME_ZARR.value,
    }
    assert len({record.ref.backend_address for record in records}) == len(records)
    assert {
        component: set(orchestrator.get_component_keys(component))
        for component in AllComponents
    } == {
        AllComponents.WELL: {"A01", "mask.png", "plain.tif"},
        AllComponents.SITE: {"1"},
        AllComponents.CHANNEL: {"1"},
        AllComponents.Z_INDEX: {"1"},
        AllComponents.TIMEPOINT: {"1"},
    }
    for alias, (source_path, pixels) in stores.items():
        virtual_path, record = records_by_alias[alias]
        filter_paths = SourceMetadataRoleView(
            record.source_metadata
        ).source_filter_paths()
        assert source_path.name in filter_paths
        assert str(source_path) in filter_paths
        np.testing.assert_array_equal(
            orchestrator.filemanager.load(
                tmp_path / virtual_path,
                Backend.VIRTUAL_WORKSPACE.value,
            ),
            pixels,
        )


def _exec_pipeline_source(source: str) -> dict[str, object]:
    namespace: dict[str, object] = {}
    exec(source, namespace)
    return namespace


def _exec_config_source(source: str) -> GlobalPipelineConfig | PipelineConfig:
    namespace: dict[str, object] = {}
    exec(source, namespace)
    config = namespace["config"]
    if not isinstance(config, (GlobalPipelineConfig, PipelineConfig)):
        raise TypeError(f"Serialized config reconstructed {type(config).__name__}.")
    return config


def test_exact_coordinate_collision_fails_in_aggregate_store(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(lambda cls: _NoJavaStores()),
    )
    pixels = np.arange(12, dtype=np.uint16).reshape(3, 4)
    _write_ngff_plate(tmp_path / "first.zarr", pixels)
    _write_ngff_plate(tmp_path / "second.zarr", pixels + 1)

    with pytest.raises(
        BioFormatsAdapterUnavailableError,
        match="Duplicate source plane address",
    ):
        SourcePlaneStoreAdapter.discover_dataset(tmp_path)


def _copy_available_format_worker_fixtures(
    tmp_path: Path,
) -> dict[str, tuple[Path, tuple[NamedSourceBinding, ...]]]:
    cases: dict[str, tuple[Path, tuple[NamedSourceBinding, ...]]] = {}
    czi_fixture = Path(os.environ.get("OPENHCS_CZI_FIXTURE", str(DEFAULT_CZI_FIXTURE)))
    ome_tiff_root = Path(
        os.environ.get(
            "OPENHCS_OME_TIFF_FIXTURE_ROOT",
            str(DEFAULT_OME_TIFF_FIXTURE_ROOT),
        )
    )

    if czi_fixture.is_file():
        czi_root = tmp_path / "czi"
        czi_root.mkdir()
        multipart_members = (
            czi_fixture,
            *sorted(
                czi_fixture.parent.glob(f"{czi_fixture.stem}(*){czi_fixture.suffix}")
            ),
        )
        for source_path in multipart_members:
            shutil.copy2(source_path, czi_root / source_path.name)
        copied_czi = czi_root / czi_fixture.name
        cases["czi"] = (czi_root, (_binding("CZI", copied_czi.name),))
    if ome_tiff_root.is_dir():
        ome_root = tmp_path / "ome_tiff"
        ome_root.mkdir()
        for source_path in sorted(ome_tiff_root.iterdir()):
            if source_path.suffix.lower() not in {".ome", ".tif", ".tiff"}:
                continue
            shutil.copy2(source_path, ome_root / source_path.name)
        cases["ome_tiff"] = (ome_root, (_binding("OME"),))

    if os.environ.get(REQUIRE_FORMAT_FIXTURES_ENV) == "1":
        missing = {"czi", "ome_tiff"}.difference(cases)
        if missing:
            raise RuntimeError(
                "Required format-worker fixtures are unavailable: "
                + ", ".join(sorted(missing))
            )
    return cases


@pytest.mark.skipif(
    os.environ.get(LIVE_ZMQ_ENV) != "1",
    reason=f"set {LIVE_ZMQ_ENV}=1 under the official30 runtime lock",
)
def test_code_mode_source_stores_compile_then_execute_over_one_zmq_session(
    tmp_path: Path,
) -> None:
    cases: dict[str, tuple[Path, tuple[NamedSourceBinding, ...]]] = {}
    mixed_root = tmp_path / "mixed"
    mixed_root.mkdir()
    stores = _write_mixed_stores(mixed_root)
    cases["mixed"] = (
        mixed_root,
        tuple(_binding(alias, path.name) for alias, (path, _pixels) in stores.items()),
    )
    cases.update(_copy_available_format_worker_fixtures(tmp_path))

    client = ZMQExecutionClient(
        port=18000 + os.getpid() % 20000,
        persistent=False,
    )
    try:
        assert client.connect(timeout=30)
        for case_name, (plate_root, bindings) in cases.items():
            _source, payload = _code_mode_round_trip(
                plate_root=plate_root,
                bindings=bindings,
                output_root=tmp_path / f"{case_name}_outputs",
            )
            observation_path = tmp_path / f"{case_name}_observation.pkl"
            submission = _submission(
                payload=payload,
                plate_root=plate_root,
                observation_path=observation_path,
            )
            compile_response = ExecutionSubmissionResponse.from_wire(
                client.submit_compile(submission)
            )
            compile_id = compile_response.require_execution_id(
                f"{case_name} source-store compilation"
            )
            ExecutionWaitResult.from_wire(
                client.wait_for_completion(compile_id)
            ).require_complete(f"{case_name} source-store compilation")

            execution_response = ExecutionSubmissionResponse.from_wire(
                client.submit_pipeline(
                    _submission(
                        payload=payload,
                        plate_root=plate_root,
                        observation_path=observation_path,
                        compile_artifact_id=compile_id,
                    )
                )
            )
            execution_id = execution_response.require_execution_id(
                f"{case_name} source-store execution"
            )
            ExecutionWaitResult.from_wire(
                client.wait_for_completion(execution_id)
            ).require_complete(f"{case_name} source-store execution")

            observation = ZMQRuntimeExecutionObservationExport.read(observation_path)
            observation.require_valid_observation()
            assert observation.axis_count > 0
            if case_name == "mixed":
                assert set(observation.execution_success_by_axis) == {
                    "A01",
                    "mask.png",
                    "plain.tif",
                }
    finally:
        client.disconnect()
