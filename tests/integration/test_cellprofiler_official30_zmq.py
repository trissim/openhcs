from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields as dataclass_fields
import json
import os
from pathlib import Path
import shutil
import traceback

import numpy as np
import pytest

from benchmark.adapters.cellprofiler import (
    DETERMINISTIC_NUMPY_DISABLED_CPU_FEATURES,
    NUMPY_DISABLED_CPU_FEATURES_ENV,
)
from benchmark.adapters.openhcs import ZMQ_RESULTS_SUMMARY_FILENAME
from benchmark.cellprofiler_comparison import (
    CellProfilerComparisonCase,
    CellProfilerComparisonObservation,
    load_comparison_cases,
    run_comparison_suite,
)
from polystore.streaming_constants import StreamingDataType
from openhcs.agent.dto.execution import ExecutionConnectionSpec
from openhcs.agent.dto.viewer import (
    ViewerWindowStateRequest,
    ViewerWindowValidationRequest,
)
from openhcs.agent.services.viewer_window_service import (
    ViewerWindowService,
    ViewerWindowValidationAuthority,
)
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazyNapariStreamingConfig,
    StreamingConfig,
    TransportMode,
    WellFilterConfig,
)
from openhcs.core.streaming_config_declarations import (
    FIJI_STREAMING_CONFIG_SPEC,
    NAPARI_STREAMING_CONFIG_SPEC,
    StreamingViewerConfigSpec,
)
from openhcs.core.orchestrator.compiled_plate_execution import (
    CompiledPlateExecutionExtras,
)
from openhcs.core.runtime_execution_validation import (
    RuntimeArtifactViewerPayloadExpectation,
    runtime_artifact_viewer_component_identity,
)
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from openhcs.runtime.viewer_protocol import (
    ViewerControlMessageRequest,
    ViewerRuntimeEndpoint,
)
from openhcs.runtime.fiji_macro_runtime import FijiMacroExecutionRequest
from openhcs.runtime.zmq_execution_observation import (
    ZMQRuntimeExecutionObservationExport,
)
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG
from zmqruntime import TcpDataControlPortPairAuthority

OFFICIAL30_MANIFEST = (
    Path(__file__).parents[2]
    / "benchmark"
    / "manifests"
    / "official30_portable_axis1.json"
)
NATIVE_REFERENCE_ROOT_ENV = "OPENHCS_CP_NATIVE_REFERENCE_ROOT"
_OFFICIAL30_CASES = load_comparison_cases(OFFICIAL30_MANIFEST)
_OFFICIAL30_NAPARI_PARAMS = tuple(
    pytest.param(case_index, case, id=f"{case_index:02d}-{case.name}")
    for case_index, case in enumerate(_OFFICIAL30_CASES)
)
_OFFICIAL30_FIJI_VIEWER_VARIANTS = (
    (FIJI_STREAMING_CONFIG_SPEC,),
    (FIJI_STREAMING_CONFIG_SPEC, NAPARI_STREAMING_CONFIG_SPEC),
)
_OFFICIAL30_FIJI_VIEWER_PARAMS = tuple(
    pytest.param(
        viewer_specs,
        case_index,
        case,
        id=(
            "+".join(spec.viewer_name for spec in viewer_specs)
            + f"-{case_index:02d}-{case.name}"
        ),
    )
    for viewer_specs in _OFFICIAL30_FIJI_VIEWER_VARIANTS
    for case_index, case in enumerate(_OFFICIAL30_CASES)
)
_FIJI_FRESH_PROCESS_IMAGE_PROBE = r"""#@ String Directory
titles = getList("image.titles");
imageCount = 0;
matchedWidth = 0;
matchedHeight = 0;
matchedChannels = 0;
matchedSlices = 0;
matchedFrames = 0;
for (index = 0; index < titles.length; index++) {
    selectWindow(titles[index]);
    getDimensions(width, height, channels, slices, frames);
    if (width > 0 && height > 0 && channels > 0 && slices > 0 && frames > 0) {
        imageCount++;
        matchedWidth = width;
        matchedHeight = height;
        matchedChannels = channels;
        matchedSlices = slices;
        matchedFrames = frames;
    }
}
newImage("openhcs_fiji_fresh_process_probe", "32-bit black", 6, 1, 1);
setPixel(0, 0, imageCount);
setPixel(1, 0, matchedWidth);
setPixel(2, 0, matchedHeight);
setPixel(3, 0, matchedChannels);
setPixel(4, 0, matchedSlices);
setPixel(5, 0, matchedFrames);
saveAs("Tiff", Directory + File.separator + "fresh_process_probe.tif");
close();
"""


@dataclass(frozen=True, slots=True)
class _Official30ViewerCaseFailure:
    case_index: int
    case_name: str
    stage: str
    detail: str

    @classmethod
    def from_exception(
        cls,
        *,
        case_index: int,
        case_name: str,
        stage: str,
        error: Exception,
    ) -> "_Official30ViewerCaseFailure":
        return cls(
            case_index=case_index,
            case_name=case_name,
            stage=stage,
            detail="".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        )


def _shutdown_viewer(endpoint: ViewerRuntimeEndpoint) -> None:
    response = ViewerControlMessageRequest(
        endpoint=endpoint,
        message_type="force_shutdown",
        timeout=2.0,
    ).send()
    assert response.succeeded(), response.payload
    assert endpoint.wait_until_released(timeout=10.0)


def _native_reference_root() -> Path:
    native_reference_root_value = os.environ.get(NATIVE_REFERENCE_ROOT_ENV)
    if native_reference_root_value is None:
        pytest.skip(f"{NATIVE_REFERENCE_ROOT_ENV} is required for official30 parity")
    disabled_cpu_features = os.environ.get(NUMPY_DISABLED_CPU_FEATURES_ENV)
    if disabled_cpu_features != DETERMINISTIC_NUMPY_DISABLED_CPU_FEATURES:
        pytest.fail(
            "Official30 fixed references require "
            f"{NUMPY_DISABLED_CPU_FEATURES_ENV}="
            f"{DETERMINISTIC_NUMPY_DISABLED_CPU_FEATURES}; got "
            f"{disabled_cpu_features!r}."
        )
    native_reference_root = Path(native_reference_root_value).resolve()
    if not native_reference_root.is_dir():
        pytest.fail(
            f"{NATIVE_REFERENCE_ROOT_ENV} is not a directory: {native_reference_root}"
        )
    return native_reference_root


def _free_zmq_port_pair(excluded: set[int]) -> int:
    """Return a currently free TCP data/control pair outside owned endpoints."""
    pair = TcpDataControlPortPairAuthority.acquire(
        OPENHCS_ZMQ_CONFIG,
        excluded=excluded,
    )
    excluded.update(pair.ports)
    return pair.data_port


def _registered_streaming_config_kwargs(
    viewer_specs: Sequence[StreamingViewerConfigSpec],
    ports_by_viewer: Mapping[str, int],
) -> dict[str, StreamingConfig]:
    """Project one viewer selection through registered config owners."""

    selected_specs = {spec.registry_key: spec for spec in viewer_specs}
    selected_viewers = {spec.viewer_name for spec in viewer_specs}
    if len(selected_specs) != len(viewer_specs) or len(selected_viewers) != len(
        viewer_specs
    ):
        raise ValueError("Official30 viewer variants require unique declarations.")
    if set(ports_by_viewer) != selected_viewers:
        raise ValueError(
            "Official30 viewer ports must exactly match enabled viewer identities."
        )

    config_kwargs: dict[str, StreamingConfig] = {}
    registered_specs: dict[str, StreamingViewerConfigSpec] = {}
    for config_type in StreamingConfig.__registry__.values():
        spec = config_type.streaming_spec
        registered_specs[spec.registry_key] = spec
        enabled = spec.registry_key in selected_specs
        init_kwargs: dict[str, object] = {
            "enabled": enabled,
            "persistent": enabled,
        }
        if enabled:
            init_kwargs.update(
                host="127.0.0.1",
                port=ports_by_viewer[spec.viewer_name],
                transport_mode=TransportMode.TCP,
            )
        config_kwargs[spec.registry_key] = config_type(**init_kwargs)

    if set(selected_specs) - set(registered_specs):
        raise ValueError("Official30 viewer variant selected an unregistered config.")
    for key, spec in selected_specs.items():
        if registered_specs[key] is not spec:
            raise ValueError(
                "Official30 viewer variant must use the registered config declaration."
            )
    return config_kwargs


def _runtime_observation(
    observation: CellProfilerComparisonObservation,
) -> ZMQRuntimeExecutionObservationExport:
    adapter_output_dir = Path(observation.openhcs.output_path).parent
    summary_path = adapter_output_dir / ZMQ_RESULTS_SUMMARY_FILENAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runtime_observation = ZMQRuntimeExecutionObservationExport.read(
        Path(summary["runtime_observation_export_path"])
    )
    runtime_observation.require_valid_observation()
    return runtime_observation


def _assert_successful_exact_observations(
    observations: tuple[CellProfilerComparisonObservation, ...],
) -> None:
    failures = tuple(
        (
            observation.case_name,
            observation.native_cellprofiler.error_message,
            observation.openhcs.error_message,
        )
        for observation in observations
        if not observation.equivalent
        or not observation.native_cellprofiler.success
        or not observation.openhcs.success
    )
    assert not failures
    assert all(not observation.openhcs.cached for observation in observations)
    assert all(
        observation.openhcs.execution_seconds is not None
        for observation in observations
    )
    assert all(observation.native_cellprofiler.cached for observation in observations)


class _CapturedViewerStateGateway:
    """Replay one exact ZMQ viewer-state response through the public validator."""

    def __init__(self, response: dict[str, object]) -> None:
        self.response = response

    def window_state(self, _request: ViewerWindowStateRequest) -> dict[str, object]:
        return self.response


def _assert_functional_viewer_state(
    observation: CellProfilerComparisonObservation,
) -> None:
    adapter_output_dir = Path(observation.openhcs.output_path).parent
    summary_path = adapter_output_dir / ZMQ_RESULTS_SUMMARY_FILENAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    states_by_port = summary[CompiledPlateExecutionExtras.RESULTS_SUMMARY_KEY]
    assert len(states_by_port) == 1
    port_text, response = next(iter(states_by_port.items()))
    assert isinstance(response, dict)
    state_response = response["payload"]
    assert isinstance(state_response, dict)

    connection = ExecutionConnectionSpec(
        port=int(port_text),
        transport_mode="ipc",
        persistent=False,
    )
    state = ViewerWindowService(
        gateway=_CapturedViewerStateGateway(state_response)
    ).window_state(ViewerWindowStateRequest(connection=connection))
    _assert_napari_state_matches_runtime(
        state=state,
        connection=connection,
        runtime_observation=_runtime_observation(observation),
    )


def _assert_napari_state_matches_runtime(
    *,
    state,
    connection: ExecutionConnectionSpec,
    runtime_observation: ZMQRuntimeExecutionObservationExport,
) -> None:
    assert state.observed
    assert not state.errors
    assert state.viewer is not None
    assert state.viewer.viewer_type == "napari"
    assert state.layer_count == len(state.layers) > 0
    assert state.component_group_count == state.layer_count
    assert state.component_item_count == sum(layer.item_count for layer in state.layers)
    assert (
        state.active_dimension_label_route is None
        or state.active_dimension_label_route
        in {layer.route_key for layer in state.layers}
    )

    validation_request = ViewerWindowValidationRequest.from_fields(
        connection=connection,
        require_nonzero_payloads=False,
    )
    window_validation = ViewerWindowValidationAuthority.validation_summary(
        connection=connection,
        request=validation_request,
        state=state,
    )
    assert window_validation.valid, window_validation
    for layer in state.layers:
        validation = ViewerWindowValidationAuthority.layer_validation_summary(
            layer,
            validation_context=validation_request,
        )
        assert validation.valid, validation
        assert layer.producer_identities
        for producer_identity in layer.producer_identities:
            assert producer_identity.origin == "pipeline"
            assert producer_identity.pipeline_position is not None
            assert producer_identity.step_name
            assert producer_identity.step_scope_id
        assert layer.item_count == layer.component_value_count
        assert layer.item_count == layer.payload_summary_count
        assert layer.item_count == len(layer.component_values)
        assert layer.item_count == len(layer.payload_summaries)
        assert not layer.component_values_truncated
        assert not layer.payload_summaries_truncated
        assert Counter(
            runtime_artifact_viewer_component_identity(values)
            for values in layer.component_values
        ) == Counter(
            runtime_artifact_viewer_component_identity(payload["components"])
            for payload in layer.payload_summaries
        )
        assert tuple(layer.axis_labels[: len(layer.stack_axes)]) == layer.stack_axes
        if layer.data_types != (StreamingDataType.IMAGE.value,):
            continue
        for axis_index, component in enumerate(layer.stack_axes):
            component_values = layer.axis_component_values[component]
            assert layer.data_shape[axis_index] == len(component_values)

    for expected in runtime_observation.expectation.artifact_viewer:
        matching_layers = tuple(
            layer
            for layer in state.layers
            if expected.producer_identity in layer.producer_identities
        )
        assert len(matching_layers) == 1
        layer = matching_layers[0]
        assert layer.mounted
        assert layer.item_count == len(expected.payloads)
        actual_payloads = tuple(
            RuntimeArtifactViewerPayloadExpectation(
                components=runtime_artifact_viewer_component_identity(
                    payload["components"]
                ),
                source_spatial_domain=SourceSpatialDomain.from_viewer_wire_mapping(
                    payload,
                    source_label="official30 Napari payload summary",
                ),
            )
            for payload in layer.payload_summaries
        )
        assert Counter(payload.identity_key for payload in actual_payloads) == Counter(
            payload.identity_key for payload in expected.payloads
        )

        payload_paths = tuple(payload["path"] for payload in layer.payload_summaries)
        assert all(isinstance(path, str) and path for path in payload_paths)

        payload_data_types = tuple(
            dict.fromkeys(payload["data_type"] for payload in layer.payload_summaries)
        )
        assert len(payload_data_types) == 1
        assert layer.data_types == payload_data_types


def _assert_live_napari_state(
    config: StreamingConfig,
    runtime_observation: ZMQRuntimeExecutionObservationExport,
) -> None:
    runtime_config = config.viewer_runtime_config()
    transport = runtime_config.transport_endpoint
    connection = ExecutionConnectionSpec(
        port=transport.port,
        host=transport.host,
        transport_mode=transport.transport_mode.value,
        persistent=True,
    )
    state = ViewerWindowService().window_state(
        ViewerWindowStateRequest.from_fields(
            connection=connection,
            timeout_ms=30_000,
            include_component_values=True,
            include_payload_summaries=True,
        )
    )
    _assert_napari_state_matches_runtime(
        state=state,
        connection=connection,
        runtime_observation=runtime_observation,
    )


def _fiji_fresh_process_image_metrics(
    *,
    config: StreamingConfig,
    macro_path: Path,
) -> tuple[int, int, int, int, int, int]:
    outputs = FijiMacroExecutionRequest.from_arrays(
        macro_path=macro_path,
        input_filenames=("unused_probe_input.tif",),
        output_filenames=("fresh_process_probe.tif",),
        directory_variable="Directory",
        macro_variables={},
        input_images=(np.zeros((1, 1), dtype=np.uint8),),
    ).send(config, timeout=300.0)
    assert len(outputs) == 1
    values = np.asarray(outputs[0]).reshape(-1)
    assert values.size == 6
    metrics = tuple(int(round(float(value))) for value in values)
    return (
        metrics[0],
        metrics[1],
        metrics[2],
        metrics[3],
        metrics[4],
        metrics[5],
    )


def _assert_fiji_fresh_process_image(
    *,
    config: StreamingConfig,
    macro_path: Path,
) -> None:
    """Require an image in the exact fresh Fiji process owned by this case."""

    metrics = _fiji_fresh_process_image_metrics(
        config=config,
        macro_path=macro_path,
    )
    image_count, width, height, channels, slices, frames = metrics
    assert image_count > 0, metrics
    assert min(width, height, channels, slices, frames) > 0


@pytest.mark.integration
def test_official30_compile_execute_and_match_native_references_over_zmq(
    tmp_path: Path,
) -> None:
    native_reference_root = _native_reference_root()

    cases = _OFFICIAL30_CASES
    assert len(cases) == 30
    global_config = GlobalPipelineConfig(
        well_filter_config=WellFilterConfig(well_filter=1),
        napari_streaming_config=LazyNapariStreamingConfig(
            enabled=False,
            persistent=False,
        ),
    )
    output_root = tmp_path / "baseline"

    observations = run_comparison_suite(
        cases,
        output_root=output_root,
        suite_id="official30-zmq-baseline",
        native_reference_root=native_reference_root,
        require_native_reference=True,
        openhcs_global_config=global_config,
        discard_openhcs_outputs=True,
        continue_on_error=True,
    )

    assert len(observations) == 30
    _assert_successful_exact_observations(observations)
    assert (output_root / "summary.csv").is_file()


def test_official30_fiji_variants_project_registered_viewer_configs() -> None:
    assert len(_OFFICIAL30_CASES) == 30
    assert len(_OFFICIAL30_FIJI_VIEWER_PARAMS) == 2 * len(_OFFICIAL30_CASES)
    assert tuple(
        parameter.values[2] for parameter in _OFFICIAL30_FIJI_VIEWER_PARAMS
    ) == _OFFICIAL30_CASES * len(_OFFICIAL30_FIJI_VIEWER_VARIANTS)
    registered_keys = set(StreamingConfig.__registry__)
    canonical_nonviewer_config = GlobalPipelineConfig(
        well_filter_config=WellFilterConfig(well_filter=1)
    )

    for viewer_specs in _OFFICIAL30_FIJI_VIEWER_VARIANTS:
        ports_by_viewer = {
            spec.viewer_name: 20_000 + index * 100
            for index, spec in enumerate(viewer_specs)
        }
        configs = _registered_streaming_config_kwargs(
            viewer_specs,
            ports_by_viewer,
        )
        variant_config = GlobalPipelineConfig(
            well_filter_config=WellFilterConfig(well_filter=1),
            **configs,
        )

        assert set(configs) == registered_keys
        assert all(
            vars(variant_config)[field.name]
            == vars(canonical_nonviewer_config)[field.name]
            for field in dataclass_fields(GlobalPipelineConfig)
            if field.name not in registered_keys
        )
        assert {
            config.viewer_type for config in configs.values() if config.enabled
        } == {spec.viewer_name for spec in viewer_specs}
        assert all(config.persistent is config.enabled for config in configs.values())
        assert {
            config.viewer_type: config.port
            for config in configs.values()
            if config.enabled
        } == ports_by_viewer


@pytest.mark.integration
@pytest.mark.parametrize(("case_index", "case"), _OFFICIAL30_NAPARI_PARAMS)
def test_official30_nonpersistent_napari_isolated_per_case(
    tmp_path: Path,
    case_index: int,
    case: CellProfilerComparisonCase,
) -> None:
    native_reference_root = _native_reference_root()
    assert len(_OFFICIAL30_CASES) == 30
    execution_port = 18000 + os.getpid() % 20000
    viewer_port = 41000 + os.getpid() % 10000
    global_config = GlobalPipelineConfig(
        well_filter_config=WellFilterConfig(well_filter=1),
        napari_streaming_config=LazyNapariStreamingConfig(
            enabled=True,
            persistent=False,
            port=viewer_port,
        ),
    )
    viewer_runtime = global_config.napari_streaming_config.viewer_runtime_config()
    endpoint = ViewerRuntimeEndpoint(
        viewer_runtime.transport_endpoint,
        OPENHCS_ZMQ_CONFIG,
    )
    observations: list[CellProfilerComparisonObservation] = []
    failures: list[_Official30ViewerCaseFailure] = []
    case_failure_count = len(failures)
    try:
        assert endpoint.wait_until_released(timeout=10.0)
    except Exception as error:
        failures.append(
            _Official30ViewerCaseFailure.from_exception(
                case_index=case_index,
                case_name=case.name,
                stage="lifecycle/pre_case_release",
                error=error,
            )
        )
        try:
            _shutdown_viewer(endpoint)
        except Exception as cleanup_error:
            failures.append(
                _Official30ViewerCaseFailure.from_exception(
                    case_index=case_index,
                    case_name=case.name,
                    stage="lifecycle/pre_case_shutdown",
                    error=cleanup_error,
                )
            )

    case_output_root = tmp_path / "napari" / f"case_{case_index:02d}"
    case_observations: tuple[CellProfilerComparisonObservation, ...] = ()
    try:
        case_observations = run_comparison_suite(
            (case,),
            output_root=case_output_root,
            suite_id=f"official30-zmq-napari-{case_index:02d}",
            native_reference_root=native_reference_root,
            require_native_reference=True,
            openhcs_global_config=global_config,
            openhcs_execution_port=execution_port,
            discard_openhcs_outputs=False,
            continue_on_error=True,
        )
        assert len(case_observations) == 1
        observations.extend(case_observations)
        _assert_successful_exact_observations(case_observations)
        assert (case_output_root / "summary.csv").is_file()
    except Exception as error:
        failures.append(
            _Official30ViewerCaseFailure.from_exception(
                case_index=case_index,
                case_name=case.name,
                stage="observation",
                error=error,
            )
        )

    if len(case_observations) == 1 and case_observations[0].openhcs.success:
        try:
            _assert_functional_viewer_state(case_observations[0])
        except Exception as error:
            failures.append(
                _Official30ViewerCaseFailure.from_exception(
                    case_index=case_index,
                    case_name=case.name,
                    stage="viewer",
                    error=error,
                )
            )

    try:
        assert endpoint.wait_until_released(timeout=10.0)
    except Exception as error:
        failures.append(
            _Official30ViewerCaseFailure.from_exception(
                case_index=case_index,
                case_name=case.name,
                stage="lifecycle/post_case_release",
                error=error,
            )
        )
        try:
            _shutdown_viewer(endpoint)
        except Exception as cleanup_error:
            failures.append(
                _Official30ViewerCaseFailure.from_exception(
                    case_index=case_index,
                    case_name=case.name,
                    stage="lifecycle/post_case_shutdown",
                    error=cleanup_error,
                )
            )

    if len(failures) == case_failure_count:
        try:
            shutil.rmtree(case_output_root / "tool_outputs")
        except Exception as error:
            failures.append(
                _Official30ViewerCaseFailure.from_exception(
                    case_index=case_index,
                    case_name=case.name,
                    stage="lifecycle/output_cleanup",
                    error=error,
                )
            )

    exact_observations = tuple(observations)
    assert not failures, tuple(failures)
    assert len(exact_observations) == 1
    _assert_successful_exact_observations(exact_observations)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("viewer_specs", "case_index", "case"),
    _OFFICIAL30_FIJI_VIEWER_PARAMS,
)
def test_official30_persistent_fiji_variants_isolated_per_case(
    tmp_path: Path,
    viewer_specs: tuple[StreamingViewerConfigSpec, ...],
    case_index: int,
    case: CellProfilerComparisonCase,
) -> None:
    native_reference_root = _native_reference_root()
    assert len(_OFFICIAL30_CASES) == 30

    excluded_ports: set[int] = set()
    execution_port = _free_zmq_port_pair(excluded_ports)
    ports_by_viewer = {
        spec.viewer_name: _free_zmq_port_pair(excluded_ports) for spec in viewer_specs
    }
    streaming_configs = _registered_streaming_config_kwargs(
        viewer_specs,
        ports_by_viewer,
    )
    global_config = GlobalPipelineConfig(
        well_filter_config=WellFilterConfig(well_filter=1),
        **streaming_configs,
    )
    endpoints = {
        spec.viewer_name: ViewerRuntimeEndpoint(
            streaming_configs[spec.registry_key]
            .viewer_runtime_config()
            .transport_endpoint,
            OPENHCS_ZMQ_CONFIG,
        )
        for spec in viewer_specs
    }
    variant_id = "+".join(spec.viewer_name for spec in viewer_specs)
    case_output_root = tmp_path / variant_id / f"case_{case_index:02d}"
    macro_path = tmp_path / f"{variant_id}_fresh_process_probe.ijm"
    macro_path.write_text(_FIJI_FRESH_PROCESS_IMAGE_PROBE, encoding="utf-8")
    failures: list[_Official30ViewerCaseFailure] = []
    observations: tuple[CellProfilerComparisonObservation, ...] = ()

    for viewer_name, endpoint in endpoints.items():
        try:
            assert endpoint.wait_until_released(timeout=10.0)
        except Exception as error:
            failures.append(
                _Official30ViewerCaseFailure.from_exception(
                    case_index=case_index,
                    case_name=case.name,
                    stage=f"{viewer_name}/lifecycle/pre_case_release",
                    error=error,
                )
            )

    if not failures:
        try:
            observations = run_comparison_suite(
                (case,),
                output_root=case_output_root,
                suite_id=f"official30-zmq-{variant_id}-{case_index:02d}",
                native_reference_root=native_reference_root,
                require_native_reference=True,
                openhcs_global_config=global_config,
                openhcs_execution_port=execution_port,
                discard_openhcs_outputs=False,
                continue_on_error=True,
            )
            assert len(observations) == 1
            _assert_successful_exact_observations(observations)
            assert (case_output_root / "summary.csv").is_file()

            runtime_observation = _runtime_observation(observations[0])
            for spec in viewer_specs:
                config = streaming_configs[spec.registry_key]
                if spec is NAPARI_STREAMING_CONFIG_SPEC:
                    _assert_live_napari_state(config, runtime_observation)
                elif spec is FIJI_STREAMING_CONFIG_SPEC:
                    _assert_fiji_fresh_process_image(
                        config=config,
                        macro_path=macro_path,
                    )
                else:
                    raise AssertionError(
                        f"Unhandled registered Official30 viewer {spec.viewer_name!r}."
                    )
        except Exception as error:
            failures.append(
                _Official30ViewerCaseFailure.from_exception(
                    case_index=case_index,
                    case_name=case.name,
                    stage=f"{variant_id}/observation",
                    error=error,
                )
            )

    for viewer_name, endpoint in endpoints.items():
        try:
            if not endpoint.wait_until_released(timeout=0.25):
                _shutdown_viewer(endpoint)
            assert endpoint.wait_until_released(timeout=10.0)
        except Exception as error:
            failures.append(
                _Official30ViewerCaseFailure.from_exception(
                    case_index=case_index,
                    case_name=case.name,
                    stage=f"{viewer_name}/lifecycle/post_case_shutdown",
                    error=error,
                )
            )

    if not failures:
        try:
            shutil.rmtree(case_output_root / "tool_outputs")
        except Exception as error:
            failures.append(
                _Official30ViewerCaseFailure.from_exception(
                    case_index=case_index,
                    case_name=case.name,
                    stage=f"{variant_id}/lifecycle/output_cleanup",
                    error=error,
                )
            )

    assert not failures, tuple(failures)
    assert len(observations) == 1
    _assert_successful_exact_observations(observations)
