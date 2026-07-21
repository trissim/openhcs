from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import traceback

import pytest

from benchmark.adapters.openhcs import ZMQ_RESULTS_SUMMARY_FILENAME
from benchmark.cellprofiler_comparison import (
    CellProfilerComparisonObservation,
    load_comparison_cases,
    run_comparison_suite,
)
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
    WellFilterConfig,
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
from openhcs.runtime.zmq_execution_observation import (
    ZMQRuntimeExecutionObservationExport,
)
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG
OFFICIAL30_MANIFEST = (
    Path(__file__).parents[2]
    / "benchmark"
    / "manifests"
    / "official30_portable_axis1.json"
)
NATIVE_REFERENCE_ROOT_ENV = "OPENHCS_CP_NATIVE_REFERENCE_ROOT"


@dataclass(frozen=True, slots=True)
class _Official30NapariCaseFailure:
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
    ) -> "_Official30NapariCaseFailure":
        return cls(
            case_index=case_index,
            case_name=case_name,
            stage=stage,
            detail="".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        )


def _shutdown_nonpersistent_viewer(endpoint: ViewerRuntimeEndpoint) -> None:
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
    native_reference_root = Path(native_reference_root_value).resolve()
    if not native_reference_root.is_dir():
        pytest.fail(
            f"{NATIVE_REFERENCE_ROOT_ENV} is not a directory: "
            f"{native_reference_root}"
        )
    return native_reference_root


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
        for axis_index, component in enumerate(layer.stack_axes):
            component_values = layer.axis_component_values[component]
            assert layer.data_shape[axis_index] == len(component_values)

    observation_path = Path(summary["runtime_observation_export_path"])
    runtime_observation = ZMQRuntimeExecutionObservationExport.read(observation_path)
    runtime_observation.require_valid_observation()
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
            dict.fromkeys(
                payload["data_type"] for payload in layer.payload_summaries
            )
        )
        assert len(payload_data_types) == 1
        assert layer.data_types == payload_data_types

@pytest.mark.integration
def test_official30_compile_execute_and_match_native_references_over_zmq(
    tmp_path: Path,
) -> None:
    native_reference_root = _native_reference_root()

    cases = load_comparison_cases(OFFICIAL30_MANIFEST)
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


@pytest.mark.integration
def test_official30_nonpersistent_napari_isolated_per_case(
    tmp_path: Path,
) -> None:
    native_reference_root = _native_reference_root()
    cases = load_comparison_cases(OFFICIAL30_MANIFEST)
    assert len(cases) == 30
    global_config = GlobalPipelineConfig(
        well_filter_config=WellFilterConfig(well_filter=1),
        napari_streaming_config=LazyNapariStreamingConfig(
            enabled=True,
            persistent=False,
            port=5563,
        ),
    )
    viewer_runtime = global_config.napari_streaming_config.viewer_runtime_config()
    endpoint = ViewerRuntimeEndpoint(
        viewer_runtime.transport_endpoint,
        OPENHCS_ZMQ_CONFIG,
    )
    observations: list[CellProfilerComparisonObservation] = []
    failures: list[_Official30NapariCaseFailure] = []
    for case_index, case in enumerate(cases):
        case_failure_count = len(failures)
        try:
            assert endpoint.wait_until_released(timeout=10.0)
        except Exception as error:
            failures.append(
                _Official30NapariCaseFailure.from_exception(
                    case_index=case_index,
                    case_name=case.name,
                    stage="lifecycle/pre_case_release",
                    error=error,
                )
            )
            try:
                _shutdown_nonpersistent_viewer(endpoint)
            except Exception as cleanup_error:
                failures.append(
                    _Official30NapariCaseFailure.from_exception(
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
                discard_openhcs_outputs=False,
                continue_on_error=True,
            )
            assert len(case_observations) == 1
            observations.extend(case_observations)
            _assert_successful_exact_observations(case_observations)
            assert (case_output_root / "summary.csv").is_file()
        except Exception as error:
            failures.append(
                _Official30NapariCaseFailure.from_exception(
                    case_index=case_index,
                    case_name=case.name,
                    stage="observation",
                    error=error,
                )
            )

        if len(case_observations) == 1:
            try:
                _assert_functional_viewer_state(case_observations[0])
            except Exception as error:
                failures.append(
                    _Official30NapariCaseFailure.from_exception(
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
                _Official30NapariCaseFailure.from_exception(
                    case_index=case_index,
                    case_name=case.name,
                    stage="lifecycle/post_case_release",
                    error=error,
                )
            )
            try:
                _shutdown_nonpersistent_viewer(endpoint)
            except Exception as cleanup_error:
                failures.append(
                    _Official30NapariCaseFailure.from_exception(
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
                    _Official30NapariCaseFailure.from_exception(
                        case_index=case_index,
                        case_name=case.name,
                        stage="lifecycle/output_cleanup",
                        error=error,
                    )
                )

    exact_observations = tuple(observations)
    assert not failures, tuple(failures)
    assert len(exact_observations) == 30
    _assert_successful_exact_observations(exact_observations)
