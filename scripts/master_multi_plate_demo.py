#!/usr/bin/env python3
"""Run every declared OpenHCS assay demo as one visible sequential showcase.

The master composes the existing showcase declarations into one canonical Plate
Manager document. It does not copy their pipelines. Each plate is selected and
executed in order, with a dedicated Napari endpoint whose window border uses the
exact accent projected by the running OpenHCS UI for that plate scope.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import openhcs  # noqa: F401  # Activate recorded source dependencies before externals.

# isort: split

from polystore.streaming.identity import StreamProducerIdentity, StreamProducerOrigin
from zmqruntime import TransportMode

from openhcs.agent.dto.ui_bridge import UiSelectedPlateWorkflowKind
from openhcs.agent.ui_bridge_actions import PlateManagerAction
from openhcs.agent.ui_bridge_identities import (
    PlateManagerOrchestratorCodeDocumentIdentity,
    PlateManagerStateSurfaceIdentityDeclaration,
    PlateManagerWidgetIdentity,
)
from openhcs.core.aligned_image_payload import AlignedImageSliceContext
from openhcs.core.artifacts import (
    MeasurementsArtifactType,
)
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.core.steps.function_output_manifest import (
    FunctionStepOutputProducerIdentityRequest,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.mcp.dev_client import McpDevClient
from openhcs.processing.presets.demo_contribution import PipelineDemoContribution
from openhcs.runtime.viewer_protocol import (
    ViewerControlField,
    ViewerLayerField,
    ViewerRuntimeEndpoint,
    ViewerTransportEndpoint,
)
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG
from openhcs.ui.shared.plate_manager_code_document import (
    PlateManagerCodeDocumentAuthority,
)
from scripts.mcp_assay_showcase import (
    ScenarioBlueprint,
    ShowcaseFailure,
    artifact_contracts,
    artifact_plan_source,
    complete_human_results_table_action,
    discover_live_measurement_surface_id,
    documentation_fixture_blueprint,
    live_measurement_evidence,
    scenario_blueprints,
)

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("PYTHON_BIN", sys.executable)
DEFAULT_OUTPUT_ROOT = ROOT / "mcp_outputs" / "master_multi_plate_demo"
DEFAULT_BASE_PORT = 5900
UI_TIMEOUT_MS = 2_000
DEFAULT_PRESENTATION_DWELL_SECONDS = 1.0
ORCHESTRATOR_DOCUMENT_ID = PlateManagerOrchestratorCodeDocumentIdentity.require_value()
PLATE_MANAGER_STATE_SURFACE_ID = (
    PlateManagerStateSurfaceIdentityDeclaration.require_value()
)
PLATE_MANAGER_WIDGET_ID = PlateManagerWidgetIdentity.require_value()


class MasterDemoFailure(RuntimeError):
    """A master-demo orchestration or acceptance condition failed."""


class MasterDemoSessionInvalidated(MasterDemoFailure):
    """The persistent MCP session can no longer execute the loaded source."""


def demo_plate_directory_name(title: str) -> str:
    """Return a meaningful filesystem leaf for PlateScopeIdentity display."""

    return title.replace("/", " or ")


@dataclass(frozen=True, slots=True)
class MasterDemoDefinition:
    """One production contribution plus scheduler-owned preparation arguments."""

    contribution: PipelineDemoContribution
    preparation_argv: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ScheduledDemo:
    """One declaration paired with its deterministic viewer endpoint."""

    definition: MasterDemoDefinition
    port: int


@dataclass(frozen=True, slots=True)
class CompiledPresentationContract:
    """Compiled proof for one declared visual and all measurement obligations."""

    visual_identity: StreamProducerIdentity
    measurement_names: tuple[str, ...]
    supporting_visual_identities: tuple[StreamProducerIdentity, ...] = ()


def compiled_presentation_contract(
    definition: MasterDemoDefinition,
    plan: Mapping[str, Any],
) -> CompiledPresentationContract:
    """Validate one content-owner target against its exact compiled plan."""

    contracts = artifact_contracts(plan)
    steps = plan.get("steps")
    if not isinstance(steps, list):
        raise MasterDemoFailure("Compiled artifact plan has no step records.")

    def resolve_identity(target: StreamProducerIdentity) -> StreamProducerIdentity:
        if target.origin != StreamProducerOrigin.PIPELINE.value or not target.step_name:
            raise MasterDemoFailure(
                "Declared visual is not a pipeline producer identity."
            )
        matching_steps = [
            step
            for step in steps
            if isinstance(step, Mapping) and step.get("step_name") == target.step_name
        ]
        if len(matching_steps) != 1:
            raise MasterDemoFailure(
                "Compiled artifact plan does not contain exactly one declared visual "
                f"step {target.step_name!r}."
            )
        target_step = matching_steps[0]
        streaming = target_step.get("viewer_streaming")
        if not isinstance(streaming, list) or not streaming:
            raise MasterDemoFailure(
                "Declared visual step has no compiled viewer stream."
            )

        artifact_output_kind = (
            FunctionStepOutputProducerIdentityRequest.ARTIFACT_OUTPUT_KIND
        )
        target_outputs = [
            output
            for output in (target_step.get("artifact_outputs") or ())
            if isinstance(output, Mapping)
            and output.get("name") == target.output_key
            and output.get("kind") == target.artifact_kind
        ]
        if target.output_kind == artifact_output_kind:
            if target.artifact_kind == MeasurementsArtifactType.require_value():
                raise MasterDemoFailure(
                    "Declared visual cannot be a Measurements artifact table."
                )
            if len(target_outputs) != 1:
                raise MasterDemoFailure(
                    "Compiled artifact plan does not contain exactly one declared "
                    f"visual output {target.output_key!r} of kind "
                    f"{target.artifact_kind!r}."
                )
        elif target.output_kind == AlignedImageSliceContext.MAIN_FLOW_OUTPUT_KIND:
            anonymous_main = (
                target.output_key
                == AlignedImageSliceContext.ANONYMOUS_MAIN_FLOW_OUTPUT_KEY
                and target.artifact_kind is None
            )
            named_artifact_main = (
                target.output_key
                != AlignedImageSliceContext.ANONYMOUS_MAIN_FLOW_OUTPUT_KEY
                and target.artifact_kind != MeasurementsArtifactType.require_value()
                and len(target_outputs) == 1
            )
            if not anonymous_main and not named_artifact_main:
                raise MasterDemoFailure(
                    "Declared main-flow visual identity is invalid."
                )
        else:
            raise MasterDemoFailure(
                f"Declared visual has unsupported output kind {target.output_kind!r}."
            )

        step_index = target_step.get("step_index")
        if type(step_index) is not int:
            raise MasterDemoFailure("Declared visual step has no compiled step index.")
        return replace(target, pipeline_position=step_index)

    return CompiledPresentationContract(
        visual_identity=resolve_identity(definition.contribution.presentation_identity),
        supporting_visual_identities=tuple(
            resolve_identity(identity)
            for identity in (definition.contribution.supporting_presentation_identities)
        ),
        measurement_names=tuple(contracts["measurement_names"]),
    )


def declared_visual_layer(
    state: Mapping[str, Any],
    identity: StreamProducerIdentity,
) -> Mapping[str, Any]:
    """Return the sole mounted layer for one exact compiled producer identity."""

    if state.get("observed") is not True:
        raise MasterDemoFailure("Napari final-result state was not observable.")
    layers = state.get(ViewerControlField.LAYERS.value)
    if not isinstance(layers, list):
        raise MasterDemoFailure("Napari final-result state has no layer list.")
    candidates: list[Mapping[str, Any]] = []
    for layer in layers:
        if (
            not isinstance(layer, Mapping)
            or layer.get(ViewerLayerField.MOUNTED.value) is not True
        ):
            continue
        identity_payloads = layer.get(ViewerLayerField.PRODUCER_IDENTITIES.value)
        if not isinstance(identity_payloads, list):
            continue
        identities = tuple(
            StreamProducerIdentity.from_payload(payload)
            for payload in identity_payloads
            if isinstance(payload, Mapping)
        )
        if any(producer.matches_declaration(identity) for producer in identities):
            candidates.append(layer)
    if len(candidates) != 1:
        raise MasterDemoFailure(
            "Napari must expose exactly one mounted route matching the "
            "declaration-owned visual identity; observed "
            f"{len(candidates)} for {identity.output_key!r}."
        )
    return candidates[0]


@dataclass(frozen=True, slots=True)
class FinalResultPresentation:
    """Verified declaration-owned Napari visual for one completed assay."""

    route_key: str
    title: str | None
    pipeline_position: int
    feature_row_count: int
    selected_data_indices: tuple[int, ...]
    visual_output_key: str | None = None
    visual_artifact_kind: str | None = None
    feature_selection_applied: bool = False

    @classmethod
    def from_viewer_state(
        cls,
        state: Mapping[str, Any],
        contract: CompiledPresentationContract,
    ) -> FinalResultPresentation:
        """Resolve the exact declaration-owned producer from viewer state."""

        target = contract.visual_identity
        layer = declared_visual_layer(state, target)
        route_key = layer.get(ViewerLayerField.ROUTE_KEY.value)
        title = layer.get(ViewerLayerField.TITLE.value)
        feature_row_count = layer.get(ViewerLayerField.FEATURE_ROW_COUNT.value)
        if not isinstance(route_key, str) or not route_key:
            raise MasterDemoFailure("Napari final-result layer has no route key.")
        if title is not None and not isinstance(title, str):
            raise MasterDemoFailure("Napari final-result layer title is invalid.")
        if type(feature_row_count) is not int or feature_row_count < 0:
            raise MasterDemoFailure("Napari final-result feature row count is invalid.")
        selected_indices = layer.get(
            ViewerLayerField.SELECTED_DATA_INDICES.value,
            (),
        )
        if not isinstance(selected_indices, list):
            raise MasterDemoFailure(
                "Napari final-result selected-data projection is invalid."
            )
        return cls(
            route_key=route_key,
            title=title,
            pipeline_position=target.pipeline_position,
            feature_row_count=feature_row_count,
            selected_data_indices=tuple(selected_indices),
            visual_output_key=target.output_key,
            visual_artifact_kind=target.artifact_kind,
            feature_selection_applied=bool(selected_indices),
        )


@dataclass(frozen=True, slots=True)
class MeasurementPresentation:
    """Populated live-table evidence for every compiled Measurements output."""

    surface_id: str
    measurement_names: tuple[str, ...]
    row_count: int
    populated_preview_count: int
    previews: tuple[dict[str, Any], ...]
    human_results_table: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DemoExecutionResult:
    demo_id: str
    title: str
    plate_path: str
    port: int
    scope_accent_color: str
    status: str
    elapsed_seconds: float
    final_overlay_route: str | None = None
    final_overlay_title: str | None = None
    final_overlay_pipeline_position: int | None = None
    feature_row_count: int = 0
    selected_data_indices: tuple[int, ...] = ()
    visual_output_key: str | None = None
    visual_artifact_kind: str | None = None
    feature_selection_applied: bool = False
    measurement_names: tuple[str, ...] = ()
    measurement_row_count: int = 0
    measurement_preview_count: int = 0
    measurement_previews: tuple[dict[str, Any], ...] = ()
    human_results_table: dict[str, Any] | None = None
    failed_stage: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MasterDemoReport:
    elapsed_seconds: float
    results: tuple[DemoExecutionResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "openhcs.master_multi_plate_demo.v1",
            "elapsed_seconds": self.elapsed_seconds,
            "demo_count": len(self.results),
            "completed_count": sum(
                result.status == "completed" for result in self.results
            ),
            "failed_count": sum(result.status == "failed" for result in self.results),
            "results": [asdict(result) for result in self.results],
        }


class MasterDemoOperations(ABC):
    """Side-effect boundary used by the deterministic sequential scheduler."""

    @abstractmethod
    def prepare_all(self, schedule: Sequence[ScheduledDemo]) -> None:
        """Prepare every declared demo input before scheduling begins."""

    @abstractmethod
    def register_all(
        self,
        schedule: Sequence[ScheduledDemo],
    ) -> Mapping[str, str]:
        """Register every prepared plate and return its authoritative scope."""

    @abstractmethod
    def assert_port_available(self, item: ScheduledDemo) -> None:
        """Require the demo's exact declared viewer endpoint to be available."""

    @abstractmethod
    def select_plate(self, item: ScheduledDemo) -> None:
        """Select the scheduled plate in the live application."""

    @abstractmethod
    def run_workflow(
        self,
        item: ScheduledDemo,
        workflow: UiSelectedPlateWorkflowKind,
    ) -> None:
        """Run one declared workflow for the scheduled plate."""

    @abstractmethod
    def wait_for_viewer(self, item: ScheduledDemo) -> None:
        """Wait until the scheduled viewer proves readiness."""

    @abstractmethod
    def present_visual_result(
        self,
        item: ScheduledDemo,
    ) -> FinalResultPresentation:
        """Return verified visual evidence for the completed demo."""

    @abstractmethod
    def present_measurements(
        self,
        item: ScheduledDemo,
    ) -> MeasurementPresentation:
        """Return verified measurement evidence for the completed demo."""


def built_in_demo_definitions(session_root: Path) -> tuple[MasterDemoDefinition, ...]:
    """Project every existing showcase blueprint into normalized demo content."""

    return tuple(
        demo_definition_from_blueprint(blueprint, session_root)
        for blueprint in scenario_blueprints()
    )


def demo_definition_from_blueprint(
    blueprint: ScenarioBlueprint,
    session_root: Path,
) -> MasterDemoDefinition:
    """Project one showcase declaration into normalized master-demo content."""

    plate_path = session_root / "plates" / demo_plate_directory_name(blueprint.title)
    output_path = session_root / "outputs" / blueprint.scenario_id
    payload = PlateManagerCodeDocumentAuthority.from_source(
        blueprint.pipeline_source(plate_path, output_path)
    )
    plate_scope_id = payload.plate_paths[0]
    return MasterDemoDefinition(
        contribution=PipelineDemoContribution(
            demo_id=blueprint.scenario_id,
            title=blueprint.title,
            biological_question=blueprint.biological_question,
            plate_path=Path(plate_scope_id),
            pipeline_config=payload.per_plate_configs[plate_scope_id],
            pipeline_steps=tuple(payload.pipeline_data[plate_scope_id]),
            presentation_identity=blueprint.presentation_identity,
            supporting_presentation_identities=(
                blueprint.supporting_presentation_identities
            ),
        ),
        preparation_argv=tuple(blueprint.generation_arguments(plate_path)),
    )


def documentation_fixture_demo_definition(
    session_root: Path,
) -> MasterDemoDefinition:
    """Project the declaration-selected UI documentation fixture."""

    return demo_definition_from_blueprint(
        documentation_fixture_blueprint(),
        session_root,
    )


def demo_definitions_with_contributions(
    built_in: Sequence[MasterDemoDefinition],
    contributions: Sequence[PipelineDemoContribution],
) -> tuple[MasterDemoDefinition, ...]:
    """Append only explicitly supplied contributors to the built-in inventory."""

    return (
        *built_in,
        *(
            MasterDemoDefinition(contribution=contribution)
            for contribution in contributions
        ),
    )


def exclude_demo_definitions(
    definitions: Sequence[MasterDemoDefinition],
    excluded_demo_ids: Sequence[str],
) -> tuple[MasterDemoDefinition, ...]:
    """Apply an explicit CLI selection without hidden scenario policy."""

    excluded = frozenset(excluded_demo_ids)
    known = {definition.contribution.demo_id for definition in definitions}
    unknown = excluded - known
    if unknown:
        raise ValueError(f"Cannot exclude unknown built-in demo ids: {sorted(unknown)}")
    return tuple(
        definition
        for definition in definitions
        if definition.contribution.demo_id not in excluded
    )


def load_contributions(
    factory_specs: Sequence[str],
    *,
    session_root: Path,
) -> tuple[PipelineDemoContribution, ...]:
    """Load explicitly named contributor factories without implicit discovery."""

    contributions: list[PipelineDemoContribution] = []
    for factory_spec in factory_specs:
        module_name, separator, function_name = factory_spec.partition(":")
        if not separator or not module_name or not function_name:
            raise ValueError(
                f"Contributor must use module:function syntax: {factory_spec!r}"
            )
        module = importlib.import_module(module_name)
        factory = vars(module)[function_name]
        produced = factory(session_root=session_root)
        values = produced if isinstance(produced, tuple) else (produced,)
        for value in values:
            if not isinstance(value, PipelineDemoContribution):
                raise TypeError(
                    f"Contributor factory {factory_spec!r} returned "
                    f"{type(value).__name__}; expected PipelineDemoContribution."
                )
            contributions.append(value)
    return tuple(contributions)


def build_demo_schedule(
    definitions: Sequence[MasterDemoDefinition],
    *,
    base_port: int,
) -> tuple[ScheduledDemo, ...]:
    """Pair declarations with stable, collision-transparent endpoint numbers."""

    demo_ids = tuple(definition.contribution.demo_id for definition in definitions)
    plate_paths = tuple(
        str(definition.contribution.plate_path) for definition in definitions
    )
    if len(set(demo_ids)) != len(demo_ids):
        raise ValueError("Master demo definitions contain a duplicate demo id.")
    if len(set(plate_paths)) != len(plate_paths):
        raise ValueError("Master demo definitions contain a duplicate plate identity.")
    data_ports = tuple(base_port + index for index in range(len(definitions)))
    control_ports = tuple(
        port + OPENHCS_ZMQ_CONFIG.control_port_offset for port in data_ports
    )
    if base_port <= 0 or (control_ports and control_ports[-1] > 65_535):
        raise ValueError("Master demo endpoint range is outside valid TCP/IPC ports.")
    overlap = sorted(set(data_ports).intersection(control_ports))
    if overlap:
        raise ValueError(
            "Master demo data/control endpoint ranges overlap at "
            f"{overlap}; choose a sparser schedule or another base port."
        )
    return tuple(
        ScheduledDemo(definition=definition, port=port)
        for definition, port in zip(definitions, data_ports, strict=True)
    )


def _scheduled_pipeline_config(
    item: ScheduledDemo,
    scope_accent_color: str | None = None,
) -> PipelineConfig:
    config = item.definition.contribution.pipeline_config
    return replace(
        config,
        napari_streaming_config=replace(
            config.napari_streaming_config,
            port=item.port,
            persistent=True,
            scope_accent_color=scope_accent_color,
        ),
    )


def _scheduled_pipeline_steps(
    item: ScheduledDemo,
    scope_accent_color: str | None = None,
) -> list[FunctionStep]:
    steps = deepcopy(item.definition.contribution.pipeline_steps)
    for step in steps:
        step.napari_streaming_config = replace(
            step.napari_streaming_config,
            port=item.port,
            persistent=True,
            scope_accent_color=scope_accent_color,
        )
    return steps


def render_master_document(
    schedule: Sequence[ScheduledDemo],
    scope_accent_colors: Mapping[str, str] | None = None,
) -> str:
    """Render one authoritative Plate Manager document for the entire schedule."""

    plate_paths = tuple(item.definition.contribution.plate_path for item in schedule)
    colors = {} if scope_accent_colors is None else dict(scope_accent_colors)
    if colors and set(colors) != {str(path) for path in plate_paths}:
        raise MasterDemoFailure(
            "Scope accents must cover the scheduled plate identities exactly."
        )
    payload = PlateManagerCodeDocumentAuthority.from_values(
        plate_paths=plate_paths,
        global_pipeline_config=GlobalPipelineConfig(),
        per_plate_configs={
            item.definition.contribution.plate_path: _scheduled_pipeline_config(
                item,
                colors.get(str(item.definition.contribution.plate_path)),
            )
            for item in schedule
        },
        pipeline_data={
            item.definition.contribution.plate_path: _scheduled_pipeline_steps(
                item,
                colors.get(str(item.definition.contribution.plate_path)),
            )
            for item in schedule
        },
    )
    return PlateManagerCodeDocumentAuthority.render(payload, clean_mode=True)


def artifact_plan_axis_filter(item: ScheduledDemo) -> tuple[str, ...]:
    """Project the exact bounded axes from the execution-filter authority."""

    configured = (
        item.definition.contribution.pipeline_config.well_filter_config.well_filter
    )
    if isinstance(configured, str) and configured:
        return (configured,)
    if (
        isinstance(configured, list)
        and configured
        and all(isinstance(axis_id, str) and axis_id for axis_id in configured)
    ):
        return tuple(configured)
    raise MasterDemoFailure(
        f"Demo {item.definition.contribution.demo_id!r} must declare exact axis "
        "labels in its "
        "pipeline well filter for bounded artifact-plan validation."
    )


def run_demo_schedule(
    schedule: Sequence[ScheduledDemo],
    operations: MasterDemoOperations,
) -> MasterDemoReport:
    """Register once, then execute one visible plate at a time."""

    started = time.perf_counter()
    operations.prepare_all(schedule)
    scope_colors = operations.register_all(schedule)
    expected_scopes = {
        str(item.definition.contribution.plate_path) for item in schedule
    }
    if set(scope_colors) != expected_scopes:
        raise MasterDemoFailure(
            "Plate Manager state did not project every registered demo scope exactly."
        )

    results: list[DemoExecutionResult] = []
    for item in schedule:
        item_started = time.perf_counter()
        scope_id = str(item.definition.contribution.plate_path)
        scope_accent_color = scope_colors[scope_id]
        failed_stage: str | None = None
        error: str | None = None
        presentation: FinalResultPresentation | None = None
        measurements: MeasurementPresentation | None = None
        stage = "port_check"
        try:

            def observe_stage(stage_name: str) -> None:
                nonlocal stage
                stage = stage_name

            run_scheduled_workflows(
                item,
                operations,
                stage_observer=observe_stage,
            )
            stage = "viewer_ready"
            operations.wait_for_viewer(item)
            stage = "present_measurements"
            measurements = operations.present_measurements(item)
            stage = "present_visual_result"
            presentation = operations.present_visual_result(item)
            status = "completed"
        except MasterDemoSessionInvalidated:
            raise
        except Exception as exc:  # noqa: BLE001 - isolate one failed demo and continue.
            status = "failed"
            failed_stage = stage
            error = str(exc)
        results.append(
            DemoExecutionResult(
                demo_id=item.definition.contribution.demo_id,
                title=item.definition.contribution.title,
                plate_path=scope_id,
                port=item.port,
                scope_accent_color=scope_accent_color,
                status=status,
                elapsed_seconds=time.perf_counter() - item_started,
                final_overlay_route=(
                    presentation.route_key if presentation is not None else None
                ),
                final_overlay_title=(
                    presentation.title if presentation is not None else None
                ),
                final_overlay_pipeline_position=(
                    presentation.pipeline_position if presentation is not None else None
                ),
                feature_row_count=(
                    presentation.feature_row_count if presentation is not None else 0
                ),
                selected_data_indices=(
                    presentation.selected_data_indices
                    if presentation is not None
                    else ()
                ),
                visual_output_key=(
                    presentation.visual_output_key if presentation is not None else None
                ),
                visual_artifact_kind=(
                    presentation.visual_artifact_kind
                    if presentation is not None
                    else None
                ),
                feature_selection_applied=(
                    presentation.feature_selection_applied
                    if presentation is not None
                    else False
                ),
                measurement_names=(
                    measurements.measurement_names if measurements is not None else ()
                ),
                measurement_row_count=(
                    measurements.row_count if measurements is not None else 0
                ),
                measurement_preview_count=(
                    measurements.populated_preview_count
                    if measurements is not None
                    else 0
                ),
                measurement_previews=(
                    measurements.previews if measurements is not None else ()
                ),
                human_results_table=(
                    measurements.human_results_table
                    if measurements is not None
                    else None
                ),
                failed_stage=failed_stage,
                error=error,
            )
        )
    return MasterDemoReport(
        elapsed_seconds=time.perf_counter() - started,
        results=tuple(results),
    )


def _ignore_workflow_stage(_stage_name: str) -> None:
    """Default stage observer for callers that only need execution."""


def run_scheduled_workflows(
    item: ScheduledDemo,
    operations: MasterDemoOperations,
    *,
    stage_observer: Callable[[str], None] = _ignore_workflow_stage,
) -> None:
    """Execute the declaration-owned UI workflow sequence for one scheduled demo."""

    stage_observer("port_check")
    operations.assert_port_available(item)
    stage_observer("select")
    operations.select_plate(item)
    for workflow in UiSelectedPlateWorkflowKind:
        stage_observer(workflow.name.casefold())
        operations.run_workflow(item, workflow)


class McpMasterDemoOperations(MasterDemoOperations):
    """Public MCP/UI/viewer implementation of the master operation boundary."""

    def __init__(
        self,
        *,
        client: McpDevClient,
        descriptor_path: Path,
        session_root: Path,
        workflow_timeout_seconds: float,
        presentation_dwell_seconds: float = DEFAULT_PRESENTATION_DWELL_SECONDS,
    ) -> None:
        self.client = client
        self.descriptor_path = descriptor_path
        self.session_root = session_root
        self.workflow_timeout_seconds = workflow_timeout_seconds
        self.presentation_dwell_seconds = presentation_dwell_seconds
        self.command_index = 0
        self.presentation_contracts: dict[str, CompiledPresentationContract] = {}
        self.live_measurement_surface_id: str | None = None

    def _ui_args(self) -> list[str]:
        return [
            "--descriptor-file-path",
            str(self.descriptor_path),
            "--timeout-ms",
            str(UI_TIMEOUT_MS),
        ]

    @staticmethod
    def _first_payload(
        response: Mapping[str, Any],
        *,
        tool_name: str | None = None,
    ) -> dict[str, Any]:
        results = response.get("results")
        if not isinstance(results, list):
            raise MasterDemoFailure("MCP command returned no result list.")
        for result in results:
            if not isinstance(result, Mapping):
                continue
            if tool_name is not None and result.get("tool") != tool_name:
                continue
            payloads = result.get("payloads")
            if (
                isinstance(payloads, list)
                and payloads
                and isinstance(payloads[0], Mapping)
            ):
                return dict(payloads[0])
        raise MasterDemoFailure(
            f"MCP command returned no {tool_name or 'tool'} payload."
        )

    def _execute(
        self,
        label: str,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        execution = self.client.execute(argv, timeout_seconds=timeout_seconds)
        evidence = {
            "argv": list(argv),
            "elapsed_seconds": time.perf_counter() - started,
            "returncode": execution.returncode,
            "payload": execution.payload,
            "server_stderr_tail": execution.server_stderr_tail,
        }
        evidence_path = (
            self.session_root / "commands" / f"{self.command_index:03d}_{label}.json"
        )
        self.command_index += 1
        _write_json(evidence_path, evidence)
        results = execution.payload.get("results")
        restart_required = isinstance(results, list) and any(
            isinstance(result, Mapping)
            and isinstance(result.get("payloads"), list)
            and any(
                isinstance(payload, Mapping) and payload.get("restart_required") is True
                for payload in result["payloads"]
            )
            for result in results
        )
        if restart_required:
            raise MasterDemoSessionInvalidated(
                f"{label} invalidated the persistent MCP session; see {evidence_path}."
            )
        has_result_error = not isinstance(results, list) or any(
            not isinstance(result, Mapping)
            or result.get("mcp_error") is True
            or result.get("payloads") in (None, [])
            for result in (results or ())
        )
        if (
            execution.returncode != 0
            or execution.payload.get("errors")
            or has_result_error
        ):
            raise MasterDemoFailure(f"{label} failed; see {evidence_path}.")
        return execution.payload

    def _execute_read_only(
        self,
        label: str,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        retry_window_seconds: float = 15.0,
    ) -> dict[str, Any]:
        """Retry a declared read-only projection without replaying mutation."""

        deadline = time.monotonic() + retry_window_seconds
        while True:
            try:
                return self._execute(
                    label,
                    argv,
                    timeout_seconds=timeout_seconds,
                )
            except MasterDemoSessionInvalidated:
                raise
            except MasterDemoFailure:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.25)

    def prepare_all(self, schedule: Sequence[ScheduledDemo]) -> None:
        for item in schedule:
            contribution = item.definition.contribution
            if contribution.prepare is not None:
                contribution.prepare()
            if item.definition.preparation_argv:
                self._execute(
                    f"prepare_{contribution.demo_id}",
                    item.definition.preparation_argv,
                    timeout_seconds=30.0,
                )
            pipeline_source_path = (
                self.session_root
                / "presentation_contracts"
                / f"{contribution.demo_id}.py"
            )
            pipeline_source_path.parent.mkdir(parents=True, exist_ok=True)
            pipeline_source_path.write_text(
                artifact_plan_source(render_master_document((item,))),
                encoding="utf-8",
            )
            plan = self._first_payload(
                self._execute_read_only(
                    f"artifact_plan_{contribution.demo_id}",
                    (
                        "artifact-plan",
                        str(contribution.plate_path),
                        "--source-file",
                        str(pipeline_source_path),
                        *(
                            argument
                            for axis_id in artifact_plan_axis_filter(item)
                            for argument in ("--axis-filter", axis_id)
                        ),
                        "--json",
                    ),
                    timeout_seconds=max(
                        30.0,
                        self.workflow_timeout_seconds,
                    ),
                    retry_window_seconds=30.0,
                ),
                tool_name="openhcs_inspect_pipeline_source_artifact_plan",
            )
            self.presentation_contracts[contribution.demo_id] = (
                compiled_presentation_contract(item.definition, plan)
            )

    def _presentation_contract(
        self,
        item: ScheduledDemo,
    ) -> CompiledPresentationContract:
        demo_id = item.definition.contribution.demo_id
        try:
            return self.presentation_contracts[demo_id]
        except KeyError as exc:
            raise MasterDemoFailure(
                f"Demo {demo_id!r} has no compiled presentation "
                "contract; prepare_all must complete first."
            ) from exc

    def register_all(self, schedule: Sequence[ScheduledDemo]) -> Mapping[str, str]:
        source_path = self.session_root / "master_plate_manager.py"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        self._apply_master_document(
            source_path=source_path,
            source=render_master_document(schedule),
            evidence_slug="master_document",
            snapshot_label="Master multi-plate lab-meeting demo",
        )
        scope_colors = self._scope_colors()
        self._apply_master_document(
            source_path=source_path,
            source=render_master_document(schedule, scope_colors),
            evidence_slug="scope_accent_document",
            snapshot_label="Project exact Plate Manager scope accents to viewers",
        )
        projected_colors = self._scope_colors()
        if projected_colors != scope_colors:
            raise MasterDemoFailure(
                "Plate Manager scope accents changed while projecting viewer config."
            )
        return projected_colors

    def _apply_master_document(
        self,
        *,
        source_path: Path,
        source: str,
        evidence_slug: str,
        snapshot_label: str,
    ) -> None:
        source_path.write_text(source, encoding="utf-8")
        current = self._first_payload(
            self._execute_read_only(
                f"read_{evidence_slug}",
                (
                    "code-document",
                    ORCHESTRATOR_DOCUMENT_ID,
                    "--selection-mode",
                    "all",
                    "--clean",
                    "--json",
                    *self._ui_args(),
                ),
                timeout_seconds=20.0,
                retry_window_seconds=15.0,
            )
        )
        revision = current.get("current_revision_token")
        if not isinstance(revision, str) or not revision:
            raise MasterDemoFailure("Plate Manager document has no current revision.")
        applied = self._first_payload(
            self._execute(
                f"apply_{evidence_slug}",
                (
                    "apply-code-document",
                    ORCHESTRATOR_DOCUMENT_ID,
                    "--source-file",
                    str(source_path),
                    "--base-revision-token",
                    revision,
                    "--no-confirmation",
                    "--snapshot-label",
                    snapshot_label,
                    "--json",
                    *self._ui_args(),
                ),
                timeout_seconds=30.0,
            )
        )
        if applied.get("applied") is not True:
            operation_id = applied.get("operation_id")
            if not isinstance(operation_id, str) or not operation_id:
                raise MasterDemoFailure(
                    "Master document apply returned no operation id."
                )
            self._wait_for_operation(
                operation_id,
                expected_outcome="applied",
                timeout_seconds=30.0,
            )

    def _wait_for_operation(
        self,
        operation_id: str,
        *,
        expected_outcome: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Wait through the public operation-progress owner."""

        status = self._first_payload(
            self._execute(
                "wait_for_ui_operation",
                (
                    "call",
                    "openhcs_ui_wait_for_operation_receipt",
                    "--arguments",
                    json.dumps(
                        {
                            "operation_id": operation_id,
                            "timeout_seconds": timeout_seconds,
                            "poll_interval_seconds": 0.25,
                            "connection": {
                                "descriptor_file_path": str(self.descriptor_path),
                                "timeout_ms": UI_TIMEOUT_MS,
                            },
                        },
                        sort_keys=True,
                    ),
                    "--json",
                    "--timeout-seconds",
                    str(timeout_seconds + 5.0),
                ),
                timeout_seconds=timeout_seconds + 5.0,
            )
        )
        if (
            status.get("status") != "completed"
            or status.get("outcome") != expected_outcome
        ):
            raise MasterDemoFailure(
                f"UI operation {operation_id} did not complete with "
                f"{expected_outcome!r}: {status}."
            )
        return status

    def _plate_manager_state(self, selection_mode: str) -> dict[str, Any]:
        response = self._execute_read_only(
            f"plate_manager_state_{selection_mode}",
            (
                "state-surface",
                PLATE_MANAGER_STATE_SURFACE_ID,
                "--selection-mode",
                selection_mode,
                "--json",
                *self._ui_args(),
            ),
            timeout_seconds=15.0,
            retry_window_seconds=15.0,
        )
        document = self._first_payload(response)
        payload = document.get("payload")
        if not isinstance(payload, Mapping):
            raise MasterDemoFailure("Plate Manager state surface has no payload.")
        return dict(payload)

    def _scope_colors(self) -> dict[str, str]:
        state = self._plate_manager_state("all")
        rows = state.get("rows")
        if not isinstance(rows, list):
            raise MasterDemoFailure("Plate Manager state has no row list.")
        colors: dict[str, str] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            scope_id = row.get("plate_scope_id")
            color = row.get("scope_accent_color")
            if isinstance(scope_id, str) and isinstance(color, str):
                colors[scope_id] = color
        return colors

    @staticmethod
    def _endpoint(item: ScheduledDemo) -> ViewerRuntimeEndpoint:
        return ViewerRuntimeEndpoint(
            transport=ViewerTransportEndpoint(
                host="localhost",
                port=item.port,
                transport_mode=TransportMode.IPC,
            ),
            config=OPENHCS_ZMQ_CONFIG,
        )

    def assert_port_available(self, item: ScheduledDemo) -> None:
        endpoint = self._endpoint(item)
        endpoint.remove_stale_addresses()
        if endpoint.in_use():
            raise MasterDemoFailure(
                f"Napari endpoint {item.port} is already in use; no fallback port "
                "was selected."
            )

    def wait_for_viewer(self, item: ScheduledDemo) -> None:
        if not self._endpoint(item).wait_ready(timeout=20.0, require_ready=True):
            raise MasterDemoFailure(
                f"Napari viewer {item.port} did not become ready after execution."
            )

    @staticmethod
    def _viewer_args(item: ScheduledDemo) -> tuple[str, ...]:
        return (
            "--port",
            str(item.port),
            "--transport-mode",
            TransportMode.IPC.value,
            "--timeout-ms",
            str(UI_TIMEOUT_MS),
        )

    def _viewer_state(
        self,
        item: ScheduledDemo,
        *,
        route_key: str | None = None,
    ) -> dict[str, Any]:
        route_args: tuple[str, ...] = ()
        if route_key is not None:
            route_args = ("--route-key", route_key)
        return self._first_payload(
            self._execute_read_only(
                f"viewer_state_{item.definition.contribution.demo_id}",
                (
                    "viewer-state",
                    *self._viewer_args(item),
                    *route_args,
                    "--no-component-values",
                    "--no-payload-summaries",
                    "--json",
                ),
                timeout_seconds=10.0,
                retry_window_seconds=10.0,
            )
        )

    def present_visual_result(
        self,
        item: ScheduledDemo,
    ) -> FinalResultPresentation:
        contract = self._presentation_contract(item)
        deadline = time.monotonic() + 20.0
        while True:
            state = self._viewer_state(item)
            layers = state.get(ViewerControlField.LAYERS.value)
            pipeline_layers = [
                layer
                for layer in (layers if isinstance(layers, list) else ())
                if isinstance(layer, Mapping)
                and any(
                    isinstance(payload, Mapping)
                    and StreamProducerIdentity.from_payload(payload).origin
                    == StreamProducerOrigin.PIPELINE.value
                    for payload in (
                        layer.get(ViewerLayerField.PRODUCER_IDENTITIES.value) or ()
                    )
                )
            ]
            if pipeline_layers and not any(
                layer.get(ViewerLayerField.PENDING_UPDATE.value) is True
                for layer in pipeline_layers
            ):
                presentation = FinalResultPresentation.from_viewer_state(
                    state,
                    contract,
                )
                break
            if time.monotonic() >= deadline:
                raise MasterDemoFailure(
                    f"Napari viewer {item.port} did not settle a pipeline result."
                )
            time.sleep(0.25)

        supporting_routes: tuple[str, ...] = tuple(
            str(
                declared_visual_layer(state, identity)[ViewerLayerField.ROUTE_KEY.value]
            )
            for identity in contract.supporting_visual_identities
        )
        for route_key in supporting_routes:
            supporting_navigation = self._first_payload(
                self._execute(
                    f"show_supporting_visual_{item.definition.contribution.demo_id}",
                    (
                        "navigate-viewer",
                        *self._viewer_args(item),
                        route_key,
                        "--visible",
                        "--json",
                    ),
                    timeout_seconds=10.0,
                )
            )
            if (
                supporting_navigation.get("observed") is not True
                or supporting_navigation.get("route_key") != route_key
                or supporting_navigation.get("visible") is not True
            ):
                raise MasterDemoFailure(
                    "Napari did not retain a declaration-owned supporting visual."
                )

        selection_indices: tuple[int, ...] = ()
        if presentation.feature_row_count == 1:
            selection_indices = (0,)
        elif presentation.feature_row_count == 2:
            selection_indices = (1, 0)
        elif presentation.feature_row_count > 2:
            selection_indices = (presentation.feature_row_count - 1, 1)
        final_selection_index: int | None = None
        for selection_number, data_index in enumerate(selection_indices, start=1):
            final_selection_index = data_index
            navigation = self._first_payload(
                self._execute(
                    "select_final_result_"
                    f"{item.definition.contribution.demo_id}_{selection_number}",
                    (
                        "navigate-viewer",
                        *self._viewer_args(item),
                        presentation.route_key,
                        "--visible",
                        "--selected",
                        "--data-index",
                        str(data_index),
                        "--json",
                    ),
                    timeout_seconds=10.0,
                )
            )
            if (
                navigation.get("observed") is not True
                or navigation.get("route_key") != presentation.route_key
                or navigation.get("visible") is not True
                or navigation.get("selected") is not True
                or navigation.get("feature_row_count") != presentation.feature_row_count
                or navigation.get("data_index") != data_index
                or navigation.get("selected_data_indices") != [data_index]
            ):
                raise MasterDemoFailure(
                    "Napari did not confirm the declaration-owned final visual."
                )
            if self.presentation_dwell_seconds:
                time.sleep(self.presentation_dwell_seconds)
        if not selection_indices:
            navigation = self._first_payload(
                self._execute(
                    f"select_final_result_{item.definition.contribution.demo_id}",
                    (
                        "navigate-viewer",
                        *self._viewer_args(item),
                        presentation.route_key,
                        "--visible",
                        "--selected",
                        "--json",
                    ),
                    timeout_seconds=10.0,
                )
            )
            if (
                navigation.get("observed") is not True
                or navigation.get("route_key") != presentation.route_key
                or navigation.get("visible") is not True
                or navigation.get("selected") is not True
                or navigation.get("feature_row_count") != presentation.feature_row_count
            ):
                raise MasterDemoFailure(
                    "Napari did not confirm the declaration-owned final visual."
                )

        verified_state = self._viewer_state(item)
        verified = FinalResultPresentation.from_viewer_state(verified_state, contract)
        layer = declared_visual_layer(verified_state, contract.visual_identity)
        supporting_visible = all(
            declared_visual_layer(verified_state, identity).get(
                ViewerLayerField.VISIBLE.value
            )
            is True
            for identity in contract.supporting_visual_identities
        )
        linked_selection_mismatch = (
            presentation.feature_row_count > 0
            and verified.selected_data_indices != (final_selection_index,)
        )
        if (
            verified.route_key != presentation.route_key
            or verified.feature_row_count != presentation.feature_row_count
            or linked_selection_mismatch
            or not isinstance(layer, Mapping)
            or layer.get(ViewerLayerField.VISIBLE.value) is not True
            or layer.get(ViewerLayerField.SELECTED.value) is not True
            or not supporting_visible
        ):
            raise MasterDemoFailure(
                "Napari final visual did not retain visibility/selection state."
            )
        if self.presentation_dwell_seconds and not selection_indices:
            time.sleep(self.presentation_dwell_seconds)
        return verified

    def _discover_live_measurement_surface(self) -> str:
        if self.live_measurement_surface_id is not None:
            return self.live_measurement_surface_id
        action_catalog = self._first_payload(
            self._execute_read_only(
                "list_plate_manager_actions",
                (
                    "actions",
                    PLATE_MANAGER_WIDGET_ID,
                    "--json",
                    *self._ui_args(),
                ),
                timeout_seconds=10.0,
                retry_window_seconds=10.0,
            )
        )
        surface_catalog = self._first_payload(
            self._execute_read_only(
                "list_state_surfaces",
                ("state-surfaces", "--json", *self._ui_args()),
                timeout_seconds=10.0,
                retry_window_seconds=10.0,
            )
        )
        try:
            surface_id = discover_live_measurement_surface_id(
                action_catalog,
                surface_catalog,
            )
        except ShowcaseFailure as exc:
            raise MasterDemoFailure(str(exc)) from exc
        self.live_measurement_surface_id = surface_id
        return surface_id

    def present_measurements(
        self,
        item: ScheduledDemo,
    ) -> MeasurementPresentation:
        contract = self._presentation_contract(item)
        surface_id = self._discover_live_measurement_surface()
        deadline = time.monotonic() + 20.0
        while True:
            surface = self._first_payload(
                self._execute_read_only(
                    f"live_measurements_{item.definition.contribution.demo_id}",
                    (
                        "state-surface",
                        surface_id,
                        "--selection-mode",
                        "all",
                        "--json",
                        *self._ui_args(),
                    ),
                    timeout_seconds=10.0,
                    retry_window_seconds=10.0,
                )
            )
            try:
                evidence = live_measurement_evidence(
                    surface,
                    surface_id=surface_id,
                    plate_path=item.definition.contribution.plate_path,
                    measurement_names=contract.measurement_names,
                )
                break
            except ShowcaseFailure as exc:
                if time.monotonic() >= deadline:
                    raise MasterDemoFailure(str(exc)) from exc
                time.sleep(0.25)

        action_result = self._first_payload(
            self._execute(
                f"open_results_{item.definition.contribution.demo_id}",
                (
                    "invoke-action",
                    PLATE_MANAGER_WIDGET_ID,
                    PlateManagerAction.VIEW_RESULTS.value,
                    "--json",
                    *self._ui_args(),
                ),
                timeout_seconds=10.0,
            )
        )
        try:
            human_results = complete_human_results_table_action(
                action_result,
                wait_for_operation=lambda operation_id: self._wait_for_operation(
                    operation_id,
                    expected_outcome="accepted",
                    timeout_seconds=10.0,
                ),
            )
        except ShowcaseFailure as exc:
            raise MasterDemoFailure(str(exc)) from exc
        previews = evidence.get("previews")
        if not isinstance(previews, list):
            raise MasterDemoFailure("Live measurement evidence has no previews.")
        return MeasurementPresentation(
            surface_id=surface_id,
            measurement_names=tuple(evidence["measurement_names"]),
            row_count=int(evidence["row_count"]),
            populated_preview_count=int(evidence["populated_preview_count"]),
            previews=tuple(dict(preview) for preview in previews),
            human_results_table=dict(human_results),
        )

    def select_plate(self, item: ScheduledDemo) -> None:
        tree = self._first_payload(
            self._execute_read_only(
                f"widget_tree_{item.definition.contribution.demo_id}",
                (
                    "widget-tree",
                    PLATE_MANAGER_WIDGET_ID,
                    "--output",
                    "json",
                    "--json",
                    *self._ui_args(),
                ),
                timeout_seconds=15.0,
                retry_window_seconds=15.0,
            )
        )
        actions = tree.get("actionable_widgets")
        if not isinstance(actions, list):
            raise MasterDemoFailure("Plate Manager widget tree has no actions.")
        scope_id = str(item.definition.contribution.plate_path)
        matches = [
            action
            for action in actions
            if isinstance(action, Mapping)
            and action.get("object_state_scope_id") == scope_id
            and action.get("action_role") == "item_select"
        ]
        if len(matches) != 1 or not isinstance(matches[0].get("path_id"), str):
            raise MasterDemoFailure(
                f"Plate row {scope_id!r} has no unique declared item-select action."
            )
        selected = self._first_payload(
            self._execute(
                f"select_{item.definition.contribution.demo_id}",
                (
                    "invoke-widget-action",
                    PLATE_MANAGER_WIDGET_ID,
                    str(matches[0]["path_id"]),
                    "--action-kind",
                    "item_select",
                    "--json",
                    *self._ui_args(),
                ),
                timeout_seconds=15.0,
            )
        )
        if selected.get("invoked") is not True:
            receipt = selected.get("receipt")
            operation_id = (
                receipt.get("bridge_operation_id")
                if isinstance(receipt, Mapping) and receipt.get("accepted") is True
                else None
            )
            if not isinstance(operation_id, str) or not operation_id:
                raise MasterDemoFailure(
                    f"Plate row selection was not accepted: {selected}."
                )
            self._wait_for_operation(
                operation_id,
                expected_outcome="invoked",
                timeout_seconds=10.0,
            )
        selected_state = self._plate_manager_state("selected")
        if selected_state.get("selected_scope_ids") != [scope_id]:
            raise MasterDemoFailure(
                f"Plate Manager selected scope differs from {scope_id!r}."
            )

    def run_workflow(
        self,
        item: ScheduledDemo,
        workflow: UiSelectedPlateWorkflowKind,
    ) -> None:
        payload = self._first_payload(
            self._execute(
                f"{workflow.name.casefold()}_{item.definition.contribution.demo_id}",
                (
                    "selected-workflow",
                    workflow.value,
                    "--wait",
                    "--wait-selection-mode",
                    "selected",
                    "--wait-interval-seconds",
                    "0.25",
                    "--wait-timeout-seconds",
                    str(self.workflow_timeout_seconds),
                    "--json",
                    *self._ui_args(),
                ),
                timeout_seconds=self.workflow_timeout_seconds + 15.0,
            ),
            tool_name="mcp_dev_selected_workflow_poll",
        )
        if (
            payload.get("poll_completed") is not True
            or payload.get("poll_status") != "completed"
        ):
            raise MasterDemoFailure(
                f"{workflow.value} did not complete for "
                f"{item.definition.contribution.demo_id}: {payload}."
            )


def dry_run_manifest(
    schedule: Sequence[ScheduledDemo], session_root: Path
) -> dict[str, Any]:
    source_path = session_root / "master_plate_manager.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(render_master_document(schedule), encoding="utf-8")
    return {
        "schema_version": "openhcs.master_multi_plate_demo.v1",
        "dry_run": True,
        "master_document": str(source_path),
        "demo_count": len(schedule),
        "demos": [
            {
                "demo_id": item.definition.contribution.demo_id,
                "title": item.definition.contribution.title,
                "plate_path": str(item.definition.contribution.plate_path),
                "port": item.port,
                "preparation_argv": list(item.definition.preparation_argv),
            }
            for item in schedule
        ],
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _session_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor-file-path", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT)
    parser.add_argument(
        "--contributor",
        action="append",
        default=[],
        metavar="MODULE:FUNCTION",
        help=(
            "Explicit contributor factory called with session_root=Path; repeat "
            "to append demos."
        ),
    )
    parser.add_argument(
        "--exclude-demo",
        action="append",
        default=[],
        metavar="DEMO_ID",
        help="Explicit built-in demo id to omit; repeat to omit multiple demos.",
    )
    parser.add_argument("--workflow-timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--presentation-dwell-seconds",
        type=float,
        default=DEFAULT_PRESENTATION_DWELL_SECONDS,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-demos", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workflow_timeout_seconds <= 0:
        raise ValueError("--workflow-timeout-seconds must be positive.")
    if args.presentation_dwell_seconds < 0:
        raise ValueError("--presentation-dwell-seconds must be nonnegative.")
    session_root = args.output_root.expanduser().resolve() / _session_id()
    built_in = exclude_demo_definitions(
        built_in_demo_definitions(session_root),
        args.exclude_demo,
    )
    contributions = load_contributions(
        args.contributor,
        session_root=session_root,
    )
    definitions = demo_definitions_with_contributions(built_in, contributions)
    schedule = build_demo_schedule(definitions, base_port=args.base_port)
    if args.list_demos:
        for item in schedule:
            print(
                f"{item.definition.contribution.demo_id}: "
                f"{item.definition.contribution.title} (port {item.port})"
            )
        return 0
    session_root.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        report = dry_run_manifest(schedule, session_root)
    else:
        if args.descriptor_file_path is None:
            raise MasterDemoFailure(
                "--descriptor-file-path is required unless --dry-run is used."
            )
        descriptor_path = args.descriptor_file_path.expanduser().resolve()
        with McpDevClient(PYTHON) as client:
            operations = McpMasterDemoOperations(
                client=client,
                descriptor_path=descriptor_path,
                session_root=session_root,
                workflow_timeout_seconds=args.workflow_timeout_seconds,
                presentation_dwell_seconds=args.presentation_dwell_seconds,
            )
            report = run_demo_schedule(schedule, operations).to_dict()
    _write_json(session_root / "summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 1 if report.get("failed_count", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
