#!/usr/bin/env python3
"""Run compact, measurement-first assay demos through one MCP session.

The showcase deliberately attaches to an already-running OpenHCS UI bridge. It
does not start, stop, or replace UI, runtime, or viewer processes. Each scenario
uses public MCP capabilities to generate one bounded plate, inspect the compiled
artifact plan, apply the corresponding Plate Manager document, execute it, read
the registered live-measurements state surface, and inspect materialized results.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from openhcs.agent.ui_bridge_actions import PlateManagerAction
from openhcs.agent.ui_bridge_identities import (
    PlateManagerOrchestratorCodeDocumentIdentity,
    PlateManagerStateSurfaceIdentityDeclaration,
    PlateManagerWidgetIdentity,
)
from openhcs.mcp.dev_client import McpDevClient


ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("PYTHON_BIN", sys.executable)
DEFAULT_OUTPUT_ROOT = ROOT / "mcp_outputs" / "assay_showcase"
ORCHESTRATOR_DOCUMENT_ID = (
    PlateManagerOrchestratorCodeDocumentIdentity.require_value()
)
WELL = "A01"
UI_TIMEOUT_MS = 2_000


class ShowcaseFailure(RuntimeError):
    """An MCP showcase acceptance condition was not met."""


@dataclass(frozen=True, slots=True)
class StageBudget:
    """Explicit wall-clock budget for one public showcase stage."""

    seconds: float

    def scaled(self, scale: float) -> float:
        return self.seconds * scale


@dataclass(frozen=True, slots=True)
class ScenarioBlueprint:
    """One curated rehearsal, not an alternate assay or callable registry."""

    scenario_id: str
    title: str
    biological_question: str
    wavelengths: int
    num_cells: int
    shared_cell_fraction: float
    random_seed: int
    assay_budget: StageBudget
    stage_budgets: Mapping[str, StageBudget]
    pipeline_source: Callable[[Path, Path], str] = field(repr=False, compare=False)

    def generation_arguments(self, plate_path: Path) -> list[str]:
        """Project this bounded fixture through the public generator CLI."""

        return [
            "generate-synthetic-plate",
            str(plate_path),
            "--grid-rows",
            "1",
            "--grid-cols",
            "1",
            "--tile-width",
            "96",
            "--tile-height",
            "96",
            "--overlap-percent",
            "10",
            "--stage-error-px",
            "1",
            "--wavelengths",
            str(self.wavelengths),
            "--z-stack-levels",
            "1",
            "--num-cells",
            str(self.num_cells),
            "--shared-cell-fraction",
            str(self.shared_cell_fraction),
            "--well",
            WELL,
            "--format",
            "ImageXpress",
            "--random-seed",
            str(self.random_seed),
            "--sample-file-limit",
            str(self.wavelengths),
            "--json",
        ]


@dataclass(slots=True)
class StageRecord:
    name: str
    elapsed_seconds: float
    budget_seconds: float
    ok: bool
    evidence_paths: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CommandRecord:
    label: str
    argv: list[str]
    elapsed_seconds: float
    returncode: int
    evidence_path: str


@dataclass(slots=True)
class ScenarioRunContext:
    blueprint: ScenarioBlueprint
    scenario_dir: Path
    plate_path: Path
    output_path: Path
    source_path: Path
    artifact_source_path: Path
    descriptor_path: Path
    budget_scale: float
    command_records: list[CommandRecord] = field(default_factory=list)
    stage_records: list[StageRecord] = field(default_factory=list)


def _stage_budgets(*, compile_seconds: float, run_seconds: float) -> dict[str, StageBudget]:
    """Return operational budgets shared by the small one-step rehearsals."""

    return {
        "generate_plate": StageBudget(4.0),
        "inspect_artifact_plan": StageBudget(12.0),
        "apply_ui_source": StageBudget(10.0),
        "initialize_plate": StageBudget(8.0),
        "compile_plate": StageBudget(compile_seconds),
        "run_plate": StageBudget(run_seconds),
        "read_live_measurements": StageBudget(8.0),
        "inspect_materialized_results": StageBudget(6.0),
    }


def scenario_blueprints() -> tuple[ScenarioBlueprint, ...]:
    """Return the three explicit stories exercised by this showcase.

    Output names and types are intentionally absent here. They are discovered
    from the compiled artifact plan for the exact pipeline source.
    """

    return (
        ScenarioBlueprint(
            scenario_id="primary_object_segmentation",
            title="Primary-object cell count and segmentation",
            biological_question=(
                "How many nucleus-like objects are present, and which labeled "
                "segmentation mask supports that count?"
            ),
            wavelengths=1,
            num_cells=12,
            shared_cell_fraction=1.0,
            random_seed=17,
            assay_budget=StageBudget(30.0),
            stage_budgets=_stage_budgets(compile_seconds=12.0, run_seconds=12.0),
            pipeline_source=_primary_object_source,
        ),
        ScenarioBlueprint(
            scenario_id="dual_channel_phenotype",
            title="Dual-channel positive/negative phenotype",
            biological_question=(
                "How many reference-channel cells are positive in the reporter "
                "channel, and what per-cell classification table supports it?"
            ),
            wavelengths=2,
            num_cells=14,
            shared_cell_fraction=0.55,
            random_seed=23,
            assay_budget=StageBudget(30.0),
            stage_budgets=_stage_budgets(compile_seconds=15.0, run_seconds=15.0),
            pipeline_source=_dual_channel_source,
        ),
        ScenarioBlueprint(
            scenario_id="image_colocalization",
            title="Two-channel image colocalization",
            biological_question=(
                "How strongly do two fluorescent signals overlap, according to "
                "correlation, Manders, rank-weighted, and overlap measurements?"
            ),
            wavelengths=2,
            num_cells=10,
            shared_cell_fraction=0.90,
            random_seed=31,
            assay_budget=StageBudget(30.0),
            stage_budgets=_stage_budgets(compile_seconds=12.0, run_seconds=12.0),
            pipeline_source=_colocalization_source,
        ),
    )


def _common_source_header(
    plate_path: Path,
    output_path: Path,
    *,
    processing_imports: str = "",
    extra_imports: str,
) -> str:
    return f'''# Compact MCP assay showcase; edit and save to apply.

from pathlib import Path

from openhcs.constants.input_source import InputSource
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazyPathPlanningConfig,
    LazyProcessingConfig,
    LazyWellFilterConfig,
    PipelineConfig,
)
from openhcs.core.steps.function_step import FunctionStep
{processing_imports.rstrip()}
{extra_imports.rstrip()}

plate_paths = [Path({str(plate_path)!r})]

global_config = GlobalPipelineConfig(
    auto_add_output_plate_to_plate_manager=True,
    num_workers=1,
)

per_plate_configs = {{
    Path({str(plate_path)!r}): PipelineConfig(
        well_filter_config=LazyWellFilterConfig(well_filter={WELL!r}),
        path_planning_config=LazyPathPlanningConfig(
            well_filter=0,
            global_output_folder=Path({str(output_path)!r}),
        ),
        materialization_results_path=Path({str(output_path / "results")!r}),
        materialize_runtime_artifacts=True,
    )
}}
'''


def _pipeline_data_header(plate_path: Path) -> str:
    return f'''pipeline_data = {{
    Path({str(plate_path)!r}): [
'''


def _document_tail() -> str:
    return '''    ]
}
'''


def artifact_plan_source(orchestrator_source: str) -> str:
    """Derive the separate PipelineDocument contract from strict UI source."""

    return f'''{orchestrator_source.rstrip()}

# Exact aliases required by the public PipelineDocument source contract.
pipeline_config = per_plate_configs[plate_paths[0]]
pipeline_steps = pipeline_data[plate_paths[0]]
'''


def _primary_object_source(plate_path: Path, output_path: Path) -> str:
    header = _common_source_header(
        plate_path,
        output_path,
        extra_imports='''from openhcs.processing.backends.analysis.cell_counting_cpu import (
    DetectionMethod,
    count_cells_single_channel,
)''',
    )
    return header + _pipeline_data_header(plate_path) + '''        FunctionStep(
        name="Segment nucleus-like primary objects",
        func=(
            count_cells_single_channel,
            {
                "detection_method": DetectionMethod.WATERSHED,
                "enable_preprocessing": False,
                "min_cell_area": 20,
                "max_cell_area": 800,
                "remove_border_cells": False,
                "return_segmentation_mask": True,
            },
        ),
        processing_config=LazyProcessingConfig(
            input_source=InputSource.PIPELINE_START,
        ),
        )
''' + _document_tail()


def _dual_channel_source(plate_path: Path, output_path: Path) -> str:
    header = _common_source_header(
        plate_path,
        output_path,
        processing_imports=(
            "from openhcs.constants.constants import GroupBy, VariableComponents"
        ),
        extra_imports='''from openhcs.processing.backends.analysis.count_cells_simple import (
    MetaXpressW2Settings,
    MetaXpressWavelengthSettings,
    StainedArea,
    count_cells_simple_dual_channel,
)''',
    )
    return header + _pipeline_data_header(plate_path) + '''        FunctionStep(
        name="Classify dual-channel reporter phenotype",
        func=(
            count_cells_simple_dual_channel,
            {
                "w1": MetaXpressWavelengthSettings(
                    channel_index=0,
                    approx_min_width=6.0,
                    approx_max_width=20.0,
                    intensity_above_local_background=2500.0,
                ),
                "w2": MetaXpressW2Settings(
                    channel_index=1,
                    approx_min_width=6.0,
                    approx_max_width=20.0,
                    intensity_above_local_background=1500.0,
                    stained_area=StainedArea.NUCLEUS,
                ),
                "minimum_stained_area": 20.0,
            },
        ),
        processing_config=LazyProcessingConfig(
            variable_components=[VariableComponents.CHANNEL],
            group_by=GroupBy.NONE,
            input_source=InputSource.PIPELINE_START,
        ),
        )
''' + _document_tail()


def _colocalization_source(plate_path: Path, output_path: Path) -> str:
    header = _common_source_header(
        plate_path,
        output_path,
        processing_imports=(
            "from openhcs.constants.constants import GroupBy, VariableComponents"
        ),
        extra_imports='''from openhcs.processing.backends.cellprofiler.colocalization import (
    measure_colocalization,
)
from openhcs.processing.backends.cellprofiler.spreadsheet_export import (
    export_to_spreadsheet,
)''',
    )
    return header + _pipeline_data_header(plate_path) + '''        FunctionStep(
        name="Measure two-channel colocalization",
        func=(
            measure_colocalization,
            {
                "channel_1": 0,
                "channel_2": 1,
                "threshold_percent": 15.0,
                "do_correlation": True,
                "do_manders": True,
                "do_rwc": True,
                "do_overlap": True,
                "do_costes": False,
            },
        ),
        processing_config=LazyProcessingConfig(
            variable_components=[VariableComponents.CHANNEL],
            group_by=GroupBy.NONE,
            input_source=InputSource.PIPELINE_START,
        ),
        ),
        FunctionStep(
        name="Export colocalization measurements",
        func=(
            export_to_spreadsheet,
            {
                "add_image_metadata": True,
                "add_image_file_names": True,
                "output_directory": "colocalization_tables",
                "export_all_measurement_types": True,
                "add_filename_prefix": True,
                "filename_prefix": "Colocalization_",
            },
        ),
        )
''' + _document_tail()


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _response_has_errors(payload: Mapping[str, Any]) -> bool:
    if payload.get("errors"):
        return True
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return True
    for result in results:
        if not isinstance(result, Mapping) or result.get("mcp_error") is True:
            return True
        result_payloads = result.get("payloads")
        if not isinstance(result_payloads, list) or not result_payloads:
            return True
        for item in result_payloads:
            if isinstance(item, Mapping) and item.get("errors"):
                return True
    return False


def _first_payload(
    response: Mapping[str, Any], tool_name: str | None = None
) -> dict[str, Any]:
    results = response.get("results")
    if not isinstance(results, list):
        raise ShowcaseFailure("MCP command returned no structured result list.")
    for result in results:
        if not isinstance(result, Mapping):
            continue
        if tool_name is not None and result.get("tool") != tool_name:
            continue
        payloads = result.get("payloads")
        if isinstance(payloads, Sequence) and payloads and isinstance(
            payloads[0], Mapping
        ):
            return dict(payloads[0])
    raise ShowcaseFailure(f"No payload returned for {tool_name or 'MCP command'}.")


def _ui_args(ctx: ScenarioRunContext) -> list[str]:
    return [
        "--descriptor-file-path",
        str(ctx.descriptor_path),
        "--timeout-ms",
        str(UI_TIMEOUT_MS),
    ]


def _run_command(
    client: McpDevClient,
    ctx: ScenarioRunContext,
    label: str,
    argv: list[str],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    execution = client.execute(argv, timeout_seconds=timeout_seconds)
    elapsed = time.perf_counter() - started
    evidence_path = (
        ctx.scenario_dir
        / "commands"
        / f"{len(ctx.command_records):02d}_{label}.json"
    )
    _json_write(
        evidence_path,
        {
            "argv": argv,
            "elapsed_seconds": elapsed,
            "returncode": execution.returncode,
            "payload": execution.payload,
            "server_stderr_tail": execution.server_stderr_tail,
        },
    )
    ctx.command_records.append(
        CommandRecord(
            label=label,
            argv=argv,
            elapsed_seconds=elapsed,
            returncode=execution.returncode,
            evidence_path=str(evidence_path),
        )
    )
    if execution.returncode != 0 or _response_has_errors(execution.payload):
        raise ShowcaseFailure(f"{label} failed; see {evidence_path}.")
    return execution.payload


def _run_stage(
    ctx: ScenarioRunContext,
    name: str,
    operation: Callable[[], Any],
) -> Any:
    budget = ctx.blueprint.stage_budgets[name].scaled(ctx.budget_scale)
    command_start = len(ctx.command_records)
    started = time.perf_counter()
    try:
        result = operation()
    except Exception:
        elapsed = time.perf_counter() - started
        ctx.stage_records.append(
            StageRecord(
                name=name,
                elapsed_seconds=elapsed,
                budget_seconds=budget,
                ok=False,
                evidence_paths=[
                    record.evidence_path
                    for record in ctx.command_records[command_start:]
                ],
            )
        )
        raise
    elapsed = time.perf_counter() - started
    ok = elapsed <= budget
    ctx.stage_records.append(
        StageRecord(
            name=name,
            elapsed_seconds=elapsed,
            budget_seconds=budget,
            ok=ok,
            evidence_paths=[
                record.evidence_path for record in ctx.command_records[command_start:]
            ],
        )
    )
    if not ok:
        raise ShowcaseFailure(
            f"{ctx.blueprint.scenario_id}:{name} took {elapsed:.2f}s, "
            f"exceeding its {budget:.2f}s budget."
        )
    return result


def _artifact_contracts(plan: Mapping[str, Any]) -> dict[str, Any]:
    steps = plan.get("steps")
    workspace = plan.get("source_workspace")
    if plan.get("axis_count") != 1 or not isinstance(steps, list) or not steps:
        raise ShowcaseFailure("Artifact plan must contain exactly one bounded axis.")
    if not isinstance(workspace, Mapping):
        raise ShowcaseFailure("Artifact plan has no source-workspace projection.")
    outputs = {
        (str(output["name"]), str(output["kind"]))
        for step in steps
        if isinstance(step, Mapping)
        for output in (step.get("artifact_outputs") or ())
        if isinstance(output, Mapping) and output.get("name") and output.get("kind")
    }
    measurement_names = sorted(
        name for name, kind in outputs if kind == "measurements"
    )
    if not measurement_names:
        raise ShowcaseFailure("Assay plan declares no measurement-table output.")
    file_count = workspace.get("file_count")
    truncated_count = workspace.get("truncated_file_count")
    if not isinstance(file_count, int) or not isinstance(truncated_count, int):
        raise ShowcaseFailure("Artifact plan source bounds are incomplete.")
    if file_count <= 0 or truncated_count != 0:
        raise ShowcaseFailure(
            "Artifact plan source scope must be non-empty and untruncated."
        )
    return {
        "axis_count": plan.get("axis_count"),
        "step_count": plan.get("step_count"),
        "source_file_count": file_count,
        "source_truncated_file_count": truncated_count,
        "outputs": [
            {"name": name, "kind": kind} for name, kind in sorted(outputs)
        ],
        "measurement_names": measurement_names,
        "final_biological_outputs": [
            {"name": name, "kind": kind} for name, kind in sorted(outputs)
        ],
    }


def _live_measurement_evidence(
    surface: Mapping[str, Any],
    *,
    surface_id: str,
    plate_path: Path,
    measurement_names: Sequence[str],
) -> dict[str, Any]:
    state = surface.get("payload")
    if not isinstance(state, Mapping):
        raise ShowcaseFailure("Live-measurements state surface has no typed payload.")
    entries = state.get("entries")
    if not isinstance(entries, list):
        raise ShowcaseFailure("Live-measurements state surface has no entry list.")
    expected_plate = str(plate_path)
    matching_entries = [
        entry
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("plate_id") == expected_plate
    ]
    previews = [
        entry["preview"]
        for entry in matching_entries
        if isinstance(entry.get("preview"), Mapping)
    ]
    populated = [
        preview
        for preview in previews
        if isinstance(preview.get("row_count"), int)
        and preview["row_count"] > 0
        and isinstance(preview.get("columns"), list)
        and bool(preview["columns"])
        and isinstance(preview.get("rows"), list)
        and bool(preview["rows"])
    ]
    observed_names = {
        str(key["name"])
        for preview in populated
        if isinstance(preview.get("address"), Mapping)
        and isinstance((key := preview["address"].get("key")), Mapping)
        and key.get("name")
    }
    missing = sorted(set(measurement_names) - observed_names)
    if missing:
        raise ShowcaseFailure(
            "Live-measurements surface did not expose populated owner-declared "
            f"tables: {missing}."
        )
    return {
        "surface_id": surface_id,
        "matching_entry_count": len(matching_entries),
        "populated_preview_count": len(populated),
        "measurement_names": sorted(observed_names),
        "row_count": sum(int(preview["row_count"]) for preview in populated),
        "previews": populated,
    }


def _discover_live_measurement_surface_id(
    action_catalog: Mapping[str, Any],
    surface_catalog: Mapping[str, Any],
) -> str:
    """Follow the Results action's owner-declared state-surface relationship."""

    actions = action_catalog.get("actions")
    if not isinstance(actions, list):
        raise ShowcaseFailure("UI action catalog returned no action list.")
    related_ids: list[str] = []
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        widget_id = action.get("widget_id")
        action_id = action.get("action_id")
        if (
            widget_id == PlateManagerWidgetIdentity.require_value()
            and action_id == PlateManagerAction.VIEW_RESULTS.value
        ):
            raw_related = action.get("related_state_surface_ids")
            if isinstance(raw_related, list):
                related_ids.extend(
                    surface_id
                    for surface_id in raw_related
                    if isinstance(surface_id, str)
                    and surface_id
                    != PlateManagerStateSurfaceIdentityDeclaration.require_value()
                )

    surfaces = surface_catalog.get("surfaces")
    if not isinstance(surfaces, list):
        raise ShowcaseFailure("UI state-surface catalog returned no surface list.")
    catalog_ids: set[str] = set()
    for surface in surfaces:
        if not isinstance(surface, Mapping):
            continue
        surface_id = surface.get("surface_id")
        if isinstance(surface_id, str) and surface_id:
            catalog_ids.add(surface_id)
    matches = tuple(dict.fromkeys(related_ids))
    if len(matches) != 1:
        raise ShowcaseFailure(
            "Plate Manager Results action must declare exactly one domain result "
            f"surface in addition to plate-manager status, found {matches}."
        )
    if matches[0] not in catalog_ids:
        raise ShowcaseFailure(
            f"Results-related state surface {matches[0]!r} is absent from the "
            "public state-surface catalog."
        )
    return matches[0]


def _result_records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    query = payload.get("query")
    if not isinstance(query, Mapping) or not isinstance(query.get("records"), list):
        raise ShowcaseFailure("Selected output query returned no structured records.")
    return [dict(record) for record in query["records"] if isinstance(record, Mapping)]


def _materialized_measurement_evidence(
    records: Sequence[Mapping[str, Any]], measurement_names: Sequence[str]
) -> dict[str, Any]:
    snapshots: list[dict[str, Any]] = []
    for record in records:
        preview = record.get("preview")
        if not isinstance(preview, Mapping):
            continue
        rows = preview.get("csv_rows")
        columns = preview.get("csv_columns")
        if isinstance(rows, list) and rows and isinstance(columns, list) and columns:
            path = record.get("full_path")
            if not isinstance(path, str) or not path:
                raise ShowcaseFailure(
                    "Materialized measurement record has no canonical full_path."
                )
            snapshots.append(
                {
                    "path": path,
                    "columns": columns,
                    "row_count": len(rows),
                }
            )
    if not snapshots:
        raise ShowcaseFailure(
            "Materialized output inventory contains no non-empty measurement CSV "
            f"for the declared tables {list(measurement_names)}."
        )
    return {
        "record_count": len(records),
        "measurement_snapshot_count": len(snapshots),
        "declared_measurement_names": list(measurement_names),
        "snapshots": snapshots,
    }


def _call_tool_args(tool_name: str, arguments: Mapping[str, Any]) -> list[str]:
    return [
        "call",
        tool_name,
        "--arguments",
        json.dumps(arguments, sort_keys=True),
        "--json",
    ]


def _apply_source(
    client: McpDevClient, ctx: ScenarioRunContext
) -> dict[str, Any]:
    read = _run_command(
        client,
        ctx,
        "read_orchestrator_document",
        [
            "code-document",
            ORCHESTRATOR_DOCUMENT_ID,
            "--selection-mode",
            "all",
            "--clean",
            "--json",
            *_ui_args(ctx),
        ],
        timeout_seconds=15.0,
    )
    token = _first_payload(read).get("current_revision_token")
    if not isinstance(token, str) or not token:
        raise ShowcaseFailure("Plate Manager document has no revision token.")
    validation = _run_command(
        client,
        ctx,
        "validate_orchestrator_document",
        [
            "validate-code-document",
            ORCHESTRATOR_DOCUMENT_ID,
            "--source-file",
            str(ctx.source_path),
            "--base-revision-token",
            token,
            "--json",
            *_ui_args(ctx),
        ],
        timeout_seconds=15.0,
    )
    if _first_payload(validation).get("valid") is not True:
        raise ShowcaseFailure("Plate Manager document validation failed.")
    applied = _run_command(
        client,
        ctx,
        "apply_orchestrator_document",
        [
            "apply-code-document",
            ORCHESTRATOR_DOCUMENT_ID,
            "--source-file",
            str(ctx.source_path),
            "--base-revision-token",
            token,
            "--no-confirmation",
            "--snapshot-label",
            f"MCP assay showcase: {ctx.blueprint.title}",
            "--json",
            *_ui_args(ctx),
        ],
        timeout_seconds=20.0,
    )
    result = _first_payload(applied)
    if result.get("applied") is True and result.get("outcome") == "applied":
        return result
    operation_id = result.get("operation_id")
    if not isinstance(operation_id, str) or not operation_id:
        raise ShowcaseFailure("UI source apply returned neither completion nor operation id.")
    waited = _run_command(
        client,
        ctx,
        "wait_for_orchestrator_apply",
        _call_tool_args(
            "openhcs_ui_wait_for_operation",
            {
                "operation_id": operation_id,
                "timeout_seconds": 15.0,
                "poll_interval_seconds": 0.25,
                "connection": {
                    "descriptor_file_path": str(ctx.descriptor_path),
                    "timeout_ms": UI_TIMEOUT_MS,
                },
            },
        ),
        timeout_seconds=20.0,
    )
    terminal = _first_payload(waited)
    if terminal.get("status") != "completed" or terminal.get("outcome") != "applied":
        raise ShowcaseFailure(f"UI source apply did not complete: {terminal}.")
    return terminal


def _workflow(
    client: McpDevClient,
    ctx: ScenarioRunContext,
    workflow: str,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    response = _run_command(
        client,
        ctx,
        f"workflow_{workflow}",
        [
            "selected-workflow",
            workflow,
            "--wait",
            "--wait-selection-mode",
            "selected",
            "--wait-interval-seconds",
            "0.25",
            "--wait-timeout-seconds",
            str(timeout_seconds),
            "--json",
            *_ui_args(ctx),
        ],
        timeout_seconds=timeout_seconds + 10.0,
    )
    summary = _first_payload(response, "mcp_dev_selected_workflow_poll")
    if summary.get("poll_completed") is not True or summary.get("poll_status") != "completed":
        raise ShowcaseFailure(f"{workflow} did not reach completed state: {summary}.")
    return summary


def _open_human_results_table(
    client: McpDevClient,
    ctx: ScenarioRunContext,
) -> dict[str, Any]:
    """Invoke the declared Results action and wait for its UI dispatch."""

    invoked = _run_command(
        client,
        ctx,
        "open_human_results_table",
        [
            "invoke-action",
            PlateManagerWidgetIdentity.require_value(),
            PlateManagerAction.VIEW_RESULTS.value,
            "--json",
            *_ui_args(ctx),
        ],
        timeout_seconds=8.0,
    )
    result = _first_payload(invoked)
    if result.get("status") == "completed":
        return result
    receipt = result.get("receipt")
    operation_id = receipt.get("bridge_operation_id") if isinstance(receipt, dict) else None
    if result.get("status") != "accepted" or not isinstance(operation_id, str):
        raise ShowcaseFailure(f"Results action was not accepted: {result}.")
    waited = _run_command(
        client,
        ctx,
        "wait_for_human_results_table",
        _call_tool_args(
            "openhcs_ui_wait_for_operation",
            {
                "operation_id": operation_id,
                "timeout_seconds": 8.0,
                "poll_interval_seconds": 0.25,
                "connection": {
                    "descriptor_file_path": str(ctx.descriptor_path),
                    "timeout_ms": UI_TIMEOUT_MS,
                },
            },
        ),
        timeout_seconds=10.0,
    )
    terminal = _first_payload(waited)
    if terminal.get("status") != "completed":
        raise ShowcaseFailure(f"Results action did not complete: {terminal}.")
    return {
        "action_id": PlateManagerAction.VIEW_RESULTS.value,
        "operation_id": operation_id,
        "outcome": terminal.get("outcome"),
        "status": terminal.get("status"),
    }


def _run_scenario(
    client: McpDevClient,
    blueprint: ScenarioBlueprint,
    *,
    session_dir: Path,
    descriptor_path: Path,
    budget_scale: float,
) -> dict[str, Any]:
    scenario_dir = session_dir / blueprint.scenario_id
    plate_path = scenario_dir / "plate"
    output_path = scenario_dir / "output"
    source_path = scenario_dir / "orchestrator.py"
    artifact_source_path = scenario_dir / "pipeline.py"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    source = blueprint.pipeline_source(plate_path, output_path)
    source_path.write_text(source, encoding="utf-8")
    artifact_source_path.write_text(
        artifact_plan_source(source),
        encoding="utf-8",
    )
    ctx = ScenarioRunContext(
        blueprint=blueprint,
        scenario_dir=scenario_dir,
        plate_path=plate_path,
        output_path=output_path,
        source_path=source_path,
        artifact_source_path=artifact_source_path,
        descriptor_path=descriptor_path,
        budget_scale=budget_scale,
    )
    started = time.perf_counter()
    try:
        generation = _run_stage(
            ctx,
            "generate_plate",
            lambda: _first_payload(
                _run_command(
                    client,
                    ctx,
                    "generate_synthetic_plate",
                    blueprint.generation_arguments(plate_path),
                    timeout_seconds=blueprint.stage_budgets[
                        "generate_plate"
                    ].scaled(budget_scale),
                )
            ),
        )
        if not plate_path.is_dir():
            raise ShowcaseFailure("Synthetic plate capability did not create its output.")
        plan_payload = _run_stage(
            ctx,
            "inspect_artifact_plan",
            lambda: _first_payload(
                _run_command(
                    client,
                    ctx,
                    "inspect_artifact_plan",
                    [
                        "artifact-plan",
                        str(plate_path),
                        "--source-file",
                        str(artifact_source_path),
                        "--axis-filter",
                        WELL,
                        "--json",
                    ],
                    timeout_seconds=blueprint.stage_budgets[
                        "inspect_artifact_plan"
                    ].scaled(budget_scale),
                ),
                "openhcs_inspect_pipeline_source_artifact_plan",
            ),
        )
        contracts = _artifact_contracts(plan_payload)
        if contracts["source_file_count"] != blueprint.wavelengths:
            raise ShowcaseFailure(
                "Compiled source scope does not match the assay channel count: "
                f"{contracts['source_file_count']} != {blueprint.wavelengths}."
            )
        _run_stage(ctx, "apply_ui_source", lambda: _apply_source(client, ctx))
        for stage_name, workflow_name in (
            ("initialize_plate", "init_plate"),
            ("compile_plate", "compile_plate"),
            ("run_plate", "run_plate"),
        ):
            budget = blueprint.stage_budgets[stage_name].scaled(budget_scale)
            _run_stage(
                ctx,
                stage_name,
                lambda workflow_name=workflow_name, budget=budget: _workflow(
                    client,
                    ctx,
                    workflow_name,
                    timeout_seconds=budget,
                ),
            )
        def read_live_measurements() -> tuple[str, dict[str, Any], dict[str, Any]]:
            action_catalog = _first_payload(
                _run_command(
                    client,
                    ctx,
                    "list_plate_manager_actions",
                    ["actions", PlateManagerWidgetIdentity.require_value(), "--json", *_ui_args(ctx)],
                    timeout_seconds=blueprint.stage_budgets[
                        "read_live_measurements"
                    ].scaled(budget_scale),
                )
            )
            surface_catalog = _first_payload(
                _run_command(
                    client,
                    ctx,
                    "list_state_surfaces",
                    ["state-surfaces", "--json", *_ui_args(ctx)],
                    timeout_seconds=blueprint.stage_budgets[
                        "read_live_measurements"
                    ].scaled(budget_scale),
                )
            )
            surface_id = _discover_live_measurement_surface_id(
                action_catalog,
                surface_catalog,
            )
            surface = _first_payload(
                _run_command(
                    client,
                    ctx,
                    "read_live_measurements",
                    [
                        "state-surface",
                        surface_id,
                        "--selection-mode",
                        "all",
                        "--json",
                        *_ui_args(ctx),
                    ],
                    timeout_seconds=blueprint.stage_budgets[
                        "read_live_measurements"
                    ].scaled(budget_scale),
                )
            )
            return surface_id, surface, _open_human_results_table(client, ctx)

        live_surface_id, surface_payload, human_results_table = _run_stage(
            ctx,
            "read_live_measurements",
            read_live_measurements,
        )
        live_measurements = _live_measurement_evidence(
            surface_payload,
            surface_id=live_surface_id,
            plate_path=plate_path,
            measurement_names=contracts["measurement_names"],
        )
        records_payload = _run_stage(
            ctx,
            "inspect_materialized_results",
            lambda: _first_payload(
                _run_command(
                    client,
                    ctx,
                    "inspect_materialized_results",
                    [
                        "selected-plate-files",
                        "--target",
                        "output",
                        "--kind",
                        "result",
                        "--limit",
                        "100",
                        "--include-previews",
                        "--json",
                        *_ui_args(ctx),
                    ],
                    timeout_seconds=blueprint.stage_budgets[
                        "inspect_materialized_results"
                    ].scaled(budget_scale),
                )
            ),
        )
        materialized = _materialized_measurement_evidence(
            _result_records(records_payload), contracts["measurement_names"]
        )
        elapsed = time.perf_counter() - started
        assay_budget = blueprint.assay_budget.scaled(budget_scale)
        if elapsed > assay_budget:
            raise ShowcaseFailure(
                f"{blueprint.scenario_id} took {elapsed:.2f}s, exceeding its "
                f"{assay_budget:.2f}s assay budget."
            )
        report = {
            "scenario_id": blueprint.scenario_id,
            "title": blueprint.title,
            "biological_question": blueprint.biological_question,
            "elapsed_seconds": elapsed,
            "assay_budget_seconds": assay_budget,
            "bounded_input": {
                "well": WELL,
                "sites": 1,
                "z_planes": 1,
                "wavelengths": blueprint.wavelengths,
                "plate_path": str(plate_path),
                "generator_result": generation,
            },
            "artifact_contracts": contracts,
            "live_measurements": live_measurements,
            "human_results_table": human_results_table,
            "materialized_measurements": materialized,
            "stages": [asdict(stage) for stage in ctx.stage_records],
            "commands": [asdict(command) for command in ctx.command_records],
        }
        _json_write(scenario_dir / "report.json", report)
        return report
    except Exception as exc:
        _json_write(
            scenario_dir / "failure.json",
            {
                "scenario_id": blueprint.scenario_id,
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
                "stages": [asdict(stage) for stage in ctx.stage_records],
                "commands": [asdict(command) for command in ctx.command_records],
            },
        )
        raise


def _dry_run_manifest(
    blueprints: Sequence[ScenarioBlueprint], session_dir: Path
) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    for blueprint in blueprints:
        scenario_dir = session_dir / blueprint.scenario_id
        source_path = scenario_dir / "orchestrator.py"
        artifact_source_path = scenario_dir / "pipeline.py"
        plate_path = scenario_dir / "plate"
        output_path = scenario_dir / "output"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        source = blueprint.pipeline_source(plate_path, output_path)
        source_path.write_text(source, encoding="utf-8")
        artifact_source_path.write_text(
            artifact_plan_source(source),
            encoding="utf-8",
        )
        scenarios.append(
            {
                "scenario_id": blueprint.scenario_id,
                "title": blueprint.title,
                "biological_question": blueprint.biological_question,
                "orchestrator_source": str(source_path),
                "pipeline_source": str(artifact_source_path),
                "assay_budget_seconds": blueprint.assay_budget.seconds,
                "stage_budgets_seconds": {
                    name: budget.seconds
                    for name, budget in blueprint.stage_budgets.items()
                },
                "bounded_input": {
                    "well": WELL,
                    "sites": 1,
                    "z_planes": 1,
                    "wavelengths": blueprint.wavelengths,
                },
            }
        )
    return {
        "schema_version": "openhcs.mcp.assay_showcase.v1",
        "dry_run": True,
        "persistent_mcp_session": False,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }


def run_showcase(
    blueprints: Sequence[ScenarioBlueprint],
    *,
    session_dir: Path,
    descriptor_path: Path,
    budget_scale: float,
    client_factory: Callable[..., McpDevClient] = McpDevClient,
) -> dict[str, Any]:
    """Execute every selected scenario through one persistent MCP session."""

    reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    with client_factory(PYTHON) as client:
        for blueprint in blueprints:
            reports.append(
                _run_scenario(
                    client,
                    blueprint,
                    session_dir=session_dir,
                    descriptor_path=descriptor_path,
                    budget_scale=budget_scale,
                )
            )
    report = {
        "schema_version": "openhcs.mcp.assay_showcase.v1",
        "dry_run": False,
        "persistent_mcp_session": True,
        "descriptor_file_path": str(descriptor_path),
        "elapsed_seconds": time.perf_counter() - started,
        "scenario_count": len(reports),
        "scenarios": reports,
    }
    _json_write(session_dir / "summary.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    blueprints = scenario_blueprints()
    parser = argparse.ArgumentParser(
        description=(
            "Run bounded, measurement-first OpenHCS assay demos through one "
            "persistent MCP session and an existing UI bridge."
        )
    )
    parser.add_argument("--descriptor-file-path", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(blueprint.scenario_id for blueprint in blueprints),
        help="Scenario to run; repeat to select multiple. Defaults to all.",
    )
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write sources and timing manifest without touching MCP or the UI.",
    )
    parser.add_argument("--list-scenarios", action="store_true")
    return parser.parse_args(argv)


def _session_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    available = scenario_blueprints()
    if args.list_scenarios:
        for blueprint in available:
            print(f"{blueprint.scenario_id}: {blueprint.title}")
        return 0
    if args.budget_scale <= 0:
        raise ShowcaseFailure("--budget-scale must be positive.")
    selected_ids = set(args.scenario or ())
    selected = tuple(
        blueprint
        for blueprint in available
        if not selected_ids or blueprint.scenario_id in selected_ids
    )
    session_dir = args.output_root.expanduser().resolve() / _session_id()
    session_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        report = _dry_run_manifest(selected, session_dir)
        _json_write(session_dir / "summary.json", report)
    else:
        if args.descriptor_file_path is None:
            raise ShowcaseFailure(
                "--descriptor-file-path is required unless --dry-run is used."
            )
        descriptor_path = args.descriptor_file_path.expanduser().resolve()
        if not descriptor_path.is_file():
            raise ShowcaseFailure(
                f"UI bridge descriptor does not exist: {descriptor_path}."
            )
        report = run_showcase(
            selected,
            session_dir=session_dir,
            descriptor_path=descriptor_path,
            budget_scale=args.budget_scale,
        )
    print(f"assay_showcase_session={session_dir}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ShowcaseFailure as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
