"""Analysis result consolidation for completed compiled plate executions."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from openhcs.constants.constants import AllComponents
from openhcs.core.config import AnalysisConsolidationConfig, PlateMetadataConfig
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.orchestrator.execution_result import (
    ExecutionResult,
    RuntimeExecutionObservation,
)
from openhcs.core.runtime_stores import StoredRuntimeValue
from openhcs.core.steps.function_artifact_materialization import (
    RuntimeArtifactMaterialization,
    runtime_artifact_materializations_from_records,
)
from openhcs.processing.backends.analysis.consolidate_analysis_results import (
    AnalysisSummaryWriter,
    RuntimeAnalysisTableOutput,
    analysis_file_path_is_included,
    consolidate_runtime_analysis_table_output_groups,
    consolidated_analysis_summary_csv,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from polystore.filemanager import FileManager


@dataclass(frozen=True, slots=True)
class RuntimeAnalysisSummaryDestination:
    """Compiled persistent destination shared by runtime analysis outputs."""

    backend: str
    images_dir: str


@dataclass(frozen=True, slots=True)
class RuntimeAnalysisConsolidationInputs:
    """Execution-ledger tables and their compiled summary destination."""

    outputs_by_directory: Mapping[Path, tuple[RuntimeAnalysisTableOutput, ...]]
    destination: RuntimeAnalysisSummaryDestination


@dataclass(frozen=True, slots=True)
class FileManagerAnalysisSummaryWriter(AnalysisSummaryWriter):
    """Persist summaries through the compiled PolyStore destination."""

    filemanager: FileManager
    destination: RuntimeAnalysisSummaryDestination

    def write(
        self,
        summary_df: pd.DataFrame,
        *,
        output_path: Path,
        results_dir: Path,
        analysis_consolidation_config: AnalysisConsolidationConfig,
        plate_metadata_config: PlateMetadataConfig,
    ) -> None:
        content = consolidated_analysis_summary_csv(
            summary_df,
            results_dir,
            analysis_consolidation_config,
            plate_metadata_config,
        )
        self.filemanager.ensure_directory(
            output_path.parent,
            self.destination.backend,
        )
        save_kwargs = self.filemanager.contextual_save_kwargs(
            self.destination.backend,
            images_dir=self.destination.images_dir,
        )
        self.filemanager.save(
            content,
            str(output_path),
            self.destination.backend,
            **save_kwargs,
        )


def consolidate_analysis_outputs(
    compiled_contexts: Mapping[str, ProcessingContext],
    execution_results: Mapping[str, ExecutionResult],
    *,
    plate_runtime_observation: RuntimeExecutionObservation,
) -> None:
    """Consolidate analysis tables materialized by this execution only."""

    first_context = next(iter(compiled_contexts.values()))
    analysis_consolidation_config = first_context.analysis_consolidation_config

    if not analysis_consolidation_config.enabled:
        logger.info("⏭️ CONSOLIDATION: Disabled")
        return

    runtime_observations = tuple(
        result.runtime_observation for result in execution_results.values()
    ) + (plate_runtime_observation,)
    consolidation_inputs = execution_analysis_outputs(
        compiled_contexts,
        runtime_observations,
    )
    if consolidation_inputs is None:
        return

    successful_dirs, failed_dirs = consolidate_runtime_analysis_table_output_groups(
        analysis_outputs_by_directory=consolidation_inputs.outputs_by_directory,
        plate_path=Path(first_context.plate_path),
        analysis_consolidation_config=analysis_consolidation_config,
        plate_metadata_config=first_context.plate_metadata_config,
        summary_writer=FileManagerAnalysisSummaryWriter(
            filemanager=first_context.filemanager,
            destination=consolidation_inputs.destination,
        ),
    )

    if failed_dirs:
        raise RuntimeError(
            "Analysis consolidation failed for execution-owned outputs: "
            f"{failed_dirs!r}."
        )
    logger.info(
        "CONSOLIDATION: %d directories consolidated",
        len(successful_dirs),
    )


def execution_analysis_outputs(
    compiled_contexts: Mapping[str, ProcessingContext],
    runtime_observations: tuple[RuntimeExecutionObservation, ...],
) -> RuntimeAnalysisConsolidationInputs | None:
    """Group exact CSV content and its compiled persistent destination."""

    records_by_context: dict[str, list[StoredRuntimeValue]] = {}
    for observation in runtime_observations:
        for context_observation in observation.contexts:
            if context_observation.context_key not in compiled_contexts:
                raise KeyError(
                    "Runtime observation references unknown compiled context "
                    f"{context_observation.context_key!r}."
                )
            records_by_context.setdefault(
                context_observation.context_key,
                [],
            ).extend(context_observation.records)

    outputs_by_directory: dict[Path, list[RuntimeAnalysisTableOutput]] = {}
    destinations: set[RuntimeAnalysisSummaryDestination] = set()
    seen_paths: set[Path] = set()
    for context_key, records in records_by_context.items():
        context = compiled_contexts[context_key]
        current_records = tuple(records)
        for step_plan in context.step_plans.values():
            if not step_plan.runtime_artifact_materialization.has_persistent_target:
                continue
            for materialization in runtime_artifact_materializations_from_records(
                step_plan,
                context,
                current_records,
            ):
                if (
                    not materialization.spec.participates_in_runtime_export_observation()
                ):
                    continue
                destinations.add(
                    RuntimeAnalysisSummaryDestination(
                        backend=(
                            step_plan.runtime_artifact_materialization.require_persistent_backend()
                        ),
                        images_dir=str(step_plan.artifact_images_dir),
                    )
                )
                for output in materialization.outputs(step_plan, context):
                    output_path = Path(output.path)
                    if not analysis_file_path_is_included(
                        output_path,
                        context.analysis_consolidation_config,
                    ):
                        continue
                    if output_path in seen_paths:
                        continue
                    seen_paths.add(output_path)
                    outputs_by_directory.setdefault(output_path.parent, []).append(
                        runtime_analysis_table_output(
                            materialization,
                            output_path=output_path,
                            csv_content=output.require_text_content(),
                            pipeline_position=step_plan.pipeline_position,
                        )
                    )

    if not outputs_by_directory:
        return None
    if len(destinations) != 1:
        raise RuntimeError(
            "Analysis outputs do not share one compiled persistent destination: "
            f"{sorted(destinations, key=lambda value: (value.backend, value.images_dir))!r}."
        )
    return RuntimeAnalysisConsolidationInputs(
        outputs_by_directory={
            results_directory: tuple(outputs)
            for results_directory, outputs in outputs_by_directory.items()
        },
        destination=destinations.pop(),
    )


def runtime_analysis_table_output(
    materialization: RuntimeArtifactMaterialization,
    *,
    output_path: Path,
    csv_content: str,
    pipeline_position: int,
) -> RuntimeAnalysisTableOutput:
    """Project table identity from the typed runtime address, never its filename."""

    scope = materialization.record.key.scope
    well_id = scope.value_text_for_component(AllComponents.WELL)
    if well_id is None:
        raise ValueError(
            "Analysis consolidation requires a well coordinate in the runtime "
            f"artifact scope for {materialization.output_plan.name!r}."
        )
    coordinate_segments = tuple(
        f"{component.value}-{value}"
        for component, value in scope.presentation_component_values
        if component is not AllComponents.WELL
    )
    analysis_type = "_".join(
        (
            *coordinate_segments,
            f"{materialization.output_plan.name}_step{pipeline_position}",
        )
    )
    return RuntimeAnalysisTableOutput(
        path=output_path,
        well_id=well_id,
        analysis_type=analysis_type,
        csv_content=csv_content,
    )
