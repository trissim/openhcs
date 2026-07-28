"""Analysis result consolidation for completed compiled plate executions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.orchestrator.execution_result import (
    ExecutionResult,
    RuntimeExecutionObservation,
)
from openhcs.core.runtime_stores import StoredRuntimeValue
from openhcs.core.steps.function_artifact_materialization import (
    observed_materialized_artifact_output_paths,
)
from openhcs.microscopes.microscope_base import MicroscopeHandler
from openhcs.processing.backends.analysis.consolidate_analysis_results import (
    consolidate_analysis_file_groups,
)


logger = logging.getLogger(__name__)


def consolidate_analysis_outputs(
    compiled_contexts: Mapping[str, ProcessingContext],
    execution_results: Mapping[str, ExecutionResult],
    microscope_handler: MicroscopeHandler,
    *,
    plate_runtime_observation: RuntimeExecutionObservation,
) -> None:
    """Consolidate analysis tables materialized by this execution only."""

    first_context = next(iter(compiled_contexts.values()))
    analysis_consolidation_config = first_context.analysis_consolidation_config

    if not analysis_consolidation_config.enabled:
        logger.info("⏭️ CONSOLIDATION: Disabled")
        return

    try:
        runtime_observations = tuple(
            result.runtime_observation for result in execution_results.values()
        ) + (plate_runtime_observation,)
        analysis_files_by_directory = execution_analysis_files(
            compiled_contexts,
            runtime_observations,
        )
        if not analysis_files_by_directory:
            return

        successful_dirs, failed_dirs = consolidate_analysis_file_groups(
            analysis_files_by_directory=analysis_files_by_directory,
            plate_path=Path(first_context.plate_path),
            analysis_consolidation_config=analysis_consolidation_config,
            plate_metadata_config=first_context.plate_metadata_config,
            filename_parser=microscope_handler.parser,
        )

        if successful_dirs:
            logger.info(
                "✅ CONSOLIDATION: %d directories consolidated",
                len(successful_dirs),
            )
        if failed_dirs:
            logger.warning(
                "⚠️ CONSOLIDATION: %d directories failed",
                len(failed_dirs),
            )
    except Exception as exc:
        logger.error("❌ CONSOLIDATION: Failed with error: %s", exc, exc_info=True)


def execution_analysis_files(
    compiled_contexts: Mapping[str, ProcessingContext],
    runtime_observations: tuple[RuntimeExecutionObservation, ...],
) -> Mapping[Path, tuple[Path, ...]]:
    """Group exact writer outputs from execution-owned runtime observations."""

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

    files_by_directory: dict[Path, list[Path]] = {}
    seen_paths: set[Path] = set()
    for context_key, records in records_by_context.items():
        context = compiled_contexts[context_key]
        current_records = tuple(records)
        for step_plan in context.step_plans.values():
            for output_path in observed_materialized_artifact_output_paths(
                step_plan,
                context,
                current_records,
            ):
                if output_path in seen_paths:
                    continue
                seen_paths.add(output_path)
                files_by_directory.setdefault(output_path.parent, []).append(
                    output_path
                )

    return {
        results_directory: tuple(output_paths)
        for results_directory, output_paths in files_by_directory.items()
    }
