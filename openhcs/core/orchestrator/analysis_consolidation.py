"""Analysis result consolidation for completed compiled plate executions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

from openhcs.core.context.processing_context import ProcessingContext
from openhcs.microscopes.microscope_base import MicroscopeHandler
from openhcs.processing.backends.analysis.consolidate_analysis_results import (
    consolidate_results_directories,
)


logger = logging.getLogger(__name__)


def consolidate_analysis_outputs(
    compiled_contexts: Mapping[str, ProcessingContext],
    microscope_handler: MicroscopeHandler,
) -> None:
    """Consolidate analysis outputs declared by completed compiled step plans."""

    first_context = next(iter(compiled_contexts.values()))
    analysis_consolidation_config = first_context.analysis_consolidation_config

    if not analysis_consolidation_config.enabled:
        logger.info("⏭️ CONSOLIDATION: Disabled")
        return

    try:
        results_dirs = analysis_result_dirs(compiled_contexts)
        if not results_dirs:
            return

        successful_dirs, failed_dirs = consolidate_results_directories(
            results_dirs=list(results_dirs),
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


def analysis_result_dirs(
    compiled_contexts: Mapping[str, ProcessingContext],
) -> set[Path]:
    """Return unique compiled analysis output directories."""

    results_dirs = set()
    for context in compiled_contexts.values():
        for step_plan in context.step_plans.values():
            if step_plan.analysis_results_dir is not None:
                results_dirs.add(Path(step_plan.analysis_results_dir))
            materialized_output = step_plan.materialized_output
            if (
                materialized_output is not None
                and materialized_output.analysis_results_dir is not None
            ):
                results_dirs.add(Path(materialized_output.analysis_results_dir))
    return results_dirs
