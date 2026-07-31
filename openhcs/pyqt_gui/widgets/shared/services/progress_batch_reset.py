"""Shared progress reset utilities for new compile/execute batches."""

from __future__ import annotations

from openhcs.core.progress.debug_projection import RuntimeProjectionBundle


def reset_progress_views_for_new_batch(
    host,
    projection_bundle: RuntimeProjectionBundle | None = None,
) -> RuntimeProjectionBundle:
    """Clear stale execution progress and reset host-facing projections.

    This removes all prior execution snapshots from the shared tracker so each
    new batch starts with a clean subtree.
    """
    runtime_projection_bundle = projection_bundle or RuntimeProjectionBundle.empty()

    execution_ids = list(host._progress_tracker.get_execution_ids())
    for execution_id in execution_ids:
        host._progress_tracker.clear_execution(execution_id)

    host.runtime_progress_projection = runtime_projection_bundle.execution
    host.debug_runtime_projection = runtime_projection_bundle.debug
    host.update_item_list()
    return runtime_projection_bundle
