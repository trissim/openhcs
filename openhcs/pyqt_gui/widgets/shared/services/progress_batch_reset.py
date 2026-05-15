"""Shared progress reset utilities for new compile/execute batches."""

from __future__ import annotations

from openhcs.core.progress.projection import ExecutionRuntimeProjection


def reset_progress_views_for_new_batch(
    host, projection: ExecutionRuntimeProjection | None = None
) -> ExecutionRuntimeProjection:
    """Clear stale execution progress and reset host-facing projections.

    This removes all prior execution snapshots from the shared tracker so each
    new batch starts with a clean subtree.
    """
    runtime_projection = projection or ExecutionRuntimeProjection()

    execution_ids = list(host._progress_tracker.get_execution_ids())
    for execution_id in execution_ids:
        host._progress_tracker.clear_execution(execution_id)

    host.runtime_progress_projection = runtime_projection
    host.execution_server_info = None
    host.update_item_list()
    return runtime_projection
