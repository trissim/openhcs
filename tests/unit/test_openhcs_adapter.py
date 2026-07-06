from __future__ import annotations

from pathlib import Path

from benchmark.adapters.openhcs import (
    OpenHCSAdapter,
    OpenHCSRunRequest,
    RuntimeExecutionCacheWritePolicy,
)
from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.source_schema_workspace import SourceSchemaImageSetSelection


def test_openhcs_adapter_stores_injected_product_config_and_source_selection() -> None:
    global_config = GlobalPipelineConfig(num_workers=2, use_threading=True)
    selection = SourceSchemaImageSetSelection(
        well_filter=("A01",),
        max_image_set_count=1,
    )
    adapter = OpenHCSAdapter(
        global_config=global_config,
        source_schema_image_set_selection=selection,
    )

    assert adapter.global_config is global_config
    assert adapter.source_schema_image_set_selection is selection


def test_openhcs_run_request_carries_source_schema_selection() -> None:
    selection = SourceSchemaImageSetSelection(
        well_filter=("A01",),
        max_image_set_count=1,
    )
    request = OpenHCSRunRequest(
        dataset_path=Path("/tmp/dataset"),
        pipeline_name="pipeline",
        pipeline_params={},
        metrics=(),
        output_dir=Path("/tmp/out"),
        source_schema_image_set_selection=selection,
    )

    assert request.source_schema_image_set_selection is selection


def test_runtime_execution_cache_policy_disables_without_manifest() -> None:
    request = OpenHCSRunRequest(
        dataset_path=Path("/tmp/dataset"),
        pipeline_name="pipeline",
        pipeline_params={"runtime_execution_cache_key": {"case": "x"}},
        metrics=(),
        output_dir=Path("/tmp/out"),
    )

    policy = RuntimeExecutionCacheWritePolicy.for_request(request)

    assert not policy.write_manifest


def test_runtime_execution_cache_policy_writes_single_validation_payload() -> None:
    request = OpenHCSRunRequest(
        dataset_path=Path("/tmp/dataset"),
        pipeline_name="pipeline",
        pipeline_params={
            "runtime_execution_cache_manifest": "/tmp/out/cache.json",
            "runtime_execution_cache_key": {"case": "x"},
            "compare_image_outputs": False,
        },
        metrics=(),
        output_dir=Path("/tmp/out"),
    )

    policy = RuntimeExecutionCacheWritePolicy.for_request(request)

    assert policy.write_manifest
