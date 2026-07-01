from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from benchmark.adapters.openhcs import (
    OpenHCSAdapter,
    OpenHCSPipelineGenerationPolicy,
    OpenHCSRunRequest,
    RuntimeExecutionCacheWritePolicy,
    _cacheable_runtime_records,
)
from benchmark.converter.execution_validation import (
    CPPipeInfrastructureProfile,
)
from openhcs.core.artifacts import ArtifactKind
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


def test_runtime_execution_cache_policy_disables_discarded_candidate_runs() -> None:
    request = OpenHCSRunRequest(
        dataset_path=Path("/tmp/dataset"),
        pipeline_name="pipeline",
        pipeline_params={
            "runtime_execution_cache_manifest": "/tmp/out/cache.json",
            "runtime_execution_cache_key": {"case": "x"},
            "cache_candidate_measurement_snapshot": False,
        },
        metrics=(),
        output_dir=Path("/tmp/out"),
    )

    policy = RuntimeExecutionCacheWritePolicy.for_request(request)

    assert not policy.write_manifest
    assert not policy.include_image_records
    assert not policy.include_non_image_records


def test_runtime_execution_cache_policy_uses_snapshots_for_value_only_runs() -> None:
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
    assert not policy.include_image_records
    assert policy.include_non_image_records


def test_pipeline_generation_policy_preserves_no_export_terminal_images() -> None:
    request = OpenHCSRunRequest(
        dataset_path=Path("/tmp/dataset"),
        pipeline_name="pipeline",
        pipeline_params={"compare_image_outputs": False},
        metrics=(),
        output_dir=Path("/tmp/out"),
    )

    policy = OpenHCSPipelineGenerationPolicy.from_request(
        request,
        CPPipeInfrastructureProfile(
            exports_tables=False,
            exports_images=False,
            image_export_specs=(),
        ),
    )

    assert policy.prune_dead_unmaterialized_artifact_steps
    assert not policy.materialize_skipped_save_images
    assert policy.materialize_terminal_images


def test_pipeline_generation_policy_prunes_terminal_images_for_table_exports() -> None:
    request = OpenHCSRunRequest(
        dataset_path=Path("/tmp/dataset"),
        pipeline_name="pipeline",
        pipeline_params={"compare_image_outputs": False},
        metrics=(),
        output_dir=Path("/tmp/out"),
    )

    policy = OpenHCSPipelineGenerationPolicy.from_request(
        request,
        CPPipeInfrastructureProfile(
            exports_tables=True,
            exports_images=False,
            image_export_specs=(),
        ),
    )

    assert policy.prune_dead_unmaterialized_artifact_steps
    assert not policy.materialize_skipped_save_images
    assert not policy.materialize_terminal_images


def test_non_image_runtime_cache_excludes_array_payload_kinds() -> None:
    image_record = SimpleNamespace(key=SimpleNamespace(kind=ArtifactKind.IMAGE))
    object_label_record = SimpleNamespace(
        key=SimpleNamespace(kind=ArtifactKind.OBJECT_LABELS)
    )
    measurement_record = SimpleNamespace(
        key=SimpleNamespace(kind=ArtifactKind.MEASUREMENTS)
    )
    relationship_record = SimpleNamespace(
        key=SimpleNamespace(kind=ArtifactKind.RELATIONSHIPS)
    )

    assert _cacheable_runtime_records(
        (
            image_record,
            object_label_record,
            measurement_record,
            relationship_record,
        ),
        include_image_records=False,
    ) == (measurement_record, relationship_record)
