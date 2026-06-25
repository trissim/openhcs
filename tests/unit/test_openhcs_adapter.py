from __future__ import annotations

import pytest
from pathlib import Path
from types import SimpleNamespace

from benchmark.adapters.openhcs import (
    OPENHCS_AXIS_FILTER_PARAM,
    OPENHCS_MAX_AXIS_COUNT_PARAM,
    OPENHCS_NUM_WORKERS_PARAM,
    OPENHCS_USE_THREADING_PARAM,
    OpenHCSAxisSelection,
    OpenHCSBenchmarkExecutionConfig,
    OpenHCSPipelineGenerationPolicy,
    OpenHCSRunRequest,
    RuntimeExecutionCacheWritePolicy,
    _cacheable_runtime_records,
)
from benchmark.converter.execution_validation import (
    CPPipeInfrastructureFeature,
    CPPipeInfrastructureProfile,
)
from openhcs.core.artifacts import ArtifactKind


def test_openhcs_axis_selection_limits_discovered_axes_in_order() -> None:
    selection = OpenHCSAxisSelection.from_pipeline_params(
        {OPENHCS_MAX_AXIS_COUNT_PARAM: 2}
    )

    assert selection.resolve(("A01", "A02", "A03")) == ("A01", "A02")


def test_openhcs_axis_selection_intersects_explicit_axes_with_discovery_order() -> None:
    selection = OpenHCSAxisSelection.from_pipeline_params(
        {
            OPENHCS_AXIS_FILTER_PARAM: ("A03", "A01"),
            OPENHCS_MAX_AXIS_COUNT_PARAM: 1,
        }
    )

    assert selection.resolve(("A01", "A02", "A03")) == ("A01",)


def test_openhcs_axis_selection_rejects_missing_axes() -> None:
    selection = OpenHCSAxisSelection.from_pipeline_params(
        {OPENHCS_AXIS_FILTER_PARAM: ("A99",)}
    )

    with pytest.raises(ValueError, match="not available"):
        selection.resolve(("A01",))


def test_openhcs_axis_selection_treats_single_string_as_one_axis() -> None:
    selection = OpenHCSAxisSelection.from_pipeline_params(
        {OPENHCS_AXIS_FILTER_PARAM: "B02"}
    )

    assert selection.resolve(("B01", "B02")) == ("B02",)


def test_openhcs_benchmark_execution_config_supports_threaded_single_process() -> None:
    config = OpenHCSBenchmarkExecutionConfig.from_pipeline_params(
        {
            OPENHCS_NUM_WORKERS_PARAM: 1,
            OPENHCS_USE_THREADING_PARAM: True,
        }
    )

    assert config.num_workers == 1
    assert config.use_threading is True


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
            features=frozenset(),
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
            features=frozenset(
                (CPPipeInfrastructureFeature.EXPORT_TO_SPREADSHEET,)
            ),
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
