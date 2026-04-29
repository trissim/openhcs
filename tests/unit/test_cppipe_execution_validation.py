from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.converter.execution_validation import (
    CPPipeExecutionValidationError,
    validate_cppipe_execution,
)
from benchmark.converter.parser import ModuleBlock
from benchmark.converter.runtime_pipeline import DirectPipelineExecution
from openhcs.core.artifacts import ArtifactKey, ArtifactKind, ArtifactScope, ArtifactSpec
from openhcs.core.runtime_semantics import FieldSpec
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_values import RuntimeValue, RuntimeValueSchema


def test_cppipe_execution_validation_rejects_header_only_csv(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "axis_Measurements_step1.csv"
    csv_path.write_text("slice_index\n", encoding="utf-8")

    with pytest.raises(CPPipeExecutionValidationError, match="has no data rows"):
        validate_cppipe_execution(
            _prepared_exporting_measurements(),
            _successful_execution_with_measurement_record(),
            tmp_path,
        )


def test_cppipe_execution_validation_accepts_csv_with_data_rows(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "axis_Measurements_step1.csv"
    csv_path.write_text("slice_index\n0\n", encoding="utf-8")

    validation = validate_cppipe_execution(
        _prepared_exporting_measurements(),
        _successful_execution_with_measurement_record(),
        tmp_path,
    )

    assert validation.observation.csv_row_counts_by_path[csv_path] == 1


def _prepared_exporting_measurements() -> SimpleNamespace:
    return SimpleNamespace(
        infrastructure_modules=(
            ModuleBlock(name="ExportToSpreadsheet", module_num=1),
        ),
        generated_pipeline=SimpleNamespace(
            artifact_contracts=(
                SimpleNamespace(
                    outputs=(
                        ArtifactSpec(
                            name="Measurements",
                            kind=ArtifactKind.MEASUREMENTS,
                        ),
                    )
                ),
            )
        ),
    )


def _successful_execution_with_measurement_record() -> DirectPipelineExecution:
    store = RuntimeValueStore()
    store.record(
        RuntimeValue(
            key=ArtifactKey(
                name="Measurements",
                kind=ArtifactKind.MEASUREMENTS,
                scope=ArtifactScope(axis_id="A01"),
            ),
            data=(),
            schema=RuntimeValueSchema(
                kind=ArtifactKind.MEASUREMENTS,
                fields=(FieldSpec("slice_index"),),
            ),
        ),
        path="results/axis_Measurements_step1.csv",
        backend="disk",
    )
    return DirectPipelineExecution(
        compiled_contexts={
            "A01": SimpleNamespace(runtime_value_store=store),
        },
        execution_results={
            "A01": SimpleNamespace(is_success=lambda: True),
        },
    )
