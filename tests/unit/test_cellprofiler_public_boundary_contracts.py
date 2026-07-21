"""Architectural guards for public CellProfiler/OpenHCS pipeline boundaries."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.pipeline_document_fields import PipelineDocumentField
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends import cellprofiler as cellprofiler_backend
from openhcs.processing.backends.cellprofiler.illumination import (
    IlluminationCorrectionMethod,
)


PROJECT_ROOT = Path(__file__).parents[2]


def _source(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def test_cellprofiler_compiler_does_not_use_selected_cppipe_contract_fallback() -> None:
    source = _source("openhcs/interop/cellprofiler/compile_time_contracts.py")

    assert "_runtime_contracts_from_selected_cppipe" not in source
    assert "_runtime_contracts_from_cppipe_path" not in source
    assert "selected_pipeline_path" not in source
    assert "input_workspace_preparation_result" not in source


def test_raw_callable_signatures_expose_behavior_not_canonical_identity() -> None:
    track_parameters = inspect.signature(cellprofiler_backend.track_objects).parameters
    calculate_math_parameters = inspect.signature(
        cellprofiler_backend.calculate_math
    ).parameters
    assert "object_name" not in track_parameters
    assert track_parameters["save_color_coded_image"].default is False
    assert track_parameters["name_the_output_image"].default == "TrackedCells"
    assert "runtime_invocation_options" not in calculate_math_parameters
    assert "output_name" in calculate_math_parameters


def test_step_fragment_transport_uses_pipeline_document_field_authority() -> None:
    assert callable(FunctionStepTransportAuthority.source_from_pipeline)
    assert callable(FunctionStepTransportAuthority.pipeline_steps_from_namespace)
    assert PipelineDocumentField.PIPELINE_STEPS.value == "pipeline_steps"


def test_pycodified_public_cp_pipeline_reconstructs_in_fresh_process(
    tmp_path: Path,
) -> None:
    pipeline_steps = [
        FunctionStep(
            func=(
                cellprofiler_backend.correct_illumination_apply,
                {"method": IlluminationCorrectionMethod.SUBTRACT},
            ),
            name="CorrectIlluminationApply",
        )
    ]
    source = FunctionStepTransportAuthority.source_from_pipeline(pipeline_steps)
    source_path = tmp_path / "pipeline.py"
    source_path.write_text(source, encoding="utf-8")
    script = (
        "from pathlib import Path\n"
        "from openhcs.core.function_step_transport import "
        "FunctionStepTransportAuthority\n"
        f"source = Path({str(source_path)!r}).read_text(encoding='utf-8')\n"
        "namespace = {}\n"
        "exec(compile(source, '<pipeline>', 'exec'), namespace)\n"
        "steps = FunctionStepTransportAuthority.pipeline_steps_from_namespace(namespace)\n"
        "assert len(steps) == 1\n"
        "assert steps[0].func[0].__name__ == 'correct_illumination_apply'\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
