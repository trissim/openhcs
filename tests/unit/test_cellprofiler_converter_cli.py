from __future__ import annotations

import ast
from pathlib import Path

from benchmark.converter import convert
from openhcs.core.config import PipelineConfig
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.processors.numpy_processor import create_projection


def test_converter_cli_writes_canonical_function_step_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cppipe_path = tmp_path / "input.cppipe"
    output_path = tmp_path / "output.py"
    cppipe_path.write_text("unused by stub", encoding="utf-8")
    steps = [FunctionStep(func=create_projection, name="Projection")]
    monkeypatch.setattr(
        convert,
        "import_cellprofiler_pipeline",
        lambda path: (steps, PipelineConfig()),
    )

    result = convert.main((str(cppipe_path), "--output", str(output_path)))

    assert result == 0
    assert output_path.read_text(encoding="utf-8") == (
        FunctionStepTransportAuthority.source_from_pipeline(steps)
    )


def test_converter_package_does_not_reexport_production_facades() -> None:
    package_path = Path(__file__).parents[2] / "benchmark" / "converter" / "__init__.py"
    tree = ast.parse(package_path.read_text(encoding="utf-8"))

    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body)


def test_converter_cli_uses_only_direct_import_and_generic_transport() -> None:
    source_path = Path(convert.__file__)
    source = source_path.read_text(encoding="utf-8")

    assert "import_cellprofiler_pipeline" in source
    assert "FunctionStepTransportAuthority.source_from_pipeline" in source
