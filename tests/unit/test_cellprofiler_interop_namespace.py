"""Public namespace guards for CellProfiler interoperability."""

from __future__ import annotations

import ast
from pathlib import Path

from openhcs.interop import cellprofiler
from openhcs.interop.cellprofiler import parser
from openhcs.interop.cellprofiler.pipeline_import import (
    import_cellprofiler_pipeline,
)


PROJECT_ROOT = Path(__file__).parents[2]
CELLPROFILER_INIT = PROJECT_ROOT / "openhcs/interop/cellprofiler/__init__.py"


def test_cellprofiler_namespace_exports_the_pure_import_boundary() -> None:
    namespace = cellprofiler.__dict__

    assert namespace["import_cellprofiler_pipeline"] is import_cellprofiler_pipeline
    assert namespace["CPPipeParser"] is parser.CPPipeParser
    assert namespace["ModuleBlock"] is parser.ModuleBlock
    assert namespace["ModuleSetting"] is parser.ModuleSetting


def test_cellprofiler_namespace_imports_the_translator_from_its_nominal_module() -> (
    None
):
    tree = ast.parse(CELLPROFILER_INIT.read_text(encoding="utf-8"))
    imports = {
        (node.module, alias.name)
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert (
        "openhcs.interop.cellprofiler.pipeline_import",
        "import_cellprofiler_pipeline",
    ) in imports
