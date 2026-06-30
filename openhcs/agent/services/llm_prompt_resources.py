"""Reusable LLM prompt resource projections for OpenHCS authoring."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

from openhcs.agent.services.function_catalog_service import FunctionCatalogService
from openhcs.processing.backends.lib_registry.unified_registry import LibraryRegistryBase


logger = logging.getLogger(__name__)


class LLMFunctionDocumentationBuilder:
    """Project registered OpenHCS functions into prompt documentation."""

    def __init__(
        self,
        *,
        function_catalog: FunctionCatalogService | None = None,
        max_functions: int = 120,
    ):
        self._function_catalog = function_catalog or FunctionCatalogService()
        self._max_functions = max_functions

    def documentation(self) -> str:
        """Build function documentation from the registry."""
        page = self._function_catalog.search(
            limit=self._max_functions,
            compact_signatures=True,
        )
        if not page.items:
            return self.fallback_function_docs()

        by_library = {}
        for entry in page.items:
            by_library.setdefault(entry.library, []).append(entry)

        docs_parts = []
        for lib_name in sorted(by_library.keys()):
            lib_docs = [f"\n## {lib_name.upper()} Functions\n"]
            for entry in by_library[lib_name]:
                summary = "" if entry.summary is None else f"  # {entry.summary}"
                lib_docs.append(
                    f"- `{entry.signature}`: {entry.import_path}{summary}"
                )
            docs_parts.append("\n".join(lib_docs))

        if page.total > len(page.items):
            remaining = page.total - len(page.items)
            docs_parts.append(
                f"... and {remaining} more functions are available through openhcs_search_functions."
            )
        return "\n".join(docs_parts)

    def fallback_function_docs(self) -> str:
        """Fallback function documentation if registry unavailable."""
        return "Function registry unavailable; call openhcs_search_functions after startup."


class LLMPromptResourceCatalog:
    """Provide prompt resource sections not owned by HTTP generation."""

    def __init__(
        self,
        *,
        function_catalog: FunctionCatalogService | None = None,
    ) -> None:
        self._function_catalog = function_catalog or FunctionCatalogService()

    def dynamic_imports_section(self) -> str:
        """Generate imports section with actual module paths."""
        return """=== REQUIRED IMPORTS (use exactly these paths) ===
# Backend decorators
from openhcs.core.memory import numpy, pyclesperanto, cupy

# Special outputs/inputs (for analysis functions)
from openhcs.core.pipeline.function_contracts import artifact_outputs, artifact_inputs

# Materializers for CSV/JSON and ROI outputs
from openhcs.processing.materialization import (
    MaterializationSpec,
    CsvOptions,
    JsonOptions,
    ROIOptions,
    TiffStackOptions,
    TextOptions,
)

# Standard library (include as needed)
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np"""

    def dynamic_materializers_section(self) -> str:
        """Generate materializers section with simple presets and advanced options."""
        try:
            from openhcs.processing.materialization import (
                CsvOptions,
                JsonOptions,
                ROIOptions,
            )
            from openhcs.processing.materialization.presets import (
                csv_only,
                json_and_csv,
                json_only,
                roi_zip,
            )

            del csv_only, json_and_csv, json_only, roi_zip

            csv_sig = str(inspect.signature(CsvOptions))
            json_sig = str(inspect.signature(JsonOptions))
            roi_sig = str(inspect.signature(ROIOptions))

            return f"""=== SIMPLE MATERIALIZATION (Use These!) ===
from openhcs.processing.materialization import json_and_csv, csv_only, json_only, roi_zip

# JSON + CSV (most common for analysis)
@artifact_outputs(("results", json_and_csv()))

# CSV only
@artifact_outputs(("measurements", csv_only()))

# JSON only
@artifact_outputs(("metadata", json_only()))

# ROI zip for ImageJ/Fiji
@artifact_outputs(("masks", roi_zip()))

=== ADVANCED CUSTOMIZATION (When needed) ===
CsvOptions{csv_sig}
JsonOptions{json_sig}
ROIOptions{roi_sig}

Usage: MaterializationSpec(CsvOptions(filename_suffix="_custom.csv", fields=["x", "y"]))"""
        except Exception:
            return """=== SIMPLE MATERIALIZATION (Use These!) ===
from openhcs.processing.materialization import json_and_csv, csv_only, json_only, roi_zip

# Most common patterns - just use these:
@artifact_outputs(("results", json_and_csv()))  # JSON + CSV
@artifact_outputs(("measurements", csv_only()))  # CSV only
@artifact_outputs(("masks", roi_zip()))  # ROIs for ImageJ

=== ADVANCED CUSTOMIZATION ===
MaterializationSpec(CsvOptions(...), JsonOptions(...))"""

    def registry_function_docs(
        self,
        registry_type: type[LibraryRegistryBase],
        *,
        max_functions: int = 18,
    ) -> str:
        """Render compact docs for functions declared by one library registry."""
        registry = registry_type()
        page = self._function_catalog.search(
            library=registry.library_name,
            limit=max_functions,
            compact_signatures=True,
        )
        if not page.items:
            return f"{registry.get_display_name()} registry unavailable or empty."
        lines = [
            f"- `{entry.signature}`: {entry.import_path}"
            for entry in page.items
        ]
        if page.total > len(page.items):
            lines.append(f"- ... {page.total - len(page.items)} more")
        return "\n".join(lines)

    def example_pipeline(self) -> str:
        """Load example pipeline from file."""
        basic_pipeline_path = Path(__file__).parents[2] / "tests" / "basic_pipeline.py"
        try:
            with open(basic_pipeline_path, encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning("Could not load example pipeline: %s", e)
            return "# Example pipeline not available"
