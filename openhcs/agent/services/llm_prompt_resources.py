"""Reusable LLM prompt resource projections for OpenHCS authoring."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Protocol

from openhcs.agent.dto.functions import FunctionCatalogPage
from openhcs.agent.services.function_catalog_service import FunctionCatalogService


logger = logging.getLogger(__name__)


class FunctionCatalogSearchReader(Protocol):
    """Catalog search projection required by prompt resource builders."""

    def search(
        self,
        *,
        query: str | None = None,
        library: str | None = None,
        limit: int = 50,
        compact_signatures: bool = True,
    ) -> FunctionCatalogPage: ...


class LLMFunctionDocumentationBuilder:
    """Project registered OpenHCS functions into prompt documentation."""

    def __init__(
        self,
        *,
        function_catalog: FunctionCatalogSearchReader | None = None,
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
        function_catalog: FunctionCatalogSearchReader | None = None,
    ) -> None:
        self._function_catalog = function_catalog or FunctionCatalogService()

    def dynamic_imports_section(self) -> str:
        """Generate imports section with actual module paths."""
        from openhcs.core.artifacts import ArtifactType
        from openhcs.processing.materialization import (
            registered_materialization_option_types,
        )

        artifact_type_names = "\n".join(
            f"    {artifact_type.__name__},"
            for artifact_type in ArtifactType.__registry__.values()
        )
        materialization_option_names = "\n".join(
            f"    {option_type.__name__},"
            for option_type in registered_materialization_option_types()
        )
        runtime_payload_types_by_module: dict[str, list[str]] = {}
        for artifact_type in ArtifactType.__registry__.values():
            for runtime_type in artifact_type.runtime_parameter_types():
                if not runtime_type.__module__.startswith("openhcs."):
                    continue
                runtime_payload_types_by_module.setdefault(
                    runtime_type.__module__,
                    [],
                ).append(runtime_type.__name__)
        runtime_payload_imports = "\n".join(
            f"from {module_name} import {', '.join(dict.fromkeys(type_names))}"
            for module_name, type_names in runtime_payload_types_by_module.items()
        )
        return f"""=== REQUIRED IMPORTS (use exactly these paths) ===
# Backend decorators
from openhcs.core.memory import numpy, pyclesperanto, cupy

# Typed artifact declarations (for analysis functions)
from openhcs.core.artifacts import (
    ArtifactMeasurementSubjectRelation,
    ArtifactSpec,
{artifact_type_names}
)
from openhcs.core.measurement_row_materialization import DataclassMeasurementColumnarRows
from openhcs.core.pipeline.function_contracts import artifact_outputs, artifact_inputs

# Nominal runtime payloads declared by the registered artifact types
{runtime_payload_imports}

# Registered materialization writer options
from openhcs.processing.materialization import (
    MaterializationSpec,
{materialization_option_names}
)

# Standard library (include as needed)
from dataclasses import dataclass
import numpy as np"""

    def dynamic_materializers_section(self) -> str:
        """Generate typed artifact materialization examples from live options."""
        from openhcs.processing.materialization import (
            registered_materialization_option_types,
        )

        option_signatures = "\n".join(
            f"{option_type.__name__}{inspect.signature(option_type)}"
            for option_type in registered_materialization_option_types()
        )

        return f"""=== TYPED ARTIFACT MATERIALIZATION ===
from openhcs.core.artifacts import (
    ArtifactMeasurementSubjectRelation,
    ArtifactSpec,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.processing.materialization import csv_only, segmentation_mask_rois

@artifact_outputs(
    ArtifactSpec(
        "measurements",
        MeasurementsArtifactType,
        materialization=csv_only(),
        relations=(ArtifactMeasurementSubjectRelation(),),
    ),
    ArtifactSpec(
        "masks",
        ObjectLabelsArtifactType,
        materialization=segmentation_mask_rois(),
    ),
)

Measurement outputs return schema-bearing ColumnarRows. Object-label outputs return
the complete integer label array. OpenHCS binds compiled names, materializes CSV/ROI
files, streams labels to Napari/Fiji, and exposes measurement snapshots.

For a path-bearing topology result, return one nominal ``SpatialGraph`` and
declare both projections on the same artifact:

```python
ArtifactSpec.output(
    "neurite_graph",
    SpatialGraphArtifactType,
    materialization=MaterializationSpec(
        SWCOptions(),
        SpatialGraphROIOptions(),
    ),
)
```

``SWCOptions`` persists a directed morphology forest. The separate
``SpatialGraphROIOptions`` writer projects those same edges and scalar features
to ``.graph.roi.zip`` so Napari receives native path Shapes and its feature table
without recomputing topology or maintaining a viewer-only graph. A saved SWC
reopens through the OpenHCS Napari reader as physical 3D samples and parent-child
paths, or through Fiji SNT. Standard SWC retains sample/type/radius/parent fields,
not arbitrary OpenHCS edge features, so use the graph-ROI projection for the full
branch measurement table.

=== ADVANCED CUSTOMIZATION (When needed) ===
{option_signatures}

Usage: MaterializationSpec(CsvOptions(filename_suffix="_custom.csv", fields=["x", "y"]))"""

    def example_pipeline(self) -> str:
        """Load example pipeline from file."""
        basic_pipeline_path = Path(__file__).parents[2] / "tests" / "basic_pipeline.py"
        try:
            with open(basic_pipeline_path, encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning("Could not load example pipeline: %s", e)
            return "# Example pipeline not available"
