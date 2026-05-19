"""
LLM Pipeline Generation Service

Handles communication with LLM endpoints to generate OpenHCS pipeline code
from natural language descriptions.

The system prompt is built dynamically from the actual function registry,
ensuring the LLM only sees real, available functions with correct signatures.
"""

import logging
import inspect
import requests
from enum import Enum
from typing import Optional, Dict, Any, Tuple, List, Set, Type, Callable
from pathlib import Path
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)

# --- Module-level constants ---
CONNECTION_TIMEOUT_S = 5
DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

# Preferred models in priority order (first available wins)
PREFERRED_MODELS = [
    "qwen2.5-coder",  # Best for code generation
    "codellama",  # Good alternative
    "deepseek-coder",  # Another good option
    "llama3",  # General purpose fallback
    "llama2",  # Older but common
    "mistral",  # General purpose
]

# Functions that get full documentation in system prompt (most commonly used)
CORE_FUNCTIONS = {
    "count_cells_single_channel",
    "count_cells_multi_channel",
    "stack_percentile_normalize",
    "create_projection",
    "create_composite",
    "assemble_stack_cpu",
    "ashlar_compute_tile_positions_cpu",
}

# Parameters to skip in signature display (internal/wrapper params)
INTERNAL_PARAMS = {"enabled", "slice_by_slice", "dtype_config"}

# Maximum functions to list per library (to keep prompt size manageable)
MAX_FUNCTIONS_PER_LIBRARY = 30


class LLMParameterDocumentationPolicy:
    """Rules for exposing callable parameters to LLM prompt documentation."""

    def __init__(self, internal_params: Set[str]):
        self._internal_params = frozenset(internal_params)
        self._variadic_parameter_kinds = frozenset(
            {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
        )

    def should_document(
        self,
        name: str,
        parameter: inspect.Parameter,
        *,
        skip_keyword_bag: bool = False,
    ) -> bool:
        if name in self._internal_params:
            return False
        if skip_keyword_bag and name == "kwargs":
            return False
        return parameter.kind not in self._variadic_parameter_kinds


class LLMParameterFormatter:
    """Formats Python signature details for prompt-safe display."""

    def __init__(self, policy: LLMParameterDocumentationPolicy):
        self._policy = policy

    def signature(self, func: Callable, name: str) -> str:
        try:
            sig = inspect.signature(func)
            params = []
            for pname, param in sig.parameters.items():
                if not self._policy.should_document(
                    pname, param, skip_keyword_bag=True
                ):
                    continue

                if param.default is inspect.Parameter.empty:
                    params.append(pname)
                else:
                    params.append(f"{pname}={self._format_default(param.default)}")

            return f"{name}({', '.join(params)})"
        except Exception:
            return f"{name}(...)"

    def parameter_block(self, func: Callable) -> str:
        try:
            sig = inspect.signature(func)
            lines = ["Parameters:"]

            for pname, param in sig.parameters.items():
                if not self._policy.should_document(pname, param):
                    continue

                lines.append(
                    f"  - {pname}: {self._format_annotation(param.annotation)}"
                    f"{self._format_default_suffix(param.default)}"
                )

            return "\n".join(lines) if len(lines) > 1 else ""
        except Exception:
            return ""

    def _format_annotation(self, annotation: object) -> str:
        if annotation is inspect.Parameter.empty:
            return ""
        if isinstance(annotation, type):
            annotation_name = annotation.__name__
            return annotation_name
        return str(annotation)

    def _format_default_suffix(self, default: object) -> str:
        if default is inspect.Parameter.empty:
            return ""
        return f" = {self._format_default(default)}"

    def _format_default(self, default: object) -> str:
        if isinstance(default, Enum):
            return f"{type(default).__name__}.{default.name}"
        if default is None:
            return "None"
        if isinstance(default, str):
            return f"'{default}'"
        return repr(default)


class LLMPipelineService:
    """
    Service for generating OpenHCS pipelines using LLM.

    Sends user requests to LLM endpoint with comprehensive system prompt
    containing OpenHCS API documentation and examples.
    """

    def __init__(
        self, api_endpoint: str = DEFAULT_OLLAMA_ENDPOINT, model: Optional[str] = None
    ):
        """
        Initialize LLM service.

        Args:
            api_endpoint: LLM API endpoint URL (default: Ollama local endpoint)
            model: Model name (auto-detected if None)
        """
        self.api_endpoint = api_endpoint
        self.base_url = self._derive_base_url(api_endpoint)
        self.model = model  # May be None, resolved on first test_connection
        self._parameter_formatter = LLMParameterFormatter(
            LLMParameterDocumentationPolicy(INTERNAL_PARAMS)
        )
        # Build system prompts for different contexts
        self._system_prompts = {
            "pipeline": self._build_pipeline_system_prompt(),
            "function": self._build_custom_function_system_prompt(),
        }

    @property
    def system_prompt(self) -> str:
        """Default system prompt (pipeline) for backward compatibility."""
        return self._system_prompts.get("pipeline", "")

    def get_system_prompt(self, code_type: str = "pipeline") -> str:
        """Return the runtime-generated system prompt for a given context."""
        if code_type == "function":
            return self._system_prompts.get("function", self.system_prompt)
        return self._system_prompts.get("pipeline", self.system_prompt)

    def _derive_base_url(self, endpoint: str) -> str:
        """Extract base URL from endpoint."""
        parsed = urlparse(endpoint)
        return urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))

    def _get_available_models(self) -> List[str]:
        """Fetch available models from Ollama."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags", timeout=CONNECTION_TIMEOUT_S
            )
            response.raise_for_status()
            data = response.json()
            return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return []

    def _select_best_model(self, available_models: List[str]) -> Optional[str]:
        """Select best model from available ones based on preference order."""
        if not available_models:
            return None

        # Check preferred models in order
        for preferred in PREFERRED_MODELS:
            for available in available_models:
                # Match base name (e.g., "qwen2.5-coder" matches "qwen2.5-coder:7b")
                if available.split(":")[0] == preferred or available.startswith(
                    preferred
                ):
                    return available

        # Fall back to first available model
        return available_models[0] if available_models else None

    def test_connection(self) -> Tuple[bool, str]:
        """
        Test connection to LLM service. Auto-selects model if not set.

        Returns:
            (is_connected, status_message)
        """
        try:
            available_models = self._get_available_models()

            if not available_models:
                return (False, "No models available")

            # Auto-select model if not set
            if self.model is None:
                self.model = self._select_best_model(available_models)
                if self.model:
                    logger.info(f"Auto-selected LLM model: {self.model}")

            if self.model is None:
                return (False, "No suitable model found")

            # Check if selected model is available
            if self.model in available_models:
                return (True, self.model)

            # Try base name match
            model_base = self.model.split(":")[0]
            for name in available_models:
                if name.split(":")[0] == model_base:
                    self.model = name  # Update to actual name
                    return (True, name)

            return (False, f"Model '{self.model}' not found")

        except requests.exceptions.ConnectionError:
            return (False, "Connection refused")
        except requests.exceptions.Timeout:
            return (False, "Connection timeout")
        except Exception as e:
            return (False, str(e))

    def _build_pipeline_system_prompt(self) -> str:
        """
        Build system prompt for pipeline generation context.

        Dynamically discovers all registered functions from the registry
        to ensure the LLM only sees real, available functions.

        Returns:
            Complete system prompt string for pipeline generation
        """
        # Build dynamic documentation from registry
        function_docs = self._get_function_documentation()
        enum_docs = self._get_enum_documentation()
        example_pipeline = self._get_example_pipeline()

        prompt = f"""You are an expert OpenHCS pipeline generator. Generate complete, runnable OpenHCS pipeline code based on user descriptions.

IMPORTANT: Only use functions from the "Available Functions" section below. Do NOT invent function names.

# OpenHCS Architecture Principles
- Prefer explicit FunctionStep declarations over hidden runtime side effects.
- Use OpenHCS enums and registry functions exactly as documented below.

# OpenHCS Pipeline API

## Core Imports (always include these)
```python
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.config import (
    LazyProcessingConfig, LazyStepWellFilterConfig, LazyStepMaterializationConfig,
    LazyNapariStreamingConfig, LazyFijiStreamingConfig
)
from openhcs.constants.constants import VariableComponents, GroupBy, DtypeConversion
from openhcs.constants.input_source import InputSource
```

## FunctionStep Structure
FunctionStep is the core building block:
- `func`: Function reference, tuple (func, kwargs), list of functions, or dict for routing
- `name`: Human-readable step name
- `processing_config`: LazyProcessingConfig for variable_components, group_by, input_source

## Function Patterns

### Single Function (no parameters)
```python
FunctionStep(func=create_projection, name="Z Projection")
```

### Function with Parameters (tuple)
```python
FunctionStep(
    func=(stack_percentile_normalize, {{'low_percentile': 1.0, 'high_percentile': 99.0}}),
    name="Normalize"
)
```

### Processing per Channel
```python
FunctionStep(
    func=some_function,
    processing_config=LazyProcessingConfig(variable_components=[VariableComponents.CHANNEL])
)
```

### Processing per Z-slice
```python
FunctionStep(
    func=some_function,
    processing_config=LazyProcessingConfig(variable_components=[VariableComponents.Z_INDEX])
)
```

# Available Enums
{enum_docs}

# Available Functions
{function_docs}

# Example Pipeline
```python
{example_pipeline}
```

# Rules
1. ONLY use functions listed in "Available Functions" section
2. Import each function from its specified module path
3. Use enums (not strings) for detection_method, dtype_conversion, etc.
4. Start with imports, then `pipeline_steps = []`, then FunctionStep definitions
5. Output ONLY Python code, no explanations"""

        return prompt

    def _build_custom_function_system_prompt(self) -> str:
        """
        Build system prompt for custom function generation context.

        Uses dynamic discovery to follow single source of truth principle.
        """
        # Dynamic discovery of imports and signatures
        imports_section = self._get_dynamic_imports_section()
        materializers_section = self._get_dynamic_materializers_section()
        pycle_docs = self._get_pyclesperanto_function_docs()

        prompt = f'''You are an expert at writing custom image processing functions for OpenHCS.
Generate COMPLETE, RUNNABLE Python code. Include ALL imports at the top.

=== CRITICAL RULES ===
1. Include ALL imports (dataclass, typing, numpy, etc.)
2. First parameter MUST be named 'image' (3D array: (C, Y, X) a.k.a. (Z, Y, X))
3. Input/Output backend types are declared by the memory decorator
   (e.g. @numpy / @cupy / @pyclesperanto). Your function must accept the
   decorator's declared input type and return the declared output type.
4. DO NOT write FunctionStep or pipeline code - just the function
5. Output ONLY Python code, no explanations
6. Do NOT manually convert between array backends inside the function
   (no cp.asnumpy(), no cle.pull(), etc.). OpenHCS handles cross-step
   conversions.
7. The decorator adds keyword-only args like slice_by_slice and
   dtype_conversion. dtype_conversion defaults to preserving the input dtype
   for the main output.

{imports_section}

=== BASIC FUNCTION (no analysis output) ===
```python
from openhcs.core.memory import numpy
import numpy as np

@numpy
def enhance_contrast(image, clip_limit: float = 0.03):
    """Enhance image contrast using CLAHE."""
    from skimage.exposure import equalize_adapthist
    result = np.stack([equalize_adapthist(ch, clip_limit=clip_limit) for ch in image])
    return result
```

=== FUNCTION WITH CSV OUTPUT ===
When you need to save measurements to CSV, use @artifact_outputs with csv_only() preset.

RETURN SEMANTICS: With N artifact_outputs, return (image, output1, output2, ..., outputN)

```python
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from skimage.measure import label, regionprops
from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import artifact_outputs
from openhcs.processing.materialization import csv_only

@dataclass
class CellMeasurement:
    slice_index: int
    cell_count: int
    total_area: float
    mean_intensity: float

@numpy
@artifact_outputs(("cell_measurements", csv_only()))
def count_cells_with_csv(
    image,
    threshold: float = 0.5,
    min_area: int = 50
) -> Tuple[np.ndarray, List[CellMeasurement]]:
    """Count cells and save measurements to CSV."""
    results = []
    for i, slice_2d in enumerate(image):
        binary = slice_2d > (np.max(slice_2d) * threshold)
        labeled = label(binary)
        props = regionprops(labeled, intensity_image=slice_2d)
        valid = [p for p in props if p.area >= min_area]

        results.append(CellMeasurement(
            slice_index=i,
            cell_count=len(valid),
            total_area=float(sum(p.area for p in valid)),
            mean_intensity=float(np.mean([p.mean_intensity for p in valid])) if valid else 0.0
        ))

    return image, results  # 1 special_output -> return (image, results)
```

=== FUNCTION WITH ROI OUTPUT (for ImageJ/Napari) ===
For segmentation masks that become ROIs, return labeled arrays (each region has unique int ID).

```python
from typing import List, Tuple
import numpy as np
from skimage.measure import label
from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import artifact_outputs
from openhcs.processing.materialization import roi_zip

@numpy
@artifact_outputs(("segmentation_masks", roi_zip()))
def segment_cells_with_rois(
    image,
    threshold: float = 0.5
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Segment cells and output ROIs for visualization."""
    masks = []
    for slice_2d in image:
        binary = slice_2d > (np.max(slice_2d) * threshold)
        labeled = label(binary)  # Each connected region gets unique ID
        masks.append(labeled)

    return image, masks  # masks -> .roi.zip for Fiji, shapes for Napari
```

=== FUNCTION WITH BOTH JSON AND CSV OUTPUT ===
Use json_and_csv() preset for analysis results (most common pattern).

```python
from typing import List, Tuple
import numpy as np
from skimage.measure import label
from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import artifact_outputs
from openhcs.processing.materialization import json_and_csv, roi_zip

@numpy
@artifact_outputs(
    ("segmentation_masks", roi_zip()),
    ("cell_measurements", json_and_csv()),
)
def analyze_cells_full(
    image,
    threshold: float = 0.5
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Segmentation + analysis with both ROIs and metrics output."""
    masks: List[np.ndarray] = []

    for i, slice_2d in enumerate(image):
        binary = slice_2d > (np.max(slice_2d) * threshold)
        labeled = label(binary)
        masks.append(labeled)

    return image, masks
```

{materializers_section}

=== PYCLESPERANTO (GPU) ===
Use cle.statistics_of_labelled_pixels() for cell stats - stays on GPU, no skimage needed!

```python
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pyclesperanto as cle
from openhcs.core.memory import pyclesperanto
from openhcs.core.pipeline.function_contracts import artifact_outputs
from openhcs.processing.materialization import csv_only, roi_zip

@dataclass
class CellStats:
    slice_index: int
    cell_count: int
    total_area: float
    mean_intensity: float

@pyclesperanto
@artifact_outputs(
    ("cell_stats", csv_only()),
    ("segmentation_masks", roi_zip())
)
def count_cells_gpu(
    image,
    sigma: float = 1.0,
    min_area: int = 50,
    max_area: int = 5000
) -> Tuple[np.ndarray, List[CellStats], List[np.ndarray]]:
    """GPU cell counting with CSV + ROI output - pure pyclesperanto."""
    stats_list = []
    masks = []

    for i, slice_2d in enumerate(image):
        # All GPU operations
        blurred = cle.gaussian_blur(slice_2d, sigma_x=sigma, sigma_y=sigma)
        binary = cle.threshold_otsu(blurred)
        labels = cle.connected_components_labeling(binary)
        labels = cle.remove_small_labels(labels, minimum_size=min_area)
        labels = cle.remove_large_labels(labels, maximum_size=max_area)

        # Get stats directly from GPU (no skimage needed!)
        stats_dict = cle.statistics_of_labelled_pixels(slice_2d, labels)

        # Extract from stats dict
        cell_count = len(stats_dict.get('label', []))
        total_area = float(sum(stats_dict.get('area', [])))
        intensities = stats_dict.get('mean_intensity', [])
        mean_int = float(np.mean(intensities)) if intensities else 0.0

        stats_list.append(CellStats(
            slice_index=i,
            cell_count=cell_count,
            total_area=total_area,
            mean_intensity=mean_int
        ))
        masks.append(labels)

    return image, stats_list, masks
```

Key pyclesperanto functions:
{pycle_docs}

IMPORTANT: cle.statistics_of_labelled_pixels(intensity_image, label_image) returns dict with:
- 'label': list of label IDs
- 'area': list of areas per label
- 'mean_intensity': list of mean intensities
- 'centroid_x', 'centroid_y': list of centroid positions

=== CUPY/CUCIM (NVIDIA CUDA GPU) ===
cucim is GPU-accelerated skimage. Use cucim.skimage.* instead of skimage.*.

```python
from dataclasses import dataclass
from typing import List, Tuple
import cupy as cp
from cucim.skimage.filters import gaussian
from cucim.skimage.measure import label, regionprops_table
from openhcs.core.memory import cupy
from openhcs.core.pipeline.function_contracts import artifact_outputs
from openhcs.processing.materialization import CsvOptions, MaterializationSpec, ROIOptions

@dataclass
class CellStats:
    slice_index: int
    cell_count: int
    total_area: float
    mean_intensity: float

@cupy
@artifact_outputs(
    ("cell_stats", MaterializationSpec(CsvOptions(filename_suffix="_stats.csv"))),
    ("segmentation_masks", MaterializationSpec(ROIOptions()))
)
def count_cells_cupy(
    image,
    sigma: float = 1.0,
    threshold: float = 0.5,
    min_area: int = 50
) -> Tuple[cp.ndarray, List[CellStats], List[cp.ndarray]]:
    """GPU cell counting with cucim (GPU-accelerated skimage)."""
    stats_list = []
    masks = []

    for i, slice_2d in enumerate(image):
        # All GPU operations using cucim (same API as skimage!)
        blurred = gaussian(slice_2d, sigma=sigma)
        binary = blurred > (cp.max(blurred) * threshold)
        labeled = label(binary)

        # regionprops_table stays on GPU, returns dict of cupy arrays
        props = regionprops_table(labeled, intensity_image=slice_2d,
                                  properties=['label', 'area', 'mean_intensity'])

        # Filter by area (GPU operations)
        areas = props['area']
        valid_mask = areas >= min_area

        stats_list.append(CellStats(
            slice_index=i,
            cell_count=int(cp.sum(valid_mask)),
            total_area=float(cp.sum(areas[valid_mask])),
            mean_intensity=float(cp.mean(props['mean_intensity'][valid_mask])) if cp.any(valid_mask) else 0.0
        ))
        masks.append(labeled)

    return image, stats_list, masks
```

Key cucim.skimage modules (same API as skimage, but GPU):
- cucim.skimage.filters: gaussian, sobel, median, threshold_otsu
- cucim.skimage.morphology: opening, closing, erosion, dilation, remove_small_objects
- cucim.skimage.measure: label, regionprops, regionprops_table
- cucim.skimage.segmentation: watershed, clear_border
- cucim.skimage.exposure: equalize_adapthist, rescale_intensity

IMPORTANT: Do not convert arrays between backends on return.

=== SPECIAL INPUTS (consume data from previous steps) ===
```python
from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import artifact_inputs

@numpy
@artifact_inputs("cell_positions")
def analyze_at_positions(image, cell_positions):
    """cell_positions is auto-loaded from a previous step's special_output."""
    return image
```
'''
        return prompt

    def _get_dynamic_imports_section(self) -> str:
        """Generate imports section with actual module paths."""
        # These are the actual import paths - single source of truth
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

    def _get_dynamic_materializers_section(self) -> str:
        """Generate materializers section with simple presets and advanced options."""
        try:
            from openhcs.processing.materialization import (
                MaterializationSpec,
                CsvOptions,
                JsonOptions,
                ROIOptions,
                TiffStackOptions,
                TextOptions,
            )
            from openhcs.processing.materialization.presets import (
                json_and_csv,
                csv_only,
                json_only,
                roi_zip,
                tiff_stack,
                text_only,
            )
            import inspect

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

    def _get_pyclesperanto_function_docs(self) -> str:
        """Get pyclesperanto functions dynamically if available."""
        try:
            import pyclesperanto as cle  # type: ignore[import-not-found]

            # Get commonly used functions that actually exist
            available_functions = frozenset(dir(cle))
            key_funcs = []
            for name in [
                "gaussian_blur",
                "median",
                "top_hat",
                "bottom_hat",
                "threshold_otsu",
                "binary_opening",
                "binary_closing",
                "erode",
                "dilate",
                "label",
                "connected_components_labeling",
                "voronoi_labeling",
                "exclude_labels_on_edges",
                "exclude_small_labels",
                "push",
                "pull",
                "maximum_z_projection",
                "mean_z_projection",
            ]:
                if name in available_functions:
                    key_funcs.append(f"cle.{name}()")
            return "\n".join(key_funcs)
        except ImportError:
            return "pyclesperanto not available"

    def _get_function_documentation(self) -> str:
        """Build function documentation from the registry."""
        try:
            from openhcs.processing.backends.lib_registry.registry_service import (
                RegistryService,
            )

            all_functions = RegistryService.get_all_functions_with_metadata()
        except Exception as e:
            logger.warning(f"Could not load function registry: {e}")
            return self._get_fallback_function_docs()

        if not all_functions:
            return self._get_fallback_function_docs()

        # Group functions by library
        by_library: Dict[str, list] = {}
        for composite_key, metadata in all_functions.items():
            lib = metadata.registry.library_name
            if lib not in by_library:
                by_library[lib] = []
            by_library[lib].append(metadata)

        docs_parts = []

        # Process each library - prioritize core functions
        for lib_name in sorted(by_library.keys()):
            functions = by_library[lib_name]
            lib_docs = [f"\n## {lib_name.upper()} Functions\n"]

            # Sort with core functions first, then alphabetically
            sorted_functions = sorted(
                functions,
                key=lambda m: (
                    0 if m.original_name in CORE_FUNCTIONS else 1,
                    m.original_name,
                ),
            )

            # Limit non-core functions to keep prompt size manageable
            count = 0
            for metadata in sorted_functions:
                is_core = metadata.original_name in CORE_FUNCTIONS
                if not is_core and count >= MAX_FUNCTIONS_PER_LIBRARY:
                    continue  # Skip excess non-core functions

                func_doc = self._format_function_doc(metadata)
                if func_doc:
                    lib_docs.append(func_doc)
                    if not is_core:
                        count += 1

            if len(sorted_functions) > MAX_FUNCTIONS_PER_LIBRARY:
                lib_docs.append(
                    f"... and {len(sorted_functions) - MAX_FUNCTIONS_PER_LIBRARY} more functions\n"
                )

            docs_parts.append("\n".join(lib_docs))

        return "\n".join(docs_parts)

    def _format_function_doc(self, metadata) -> str:
        """Format documentation for a single function."""
        try:
            func = metadata.func
            original_name = metadata.original_name
            module = metadata.module

            # Build import path
            if metadata.registry.library_name in ("pyclesperanto", "skimage", "cupy"):
                # External libs use virtual modules
                import_path = f"from openhcs.{metadata.registry.library_name} import {original_name}"
            else:
                # OpenHCS native functions
                import_path = f"from {module} import {original_name}"

            # Get signature (filtering internal params)
            sig_str = self._format_signature(func, original_name)

            # Get description (first line of docstring)
            desc = ""
            if metadata.doc:
                first_line = metadata.doc.split("\n")[0].strip()
                if first_line:
                    desc = f"  # {first_line}"

            # Core functions get detailed documentation
            if original_name in CORE_FUNCTIONS:
                param_doc = self._format_parameters(func)
                return f"""
### {original_name}
{import_path}
Signature: `{sig_str}`{desc}
{param_doc}
"""
            else:
                # Non-core functions: compact format (just name and import)
                return f"- `{original_name}`: {import_path}\n"

        except Exception as e:
            logger.debug(f"Could not format function {metadata.name}: {e}")
            return ""

    def _format_signature(self, func: Callable, name: str) -> str:
        """Format function signature, filtering internal params."""
        return self._parameter_formatter.signature(func, name)

    def _format_parameters(self, func: Callable) -> str:
        """Format parameter documentation for core functions."""
        return self._parameter_formatter.parameter_block(func)

    def _get_enum_documentation(self) -> str:
        """Build documentation for relevant enums."""
        enums_to_document = []

        # Core enums that are always useful
        try:
            from openhcs.processing.backends.analysis.cell_counting_cpu import (
                DetectionMethod,
                ThresholdMethod,
                ColocalizationMethod,
            )

            enums_to_document.extend(
                [DetectionMethod, ThresholdMethod, ColocalizationMethod]
            )
        except ImportError:
            pass

        try:
            from openhcs.constants.constants import (
                DtypeConversion,
                VariableComponents,
                GroupBy,
            )

            enums_to_document.extend([DtypeConversion, VariableComponents, GroupBy])
        except ImportError:
            pass

        try:
            from openhcs.constants.input_source import InputSource

            enums_to_document.append(InputSource)
        except ImportError:
            pass

        docs = []
        for enum_type in enums_to_document:
            values = ", ".join(m.name for m in enum_type)
            module = enum_type.__module__
            docs.append(f"- `{enum_type.__name__}`: from {module} → Values: {values}")

        return "\n".join(docs) if docs else "# No enum documentation available"

    def _get_example_pipeline(self) -> str:
        """Load example pipeline from file."""
        basic_pipeline_path = (
            Path(__file__).parent.parent.parent / "tests" / "basic_pipeline.py"
        )
        try:
            with open(basic_pipeline_path, "r") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not load example pipeline: {e}")
            return "# Example pipeline not available"

    def _get_fallback_function_docs(self) -> str:
        """Fallback function documentation if registry unavailable."""
        return """
## Core Functions (fallback - registry not loaded)
- `stack_percentile_normalize`: from openhcs.processing.backends.processors.numpy_processor
- `create_projection`: from openhcs.processing.backends.processors.numpy_processor
- `create_composite`: from openhcs.processing.backends.processors.numpy_processor
- `count_cells_single_channel`: from openhcs.processing.backends.analysis.cell_counting_cpu
- `assemble_stack_cpu`: from openhcs.processing.backends.assemblers.assemble_stack_cpu
"""

    def generate_code(self, user_request: str, code_type: str = "pipeline") -> str:
        """
        Generate code from user request based on context.

        Args:
            user_request: Natural language description of desired code
            code_type: Type of code to generate ('pipeline', 'step', 'config', 'function', 'orchestrator')

        Returns:
            Generated Python code as string

        Raises:
            Exception: If LLM request fails
        """
        try:
            # Select appropriate system prompt based on code_type
            if code_type == "function":
                system_prompt = self._system_prompts.get("function", self.system_prompt)
                context_suffix = (
                    "Generate a standalone custom function with @decorator."
                )
            else:
                system_prompt = self._system_prompts.get("pipeline", self.system_prompt)
                context_suffix = {
                    "pipeline": "Generate complete pipeline_steps list with FunctionStep objects.",
                    "step": "Generate a single FunctionStep object.",
                    "config": "Generate a configuration object (LazyProcessingConfig, LazyStepWellFilterConfig, etc.).",
                    "orchestrator": "Generate complete orchestrator code with plate_paths, pipeline_data, and configs.",
                }.get(code_type, "Generate OpenHCS code.")

            # Construct request payload (Ollama format)
            payload = {
                "model": self.model,
                "prompt": f"{system_prompt}\n\nContext: {context_suffix}\n\nUser Request:\n{user_request}\n\nGenerated Code:",
                "stream": False,
                "options": {
                    "temperature": 0.2,  # Low temperature for more deterministic code generation
                    "top_p": 0.9,
                },
            }

            logger.info(
                f"Sending request to LLM: {self.api_endpoint} (code_type={code_type})"
            )
            try:
                response = requests.post(self.api_endpoint, json=payload, timeout=60)
                response.raise_for_status()
            except Exception as e:
                logger.error(f"LLM request failed: {e}")
                raise Exception(f"Failed to connect to LLM service: {e}") from e

            result = response.json()
            generated_code = result.get("response", "")

            # Clean up code (remove markdown code blocks if present)
            generated_code = self._clean_generated_code(generated_code)

            logger.info(f"Successfully generated {code_type} code")
            return generated_code

        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            raise

    def _clean_generated_code(self, code: str) -> str:
        """
        Clean generated code by removing markdown formatting.

        Args:
            code: Raw generated code

        Returns:
            Cleaned Python code
        """
        # Remove markdown code blocks
        if code.startswith("```python"):
            code = code[len("```python") :].lstrip()
        if code.startswith("```"):
            code = code[3:].lstrip()
        if code.endswith("```"):
            code = code[:-3].rstrip()

        return code.strip()
