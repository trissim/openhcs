"""
LLM Pipeline Generation Service

Handles communication with LLM endpoints to generate OpenHCS pipeline code
from natural language descriptions.

The system prompt is built dynamically from the actual function registry,
ensuring the LLM only sees real, available functions with correct signatures.
"""

import logging
import requests
from abc import ABC, abstractmethod
from types import FunctionType
from typing import List, Optional, Tuple, TypeVar
from urllib.parse import urlparse, urlunparse

from metaclass_registry import AutoRegisterMeta
from pyqt_reactive.services.function_pattern_code_document import FunctionPatternValue
from python_introspect import AnnotatedDataclassValidationMixin

from openhcs.agent.services.llm_prompt_resources import (
    FunctionCatalogSearchReader,
    LLMFunctionDocumentationBuilder,
    LLMPromptResourceCatalog,
)
from openhcs.core.config_document import ConfigDocumentAuthority
from openhcs.core.function_step_document import (
    FunctionStepDocument,
    FunctionStepDocumentAuthority,
)
from openhcs.core.pipeline_document import PipelineDocument, PipelineDocumentAuthority
from openhcs.core.registry_strategies import NominalTypeStrategyFamilyMixin
from openhcs.core.steps.function_step import FunctionStep
from openhcs.ui.shared.plate_manager_code_document import (
    PlateManagerCodeDocumentAuthority,
    PlateManagerOrchestratorCodePayload,
)

logger = logging.getLogger(__name__)
DeclarationT = TypeVar("DeclarationT")

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


class LLMPromptBuilder:
    """Assemble context-specific LLM prompts from documented authorities."""

    def __init__(self, function_catalog: FunctionCatalogSearchReader):
        self._catalog = LLMPromptResourceCatalog(function_catalog=function_catalog)
        self._function_docs = LLMFunctionDocumentationBuilder(
            function_catalog=function_catalog
        )

    def build_pipeline_system_prompt(self) -> str:
        """
        Build system prompt for pipeline generation context.

        Dynamically discovers all registered functions from the registry
        to ensure the LLM only sees real, available functions.

        Returns:
            Complete system prompt string for pipeline generation
        """
        # Build dynamic documentation from registry
        function_docs = self._function_docs.documentation()
        example_pipeline = self._catalog.example_pipeline()

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
from openhcs.constants.constants import VariableComponents, GroupBy
from arraybridge.decorators import DtypeConversion
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
FunctionStep(func=registered_function, name="Processing step")
```

### Function with Parameters (tuple)
```python
FunctionStep(
    func=(registered_function, {{"parameter_name": 1.0}}),
    name="Configured processing step"
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

# Available Functions
{function_docs}

# Example Pipeline
```python
{example_pipeline}
```

# Rules
1. ONLY use functions listed in "Available Functions" section
2. Import each function from its specified module path
3. Use enums (not strings) for enum-typed function parameters shown by registry signatures or config schema.
4. Start with imports, then `pipeline_steps = []`, then FunctionStep definitions
5. Output ONLY Python code, no explanations"""

        return prompt

    def build_custom_function_system_prompt(self) -> str:
        """
        Build system prompt for custom function generation context.

        Uses dynamic discovery to follow single source of truth principle.
        """
        # Dynamic discovery of imports and signatures
        imports_section = self._catalog.dynamic_imports_section()
        materializers_section = self._catalog.dynamic_materializers_section()
        function_docs = self._function_docs.documentation()

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
7. Do not invent infrastructure keyword arguments. Use registry-displayed
   signatures for user parameters; OpenHCS supplies hidden runtime controls
   from registered contracts and resolved configs.

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
Declare measurement semantics on the artifact and return schema-bearing ColumnarRows.

RETURN SEMANTICS: With N artifact_outputs, return (image, output1, output2, ..., outputN)

```python
from dataclasses import dataclass
import numpy as np
from skimage.measure import label, regionprops
from openhcs.core.artifacts import (
    ArtifactMeasurementSubjectRelation,
    ArtifactSpec,
    MeasurementsArtifactType,
)
from openhcs.core.memory import numpy
from openhcs.core.measurement_row_materialization import DataclassMeasurementColumnarRows
from openhcs.core.pipeline.function_contracts import artifact_outputs
from openhcs.processing.materialization import csv_only

@dataclass
class CellMeasurement:
    slice_index: int
    cell_count: int
    total_area: float
    mean_intensity: float

@numpy
@artifact_outputs(
    ArtifactSpec(
        "cell_measurements",
        MeasurementsArtifactType,
        materialization=csv_only(),
        relations=(ArtifactMeasurementSubjectRelation(),),
    )
)
def count_cells_with_csv(
    image,
    threshold: float = 0.5,
    min_area: int = 50
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows]:
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

    return image, DataclassMeasurementColumnarRows(results, row_type=CellMeasurement)
```

=== FUNCTION WITH ROI OUTPUT (for ImageJ/Napari) ===
For segmentation masks that become ROIs, return labeled arrays (each region has unique int ID).

```python
import numpy as np
from skimage.measure import label
from openhcs.core.artifacts import ArtifactSpec, ObjectLabelsArtifactType
from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import artifact_outputs
from openhcs.processing.materialization import segmentation_mask_rois

@numpy
@artifact_outputs(
    ArtifactSpec(
        "segmentation_masks",
        ObjectLabelsArtifactType,
        materialization=segmentation_mask_rois(),
    )
)
def segment_cells_with_rois(
    image,
    threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Segment cells and output ROIs for visualization."""
    masks = []
    for slice_2d in image:
        binary = slice_2d > (np.max(slice_2d) * threshold)
        labeled = label(binary)  # Each connected region gets unique ID
        masks.append(labeled)

    return image, np.stack(masks)
```

To produce measurements and labels together, list both typed artifact declarations and
return `(image, measurement_rows, label_array)` in the same declared order.

{materializers_section}

=== FUNCTIONS AVAILABLE ON THE CONNECTED EXECUTION SERVER ===
Use backend decorators for memory semantics, then choose callable operations from
the connected server catalog instead of copied function lists.

{function_docs}

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


class CodeDeclarationStrategy(
    NominalTypeStrategyFamilyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Code-authoring behavior selected by the declaration already being edited."""

    @classmethod
    def for_declaration_type(
        cls,
        declaration_type: type[DeclarationT],
    ) -> "CodeDeclarationStrategy":
        strategy_types = cls.strategy_types_for_nominal_type(declaration_type)
        if not strategy_types:
            raise TypeError(
                "Code authoring has no registered strategy for declaration type "
                f"{declaration_type.__module__}.{declaration_type.__qualname__}."
            )
        return strategy_types[0]()

    def system_prompt(self, builder: LLMPromptBuilder) -> str:
        return builder.build_pipeline_system_prompt()

    @abstractmethod
    def context_suffix(self) -> str:
        """Return the user-request instruction owned by this editor context."""

    @abstractmethod
    def normalize_source(
        self,
        source: str,
        *,
        declaration_type: type[DeclarationT],
        clean_mode: bool,
    ) -> str:
        """Render edited source through its existing declaration authority."""


class PipelineCodeDeclarationStrategy(CodeDeclarationStrategy):
    value_type = PipelineDocument

    def context_suffix(self) -> str:
        return "Generate complete pipeline_steps list with FunctionStep objects."

    def normalize_source(
        self,
        source: str,
        *,
        declaration_type: type[DeclarationT],
        clean_mode: bool,
    ) -> str:
        del declaration_type
        return PipelineDocumentAuthority.render(
            PipelineDocumentAuthority.from_source(source),
            clean_mode=clean_mode,
        )


class StepCodeDeclarationStrategy(CodeDeclarationStrategy):
    value_type = (FunctionStep, FunctionStepDocument)

    def context_suffix(self) -> str:
        return "Generate a single FunctionStep object."

    def normalize_source(
        self,
        source: str,
        *,
        declaration_type: type[DeclarationT],
        clean_mode: bool,
    ) -> str:
        del declaration_type
        return FunctionStepDocumentAuthority.render(
            FunctionStepDocumentAuthority.from_source(source),
            clean_mode=clean_mode,
        )


class ConfigCodeDeclarationStrategy(CodeDeclarationStrategy):
    value_type = AnnotatedDataclassValidationMixin

    def context_suffix(self) -> str:
        return (
            "Generate a configuration object using the declared lazy config types."
        )

    def normalize_source(
        self,
        source: str,
        *,
        declaration_type: type[DeclarationT],
        clean_mode: bool,
    ) -> str:
        config = ConfigDocumentAuthority.from_source(
            source,
            expected_config_type=declaration_type,
        )
        return ConfigDocumentAuthority.render(
            config,
            expected_config_type=declaration_type,
            clean_mode=clean_mode,
        )


class CustomFunctionCodeDeclarationStrategy(CodeDeclarationStrategy):
    value_type = FunctionType

    def system_prompt(self, builder: LLMPromptBuilder) -> str:
        return builder.build_custom_function_system_prompt()

    def context_suffix(self) -> str:
        return "Generate a standalone custom function with its backend decorator."

    def normalize_source(
        self,
        source: str,
        *,
        declaration_type: type[DeclarationT],
        clean_mode: bool,
    ) -> str:
        del declaration_type, clean_mode
        compile(source, "<openhcs-custom-function-document>", "exec")
        return source


class FunctionPatternCodeDeclarationStrategy(
    CustomFunctionCodeDeclarationStrategy
):
    value_type = FunctionPatternValue

    def normalize_source(
        self,
        source: str,
        *,
        declaration_type: type[DeclarationT],
        clean_mode: bool,
    ) -> str:
        del declaration_type
        from pyqt_reactive.services.function_pattern_code_document import (
            FunctionPatternCodeDocumentService,
        )

        service = FunctionPatternCodeDocumentService()
        return service.generate_complete_function_pattern_code(
            service.pattern_from_source(source),
            clean_mode=clean_mode,
        )


class OrchestratorCodeDeclarationStrategy(CodeDeclarationStrategy):
    value_type = PlateManagerOrchestratorCodePayload

    def context_suffix(self) -> str:
        return (
            "Generate complete orchestrator code with plate paths, pipeline data, "
            "and configs."
        )

    def normalize_source(
        self,
        source: str,
        *,
        declaration_type: type[DeclarationT],
        clean_mode: bool,
    ) -> str:
        del declaration_type
        return PlateManagerCodeDocumentAuthority.render(
            PlateManagerCodeDocumentAuthority.from_source(source),
            clean_mode=clean_mode,
        )


class LLMPipelineService:
    """
    Service for generating OpenHCS pipelines using LLM.

    Sends user requests to LLM endpoint with comprehensive system prompt
    containing OpenHCS API documentation and examples.
    """

    def __init__(
        self,
        function_catalog: FunctionCatalogSearchReader,
        api_endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
        model: Optional[str] = None,
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
        self._prompt_builder = LLMPromptBuilder(function_catalog)
        self._system_prompts: dict[type[CodeDeclarationStrategy], str] = {}

    @property
    def system_prompt(self) -> str:
        """Return the endpoint-derived pipeline prompt."""
        return self.get_system_prompt(PipelineDocument)

    def get_system_prompt(self, declaration_type: type[DeclarationT]) -> str:
        """Build each endpoint-derived prompt only when its editor requests it."""
        strategy = CodeDeclarationStrategy.for_declaration_type(declaration_type)
        strategy_type = type(strategy)
        if strategy_type not in self._system_prompts:
            self._system_prompts[strategy_type] = strategy.system_prompt(
                self._prompt_builder
            )
        return self._system_prompts[strategy_type]

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

    def generate_code(
        self,
        user_request: str,
        declaration_type: type[DeclarationT],
    ) -> str:
        """
        Generate code from user request based on context.

        Args:
            user_request: Natural language description of desired code
            declaration_type: Existing nominal type of the declaration to generate.

        Returns:
            Generated Python code as string

        Raises:
            Exception: If LLM request fails
        """
        try:
            strategy = CodeDeclarationStrategy.for_declaration_type(declaration_type)
            system_prompt = self.get_system_prompt(declaration_type)
            context_suffix = strategy.context_suffix()

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
                "Sending request to LLM: %s (declaration_type=%s)",
                self.api_endpoint,
                declaration_type.__qualname__,
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

            logger.info(
                "Successfully generated %s code",
                declaration_type.__qualname__,
            )
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
