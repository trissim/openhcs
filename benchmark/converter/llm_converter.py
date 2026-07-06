"""
LLMFunctionConverter - Convert CellProfiler functions to OpenHCS using LLM.

Supports two backends:
1. Ollama (local): http://localhost:11434/api/generate
2. OpenRouter (cloud): https://openrouter.ai/api/v1/chat/completions

OpenRouter provides access to frontier models like MiniMax-01 (456B params).
"""

import json
import logging
import os
import re
import requests
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from openhcs.interop.cellprofiler.parser import ModuleBlock
from .source_locator import SourceLocation
from .system_prompt import build_conversion_prompt

logger = logging.getLogger(__name__)
CONNECTION_TIMEOUT_S = 5
GENERATION_TIMEOUT_S = 300
DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
PREFERRED_OLLAMA_MODELS = ["qwen2.5-coder", "codellama", "deepseek-coder", "llama3"]
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
PREFERRED_OPENROUTER_MODELS = [
    "minimax/minimax-m2.1",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-2.0-flash-exp:free",
    "qwen/qwen-2.5-coder-32b-instruct",
]


class LLMBackend(Enum):
    OLLAMA = auto()
    OPENROUTER = auto()


def detect_backend(model: str) -> LLMBackend:
    """Detect backend from model name format."""
    if "/" in model and (not model.startswith("http")):
        return LLMBackend.OPENROUTER
    return LLMBackend.OLLAMA


@dataclass
class ConversionResult:
    """Result of converting a CellProfiler function."""

    module_name: str
    success: bool
    converted_code: str = ""
    error_message: str = ""
    original_source: str = ""
    settings: Dict[str, str] = None
    confidence: float = 0.0
    reasoning: str = ""
    parameter_mapping: Dict[str, any] = None

    def __post_init__(self):
        if self.settings is None:
            self.settings = {}
        if self.parameter_mapping is None:
            self.parameter_mapping = {}


class LLMFunctionConverter:
    """
    LLM-powered converter for CellProfiler → OpenHCS functions.

    Supports:
    - Ollama (local): model names like "qwen2.5-coder:7b"
    - OpenRouter (cloud): model names like "minimax/minimax-m2.1"

    OpenRouter requires OPENROUTER_API_KEY environment variable.
    """

    def __init__(self, model: str = None):
        """
        Initialize converter.

        Args:
            model: Model name. Format determines backend:
                   - "qwen2.5-coder:7b" → Ollama
                   - "minimax/minimax-m2.1" → OpenRouter
        """
        self.model = model
        self.backend = detect_backend(model) if model else LLMBackend.OLLAMA

    def test_connection(self) -> Tuple[bool, str]:
        """Test connection to LLM service."""
        if self.backend == LLMBackend.OPENROUTER:
            return self._test_openrouter()
        return self._test_ollama()

    def _test_ollama(self) -> Tuple[bool, str]:
        """Test Ollama connection and auto-detect model."""
        try:
            response = requests.get(
                f"{DEFAULT_OLLAMA_ENDPOINT.rsplit('/api', 1)[0]}/api/tags",
                timeout=CONNECTION_TIMEOUT_S,
            )
            response.raise_for_status()
            data = response.json()
            available = [m.get("name", "") for m in data.get("models", [])]
            if not available:
                return (False, "No models available")
            if self.model is None:
                for preferred in PREFERRED_OLLAMA_MODELS:
                    for name in available:
                        if preferred in name.lower():
                            self.model = name
                            return (True, f"Using model: {name}")
                self.model = available[0]
                return (True, f"Using model: {self.model}")
            if self.model in available or any((self.model in a for a in available)):
                return (True, f"Model ready: {self.model}")
            return (False, f"Model '{self.model}' not found. Available: {available}")
        except requests.exceptions.ConnectionError:
            return (False, "Connection refused - is Ollama running?")
        except Exception as e:
            return (False, str(e))

    def _test_openrouter(self) -> Tuple[bool, str]:
        """Test OpenRouter connection."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return (False, "OPENROUTER_API_KEY environment variable not set")
        if not self.model:
            self.model = PREFERRED_OPENROUTER_MODELS[0]
        return (True, f"OpenRouter ready: {self.model}")

    def convert(self, module: ModuleBlock, source: SourceLocation) -> ConversionResult:
        """
        Convert a CellProfiler module to OpenHCS format.

        Args:
            module: ModuleBlock with settings from .cppipe
            source: SourceLocation with source code

        Returns:
            ConversionResult with converted code or error
        """
        if not source.source_code:
            return ConversionResult(
                module_name=module.name,
                success=False,
                error_message="No source code found",
            )
        prompt = build_conversion_prompt(
            module_name=module.name,
            source_code=source.source_code,
            settings=module.settings,
        )
        if self.backend == LLMBackend.OPENROUTER:
            return self._convert_openrouter(module, source, prompt)
        return self._convert_ollama(module, source, prompt)

    def _convert_ollama(
        self, module: ModuleBlock, source: SourceLocation, prompt: str
    ) -> ConversionResult:
        """Convert using Ollama backend."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "top_p": 0.9},
            }
            logger.info(f"Converting {module.name} with Ollama ({self.model})...")
            response = requests.post(
                DEFAULT_OLLAMA_ENDPOINT, json=payload, timeout=GENERATION_TIMEOUT_S
            )
            response.raise_for_status()
            result = response.json()
            raw_response = result.get("response", "")
            parsed = self._parse_response(raw_response)
            logger.info(f"Successfully converted {module.name}")
            return ConversionResult(
                module_name=module.name,
                success=True,
                converted_code=parsed.get("code", ""),
                original_source=source.source_code,
                settings=module.settings,
                confidence=parsed.get("confidence", 0.5),
                reasoning=parsed.get("reasoning", ""),
                parameter_mapping=parsed.get("parameter_mapping", {}),
            )
        except Exception as e:
            logger.error(f"Conversion failed for {module.name}: {e}")
            return ConversionResult(
                module_name=module.name,
                success=False,
                error_message=str(e),
                original_source=source.source_code,
                settings=module.settings,
            )

    def _convert_openrouter(
        self, module: ModuleBlock, source: SourceLocation, prompt: str
    ) -> ConversionResult:
        """Convert using OpenRouter backend."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return ConversionResult(
                module_name=module.name,
                success=False,
                error_message="OPENROUTER_API_KEY not set",
                original_source=source.source_code,
                settings=module.settings,
            )
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/trissim/openhcs",
                "X-Title": "OpenHCS CellProfiler Converter",
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "top_p": 0.9,
                "max_tokens": 8192,
            }
            logger.info(f"Converting {module.name} with OpenRouter ({self.model})...")
            response = requests.post(
                OPENROUTER_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=GENERATION_TIMEOUT_S,
            )
            response.raise_for_status()
            result = response.json()
            choices = result.get("choices", [])
            if not choices:
                return ConversionResult(
                    module_name=module.name,
                    success=False,
                    error_message="No response from model",
                    original_source=source.source_code,
                    settings=module.settings,
                )
            raw_response = choices[0].get("message", {}).get("content", "")
            parsed = self._parse_response(raw_response)
            logger.info(f"Successfully converted {module.name}")
            return ConversionResult(
                module_name=module.name,
                success=True,
                converted_code=parsed.get("code", ""),
                original_source=source.source_code,
                settings=module.settings,
                confidence=parsed.get("confidence", 0.5),
                reasoning=parsed.get("reasoning", ""),
                parameter_mapping=parsed.get("parameter_mapping", {}),
            )
        except Exception as e:
            logger.error(f"Conversion failed for {module.name}: {e}")
            return ConversionResult(
                module_name=module.name,
                success=False,
                error_message=str(e),
                original_source=source.source_code,
                settings=module.settings,
            )

    def convert_all(
        self, modules: List[ModuleBlock], sources: Dict[str, SourceLocation]
    ) -> List[ConversionResult]:
        """Convert multiple modules."""
        results = []
        for module in modules:
            source = sources.get(module.name)
            if source:
                result = self.convert(module, source)
                results.append(result)
            else:
                results.append(
                    ConversionResult(
                        module_name=module.name,
                        success=False,
                        error_message="Source not found",
                    )
                )
        return results

    def _parse_response(self, response: str) -> Dict[str, any]:
        """Parse LLM response as strict JSON conversion payload."""
        response = response.strip()
        if response.startswith("```json"):
            response = response[len("```json") :].lstrip()
        if response.startswith("```"):
            response = response[3:].lstrip()
        if response.endswith("```"):
            response = response[:-3].rstrip()
        try:
            data = json.loads(response)
        except json.JSONDecodeError as exc:
            first_brace = response.find("{")
            last_brace = response.rfind("}")
            if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
                raise ValueError(
                    "LLM conversion response must be a JSON object payload."
                ) from exc
            data = json.loads(response[first_brace : last_brace + 1])
        if not isinstance(data, dict):
            raise TypeError("LLM conversion response must decode to a JSON object.")
        code = data.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("LLM conversion response must include non-empty code.")
        if code.startswith("```python"):
            code = code[len("```python") :].lstrip()
        if code.startswith("```"):
            code = code[3:].lstrip()
        if code.endswith("```"):
            code = code[:-3].rstrip()
        data["code"] = code.strip()
        return data
