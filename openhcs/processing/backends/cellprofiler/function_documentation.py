"""Rendered documentation for exported CellProfiler backend functions."""

from __future__ import annotations

import ast
import inspect
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


CELLPROFILER_FUNCTION_DOCUMENTATION_ATTR = "__openhcs_cellprofiler_documentation__"

_PARAMETER_SETTING_ALIASES: dict[str, tuple[str, ...]] = {
    "threshold_scope": ("threshold_strategy",),
    "threshold_method": ("thresholding_method",),
    "window_size": ("adaptive_window_size", "size_of_adaptive_window"),
    "otsu_class_count": ("two_class_or_three_class_thresholding",),
    "predefined_threshold": ("manual_threshold",),
}


class CellProfilerSourceAstInspector:
    """Static AST projection for CellProfiler source documentation."""

    def module_doc(self, tree: ast.Module) -> str:
        """Return the module-level CellProfiler help text."""
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "__doc__"
                for target in node.targets
            ):
                continue
            literal = self.literal_string(node.value)
            if literal is not None:
                return literal
        return ast.get_docstring(tree) or ""

    def settings(
        self,
        tree: ast.Module,
    ) -> tuple[CellProfilerSourceSettingDocumentation, ...]:
        """Return documented settings from CellProfiler setting constructors."""
        settings: list[CellProfilerSourceSettingDocumentation] = []
        seen: set[str] = set()

        for node in ast.walk(tree):
            setting = self.assigned_setting(node)
            if setting is None:
                continue
            key = _normalized_doc_key(setting.setting_name)
            if key in seen:
                continue
            seen.add(key)
            settings.append(setting)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            setting = self.setting_documentation(node)
            if setting is None:
                continue
            key = _normalized_doc_key(setting.setting_name)
            if key in seen:
                continue
            seen.add(key)
            settings.append(setting)
        return tuple(settings)

    def assigned_setting(
        self,
        node: ast.AST,
    ) -> CellProfilerSourceSettingDocumentation | None:
        """Return setting docs for ``self.<attribute> = Setting(...)`` nodes."""
        if isinstance(node, ast.Assign):
            attribute_name = next(
                (
                    attribute
                    for target in node.targets
                    if (attribute := self.self_attribute_name(target)) is not None
                ),
                None,
            )
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            attribute_name = self.self_attribute_name(node.target)
            value = node.value
        else:
            return None
        if attribute_name is None or not isinstance(value, ast.Call):
            return None
        return self.setting_documentation(value, attribute_name=attribute_name)

    def self_attribute_name(self, node: ast.AST) -> str | None:
        """Return the attribute in a ``self.<name>`` assignment target."""
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            return node.attr
        return None

    def setting_documentation(
        self,
        node: ast.Call,
        *,
        attribute_name: str | None = None,
    ) -> "CellProfilerSourceSettingDocumentation | None":
        """Return source docs for one CellProfiler setting constructor call."""
        setting_name = self.call_setting_name(node)
        if setting_name is None:
            return None
        return CellProfilerSourceSettingDocumentation(
            setting_name=setting_name,
            attribute_name=attribute_name,
            doc=self.call_keyword_string(node, "doc"),
            default_value=self.call_keyword_value_text(node, "value"),
        )

    def call_setting_name(self, node: ast.Call) -> str | None:
        """Return the UI setting label owned by a documented setting call."""
        has_doc_keyword = any(keyword.arg == "doc" for keyword in node.keywords)
        text = self.call_keyword_string(node, "text")
        if text is not None:
            return text
        if not has_doc_keyword:
            return None
        if node.args:
            return self.literal_string(node.args[0])
        return None

    def call_keyword_string(self, node: ast.Call, keyword_name: str) -> str | None:
        """Return a literal string keyword from a CellProfiler setting call."""
        for keyword in node.keywords:
            if keyword.arg == keyword_name:
                return self.literal_string(keyword.value)
        return None

    def call_keyword_value_text(self, node: ast.Call, keyword_name: str) -> str | None:
        """Return a printable literal keyword value from a setting call."""
        for keyword in node.keywords:
            if keyword.arg != keyword_name:
                continue
            value = keyword.value
            if isinstance(value, ast.Constant) and value.value is not None:
                return repr(value.value)
        return None

    def literal_string(self, node: ast.AST) -> str | None:
        """Return a statically recoverable string literal."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
            return "".join(parts) if parts else None
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "format"
        ):
            return self.literal_string(node.func.value)
        return None


_SOURCE_AST_INSPECTOR = CellProfilerSourceAstInspector()


@dataclass(frozen=True, slots=True)
class CellProfilerSourceSettingDocumentation:
    """Static documentation extracted from one CellProfiler source setting."""

    setting_name: str
    attribute_name: str | None = None
    doc: str | None = None
    default_value: str | None = None


@dataclass(frozen=True, slots=True)
class CellProfilerParameterDocumentation:
    """Rendered documentation for one exported function parameter."""

    parameter_name: str
    annotation: str
    default_value: str | None
    description: str
    cellprofiler_setting: CellProfilerSourceSettingDocumentation | None = None


@dataclass(frozen=True, slots=True)
class CellProfilerFunctionDocumentation:
    """Structured documentation attached to an exported CellProfiler callable."""

    module_name: str
    function_name: str
    summary: str
    description: str
    parameters: tuple[CellProfilerParameterDocumentation, ...]
    returns: str | None


def enrich_cellprofiler_function_documentation(
    wrapper: Callable[..., Any],
    *,
    module_name: str,
    source_function: Callable[..., Any],
) -> None:
    """Attach rendered and structured CellProfiler documentation to a wrapper."""
    documentation = build_cellprofiler_function_documentation(
        module_name=module_name,
        function_name=wrapper.__name__,
        source_function=source_function,
    )
    setattr(wrapper, CELLPROFILER_FUNCTION_DOCUMENTATION_ATTR, documentation)
    wrapper.__doc__ = render_cellprofiler_function_docstring(documentation)


def build_cellprofiler_function_documentation(
    *,
    module_name: str,
    function_name: str,
    source_function: Callable[..., Any],
) -> CellProfilerFunctionDocumentation:
    """Build structured docs from the callable signature and CP source docs."""
    source_docs = _source_module_documentation(module_name)
    original_doc = inspect.getdoc(source_function) or ""
    summary, description = _summary_and_description(
        module_name,
        original_doc=original_doc,
        source_doc=source_docs.module_doc,
    )
    existing_parameter_docs = _parse_existing_parameter_docs(original_doc)
    settings_by_key = {
        _normalized_doc_key(setting.setting_name): setting
        for setting in source_docs.settings
    }

    parameters = tuple(
        _parameter_documentation(
            module_name=module_name,
            parameter_name=name,
            parameter=parameter,
            existing_parameter_docs=existing_parameter_docs,
            settings_by_key=settings_by_key,
        )
        for name, parameter in inspect.signature(source_function).parameters.items()
    )

    return CellProfilerFunctionDocumentation(
        module_name=module_name,
        function_name=function_name,
        summary=summary,
        description=description,
        parameters=parameters,
        returns=_return_annotation_text(inspect.signature(source_function)),
    )


def render_cellprofiler_function_docstring(
    documentation: CellProfilerFunctionDocumentation,
) -> str:
    """Render structured CellProfiler documentation as parser-friendly text."""
    lines = [documentation.summary, ""]
    if documentation.description:
        lines.extend(_wrapped_lines(documentation.description))
        lines.append("")
    lines.append("Args:")
    for parameter in documentation.parameters:
        lines.append(
            f"    {parameter.parameter_name}: "
            f"{_parameter_type_default_prefix(parameter)}{parameter.description}"
        )
    if documentation.returns:
        lines.extend(("", "Returns:", f"    {documentation.returns}"))
    return "\n".join(lines).rstrip()


def cellprofiler_source_setting_parameter_mapping(
    module_name: str,
    parameter_names: Iterable[str],
) -> dict[str, str]:
    """Map normalized CP setting labels to same-named absorbed parameters.

    CellProfiler modules usually store each UI setting on ``self.<attribute>``.
    When that attribute is also an absorbed callable parameter, the source
    declaration is the authority for translating UI label to runtime kwarg.
    """
    parameters = frozenset(parameter_names)
    return {
        _normalized_doc_key(setting.setting_name): setting.attribute_name
        for setting in _source_module_documentation(module_name).settings
        if setting.attribute_name in parameters
    }


@dataclass(frozen=True, slots=True)
class _SourceModuleDocumentation:
    module_doc: str
    settings: tuple[CellProfilerSourceSettingDocumentation, ...]


def _parameter_documentation(
    *,
    module_name: str,
    parameter_name: str,
    parameter: inspect.Parameter,
    existing_parameter_docs: dict[str, str],
    settings_by_key: dict[str, CellProfilerSourceSettingDocumentation],
) -> CellProfilerParameterDocumentation:
    annotation = _annotation_text(parameter.annotation)
    default_value = _default_text(parameter.default)
    existing_doc = existing_parameter_docs.get(parameter_name)
    setting = _matching_setting(parameter_name, settings_by_key)
    description = _parameter_description(
        module_name=module_name,
        parameter_name=parameter_name,
        existing_doc=existing_doc,
        setting=setting,
    )
    return CellProfilerParameterDocumentation(
        parameter_name=parameter_name,
        annotation=annotation,
        default_value=default_value,
        description=description,
        cellprofiler_setting=setting,
    )


def _parameter_description(
    *,
    module_name: str,
    parameter_name: str,
    existing_doc: str | None,
    setting: CellProfilerSourceSettingDocumentation | None,
) -> str:
    parts: list[str] = []
    if setting is not None:
        parts.append(f"CellProfiler setting '{setting.setting_name}'.")
        if setting.doc:
            parts.append(_clean_rst(setting.doc))
        if setting.default_value is not None:
            parts.append(f"CellProfiler default: {setting.default_value}.")
    if existing_doc:
        parts.append(_clean_rst(existing_doc))
    if not parts:
        readable_name = parameter_name.replace("_", " ")
        parts.append(
            f"Controls {readable_name} for CellProfiler {module_name} execution."
        )
    return " ".join(part for part in parts if part).strip()


def _parameter_type_default_prefix(
    parameter: CellProfilerParameterDocumentation,
) -> str:
    pieces = []
    if parameter.annotation:
        pieces.append(parameter.annotation)
    if parameter.default_value is not None:
        pieces.append(f"default {parameter.default_value}")
    if not pieces:
        return ""
    return "; ".join(pieces) + ". "


def _matching_setting(
    parameter_name: str,
    settings_by_key: dict[str, CellProfilerSourceSettingDocumentation],
) -> CellProfilerSourceSettingDocumentation | None:
    parameter_key = _normalized_doc_key(parameter_name)
    candidate_keys = (parameter_key, *_PARAMETER_SETTING_ALIASES.get(parameter_key, ()))
    for candidate_key in candidate_keys:
        if candidate_key in settings_by_key:
            return settings_by_key[candidate_key]

    parameter_compact = parameter_key.replace("_", "")
    best_score = 0
    best_setting: CellProfilerSourceSettingDocumentation | None = None
    for setting_key, setting in settings_by_key.items():
        setting_compact = setting_key.replace("_", "")
        if parameter_compact == setting_compact:
            score = 4
        elif parameter_compact and parameter_compact in setting_compact:
            score = 3
        elif setting_compact and setting_compact in parameter_compact:
            score = 2
        else:
            score = 0
        if score > best_score:
            best_score = score
            best_setting = setting
    return best_setting if best_score >= 2 else None


@lru_cache(maxsize=None)
def _source_module_documentation(module_name: str) -> _SourceModuleDocumentation:
    source_path = _source_module_path(module_name)
    if source_path is None:
        return _SourceModuleDocumentation(module_doc="", settings=())
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return _SourceModuleDocumentation(module_doc="", settings=())
    return _SourceModuleDocumentation(
        module_doc=_SOURCE_AST_INSPECTOR.module_doc(tree),
        settings=_SOURCE_AST_INSPECTOR.settings(tree),
    )


def _source_module_path(module_name: str) -> Path | None:
    root = Path(__file__).resolve().parents[4] / "benchmark" / "cellprofiler_source"
    candidates = (
        root / "modules" / f"{module_name.lower()}.py",
        root / "library" / "modules" / f"_{module_name.lower()}.py",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _summary_and_description(
    module_name: str,
    *,
    original_doc: str,
    source_doc: str,
) -> tuple[str, str]:
    doc = source_doc or original_doc
    cleaned = _clean_rst(doc)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    filtered = [
        line
        for line in lines
        if line != module_name and not set(line) <= {"=", "-", "^"}
    ]
    if not filtered:
        return (
            f"CellProfiler-compatible {module_name} processing function.",
            "",
        )
    summary = filtered[0]
    description = "\n".join(filtered[1:8]).strip()
    return summary, description


def _parse_existing_parameter_docs(docstring: str) -> dict[str, str]:
    docs: dict[str, str] = {}
    current: str | None = None
    in_args = False
    for raw_line in docstring.splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if lowered in {"args:", "arguments:", "parameters:"}:
            in_args = True
            current = None
            continue
        if lowered in {"returns:", "return:", "examples:"}:
            in_args = False
            current = None
            continue
        if not in_args:
            continue
        match = re.match(r"^(\w+):\s*(.+)", line)
        if match:
            current, description = match.groups()
            docs[current] = description.strip()
        elif current and raw_line.startswith(("    ", "\t")) and line:
            docs[current] = f"{docs[current]} {line}"
    return docs


def _return_annotation_text(signature: inspect.Signature) -> str | None:
    if signature.return_annotation is inspect.Signature.empty:
        return None
    return _annotation_text(signature.return_annotation)


def _annotation_text(annotation: Any) -> str:
    if annotation is inspect.Signature.empty:
        return ""
    return inspect.formatannotation(annotation).replace("typing.", "")


def _default_text(default: Any) -> str | None:
    if default is inspect.Signature.empty:
        return None
    if isinstance(default, Enum):
        return repr(default.value)
    return repr(default)


def _normalized_doc_key(value: str) -> str:
    without_parentheses = re.sub(r"\([^)]*\)", "", value)
    words = re.sub(r"[^\w\s]", " ", without_parentheses).lower().split()
    return "_".join(words)


def _clean_rst(value: str) -> str:
    cleaned = value.replace("*", "").replace("`", "")
    cleaned = re.sub(r"\|[^|]+\|", "", cleaned)
    cleaned = re.sub(r":[a-zA-Z]+:`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _wrapped_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]
