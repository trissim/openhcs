"""Headless ObjectState field documentation service."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Protocol, runtime_checkable

from python_introspect import DocstringExtractor, UnifiedParameterAnalyzer

from openhcs.agent.dto.common import AgentError, AgentWarning, JsonValue, SCHEMA_VERSION
from openhcs.agent.dto.functions import FunctionArtifactSpec, FunctionDetail
from openhcs.agent.dto.ui_bridge import (
    UiBridgeConnectionSpec,
    UiObjectStateFieldHelpQuery,
    UiObjectStateFieldHelpRequest,
    UiObjectStateFieldHelpResult,
    UiObjectStateFieldSummary,
    UiObjectStateFieldListOptions,
    UiObjectStateScopeListRequest,
    UiObjectStateScopeSummary,
    UiObjectStateScopeVisibility,
)
from openhcs.agent.services.function_catalog_service import FunctionCatalogService
from openhcs.agent.services.ui_bridge_service import UiBridgeService


NO_PARAMETER_DESCRIPTION = "No description available"


@runtime_checkable
class CallableImportIdentity(Protocol):
    """Runtime-visible identity carried by importable Python callables."""

    __module__: str
    __qualname__: str


@runtime_checkable
class CallableDisplayName(Protocol):
    """Runtime-visible display name carried by named Python callables."""

    __name__: str


@dataclass(frozen=True, slots=True)
class ObjectStateFieldHelpInferencePolicy:
    """Bounds for resolving a field path to one ObjectState scope."""

    field_query_scan_limit: int = 1_000
    candidate_display_limit: int = 8


@dataclass(frozen=True, slots=True)
class ObjectStateFieldHelpService:
    """Describe ObjectState fields through path types and Python introspection."""

    ui_bridge_service: UiBridgeService
    function_catalog_service: FunctionCatalogService | None = None
    inference_policy: ObjectStateFieldHelpInferencePolicy = (
        ObjectStateFieldHelpInferencePolicy()
    )

    def describe_query(
        self,
        query: UiObjectStateFieldHelpQuery,
        connection: UiBridgeConnectionSpec,
    ) -> UiObjectStateFieldHelpResult:
        if query.object_state_scope_id is not None:
            return self.describe(
                query.concrete_request(query.object_state_scope_id),
                connection,
            )
        scope_id, error_result = self._infer_scope_id(query, connection)
        if error_result is not None:
            return error_result
        if scope_id is None:
            return self._query_error_result(
                query,
                AgentError(
                    code="ui_object_state_scope_inference_failed",
                    message="ObjectState field help scope inference returned no scope.",
                ),
            )
        return self.describe(query.concrete_request(scope_id), connection)

    def describe(
        self,
        request: UiObjectStateFieldHelpRequest,
        connection: UiBridgeConnectionSpec,
    ) -> UiObjectStateFieldHelpResult:
        catalog = self.ui_bridge_service.list_object_state_scopes(
            UiObjectStateScopeListRequest.from_visibility_options(
                UiObjectStateScopeVisibility(include_system_scopes=True),
                UiObjectStateFieldListOptions(
                    include_fields=True,
                    include_field_values=True,
                    include_field_descriptions=True,
                    field_limit=1,
                    field_paths=(request.field_path,),
                ),
            ),
            connection,
        )
        if catalog.errors:
            return UiObjectStateFieldHelpResult(
                schema_version=SCHEMA_VERSION,
                address=request,
                errors=catalog.errors,
                warnings=catalog.warnings,
            )

        scope, field = self._scope_and_field(
            catalog.scopes,
            request.object_state_scope_id,
            request.field_path,
        )
        if scope is None:
            return self._error_result(
                request,
                AgentError(
                    code="ui_object_state_scope_not_found",
                    message=(
                        "ObjectState scope is not registered in the running UI: "
                        f"{request.object_state_scope_id!r}"
                    ),
                ),
            )
        if field is None:
            return self._error_result(
                request,
                AgentError(
                    code="ui_object_state_field_not_found",
                    message=(
                        "ObjectState field is not registered on this scope: "
                        f"{request.field_path!r}"
                    ),
                ),
            )

        try:
            help_target = self._import_qualified_target(field.object_state_path_type)
            parameter_info = UnifiedParameterAnalyzer.analyze(help_target).get(
                field.field_name
            )
            parameter_type = None
            parameter_description = field.parameter_description
            if parameter_info is not None:
                parameter_type = parameter_info.param_type
                parameter_description = (
                    parameter_info.description or parameter_description
                )
            parameter_description = self._description_with_function_parameter_context(
                parameter_description,
                help_target,
                field.field_name,
                request.max_description_chars,
            )
            parameter_description = self._description_with_value_context(
                parameter_description,
                field,
                request.max_description_chars,
            )
            description, description_truncated = self._bounded_text(
                parameter_description or NO_PARAMETER_DESCRIPTION,
                request.max_description_chars,
            )
            target_docstring = DocstringExtractor.extract(help_target)
            return UiObjectStateFieldHelpResult(
                schema_version=SCHEMA_VERSION,
                address=field.address,
                field=field,
                object_type=scope.object_type,
                help_target_type=field.object_state_path_type,
                parameter_name=field.field_name,
                target_summary=target_docstring.summary,
                target_description=target_docstring.description,
                summary=self._parameter_summary(field.field_name, parameter_type),
                description=description,
                description_truncated=description_truncated,
                warnings=catalog.warnings,
            )
        except Exception as exc:
            return self._error_result(
                request,
                AgentError.from_exception(
                    "object_state_field_help_failed",
                    exc,
                ),
                field=field,
            )

    def _infer_scope_id(
        self,
        query: UiObjectStateFieldHelpQuery,
        connection: UiBridgeConnectionSpec,
    ) -> tuple[str | None, UiObjectStateFieldHelpResult | None]:
        catalog = self.ui_bridge_service.list_object_state_scopes(
            UiObjectStateScopeListRequest.from_visibility_options(
                UiObjectStateScopeVisibility(include_system_scopes=True),
                UiObjectStateFieldListOptions(
                    include_fields=True,
                    include_field_descriptions=False,
                    field_limit=self.inference_policy.field_query_scan_limit,
                    field_offset=0,
                    field_paths=(query.field_path,),
                ),
            ),
            connection,
        )
        if catalog.errors:
            return None, self._query_error_result(
                query,
                catalog.errors[0],
                warnings=catalog.warnings,
            )

        candidate_scope_ids = tuple(
            scope.identity.object_state_scope_id
            for scope in catalog.scopes
            for field in scope.fields
            if field.address.field_path == query.field_path
        )
        if len(candidate_scope_ids) == 1:
            return candidate_scope_ids[0], None

        if not candidate_scope_ids:
            return None, self._query_error_result(
                query,
                AgentError(
                    code="ui_object_state_field_not_found",
                    message=(
                        "ObjectState field is not registered in the running UI: "
                        f"{query.field_path!r}"
                    ),
                    hint=(
                        "Run openhcs_ui_get_object_state_fields with field_paths "
                        "or field_path_contains to inspect available fields."
                    ),
                    path=query.field_path,
                ),
                warnings=catalog.warnings,
            )

        visible_candidates = candidate_scope_ids[
            :self.inference_policy.candidate_display_limit
        ]
        more_count = len(candidate_scope_ids) - len(visible_candidates)
        candidate_text = ", ".join(repr(scope_id) for scope_id in visible_candidates)
        if more_count:
            candidate_text += f", ... (+{more_count} more)"
        return None, self._query_error_result(
            query,
            AgentError(
                code="ambiguous_ui_object_state_field",
                message=(
                    "ObjectState field path matched multiple scopes; pass "
                    "object_state_scope_id explicitly."
                ),
                hint=f"Candidate object_state_scope_id values: {candidate_text}",
                path=query.field_path,
            ),
            warnings=catalog.warnings,
        )

    @staticmethod
    def _scope_and_field(
        scopes: tuple[UiObjectStateScopeSummary, ...],
        scope_id: str,
        field_path: str,
    ) -> tuple[
        UiObjectStateScopeSummary | None,
        UiObjectStateFieldSummary | None,
    ]:
        for scope in scopes:
            if scope.identity.object_state_scope_id != scope_id:
                continue
            for field in scope.fields:
                if field.address.field_path == field_path:
                    return scope, field
            return scope, None
        return None, None

    @staticmethod
    def _import_qualified_target(
        qualified_name: str,
    ) -> type | Callable[..., object]:
        module_name, _, qualname = qualified_name.partition(":")
        if not qualname:
            module_name, _, qualname = qualified_name.rpartition(".")
        if not module_name or not qualname:
            raise ValueError(f"Help target name is not importable: {qualified_name!r}")
        module = importlib.import_module(module_name)
        symbol = ObjectStateFieldHelpService._resolve_qualname(module, qualname)
        if not isinstance(symbol, type) and not callable(symbol):
            raise TypeError(
                f"Imported symbol is not a class or callable: {qualified_name!r}"
            )
        return symbol

    @staticmethod
    def _resolve_qualname(module: ModuleType, qualname: str) -> object:
        symbol: object = module
        for name in qualname.split("."):
            symbol = vars(symbol)[name]
        return symbol

    @staticmethod
    def _parameter_summary(parameter_name: str, parameter_type: type | None) -> str:
        type_name = ObjectStateFieldHelpService._parameter_type_name(parameter_type)
        suffix = f" ({type_name})" if type_name else ""
        return f"• {parameter_name}{suffix}"

    @staticmethod
    def _parameter_type_name(parameter_type: type | None) -> str | None:
        if parameter_type is None:
            return None
        if isinstance(parameter_type, type):
            return parameter_type.__name__
        return inspect.formatannotation(parameter_type)

    def _description_with_value_context(
        self,
        parameter_description: str | None,
        field: UiObjectStateFieldSummary,
        max_description_chars: int,
    ) -> str | None:
        callable_context = self._callable_value_context(
            field.resolved_value,
            max_description_chars=max_description_chars,
        )
        if callable_context is None:
            callable_context = self._callable_value_context(
                field.raw_value,
                max_description_chars=max_description_chars,
            )
        if callable_context is None:
            return parameter_description
        if parameter_description:
            return f"{parameter_description}\n\n{callable_context}"
        return callable_context

    def _description_with_function_parameter_context(
        self,
        parameter_description: str | None,
        help_target: type | Callable[..., object],
        parameter_name: str,
        max_description_chars: int,
    ) -> str | None:
        context = self._function_parameter_context(
            help_target,
            parameter_name,
            max_description_chars=max_description_chars,
        )
        if context is None:
            return parameter_description
        if parameter_description:
            return f"{parameter_description}\n\n{context}"
        return context

    def _function_parameter_context(
        self,
        help_target: type | Callable[..., object],
        parameter_name: str,
        *,
        max_description_chars: int,
    ) -> str | None:
        if isinstance(help_target, type):
            return None
        import_path = _callable_import_path(help_target)
        if import_path is None:
            return None
        detail = self._function_detail(
            _CallableValueReference(
                import_path=import_path,
                name=(
                    help_target.__name__
                    if isinstance(help_target, CallableDisplayName)
                    else parameter_name
                ),
            ),
            max_description_chars=max_description_chars,
        )
        if detail is None:
            return None
        parameter = next(
            (
                candidate
                for candidate in detail.parameters
                if candidate.name == parameter_name
            ),
            None,
        )
        if parameter is None or parameter.supplied_by == "agent":
            return None

        pieces = [
            (
                "Function parameter contract: "
                f"{parameter.name} is supplied by OpenHCS "
                f"({parameter.supplied_by}); do not pass it as a FunctionStep kwarg."
            )
        ]
        if parameter.annotation:
            pieces.append(f"- type: {parameter.annotation}")
        artifact = _function_artifact_for_parameter(detail, parameter.name)
        if artifact is not None:
            pieces.append(
                "- artifact: "
                f"{artifact.name}:{artifact.kind} required={artifact.required}"
            )
        return "\n".join(pieces)

    def _callable_value_context(
        self,
        value: JsonValue | None,
        *,
        max_description_chars: int,
    ) -> str | None:
        references = self._callable_references(value)
        if not references:
            return None
        lines = ["Callable value:"]
        for reference in references:
            lines.append(f"- {reference.label}")
            if reference.kwargs:
                lines.append(f"  kwargs: {reference.kwargs}")
            function_detail = self._function_detail(reference, max_description_chars)
            if function_detail is not None:
                lines.extend(
                    _function_detail_context_lines(
                        function_detail,
                        active_kwarg_names=reference.kwarg_names,
                    )
                )
        return "\n".join(lines)

    def _function_detail(
        self,
        reference: "_CallableValueReference",
        max_description_chars: int,
    ) -> FunctionDetail | None:
        catalog = self.function_catalog_service
        if catalog is None:
            catalog = FunctionCatalogService()
        try:
            return catalog.get_by_import_path(
                reference.import_path,
                max_doc_chars=max(0, min(max_description_chars, 1_200)),
                compact_signature=True,
            )
        except Exception:
            return None

    @classmethod
    def _callable_references(
        cls,
        value: JsonValue | None,
    ) -> tuple["_CallableValueReference", ...]:
        references: list[_CallableValueReference] = []
        cls._collect_callable_references(value, references)
        return tuple(references)

    @classmethod
    def _collect_callable_references(
        cls,
        value: JsonValue | None,
        references: list["_CallableValueReference"],
    ) -> None:
        if isinstance(value, Mapping):
            callable_reference = _CallableValueReference.from_mapping(value)
            if callable_reference is not None:
                references.append(callable_reference)
                return
            for item in value.values():
                cls._collect_callable_references(item, references)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            callable_reference = _CallableValueReference.from_sequence(value)
            if callable_reference is not None:
                references.append(callable_reference)
                return
            for item in value:
                cls._collect_callable_references(item, references)

    @staticmethod
    def _bounded_text(value: str, max_chars: int) -> tuple[str, bool]:
        bounded_max = max(0, max_chars)
        if len(value) <= bounded_max:
            return value, False
        return (
            value[:bounded_max]
            + f"\n...<truncated {len(value) - bounded_max} chars>",
            True,
        )

    @staticmethod
    def _error_result(
        request: UiObjectStateFieldHelpRequest,
        error: AgentError,
        *,
        field: UiObjectStateFieldSummary | None = None,
    ) -> UiObjectStateFieldHelpResult:
        return UiObjectStateFieldHelpResult(
            schema_version=SCHEMA_VERSION,
            address=request,
            field=field,
            errors=(error,),
        )

    @staticmethod
    def _query_error_result(
        query: UiObjectStateFieldHelpQuery,
        error: AgentError,
        *,
        warnings: tuple[AgentWarning, ...] = (),
    ) -> UiObjectStateFieldHelpResult:
        return UiObjectStateFieldHelpResult(
            schema_version=SCHEMA_VERSION,
            address=query.error_address(),
            errors=(error,),
            warnings=warnings,
        )


@dataclass(frozen=True, slots=True)
class _CallableValueReference:
    import_path: str
    name: str
    kwargs: str | None = None
    kwarg_names: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return f"{self.name} ({self.import_path})"

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[object, object],
    ) -> "_CallableValueReference | None":
        if value.get("kind") != "callable":
            return None
        import_path = value.get("import_path")
        name = value.get("name")
        if not isinstance(import_path, str) or not isinstance(name, str):
            return None
        return cls(import_path=import_path, name=name)

    @classmethod
    def from_sequence(
        cls,
        value: Sequence[object],
    ) -> "_CallableValueReference | None":
        if len(value) != 2:
            return None
        first, second = value
        if not isinstance(first, Mapping):
            return None
        reference = cls.from_mapping(first)
        if reference is None:
            return None
        if not isinstance(second, Mapping):
            return reference
        return cls(
            import_path=reference.import_path,
            name=reference.name,
            kwargs=_kwargs_summary(second),
            kwarg_names=tuple(str(key) for key in second),
        )


def _kwargs_summary(value: Mapping[object, object]) -> str | None:
    if not value:
        return None
    parts = []
    for key, item in value.items():
        parts.append(f"{key}={_bounded_value_text(item)}")
    return ", ".join(parts)


def _callable_import_path(value: Callable[..., object]) -> str | None:
    if not isinstance(value, CallableImportIdentity):
        return None
    return f"{value.__module__}.{value.__qualname__}"


def _function_artifact_for_parameter(
    detail: FunctionDetail,
    parameter_name: str,
) -> FunctionArtifactSpec | None:
    contract = detail.runtime_contract
    if contract is None:
        return None
    for artifact in (*contract.artifact_inputs, *contract.runtime_artifact_inputs):
        if artifact.name == parameter_name:
            return artifact
    return None


def _function_detail_context_lines(
    detail: FunctionDetail,
    *,
    active_kwarg_names: tuple[str, ...],
) -> list[str]:
    lines = [
        f"  function_id: {detail.entry.function_id}",
        f"  signature: {detail.entry.signature}",
    ]
    if detail.entry.summary:
        lines.append(f"  summary: {detail.entry.summary}")
    lines.extend(_active_kwarg_context_lines(detail, active_kwarg_names))
    runtime_parameter_lines = _runtime_parameter_context_lines(detail)
    if runtime_parameter_lines:
        lines.append(f"  runtime supplied: {', '.join(runtime_parameter_lines)}")
    artifact_output_lines = _artifact_output_context_lines(detail)
    if artifact_output_lines:
        lines.append(f"  artifact outputs: {', '.join(artifact_output_lines)}")
    doc_lines = _doc_excerpt_lines(detail.doc)
    if doc_lines:
        lines.extend(doc_lines)
    return lines


def _active_kwarg_context_lines(
    detail: FunctionDetail,
    active_kwarg_names: tuple[str, ...],
) -> list[str]:
    if not active_kwarg_names:
        return []
    parameter_by_name = {
        parameter.name: parameter
        for parameter in detail.parameters
        if parameter.supplied_by == "agent"
    }
    lines = ["  active kwargs:"]
    for name in active_kwarg_names:
        parameter = parameter_by_name.get(name)
        if parameter is None:
            lines.append(f"    - {name}: not an agent-settable parameter")
            continue
        pieces = [name]
        if parameter.annotation:
            pieces.append(f"type={parameter.annotation}")
        if parameter.default_repr is not None:
            pieces.append(f"default={parameter.default_repr}")
        if parameter.required:
            pieces.append("required=True")
        if parameter.description:
            pieces.append(parameter.description)
        lines.append(f"    - {', '.join(pieces)}")
    return lines


def _runtime_parameter_context_lines(detail: FunctionDetail) -> list[str]:
    return [
        f"{parameter.name} ({parameter.supplied_by})"
        for parameter in detail.parameters
        if parameter.supplied_by != "agent"
    ]


def _artifact_output_context_lines(detail: FunctionDetail) -> list[str]:
    contract = detail.runtime_contract
    if contract is None:
        return []
    return [
        f"{artifact.name}:{artifact.kind}"
        for artifact in contract.artifact_outputs
    ]


def _doc_excerpt_lines(doc: str | None) -> list[str]:
    if not doc:
        return []
    excerpt = doc.strip()
    if not excerpt:
        return []
    if len(excerpt) > 800:
        excerpt = f"{excerpt[:797].rstrip()}..."
    return ["  doc excerpt:", *(f"    {line}" for line in excerpt.splitlines())]


def _bounded_value_text(value: object) -> str:
    text = repr(value)
    if len(text) <= 120:
        return text
    return f"{text[:117]}..."
