"""Lightweight authoring-context declarations for agent-facing guidance."""

from __future__ import annotations

from abc import ABC
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta


class AuthoringContextDeclaration(ABC, metaclass=AutoRegisterMeta):
    """Registered authoring-context guide exposed through MCP."""

    __registry__: ClassVar[dict[str, type["AuthoringContextDeclaration"]]] = {}
    __registry_key__ = "kind"
    __skip_if_no_key__ = True

    kind: ClassVar[str | None] = None

    @classmethod
    def context_facets(cls) -> tuple[type["AuthoringContextDeclaration"], ...]:
        return AuthoringContextDeclaration.facets_for(cls)

    @staticmethod
    def facets_for(
        target_cls: type,
    ) -> tuple[type["AuthoringContextDeclaration"], ...]:
        facets: list[type[AuthoringContextDeclaration]] = []
        for base in target_cls.__mro__:
            if base is target_cls or base is AuthoringContextDeclaration:
                continue
            if not issubclass(base, AuthoringContextDeclaration):
                continue
            facets.append(base)
        return tuple(facets)

    @classmethod
    def from_request(cls, kind: str) -> type["AuthoringContextDeclaration"]:
        request_kind = kind.casefold()
        try:
            return cls.__registry__[request_kind]
        except KeyError as exc:
            raise ValueError(
                f"kind must be one of: {', '.join(cls.allowed_values())}"
            ) from exc

    @classmethod
    def allowed_values(cls) -> tuple[str, ...]:
        return tuple(
            declaration.kind
            for declaration in cls.__registry__.values()
            if declaration.kind is not None
        )

    @classmethod
    def require_kind(cls) -> str:
        if cls.kind is None:
            raise ValueError(f"{cls.__name__} must declare an authoring context kind.")
        return cls.kind


class PipelineSystemModelContext(AuthoringContextDeclaration):
    """Sections that explain the FunctionStep/config/compiler model."""


class CellProfilerTranslationContext(AuthoringContextDeclaration):
    """Sections that bridge CellProfiler concepts into OpenHCS concepts."""


class RuntimeUiCoordinationContext(AuthoringContextDeclaration):
    """Sections that explain UI-visible execution versus headless execution."""


class StateCodeRoundtripContext(AuthoringContextDeclaration):
    """Sections that explain ObjectState and code-document round trips."""


class CustomFunctionRuntimeContext(AuthoringContextDeclaration):
    """Sections that explain runtime callable and artifact semantics."""


class SourceBindingWorkflowContext(AuthoringContextDeclaration):
    """Sections that explain source bindings and virtual workspaces."""


class ExampleCorpusContext(AuthoringContextDeclaration):
    """Sections that point agents at checked-in examples first."""


class PipelineAuthoringRulesContext(AuthoringContextDeclaration):
    """Sections that describe pipeline authoring syntax and schema hints."""


class CustomFunctionAuthoringRulesContext(AuthoringContextDeclaration):
    """Sections that describe custom function authoring and registration."""


class FirstUseWorkflowContext(AuthoringContextDeclaration):
    """Sections for contextless first-use MCP onboarding."""


class CoreOpenHCSModelContext(AuthoringContextDeclaration):
    """Sections that explain OpenHCS as a compiler/runtime system."""


class FolderOnboardingContext(AuthoringContextDeclaration):
    """Sections for taking a user from an image folder to a validated setup."""


class DomainExpertAssistedSetupContext(AuthoringContextDeclaration):
    """Sections for serving a domain expert who wants autonomous setup."""


class UiVisibleWorkflowContext(AuthoringContextDeclaration):
    """Sections for workflows that must be visible in the running UI."""


class HeadlessExecutionContext(AuthoringContextDeclaration):
    """Sections for direct orchestrator compile/run sessions."""


class ViewerReviewContext(AuthoringContextDeclaration):
    """Sections for image, payload, and ROI review through viewer tools."""


class ObjectStateEditingContext(AuthoringContextDeclaration):
    """Sections for typed ObjectState inspection and mutation."""


class PipelineMentalModelContext(
    PipelineSystemModelContext,
    CellProfilerTranslationContext,
    RuntimeUiCoordinationContext,
    StateCodeRoundtripContext,
    CustomFunctionRuntimeContext,
    SourceBindingWorkflowContext,
    ExampleCorpusContext,
):
    """Composable bundle for the core OpenHCS authoring mental model."""


class AgentCapabilityIndexContext(AuthoringContextDeclaration):
    """Sections that project the capability registry into prompt context."""


class PipelineAuthoringContext(
    PipelineMentalModelContext,
    PipelineAuthoringRulesContext,
    AuthoringContextDeclaration,
):
    kind = "pipeline"


class CustomFunctionAuthoringContext(
    StateCodeRoundtripContext,
    CustomFunctionRuntimeContext,
    CustomFunctionAuthoringRulesContext,
    AuthoringContextDeclaration,
):
    kind = "custom_function"


class FirstUseAuthoringContext(
    CoreOpenHCSModelContext,
    CellProfilerTranslationContext,
    FirstUseWorkflowContext,
    FolderOnboardingContext,
    UiVisibleWorkflowContext,
    ViewerReviewContext,
    AgentCapabilityIndexContext,
    AuthoringContextDeclaration,
):
    kind = "first_use"


class FolderOnboardingAuthoringContext(
    PipelineSystemModelContext,
    SourceBindingWorkflowContext,
    ExampleCorpusContext,
    FolderOnboardingContext,
    AuthoringContextDeclaration,
):
    kind = "folder_onboarding"


class DomainExpertAssistedSetupAuthoringContext(
    DomainExpertAssistedSetupContext,
    FolderOnboardingContext,
    UiVisibleWorkflowContext,
    ViewerReviewContext,
    AuthoringContextDeclaration,
):
    kind = "domain_expert_assisted_setup"


class UiVisibleWorkflowAuthoringContext(
    RuntimeUiCoordinationContext,
    StateCodeRoundtripContext,
    UiVisibleWorkflowContext,
    AuthoringContextDeclaration,
):
    kind = "ui_visible_workflow"


class HeadlessExecutionAuthoringContext(
    PipelineSystemModelContext,
    HeadlessExecutionContext,
    ViewerReviewContext,
    AuthoringContextDeclaration,
):
    kind = "headless_execution"


class ViewerReviewAuthoringContext(
    ViewerReviewContext,
    AuthoringContextDeclaration,
):
    kind = "viewer_review"


class ObjectStateEditingAuthoringContext(
    StateCodeRoundtripContext,
    ObjectStateEditingContext,
    AuthoringContextDeclaration,
):
    kind = "objectstate_editing"


class CellProfilerTranslationAuthoringContext(
    CellProfilerTranslationContext,
    ExampleCorpusContext,
    PipelineSystemModelContext,
    AuthoringContextDeclaration,
):
    kind = "cellprofiler_translation"
