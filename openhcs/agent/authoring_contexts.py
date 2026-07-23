"""Lightweight authoring-context declarations for agent-facing guidance."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.agent.dto.knowledge import KnowledgeBaseDocumentTarget


@dataclass(frozen=True, slots=True)
class AuthoringContextRoute:
    """Intent and deepening targets owned by one authoring-context kind."""

    title: str
    use_when: str
    knowledge_targets: tuple[KnowledgeBaseDocumentTarget, ...] = ()


class AuthoringContextDeclaration(ABC, metaclass=AutoRegisterMeta):
    """Registered authoring-context guide exposed through MCP."""

    __registry__: ClassVar[dict[str, type["AuthoringContextDeclaration"]]] = {}
    __registry_key__ = "kind"
    __skip_if_no_key__ = True

    kind: ClassVar[str | None] = None
    route: ClassVar[AuthoringContextRoute | None] = None

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

    @classmethod
    def require_route(cls) -> AuthoringContextRoute:
        if cls.route is None:
            raise ValueError(f"{cls.__name__} must declare an authoring context route.")
        return cls.route


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


class FolderOnboardingContext(AuthoringContextDeclaration):
    """Sections for taking a user from an image folder to a validated setup."""


class DomainExpertAssistedSetupContext(AuthoringContextDeclaration):
    """Sections for serving a domain expert who wants autonomous setup."""


class UiVisibleWorkflowContext(AuthoringContextDeclaration):
    """Sections for workflows that must be visible in the running UI."""


class HeadlessExecutionContext(AuthoringContextDeclaration):
    """Sections for direct orchestrator compile/run sessions."""


class DebuggingWorkflowContext(AuthoringContextDeclaration):
    """Sections for compiled-plan and paused-runtime diagnosis."""


class ViewerReviewContext(AuthoringContextDeclaration):
    """Sections for image, payload, and ROI review through viewer tools."""


class ObjectStateEditingContext(AuthoringContextDeclaration):
    """Sections for typed ObjectState inspection and mutation."""


class FirstUseAuthoringContext(
    FirstUseWorkflowContext,
    AuthoringContextDeclaration,
):
    kind = "first_use"
    route = AuthoringContextRoute(
        title="Choose an OpenHCS workflow",
        use_when=(
            "you do not yet know which OpenHCS state owner or workflow matches "
            "the request"
        ),
        knowledge_targets=(
            KnowledgeBaseDocumentTarget("openhcs_architecture_quick_start"),
        ),
    )


class UiVisibleWorkflowAuthoringContext(
    RuntimeUiCoordinationContext,
    StateCodeRoundtripContext,
    UiVisibleWorkflowContext,
    AuthoringContextDeclaration,
):
    kind = "ui_visible_workflow"
    route = AuthoringContextRoute(
        title="Work in the running desktop UI",
        use_when=(
            "the user asks to open, show, continue, or edit work in the visible "
            "OpenHCS desktop"
        ),
        knowledge_targets=(
            KnowledgeBaseDocumentTarget("openhcs_basic_interface"),
            KnowledgeBaseDocumentTarget("openhcs_code_ui_editing"),
        ),
    )


class DomainExpertAssistedSetupAuthoringContext(
    DomainExpertAssistedSetupContext,
    AuthoringContextDeclaration,
):
    kind = "domain_expert_assisted_setup"
    route = AuthoringContextRoute(
        title="Set up a workflow from biological intent",
        use_when=(
            "a domain expert describes the experiment and expected result but wants "
            "the agent to handle OpenHCS mechanics"
        ),
        knowledge_targets=(
            KnowledgeBaseDocumentTarget("openhcs_domain_expert_onboarding"),
            KnowledgeBaseDocumentTarget("openhcs_example_corpus_map"),
        ),
    )


class FolderOnboardingAuthoringContext(
    SourceBindingWorkflowContext,
    ExampleCorpusContext,
    FolderOnboardingContext,
    AuthoringContextDeclaration,
):
    kind = "folder_onboarding"
    route = AuthoringContextRoute(
        title="Inspect and onboard image data",
        use_when=(
            "the task starts from a local plate, image folder, or rich image "
            "container and does not require visible UI ownership"
        ),
        knowledge_targets=(
            KnowledgeBaseDocumentTarget("openhcs_image_sources"),
            KnowledgeBaseDocumentTarget("openhcs_source_model"),
            KnowledgeBaseDocumentTarget("openhcs_example_corpus_map"),
            KnowledgeBaseDocumentTarget("openhcs_official30_benchmark_recipes"),
        ),
    )


class PipelineAuthoringContext(
    PipelineSystemModelContext,
    ExampleCorpusContext,
    PipelineAuthoringRulesContext,
    AuthoringContextDeclaration,
):
    kind = "pipeline"
    route = AuthoringContextRoute(
        title="Author or revise a typed pipeline",
        use_when=(
            "the sources are understood and the next task is choosing functions, "
            "axes, grouping, configuration, or artifact contracts"
        ),
        knowledge_targets=(
            KnowledgeBaseDocumentTarget("openhcs_configuration_reference"),
            KnowledgeBaseDocumentTarget("openhcs_data_dimensions"),
            KnowledgeBaseDocumentTarget("openhcs_function_patterns"),
            KnowledgeBaseDocumentTarget("openhcs_processing_semantics"),
            KnowledgeBaseDocumentTarget("openhcs_artifact_contract_system"),
            KnowledgeBaseDocumentTarget("openhcs_official30_benchmark_recipes"),
        ),
    )


class CustomFunctionAuthoringContext(
    CustomFunctionRuntimeContext,
    CustomFunctionAuthoringRulesContext,
    AuthoringContextDeclaration,
):
    kind = "custom_function"
    route = AuthoringContextRoute(
        title="Author a custom processing function",
        use_when=(
            "the required operation is not already in the function registry or "
            "needs a typed custom artifact contract"
        ),
        knowledge_targets=(
            KnowledgeBaseDocumentTarget("openhcs_custom_functions"),
            KnowledgeBaseDocumentTarget("openhcs_custom_function_management"),
            KnowledgeBaseDocumentTarget("openhcs_artifact_contract_system"),
        ),
    )


class HeadlessExecutionAuthoringContext(
    PipelineSystemModelContext,
    HeadlessExecutionContext,
    ViewerReviewContext,
    AuthoringContextDeclaration,
):
    kind = "headless_execution"
    route = AuthoringContextRoute(
        title="Compile and execute headlessly",
        use_when=(
            "a reviewed pipeline should run without becoming selected or editable "
            "in the desktop UI"
        ),
        knowledge_targets=(
            KnowledgeBaseDocumentTarget("openhcs_pipeline_compilation_system"),
            KnowledgeBaseDocumentTarget("openhcs_runtime_value_system"),
            KnowledgeBaseDocumentTarget("openhcs_measurement_equivalence"),
        ),
    )


class DebuggingAuthoringContext(
    DebuggingWorkflowContext,
    AuthoringContextDeclaration,
):
    kind = "debugging"
    route = AuthoringContextRoute(
        title="Diagnose compiled and runtime behavior",
        use_when=(
            "a compile, execution, step output, runtime value, or artifact must "
            "be inspected at its owning boundary"
        ),
        knowledge_targets=(
            KnowledgeBaseDocumentTarget("openhcs_pipeline_debugging"),
            KnowledgeBaseDocumentTarget("openhcs_pipeline_compilation_system"),
            KnowledgeBaseDocumentTarget("openhcs_runtime_value_system"),
            KnowledgeBaseDocumentTarget("openhcs_artifact_contract_system"),
        ),
    )


class ViewerReviewAuthoringContext(
    ViewerReviewContext,
    AuthoringContextDeclaration,
):
    kind = "viewer_review"
    route = AuthoringContextRoute(
        title="Validate results in a managed viewer",
        use_when=(
            "execution produced images, labels, or ROIs that need bounded visual "
            "and structured validation"
        ),
        knowledge_targets=(
            KnowledgeBaseDocumentTarget("openhcs_viewer_management"),
        ),
    )


class ObjectStateEditingAuthoringContext(
    StateCodeRoundtripContext,
    ObjectStateEditingContext,
    AuthoringContextDeclaration,
):
    kind = "objectstate_editing"
    route = AuthoringContextRoute(
        title="Inspect or edit typed UI state",
        use_when=(
            "one exact UI-backed field, inheritance value, snapshot, or branch "
            "must be inspected or changed"
        ),
        knowledge_targets=(
            KnowledgeBaseDocumentTarget("openhcs_code_ui_editing"),
            KnowledgeBaseDocumentTarget("openhcs_code_ui_interconversion"),
        ),
    )


class CellProfilerTranslationAuthoringContext(
    CellProfilerTranslationContext,
    ExampleCorpusContext,
    PipelineSystemModelContext,
    AuthoringContextDeclaration,
):
    kind = "cellprofiler_translation"
    route = AuthoringContextRoute(
        title="Translate or validate a CellProfiler workflow",
        use_when=(
            "the intent begins with a .cppipe, CellProfiler module vocabulary, or "
            "an exact parity claim"
        ),
        knowledge_targets=(
            KnowledgeBaseDocumentTarget("openhcs_cellprofiler_interop"),
            KnowledgeBaseDocumentTarget("openhcs_official30_benchmark_recipes"),
        ),
    )
