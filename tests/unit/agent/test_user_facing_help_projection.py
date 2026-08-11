from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, fields
import inspect
from pathlib import Path

from objectstate import ObjectState
from pyqt_reactive.services.parameter_help_service import (
    dataclass_parameter_descriptions,
    docstring_info_for_target,
)
from python_introspect import (
    DocstringExtractor,
    mark_enableable,
    signature_analysis_target,
)
from python_introspect.signature_analyzer import ParametersDocstringSection

from openhcs.agent.services.function_catalog_service import FunctionCatalogService
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.processing.backends.analysis.cell_counting_cpu import (
    count_cells_single_channel,
)
from openhcs.processing.custom_functions import manager as custom_function_manager
from openhcs.pyqt_gui.config import AgentUiBridgeConfig, UIConfig
from openhcs.runtime.zmq_config import OpenHCSZMQConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REJECTED_HELP_FRAGMENTS = (
    "input image payload processed by this callable",
    "object-label payload whose labeled regions are processed",
    "backend used by this callable",
)
FORBIDDEN_HELP_MIRROR_SYMBOLS = frozenset(
    {
        "AdditionalParametersDocstringSection",
        "CellProfilerSettingParameterDescriptionProvider",
        "EXACT_DESCRIPTIONS",
        "LABEL_DESCRIPTIONS",
        "PROVIDER_DESCRIPTIONS",
        "UnifiedParameterDescriptionProvider",
        "description_for",
    }
)


@dataclass(frozen=True)
class _Metadata:
    func: Callable
    original_name: str
    name: str
    module: str
    doc: str
    tags: tuple[str, ...] = ("custom",)

    @classmethod
    def from_function(cls, func: Callable) -> "_Metadata":
        return cls(
            func=func,
            original_name=func.__name__,
            name=func.__name__,
            module=func.__module__,
            doc=func.__doc__ or "",
        )

    @property
    def display_name(self) -> str:
        return self.original_name

    def get_registry_name(self) -> str:
        return "openhcs"


def _is_repository_owned_metadata(metadata: object) -> bool:
    return metadata.get_registry_name() == "openhcs" and "custom" not in {
        str(tag).casefold() for tag in (metadata.tags or ())
    }


def _repository_owned_metadata(
    service: FunctionCatalogService,
) -> dict[str, object]:
    return {
        function_id: metadata
        for function_id, metadata in sorted(service._all_metadata().items())
        if _is_repository_owned_metadata(metadata)
    }


def _source_docstring_parameters(docstring: str) -> dict[str, str]:
    def source_docstring_target() -> None:
        pass

    source_docstring_target.__doc__ = docstring
    return DocstringExtractor.extract(source_docstring_target).parameters or {}


def _assignment_names(node: ast.Assign | ast.AnnAssign) -> tuple[str, ...]:
    targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
    return tuple(target.id for target in targets if isinstance(target, ast.Name))


def test_public_ui_and_zmq_config_fields_project_declaration_help() -> None:
    for config_type in (AgentUiBridgeConfig, OpenHCSZMQConfig):
        descriptions = dataclass_parameter_descriptions(config_type)
        field_names = {field.name for field in fields(config_type)}

        assert set(descriptions) == field_names
        assert all(descriptions[name].strip() for name in field_names)

        state = ObjectState(config_type(), scope_id=config_type.__name__)
        assert state.parameter_descriptions == descriptions

    assert "UI bridge" in dataclass_parameter_descriptions(
        AgentUiBridgeConfig
    )["enabled"]
    assert "First data port" in dataclass_parameter_descriptions(
        OpenHCSZMQConfig
    )["default_port"]


def test_custom_registered_callable_uses_shared_authored_parameter_help(
    monkeypatch,
    tmp_path,
) -> None:
    registrations: list[Callable] = []
    monkeypatch.setattr(
        custom_function_manager,
        "register_function",
        lambda func, backend: registrations.append(func),
    )
    manager = custom_function_manager.CustomFunctionManager()
    manager.storage_dir = tmp_path
    (custom_func,) = manager.register_from_code(
        '''
@numpy
def codex_help_projection_probe(image, scale: float = 1.0):
    """Scale an image for the help-projection acceptance test.

    Args:
        image: Input image stack to scale.
        scale: Multiplicative intensity factor.

    Returns:
        The scaled image stack.
    """
    return image * scale
''',
        persist=False,
        clear_caches=False,
        emit_signal=False,
        collision_metadata={},
    )
    metadata = _Metadata.from_function(custom_func)
    function_id = "openhcs:codex_help_projection_probe"
    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self: {function_id: metadata},
    )

    detail = FunctionCatalogService().get(function_id)
    projected = {parameter.name: parameter for parameter in detail.parameters}
    authored = docstring_info_for_target(custom_func).parameters

    assert registrations == [custom_func]
    assert authored is not None
    assert projected["scale"].description == authored["scale"]
    assert projected["image"].description is not None
    assert projected["image"].description.startswith(authored["image"])
    assert "Supplied by OpenHCS" in projected["image"].description
    assert projected["slice_by_slice"].description == authored["slice_by_slice"]
    assert "numpy memory decorator" in projected["slice_by_slice"].description


def test_enableable_callable_projects_its_nominal_parameter_help(monkeypatch) -> None:
    def enableable_probe(image, *, enabled: bool = True):
        """Return an image when this step is enabled.

        Args:
            image: Input image stack.
        """

        return image

    mark_enableable(enableable_probe)
    metadata = _Metadata.from_function(enableable_probe)
    function_id = "openhcs:enableable_probe"
    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self: {function_id: metadata},
    )

    detail = FunctionCatalogService().get(function_id)
    projected = {parameter.name: parameter for parameter in detail.parameters}
    authored = docstring_info_for_target(enableable_probe).parameters

    assert authored is not None
    assert authored["enabled"] == (
        "Run this callable or configuration when enabled; skip it when disabled."
    )
    assert projected["enabled"].description == authored["enabled"]


def test_builtin_callable_projects_complete_authored_parameter_help(
    monkeypatch,
) -> None:
    metadata = _Metadata.from_function(count_cells_single_channel)
    function_id = "openhcs:count_cells_single_channel"
    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self: {function_id: metadata},
    )

    detail = FunctionCatalogService().get(function_id)
    projected = {parameter.name: parameter for parameter in detail.parameters}

    assert all(parameter.description for parameter in detail.parameters)
    assert projected["threshold"].description == (
        "Detection threshold (method-dependent)"
    )
    assert projected["watershed_max_eccentricity"].description == (
        "Maximum object eccentricity eligible for watershed splitting "
        "(0=circle, 1=line)"
    )


def test_preprocessing_guidance_projects_from_callable_owners(monkeypatch) -> None:
    from openhcs.processing.backends.cellprofiler.illumination import (
        correct_illumination_calculate,
    )
    from openhcs.processing.backends.enhance.basic_processor_numpy import (
        basic_flatfield_correction_numpy,
    )
    from openhcs.processing.backends.processors.numpy_processor import (
        percentile_normalize,
        stack_percentile_normalize,
        tophat,
    )

    expected_phrases = {
        percentile_normalize: (
            "Normalize each plane independently",
            "Absolute intensity",
            "Inspect representative raw/normalized pairs",
        ),
        stack_percentile_normalize: (
            "declared leading array axis",
            "relative intensities between unclipped pixels are preserved",
            "fraction clipped at each endpoint",
        ),
        tophat: (
            "white top-hat background subtraction",
            "foreground/background size boundary",
            "raw/corrected overlays",
        ),
        basic_flatfield_correction_numpy: (
            "low-rank and sparse decomposition",
            "Estimate fields separately for independent acquisition channels",
            "does not restore an external intensity calibration",
        ),
        correct_illumination_calculate: (
            "smooth illumination correction function",
            "local object-scale background removal",
            "foreground biology",
        ),
    }
    metadata_by_id = {
        f"test:{func.__name__}": _Metadata.from_function(func)
        for func in expected_phrases
    }
    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self: metadata_by_id,
    )
    service = FunctionCatalogService()

    for function_id, metadata in metadata_by_id.items():
        detail = service.get(function_id, compact_signature=False)
        shared_help = docstring_info_for_target(metadata.func)
        shared_text = "\n".join(
            part
            for part in (shared_help.summary, shared_help.description)
            if part
        )
        normalized_detail_doc = " ".join(detail.doc.split())
        normalized_shared_text = " ".join(shared_text.split())

        assert detail.doc == inspect.getdoc(metadata.func)
        assert detail.entry.summary == shared_help.summary
        for phrase in expected_phrases[metadata.func]:
            assert phrase in normalized_detail_doc
            assert phrase in normalized_shared_text


def test_cellprofiler_setting_binding_owns_callable_parameter_help() -> None:
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
    from openhcs.processing.backends.cellprofiler import align

    module_type = CellProfilerModule.require_module("Align")
    authored = docstring_info_for_target(align).parameters or {}
    parsed = DocstringExtractor.extract(align).parameters or {}
    binding = next(
        binding
        for binding in module_type.declared_setting_bindings()
        if binding.require_parameter_name() == "method"
    )

    assert parsed["method"] == binding.parameter_help_description()
    assert authored["method"] == binding.parameter_help_description()
    assert "Select the alignment method" in authored["method"]


def test_cellprofiler_variant_help_follows_its_analysis_target() -> None:
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule

    module_type = CellProfilerModule.require_module("MeasureTexture")
    variant = module_type.require_callable("measure_texture_objects")
    help_target = signature_analysis_target(variant)
    authored = docstring_info_for_target(variant).parameters or {}
    binding = next(
        binding
        for binding in module_type.declared_setting_bindings()
        if binding.require_parameter_name() == "measurement_scope"
    )

    assert help_target is variant
    assert authored["measurement_scope"] == binding.parameter_help_description()


def test_canonical_cellprofiler_catalog_preserves_final_callable_help() -> None:
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule

    service = FunctionCatalogService()
    function_id = "openhcs:cellprofiler_align"
    detail = service.get(function_id, compact_signature=False)
    projected = {parameter.name: parameter for parameter in detail.parameters}
    module_type = CellProfilerModule.require_module("Align")
    bindings = {
        binding.require_parameter_name(): binding
        for binding in module_type.declared_setting_bindings()
    }

    for name in ("method", "crop_mode", "additional_alignment_modes"):
        assert projected[name].description == bindings[name].parameter_help_description()

    assert projected["enabled"].description == (
        "Run this callable or configuration when enabled; skip it when disabled."
    )
    catalog_callable = service.resolve(function_id)
    assert signature_analysis_target(catalog_callable) is catalog_callable.__wrapped__


def test_untangle_worm_help_targets_follow_module_declaration_authority() -> None:
    from openhcs.processing.backends.cellprofiler import worms

    tree = ast.parse(Path(worms.__file__).read_text(encoding="utf-8"))
    target_loops = []
    for node in tree.body:
        if not isinstance(node, ast.For):
            continue
        target_calls = [
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "set_signature_analysis_target"
        ]
        if target_calls:
            target_loops.append((node, target_calls))

    assert len(target_loops) == 1
    loop, target_calls = target_loops[0]
    assert isinstance(loop.target, ast.Name)
    assert isinstance(loop.iter, ast.Call)
    assert isinstance(loop.iter.func, ast.Attribute)
    assert isinstance(loop.iter.func.value, ast.Name)
    assert loop.iter.func.value.id == "UntangleWormsModule"
    assert loop.iter.func.attr == "declared_function_names"

    assert len(target_calls) == 1
    target_call = target_calls[0]
    resolved_callable = target_call.args[0]
    assert isinstance(resolved_callable, ast.Call)
    assert isinstance(resolved_callable.func, ast.Attribute)
    assert isinstance(resolved_callable.func.value, ast.Name)
    assert resolved_callable.func.value.id == "UntangleWormsModule"
    assert resolved_callable.func.attr == "require_callable"
    assert len(resolved_callable.args) == 1
    assert isinstance(resolved_callable.args[0], ast.Name)
    assert resolved_callable.args[0].id == loop.target.id

    copied_variant_collections = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Tuple, ast.List))
        and sum(
            isinstance(element, ast.Name)
            and element.id.startswith("untangle_worms")
            for element in node.elts
        )
        > 1
    ]
    assert copied_variant_collections == []


def test_all_public_config_paths_project_declaration_help() -> None:
    for config_type in (UIConfig, GlobalPipelineConfig, PipelineConfig):
        state = ObjectState(
            config_type(),
            scope_id=f"user-help-coverage:{config_type.__name__}",
        )
        descriptions = state.parameter_descriptions

        assert set(descriptions) == set(state.parameters)
        assert {
            path: description
            for path, description in descriptions.items()
            if not (description or "").strip()
        } == {}


def test_repository_catalog_has_actionable_parameter_help() -> None:
    service = FunctionCatalogService()
    missing: dict[str, tuple[str, ...]] = {}
    weak: dict[str, tuple[tuple[str, str], ...]] = {}
    supplier_only_without_summary: dict[str, tuple[str, ...]] = {}

    for function_id in _repository_owned_metadata(service):
        detail = service.get(function_id, compact_signature=False)
        missing_parameters = tuple(
            parameter.name
            for parameter in detail.parameters
            if not (parameter.description or "").strip()
        )
        if missing_parameters:
            missing[function_id] = missing_parameters

        weak_parameters = tuple(
            (parameter.name, parameter.description or "")
            for parameter in detail.parameters
            if any(
                fragment in (parameter.description or "").casefold()
                for fragment in REJECTED_HELP_FRAGMENTS
            )
        )
        if weak_parameters:
            weak[function_id] = weak_parameters

        if not (detail.entry.summary or "").strip():
            supplier_only = tuple(
                parameter.name
                for parameter in detail.parameters
                if (parameter.description or "").startswith("Supplied by OpenHCS")
            )
            if supplier_only:
                supplier_only_without_summary[function_id] = supplier_only

    assert missing == {}
    assert weak == {}
    assert supplier_only_without_summary == {}


def test_undocumented_custom_code_is_not_subject_to_repository_authored_gate(
    monkeypatch,
) -> None:
    def undocumented_custom_function(image, scale: float = 1.0):
        return image * scale

    metadata = _Metadata.from_function(undocumented_custom_function)
    function_id = "openhcs:undocumented_custom_function"
    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self: {function_id: metadata},
    )
    service = FunctionCatalogService()
    detail = service.get(function_id, compact_signature=False)
    parameters = {parameter.name: parameter for parameter in detail.parameters}

    assert parameters["scale"].description is None
    assert _repository_owned_metadata(service) == {}


def test_repository_callable_source_has_no_duplicated_leaf_semantics() -> None:
    service = FunctionCatalogService()
    source_trees: dict[Path, ast.Module] = {}
    descriptions: dict[str, list[tuple[tuple[Path, int, str], str]]] = defaultdict(
        list
    )

    for metadata in _repository_owned_metadata(service).values():
        owner = inspect.unwrap(metadata.func)
        source_filename = inspect.getsourcefile(owner)
        assert source_filename is not None
        source_path = Path(source_filename).resolve()
        relative_path = source_path.relative_to(REPOSITORY_ROOT)
        tree = source_trees.setdefault(
            source_path,
            ast.parse(source_path.read_text(encoding="utf-8")),
        )
        candidates = tuple(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == owner.__name__
        )
        assert candidates
        declaration = min(
            candidates,
            key=lambda node: abs(node.lineno - owner.__code__.co_firstlineno),
        )
        docstring = ast.get_docstring(declaration, clean=True)
        if not docstring:
            continue
        signature_names = set(inspect.signature(metadata.func).parameters)
        declaration_id = (relative_path, declaration.lineno, declaration.name)
        for name, description in _source_docstring_parameters(docstring).items():
            if name not in signature_names or not description.strip():
                continue
            normalized_description = " ".join(description.split())
            descriptions[normalized_description].append((declaration_id, name))

    duplicates = {
        description: tuple(
            f"{owner[0]}:{owner[1]}:{owner[2]}.{name}"
            for owner, name in references
        )
        for description, references in descriptions.items()
        if len({owner for owner, _name in references}) > 1
    }
    assert duplicates == {}


def test_help_projection_has_no_semantic_mirror_or_name_dispatch() -> None:
    assert not (REPOSITORY_ROOT / "scripts/_user_help_docstring_codemod.py").exists()
    production_roots = (
        REPOSITORY_ROOT / "openhcs",
        REPOSITORY_ROOT / "external/arraybridge/src",
        REPOSITORY_ROOT / "external/pyqt-reactive/src",
        REPOSITORY_ROOT / "external/python-introspect/src",
        REPOSITORY_ROOT / "scripts",
    )
    forbidden_definitions: list[str] = []
    module_description_maps: list[str] = []

    for root in production_roots:
        for source_path in root.rglob("*.py"):
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            relative_path = source_path.relative_to(REPOSITORY_ROOT)
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and (
                    node.name in FORBIDDEN_HELP_MIRROR_SYMBOLS
                ):
                    forbidden_definitions.append(
                        f"{relative_path}:{node.lineno}:{node.name}"
                    )
            for node in tree.body:
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                value = node.value
                if not isinstance(value, ast.Dict):
                    continue
                for name in _assignment_names(node):
                    normalized_name = name.casefold()
                    if name in FORBIDDEN_HELP_MIRROR_SYMBOLS or normalized_name.endswith(
                        ("descriptions", "description_map")
                    ):
                        module_description_maps.append(
                            f"{relative_path}:{node.lineno}:{name}"
                        )

    assert forbidden_definitions == []
    assert module_description_maps == []
    assert "additional parameters:" in ParametersDocstringSection.colon_headers
    assert "additional parameters" in ParametersDocstringSection.numpy_headers
