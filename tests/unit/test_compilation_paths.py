"""Deterministic compiler resolution for declaration-owned plate paths."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import get_type_hints

import pytest
from objectstate import ObjectState, ObjectStateRegistry
from python_introspect import Enableable

from openhcs.core.callable_contract import CallableContract
from openhcs.core.config import (
    CompilationDebugConfig,
    GlobalPipelineConfig,
    PathPlanningConfig,
    PipelineConfig,
)
from objectstate.lazy_factory import ensure_global_config_context
from openhcs.core.function_patterns import compile_function_pattern
from openhcs.core.pipeline.compilation_session import (
    CompilationPathResolver,
    CompilationPlateScope,
    resolve_declared_dataclass_paths,
)
from openhcs.core.pipeline.path_planner import _cached_output_plate_root
from openhcs.core.vfs_protocol import (
    FileManagerLike,
    PlateInputDirectoryDeclaration,
    PlateInputFile,
    PlateInputFileDeclaration,
    PlateOutputDirectory,
    PlateOutputDirectoryDeclaration,
    PlateOutputFile,
    PlateOutputFileDeclaration,
    PlatePathDeclaration,
)


class RecordingFileManager(FileManagerLike):
    def __init__(
        self,
        *,
        files: Sequence[Path] = (),
        directories: Sequence[Path] = (),
    ) -> None:
        self.files = frozenset(files)
        self.directories = frozenset(directories)
        self.resolve_calls: list[tuple[Path, str, Path]] = []
        self.validation_calls: list[tuple[str, Path, str]] = []

    def list_files(
        self, directory: str | Path, backend: str, **kwargs: object
    ) -> list[str]:
        del directory, backend, kwargs
        return []

    def exists(self, path: str | Path, backend: str) -> bool:
        target = Path(path)
        self.validation_calls.append(("exists", target, backend))
        return target in self.files or target in self.directories

    def is_dir(self, path: str | Path, backend: str) -> bool:
        target = Path(path)
        self.validation_calls.append(("is_dir", target, backend))
        return target in self.directories

    def load(self, file_path: str | Path, backend: str, **kwargs: object) -> object:
        raise NotImplementedError

    def load_batch(
        self,
        file_paths: Sequence[str | Path],
        backend: str,
        **kwargs: object,
    ) -> Sequence[object]:
        raise NotImplementedError

    def resolve_address(
        self,
        backend_address: str | Path,
        backend: str,
        *,
        base_path: str | Path,
    ) -> str | Path:
        authored = Path(backend_address)
        base = Path(base_path)
        self.resolve_calls.append((authored, backend, base))
        return authored if authored.is_absolute() else base / authored

    physical_source_path = resolve_address

    def save(
        self,
        data: object,
        output_path: str | Path,
        backend: str,
        **kwargs: object,
    ) -> None:
        raise NotImplementedError

    def ensure_directory(self, directory: str | Path, backend: str) -> str:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class NestedPathConfig:
    debug_bundle: PlateOutputFile = Path("debug/bundle.pkl")


@dataclass(frozen=True, slots=True)
class DeclaredPathConfig:
    template: PlateInputFile = Path("templates/template.tif")
    output_root: PlateOutputDirectory = Path("exports")
    nested: NestedPathConfig = NestedPathConfig()
    ordinary_path: Path = Path("owned/by/another/subsystem")


def _path_callable(
    image: object,
    template_path: PlateInputFile = "templates/template.tif",
) -> object:
    return image


def _resolver() -> tuple[CompilationPathResolver, RecordingFileManager]:
    plate_root = Path("/plates/plate-a")
    filemanager = RecordingFileManager(
        files=(plate_root / "templates/template.tif",),
        directories=(plate_root / "templates",),
    )
    return (
        CompilationPathResolver(
            plate_scope=CompilationPlateScope.from_path(plate_root),
            filemanager=filemanager,
            backend="disk",
        ),
        filemanager,
    )


def test_plate_scope_delegates_relative_resolution_to_vfs_with_explicit_base() -> None:
    resolver, filemanager = _resolver()

    resolved = resolver.resolve(
        "templates/template.tif",
        PlateInputFileDeclaration(),
        owner="test template",
    )

    assert resolved == Path("/plates/plate-a/templates/template.tif")
    assert filemanager.resolve_calls == [
        (Path("templates/template.tif"), "disk", Path("/plates/plate-a"))
    ]
    assert filemanager.validation_calls == [
        ("exists", resolved, "disk"),
        ("is_dir", resolved, "disk"),
    ]


def test_plate_scope_rejects_relative_compilation_root_at_construction() -> None:
    with pytest.raises(ValueError, match="must be absolute"):
        CompilationPlateScope.from_path("relative/plate")


def test_input_declarations_validate_kind_and_output_declarations_do_not_probe() -> None:
    resolver, filemanager = _resolver()

    with pytest.raises(NotADirectoryError, match="not a directory"):
        resolver.resolve(
            "templates/template.tif",
            PlateInputDirectoryDeclaration(),
            owner="test directory",
        )

    filemanager.validation_calls.clear()
    output = resolver.resolve(
        "new/output.tif",
        PlateOutputFileDeclaration(),
        owner="test output",
    )
    assert output == Path("/plates/plate-a/new/output.tif")
    assert filemanager.validation_calls == []


def test_dataclass_resolution_returns_absolute_copy_and_preserves_authored_value() -> None:
    resolver, _filemanager = _resolver()
    authored = DeclaredPathConfig()

    compiled = resolve_declared_dataclass_paths(
        authored,
        resolver,
        owner="DeclaredPathConfig",
    )

    assert compiled is not authored
    assert compiled.template == Path("/plates/plate-a/templates/template.tif")
    assert compiled.output_root == Path("/plates/plate-a/exports")
    assert compiled.nested.debug_bundle == Path("/plates/plate-a/debug/bundle.pkl")
    assert compiled.ordinary_path == Path("owned/by/another/subsystem")
    assert authored == DeclaredPathConfig()


def test_callable_compilation_resolves_authored_and_default_declared_paths() -> None:
    resolver, _filemanager = _resolver()

    compiled_default = compile_function_pattern(
        _path_callable,
        {},
        {},
        path_resolver=resolver,
    )
    compiled_authored = compile_function_pattern(
        (_path_callable, {"template_path": Path("templates/template.tif")}),
        {},
        {},
        path_resolver=resolver,
    )

    expected = Path("/plates/plate-a/templates/template.tif")
    assert next(compiled_default.iter_invocations()).kwargs_dict == {
        "template_path": expected
    }
    assert next(compiled_authored.iter_invocations()).kwargs_dict == {
        "template_path": expected
    }


def test_relative_declared_path_without_compilation_scope_fails_loudly() -> None:
    with pytest.raises(ValueError, match="CompilationPathResolver"):
        compile_function_pattern(_path_callable, {}, {})


def test_config_annotations_delegate_to_nominal_path_declarations() -> None:
    path_planning_annotations = get_type_hints(
        PathPlanningConfig,
        include_extras=True,
    )
    debug_annotations = get_type_hints(
        CompilationDebugConfig,
        include_extras=True,
    )

    assert isinstance(
        PlatePathDeclaration.from_annotation(
            path_planning_annotations["global_output_folder"]
        ),
        PlateOutputDirectoryDeclaration,
    )
    assert PlatePathDeclaration.from_annotation(
        debug_annotations["compiled_execution_bundle_path"]
    ) == PlatePathDeclaration.from_annotation(PlateOutputFile)
    assert CallableContract.from_callable(
        _path_callable
    ).declared_path_parameters.keys() == {"template_path"}


def test_compilation_debug_enabled_field_is_inherited_from_enableable() -> None:
    enabled_field = next(
        config_field
        for config_field in fields(CompilationDebugConfig)
        if config_field.name == "enabled"
    )

    assert "enabled" not in CompilationDebugConfig.__annotations__
    assert enabled_field is Enableable.callable_field()
    assert enabled_field.default is True
    assert CompilationDebugConfig().enabled is True

    ensure_global_config_context(
        GlobalPipelineConfig,
        GlobalPipelineConfig(
            compilation_debug_config=CompilationDebugConfig(enabled=False),
        ),
    )
    try:
        pipeline_config = PipelineConfig()
        state = ObjectState(pipeline_config, scope_id="compilation-debug")
        ObjectStateRegistry.register(state, _skip_snapshot=True)
        assert state.parameters["compilation_debug_config.enabled"] is None
        assert state.get_resolved_value("compilation_debug_config.enabled") is False
    finally:
        ObjectStateRegistry.clear()
        ensure_global_config_context(
            GlobalPipelineConfig,
            GlobalPipelineConfig(),
        )


def test_compilation_debug_is_last_in_generated_pipeline_config_order() -> None:
    global_field_names = tuple(
        config_field.name for config_field in fields(GlobalPipelineConfig)
    )
    pipeline_field_names = tuple(
        config_field.name for config_field in fields(PipelineConfig)
    )

    assert pipeline_field_names == global_field_names
    assert global_field_names[-1] == "compilation_debug_config"


def test_path_planner_rejects_uncompiled_relative_global_output_folder() -> None:
    _cached_output_plate_root.cache_clear()
    with pytest.raises(ValueError, match="requires compiled global_output_folder"):
        _cached_output_plate_root("/plates/plate-a", "exports", "_openhcs")
