"""Generated public source and plate-relative compilation integration."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path

from openhcs.core.function_patterns import compile_function_pattern
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.pipeline.compilation_session import (
    CompilationPathResolver,
    CompilationPlateScope,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.vfs_protocol import FileManagerLike
from openhcs.processing.backends.cellprofiler.imagej_macro import run_imagej_macro


class FilesystemFileManager(FileManagerLike):
    def list_files(
        self,
        directory: str | Path,
        backend: str,
        **kwargs: object,
    ) -> list[str]:
        del backend, kwargs
        return [str(path) for path in Path(directory).iterdir()]

    def exists(self, path: str | Path, backend: str) -> bool:
        del backend
        return Path(path).exists()

    def is_dir(self, path: str | Path, backend: str) -> bool:
        del backend
        return Path(path).is_dir()

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
        del backend
        authored = Path(backend_address)
        return authored if authored.is_absolute() else Path(base_path) / authored

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


def _compiled_template_path(
    step: FunctionStep,
    *,
    plate_root: Path,
) -> Path:
    resolver = CompilationPathResolver(
        plate_scope=CompilationPlateScope.from_path(plate_root),
        filemanager=FilesystemFileManager(),
        backend="disk",
    )
    pattern = compile_function_pattern(
        step.func,
        {},
        {},
        path_resolver=resolver,
    )
    return next(pattern.iter_invocations()).kwargs_dict["macro_path"]


def test_generated_relative_path_compiles_against_explicit_plate_not_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plate_a = tmp_path / "plate-a"
    plate_b = tmp_path / "plate-b"
    for plate_root in (plate_a, plate_b):
        template = plate_root / "templates" / "template.tif"
        template.parent.mkdir(parents=True)
        template.touch()

    authored_path = Path("templates/template.tif")
    source = FunctionStepTransportAuthority.source_from_pipeline(
        [
            FunctionStep(
                func=(
                    run_imagej_macro,
                    {"macro_path": authored_path},
                ),
                name="Relative template",
            )
        ]
    )
    namespace: dict[str, object] = {}
    exec(source, namespace)
    step = FunctionStepTransportAuthority.pipeline_steps_from_namespace(namespace)[0]

    cwd_a = tmp_path / "unrelated-a"
    cwd_b = tmp_path / "unrelated-b"
    cwd_a.mkdir()
    cwd_b.mkdir()
    monkeypatch.chdir(cwd_a)
    first = _compiled_template_path(step, plate_root=plate_a)
    monkeypatch.chdir(cwd_b)
    repeated = _compiled_template_path(step, plate_root=plate_a)
    second_plate = _compiled_template_path(step, plate_root=plate_b)

    assert first == repeated == plate_a / authored_path
    assert second_plate == plate_b / authored_path
    assert step.func[1]["macro_path"] == authored_path
    generated_paths = tuple(
        node.args[0].value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Path"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    )
    assert str(authored_path) in generated_paths
