from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import openhcs.serialization.pycodify_formatters  # noqa: F401
from openhcs.core.vfs_protocol import PlateInputFile
from openhcs.serialization.source_path_factoring import OpenHCSPythonSourceDocument
from pycodify import Assignment, CodeBlock


def declared_template_callable(
    image: object,
    *,
    template_path: PlateInputFile,
) -> object:
    return image


@dataclass(frozen=True, slots=True)
class PathBearingConfig:
    actual_path: Path
    path_lookalike: str


def test_document_factors_deepest_common_root_and_repeated_values() -> None:
    plate_a = Path("/media/alice/T7/screen/plate_A")
    plate_b = Path("/media/alice/T7/screen/plate_B")
    template = Path("/media/alice/T7/screen/templates/fiducial.png")
    source = OpenHCSPythonSourceDocument(
        CodeBlock.from_items(
            (
                Assignment("plate_paths", [plate_a, plate_b]),
                Assignment("pipeline_data", {plate_a: template, plate_b: None}),
            )
        )
    ).render()

    assert "path_root = Path('/media/alice/T7/screen')" in source
    assert "path_1 = path_root / 'plate_A'" in source
    assert "path_2 = path_root / 'plate_B'" in source
    assert "path_root / 'templates' / 'fiducial.png'" in source
    assert source.count("/media/alice/T7/screen") == 1

    namespace: dict[str, object] = {}
    exec(source, namespace)
    assert namespace["plate_paths"] == [plate_a, plate_b]
    assert namespace["pipeline_data"] == {plate_a: template, plate_b: None}


def test_document_does_not_factor_relative_or_string_lookalike_paths() -> None:
    source = OpenHCSPythonSourceDocument(
        Assignment(
            "values",
            [
                Path("templates/fiducial.png"),
                "/media/alice/T7/screen/plate_A",
                "/media/alice/T7/screen/plate_A",
            ],
        )
    ).render()

    assert "path_root" not in source
    assert "path_1" not in source
    assert "Path('templates/fiducial.png')" in source
    assert source.count("'/media/alice/T7/screen/plate_A'") == 2


def test_document_keeps_unrelated_absolute_trees_separate() -> None:
    source = OpenHCSPythonSourceDocument(
        Assignment(
            "values",
            [Path("/data/a/one"), Path("/data/a/two"), Path("/mnt/b/three")],
        )
    ).render()

    assert "path_root = Path('/data/a')" in source
    assert "path_root_2" not in source
    assert "Path('/mnt/b/three')" in source


def test_document_extracts_one_repeated_absolute_path_without_root_binding() -> None:
    source = OpenHCSPythonSourceDocument(
        Assignment(
            "values",
            [Path("/opt/models/model.bin"), Path("/opt/models/model.bin")],
        )
    ).render()

    assert "path_root" not in source
    assert "path_1 = Path('/opt/models/model.bin')" in source
    assert source.count("/opt/models/model.bin") == 1


def test_document_uses_first_occurrence_order_for_multiple_coherent_roots() -> None:
    source = OpenHCSPythonSourceDocument(
        Assignment(
            "values",
            [
                Path("/mnt/zeta/one"),
                Path("/mnt/zeta/two"),
                Path("/data/alpha/one"),
                Path("/data/alpha/two"),
            ],
        )
    ).render()

    assert source.index("path_root = Path('/mnt/zeta')") < source.index(
        "path_root_2 = Path('/data/alpha')"
    )


def test_document_collects_dataclass_path_values_but_not_string_lookalikes() -> None:
    absolute = Path("/data/screen/config/model.bin")
    source = OpenHCSPythonSourceDocument(
        Assignment(
            "configs",
            [
                PathBearingConfig(absolute, str(absolute)),
                PathBearingConfig(absolute, str(absolute)),
            ],
        )
    ).render()

    assert source.count("/data/screen/config/model.bin") == 3
    assert "path_1 = Path('/data/screen/config/model.bin')" in source


def test_declared_string_callable_path_is_collected_structurally() -> None:
    source = OpenHCSPythonSourceDocument(
        Assignment(
            "pattern",
            (
                declared_template_callable,
                {"template_path": "/data/screen/templates/fiducial.png"},
            ),
        )
    ).render()

    assert "template_path" in source
    assert "Path('/data/screen/templates/fiducial.png')" in source


def test_pipeline_declarations_share_their_deepest_common_absolute_root() -> None:
    source = OpenHCSPythonSourceDocument(
        Assignment(
            "pattern",
            [
                (
                    declared_template_callable,
                    {"template_path": "/media/alice/T7/screen/a/template.tif"},
                ),
                (
                    declared_template_callable,
                    {"template_path": "/media/alice/T7/screen/b/template.tif"},
                ),
            ],
        )
    ).render()

    assert "path_root = Path('/media/alice/T7/screen')" in source
    assert "path_root / 'a' / 'template.tif'" in source
    assert "path_root / 'b' / 'template.tif'" in source
    assert source.count("/media/alice/T7/screen") == 1
