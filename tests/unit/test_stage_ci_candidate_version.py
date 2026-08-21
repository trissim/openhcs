"""Tests for the native-installer candidate version boundary."""

from pathlib import Path

import pytest
from packaging.version import Version

from scripts.stage_ci_candidate_version import stage_ci_candidate_version


def test_stage_ci_candidate_version_rewrites_the_literal_declaration(
    tmp_path: Path,
) -> None:
    source = tmp_path / "__init__.py"
    source.write_text(
        '"""Package."""\n\n__version__ = "1.2.3"\nVALUE = 4\n',
        encoding="utf-8",
    )

    candidate = stage_ci_candidate_version(source, "12345")

    assert candidate == Version("1.2.3.dev12345")
    assert source.read_text(encoding="utf-8") == (
        '"""Package."""\n\n__version__ = "1.2.3.dev12345"\nVALUE = 4\n'
    )


@pytest.mark.parametrize("run_id", ["", "abc", "12-3"])
def test_stage_ci_candidate_version_rejects_non_numeric_run_ids(
    tmp_path: Path,
    run_id: str,
) -> None:
    source = tmp_path / "__init__.py"
    source.write_text('__version__ = "1.2.3"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="decimal digits"):
        stage_ci_candidate_version(source, run_id)


def test_stage_ci_candidate_version_requires_one_literal_owner(tmp_path: Path) -> None:
    source = tmp_path / "__init__.py"
    source.write_text(
        'BASE = "1.2.3"\n__version__ = BASE\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one literal"):
        stage_ci_candidate_version(source, "12345")
