import ast
from dataclasses import FrozenInstanceError

import pytest

from openhcs.validation.ast_validator import (
    ASTValidator,
    BACKEND_PARAM,
    DEFAULT_VALIDATION_KINDS,
    PATH_TYPE,
    PathTypeValidator,
    ValidationKind,
    ValidationViolation,
    VFS_BOUNDARY,
    run_ast_validators,
    validate_file,
)


def test_validation_violation_preserves_string_kind_and_is_frozen():
    violation = ValidationViolation(
        file_path="example.py",
        line_number=12,
        violation_type=ValidationKind.PATH_TYPE,
        message="bad path",
    )

    assert violation.violation_type == PATH_TYPE
    assert str(violation) == "example.py:12 - path_type: bad path"
    with pytest.raises(FrozenInstanceError):
        violation.message = "changed"


def test_validator_registry_returns_stable_default_order():
    validator_classes = ASTValidator.validators_for()

    assert tuple(cls.validation_kind for cls in validator_classes) == DEFAULT_VALIDATION_KINDS
    assert validator_classes[0] is PathTypeValidator
    assert ASTValidator.validators_for([PATH_TYPE]) == (PathTypeValidator,)


def test_run_ast_validators_supports_kind_subset():
    tree = ast.parse(
        """
from os.path import join

@validate_path_types()
def load_image(path: int):
    return path
"""
    )

    path_violations = run_ast_validators(tree, "openhcs/example.py", [ValidationKind.PATH_TYPE])
    vfs_violations = run_ast_validators(tree, "openhcs/example.py", [ValidationKind.VFS_BOUNDARY])

    assert [violation.violation_type for violation in path_violations] == [PATH_TYPE]
    assert [violation.violation_type for violation in vfs_violations] == [VFS_BOUNDARY]


def test_validate_file_uses_registered_validators(tmp_path):
    source_path = tmp_path / "sample.py"
    source_path.write_text(
        """
FileManager.load("path")

@validate_path_types()
def load_image(path: int):
    return path
""",
        encoding="utf-8",
    )

    violations = validate_file(str(source_path), kinds=[ValidationKind.PATH_TYPE, ValidationKind.BACKEND_PARAM])

    assert [violation.violation_type for violation in violations] == [PATH_TYPE, BACKEND_PARAM]


def test_validate_file_keeps_syntax_error_violation(tmp_path):
    source_path = tmp_path / "broken.py"
    source_path.write_text("def broken(:\n", encoding="utf-8")

    violations = validate_file(str(source_path))

    assert len(violations) == 1
    assert violations[0].violation_type == "syntax_error"
