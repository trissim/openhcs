import pytest

from openhcs.constants import GroupBy, VariableComponents
from openhcs.core.function_patterns import compile_function_pattern
from openhcs.core.pipeline.funcstep_contract_validator import (
    FuncStepContractValidator,
)


def _function(name="function"):
    def func(image):
        return image

    func.__name__ = name
    func.__module__ = "builtins"
    return func


def _compiled_pattern(func):
    return compile_function_pattern(func, {}, {})


def test_validate_compiled_function_pattern_uses_callable_contract_memory_types():
    func = _function("valid")
    func.input_memory_type = "numpy"
    func.output_memory_type = "cupy"

    assert FuncStepContractValidator.validate_compiled_function_pattern(
        _compiled_pattern(func),
        "step",
    ) == ("numpy", "cupy")


def test_validate_compiled_function_pattern_rejects_missing_contract_memory_types():
    func = _function("missing")

    with pytest.raises(ValueError, match="needs memory type decorator"):
        FuncStepContractValidator.validate_compiled_function_pattern(
            _compiled_pattern(func),
            "step",
        )


def test_validate_compiled_function_pattern_reports_invocation_identity():
    func = _function("invalid")
    func.input_memory_type = "bogus"
    func.output_memory_type = "numpy"

    with pytest.raises(ValueError, match=r"invalid\[default:0\]"):
        FuncStepContractValidator.validate_compiled_function_pattern(
            _compiled_pattern(func),
            "step",
        )


def test_normalized_group_by_resolves_variable_component_conflict_to_none():
    assert (
        FuncStepContractValidator.normalized_group_by(
            GroupBy.CHANNEL,
            (VariableComponents.CHANNEL,),
            "step",
        )
        is GroupBy.NONE
    )
