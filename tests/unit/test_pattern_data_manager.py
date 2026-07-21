import pytest

from openhcs.ui.shared.pattern_data_manager import PatternDataManager


def _identity(image):
    return image


def test_extract_func_and_kwargs_accepts_exact_two_member_leaf() -> None:
    kwargs = {"sigma": 2}

    func, extracted_kwargs = PatternDataManager.extract_func_and_kwargs(
        (_identity, kwargs)
    )

    assert func is _identity
    assert extracted_kwargs is kwargs


def test_extract_func_and_kwargs_rejects_three_member_leaf() -> None:
    with pytest.raises(TypeError, match="exactly two"):
        PatternDataManager.extract_func_and_kwargs((_identity, {}, object()))
