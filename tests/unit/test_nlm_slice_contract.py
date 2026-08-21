"""Regression tests for decorator-owned NLM slice execution semantics."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from openhcs.processing.backends.enhance import jax_nlm_processor
from openhcs.processing.backends.enhance import torch_nlm_processor


@pytest.mark.parametrize(
    ("func", "expected_default"),
    (
        (jax_nlm_processor.non_local_means_denoise_jax, False),
        (torch_nlm_processor.non_local_means_denoise_torch, True),
    ),
)
def test_nlm_slice_control_has_one_declaration(func, expected_default) -> None:
    """The decorator exposes the control and the backend does not redeclare it."""
    decorated_parameter = inspect.signature(func).parameters["slice_by_slice"]
    backend_parameters = inspect.signature(inspect.unwrap(func)).parameters

    assert decorated_parameter.default is expected_default
    assert "slice_by_slice" not in backend_parameters


def test_torch_nlm_dimensionality_members_execute_their_own_leaf(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        torch_nlm_processor,
        "nlm2d",
        lambda image, **kwargs: calls.append(("2d", image, kwargs)) or "two",
    )
    monkeypatch.setattr(
        torch_nlm_processor,
        "nlm3d",
        lambda image, **kwargs: calls.append(("3d", image, kwargs)) or "three",
    )

    assert (
        torch_nlm_processor.TorchNlmInputDimensionality.IMAGE_2D.denoise(
            "plane",
            kernel_size=3,
        )
        == "two"
    )
    assert (
        torch_nlm_processor.TorchNlmInputDimensionality.VOLUME_3D.denoise(
            "volume",
            kernel_size=5,
        )
        == "three"
    )
    assert calls == [
        ("2d", "plane", {"kernel_size": 3}),
        ("3d", "volume", {"kernel_size": 5}),
    ]


def test_jax_nlm_dimensionality_members_own_estimation_and_rejection() -> None:
    volume = np.arange(24).reshape(2, 3, 4)

    assert np.array_equal(
        jax_nlm_processor.JaxNlmInputDimensionality.IMAGE_2D.estimation_slice(
            volume[0]
        ),
        volume[0],
    )
    assert np.array_equal(
        jax_nlm_processor.JaxNlmInputDimensionality.VOLUME_3D.estimation_slice(volume),
        volume[0],
    )
    with pytest.raises(NotImplementedError, match="slice_by_slice=True"):
        jax_nlm_processor.JaxNlmInputDimensionality.VOLUME_3D.denoise(
            volume,
            search_window_radius=7,
            filter_radius=1,
            h=0.1,
            sigma=0.1,
        )
