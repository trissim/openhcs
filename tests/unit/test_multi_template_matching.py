from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from openhcs.processing.backends.analysis import multi_template_matching
from openhcs.processing.backends.analysis.multi_template_matching import (
    OpenCVTemplateMatchMethod,
    multi_template_crop,
    multi_template_crop_reference_channel,
    multi_template_crop_subset,
)


@pytest.mark.parametrize(
    "function",
    (
        multi_template_crop,
        multi_template_crop_reference_channel,
        multi_template_crop_subset,
    ),
)
def test_public_template_matching_routes_selected_method_to_mtm(
    monkeypatch: pytest.MonkeyPatch,
    function,
) -> None:
    received_methods: list[int] = []

    monkeypatch.setattr(
        multi_template_matching.cv2,
        "imread",
        lambda *_args, **_kwargs: np.ones((2, 2), dtype=np.uint8),
    )

    def match_templates(*_args, **kwargs):
        received_methods.append(kwargs["method"])
        return []

    monkeypatch.setattr(
        multi_template_matching.MTM,
        "matchTemplates",
        match_templates,
    )

    inspect.unwrap(function)(
        np.ones((1, 4, 4), dtype=np.uint8),
        Path("template.tif"),
        method=OpenCVTemplateMatchMethod.SQDIFF,
        crop_enabled=False,
    )

    assert received_methods == [int(OpenCVTemplateMatchMethod.SQDIFF)]
