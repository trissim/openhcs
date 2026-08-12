from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openhcs.processing.backends.analysis import multi_template_matching
from openhcs.processing.backends.analysis.multi_template_matching import (
    OpenCVTemplateMatchMethod,
    TemplateMatchResult,
    multi_template_crop,
    multi_template_crop_reference_channel,
    multi_template_crop_subset,
)
from openhcs.core.function_contract_metadata import FunctionContractAttribute
from openhcs.processing.materialization import materialization_outputs


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


def test_template_match_artifact_materializes_every_match_column() -> None:
    [artifact_spec] = vars(multi_template_crop_reference_channel)[
        FunctionContractAttribute.artifact_outputs
    ]
    results = [
        TemplateMatchResult(
            slice_index=2,
            matches=[
                ("first", (1, 2, 3, 4), 0.95),
                ("second", (5, 6, 7, 8), 0.85),
            ],
            best_match=("first", (1, 2, 3, 4), 0.95),
            crop_bbox=(1, 2, 3, 4),
            match_score=0.95,
            num_matches=2,
            best_rotation_angle=0.0,
        )
    ]

    [output] = materialization_outputs(
        artifact_spec.materialization,
        results,
        "/analysis/match_results.pkl",
        SimpleNamespace(),
    )

    assert output.path == "/analysis/match_results_mtm_matches.csv"
    lines = output.content.splitlines()
    assert lines[0].split(",") == [
        "slice_index",
        "match_id",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "confidence_score",
        "template_name",
        "is_best_match",
        "was_cropped",
    ]
    assert len(lines) == 3
    assert "slice_2_match_0" in lines[1]
    assert "slice_2_match_1" in lines[2]
