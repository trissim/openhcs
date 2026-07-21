"""Exact deletion gates for persistent-only artifact materialization."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from polystore.streaming.identity import StreamProducerIdentity

from openhcs.core.steps.function_artifact_materialization import (
    ArtifactMaterializationBackendPlan,
    ArtifactStreamSourceMetadataAuthority,
)
from openhcs.processing.materialization.core import (
    MaterializationSpec,
    RawBackendKwargs,
)
from openhcs.processing.materialization.options import ImageFileOptions


class _PersistentBackend:
    def contextual_save_kwargs(self, *, images_dir: str) -> dict[str, object]:
        del images_dir
        return {}

    def supports_file_path(self, path: str) -> bool:
        del path
        return False


class _FileManager:
    def __init__(self) -> None:
        self.backend = _PersistentBackend()

    def _get_backend(self, backend: str) -> _PersistentBackend:
        del backend
        return self.backend


@pytest.mark.parametrize(
    ("payload", "streaming_viewer_surfaces"),
    (
        (np.empty((0, 4, 4), dtype=np.uint16), {}),
        (np.ones((60, 4, 4), dtype=np.uint16), {"viewer": object()}),
    ),
)
def test_persistent_only_backend_kwargs_skip_stream_payload_projection(
    monkeypatch: pytest.MonkeyPatch,
    payload: np.ndarray,
    streaming_viewer_surfaces: dict[str, object],
) -> None:
    def unexpected_stream_metadata(**_kwargs: object) -> None:
        raise AssertionError("persistent-only materialization projected stream metadata")

    monkeypatch.setattr(
        ArtifactStreamSourceMetadataAuthority,
        "metadata_items",
        unexpected_stream_metadata,
    )
    backend_plan = ArtifactMaterializationBackendPlan(
        persistent_backend_kwargs={"disk": RawBackendKwargs()},
        streaming_viewer_surfaces=streaming_viewer_surfaces,
    )

    kwargs = backend_plan.backend_kwargs(
        materialization_spec=MaterializationSpec(
            ImageFileOptions(filename_suffix=".tif")
        ),
        data=payload,
        fallback_source_identity=None,
        producer_identity=StreamProducerIdentity(
            origin="pipeline",
            output_kind="artifact",
            output_key="SavedImage",
            projection_key="SavedImage",
        ),
        context=object(),
        filemanager=_FileManager(),
        images_dir="/images",
        stream_output_paths=("/images/SavedImage.tif",),
    )

    assert tuple(kwargs) == ("disk",)
    assert dict(kwargs["disk"]) == {}


def test_backend_kwargs_no_longer_accepts_eager_stream_metadata() -> None:
    assert "source_metadata_items" not in inspect.signature(
        ArtifactMaterializationBackendPlan.backend_kwargs
    ).parameters
