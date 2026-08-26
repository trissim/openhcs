"""Canonical configuration Python document contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import pytest

from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.core.config_document import (
    ConfigDocumentAuthority,
    ConfigDocumentField,
)


@dataclass(frozen=True)
class _ProjectionRoot:
    target: timedelta = field(default_factory=lambda: timedelta(seconds=1))


def test_config_document_field_owns_exact_public_name() -> None:
    assert tuple(field.value for field in ConfigDocumentField) == ("config",)


@pytest.mark.parametrize(
    "config",
    (
        GlobalPipelineConfig(num_workers=3),
        PipelineConfig(),
    ),
)
def test_config_document_round_trip(config: object) -> None:
    config_type = type(config)

    source = ConfigDocumentAuthority.render(
        config,
        expected_config_type=config_type,
    )
    restored = ConfigDocumentAuthority.from_source(
        source,
        expected_config_type=config_type,
    )

    assert source.startswith("# OpenHCS configuration")
    assert "config = " in source
    assert restored == config


def test_config_document_from_namespace_requires_exact_field() -> None:
    with pytest.raises(ValueError, match="'config'"):
        ConfigDocumentAuthority.from_namespace(
            {"pipeline_config": PipelineConfig()},
            expected_config_type=PipelineConfig,
        )


def test_config_document_rejects_wrong_expected_type() -> None:
    with pytest.raises(TypeError, match="PipelineConfig"):
        ConfigDocumentAuthority.from_namespace(
            {"config": GlobalPipelineConfig()},
            expected_config_type=PipelineConfig,
        )

    with pytest.raises(TypeError, match="PipelineConfig"):
        ConfigDocumentAuthority.render(
            GlobalPipelineConfig(),
            expected_config_type=PipelineConfig,
        )


def test_config_document_projects_one_explicit_dataclass_field() -> None:
    projected = ConfigDocumentAuthority.project_dataclass_field_from_source(
        "from datetime import timedelta\nconfig = Root(target=timedelta(seconds=2))",
        expected_config_type=_ProjectionRoot,
        expected_field_type=timedelta,
    )

    assert projected == timedelta(seconds=2)


def test_config_document_projection_uses_the_declared_field_default() -> None:
    projected = ConfigDocumentAuthority.project_dataclass_field_from_source(
        "config = Root()",
        expected_config_type=_ProjectionRoot,
        expected_field_type=timedelta,
    )

    assert projected == timedelta(seconds=1)


def test_config_document_projection_rejects_the_wrong_field_type() -> None:
    with pytest.raises(TypeError, match="target.*timedelta.*list"):
        ConfigDocumentAuthority.project_dataclass_field_from_source(
            'config = Root(target=["one"])',
            expected_config_type=_ProjectionRoot,
            expected_field_type=timedelta,
        )
