from __future__ import annotations

import pytest

from openhcs.core.config import (
    FijiDisplayConfig,
    FijiStreamingConfig,
    GlobalPipelineConfig,
    LazyFijiDisplayConfig,
    LazyFijiStreamingConfig,
    LazyNapariDisplayConfig,
    LazyNapariStreamingConfig,
    NapariDisplayConfig,
    NapariStreamingConfig,
    PlateMetadataConfig,
)
from python_introspect import AnnotationValidationError


@pytest.mark.parametrize(
    ("config_type", "values"),
    (
        (GlobalPipelineConfig, {"num_workers": 0}),
        (PlateMetadataConfig, {"z_step": 0}),
        (NapariDisplayConfig, {"colormap": " "}),
        (FijiDisplayConfig, {"lut": " "}),
        (NapariStreamingConfig, {"port": 0}),
        (FijiStreamingConfig, {"port": 65_536}),
        (NapariStreamingConfig, {"host": " "}),
        (LazyNapariDisplayConfig, {"colormap": " "}),
        (LazyFijiDisplayConfig, {"lut": " "}),
        (LazyNapariStreamingConfig, {"port": 0}),
        (LazyFijiStreamingConfig, {"port": 65_536}),
        (LazyNapariStreamingConfig, {"host": " "}),
    ),
)
def test_config_type_annotations_are_enforced_after_construction(
    config_type,
    values,
) -> None:
    with pytest.raises(AnnotationValidationError):
        config_type(**values)
