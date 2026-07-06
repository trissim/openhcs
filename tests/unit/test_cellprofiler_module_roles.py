"""CellProfiler module-role policy tests."""

from openhcs.core.artifacts import ImageArtifactType
from openhcs.interop import cellprofiler
from openhcs.interop.cellprofiler.module_roles import (
    ArtifactSpecKey,
    cellprofiler_infrastructure_retained_artifacts,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock


def test_saveimages_infrastructure_policy_declares_retained_image_artifacts() -> None:
    module = ModuleBlock(
        name="SaveImages",
        module_num=5,
        settings={
            "Select the image to save": "OverlayImage, CorrectedImage",
        },
    )

    assert cellprofiler_infrastructure_retained_artifacts(
        module,
        contracts_by_module_num={},
    ) == frozenset(
        (
            ArtifactSpecKey(ImageArtifactType, "OverlayImage"),
            ArtifactSpecKey(ImageArtifactType, "CorrectedImage"),
        )
    )


def test_default_infrastructure_policy_declares_no_retained_artifacts() -> None:
    module = ModuleBlock(name="ExportToSpreadsheet", module_num=6)

    assert (
        cellprofiler_infrastructure_retained_artifacts(
            module,
            contracts_by_module_num={},
        )
        == frozenset()
    )


def test_cellprofiler_namespace_exposes_infrastructure_retention_query() -> None:
    assert (
        cellprofiler.cellprofiler_infrastructure_retained_artifacts
        is cellprofiler_infrastructure_retained_artifacts
    )
