from __future__ import annotations

from openhcs.core.artifact_contract_preview import (
    ArtifactContractPreview,
    ArtifactPreviewDirection,
    ArtifactPreviewOrigin,
    SourceBindingRuntimeContractGuard,
)
from openhcs.core.artifacts import ArtifactKind, ArtifactSidecarRole, ArtifactSpec
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.core.module_artifact_contract import module_artifact_contract
from openhcs.core.source_bindings import (
    NamedSourceBinding,
    StepSourceBindingsConfig,
)
from openhcs.pyqt_gui.widgets.artifact_contract_preview import (
    ArtifactContractPreviewProjection,
)


def test_artifact_contract_preview_projects_inputs_outputs_and_sidecars() -> None:
    contract = ModuleArtifactContract(
        module_name="Crop",
        inputs=(
            ArtifactSpec("OrigBlue", ArtifactKind.IMAGE),
            ArtifactSpec("Nuclei", ArtifactKind.OBJECT_LABELS),
        ),
        runtime_artifact_inputs=(ArtifactSpec("Nuclei", ArtifactKind.OBJECT_LABELS),),
        outputs=(
            ArtifactSpec("CropBlue", ArtifactKind.IMAGE),
            ArtifactSpec(
                "CropBlue__crop_mask",
                ArtifactKind.IMAGE,
                sidecar_role=ArtifactSidecarRole.CROP_MASK,
            ),
        ),
    )

    preview = ArtifactContractPreview.from_module_contract(contract)

    assert preview.module_name == "Crop"
    assert [row.name for row in preview.inputs] == ["OrigBlue", "Nuclei"]
    assert preview.inputs[0].origin is ArtifactPreviewOrigin.SOURCE_BINDING
    assert preview.inputs[1].origin is ArtifactPreviewOrigin.RUNTIME_ARTIFACT
    assert all(row.direction is ArtifactPreviewDirection.OUTPUT for row in preview.outputs)
    assert preview.outputs[1].sidecar_role is ArtifactSidecarRole.CROP_MASK


def test_artifact_contract_projection_reads_callable_contracts_from_function_spec() -> None:
    contract = ModuleArtifactContract(
        module_name="Measure",
        inputs=(ArtifactSpec("Nuclei", ArtifactKind.OBJECT_LABELS),),
        runtime_artifact_inputs=(ArtifactSpec("Nuclei", ArtifactKind.OBJECT_LABELS),),
        outputs=(ArtifactSpec("Measurements", ArtifactKind.MEASUREMENTS),),
    )

    @module_artifact_contract(contract)
    def measure_objects():
        return None

    previews = ArtifactContractPreviewProjection(
        [(measure_objects, {"scale": 3})]
    ).previews()

    assert len(previews) == 1
    assert previews[0].module_name == "Measure"
    assert [row.name for row in previews[0].rows] == ["Nuclei", "Measurements"]


def test_artifact_contract_projection_reads_dict_function_patterns() -> None:
    contract = ModuleArtifactContract(
        module_name="Segment",
        outputs=(ArtifactSpec("Nuclei", ArtifactKind.OBJECT_LABELS),),
    )

    @module_artifact_contract(contract)
    def segment():
        return None

    previews = ArtifactContractPreviewProjection(
        {"default": [(segment, {"enabled": True})]}
    ).previews()

    assert len(previews) == 1
    assert previews[0].module_name == "Segment"


def test_source_binding_contract_alignment_reports_drift() -> None:
    contract = ModuleArtifactContract(
        module_name="Crop",
        inputs=(ArtifactSpec("OrigBlue", ArtifactKind.IMAGE),),
    )
    source_bindings = StepSourceBindingsConfig(bindings=(NamedSourceBinding(alias="OrigGreen"),),
    )

    alignment = SourceBindingRuntimeContractGuard(
        contract,
        source_bindings,
    ).alignment()

    assert not alignment.ok
    assert alignment.missing == (("OrigBlue", ArtifactKind.IMAGE),)
    assert alignment.unexpected == (("OrigGreen", ArtifactKind.IMAGE),)
    assert "missing: image:OrigBlue" in alignment.message
    assert "unexpected: image:OrigGreen" in alignment.message


def test_artifact_contract_projection_message_surfaces_source_binding_drift() -> None:
    contract = ModuleArtifactContract(
        module_name="Crop",
        inputs=(ArtifactSpec("OrigBlue", ArtifactKind.IMAGE),),
    )

    @module_artifact_contract(contract)
    def crop():
        return None

    projection = ArtifactContractPreviewProjection(
        crop,
        source_bindings=StepSourceBindingsConfig(bindings=(NamedSourceBinding(alias="OrigGreen"),),
        ),
    )

    assert "Source-binding drift detected" in projection.message()
    assert "missing: image:OrigBlue" in projection.alignment_for("Crop").message
