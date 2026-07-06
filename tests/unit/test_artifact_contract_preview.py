from __future__ import annotations

from dataclasses import fields

from openhcs.core.artifact_contract_preview import (
    ArtifactContractPreview,
    ArtifactPreviewDirection,
    ArtifactPreviewOrigin,
    SourceBindingRuntimeContractGuard,
)
from openhcs.core.artifacts import (
    ArtifactSidecarRole,
    ArtifactSpec,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
)
from openhcs.core.module_artifact_contract import (
    DeclaredArtifactOutputPartition,
    ModuleArtifactContract,
    ModuleArtifactContractItem,
    RecordedArtifactOutputPartition,
    RuntimeArtifactInputPartition,
    SourceArtifactInputPartition,
    module_artifact_contract,
)
from openhcs.core.source_bindings import (
    NamedSourceBinding,
    StepSourceBindingsConfig,
)
from openhcs.pyqt_gui.widgets.artifact_contract_preview import (
    ArtifactContractPreviewProjection,
)


def test_module_artifact_contract_stores_partitioned_items() -> None:
    contract = ModuleArtifactContract(
        module_name="Measure",
        items=(
            *ModuleArtifactContract.items_for_partition(
                SourceArtifactInputPartition,
                (ArtifactSpec.input("DNA", ImageArtifactType),),
            ),
            *ModuleArtifactContract.items_for_partition(
                RuntimeArtifactInputPartition,
                (ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),),
            ),
            *ModuleArtifactContract.items_for_partition(
                RecordedArtifactOutputPartition,
                (ArtifactSpec.output("Measurements", MeasurementsArtifactType),),
            ),
            *ModuleArtifactContract.items_for_partition(
                DeclaredArtifactOutputPartition,
                (ArtifactSpec.output("Measurements", MeasurementsArtifactType),),
            ),
        ),
    )

    assert [field.name for field in fields(ModuleArtifactContract)] == [
        "module_name",
        "items",
        "required_variable_components",
    ]
    assert [(item.partition_type, item.spec.name) for item in contract.items] == [
        (SourceArtifactInputPartition, "DNA"),
        (RuntimeArtifactInputPartition, "Nuclei"),
        (RecordedArtifactOutputPartition, "Measurements"),
        (DeclaredArtifactOutputPartition, "Measurements"),
    ]
    assert all(isinstance(item, ModuleArtifactContractItem) for item in contract.items)
    assert [spec.name for spec in contract.inputs] == ["DNA"]
    assert [spec.name for spec in contract.runtime_artifact_inputs] == ["Nuclei"]
    assert [spec.name for spec in contract.outputs] == ["Measurements"]
    assert [spec.name for spec in contract.declared_outputs] == ["Measurements"]


def test_artifact_contract_preview_projects_inputs_outputs_and_sidecars() -> None:
    contract = ModuleArtifactContract(
        module_name="Crop",
        items=(
            *ModuleArtifactContract.items_for_partition(
                SourceArtifactInputPartition,
                (
                    ArtifactSpec.input("OrigBlue", ImageArtifactType),
                    ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),
                ),
            ),
            *ModuleArtifactContract.items_for_partition(
                RuntimeArtifactInputPartition,
                (ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),),
            ),
            *ModuleArtifactContract.items_for_partition(
                RecordedArtifactOutputPartition,
                (
                    ArtifactSpec.output("CropBlue", ImageArtifactType),
                    ArtifactSpec.output(
                        "CropBlue__crop_mask",
                        ImageArtifactType,
                        sidecar_role=ArtifactSidecarRole.CROP_MASK,
                    ),
                ),
            ),
            *ModuleArtifactContract.items_for_partition(
                DeclaredArtifactOutputPartition,
                (
                    ArtifactSpec.output("CropBlue", ImageArtifactType),
                    ArtifactSpec.output(
                        "CropBlue__crop_mask",
                        ImageArtifactType,
                        sidecar_role=ArtifactSidecarRole.CROP_MASK,
                    ),
                ),
            ),
        ),
    )

    preview = ArtifactContractPreview.from_module_contract(contract)

    assert preview.module_name == "Crop"
    assert [row.name for row in preview.inputs] == ["OrigBlue", "Nuclei"]
    assert preview.inputs[0].origin is ArtifactPreviewOrigin.SOURCE_BINDING
    assert preview.inputs[1].origin is ArtifactPreviewOrigin.RUNTIME_ARTIFACT
    assert all(
        row.direction is ArtifactPreviewDirection.OUTPUT for row in preview.outputs
    )
    assert preview.outputs[1].sidecar_role is ArtifactSidecarRole.CROP_MASK


def test_artifact_contract_projection_reads_callable_contracts_from_function_spec() -> (
    None
):
    contract = ModuleArtifactContract(
        module_name="Measure",
        items=(
            *ModuleArtifactContract.items_for_partition(
                SourceArtifactInputPartition,
                (ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),),
            ),
            *ModuleArtifactContract.items_for_partition(
                RuntimeArtifactInputPartition,
                (ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),),
            ),
            *ModuleArtifactContract.items_for_partition(
                RecordedArtifactOutputPartition,
                (ArtifactSpec.output("Measurements", MeasurementsArtifactType),),
            ),
            *ModuleArtifactContract.items_for_partition(
                DeclaredArtifactOutputPartition,
                (ArtifactSpec.output("Measurements", MeasurementsArtifactType),),
            ),
        ),
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
        items=(
            *ModuleArtifactContract.items_for_partition(
                RecordedArtifactOutputPartition,
                (ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType),),
            ),
            *ModuleArtifactContract.items_for_partition(
                DeclaredArtifactOutputPartition,
                (ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType),),
            ),
        ),
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
        items=(
            *ModuleArtifactContract.items_for_partition(
                SourceArtifactInputPartition,
                (ArtifactSpec.input("OrigBlue", ImageArtifactType),),
            ),
        ),
    )
    source_bindings = StepSourceBindingsConfig(
        bindings=(NamedSourceBinding(alias="OrigGreen"),),
    )

    alignment = SourceBindingRuntimeContractGuard(
        contract,
        source_bindings,
    ).alignment()

    assert not alignment.ok
    assert alignment.missing == (("OrigBlue", ImageArtifactType),)
    assert alignment.unexpected == (("OrigGreen", ImageArtifactType),)
    assert "missing: image:OrigBlue" in alignment.message
    assert "unexpected: image:OrigGreen" in alignment.message


def test_artifact_contract_projection_message_surfaces_source_binding_drift() -> None:
    contract = ModuleArtifactContract(
        module_name="Crop",
        items=(
            *ModuleArtifactContract.items_for_partition(
                SourceArtifactInputPartition,
                (ArtifactSpec.input("OrigBlue", ImageArtifactType),),
            ),
        ),
    )

    @module_artifact_contract(contract)
    def crop():
        return None

    projection = ArtifactContractPreviewProjection(
        crop,
        source_bindings=StepSourceBindingsConfig(
            bindings=(NamedSourceBinding(alias="OrigGreen"),),
        ),
    )

    assert "Source-binding drift detected" in projection.message()
    assert "missing: image:OrigBlue" in projection.alignment_for("Crop").message
