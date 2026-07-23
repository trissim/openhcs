"""Focused runtime binding-identity regressions for CellProfiler modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    InputStackBroadcastSourceRelation,
    ObjectLabelsArtifactType,
)
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    FunctionInvocationKey,
    normalize_function_pattern,
)
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.runtime_image_values import image_payload_data
from openhcs.core.runtime_object_labels import ObjectLabelSet, ObjectLabelVariantData
from openhcs.interop.cellprofiler.parser import CPPipeParser
from openhcs.interop.cellprofiler.runtime.invocation import CellProfilerImageRequest
from openhcs.interop.cellprofiler.settings_binder import SettingsBinder
from openhcs.processing.backends.cellprofiler.crop import CropModule, crop
from openhcs.processing.backends.cellprofiler.morphology import MaskObjectsModule
from openhcs.processing.backends.cellprofiler.neighbors import (
    MeasureObjectNeighborsModule,
)
from openhcs.processing.backends.cellprofiler.secondary import (
    IdentifyTertiaryObjectsModule,
)
from openhcs.processing.backends.cellprofiler.tracking import TrackObjectsModule


def test_official_image_crop_settings_reconstruct_public_contract(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "official-image-crop.cppipe"
    cppipe_path.write_text(
        """CellProfiler Pipeline: https://cellprofiler.org
Crop:[module_num:10|svn_version:'Unknown'|variable_revision_number:3|show_window:True|notes:['Cut out (crop) the region within the eroded well edge and continue analysis on this cropped image.']|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    Select the input image:Worms
    Name the output image:WormsCropped
    Select the cropping shape:Image
    Select the cropping method:Coordinates
    Apply which cycle's cropping pattern?:Every
    Left and right rectangle positions:0,end
    Top and bottom rectangle positions:0,end
    Coordinates of ellipse center:500,500
    Ellipse radius, X direction:400
    Ellipse radius, Y direction:200
    Remove empty rows and columns?:No
    Select the masking image:ErodedWellEdge
    Select the image with a cropping mask:None
    Select the objects:None
""",
        encoding="utf-8",
    )
    (module,) = tuple(CPPipeParser().parse(cppipe_path))
    assert module.variable_revision_number == 3
    assert len(module.setting_records) == 14

    available = ArtifactSpecCollection(
        (
            ArtifactSpec.output("Worms", ImageArtifactType),
            ArtifactSpec.output("ErodedWellEdge", ImageArtifactType),
        )
    )
    step_context = ArtifactDeclarationStepContext(
        step_name="Crop",
        step_index=0,
        available_artifacts=available,
        main_flow_artifacts=available,
    )
    invocation_key = FunctionInvocationKey(
        function_name="crop",
        group_key=DEFAULT_GROUP_KEY,
        position=0,
    )
    parsed_contract = CropModule.callable_contract(
        module=module,
        invocation_key=invocation_key,
        step_context=step_context,
    )
    bound = CropModule.bind_settings(module, binder=SettingsBinder())
    invocation = next(
        normalize_function_pattern((crop, dict(bound.kwargs))).iter_items()
    )
    blocks, consumed_names = CropModule.module_blocks_for_invocation(
        invocation=invocation,
        step_context=step_context,
    )
    (numbered_blocks,), _next_module_num = CropModule.number_step_invocation_blocks(
        (blocks,),
        first_module_num=module.module_num,
    )

    public_contract, _provider_consumed_names = (
        CropModule.invocation_callable_contract(
            invocation=invocation,
            numbered_module_blocks=numbered_blocks,
            consumed_kwarg_names=consumed_names,
            step_context=step_context,
        )
    )

    assert public_contract == parsed_contract
    primary_ref = ArtifactSpec.input("Worms", ImageArtifactType).ref()
    mask_input = public_contract.artifact_inputs.require_by_name_and_artifact_type(
        "ErodedWellEdge",
        ImageArtifactType,
    )
    assert mask_input.parameter_name == "topology_inputs"
    assert mask_input.relations == (
        InputStackBroadcastSourceRelation(source=primary_ref),
    )
    assert tuple(
        output.source_context_sources()
        for output in public_contract.artifact_outputs.of_artifact_type(
            ImageArtifactType
        )
    ) == ((primary_ref,), (primary_ref,))


@pytest.mark.parametrize(
    ("module_type", "expected_binding"),
    (
        (MaskObjectsModule, MaskObjectsModule.input_objects_binding),
        (
            MeasureObjectNeighborsModule,
            MeasureObjectNeighborsModule.measured_objects_binding,
        ),
        (
            IdentifyTertiaryObjectsModule,
            IdentifyTertiaryObjectsModule.larger_objects_binding,
        ),
        (TrackObjectsModule, TrackObjectsModule.tracked_objects_binding),
    ),
)
def test_multi_label_primary_image_domain_has_one_exact_binding(
    module_type,
    expected_binding,
) -> None:
    declared_labels = module_type.declared_artifact_bindings(
        plan_type=ArtifactInputPlan,
        artifact_type=ObjectLabelsArtifactType,
    )

    assert module_type.primary_image_domain_input_binding() is expected_binding
    assert expected_binding in declared_labels


def test_primary_image_projection_consumes_nominal_binding_hook(monkeypatch) -> None:
    larger = ObjectLabelSet(
        name="Larger",
        variant_data=ObjectLabelVariantData(
            labels=np.ones((2, 3), dtype=np.int32),
        ),
    )
    smaller = ObjectLabelSet(
        name="Smaller",
        variant_data=ObjectLabelVariantData(
            labels=np.ones((4, 5), dtype=np.int32),
        ),
    )
    monkeypatch.setattr(
        IdentifyTertiaryObjectsModule,
        "primary_image_domain_input_binding",
        classmethod(lambda cls: cls.smaller_objects_binding),
    )

    projected = IdentifyTertiaryObjectsModule.project_invocation_image_request(
        image_request=CellProfilerImageRequest(
            source_image_name="Carrier",
            source_aliases=("Carrier",),
            image_count=1,
            payload=np.zeros((3, 3), dtype=np.float32),
        ),
        runtime_kwargs={
            "secondary_labels": larger,
            "primary_labels": smaller,
        },
    )

    assert np.shape(image_payload_data(projected.payload)) == (4, 5)
    assert projected.source_image_name is None
    assert projected.source_aliases == ()
