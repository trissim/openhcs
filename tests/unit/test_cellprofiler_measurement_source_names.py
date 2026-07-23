"""Focused CellProfiler composed measurement-source identity tests."""

from types import SimpleNamespace

import numpy as np

from openhcs.core.artifacts import (
    ArtifactSpec,
    ImageArtifactType,
    MeasurementsArtifactType,
)
from openhcs.core.callable_contract import ImagePayloadConsumption
from openhcs.core.equivalence.keys import RuntimeMeasurementSourcePair
import openhcs.interop.cellprofiler.runtime.module_execution as module_execution
from openhcs.interop.cellprofiler.runtime.invocation import CellProfilerImageRequest
from openhcs.interop.cellprofiler.runtime.measurement_source_names import (
    measurement_source_name_for_specs,
)
from openhcs.interop.cellprofiler.runtime.module_execution import (
    CellProfilerModuleExecutor,
)


def _image_specs(*names: str) -> tuple[ArtifactSpec, ...]:
    return tuple(ArtifactSpec.input(name, ImageArtifactType) for name in names)


def test_measurement_source_name_uses_zero_and_single_distinct_semantics() -> None:
    assert measurement_source_name_for_specs(()) is None
    assert measurement_source_name_for_specs(_image_specs("DNA")) == "DNA"
    assert measurement_source_name_for_specs(_image_specs("DNA", "DNA")) == "DNA"


def test_measurement_source_name_uses_ordered_runtime_pair_identity() -> None:
    forward = measurement_source_name_for_specs(_image_specs("DNA", "GFP"))
    reverse = measurement_source_name_for_specs(_image_specs("GFP", "DNA"))

    assert forward == RuntimeMeasurementSourcePair("DNA", "GFP").source_name
    assert reverse == RuntimeMeasurementSourcePair("GFP", "DNA").source_name
    assert forward != reverse


def test_larger_composition_does_not_collide_with_real_source_name() -> None:
    real_source_name = "DNA__GFP__Actin"

    assert (
        measurement_source_name_for_specs(_image_specs("DNA", "GFP", "Actin")) is None
    )
    assert (
        measurement_source_name_for_specs(_image_specs(real_source_name))
        == real_source_name
    )


def test_input_source_name_resolves_only_declared_context_carriers(
    monkeypatch,
) -> None:
    image_spec = ArtifactSpec.input("DNA", ImageArtifactType)
    measurement_spec = ArtifactSpec.input(
        "MeasureObjectIntensity_measurements",
        MeasurementsArtifactType,
    )
    requested_specs: list[ArtifactSpec] = []

    class BindingRequest:
        def artifact_request_for_spec(self, spec: ArtifactSpec) -> SimpleNamespace:
            requested_specs.append(spec)
            return SimpleNamespace(spec=spec)

    class ArtifactStrategy:
        @staticmethod
        def source_image_name(request: SimpleNamespace) -> str:
            return request.spec.name

    monkeypatch.setattr(
        module_execution,
        "RuntimeInputBindingRequest",
        lambda **_kwargs: BindingRequest(),
    )
    monkeypatch.setattr(
        module_execution,
        "RuntimeArtifactTypeStrategy",
        SimpleNamespace(for_artifact_type=lambda _artifact_type: ArtifactStrategy()),
    )

    source_name = CellProfilerModuleExecutor._input_source_image_name(
        SimpleNamespace(),
        SimpleNamespace(),
        np.zeros((2, 2), dtype=np.float32),
        active_input_specs=(measurement_spec, image_spec),
    )

    assert source_name == "DNA"
    assert requested_specs == [image_spec]


def test_composed_measurement_caller_preserves_ordered_source_aliases() -> None:
    source_aliases = ("DNA", "GFP", "Actin")
    image_inputs = _image_specs(*source_aliases)
    payload = np.zeros((3, 2, 2), dtype=np.float32)
    image_request = CellProfilerImageRequest(
        source_image_name=None,
        source_aliases=source_aliases,
        image_count=len(source_aliases),
        payload=payload,
    )

    executor = SimpleNamespace(
        callable_contract=SimpleNamespace(
            image_payload_consumption=ImagePayloadConsumption.COMPOSED
        ),
        _primary_image_inputs=lambda current_image, adapter, *, module_type: (
            image_inputs
        ),
    )

    measurement_images = CellProfilerModuleExecutor._measurement_image_inputs(
        executor,
        adapter=SimpleNamespace(),
        current_image=payload,
        image_request=image_request,
        module_type=SimpleNamespace(),
    )

    assert len(measurement_images) == 1
    measurement_image = measurement_images[0]
    assert measurement_image.source_image_name is None
    assert measurement_image.source_aliases == source_aliases
    assert measurement_image.payload is payload
    assert tuple(
        (pair.first.name, pair.second.name, pair.runtime_pair.source_name)
        for pair in measurement_image.source_image_pairs()
    ) == (
        ("DNA", "GFP", RuntimeMeasurementSourcePair("DNA", "GFP").source_name),
        ("DNA", "Actin", RuntimeMeasurementSourcePair("DNA", "Actin").source_name),
        ("GFP", "Actin", RuntimeMeasurementSourcePair("GFP", "Actin").source_name),
    )
