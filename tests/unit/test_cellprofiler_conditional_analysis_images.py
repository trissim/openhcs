from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect

import numpy as np
import pytest

from openhcs.core.aligned_image_payload import AlignedImageStack
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRelation,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    FunctionInvocationKey,
    normalize_function_pattern,
)
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_metadata,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelVariantData,
)
from openhcs.core.runtime_object_label_domains import ObjectLabelDomain
from openhcs.core.steps.function_step import FunctionStep
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.processing.backends.cellprofiler.classification import (
    ClassifiedImageSourceRelation,
    ClassifyObjectsSingleMeasurementModule,
    classification_rgb_image,
    classify_objects_single_measurement,
)
from openhcs.processing.backends.cellprofiler.colocalization import (
    ColocalizationThresholdMaskGroup,
    ColocalizationThresholdMaskObjectRelation,
    ColocalizationThresholdMaskRuntimeOutput,
    ColocalizationThresholdMaskSourceRelation,
    MeasureColocalizationModule,
    measure_colocalization,
    measure_colocalization_objects,
)
from openhcs.processing.backends.cellprofiler.intensity_distribution import (
    IntensityDistributionHeatmapGroup,
    IntensityDistributionHeatmapMeasurement,
    IntensityDistributionHeatmapObjectRelation,
    IntensityDistributionHeatmapRuntimeOutput,
    IntensityDistributionHeatmapSourceRelation,
    MeasureObjectIntensityDistributionModule,
    measure_object_intensity_distribution,
)


def _module_block(
    module_type,
    records: tuple[tuple[str, str], ...],
) -> ModuleBlock:
    return ModuleBlock(
        name=str(module_type.module_name),
        module_num=1,
        setting_records=[ModuleSetting(name, value) for name, value in records],
    )


def _invocation(func: Callable, kwargs: Mapping[str, object] | None = None):
    pattern = func if kwargs is None else (func, dict(kwargs))
    return next(normalize_function_pattern(pattern).iter_items())


def _contract(module_type, module: ModuleBlock, func: Callable, context):
    invocation = _invocation(func)
    contract = module_type.callable_contract(
        module=module,
        invocation_key=invocation.key,
        step_context=context,
    )
    resolved = module_type.resolve_function(
        module,
        contract=contract,
        source_bindings=context.source_bindings,
    )
    module_type.validate_callable_artifact_abi(resolved, contract)
    return contract


def _public_function_step_contract(
    module_type,
    func: Callable,
    kwargs: Mapping[str, object],
    context: ArtifactDeclarationStepContext,
):
    step = FunctionStep(func=(func, dict(kwargs)), name=str(module_type.module_name))
    invocation = next(normalize_function_pattern(step.func).iter_items())
    blocks, consumed_names = module_type.module_blocks_for_invocation(
        invocation=invocation,
        step_context=context,
    )
    (numbered_blocks,), _next_module_num = (
        module_type.number_step_invocation_blocks(
            (blocks,),
            first_module_num=1,
        )
    )
    contract, _consumed = module_type.invocation_callable_contract(
        invocation=invocation,
        numbered_module_blocks=numbered_blocks,
        consumed_kwarg_names=consumed_names,
        step_context=context,
    )
    return contract


def _image_outputs(contract):
    return contract.artifact_outputs.of_artifact_type(ImageArtifactType)


def _runtime_images(output: object) -> tuple[np.ndarray, ...]:
    values = output.slices if isinstance(output, AlignedImageStack) else (output,)
    return tuple(np.asarray(image_payload_data(value)) for value in values)


def _object_payload() -> ObjectLabelPayload:
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[1:4, 1:4] = 1
    labels[4:7, 4:7] = 2
    return ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
    )


def _classification_context() -> ArtifactDeclarationStepContext:
    objects = ArtifactSpec.output("Cells", ObjectLabelsArtifactType)
    object_ref = objects.for_plan_type(ArtifactInputPlan).ref()
    measurements = ArtifactSpec.output(
        "MeasureObjectSizeShape_1_measurements",
        MeasurementsArtifactType,
        relations=(ArtifactSpecRelation(object_ref),),
    )
    return ArtifactDeclarationStepContext(
        step_name="ClassifyObjects",
        step_index=3,
        available_artifacts=ArtifactSpecCollection((objects, measurements)),
        main_flow_artifacts=ArtifactSpecCollection(
            (objects.for_plan_type(ArtifactInputPlan),)
        ),
        available_artifact_producers=artifact_producers_for_outputs(
            (objects, measurements),
            groups=(None,),
            invocation_keys=(
                FunctionInvocationKey(
                    "measure_object_size_shape",
                    DEFAULT_GROUP_KEY,
                    0,
                ),
            ),
        ),
    )


def _classification_rules(active_count: int) -> tuple[dict[str, object], ...]:
    active_indices = {1} if active_count == 1 else set(range(active_count))
    return tuple(
        {
            "measurement_feature": "AreaShape_Area",
            "bin_choice": "custom",
            "custom_thresholds": "0,0.5,1",
            "bin_names": "Low,High",
            "retained_image_name": (
                ("FirstClassified", "SecondClassified")[rule_index]
                if rule_index in active_indices
                else None
            ),
        }
        for rule_index in range(2)
    )


def _classification_records(active_count: int) -> tuple[tuple[str, str], ...]:
    records: list[tuple[str, str]] = [
        (
            "Make each classification decision on how many measurements?",
            "Single measurement",
        ),
        ("Hidden", "2"),
        ("Select the object to be classified", "Cells"),
    ]
    for rule in _classification_rules(active_count):
        output_name = rule["retained_image_name"]
        records.extend(
            (
                ("Select the measurement to classify by", "AreaShape_Area"),
                ("Select bin spacing", "Custom-defined bins"),
                (
                    "Enter the custom thresholds separating the values between bins",
                    "0,0.5,1",
                ),
                (
                    "Retain an image of the classified objects?",
                    "Yes" if output_name else "No",
                ),
                ("Name the output image", str(output_name or "UnusedClassified")),
            )
        )
    return tuple(records)


@pytest.mark.parametrize(
    ("active_count", "expected_names", "expected_rule_indices"),
    (
        (0, (), ()),
        (1, ("SecondClassified",), (1,)),
        (2, ("FirstClassified", "SecondClassified"), (0, 1)),
    ),
)
def test_classify_objects_repeated_groups_declare_and_reconstruct_every_image(
    active_count: int,
    expected_names: tuple[str, ...],
    expected_rule_indices: tuple[int, ...],
) -> None:
    context = _classification_context()
    parsed = _module_block(
        ClassifyObjectsSingleMeasurementModule,
        _classification_records(active_count),
    )
    parsed_contract = _contract(
        ClassifyObjectsSingleMeasurementModule,
        parsed,
        classify_objects_single_measurement,
        context,
    )
    public_contract = _public_function_step_contract(
        ClassifyObjectsSingleMeasurementModule,
        classify_objects_single_measurement,
        {"classification_rules": _classification_rules(active_count)},
        context,
    )

    for contract in (parsed_contract, public_contract):
        outputs = _image_outputs(contract)
        assert tuple(output.name for output in outputs) == expected_names
        assert (
            tuple(
                relation.rule_index
                for output in outputs
                for relation in output.relations
                if isinstance(relation, ClassifiedImageSourceRelation)
            )
            == expected_rule_indices
        )


@pytest.mark.parametrize("active_count", (0, 1, 2))
def test_classify_objects_runtime_returns_active_images_in_rule_order(
    active_count: int,
) -> None:
    object_payload = _object_payload()
    labels = object_payload.variant_data.labels
    image = np.arange(labels.size, dtype=float).reshape(labels.shape) / labels.size
    active_indices = (
        () if active_count == 0 else ((1,) if active_count == 1 else (0, 1))
    )
    output, _rows = inspect.unwrap(classify_objects_single_measurement)(
        image,
        object_payload,
        classification_rules=_classification_rules(active_count),
        measurement_values_by_rule=(
            np.array((0.25, 0.75)),
            np.array((0.75, 0.25)),
        ),
        classified_image_rule_indices=active_indices,
    )

    if not active_indices:
        np.testing.assert_array_equal(image_payload_data(output), labels)
        return
    first_classes = np.where(labels == 1, 1, np.where(labels == 2, 2, 0))
    second_classes = np.where(labels == 1, 2, np.where(labels == 2, 1, 0))
    expected = tuple(
        (
            classification_rgb_image(first_classes),
            classification_rgb_image(second_classes),
        )[index]
        for index in active_indices
    )
    actual = _runtime_images(output)
    assert len(actual) == len(expected)
    values = output.slices if isinstance(output, AlignedImageStack) else (output,)
    assert all(
        image_payload_metadata(value).source_channel_axis == -1 for value in values
    )
    for actual_image, expected_image in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_image, expected_image)


def _analysis_context() -> ArtifactDeclarationStepContext:
    images = tuple(
        ArtifactSpec.output(name, ImageArtifactType) for name in ("DNA", "RNA")
    )
    objects = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType)
    return ArtifactDeclarationStepContext(
        step_name="conditional-analysis-images",
        step_index=0,
        available_artifacts=ArtifactSpecCollection((*images, objects)),
        main_flow_artifacts=ArtifactSpecCollection(
            image.for_plan_type(ArtifactInputPlan) for image in images
        ),
        available_artifact_producers=artifact_producers_for_outputs(
            (*images, objects),
            groups=(None,),
            invocation_keys=(
                FunctionInvocationKey("fixture_producer", DEFAULT_GROUP_KEY, 0),
            ),
        ),
    )


def _colocalization_groups(
    active_count: int,
) -> tuple[ColocalizationThresholdMaskGroup, ...]:
    groups = (
        ColocalizationThresholdMaskGroup("RNA", "RNAMask", 20.0),
        ColocalizationThresholdMaskGroup("DNA", "DNANucleiMask", 50.0, "Nuclei"),
    )
    return groups[:active_count]


def _colocalization_records(active_count: int) -> tuple[tuple[str, str], ...]:
    groups = _colocalization_groups(active_count)
    records: list[tuple[str, str]] = [
        ("Select images to measure", "DNA"),
        ("Select images to measure", "RNA"),
        ("Select where to measure correlation", "Across entire image"),
        (
            "Set threshold as percentage of maximum intensity for the images",
            "15",
        ),
        ("Enable image specific thresholds?", "Yes" if groups else "No"),
        ("Threshold count", str(len(groups))),
    ]
    for group in groups:
        records.extend(
            (
                ("Select the image", group.source_image_name),
                (
                    "Set threshold as percentage of maximum intensity of selected image",
                    str(group.threshold_percent),
                ),
            )
        )
    records.extend(
        (
            ("Save thresholded mask?", "Yes" if groups else "No"),
            ("Save mask count", str(len(groups))),
        )
    )
    for group in groups:
        records.extend(
            (
                ("Which image mask would you like to save", group.source_image_name),
                ("Use object for thresholding?", "Yes" if group.object_name else "No"),
            )
        )
        if group.object_name is not None:
            records.append(("Select an Object for threhsolding", group.object_name))
        records.append(("Name the output image", group.output_image_name))
    return tuple(records)


@pytest.mark.parametrize("active_count", (0, 1, 2))
def test_measure_colocalization_groups_declare_and_reconstruct_every_mask(
    active_count: int,
) -> None:
    context = _analysis_context()
    groups = _colocalization_groups(active_count)
    parsed = _module_block(
        MeasureColocalizationModule,
        _colocalization_records(active_count),
    )
    parsed_contract = _contract(
        MeasureColocalizationModule,
        parsed,
        measure_colocalization,
        context,
    )
    public_kwargs: dict[str, object] = {
        "select_images_to_measure": ("DNA", "RNA"),
        "threshold_mask_groups": groups,
    }
    if any(group.object_name is not None for group in groups):
        public_kwargs["select_object_sets_to_measure"] = ("Nuclei",)
    public_func = (
        measure_colocalization_objects
        if any(group.object_name is not None for group in groups)
        else measure_colocalization
    )
    public_contract = _public_function_step_contract(
        MeasureColocalizationModule,
        public_func,
        public_kwargs,
        context,
    )

    for contract in (parsed_contract, public_contract):
        outputs = _image_outputs(contract)
        assert tuple(output.name for output in outputs) == tuple(
            group.output_image_name for group in groups
        )
        for output, group in zip(outputs, groups, strict=True):
            (source_relation,) = tuple(
                relation
                for relation in output.relations
                if isinstance(relation, ColocalizationThresholdMaskSourceRelation)
            )
            assert source_relation.source.name == group.source_image_name
            assert source_relation.threshold_percent == group.threshold_percent
            object_relations = tuple(
                relation
                for relation in output.relations
                if isinstance(relation, ColocalizationThresholdMaskObjectRelation)
            )
            assert tuple(relation.source.name for relation in object_relations) == (
                () if group.object_name is None else (group.object_name,)
            )


@pytest.mark.parametrize("active_count", (0, 1, 2))
def test_measure_colocalization_runtime_returns_masks_in_contract_order(
    active_count: int,
) -> None:
    dna = np.arange(64, dtype=float).reshape(8, 8) / 63.0
    rna = np.flipud(dna)
    image = ImagePayloadMetadata(source_image_names=("DNA", "RNA")).attach_to(
        np.stack((dna, rna))
    )
    object_payload = _object_payload()
    all_groups = _colocalization_groups(2)
    groups = all_groups[:active_count]
    requests = (
        ColocalizationThresholdMaskRuntimeOutput(all_groups[0], 1),
        ColocalizationThresholdMaskRuntimeOutput(all_groups[1], 0, object_payload),
    )[:active_count]
    output, _rows = inspect.unwrap(measure_colocalization)(
        image,
        do_correlation=False,
        do_manders=False,
        do_rwc=False,
        do_overlap=False,
        do_costes=False,
        threshold_mask_groups=groups,
        threshold_mask_outputs=requests,
    )

    if not groups:
        np.testing.assert_array_equal(np.squeeze(image_payload_data(output)), dna)
        return
    expected_rna = rna > 0.2 * np.max(rna)
    labels = object_payload.variant_data.labels
    expected_dna = np.zeros(dna.shape, dtype=bool)
    for object_id in (1, 2):
        object_mask = labels == object_id
        expected_dna[object_mask] = dna[object_mask] >= 0.5 * np.max(dna[object_mask])
    expected = (expected_rna, expected_dna)[:active_count]
    actual = _runtime_images(output)
    assert len(actual) == len(expected)
    for actual_image, expected_image in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_image, expected_image)


def _heatmap_groups(active_count: int) -> tuple[IntensityDistributionHeatmapGroup, ...]:
    active_indices = {1} if active_count == 1 else set(range(active_count))
    return (
        IntensityDistributionHeatmapGroup(
            "DNA",
            "Nuclei",
            4,
            True,
            100,
            IntensityDistributionHeatmapMeasurement.FRACTION_AT_DISTANCE,
            "Default",
            0 in active_indices,
            "FractionHeatmap",
        ),
        IntensityDistributionHeatmapGroup(
            "DNA",
            "Nuclei",
            4,
            True,
            100,
            IntensityDistributionHeatmapMeasurement.RADIAL_CV,
            "gray",
            1 in active_indices,
            "RadialCVHeatmap",
        ),
    )


def _intensity_distribution_records(
    active_count: int,
) -> tuple[tuple[str, str], ...]:
    groups = _heatmap_groups(active_count)
    records: list[tuple[str, str]] = [
        ("Hidden", "1"),
        ("Hidden", "1"),
        ("Hidden", "2"),
        ("Select images to measure", "DNA"),
        ("Select object sets to measure", "Nuclei"),
        ("Scale the bins?", "Yes"),
        ("Number of bins", "4"),
        ("Maximum radius", "100"),
    ]
    for group in groups:
        records.extend(
            (
                ("Image", group.source_image_name),
                ("Objects to display", group.object_name),
                ("Number of bins", str(group.bin_count)),
                ("Measurement", group.measurement.value),
                ("Color map", group.colormap),
                ("Save display as image?", "Yes" if group.save_display else "No"),
                ("Output image name", group.output_image_name),
            )
        )
    return tuple(records)


@pytest.mark.parametrize("active_count", (0, 1, 2))
def test_intensity_distribution_groups_declare_and_reconstruct_every_heatmap(
    active_count: int,
) -> None:
    context = _analysis_context()
    groups = _heatmap_groups(active_count)
    active_groups = tuple(group for group in groups if group.save_display)
    parsed = _module_block(
        MeasureObjectIntensityDistributionModule,
        _intensity_distribution_records(active_count),
    )
    parsed_contract = _contract(
        MeasureObjectIntensityDistributionModule,
        parsed,
        measure_object_intensity_distribution,
        context,
    )
    public_contract = _public_function_step_contract(
        MeasureObjectIntensityDistributionModule,
        measure_object_intensity_distribution,
        {
            "select_images_to_measure": ("DNA",),
            "select_object_sets_to_measure": ("Nuclei",),
            "bin_count": 4,
            "wants_scaled": True,
            "maximum_radius": 100,
            "heatmap_groups": groups,
        },
        context,
    )

    for contract in (parsed_contract, public_contract):
        outputs = _image_outputs(contract)
        assert tuple(output.name for output in outputs) == tuple(
            group.output_image_name for group in active_groups
        )
        for output, group in zip(outputs, active_groups, strict=True):
            (source_relation,) = tuple(
                relation
                for relation in output.relations
                if isinstance(relation, IntensityDistributionHeatmapSourceRelation)
            )
            (object_relation,) = tuple(
                relation
                for relation in output.relations
                if isinstance(relation, IntensityDistributionHeatmapObjectRelation)
            )
            assert source_relation.source.name == group.source_image_name
            assert source_relation.bin_count == group.bin_count
            assert source_relation.wants_scaled is group.wants_scaled
            assert source_relation.maximum_radius == group.maximum_radius
            assert source_relation.measurement is group.measurement
            assert source_relation.colormap == group.colormap
            assert object_relation.source.name == group.object_name


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
@pytest.mark.parametrize("active_count", (0, 1, 2))
def test_intensity_distribution_runtime_returns_heatmaps_in_contract_order(
    active_count: int,
) -> None:
    source = np.arange(64, dtype=float).reshape(8, 8) / 63.0
    image = ImagePayloadMetadata(source_image_names=("DNA",)).attach_to(source)
    object_payload = _object_payload()
    groups = _heatmap_groups(active_count)
    active_groups = tuple(group for group in groups if group.save_display)
    requests = tuple(
        IntensityDistributionHeatmapRuntimeOutput(group, object_payload)
        for group in active_groups
    )
    output, _rows = inspect.unwrap(measure_object_intensity_distribution)(
        image,
        object_payload,
        bin_count=4,
        wants_scaled=True,
        maximum_radius=100,
        wants_zernikes="none",
        heatmap_groups=groups,
        heatmap_outputs=requests,
    )

    if not active_groups:
        np.testing.assert_array_equal(image_payload_data(output), source)
        return
    actual = _runtime_images(output)
    assert len(actual) == len(active_groups)
    values = output.slices if isinstance(output, AlignedImageStack) else (output,)
    assert tuple(
        image_payload_metadata(value).source_channel_axis for value in values
    ) == tuple(None if group.colormap == "gray" else -1 for group in active_groups)
    assert tuple(image.ndim for image in actual) == tuple(
        2 if group.colormap == "gray" else 3 for group in active_groups
    )
