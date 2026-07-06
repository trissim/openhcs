"""
Converted from CellProfiler: MeasureImageAreaOccupied
Measures the total area in an image that is occupied by objects or foreground.
"""

from openhcs.interop.cellprofiler.setting_names import split_symbol_names
import numpy as np
from typing import Annotated, Optional, Sequence, Tuple
from dataclasses import dataclass
from enum import Enum
from openhcs.core.callable_contract import processing_prepare
from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_semantics import (
    MeasurementRowAxisField,
    RuntimeMeasurementFeature,
)
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.processing.backends.analysis.region_properties import (
    binary_area_and_perimeter_2d,
    label_area_and_rounded_perimeter_2d,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import (
    composed_image_payload,
    special_inputs,
    special_outputs,
)
from openhcs.processing.materialization import csv_materializer
from openhcs.interop.cellprofiler.module_declarations import (
    ProcessingContract,
    ImageArtifactInputCapability,
    ImageArtifactInputModule,
    ImageArtifactOutputCapability,
    ImageArtifactOutputModule,
    MeasurementArtifactOutputModule,
    ModuleSettingsSourceModule,
    ObjectArtifactInputModule,
    ObjectLabelArtifactInputCapability,
    SourceQualifiedMeasurementFeatureModule,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    block_setting_value,
    optional_setting_value,
    repeating_setting_blocks,
    required_setting_value,
    setting_names,
)
from openhcs.interop.cellprofiler.parser import ModuleSetting
from openhcs.interop.cellprofiler.setting_names import normalized_symbol_name
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    ObjectRowsInputPolicy,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    FieldDerivedMeasurementFeatureModule,
    MeasurementFeatureRecord,
    NoFieldsMeasurementRecordMixin,
    NoObjectNameMeasurementRecordMixin,
    SourceQualifiedInputPayloadMeasurementRecordMixin,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    ModuleOwnedResultMeasurementRows,
)
from openhcs.interop.cellprofiler.runtime.payload_types import CellProfilerKwargs


class MeasureImageAreaOccupiedBinaryModule(
    ObjectRowsInputPolicy,
    NoObjectNameMeasurementRecordMixin,
    SourceQualifiedInputPayloadMeasurementRecordMixin,
    SourceQualifiedMeasurementFeatureModule,
    FieldDerivedMeasurementFeatureModule,
    NoFieldsMeasurementRecordMixin,
    ImageArtifactInputModule,
    ObjectArtifactInputModule,
    ImageArtifactOutputModule,
    MeasurementArtifactOutputModule,
    ModuleSettingsSourceModule,
):
    module_name = "MeasureImageAreaOccupiedBinary"
    function_name = "measure_image_area_occupied"
    validated = True
    aliases = ("MeasureImageAreaOccupied",)
    function_variants = (
        "measure_image_area_occupied_binary",
        "measure_image_area_occupied_objects",
        "measure_image_volume_occupied_binary",
        "measure_image_volume_occupied_objects",
    )
    contract = ProcessingContract.FLEXIBLE
    confidence = 1.0
    measurement_category_prefixes = (("area", "occupied"),)
    measurement_feature_family = "AreaOccupied"

    class MeasurementFeature(RuntimeMeasurementFeature):
        """Feature families emitted by MeasureImageAreaOccupied."""

        AREA_OCCUPIED = "AreaOccupied"
        PERIMETER = "Perimeter"
        TOTAL_AREA = "TotalArea"

    @dataclass(frozen=True, slots=True)
    class MeasurementRows(ModuleOwnedResultMeasurementRows):
        """Project AreaOccupied result records into source-qualified CP features."""

        registry_key = "area_occupied"

        @classmethod
        def for_request(cls, module_type, request):
            return cls(request.output_value, module_type=module_type)

        def rows(self) -> list[CellProfilerKwargs]:
            records: list[AreaOccupiedMeasurement] = []
            for source_record in self.source_rows():
                if not isinstance(source_record, AreaOccupiedMeasurement):
                    raise TypeError(
                        "MeasureImageAreaOccupied measurement rows must be emitted "
                        "as AreaOccupiedMeasurement dataclasses."
                    )
                records.append(source_record)
            return self.module_type.source_qualified_measurement_feature_rows_from_records(
                tuple(records)
            )

    mode_setting = SettingNameFamily(
        "Measure the area occupied in a binary image, or in objects?",
        aliases=("Measure the area occupied by",),
    )
    binary_image_setting = SettingNameFamily(
        "Select a binary image to measure", aliases=("Select binary images to measure",)
    )
    objects_setting = SettingNameFamily(
        "Select objects to measure", aliases=("Select object sets to measure",)
    )
    retain_image_setting = "Retain a binary image of the object regions?"
    output_image_setting = "Name the output binary image"

    class Operand(str, Enum):
        BINARY_IMAGE = "binary_image"
        OBJECTS = "objects"

    @dataclass(frozen=True, slots=True)
    class MeasurementRow:
        operand: "MeasureImageAreaOccupiedBinaryModule.Operand"
        input_name: str
        binary_image_name: str | None
        objects_name: str | None
        retained_image_name: str | None

    @classmethod
    def settings_source(cls, module: "ModuleBlock") -> "CellProfilerKwargs":
        rows = cls.measurement_rows(module)
        return {
            "operand_choices": tuple((row.operand.value for row in rows)),
            "input_names": tuple((row.input_name for row in rows)),
            "retained_image_names": tuple((row.retained_image_name for row in rows)),
        }

    @classmethod
    def compile_time_setting_records_from_kwargs(cls, kwargs):
        if not {
            "operand_choices",
            "input_names",
            "retained_image_names",
        }.issubset(kwargs):
            return ()
        operand_choices = tuple(kwargs["operand_choices"])
        input_names = tuple(kwargs["input_names"])
        retained_image_names = tuple(kwargs["retained_image_names"])
        if (
            len(operand_choices) != len(input_names)
            or len(input_names) != len(retained_image_names)
        ):
            raise ValueError(
                "MeasureImageAreaOccupied compile-time kwargs must align by row."
            )

        records: list[ModuleSetting] = []
        for operand_literal, input_name, retained_image_name in zip(
            operand_choices,
            input_names,
            retained_image_names,
            strict=True,
        ):
            operand = cls._operand_from_kwarg(operand_literal)
            records.extend(
                cls._compile_time_setting_records_for_row(
                    operand,
                    input_name=str(input_name),
                    retained_image_name=(
                        None
                        if retained_image_name is None
                        else str(retained_image_name)
                    ),
                )
            )
        return tuple(records)

    @classmethod
    def measurement_rows(
        cls, module: "ModuleBlock"
    ) -> tuple["MeasureImageAreaOccupiedBinaryModule.MeasurementRow", ...]:
        rows: list[MeasureImageAreaOccupiedBinaryModule.MeasurementRow] = []
        for block in repeating_setting_blocks(
            module.iter_settings(), start_name=cls.mode_setting
        ):
            rows.extend(cls._expanded_rows_from_block(module, block))
        return tuple(rows)

    @classmethod
    def _expanded_rows_from_block(
        cls, module: "ModuleBlock", block: Sequence["ModuleSetting"]
    ) -> tuple["MeasureImageAreaOccupiedBinaryModule.MeasurementRow", ...]:
        operand = cls._operand_from_literal(
            block_setting_value(block, cls.mode_setting)
        )
        binary_image_name = normalized_symbol_name(
            block_setting_value(block, cls.binary_image_setting)
        )
        objects_name = normalized_symbol_name(
            block_setting_value(block, cls.objects_setting)
        )
        input_name = cls._input_name_for_operand(
            module,
            operand,
            binary_image_name=binary_image_name,
            objects_name=objects_name,
        )
        retained_image_name = cls._retained_image_name(block)
        expanded_input_names = cls._expanded_input_names(
            module, operand, input_name=input_name, block=block
        )
        return tuple(
            (
                cls.MeasurementRow(
                    operand=operand,
                    input_name=expanded_input_name,
                    binary_image_name=(
                        expanded_input_name
                        if operand is cls.Operand.BINARY_IMAGE
                        else None
                    ),
                    objects_name=(
                        expanded_input_name if operand is cls.Operand.OBJECTS else None
                    ),
                    retained_image_name=retained_image_name,
                )
                for expanded_input_name in expanded_input_names
            )
        )

    @classmethod
    def _operand_from_literal(
        cls, value: str
    ) -> "MeasureImageAreaOccupiedBinaryModule.Operand":
        normalized = value.strip().lower()
        if "binary" in normalized:
            return cls.Operand.BINARY_IMAGE
        if "object" in normalized:
            return cls.Operand.OBJECTS
        raise ValueError(f"Unsupported MeasureImageAreaOccupied mode {value!r}.")

    @classmethod
    def _operand_from_kwarg(
        cls, value: object
    ) -> "MeasureImageAreaOccupiedBinaryModule.Operand":
        if isinstance(value, cls.Operand):
            return value
        return cls.Operand(str(value))

    @classmethod
    def _compile_time_setting_records_for_row(
        cls,
        operand: "MeasureImageAreaOccupiedBinaryModule.Operand",
        *,
        input_name: str,
        retained_image_name: str | None,
    ) -> tuple[ModuleSetting, ...]:
        binary_image_name = input_name if operand is cls.Operand.BINARY_IMAGE else "None"
        objects_name = input_name if operand is cls.Operand.OBJECTS else "None"
        records = [
            ModuleSetting(
                setting_names(cls.mode_setting)[0],
                "Binary image" if operand is cls.Operand.BINARY_IMAGE else "Objects",
            ),
            ModuleSetting(setting_names(cls.binary_image_setting)[0], binary_image_name),
            ModuleSetting(setting_names(cls.objects_setting)[0], objects_name),
            ModuleSetting(
                cls.retain_image_setting,
                "Yes" if retained_image_name is not None else "No",
            ),
        ]
        if retained_image_name is not None:
            records.append(ModuleSetting(cls.output_image_setting, retained_image_name))
        return tuple(records)

    @classmethod
    def _input_name_for_operand(
        cls,
        module: "ModuleBlock",
        operand: "MeasureImageAreaOccupiedBinaryModule.Operand",
        *,
        binary_image_name: str | None,
        objects_name: str | None,
    ) -> str:
        if operand is cls.Operand.BINARY_IMAGE:
            input_name = binary_image_name
            role = "binary image"
        else:
            input_name = objects_name
            role = "object"
        if input_name is None:
            raise ValueError(
                f"Module {module.name}({module.module_num}) has an area-occupied row with no {role} input."
            )
        return input_name

    @classmethod
    def _expanded_input_names(
        cls,
        module: "ModuleBlock",
        operand: "MeasureImageAreaOccupiedBinaryModule.Operand",
        *,
        input_name: str,
        block: Sequence["ModuleSetting"],
    ) -> tuple[str, ...]:
        if operand is cls.Operand.BINARY_IMAGE:
            return split_symbol_names(
                block_setting_value(block, cls.binary_image_setting)
            ) or (input_name,)
        return split_symbol_names(block_setting_value(block, cls.objects_setting)) or (
            input_name,
        )

    @classmethod
    def _retained_image_name(cls, block: Sequence["ModuleSetting"]) -> str | None:
        retain_literal = block_setting_value(block, cls.retain_image_setting)
        if retain_literal.strip().lower() != "yes":
            return None
        return normalized_symbol_name(
            block_setting_value(block, cls.output_image_setting)
        )

    @classmethod
    def artifact_contract(cls, assembler, builder, module):
        rows = cls.measurement_rows(module)
        if not rows:
            raise ValueError(
                f"Module {module.name}({module.module_num}) declares no MeasureImageAreaOccupied measurement rows."
            )
        inputs = []
        retained_images = []
        for row in rows:
            if (
                row.operand is cls.Operand.BINARY_IMAGE
                and row.binary_image_name is not None
            ):
                inputs.append(
                    ImageArtifactInputCapability.bind_artifact(cls, builder, module, ImageArtifactInputCapability.spec(row.binary_image_name))
                )
            if row.operand is cls.Operand.OBJECTS and row.objects_name is not None:
                inputs.append(
                    ObjectLabelArtifactInputCapability.bind_artifact(cls, builder, module, ObjectLabelArtifactInputCapability.spec(row.objects_name))
                )
            if row.retained_image_name is not None:
                retained_images.append(
                    ImageArtifactOutputCapability.bind_artifact(cls, builder, module, ImageArtifactOutputCapability.spec(row.retained_image_name))
                )
        measurements = cls.measurement_output_artifact(builder, module)
        return assembler.assemble_contract(
            module, builder, inputs=inputs, outputs=[*retained_images, measurements]
        )


class OperandChoice(Enum):
    BINARY_IMAGE = "binary_image"
    OBJECTS = "objects"

    @classmethod
    def from_literal(cls, value: "OperandChoice | str") -> "OperandChoice":
        if isinstance(value, cls):
            return value
        normalized = value.strip().lower()
        if "binary" in normalized:
            return cls.BINARY_IMAGE
        if "object" in normalized:
            return cls.OBJECTS
        return cls(normalized)


@dataclass(frozen=True, slots=True)
class AreaOccupiedRuntimeRow:
    """One typed runtime row for the generic area-occupied runner."""

    operand: OperandChoice
    input_name: str
    retained_image_name: str | None

    @classmethod
    def from_literals(
        cls,
        *,
        operand: OperandChoice | str,
        input_name: str,
        retained_image_name: str | None,
    ) -> "AreaOccupiedRuntimeRow":
        normalized_input_name = input_name.strip()
        if not normalized_input_name:
            raise ValueError("AreaOccupiedRuntimeRow.input_name cannot be empty.")
        return cls(
            operand=OperandChoice.from_literal(operand),
            input_name=normalized_input_name,
            retained_image_name=retained_image_name,
        )


@dataclass
class AreaOccupiedMeasurement(MeasurementFeatureRecord):
    """Measurements for area occupied analysis."""

    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
    area_occupied: float
    perimeter: float
    total_area: float
    source_image_name: Annotated[
        str | None,
        MeasurementRowAxisField.SOURCE_IMAGE_NAME,
    ] = None

    @classmethod
    def from_area(
        cls,
        *,
        area_occupied: float,
        perimeter: float,
        total_area: float,
        slice_index: int = 0,
        source_image_name: str | None = None,
    ) -> "AreaOccupiedMeasurement":
        return cls(
            slice_index=slice_index,
            area_occupied=area_occupied,
            perimeter=perimeter,
            total_area=total_area,
            source_image_name=source_image_name,
        )


@dataclass(frozen=True, slots=True)
class BinaryAreaOccupiedRequest:
    """Measure occupied area for one binary image plane."""

    image: np.ndarray
    slice_index: int = 0
    source_image_name: str | None = None

    def measure(self) -> tuple[np.ndarray, AreaOccupiedMeasurement]:
        binary_mask = self.image > 0
        area_occupied, perimeter_value = binary_area_and_perimeter_2d(binary_mask)
        measurement = AreaOccupiedMeasurement.from_area(
            area_occupied=area_occupied,
            perimeter=perimeter_value,
            total_area=float(np.prod(self.image.shape)),
            slice_index=self.slice_index,
            source_image_name=self.source_image_name,
        )
        return (self.image, measurement)


@dataclass(frozen=True, slots=True)
class ObjectLabelsAreaOccupiedRequest:
    """Measure occupied area for one object-label plane."""

    image: np.ndarray
    labels: np.ndarray
    slice_index: int = 0
    source_image_name: str | None = None

    def measure(self) -> tuple[np.ndarray, AreaOccupiedMeasurement]:
        label_array = object_label_dense_array(self.labels)
        area_occupied, perimeter_value = label_area_and_perimeter(label_array)
        measurement = AreaOccupiedMeasurement.from_area(
            area_occupied=area_occupied,
            perimeter=perimeter_value,
            total_area=float(np.prod(label_array.shape)),
            slice_index=self.slice_index,
            source_image_name=self.source_image_name,
        )
        object_region_mask = (label_array > 0).astype(np.asarray(self.image).dtype)
        return (object_region_mask, measurement)


@composed_image_payload
@numpy(contract=ProcessingContract.FLEXIBLE)
@special_outputs(
    (
        "area_measurements",
        csv_materializer(
            fields=["slice_index", "area_occupied", "perimeter", "total_area"],
            analysis_type="area_occupied",
        ),
    )
)
def measure_image_area_occupied(
    image: np.ndarray,
    *,
    operand_choices: Sequence[OperandChoice | str] = (OperandChoice.BINARY_IMAGE,),
    input_names: Sequence[str] = ("image",),
    retained_image_names: Sequence[str | None] = (None,),
    object_labels: Sequence[np.ndarray] = (),
    slice_by_slice: bool = True,
) -> tuple:
    """Measure area occupied for ordered binary-image and object rows."""
    rows = _area_occupied_runtime_rows(
        operand_choices, input_names, retained_image_names
    )
    binary_images = _binary_images_from_payload(
        image, sum((row.operand is OperandChoice.BINARY_IMAGE for row in rows))
    )
    expected_object_rows = sum((row.operand is OperandChoice.OBJECTS for row in rows))
    if len(object_labels) != expected_object_rows:
        raise ValueError(
            "MeasureImageAreaOccupied object_labels count must match object rows: "
            f"got {len(object_labels)} object label input(s) for "
            f"{expected_object_rows} object row(s) from "
            f"operand_choices={tuple(row.operand.value for row in rows)!r}, "
            f"input_names={tuple(row.input_name for row in rows)!r}; "
            f"object_labels_type={type(object_labels).__name__}, "
            f"object_label_items={_object_label_items_diagnostic(object_labels)}."
        )
    retained_outputs = []
    measurements = []
    binary_index = 0
    object_index = 0
    for row_index, row in enumerate(rows):
        if row.operand is OperandChoice.BINARY_IMAGE:
            output_image, measurement = BinaryAreaOccupiedRequest(
                image=binary_images[binary_index],
                slice_index=row_index,
                source_image_name=row.input_name,
            ).measure()
            binary_index += 1
        else:
            labels = object_labels[object_index]
            output_image, measurement = ObjectLabelsAreaOccupiedRequest(
                image=_reference_image_for_labels(image, labels),
                labels=labels,
                slice_index=row_index,
                source_image_name=row.input_name,
            ).measure()
            object_index += 1
        measurements.append(measurement)
        if row.retained_image_name is not None:
            retained_outputs.append(output_image)
    if retained_outputs:
        return (*retained_outputs, measurements)
    return (image, measurements)


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(
    (
        "area_measurements",
        csv_materializer(
            fields=["slice_index", "area_occupied", "perimeter", "total_area"],
            analysis_type="area_occupied",
        ),
    )
)
def measure_image_area_occupied_binary(
    image: np.ndarray, source_image_name: str | None = None
) -> Tuple[np.ndarray, AreaOccupiedMeasurement]:
    """
    Measure area occupied by foreground in a binary image.

    Args:
        image: Binary image (H, W) where foreground > 0

    Returns:
        Tuple of (original image, AreaOccupiedMeasurement)
    """
    return BinaryAreaOccupiedRequest(
        image=image, source_image_name=source_image_name
    ).measure()


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    (
        "area_measurements",
        csv_materializer(
            fields=["slice_index", "area_occupied", "perimeter", "total_area"],
            analysis_type="area_occupied",
        ),
    )
)
def measure_image_area_occupied_objects(
    image: np.ndarray, labels: np.ndarray, source_image_name: str | None = None
) -> Tuple[np.ndarray, AreaOccupiedMeasurement]:
    """
    Measure area occupied by labeled objects.

    Args:
        image: Intensity image (H, W)
        labels: Label image from segmentation (H, W)

    Returns:
        Tuple of (original image, AreaOccupiedMeasurement)
    """
    return ObjectLabelsAreaOccupiedRequest(
        image=image, labels=labels, source_image_name=source_image_name
    ).measure()


def _area_occupied_runtime_rows(
    operand_choices: Sequence[OperandChoice | str],
    input_names: Sequence[str],
    retained_image_names: Sequence[str | None],
) -> tuple[AreaOccupiedRuntimeRow, ...]:
    if len(operand_choices) != len(input_names) or len(input_names) != len(
        retained_image_names
    ):
        raise ValueError(
            "MeasureImageAreaOccupied row kwargs must have matching lengths."
        )
    return tuple(
        (
            AreaOccupiedRuntimeRow.from_literals(
                operand=operand,
                input_name=input_name,
                retained_image_name=retained_image_name,
            )
            for operand, input_name, retained_image_name in zip(
                operand_choices, input_names, retained_image_names, strict=True
            )
        )
    )


def _object_label_items_diagnostic(object_labels: Sequence[np.ndarray]) -> tuple[str, ...]:
    diagnostics: list[str] = []
    for label in object_labels:
        shape = getattr(label, "shape", None)
        if shape is None:
            data = getattr(label, "data", None)
            shape = getattr(data, "shape", None)
        diagnostics.append(f"{type(label).__name__}(shape={shape!r})")
    return tuple(diagnostics)


def _binary_images_from_payload(
    image: np.ndarray, binary_image_count: int
) -> tuple[np.ndarray, ...]:
    if binary_image_count == 0:
        return ()
    if binary_image_count == 1:
        if isinstance(image, np.ndarray) and image.ndim == 3 and (image.shape[0] == 1):
            return (image[0],)
        return (image,)
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError(
            "MeasureImageAreaOccupied requires a stacked image payload for multiple binary-image rows."
        )
    if image.shape[0] != binary_image_count:
        raise ValueError(
            "MeasureImageAreaOccupied binary image stack length must match binary-image row count."
        )
    return tuple((image[index] for index in range(binary_image_count)))


def label_area_and_perimeter(labels: np.ndarray) -> tuple[float, float]:
    label_array = object_label_dense_array(labels)
    if label_array.ndim <= 2:
        return _label_plane_area_and_perimeter(label_array)
    plane_measurements = tuple(
        (
            _label_plane_area_and_perimeter(label_array[index])
            for index in range(label_array.shape[0])
        )
    )
    return (
        float(sum((area for area, _perimeter in plane_measurements))),
        float(sum((perimeter for _area, perimeter in plane_measurements))),
    )


def _label_plane_area_and_perimeter(labels: np.ndarray) -> tuple[float, float]:
    labels_array = object_label_dense_array(labels, dtype=np.int32)
    return label_area_and_rounded_perimeter_2d(labels_array)


def _reference_image_for_labels(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    label_array = object_label_dense_array(labels)
    if image.ndim == label_array.ndim:
        return image
    if image.ndim == label_array.ndim + 1 and image.shape[0] >= 1:
        return image[0]
    return image


@dataclass
class VolumeOccupiedMeasurement:
    """Measurements for volume occupied analysis (3D)."""

    volume_occupied: float
    surface_area: float
    total_volume: float

    @classmethod
    def from_volume(
        cls, *, volume_occupied: float, surface_area: float, total_volume: float
    ) -> "VolumeOccupiedMeasurement":
        return cls(
            volume_occupied=volume_occupied,
            surface_area=surface_area,
            total_volume=total_volume,
        )


@dataclass(frozen=True, slots=True)
class SurfaceAreaRequest:
    """Compute rounded surface area for one 3D label image."""

    label_image: np.ndarray
    spacing: Optional[Tuple[float, ...]] = None

    def surface_area(self) -> float:
        from skimage.measure import marching_cubes, mesh_surface_area

        spacing = self.spacing
        label_image = np.asarray(self.label_image)
        if spacing is None:
            spacing = (1.0,) * label_image.ndim
        unique_labels = np.unique(label_image)
        unique_labels = unique_labels[unique_labels != 0]
        if len(unique_labels) == 0:
            return 0.0
        total_surface = 0.0
        for label in unique_labels:
            binary_mask = (label_image == label).astype(np.float32)
            try:
                verts, faces, _, _ = marching_cubes(
                    binary_mask, spacing=spacing, level=0.5, method="lorensen"
                )
                total_surface += mesh_surface_area(verts, faces)
            except (ValueError, RuntimeError):
                continue
        return float(np.round(total_surface))


@dataclass(frozen=True, slots=True)
class VolumeOccupiedRequest:
    """Materialize a volume-occupied measurement from voxel totals."""

    volume_occupied: float
    surface_area: float
    total_volume: float

    def measurement(self) -> VolumeOccupiedMeasurement:
        return VolumeOccupiedMeasurement.from_volume(
            volume_occupied=self.volume_occupied,
            surface_area=self.surface_area,
            total_volume=self.total_volume,
        )


@numpy(contract=ProcessingContract.PURE_3D)
@special_outputs(
    (
        "volume_measurements",
        csv_materializer(
            fields=["volume_occupied", "surface_area", "total_volume"],
            analysis_type="volume_occupied",
        ),
    )
)
def measure_image_volume_occupied_binary(
    image: np.ndarray, spacing: Optional[Tuple[float, float, float]] = None
) -> Tuple[np.ndarray, VolumeOccupiedMeasurement]:
    """
    Measure volume occupied by foreground in a 3D binary image.

    Args:
        image: 3D binary image (D, H, W) where foreground > 0
        spacing: Voxel spacing (z, y, x) for surface area calculation

    Returns:
        Tuple of (original image, VolumeOccupiedMeasurement)
    """
    binary_mask = image > 0
    volume_occupied = float(np.sum(binary_mask))
    if volume_occupied > 0:
        surface_area_value = SurfaceAreaRequest(
            binary_mask.astype(np.int32), spacing=spacing
        ).surface_area()
    else:
        surface_area_value = 0.0
    total_volume = float(np.prod(image.shape))
    measurement = VolumeOccupiedRequest(
        volume_occupied=volume_occupied,
        surface_area=surface_area_value,
        total_volume=total_volume,
    ).measurement()
    return (image, measurement)


@numpy(contract=ProcessingContract.PURE_3D)
@special_inputs("labels")
@special_outputs(
    (
        "volume_measurements",
        csv_materializer(
            fields=["volume_occupied", "surface_area", "total_volume"],
            analysis_type="volume_occupied",
        ),
    )
)
def measure_image_volume_occupied_objects(
    image: np.ndarray,
    labels: np.ndarray,
    spacing: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, VolumeOccupiedMeasurement]:
    """
    Measure volume occupied by labeled objects in 3D.

    Args:
        image: 3D intensity image (D, H, W)
        labels: 3D label image from segmentation (D, H, W)
        spacing: Voxel spacing (z, y, x) for surface area calculation

    Returns:
        Tuple of (original image, VolumeOccupiedMeasurement)
    """
    labels_array = object_label_dense_array(labels, dtype=np.int32)
    volume_occupied = float(np.count_nonzero(labels_array))
    if volume_occupied > 0:
        surface_area_value = SurfaceAreaRequest(
            labels_array, spacing=spacing
        ).surface_area()
    else:
        surface_area_value = 0.0
    total_volume = float(np.prod(labels_array.shape))
    measurement = VolumeOccupiedRequest(
        volume_occupied=volume_occupied,
        surface_area=surface_area_value,
        total_volume=total_volume,
    ).measurement()
    return (image, measurement)


@processing_prepare(measure_image_area_occupied)
def _prepare_measure_image_area_occupied() -> None:
    """Compile reusable area/perimeter kernels before timed execution."""
    binary = np.zeros((64, 64), dtype=np.float32)
    binary[8:40, 12:48] = 1.0
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[8:24, 8:24] = 1
    labels[32:56, 32:56] = 2
    BinaryAreaOccupiedRequest(binary).measure()
    ObjectLabelsAreaOccupiedRequest(binary, labels).measure()
