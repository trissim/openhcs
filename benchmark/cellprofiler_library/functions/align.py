"""Compatibility implementation for legacy CellProfiler Align."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import numpy as np
import scipy.ndimage as scind
import scipy.sparse
from metaclass_registry import AutoRegisterMeta
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    MaskedImagePayload,
    image_payload_data,
    image_payload_metadata,
    image_payload_mask,
    image_payload_with_context,
)
from openhcs.processing.materialization import csv_materializer
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
)
from openhcs.processing.backends.cellprofiler.alignment import AlignmentBackendStrategy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from scipy.fftpack import fft2, ifft2


class AlignCropMode(str, Enum):
    """Closed crop modes from legacy CellProfiler Align."""

    KEEP_SIZE = "Keep size"
    CROP_TO_ALIGNED_REGION = "Crop to aligned region"
    PAD_IMAGES = "Pad images"

    @classmethod
    def from_literal(cls, value: "AlignCropMode | str") -> "AlignCropMode":
        if isinstance(value, cls):
            return value
        normalized = value.strip().lower()
        for mode in cls:
            if normalized == mode.value.lower():
                return mode
        raise ValueError(f"Unsupported Align crop mode {value!r}.")


class AlignAdditionalMode(str, Enum):
    """Closed modes for applying Align shifts to additional outputs."""

    SIMILARLY = "Similarly"

    @classmethod
    def from_literal(cls, value: "AlignAdditionalMode | str") -> "AlignAdditionalMode":
        if isinstance(value, cls):
            return value
        normalized = value.strip().lower()
        for mode in cls:
            if normalized == mode.value.lower():
                return mode
        raise ValueError(f"Unsupported Align additional-image mode {value!r}.")


@dataclass(frozen=True, slots=True)
class AlignCropRequest:
    """Inputs shared by Align crop-mode strategies."""

    first_image: np.ndarray
    second_image: np.ndarray
    first_mask: np.ndarray | None
    second_mask: np.ndarray | None
    offsets: tuple[tuple[int, int], tuple[int, int]]
    shapes: tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True, slots=True)
class AlignShiftMeasurement:
    """Per-output translation reported by CellProfiler Align."""

    slice_index: int
    output_index: int
    x_shift: float
    y_shift: float


@numpy(contract=ProcessingContract.FLEXIBLE)
@special_outputs(("align_measurements", csv_materializer(
    fields=["slice_index", "output_index", "x_shift", "y_shift"],
    analysis_type="alignment",
)))
def align(
    image: np.ndarray,
    *,
    method: str = "Mutual Information",
    crop_mode: AlignCropMode | str = AlignCropMode.KEEP_SIZE,
    additional_alignment_modes: tuple[AlignAdditionalMode | str, ...] = (),
    alignment_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[object, ...]:
    """Align primary images and apply declared additional-image shifts."""
    images = _image_payloads(image)
    first_image, second_image = images[:2]
    masks = _image_masks(image, images)
    metadata = _image_metadata(image, len(images))
    additional_modes = _additional_alignment_modes(
        additional_alignment_modes,
        additional_count=len(images) - 2,
    )
    row_offset, column_offset = _translation_offset(
        first_image,
        second_image,
        method=method,
        first_mask=masks[0],
        second_mask=masks[1],
        alignment_backend_provider=alignment_backend_provider,
    )
    normalized_crop_mode = AlignCropMode.from_literal(crop_mode)
    offsets, shapes = _adjust_offsets(
        ((0, 0), (row_offset, column_offset)),
        (first_image.shape[:2], second_image.shape[:2]),
        normalized_crop_mode,
    )
    outputs = list(
        _crop_mode_outputs(
            first_image,
            second_image,
            first_mask=masks[0],
            second_mask=masks[1],
            first_metadata=metadata[0],
            second_metadata=metadata[1],
            offsets=offsets,
            shapes=shapes,
        )
    )
    additional_measurements: list[AlignShiftMeasurement] = []
    for output_index, (additional_image, additional_mask, additional_metadata, mode) in enumerate(
        zip(images[2:], masks[2:], metadata[2:], additional_modes, strict=True),
        start=2,
    ):
        if mode is not AlignAdditionalMode.SIMILARLY:
            raise ValueError(f"Unsupported Align additional-image mode {mode.value!r}.")
        additional_offset, additional_shape = _similarly_aligned_output_geometry(
            additional_image,
            second_image,
            second_offset=offsets[1],
            second_shape=shapes[1],
            crop_mode=normalized_crop_mode,
        )
        outputs.append(
            _apply_alignment(
                additional_image,
                additional_mask,
                additional_metadata,
                additional_offset,
                additional_shape,
            )
        )
        additional_measurements.append(
            AlignShiftMeasurement(
                slice_index=0,
                output_index=output_index,
                x_shift=float(-additional_offset[1]),
                y_shift=float(-additional_offset[0]),
            )
        )
    measurements = (
        AlignShiftMeasurement(
            slice_index=0,
            output_index=0,
            x_shift=float(-offsets[0][1]),
            y_shift=float(-offsets[0][0]),
        ),
        AlignShiftMeasurement(
            slice_index=0,
            output_index=1,
            x_shift=float(-offsets[1][1]),
            y_shift=float(-offsets[1][0]),
        ),
        *additional_measurements,
    )
    return (
        *outputs,
        measurements,
    )


def _image_payloads(image: np.ndarray) -> tuple[np.ndarray, ...]:
    data = np.asarray(image_payload_data(image))
    if not hasattr(data, "ndim") or data.ndim not in (3, 4) or data.shape[0] < 2:
        raise ValueError("Align requires at least two stacked image inputs.")
    return tuple(data[index] for index in range(data.shape[0]))


def _two_image_payload(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    payloads = _image_payloads(image)
    if len(payloads) != 2:
        raise ValueError("Align requires exactly two stacked image inputs.")
    return payloads[0], payloads[1]


def _image_masks(
    image: np.ndarray,
    images: tuple[np.ndarray, ...],
) -> tuple[np.ndarray | None, ...]:
    mask = image_payload_mask(image)
    if mask is None:
        return (None,) * len(images)
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.ndim == 2:
        return (mask_array,) * len(images)
    if mask_array.ndim == 3 and mask_array.shape[0] == len(images):
        return tuple(mask_array[index] for index in range(len(images)))
    spatial_shapes = {tuple(np.asarray(input_image).shape[:2]) for input_image in images}
    if len(spatial_shapes) == 1 and mask_array.shape[:2] == next(iter(spatial_shapes)):
        return (mask_array,) * len(images)
    raise ValueError("Align mask must be shared 2D mask or one mask per stacked image.")


def _two_image_masks(
    image: np.ndarray,
    first_image: np.ndarray,
    second_image: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    masks = _image_masks(image, (first_image, second_image))
    if len(masks) != 2:
        raise ValueError("Align requires exactly two stacked image inputs.")
    return masks[0], masks[1]


def _image_metadata(
    image: np.ndarray,
    count: int,
) -> tuple[ImagePayloadMetadata, ...]:
    metadata = image_payload_metadata(image)
    return tuple(metadata.for_channel(index) for index in range(count))


def _additional_alignment_modes(
    modes: tuple[AlignAdditionalMode | str, ...],
    *,
    additional_count: int,
) -> tuple[AlignAdditionalMode, ...]:
    if additional_count == 0:
        if modes:
            raise ValueError("Align got additional alignment modes without extra images.")
        return ()
    if not modes:
        return (AlignAdditionalMode.SIMILARLY,) * additional_count
    normalized = tuple(AlignAdditionalMode.from_literal(mode) for mode in modes)
    if len(normalized) != additional_count:
        raise ValueError(
            "Align additional alignment mode count must match additional image count; "
            f"got {len(normalized)} modes for {additional_count} images."
        )
    return normalized


def _similarly_aligned_output_geometry(
    additional_image: np.ndarray,
    second_image: np.ndarray,
    *,
    second_offset: tuple[int, int],
    second_shape: tuple[int, int],
    crop_mode: AlignCropMode,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if crop_mode is AlignCropMode.KEEP_SIZE:
        return second_offset, tuple(np.asarray(additional_image).shape[:2])
    if tuple(np.asarray(additional_image).shape[:2]) != tuple(
        np.asarray(second_image).shape[:2]
    ):
        raise ValueError(
            "Align additional images with non-keep-size crop modes must share "
            "the second input image spatial shape."
        )
    return second_offset, second_shape


def _two_image_metadata(
    image: np.ndarray,
) -> tuple[ImagePayloadMetadata, ImagePayloadMetadata]:
    metadata = image_payload_metadata(image)
    return metadata.for_channel(0), metadata.for_channel(1)


def _translation_offset(
    reference_image: np.ndarray,
    moving_image: np.ndarray,
    *,
    method: str,
    first_mask: np.ndarray | None,
    second_mask: np.ndarray | None,
    alignment_backend_provider: CellProfilerBackendProvider | None,
) -> tuple[int, int]:
    """Return integer row/column offsets in CellProfiler's native convention."""
    reference_pixels = _alignment_pixels(reference_image)
    moving_pixels = _alignment_pixels(moving_image)
    if method.strip().lower() == "normalized cross correlation":
        column_offset, row_offset = _cross_correlation_offset(
            reference_pixels,
            moving_pixels,
        )
    else:
        alignment_backend = AlignmentBackendStrategy.for_memory_type(
            backend_provider=alignment_backend_provider,
        )
        column_offset, row_offset = alignment_backend.mutual_information_offset(
            reference_pixels,
            moving_pixels,
            _alignment_mask(first_mask, reference_pixels.shape),
            _alignment_mask(second_mask, moving_pixels.shape),
        )
    return int(row_offset), int(column_offset)


def _alignment_pixels(image: np.ndarray) -> np.ndarray:
    pixels = np.asarray(image, dtype=float)
    if pixels.ndim == 3:
        return np.mean(pixels, axis=2)
    return pixels


def _alignment_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    return np.asarray(mask, dtype=bool)


def _cross_correlation_offset(
    reference_pixels: np.ndarray,
    moving_pixels: np.ndarray,
) -> tuple[int, int]:
    shape = np.maximum(reference_pixels.shape, moving_pixels.shape)
    fft_shape = shape * 2
    row_grid, column_grid = np.mgrid[-shape[0] : shape[0], -shape[1] : shape[1]]
    overlap_count = np.abs(row_grid * column_grid).astype(float)
    overlap_count[overlap_count < 1] = 1

    reference_pixels = reference_pixels - np.mean(reference_pixels)
    moving_pixels = moving_pixels - np.mean(moving_pixels)
    reference_fft = fft2(reference_pixels, fft_shape.tolist())
    moving_fft = fft2(moving_pixels, fft_shape.tolist())
    correlation = ifft2(reference_fft * moving_fft.conj()).real

    ref_rows, ref_columns = reference_pixels.shape
    ref_sum = np.zeros(fft_shape)
    ref_sum[:ref_rows, :ref_columns] = _cumsum_quadrant(
        reference_pixels, row_forwards=False, column_forwards=False
    )
    ref_sum[:ref_rows, -ref_columns:] = _cumsum_quadrant(
        reference_pixels, row_forwards=False, column_forwards=True
    )
    ref_sum[-ref_rows:, :ref_columns] = _cumsum_quadrant(
        reference_pixels, row_forwards=True, column_forwards=False
    )
    ref_sum[-ref_rows:, -ref_columns:] = _cumsum_quadrant(
        reference_pixels, row_forwards=True, column_forwards=True
    )
    ref_mean = ref_sum / overlap_count

    moving_rows, moving_columns = moving_pixels.shape
    moving_sum = np.zeros(fft_shape)
    moving_sum[:moving_rows, :moving_columns] = _cumsum_quadrant(
        moving_pixels, row_forwards=False, column_forwards=False
    )
    moving_sum[:moving_rows, -moving_columns:] = _cumsum_quadrant(
        moving_pixels, row_forwards=False, column_forwards=True
    )
    moving_sum[-moving_rows:, :moving_columns] = _cumsum_quadrant(
        moving_pixels, row_forwards=True, column_forwards=False
    )
    moving_sum[-moving_rows:, -moving_columns:] = _cumsum_quadrant(
        moving_pixels, row_forwards=True, column_forwards=True
    )
    moving_mean = np.fliplr(np.flipud(moving_sum)) / overlap_count

    ref_sd = np.sum(reference_pixels ** 2) - ref_mean ** 2 * np.prod(shape)
    moving_sd = np.sum(moving_pixels ** 2) - moving_mean ** 2 * np.prod(shape)
    sd = np.sqrt(np.maximum(ref_sd * moving_sd, 0))
    normalized = np.divide(correlation, sd, out=np.zeros_like(correlation), where=sd != 0)
    normalized[(overlap_count < np.prod(shape) / 2) & (sd < np.mean(sd) / 100)] = 0

    row_offset, column_offset = np.unravel_index(np.argmax(normalized), fft_shape)
    if row_offset > reference_pixels.shape[0]:
        row_offset -= int(fft_shape[0])
    if column_offset > reference_pixels.shape[1]:
        column_offset -= int(fft_shape[1])
    return int(column_offset), int(row_offset)


def _mutual_information_offset_python(
    reference_pixels: np.ndarray,
    moving_pixels: np.ndarray,
    reference_mask: np.ndarray,
    moving_mask: np.ndarray,
) -> tuple[int, int]:
    """Readable reference implementation of legacy MI offset search."""
    best = _mutual_information(
        reference_pixels,
        moving_pixels,
        reference_mask,
        moving_mask,
    )
    row_offset = 0
    column_offset = 0
    while True:
        previous_row_offset = row_offset
        previous_column_offset = column_offset
        for candidate_row in range(previous_row_offset - 1, previous_row_offset + 2):
            for candidate_column in range(
                previous_column_offset - 1,
                previous_column_offset + 2,
            ):
                if candidate_row == 0 and candidate_column == 0:
                    continue
                moving_slice, reference_slice = _offset_slice(
                    moving_pixels,
                    reference_pixels,
                    candidate_row,
                    candidate_column,
                )
                moving_mask_slice, reference_mask_slice = _offset_slice(
                    moving_mask,
                    reference_mask,
                    candidate_row,
                    candidate_column,
                )
                information = _mutual_information(
                    reference_slice,
                    moving_slice,
                    reference_mask_slice,
                    moving_mask_slice,
                )
                if information > best:
                    best = information
                    row_offset = candidate_row
                    column_offset = candidate_column
        if row_offset == previous_row_offset and column_offset == previous_column_offset:
            return int(column_offset), int(row_offset)


def _mutual_information(
    reference_pixels: np.ndarray,
    moving_pixels: np.ndarray,
    reference_mask: np.ndarray,
    moving_mask: np.ndarray,
) -> float:
    mask = reference_mask & moving_mask
    reference_values = reference_pixels[mask]
    moving_values = moving_pixels[mask]
    return (
        _entropy(reference_values)
        + _entropy(moving_values)
        - _joint_entropy(reference_values, moving_values)
    )


def _entropy(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    histogram = scind.histogram(
        values.astype(float),
        float(np.min(values)),
        float(np.max(values)),
        256,
    )
    count = np.sum(histogram)
    if count <= 0 or np.max(histogram) <= 0:
        return 0.0
    nonzero = histogram[histogram != 0]
    return float(np.log2(count) - np.sum(nonzero * np.log2(nonzero)) / count)


def _joint_entropy(x_values: np.ndarray, y_values: np.ndarray) -> float:
    if x_values.size == 0 or y_values.size == 0:
        return 0.0
    x_bins = (_stretch_to_unit_interval(x_values) * 255).astype(int)
    y_bins = (_stretch_to_unit_interval(y_values) * 255).astype(int)
    paired_bins = (256 * x_bins + y_bins).flatten()
    histogram = scipy.sparse.coo_matrix(
        (
            np.ones(paired_bins.shape, dtype=np.int32),
            (paired_bins, np.zeros(paired_bins.shape, dtype=np.int32)),
        )
    ).toarray()
    count = np.sum(histogram)
    if count <= 0 or np.max(histogram) <= 0:
        return 0.0
    nonzero = histogram[histogram > 0]
    return float(np.log2(count) - np.sum(nonzero * np.log2(nonzero)) / count)


def _stretch_to_unit_interval(values: np.ndarray) -> np.ndarray:
    """Linearly stretch finite values to [0, 1] for MI histogram binning."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum <= minimum:
        return np.zeros(array.shape, dtype=float)
    return (array - minimum) / (maximum - minimum)


def _cumsum_quadrant(
    values: np.ndarray,
    *,
    row_forwards: bool,
    column_forwards: bool,
) -> np.ndarray:
    if row_forwards:
        values = values.cumsum(0)
    else:
        values = np.flipud(np.flipud(values).cumsum(0))
    if column_forwards:
        return values.cumsum(1)
    return np.fliplr(np.fliplr(values).cumsum(1))


def _prepare_align() -> None:
    """Compile alignment backend kernels outside measured execution."""
    reference = np.zeros((32, 32), dtype=np.float32)
    moving = np.zeros((32, 32), dtype=np.float32)
    reference[8:20, 9:21] = 1.0
    moving[9:21, 8:20] = 1.0
    _translation_offset(
        reference,
        moving,
        method="Mutual Information",
        first_mask=None,
        second_mask=None,
        alignment_backend_provider=None,
    )


def _crop_mode_outputs(
    first_image: np.ndarray,
    second_image: np.ndarray,
    *,
    first_mask: np.ndarray | None,
    second_mask: np.ndarray | None,
    first_metadata: ImagePayloadMetadata,
    second_metadata: ImagePayloadMetadata,
    offsets: tuple[tuple[int, int], tuple[int, int]],
    shapes: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        _apply_alignment(
            first_image,
            first_mask,
            first_metadata,
            offsets[0],
            shapes[0],
        ),
        _apply_alignment(
            second_image,
            second_mask,
            second_metadata,
            offsets[1],
            shapes[1],
        ),
    )


class AlignCropModeStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal strategy family for legacy Align crop modes."""

    __registry_key__ = "crop_mode_label"
    __skip_if_no_key__ = True
    crop_mode_label: ClassVar[str | None] = None
    crop_mode: ClassVar[AlignCropMode | None] = None

    @classmethod
    def for_crop_mode(cls, crop_mode: AlignCropMode) -> "AlignCropModeStrategy":
        return cls.__registry__[crop_mode.value]()

    @abstractmethod
    def apply(self, request: AlignCropRequest) -> tuple[
        tuple[tuple[int, int], tuple[int, int]],
        tuple[tuple[int, int], tuple[int, int]],
    ]:
        """Return first/second image outputs for one crop mode."""


class KeepSizeAlignCropModeStrategy(AlignCropModeStrategy):
    """Keep aligned images in their original shape."""

    crop_mode = AlignCropMode.KEEP_SIZE
    crop_mode_label = crop_mode.value

    def apply(self, request: AlignCropRequest) -> tuple[
        tuple[tuple[int, int], tuple[int, int]],
        tuple[tuple[int, int], tuple[int, int]],
    ]:
        return request.offsets, request.shapes


class PadImagesAlignCropModeStrategy(AlignCropModeStrategy):
    """Pad both images to preserve all shifted content."""

    crop_mode = AlignCropMode.PAD_IMAGES
    crop_mode_label = crop_mode.value

    def apply(self, request: AlignCropRequest) -> tuple[
        tuple[tuple[int, int], tuple[int, int]],
        tuple[tuple[int, int], tuple[int, int]],
    ]:
        return _adjust_offsets_for_padding(request.offsets, request.shapes)


class CropToOverlapAlignCropModeStrategy(AlignCropModeStrategy):
    """Crop both images to the overlapping aligned region."""

    crop_mode = AlignCropMode.CROP_TO_ALIGNED_REGION
    crop_mode_label = crop_mode.value

    def apply(self, request: AlignCropRequest) -> tuple[
        tuple[tuple[int, int], tuple[int, int]],
        tuple[tuple[int, int], tuple[int, int]],
    ]:
        return _adjust_offsets_for_cropping(request.offsets, request.shapes)


def _adjust_offsets(
    offsets: tuple[tuple[int, int], tuple[int, int]],
    shapes: tuple[tuple[int, int], tuple[int, int]],
    crop_mode: AlignCropMode,
) -> tuple[
    tuple[tuple[int, int], tuple[int, int]],
    tuple[tuple[int, int], tuple[int, int]],
]:
    return AlignCropModeStrategy.for_crop_mode(crop_mode).apply(
        AlignCropRequest(
            first_image=np.empty(shapes[0]),
            second_image=np.empty(shapes[1]),
            first_mask=None,
            second_mask=None,
            offsets=offsets,
            shapes=shapes,
        )
    )


def _adjust_offsets_for_cropping(
    offsets: tuple[tuple[int, int], tuple[int, int]],
    shapes: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[
    tuple[tuple[int, int], tuple[int, int]],
    tuple[tuple[int, int], tuple[int, int]],
]:
    offsets_array = np.asarray(offsets, dtype=int)
    shapes_array = np.asarray(shapes, dtype=int)
    offsets_array = offsets_array - np.max(offsets_array, axis=0)[np.newaxis, :]
    shapes_array = shapes_array + offsets_array
    output_shape = np.min(shapes_array, axis=0)
    return _offsets_and_shapes_tuple(
        offsets_array,
        np.tile(output_shape, (len(shapes), 1)),
    )


def _adjust_offsets_for_padding(
    offsets: tuple[tuple[int, int], tuple[int, int]],
    shapes: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[
    tuple[tuple[int, int], tuple[int, int]],
    tuple[tuple[int, int], tuple[int, int]],
]:
    offsets_array = np.asarray(offsets, dtype=int)
    shapes_array = np.asarray(shapes, dtype=int)
    offsets_array = offsets_array - np.min(offsets_array, axis=0)[np.newaxis, :]
    shapes_array = shapes_array + offsets_array
    output_shape = np.max(shapes_array, axis=0)
    return _offsets_and_shapes_tuple(
        offsets_array,
        np.tile(output_shape, (len(shapes), 1)),
    )


def _offsets_and_shapes_tuple(
    offsets: np.ndarray,
    shapes: np.ndarray,
) -> tuple[
    tuple[tuple[int, int], tuple[int, int]],
    tuple[tuple[int, int], tuple[int, int]],
]:
    return (
        tuple(tuple(int(value) for value in row) for row in offsets),
        tuple(tuple(int(value) for value in row) for row in shapes),
    )


def _apply_alignment(
    image: np.ndarray,
    mask: np.ndarray | None,
    metadata: ImagePayloadMetadata,
    offset: tuple[int, int],
    shape: tuple[int, int],
) -> np.ndarray | MaskedImagePayload:
    output_shape = tuple(shape) + tuple(np.asarray(image).shape[2:])
    output = np.zeros(output_shape, dtype=np.asarray(image).dtype)
    source_view, output_view = _offset_slice(np.asarray(image), output, *offset)
    output_view[...] = source_view

    source_mask = (
        np.ones(np.asarray(image).shape[:2], dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    output_mask = np.zeros(tuple(shape), dtype=bool)
    source_mask_view, output_mask_view = _offset_slice(source_mask, output_mask, *offset)
    output_mask_view[...] = source_mask_view
    return image_payload_with_context(
        output,
        mask=None if np.all(output_mask) else output_mask,
        metadata=metadata,
    )


def _offset_slice(
    source: np.ndarray,
    target: np.ndarray,
    row_offset: int,
    column_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    if row_offset < 0:
        height = min(source.shape[0] + row_offset, target.shape[0])
        source_row_start = -row_offset
        target_row_start = 0
    else:
        height = min(source.shape[0], target.shape[0] - row_offset)
        source_row_start = 0
        target_row_start = row_offset
    if column_offset < 0:
        width = min(source.shape[1] + column_offset, target.shape[1])
        source_column_start = -column_offset
        target_column_start = 0
    else:
        width = min(source.shape[1], target.shape[1] - column_offset)
        source_column_start = 0
        target_column_start = column_offset

    if height <= 0 or width <= 0:
        empty = (slice(0, 0), slice(0, 0))
        return source[empty], target[empty]

    source_slices = (
        slice(source_row_start, source_row_start + height),
        slice(source_column_start, source_column_start + width),
        *(slice(None),) * max(0, source.ndim - 2),
    )
    target_slices = (
        slice(target_row_start, target_row_start + height),
        slice(target_column_start, target_column_start + width),
        *(slice(None),) * max(0, target.ndim - 2),
    )
    return source[source_slices], target[target_slices]


align.__openhcs_prepare__ = _prepare_align
