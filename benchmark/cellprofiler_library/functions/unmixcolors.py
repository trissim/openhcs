"""Converted from CellProfiler: UnmixColors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum


class StainType(Enum):
    """Closed family of CellProfiler UnmixColors stain choices."""

    HEMATOXYLIN = ("Hematoxylin", (0.644, 0.717, 0.267))
    EOSIN = ("Eosin", (0.093, 0.954, 0.283))
    DAB = ("DAB", (0.268, 0.570, 0.776))
    FAST_RED = ("Fast red", (0.214, 0.851, 0.478))
    FAST_BLUE = ("Fast blue", (0.749, 0.606, 0.267))
    METHYL_BLUE = ("Methyl blue", (0.799, 0.591, 0.105))
    METHYL_GREEN = ("Methyl green", (0.980, 0.144, 0.133))
    AEC = ("AEC", (0.274, 0.679, 0.680))
    ANILINE_BLUE = ("Aniline blue", (0.853, 0.509, 0.113))
    AZOCARMINE = ("Azocarmine", (0.071, 0.977, 0.198))
    ALCIAN_BLUE = ("Alcian blue", (0.875, 0.458, 0.158))
    PAS = ("PAS", (0.175, 0.972, 0.155))
    HEMATOXYLIN_AND_PAS = ("Hematoxylin and PAS", (0.553, 0.754, 0.354))
    FEULGEN = ("Feulgen", (0.464, 0.830, 0.308))
    METHYLENE_BLUE = ("Methylene blue", (0.553, 0.754, 0.354))
    ORANGE_G = ("Orange-G", (0.107, 0.368, 0.923))
    PONCEAU_FUCHSIN = ("Ponceau-fuchsin", (0.100, 0.737, 0.668))
    CUSTOM = ("Custom", None)

    @property
    def display_name(self) -> str:
        return self.value[0]

    @property
    def calibrated_absorbance(self) -> tuple[float, float, float]:
        absorbance = self.value[1]
        if absorbance is None:
            raise ValueError("Custom stains require explicit absorbance values.")
        return absorbance


@dataclass(frozen=True, slots=True)
class StainDefinition:
    """One stain row participating in CellProfiler color deconvolution."""

    stain: StainType
    custom_absorbance: tuple[float, float, float] | None = None

    @property
    def absorbance(self) -> np.ndarray:
        if self.stain is StainType.CUSTOM:
            if self.custom_absorbance is None:
                raise ValueError("Custom UnmixColors rows require absorbance values.")
            absorbance = self.custom_absorbance
        else:
            absorbance = self.stain.calibrated_absorbance
        return _normalized_absorbance(absorbance)


@numpy(contract=ProcessingContract.FLEXIBLE)
def unmix_colors(
    image: np.ndarray,
    stain_names: Sequence[StainType | str] = (),
    custom_absorbances: Sequence[Sequence[float] | None] = (),
    stain1: StainType | str = StainType.HEMATOXYLIN,
    stain2: StainType | str = StainType.EOSIN,
    stain3: StainType | str | None = None,
    output_stain_index: int = 0,
    custom_red_absorbance_1: float = 0.5,
    custom_green_absorbance_1: float = 0.5,
    custom_blue_absorbance_1: float = 0.5,
    custom_red_absorbance_2: float = 0.5,
    custom_green_absorbance_2: float = 0.5,
    custom_blue_absorbance_2: float = 0.5,
    custom_red_absorbance_3: float = 0.5,
    custom_green_absorbance_3: float = 0.5,
    custom_blue_absorbance_3: float = 0.5,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Unmix one RGB image into one image per configured CellProfiler stain row.

    CellProfiler Parameter Mapping:
        'Select the input color image' -> (pipeline-handled)
        'Color image' -> (pipeline-handled)
        'Name the output image' -> (pipeline-handled)
        'Image name' -> (pipeline-handled)
        'Stain' -> stain_names
        'Red absorbance' -> custom_absorbances
        'Green absorbance' -> custom_absorbances
        'Blue absorbance' -> custom_absorbances
    """
    rgb_image = _as_rgb_image(image)
    if stain_names:
        return _unmix_stain_outputs(
            rgb_image,
            _stain_definitions(stain_names, custom_absorbances),
        )

    definitions = _legacy_stain_definitions(
        stain1=stain1,
        stain2=stain2,
        stain3=stain3,
        custom_absorbances=(
            (
                custom_red_absorbance_1,
                custom_green_absorbance_1,
                custom_blue_absorbance_1,
            ),
            (
                custom_red_absorbance_2,
                custom_green_absorbance_2,
                custom_blue_absorbance_2,
            ),
            (
                custom_red_absorbance_3,
                custom_green_absorbance_3,
                custom_blue_absorbance_3,
            ),
        ),
    )
    outputs = _unmix_stain_outputs(rgb_image, definitions)
    if output_stain_index < 0 or output_stain_index >= len(outputs):
        raise ValueError(
            f"output_stain_index must be in [0, {len(outputs) - 1}], "
            f"got {output_stain_index}."
        )
    return outputs[output_stain_index]


def _stain_definitions(
    stain_names: Sequence[StainType | str],
    custom_absorbances: Sequence[Sequence[float] | None],
) -> tuple[StainDefinition, ...]:
    if len(stain_names) != len(custom_absorbances):
        raise ValueError(
            "UnmixColors stain_names and custom_absorbances must have the "
            "same length."
        )
    return tuple(
        StainDefinition(
            stain=coerce_cellprofiler_enum(StainType, stain_name),
            custom_absorbance=_coerce_custom_absorbance(custom_absorbance),
        )
        for stain_name, custom_absorbance in zip(
            stain_names,
            custom_absorbances,
            strict=True,
        )
    )


def _legacy_stain_definitions(
    *,
    stain1: StainType | str,
    stain2: StainType | str,
    stain3: StainType | str | None,
    custom_absorbances: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
) -> tuple[StainDefinition, ...]:
    stains = (stain1, stain2, stain3)
    return tuple(
        StainDefinition(
            stain=coerce_cellprofiler_enum(StainType, stain),
            custom_absorbance=custom_absorbances[index],
        )
        for index, stain in enumerate(stains)
        if stain is not None
    )


def _unmix_stain_outputs(
    image: np.ndarray,
    definitions: tuple[StainDefinition, ...],
) -> tuple[np.ndarray, ...]:
    if not definitions:
        raise ValueError("UnmixColors requires at least one stain definition.")
    inverse_matrix = np.linalg.pinv(
        np.asarray([definition.absorbance for definition in definitions])
    )
    return tuple(
        _run_unmix_output(image, inverse_matrix[:, index])
        for index in range(len(definitions))
    )


def _run_unmix_output(
    image: np.ndarray,
    inverse_absorbances: np.ndarray,
) -> np.ndarray:
    eps = 1.0 / 256.0 / 2.0
    log_image = np.log(image + eps)
    broadcast_shape = (1,) * (log_image.ndim - 1) + (3,)
    scaled_image = log_image * inverse_absorbances.reshape(broadcast_shape)
    result = np.exp(np.sum(scaled_image, axis=-1)) - eps
    return (1.0 - np.clip(result, 0.0, 1.0)).astype(np.float32)


def _as_rgb_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        return np.stack((array, array, array), axis=-1)
    if array.ndim >= 3 and array.shape[-1] == 3:
        return array
    if array.ndim == 3 and array.shape[0] == 3:
        return np.moveaxis(array, 0, -1)
    if array.ndim >= 4 and array.shape[1] == 3:
        return np.moveaxis(array, 1, -1)
    raise ValueError(
        "UnmixColors expects an RGB image with three color channels on the "
        f"first or last channel axis, got shape {array.shape}."
    )


def _normalized_absorbance(absorbance: Sequence[float]) -> np.ndarray:
    vector = np.asarray(tuple(float(channel) for channel in absorbance))
    if vector.shape != (3,):
        raise ValueError(
            f"UnmixColors absorbance vectors must have three channels, got {vector}."
        )
    norm = np.sqrt(np.sum(vector**2))
    if norm <= 0:
        raise ValueError("UnmixColors absorbance vectors cannot be zero.")
    return vector / norm


def _coerce_custom_absorbance(
    absorbance: Sequence[float] | None,
) -> tuple[float, float, float] | None:
    if absorbance is None:
        return None
    red, green, blue = absorbance
    return float(red), float(green), float(blue)
