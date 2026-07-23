"""Production authority for bounded synthetic plate generation profiles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SyntheticPlateFormat(str, Enum):
    """Synthetic plate filename/metadata formats supported by the generator."""

    IMAGE_XPRESS = "ImageXpress"
    OPERA_PHENIX = "OperaPhenix"


@dataclass(frozen=True, slots=True)
class SyntheticPlateGenerationParameters:
    """Concrete parameters accepted by the synthetic microscopy generator."""

    grid_rows: int = 2
    grid_cols: int = 2
    tile_width: int = 128
    tile_height: int = 128
    overlap_percent: int = 10
    stage_error_px: int = 2
    wavelengths: int = 2
    z_stack_levels: int = 1
    num_cells: int = 80
    shared_cell_fraction: float = 0.95
    wells: tuple[str, ...] = ("A01",)
    format: SyntheticPlateFormat = SyntheticPlateFormat.IMAGE_XPRESS
    openhcs_format: bool = False
    include_all_components: bool = True
    random_seed: int | None = 1
    sample_file_limit: int = 20


@dataclass(frozen=True, slots=True)
class SyntheticPlateGenerationBounds:
    """Validation bounds for synthetic microscopy generator requests."""

    max_grid_rows: int = 8
    max_grid_cols: int = 8
    max_tile_width: int = 2048
    max_tile_height: int = 2048
    max_overlap_percent: int = 80
    max_stage_error_px: int = 128
    max_wavelengths: int = 8
    max_z_stack_levels: int = 16
    max_wells: int = 96
    max_num_cells: int = 10_000
    max_sample_file_limit: int = 10_000


@dataclass(frozen=True, slots=True)
class SyntheticPlateGenerationProfile:
    """Supported synthetic generation profile shared by agent and transport."""

    default_request: SyntheticPlateGenerationParameters
    bounds: SyntheticPlateGenerationBounds
    supported_formats: tuple[SyntheticPlateFormat, ...]

    @classmethod
    def default(cls) -> "SyntheticPlateGenerationProfile":
        return cls(
            default_request=SyntheticPlateGenerationParameters(),
            bounds=SyntheticPlateGenerationBounds(),
            supported_formats=tuple(SyntheticPlateFormat),
        )

    def format_from_value(
        self,
        value: SyntheticPlateFormat | str,
    ) -> SyntheticPlateFormat:
        synthetic_format = SyntheticPlateFormat(value)
        if synthetic_format not in self.supported_formats:
            raise ValueError(
                f"Unsupported synthetic plate format: {synthetic_format.value!r}."
            )
        return synthetic_format

    def validate(self, request: SyntheticPlateGenerationParameters) -> None:
        self._bounded_int(
            "grid_rows",
            request.grid_rows,
            minimum=1,
            maximum=self.bounds.max_grid_rows,
        )
        self._bounded_int(
            "grid_cols",
            request.grid_cols,
            minimum=1,
            maximum=self.bounds.max_grid_cols,
        )
        self._bounded_int(
            "tile_width",
            request.tile_width,
            minimum=1,
            maximum=self.bounds.max_tile_width,
        )
        self._bounded_int(
            "tile_height",
            request.tile_height,
            minimum=1,
            maximum=self.bounds.max_tile_height,
        )
        self._bounded_int(
            "overlap_percent",
            request.overlap_percent,
            minimum=0,
            maximum=self.bounds.max_overlap_percent,
        )
        self._bounded_int(
            "stage_error_px",
            request.stage_error_px,
            minimum=0,
            maximum=self.bounds.max_stage_error_px,
        )
        self._bounded_int(
            "wavelengths",
            request.wavelengths,
            minimum=1,
            maximum=self.bounds.max_wavelengths,
        )
        self._bounded_int(
            "z_stack_levels",
            request.z_stack_levels,
            minimum=1,
            maximum=self.bounds.max_z_stack_levels,
        )
        self._bounded_int(
            "num_cells",
            request.num_cells,
            minimum=0,
            maximum=self.bounds.max_num_cells,
        )
        self._bounded_float(
            "shared_cell_fraction",
            request.shared_cell_fraction,
            minimum=0.0,
            maximum=1.0,
        )
        self._bounded_int(
            "sample_file_limit",
            request.sample_file_limit,
            minimum=0,
            maximum=self.bounds.max_sample_file_limit,
        )
        if not request.wells:
            raise ValueError("wells must contain at least one well.")
        if len(request.wells) > self.bounds.max_wells:
            raise ValueError(
                f"wells must not contain more than {self.bounds.max_wells} entries."
            )
        for well in request.wells:
            if not well:
                raise ValueError("wells must not contain empty values.")
        self.format_from_value(request.format)

    @staticmethod
    def _bounded_int(
        name: str,
        value: int,
        *,
        minimum: int,
        maximum: int,
    ) -> None:
        if isinstance(value, bool) or value < minimum or value > maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}.")

    @staticmethod
    def _bounded_float(
        name: str,
        value: float,
        *,
        minimum: float,
        maximum: float,
    ) -> None:
        if isinstance(value, bool) or value < minimum or value > maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}.")


SYNTHETIC_PLATE_GENERATION_PROFILE = SyntheticPlateGenerationProfile.default()
