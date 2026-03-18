"""
File I/O operations for loading and saving projections.

This module provides functions to:
- Load Z-stacks from files
- Build projection filenames using FilenameParser API
- Save projections to disk
- Create Z-stack movies (XY, XZ, YZ slice animations)

Invariants:
- I/O is isolated - can be mocked, replaced, or parallelized
- All outputs are tracked via immutable records
- Functions are idempotent (same inputs → same outputs/files)
- Naming delegated to FilenameParser API - no custom naming logic
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tifffile

from .discovery import WellChannelKey


@dataclass(frozen=True)
class ProjectionOutput:
    """Immutable record of saved projection."""

    projection_type: str
    output_path: Path
    well_id: str
    channel_id: str


@dataclass(frozen=True)
class MovieOutput:
    """Immutable record of saved movie."""

    movie_type: str
    output_path: Path
    well_id: str
    channel_id: str


def load_z_stack(file_paths: Tuple[Path, ...]) -> np.ndarray:
    """
    Load multiple Z-slice files into a 3D stack.

    Args:
        file_paths: Ordered tuple of paths (Z-order preserved)

    Returns:
        3D NumPy array of shape (Z, Y, X)

    Raises:
        FileNotFoundError: If any file is missing
    """
    if not file_paths:
        raise ValueError("No file paths provided")

    stacks = []
    for path in file_paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        img = tifffile.imread(str(path))

        if img.ndim == 2:
            stacks.append(img)
        elif img.ndim == 3:
            for z in range(img.shape[0]):
                stacks.append(img[z])
        else:
            raise ValueError(f"Unexpected image dimensions: {img.ndim}")

    if not stacks:
        raise ValueError("No valid images loaded")

    stack = np.stack(stacks, axis=0)
    stack = np.flip(stack, axis=0)
    return stack


def build_projection_filename(
    well_key: WellChannelKey, projection_type: str, extension: str = ".tiff"
) -> str:
    """
    Build output filename using well/channel info.

    Args:
        well_key: WellChannelKey with well_id and channel_id
        projection_type: "xy", "xz", or "yz"
        extension: File extension

    Returns:
        Filename string like "r01c06_ch1_XY_max.tiff"
    """
    proj_upper = projection_type.upper()
    filename = f"{well_key.well_id}_ch{well_key.channel_id}_{proj_upper}_max{extension}"
    return filename


def save_projection(
    projection: np.ndarray,
    projection_type: str,
    well_key: WellChannelKey,
    output_dir: Path,
    output_format: str = "tiff",
) -> ProjectionOutput:
    """
    Save a single projection to disk.

    Args:
        projection: 2D NumPy array
        projection_type: "xy", "xz", or "yz"
        well_key: WellChannelKey for naming
        output_dir: Directory to save to
        output_format: File format ("tiff", "png")

    Returns:
        ProjectionOutput record
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extension = (
        f".{output_format}" if not output_format.startswith(".") else output_format
    )
    filename = build_projection_filename(well_key, projection_type, extension)
    output_path = output_dir / filename

    if output_format.lower() in ("tiff", "tif"):
        tifffile.imwrite(str(output_path), projection)
    else:
        from PIL import Image

        img = Image.fromarray(projection)
        img.save(str(output_path))

    return ProjectionOutput(
        projection_type=projection_type,
        output_path=output_path,
        well_id=well_key.well_id,
        channel_id=well_key.channel_id,
    )


def save_all_projections(
    projections: dict,
    well_key: WellChannelKey,
    output_dir: Path,
    output_format: str = "tiff",
) -> Tuple[ProjectionOutput, ...]:
    """
    Save all projections for a well/channel.

    Args:
        projections: Dict {"xy": arr, "xz": arr, "yz": arr}
        well_key: WellChannelKey for naming
        output_dir: Directory to save to
        output_format: File format

    Returns:
        Tuple of ProjectionOutput records
    """
    outputs = []
    for proj_type, proj_data in projections.items():
        output = save_projection(
            proj_data, proj_type, well_key, output_dir, output_format
        )
        outputs.append(output)

    return tuple(outputs)


def create_slice_movie(
    z_stack: np.ndarray,
    output_path: Path,
    slice_type: str = "xy",
    fps: int = 10,
    global_max: float = None,
) -> Path:
    """
    Create a movie going through slices of a Z-stack.

    Args:
        z_stack: 3D array of shape (Z, Y, X)
        output_path: Path to save movie (should end in .mp4 or .gif)
        slice_type: "xy" (go through Z), "xz" (go through Y), "yz" (go through X)
        fps: Frames per second
        global_max: Optional max value for normalization (use same for all movies)

    Returns:
        Path to saved movie
    """
    import imageio.v3 as iio

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if global_max is None:
        global_max = z_stack.max()
    if global_max == 0:
        global_max = 1

    frames = []

    if slice_type == "xy":
        for z in range(z_stack.shape[0]):
            frame = z_stack[z, :, :]
            frame_norm = (frame / global_max * 255).astype(np.uint8)
            frames.append(frame_norm)
    elif slice_type == "xz":
        for y in range(z_stack.shape[1]):
            frame = z_stack[:, y, :]
            frame_norm = (frame / global_max * 255).astype(np.uint8)
            frames.append(frame_norm)
    elif slice_type == "yz":
        for x in range(z_stack.shape[2]):
            frame = z_stack[:, :, x]
            frame_norm = (frame / global_max * 255).astype(np.uint8)
            frames.append(frame_norm)
    else:
        raise ValueError(f"Unknown slice_type: {slice_type}")

    output_path = output_path.with_suffix(".gif")
    iio.imwrite(str(output_path), frames, duration=1000 // fps, loop=0)

    return output_path


def create_multi_channel_slice_movie(
    all_channel_stacks: Dict[str, np.ndarray],
    output_path: Path,
    slice_type: str = "xy",
    fps: int = 10,
    z_gap: float = 1.0,
    channel_colors: Tuple = None,
) -> Path:
    """
    Create a multi-channel color overlay movie going through slices.

    Args:
        all_channel_stacks: Dict {channel_id: 3D array (Z, Y, X)}
        output_path: Path to save movie
        slice_type: "xy", "xz", or "yz"
        fps: Frames per second
        z_gap: Vertical stretch factor for XZ/YZ frames (same as composite)
        channel_colors: Tuple of ChannelColorMapping

    Returns:
        Path to saved movie
    """
    import imageio.v3 as iio
    import matplotlib.colors as mcolors
    from scipy.ndimage import zoom

    from .constants import DEFAULT_CHANNEL_COLORS

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    channel_colors = channel_colors or DEFAULT_CHANNEL_COLORS
    color_map = {cc.channel_id: cc for cc in channel_colors}

    active_channels = [
        cc
        for cc in channel_colors
        if cc.visible and cc.channel_id in all_channel_stacks
    ]

    global_max = 0
    for cc in active_channels:
        stack = all_channel_stacks[cc.channel_id]
        global_max = max(global_max, stack.max())
    if global_max == 0:
        global_max = 1

    first_stack = list(all_channel_stacks.values())[0]
    z_size, y_size, x_size = first_stack.shape

    frames = []

    if slice_type == "xy":
        num_frames = z_size
    elif slice_type == "xz":
        num_frames = y_size
    elif slice_type == "yz":
        num_frames = x_size
    else:
        raise ValueError(f"Unknown slice_type: {slice_type}")

    apply_stretch = slice_type in ("xz", "yz") and z_gap > 1.0

    for i in range(num_frames):
        if slice_type == "xy":
            h, w = y_size, x_size
        elif slice_type == "xz":
            h, w = z_size, x_size
        else:
            h, w = z_size, y_size

        rgb_frame = np.zeros((h, w, 3), dtype=np.float32)

        for cc in active_channels:
            stack = all_channel_stacks[cc.channel_id]

            if slice_type == "xy":
                slice_data = stack[i, :, :]
            elif slice_type == "xz":
                slice_data = stack[:, i, :]
            else:
                slice_data = stack[:, :, i]

            slice_norm = slice_data.astype(np.float32) / global_max
            rgb_color = np.array(mcolors.to_rgb(cc.color))

            for c in range(3):
                rgb_frame[..., c] += slice_norm * rgb_color[c]

        rgb_frame = np.clip(rgb_frame, 0, 1)

        if apply_stretch:
            rgb_frame = zoom(rgb_frame, (z_gap, 1.0, 1.0), order=1)

        frame_uint8 = (rgb_frame * 255).astype(np.uint8)
        frames.append(frame_uint8)

    output_path = output_path.with_suffix(".gif")
    iio.imwrite(str(output_path), frames, duration=1000 // fps, loop=0)

    return output_path


def save_slice_movies_for_well(
    all_channel_stacks: Dict[str, np.ndarray],
    well_id: str,
    output_dir: Path,
    slice_types: Tuple[str, ...] = ("xy", "xz", "yz"),
    fps: int = 10,
    z_gap: float = 1.0,
    channel_colors: Tuple = None,
) -> List[MovieOutput]:
    """
    Save all slice movies for a well (multi-channel color overlay).

    Args:
        all_channel_stacks: Dict {channel_id: 3D array (Z, Y, X)}
        well_id: Well identifier for naming
        output_dir: Directory to save movies
        slice_types: Which slice types to create
        fps: Frames per second
        z_gap: Vertical stretch factor for XZ/YZ (same as composite)
        channel_colors: Color mappings

    Returns:
        List of MovieOutput records
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []

    for slice_type in slice_types:
        filename = f"{well_id}_{slice_type}_movie.gif"
        output_path = output_dir / filename

        actual_path = create_multi_channel_slice_movie(
            all_channel_stacks,
            output_path,
            slice_type=slice_type,
            fps=fps,
            z_gap=z_gap,
            channel_colors=channel_colors,
        )

        outputs.append(
            MovieOutput(
                movie_type=slice_type,
                output_path=actual_path,
                well_id=well_id,
                channel_id="composite",
            )
        )

    return outputs
