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


@dataclass(frozen=True)
class AnimatedGifOutput:
    """Immutable record of saved animated GIF."""

    well_id: str
    output_path: Path
    movie_type: str


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
    bit_depth: int = 8,
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
        bit_depth: 8 or 16 bit output
        channel_colors: Tuple of ChannelColorMapping

    Returns:
        Path to saved movie
    """
    import logging
    import subprocess
    import matplotlib.colors as mcolors
    from scipy.ndimage import zoom

    from .constants import DEFAULT_CHANNEL_COLORS

    movie_logger = logging.getLogger(__name__)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    channel_colors = channel_colors or DEFAULT_CHANNEL_COLORS
    color_map = {cc.channel_id: cc for cc in channel_colors}

    active_channels = [
        cc
        for cc in channel_colors
        if cc.visible and cc.channel_id in all_channel_stacks
    ]

    channel_maxes_movie = {}
    for cc in active_channels:
        stack = all_channel_stacks[cc.channel_id]
        ch_max = float(stack.max())
        if ch_max <= 0:
            ch_max = 1.0
        channel_maxes_movie[cc.channel_id] = ch_max

    first_stack = list(all_channel_stacks.values())[0]
    z_size, y_size, x_size = first_stack.shape

    if slice_type == "xy":
        num_frames = z_size
        frame_height, frame_width = y_size, x_size
    elif slice_type == "xz":
        num_frames = y_size
        frame_height, frame_width = z_size, x_size
    elif slice_type == "yz":
        num_frames = x_size
        frame_height, frame_width = z_size, y_size
    else:
        raise ValueError(f"Unknown slice_type: {slice_type}")

    apply_stretch = slice_type in ("xz", "yz") and z_gap > 1.0

    if apply_stretch:
        frame_height = int(round(frame_height * z_gap))

    output_path = output_path.with_suffix(".mp4")
    movie_logger.info(
        f"  Writing {slice_type} movie to {output_path.name} ({num_frames} frames)"
    )

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{frame_width}x{frame_height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        for i in range(num_frames):
            if i % 20 == 0 or i == num_frames - 1:
                movie_logger.info(f"  Encoding {slice_type} frame {i + 1}/{num_frames}")

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

                ch_max = channel_maxes_movie[cc.channel_id]
                slice_norm = np.clip(slice_data.astype(np.float32) / ch_max, 0, 1)
                rgb_color = np.array(mcolors.to_rgb(cc.color), dtype=np.float32)

                for c in range(3):
                    rgb_frame[..., c] += slice_norm * rgb_color[c]

            rgb_frame = np.clip(rgb_frame, 0, 1)

            if apply_stretch:
                rgb_frame = zoom(rgb_frame, (z_gap, 1.0, 1.0), order=1)

            frame_uint8 = (rgb_frame * 255).astype(np.uint8)
            process.stdin.write(frame_uint8.tobytes())

        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="ignore")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg failed for {output_path.name}: {stderr.strip()}"
            )
    finally:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.stderr:
            process.stderr.close()

    movie_logger.info(
        f"  Completed {slice_type} movie: {output_path.stat().st_size / 1024 / 1024:.1f} MB"
    )

    return output_path


def save_slice_movies_for_well(
    all_channel_stacks: Dict[str, np.ndarray],
    well_id: str,
    output_dir: Path,
    slice_types: Tuple[str, ...] = ("xy", "xz", "yz"),
    fps: int = 10,
    z_gap: float = 1.0,
    bit_depth: int = 8,
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
        bit_depth: 8 or 16 bit output
        channel_colors: Color mappings

    Returns:
        List of MovieOutput records
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []

    for slice_type in slice_types:
        filename = f"{well_id}_{slice_type}_movie.mp4"
        output_path = output_dir / filename

        actual_path = create_multi_channel_slice_movie(
            all_channel_stacks,
            output_path,
            slice_type=slice_type,
            fps=fps,
            z_gap=z_gap,
            bit_depth=bit_depth,
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


def _build_sync_composite_frame(
    all_channel_stacks,
    channel_maxes,
    z_idx,
    xy_h,
    xy_w,
    z_size,
    xz_w,
    yz_w,
    xz_full_h,
    total_height,
    total_width,
    right_col_width,
    GAP,
    z_gap,
    layout,
    active_channels,
):
    """Build one sync composite frame. Returns uint8 RGB array."""
    import matplotlib.colors as mcolors
    from scipy.ndimage import zoom

    xy_rgb = np.zeros((xy_h, xy_w, 3), dtype=np.float32)
    for cc in active_channels:
        stack = all_channel_stacks[cc.channel_id]
        slice_data = stack[z_idx, :, :].astype(np.float32)
        ch_max = channel_maxes[cc.channel_id]
        slice_norm = np.clip(slice_data / ch_max, 0, 1)
        rgb_color = np.array(mcolors.to_rgb(cc.color), dtype=np.float32)
        for c in range(3):
            xy_rgb[..., c] += slice_norm * rgb_color[c]
    xy_rgb = np.clip(xy_rgb, 0, 1)

    xz_rgb = np.zeros((z_size, xz_w, 3), dtype=np.float32)
    for cc in active_channels:
        stack = all_channel_stacks[cc.channel_id]
        slice_data = stack[:, z_idx, :].astype(np.float32)
        ch_max = channel_maxes[cc.channel_id]
        slice_norm = np.clip(slice_data / ch_max, 0, 1)
        rgb_color = np.array(mcolors.to_rgb(cc.color), dtype=np.float32)
        for c in range(3):
            xz_rgb[..., c] += slice_norm * rgb_color[c]
    xz_rgb = np.clip(xz_rgb, 0, 1)

    yz_rgb = np.zeros((z_size, yz_w, 3), dtype=np.float32)
    for cc in active_channels:
        stack = all_channel_stacks[cc.channel_id]
        slice_data = stack[:, :, z_idx].astype(np.float32)
        ch_max = channel_maxes[cc.channel_id]
        slice_norm = np.clip(slice_data / ch_max, 0, 1)
        rgb_color = np.array(mcolors.to_rgb(cc.color), dtype=np.float32)
        for c in range(3):
            yz_rgb[..., c] += slice_norm * rgb_color[c]
    yz_rgb = np.clip(yz_rgb, 0, 1)

    if z_gap > 1.0:
        factor = int(round(z_gap))
        remainder = z_gap - factor
        xz_rgb = np.repeat(xz_rgb, factor, axis=0)
        yz_rgb = np.repeat(yz_rgb, factor, axis=0)
        if remainder > 0:
            xz_rgb = zoom(xz_rgb, (1.0 + remainder / factor, 1.0, 1.0), order=1)
            yz_rgb = zoom(yz_rgb, (1.0 + remainder / factor, 1.0, 1.0), order=1)

    xz_h_actual = xz_rgb.shape[0]
    yz_h_actual = yz_rgb.shape[0]

    composite = np.zeros((total_height, total_width, 3), dtype=np.float32)

    xy_y_start = (total_height - xy_h) // 2
    composite[xy_y_start : xy_y_start + xy_h, :xy_w, :] = xy_rgb

    right_group_start = (total_height - (xz_h_actual + GAP + yz_h_actual)) // 2

    xz_y_start = right_group_start
    xz_x_start = xy_w + (right_col_width - xz_w) // 2
    composite[
        xz_y_start : xz_y_start + xz_h_actual,
        xz_x_start : xz_x_start + xz_w,
        :,
    ] = xz_rgb

    yz_y_start = right_group_start + xz_h_actual + GAP
    yz_x_start = xy_w + (right_col_width - yz_w) // 2
    composite[
        yz_y_start : yz_y_start + yz_h_actual,
        yz_x_start : yz_x_start + yz_w,
        :,
    ] = yz_rgb

    if layout.panel_titles:
        composite[xy_y_start : xy_y_start + 15, 5:60, :] = 1.0
        composite[
            right_group_start : right_group_start + 15,
            xz_x_start : xz_x_start + 50,
            :,
        ] = 1.0
        composite[yz_y_start : yz_y_start + 15, yz_x_start : yz_x_start + 50, :] = 1.0

    return (composite * 255).astype(np.uint8)


def create_synchronized_composite_gif(
    all_channel_stacks: Dict[str, np.ndarray],
    output_path: Path,
    layout,
    channel_colors,
    z_gap: float = 1.0,
    fps: int = 10,
    frame_step: int = 1,
) -> Path:
    """
    Create a synchronized composite GIF with XY on left, XZ/YZ stacked on right.
    """
    import logging
    import subprocess
    import tempfile
    import os

    from .constants import DEFAULT_CHANNEL_COLORS

    sync_logger = logging.getLogger(__name__)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    channel_colors = channel_colors or DEFAULT_CHANNEL_COLORS

    active_channels = [
        cc
        for cc in channel_colors
        if cc.visible and cc.channel_id in all_channel_stacks
    ]

    channel_maxes = {}
    for cc in active_channels:
        stack = all_channel_stacks[cc.channel_id]
        ch_max = float(stack.max())
        if ch_max <= 0:
            ch_max = 1.0
        channel_maxes[cc.channel_id] = ch_max

    first_stack = list(all_channel_stacks.values())[0]
    z_size, y_size, x_size = first_stack.shape

    effective_z = z_size // frame_step
    num_frames = min(effective_z, 120)

    xz_full_h = int(z_size * z_gap) if z_gap > 1.0 else z_size
    yz_full_h = xz_full_h

    GAP = 50

    xy_h, xy_w = y_size, x_size
    xz_h, xz_w = xz_full_h, x_size
    yz_h, yz_w = yz_full_h, y_size

    right_col_width = max(xz_w, yz_w)
    right_column_height = xz_h + GAP + yz_h
    total_width = xy_w + right_col_width
    total_height = max(xy_h, right_column_height)

    output_path = output_path.with_suffix(".gif")
    sync_logger.info(
        f"  Writing sync composite GIF to {output_path.name} ({num_frames} frames, "
        f"{total_width}x{total_height})"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        palette_path = os.path.join(tmpdir, "palette.png")

        palette_cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{total_width}x{total_height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-vf",
            # This GIF is always opaque. Reserving a transparent palette entry
            # reduces the usable color budget and can cause viewer-dependent
            # black/transparent artifacts in later frames.
            "palettegen=stats_mode=full:reserve_transparent=0",
            palette_path,
        ]

        encode_cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{total_width}x{total_height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-i",
            palette_path,
            "-lavfi",
            "paletteuse",
            # Encode each frame as a full opaque image. ffmpeg's default GIF
            # encoder flags use offsetting/transdiff optimization, which can
            # introduce viewer-dependent corruption in later frames.
            "-gifflags",
            "-offsetting-transdiff",
            "-loop",
            "0",
            "-r",
            str(fps),
            str(output_path),
        ]

        palette_proc = subprocess.Popen(
            palette_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        for i in range(num_frames):
            if i % 20 == 0 or i == num_frames - 1:
                sync_logger.info(f"  Building frame {i + 1}/{num_frames}")
            frame = _build_sync_composite_frame(
                all_channel_stacks,
                channel_maxes,
                z_size - 1 - i * frame_step,
                xy_h,
                xy_w,
                z_size,
                xz_w,
                yz_w,
                xz_full_h,
                total_height,
                total_width,
                right_col_width,
                GAP,
                z_gap,
                layout,
                active_channels,
            )
            palette_proc.stdin.write(frame.tobytes())

        palette_proc.stdin.close()
        palette_proc.wait()
        palette_stderr = palette_proc.stderr.read().decode("utf-8", errors="ignore")
        if palette_proc.returncode != 0:
            raise RuntimeError(f"palettegen failed: {palette_stderr}")

        sync_logger.info("  Encoding GIF with palette...")

        encode_proc = subprocess.Popen(
            encode_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        for i in range(num_frames):
            if i % 20 == 0 or i == num_frames - 1:
                sync_logger.info(f"  Encoding frame {i + 1}/{num_frames}")
            frame = _build_sync_composite_frame(
                all_channel_stacks,
                channel_maxes,
                z_size - 1 - i * frame_step,
                xy_h,
                xy_w,
                z_size,
                xz_w,
                yz_w,
                xz_full_h,
                total_height,
                total_width,
                right_col_width,
                GAP,
                z_gap,
                layout,
                active_channels,
            )
            try:
                encode_proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                break

        encode_proc.stdin.close()
        encode_proc.wait()
        encode_stderr = encode_proc.stderr.read().decode("utf-8", errors="ignore")
        if encode_proc.returncode != 0:
            raise RuntimeError(f"ffmpeg paletteuse failed: {encode_stderr}")

    sync_logger.info(
        f"  Completed sync composite GIF: {output_path.stat().st_size / 1024 / 1024:.1f} MB"
    )
    return output_path


def save_synchronized_gif_for_well(
    all_channel_stacks: Dict[str, np.ndarray],
    well_id: str,
    output_dir: Path,
    layout=None,
    channel_colors=None,
    z_gap: float = 1.0,
    fps: int = 10,
    frame_step: int = 1,
) -> List[AnimatedGifOutput]:
    """
    Save synchronized composite GIF for a well.

    Args:
        all_channel_stacks: Dict {channel_id: 3D array (Z, Y, X)}
        well_id: Well identifier for naming
        output_dir: Directory to save GIF
        layout: CompositeLayout for layout parameters
        channel_colors: Color mappings
        z_gap: Vertical stretch factor
        fps: Frames per second
        frame_step: Step through XY frames

    Returns:
        List of AnimatedGifOutput records
    """
    from .constants import DEFAULT_CHANNEL_COLORS, CompositeLayout

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layout = layout or CompositeLayout()

    filename = f"{well_id}_composite_sync.gif"
    output_path = output_dir / filename

    actual_path = create_synchronized_composite_gif(
        all_channel_stacks,
        output_path,
        layout=layout,
        channel_colors=channel_colors or DEFAULT_CHANNEL_COLORS,
        z_gap=z_gap,
        fps=fps,
        frame_step=frame_step,
    )

    return [
        AnimatedGifOutput(
            well_id=well_id,
            output_path=actual_path,
            movie_type="composite_sync",
        )
    ]
