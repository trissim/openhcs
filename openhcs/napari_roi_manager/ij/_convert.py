from __future__ import annotations

import struct
import warnings

import numpy as np
from napari.layers.shapes._shapes_constants import ShapeType
from numpy.typing import NDArray
from roifile import ROI_SUBTYPE, ROI_TYPE, ImagejRoi

from openhcs.napari_roi_manager._dataclasses import RoiTuple


def roi_to_shape(ijroi: ImagejRoi) -> RoiTuple | None:
    """Convert an ImageJ ROI to a shape coordinates and type."""
    p = _imagej_position_to_index(ijroi.position)
    t = _imagej_position_to_index(ijroi.t_position)
    z = _imagej_position_to_index(ijroi.z_position)
    name = ijroi.name
    multidim = (p, t, z)

    if ijroi.subtype == ROI_SUBTYPE.UNDEFINED:
        if ijroi.roitype == ROI_TYPE.RECT:
            if ijroi.subpixelresolution:
                x = ijroi.xd
                y = ijroi.yd
                width = ijroi.widthd
                height = ijroi.heightd
            else:
                x = ijroi.left
                y = ijroi.top
                width = ijroi.right - ijroi.left
                height = ijroi.bottom - ijroi.top
            data = np.array(
                [
                    [[y, x], [y, x + width], [y + height, x + width], [y + height, x]],
                ]
            )
            out = RoiTuple(
                data=data - 0.5,
                shape_type=ShapeType.RECTANGLE,
                name=name,
                multidim=multidim,
            )
        elif ijroi.roitype == ROI_TYPE.LINE:
            start = (ijroi.y1, ijroi.x1)
            end = (ijroi.y2, ijroi.x2)
            out = RoiTuple(
                data=[start, end],
                shape_type=ShapeType.LINE,
                name=name,
                multidim=multidim,
            )
        elif ijroi.roitype == ROI_TYPE.POINT:
            warnings.warn(
                "ImageJ point ROI encountered, but napari shapes layer does not "
                "support points. Point ROIs will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            return
        elif ijroi.roitype in (ROI_TYPE.POLYGON, ROI_TYPE.FREEHAND):
            coords = _get_coords(ijroi)
            out = RoiTuple(
                data=coords,
                shape_type=ShapeType.POLYGON,
                name=name,
                multidim=multidim,
            )
        elif ijroi.roitype in (ROI_TYPE.POLYLINE, ROI_TYPE.FREELINE):
            coords = _get_coords(ijroi)
            out = RoiTuple(
                data=coords,
                shape_type=ShapeType.PATH,
                name=name,
                multidim=multidim,
            )
        elif ijroi.roitype == ROI_TYPE.OVAL:
            if ijroi.subpixelresolution:
                x = ijroi.xd
                y = ijroi.yd
                width = ijroi.widthd
                height = ijroi.heightd
            else:
                x = ijroi.left
                y = ijroi.top
                width = ijroi.right - ijroi.left
                height = ijroi.bottom - ijroi.top
            data = np.array(
                [
                    [[y, x], [y, x + width], [y + height, x + width], [y + height, x]],
                ]
            )
            out = RoiTuple(
                data=data - 0.5,
                shape_type=ShapeType.ELLIPSE,
                name=name,
                multidim=multidim,
            )
        else:
            raise ValueError(f"Unsupported ROI type: {ijroi.roitype!r}")
    elif ijroi.subtype == ROI_SUBTYPE.ROTATED_RECT:
        coords = _get_coords(ijroi)
        out = RoiTuple(
            data=coords,
            shape_type=ShapeType.RECTANGLE,
            name=name,
            multidim=multidim,
        )
    elif ijroi.subtype == ROI_SUBTYPE.ELLIPSE:
        ratio = decode_rotated_roi_width(
            (
                ijroi.arrow_style_or_aspect_ratio,
                ijroi.arrow_head_size,
                ijroi.rounded_rect_arc_size,
            ),
            ijroi.byteorder,
        )
        start = np.array([ijroi.y1, ijroi.x1])
        end = np.array([ijroi.y2, ijroi.x2])
        length = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        width = length * ratio
        vec_longitudinal = end - start
        vec_lateral = (
            np.array([-vec_longitudinal[1], vec_longitudinal[0]])
            / np.linalg.norm(vec_longitudinal)
            * width
        )
        r00 = start - vec_lateral / 2
        r01 = start + vec_lateral / 2
        r10 = end - vec_lateral / 2
        r11 = end + vec_lateral / 2
        out = RoiTuple(
            data=np.array([[r00, r01, r11, r10]]),
            shape_type=ShapeType.ELLIPSE,
            name=name,
            multidim=multidim,
        )
    else:
        raise ValueError(f"Unsupported ROI subtype: {ijroi.subtype}")

    return out


def _imagej_position_to_index(position: int) -> int:
    """Project ImageJ's positive 1-based positions to native 0-based indices."""
    return position - 1 if position > 0 else 0


def shape_to_roi(shape: RoiTuple) -> ImagejRoi:
    """Convert a shape to an ImageJ ROI."""
    ys = shape.data[:, -2]
    xs = shape.data[:, -1]
    if shape.shape_type is ShapeType.RECTANGLE:
        if ys[0] == ys[1] or xs[0] == xs[1]:
            # not rotated
            y0, y1 = ys.min() + 0.5, ys.max() + 0.5
            x0, x1 = xs.min() + 0.5, xs.max() + 0.5
            roi = ImagejRoi.frompoints(
                np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]),
                name=shape.name,
                **_get_multidim_kwargs(shape),
            )
            roi.roitype = ROI_TYPE.RECT
            roi.y1 = roi.yd = y0
            roi.x1 = roi.xd = x0
            roi.y2 = y1
            roi.x2 = x1
            roi.widthd = x1 - x0
            roi.heightd = y1 - y0
        else:
            roi = ImagejRoi.frompoints(
                np.stack([xs, ys], axis=1),
                name=shape.name,
                **_get_multidim_kwargs(shape),
            )
            roi.roitype = ROI_TYPE.FREEHAND
            roi.subtype = ROI_SUBTYPE.ROTATED_RECT
            roi.x1 = xs[1:3].mean()
            roi.y1 = ys[1:3].mean()
            roi.x2 = xs[:4:3].mean()
            roi.y2 = ys[:4:3].mean()
            width = np.sqrt((xs[1] - xs[2]) ** 2 + (ys[1] - ys[2]) ** 2)
            (
                roi.arrow_style_or_aspect_ratio,
                roi.arrow_head_size,
                roi.rounded_rect_arc_size,
            ) = encode_rotated_roi_width(width, roi.byteorder)
        return roi
    elif shape.shape_type is ShapeType.LINE:
        roi = ImagejRoi.frompoints(
            np.stack([xs, ys], axis=1),
            name=shape.name,
            **_get_multidim_kwargs(shape),
        )
        roi.x1 = xs[0]
        roi.y1 = ys[0]
        roi.x2 = xs[1]
        roi.y2 = ys[1]
        roi.roitype = ROI_TYPE.LINE
        return roi
    elif shape.shape_type is ShapeType.PATH:
        roi = ImagejRoi.frompoints(
            np.stack([xs, ys], axis=1),
            name=shape.name,
            **_get_multidim_kwargs(shape),
        )
        roi.roitype = ROI_TYPE.POLYLINE
        return roi
    elif shape.shape_type is ShapeType.POLYGON:
        roi = ImagejRoi.frompoints(
            np.stack([xs, ys], axis=1),
            name=shape.name,
            **_get_multidim_kwargs(shape),
        )
        roi.roitype = ROI_TYPE.POLYGON
        return roi
    elif shape.shape_type is ShapeType.ELLIPSE:
        if ys[0] == ys[1] or xs[0] == xs[1]:
            # not rotated
            y0, y1 = ys.min() + 0.5, ys.max() + 0.5
            x0, x1 = xs.min() + 0.5, xs.max() + 0.5
            roi = ImagejRoi.frompoints(
                np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]),
                name=shape.name,
                **_get_multidim_kwargs(shape),
            )
            roi.roitype = ROI_TYPE.OVAL
            roi.y1 = roi.yd = y0
            roi.x1 = roi.xd = x0
            roi.y2 = y1
            roi.x2 = x1
            roi.widthd = x1 - x0
            roi.heightd = y1 - y0
        else:
            b = np.sqrt((xs[0] - xs[1]) ** 2 + (ys[0] - ys[1]) ** 2) / 2
            a = np.sqrt((xs[1] - xs[2]) ** 2 + (ys[1] - ys[2]) ** 2) / 2
            phi = -np.arctan2(ys[1] - ys[2], xs[1] - xs[2])
            ts = np.linspace(0, 2 * np.pi, 72, endpoint=False)
            x1 = xs[:2].mean()
            y1 = ys[:2].mean()
            x2 = xs[2:4].mean()
            y2 = ys[2:4].mean()
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            xsamples = (
                a * np.cos(ts) * np.cos(phi) - b * np.sin(ts) * np.sin(phi) + center[0]
            )
            ysamples = (
                a * np.cos(ts) * np.sin(phi) + b * np.sin(ts) * np.cos(phi) + center[1]
            )
            roi = ImagejRoi.frompoints(
                np.stack([xsamples, ysamples], axis=1),
                name=shape.name,
                **_get_multidim_kwargs(shape),
            )
            roi.roitype = ROI_TYPE.FREEHAND
            roi.subtype = ROI_SUBTYPE.ELLIPSE
            roi.x1 = x1
            roi.y1 = y1
            roi.x2 = x2
            roi.y2 = y2
            (
                roi.arrow_style_or_aspect_ratio,
                roi.arrow_head_size,
                roi.rounded_rect_arc_size,
            ) = encode_rotated_roi_width(b / a, roi.byteorder)
        return roi
    raise ValueError(f"Unsupported shape type: {shape.shape_type}")


def _get_coords(ijroi: ImagejRoi) -> NDArray[np.number]:
    if ijroi.subpixelresolution:
        xy = ijroi.subpixel_coordinates
    else:
        xy = ijroi.integer_coordinates + [ijroi.left, ijroi.top]
    return xy[:, ::-1]


def _get_multidim_kwargs(shape: RoiTuple) -> dict[str, int]:
    """Get the multidim kwargs for a shape."""
    if len(shape.multidim) == 0:
        return {}
    if len(shape.multidim) == 1:
        return {"z": shape.multidim[0]}
    if len(shape.multidim) == 2:
        return {
            "t": shape.multidim[0],
            "z": shape.multidim[1],
        }
    else:
        return {
            "position": shape.multidim[0],
            "t": shape.multidim[1],
            "z": shape.multidim[2],
        }


def encode_rotated_roi_width(width: float, byteorder: str) -> tuple[int, int, int]:
    s = struct.pack(byteorder + "f", width)
    return struct.unpack(byteorder + "BBh", s)


def decode_rotated_roi_width(ints: tuple[int, int, int], byteorder: str) -> float:
    s = struct.pack(byteorder + "BBh", *ints)
    return struct.unpack(byteorder + "f", s)[0]
