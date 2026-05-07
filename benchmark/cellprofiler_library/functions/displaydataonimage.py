"""Converted from CellProfiler: DisplayDataOnImage"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass, replace
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs
from openhcs.core.runtime_values import object_label_dense_array


class DisplayMode(Enum):
    TEXT = "text"
    COLOR = "color"


class ObjectsOrImage(Enum):
    OBJECTS = "objects"
    IMAGE = "image"


class ColorMapScale(Enum):
    USE_MEASUREMENT_RANGE = "use_measurement_range"
    MANUAL = "manual"


class SavedImageContents(Enum):
    IMAGE = "image"
    AXES = "axes"
    FIGURE = "figure"


@dataclass(frozen=True)
class DisplayDataOnImageRequest:
    """Typed request for rendering CellProfiler measurements onto an image."""

    image: np.ndarray
    labels: Optional[np.ndarray]
    measurements: Optional[np.ndarray]
    objects_or_image: ObjectsOrImage
    display_mode: DisplayMode
    wants_background_image: bool
    text_color: Tuple[float, float, float]
    font_size: int
    decimals: int
    offset: int
    colormap: str
    color_map_scale_choice: ColorMapScale
    color_map_scale_min: float
    color_map_scale_max: float
    use_scientific_notation: bool
    image_measurement_value: Optional[float]
    center_x: Optional[np.ndarray]
    center_y: Optional[np.ndarray]

    def for_slice(self, index: int) -> "DisplayDataOnImageRequest":
        labels = (
            self.labels[index]
            if self.labels is not None and self.labels.ndim == 3
            else self.labels
        )
        return replace(self, image=self.image[index], labels=labels)


@numpy
@special_inputs("labels", "measurements")
def display_data_on_image(
    image: np.ndarray,
    labels: Optional[np.ndarray] = None,
    measurements: Optional[np.ndarray] = None,
    measurement_feature: Optional[str] = None,
    objects_or_image: ObjectsOrImage = ObjectsOrImage.OBJECTS,
    display_mode: DisplayMode = DisplayMode.TEXT,
    wants_background_image: bool = True,
    text_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    font_size: int = 10,
    decimals: int = 2,
    offset: int = 0,
    colormap: str = "viridis",
    color_map_scale_choice: ColorMapScale = ColorMapScale.USE_MEASUREMENT_RANGE,
    color_map_scale_min: float = 0.0,
    color_map_scale_max: float = 1.0,
    use_scientific_notation: bool = False,
    image_measurement_value: Optional[float] = None,
    center_x: Optional[np.ndarray] = None,
    center_y: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Display measurement data on top of an image.
    
    This function overlays measurement values on an image, either as text
    annotations at object centers or as a color map applied to object regions.
    
    Args:
        image: Input image, shape (D, H, W) or (H, W)
        labels: Optional label image for objects, shape matching image
        measurements: Optional array of measurement values per object
        measurement_feature: CellProfiler feature selected for runtime measurement lookup
        objects_or_image: Whether displaying object or image measurements
        display_mode: TEXT for numeric values, COLOR for colormap overlay
        wants_background_image: Whether to show background image or black
        text_color: RGB tuple for text color (0-1 range)
        font_size: Font size in points
        decimals: Number of decimal places to display
        offset: Pixel offset for text placement
        colormap: Name of matplotlib colormap
        color_map_scale_choice: Use measurement range or manual scale
        color_map_scale_min: Manual minimum for color scale
        color_map_scale_max: Manual maximum for color scale
        use_scientific_notation: Display values in scientific notation
        image_measurement_value: Single value for image-level measurement
        center_x: X coordinates of object centers
        center_y: Y coordinates of object centers
    
    Returns:
        RGB image with measurements displayed, shape (D, H, W, 3) or (H, W, 3)
    """
    request = DisplayDataOnImageRequest(
        image=image,
        labels=labels,
        measurements=measurements,
        objects_or_image=objects_or_image,
        display_mode=display_mode,
        wants_background_image=wants_background_image,
        text_color=text_color,
        font_size=font_size,
        decimals=decimals,
        offset=offset,
        colormap=colormap,
        color_map_scale_choice=color_map_scale_choice,
        color_map_scale_min=color_map_scale_min,
        color_map_scale_max=color_map_scale_max,
        use_scientific_notation=use_scientific_notation,
        image_measurement_value=image_measurement_value,
        center_x=center_x,
        center_y=center_y,
    )

    # Handle dimensionality
    if image.ndim == 3:
        # Process each slice
        results = []
        for i in range(image.shape[0]):
            results.append(_display_data_on_slice(request.for_slice(i)))
        return np.stack(results, axis=0)
    return _display_data_on_slice(request)


def _display_data_on_slice(request: DisplayDataOnImageRequest) -> np.ndarray:
    """Process a single 2D slice."""
    from skimage.measure import regionprops
    import cv2

    image = request.image
    labels = request.labels
    measurements = request.measurements
    h, w = image.shape[:2]
    
    # Prepare background
    if request.wants_background_image:
        if image.ndim == 2:
            # Grayscale to RGB
            background = np.stack([image, image, image], axis=-1)
        else:
            background = image.copy()
    else:
        background = np.zeros((h, w, 3), dtype=np.float32)
    
    # Normalize to 0-1 range if needed
    if background.max() > 1.0:
        background = background / 255.0
    background = background.astype(np.float32)
    
    if request.objects_or_image == ObjectsOrImage.IMAGE:
        # Display single image measurement at center
        if request.image_measurement_value is not None:
            x = w // 2
            y = h // 2
            x_offset = np.random.uniform(-1.0, 1.0)
            y_offset = np.sqrt(1 - x_offset ** 2)
            x = int(x + request.offset * x_offset)
            y = int(y + request.offset * y_offset)
            
            if request.use_scientific_notation:
                text = f"{request.image_measurement_value:.{request.decimals}e}"
            else:
                text = f"{request.image_measurement_value:.{request.decimals}f}"
            
            # Convert to uint8 for cv2
            output = (background * 255).astype(np.uint8)
            color_bgr = _text_color_bgr(request)
            font_scale = request.font_size / 20.0
            cv2.putText(output, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, color_bgr, 1, cv2.LINE_AA)
            return output.astype(np.float32) / 255.0
    
    elif request.objects_or_image == ObjectsOrImage.OBJECTS and labels is not None:
        labels = object_label_dense_array(labels, dtype=np.int32)
        if request.display_mode == DisplayMode.COLOR and measurements is not None:
            # Color map mode
            from matplotlib import cm
            
            # Get colormap
            cmap = cm.get_cmap(request.colormap)
            
            # Determine scale
            valid_measurements = measurements[~np.isnan(measurements)] if len(measurements) > 0 else np.array([0, 1])
            if request.color_map_scale_choice == ColorMapScale.MANUAL:
                vmin, vmax = request.color_map_scale_min, request.color_map_scale_max
            else:
                vmin = valid_measurements.min() if len(valid_measurements) > 0 else 0
                vmax = valid_measurements.max() if len(valid_measurements) > 0 else 1
            
            if vmax == vmin:
                vmax = vmin + 1
            
            # Normalize measurements
            normalized = (measurements - vmin) / (vmax - vmin)
            normalized = np.clip(normalized, 0, 1)
            
            # Create colored output
            output = background.copy()
            if output.ndim == 2:
                output = np.stack([output, output, output], axis=-1)
            
            # Apply colors to each labeled region
            for i, val in enumerate(normalized):
                if not np.isnan(val):
                    color = cmap(val)[:3]
                    mask = labels == (i + 1)
                    for c in range(3):
                        output[:, :, c] = np.where(mask, 
                            output[:, :, c] * 0.5 + color[c] * 0.5,
                            output[:, :, c])
            
            return output
        
        else:
            # Text mode
            # Get object centers
            if request.center_x is None or request.center_y is None:
                props = regionprops(labels)
                centers = [(p.centroid[1], p.centroid[0]) for p in props]
            else:
                centers = list(zip(request.center_x, request.center_y))
            
            # Convert to uint8 for cv2
            output = (background * 255).astype(np.uint8)
            color_bgr = _text_color_bgr(request)
            font_scale = request.font_size / 20.0
            
            if measurements is not None:
                for idx, (cx, cy) in enumerate(centers):
                    if idx < len(measurements):
                        val = measurements[idx]
                        if np.isnan(val):
                            continue
                        
                        # Apply offset
                        x_off = np.random.uniform(-1.0, 1.0)
                        y_off = np.sqrt(1 - x_off ** 2)
                        x = int(cx + request.offset * x_off)
                        y = int(cy + request.offset * y_off)
                        
                        if request.use_scientific_notation:
                            text = f"{val:.{request.decimals}e}"
                        else:
                            text = f"{val:.{request.decimals}f}"
                        
                        cv2.putText(output, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                   font_scale, color_bgr, 1, cv2.LINE_AA)
            
            return output.astype(np.float32) / 255.0
    
    return background


def _text_color_bgr(
    request: DisplayDataOnImageRequest,
) -> Tuple[int, int, int]:
    return (
        int(request.text_color[2] * 255),
        int(request.text_color[1] * 255),
        int(request.text_color[0] * 255),
    )
