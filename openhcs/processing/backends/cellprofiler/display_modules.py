"""CellProfiler display module declarations."""

from __future__ import annotations
from openhcs.interop.cellprofiler.runtime.binding_authorities import (
    CellProfilerStringKwargAuthority,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_vectors import (
    CellProfilerObjectInputCountAuthority,
    CellProfilerObjectMeasurementVectorBinding,
)
from openhcs.interop.cellprofiler.runtime.payload_types import CellProfilerKwargDict
from openhcs.interop.cellprofiler.runtime.special_input_policies import (
    NoSpecialImageInputsMixin,
    SpecialInputBindingRequest,
)
from openhcs.interop.cellprofiler.setting_names import SettingNameFamily
from openhcs.interop.cellprofiler.module_declarations import (
    ProcessingContract,
    BoundModuleSettings,
    CellProfilerModule,
    ImageArtifactInputModule,
    ImageArtifactOutputModule,
    ObjectArtifactInputModule,
    PriorMeasurementArtifactInputModule,
)
from openhcs.interop.cellprofiler.display_data_on_image import display_data_on_image
from openhcs.interop.cellprofiler.display_density_plot import display_density_plot
from openhcs.interop.cellprofiler.display_histogram import display_histogram
from openhcs.interop.cellprofiler.display_plate_map import display_platemap
from openhcs.interop.cellprofiler.display_scatter_plot import display_scatter_plot


class DisplayModule(CellProfilerModule):
    """Parent for CellProfiler display/export debug sections."""


class DisplayDataOnImageSpecialInputPolicy(NoSpecialImageInputsMixin):
    """Resolve display annotations from object labels and measurement tables."""

    def bind(self, request: SpecialInputBindingRequest) -> CellProfilerKwargDict:
        object_inputs = request.object_inputs
        CellProfilerObjectInputCountAuthority.require_exact(
            request.module_name, object_inputs, 1
        )
        object_spec = object_inputs[0]
        labels = request.labels_for(object_spec)
        feature_name = CellProfilerStringKwargAuthority.required(
            request.kwargs, "measurement_feature", request.module_name
        )
        return {
            "labels": labels,
            "measurements": CellProfilerObjectMeasurementVectorBinding.for_object(
                request,
                object_ref=object_spec,
                feature_name=feature_name,
                labels=labels,
            )
            .vector()
            .slice_aligned_value,
        }


class DisplayDataOnImageModule(
    DisplayDataOnImageSpecialInputPolicy,
    PriorMeasurementArtifactInputModule,
    ImageArtifactInputModule,
    ImageArtifactOutputModule,
    ObjectArtifactInputModule,
    DisplayModule,
):
    module_name = "DisplayDataOnImage"
    function_name = "display_data_on_image"
    validated = True
    contract = ProcessingContract.FLEXIBLE
    confidence = 1.0
    image_input_settings = ("Select the image on which to display the measurements",)
    object_input_settings = ("Select the input objects",)
    image_output_settings = (
        "Name the output image that has the measurements displayed",
    )
    object_or_image_setting = SettingNameFamily("Display object or image measurements?")
    measurement_feature_setting = SettingNameFamily("Measurement to display")
    object_or_image_parameter = "objects_or_image"
    measurement_feature_parameter = "measurement_feature"
    default_object_or_image = "Object"
    setting_parameter_aliases = {
        object_or_image_setting: object_or_image_parameter,
        measurement_feature_setting: measurement_feature_parameter,
        "Display mode": "display_mode",
        "Display background image?": "wants_background_image",
        "Text color": "text_color",
        "Font size": "font_size",
        "Number of decimals": "decimals",
        "Annotation offset": "offset",
        "Color map": "colormap",
        "Color map scale": "color_map_scale_choice",
        "Color map range": ["color_map_scale_min", "color_map_scale_max"],
    }
    ignored_settings = (
        "Image elements to save",
    )

    @classmethod
    def postprocess_bound_settings(
        cls, module: "ModuleBlock", bound: "BoundModuleSettings"
    ) -> "BoundModuleSettings":
        del module
        if cls.measurement_feature_parameter not in bound.kwargs:
            raise ValueError("DisplayDataOnImage requires a measurement feature.")
        kwargs = dict(bound.kwargs)
        if "text_color" in kwargs:
            from openhcs.processing.backends.cellprofiler.color import coerce_rgb_color

            kwargs["text_color"] = coerce_rgb_color(kwargs["text_color"])
        if kwargs.get("colormap") == "Default":
            kwargs["colormap"] = "viridis"
        bound = BoundModuleSettings(
            kwargs,
            bound.unmapped_kwargs,
            bound.invocation_options,
            bound.setting_coverage,
        )
        if cls.object_or_image_parameter in bound.kwargs:
            return bound
        return bound.with_kwargs(
            {cls.object_or_image_parameter: cls.default_object_or_image}
        )


class DisplayDensityPlotModule(DisplayModule):
    module_name = "DisplayDensityPlot"
    function_name = "display_density_plot"
    validated = True
    contract = ProcessingContract.FLEXIBLE
    confidence = 1.0


class DisplayHistogramModule(DisplayModule):
    module_name = "DisplayHistogram"
    function_name = "display_histogram"
    validated = True
    contract = None
    confidence = 1.0


class DisplayPlatemapModule(DisplayModule):
    module_name = "DisplayPlatemap"
    function_name = "display_platemap"
    validated = True
    confidence = 1.0


class DisplayScatterPlotModule(DisplayModule):
    module_name = "DisplayScatterPlot"
    function_name = "display_scatter_plot"
    validated = True
    contract = None
    confidence = 1.0
