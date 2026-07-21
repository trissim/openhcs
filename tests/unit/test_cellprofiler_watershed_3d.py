from __future__ import annotations

import numpy as np
import pytest

from openhcs.core.runtime_object_labels import object_label_dense_array
from openhcs.core.runtime_object_label_domains import ObjectLabelDomainScope
from openhcs.processing.backends.cellprofiler.watershed import (
    WatershedMethod,
    watershed_cellprofiler4,
)


def test_marker_watershed_3d_defaults_to_thresholded_image_domain() -> None:
    image = np.zeros((3, 7, 9), dtype=bool)
    image[:, 1:6, 1:8] = True
    markers = np.zeros(image.shape, dtype=np.int32)
    markers[1, 3, 2] = 1
    markers[1, 3, 6] = 2

    _, stats, label_value = watershed_cellprofiler4(
        image,
        topology_inputs=(markers,),
        watershed_method=WatershedMethod.MARKERS,
        use_advanced_settings=False,
    )

    labels = object_label_dense_array(label_value)
    np.testing.assert_array_equal(labels > 0, image)
    assert np.count_nonzero(labels[~image]) == 0
    assert np.unique(labels).tolist() == [0, 1, 2]
    assert label_value.domain.scope is ObjectLabelDomainScope.PAYLOAD
    assert label_value.domain.declared_object_ids == (1, 2)
    (stats_row,) = stats.row_mappings()
    assert stats_row["object_count"] == 2
    assert stats_row["mean_area"] == pytest.approx(np.count_nonzero(image) / 2)
