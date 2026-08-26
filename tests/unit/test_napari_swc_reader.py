from importlib.resources import files

import numpy as np
import pytest

from openhcs.runtime.napari_swc_reader import read_swc_layers


@pytest.mark.unit
def test_installed_npe2_manifest_discovers_and_reads_swc_points_and_shapes(
    tmp_path,
) -> None:
    npe2 = pytest.importorskip("npe2")
    swc_path = tmp_path / "morphology.swc"
    swc_path.write_text(
        "\n".join(
            (
                "# OpenHCS spatial graph: traced_neuron",
                "# id type x y z radius parent",
                "1 1 10 20 30 4 -1",
                "2 2 11 22 33 2 1",
                "3 3 15 25 35 1 2",
                "",
            )
        ),
        encoding="utf-8",
    )

    manager = npe2.PluginManager.instance()
    manager.discover()
    compatible = tuple(manager.iter_compatible_readers([str(swc_path)]))

    assert any(reader.command == "openhcs.get_swc_reader" for reader in compatible)
    layer_data, reader = npe2.io_utils._read(
        [str(swc_path)],
        stack=False,
        plugin_name="openhcs",
        return_reader=True,
        _pm=manager,
    )

    assert reader.command == "openhcs.get_swc_reader"
    assert [layer_type for _data, _kwargs, layer_type in layer_data] == [
        "points",
        "shapes",
    ]
    points, point_kwargs, _point_type = layer_data[0]
    shapes, shape_kwargs, _shape_type = layer_data[1]
    assert point_kwargs["ndim"] == 3
    assert shape_kwargs["ndim"] == 3
    np.testing.assert_array_equal(
        points,
        np.array(((30.0, 20.0, 10.0), (33.0, 22.0, 11.0), (35.0, 25.0, 15.0))),
    )
    assert len(shapes) == 2
    expected_features = {
        "sample_id",
        "sample_type",
        "radius",
        "parent_sample_id",
    }
    assert set(point_kwargs["features"]) == expected_features
    assert set(shape_kwargs["features"]) == expected_features
    np.testing.assert_array_equal(
        point_kwargs["features"]["parent_sample_id"],
        np.array((-1, 1, 2)),
    )
    np.testing.assert_array_equal(
        shape_kwargs["features"]["sample_type"],
        np.array((2, 3)),
    )
    assert files("openhcs").joinpath("napari.yaml").is_file()


@pytest.mark.unit
def test_root_only_swc_mounts_empty_three_dimensional_shapes_layer(tmp_path) -> None:
    swc_path = tmp_path / "root-only.swc"
    swc_path.write_text("1 1 10 20 30 4 -1\n", encoding="utf-8")

    point_layer, shape_layer = read_swc_layers(str(swc_path))

    assert point_layer[0].shape == (1, 3)
    assert point_layer[1]["ndim"] == 3
    assert shape_layer[0] == []
    assert shape_layer[1]["ndim"] == 3
    assert all(len(values) == 0 for values in shape_layer[1]["features"].values())
