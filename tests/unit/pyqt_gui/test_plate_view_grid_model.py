from openhcs.pyqt_gui.widgets.shared.plate_view_widget import PlateGridModel


def test_plate_grid_model_parses_standard_well_coordinates() -> None:
    model = PlateGridModel.from_wells({"B02", "C04", "AA10"})

    assert model.coord_to_well[(2, 2)] == "B02"
    assert model.coord_to_well[(3, 4)] == "C04"
    assert model.coord_to_well[(27, 10)] == "AA10"
    assert model.well_to_coord["AA10"] == (27, 10)
    assert model.plate_dimensions == (26, 9)
    assert model.row_offset == 1
    assert model.col_offset == 1


def test_plate_grid_model_uses_supplied_nonstandard_coordinates() -> None:
    model = PlateGridModel.from_wells(
        {"R02C03", "R04C05"},
        coord_to_well={(2, 3): "R02C03", (4, 5): "R04C05"},
    )

    assert model.actual_rows == [2, 3, 4]
    assert model.actual_cols == [3, 4, 5]
    assert model.well_at(4, 5) == "R04C05"
    assert model.well_at(3, 4) is None
    assert model.plate_dimensions == (3, 3)


def test_plate_grid_model_preserves_explicit_plate_dimensions() -> None:
    model = PlateGridModel.from_wells(
        {"A01", "B02"},
        plate_dimensions=(8, 12),
    )

    assert model.plate_dimensions == (8, 12)
    assert model.actual_rows == [1, 2]
    assert model.actual_cols == [1, 2]


def test_plate_grid_model_projects_axis_membership() -> None:
    model = PlateGridModel.from_wells({"A01", "A02", "B01"})

    assert model.wells_on_axis(axis_index=0, axis_value=1) == ["A01", "A02"]
    assert model.wells_on_axis(axis_index=1, axis_value=1) == ["A01", "B01"]

