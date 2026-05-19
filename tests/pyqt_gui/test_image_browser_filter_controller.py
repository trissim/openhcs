"""Tests for ImageBrowser filter projection without constructing the widget."""

from openhcs.pyqt_gui.widgets.image_browser import ImageBrowserFilterController


class _FolderItem:
    def __init__(self, folder_path: str | None):
        self.folder_path = folder_path

    def data(self, *_):
        return self.folder_path


class _FolderTree:
    def __init__(self, folder_path: str | None = None):
        self.folder_path = folder_path

    def selectedItems(self):
        return [] if self.folder_path is None else [_FolderItem(self.folder_path)]


class _ColumnFilterPanel:
    def __init__(self, filters):
        self.filters = filters

    def get_active_filters(self):
        return self.filters


class _Browser:
    def __init__(self):
        self.folder_tree = _FolderTree("PlateA")
        self.column_filter_panel = _ColumnFilterPanel({"Channel": {"DAPI"}})
        self.selected_wells = {"A01"}
        self.filtered_files = {
            "PlateA/A01_DAPI.tif": {
                "filename": "PlateA/A01_DAPI.tif",
                "well": "A01",
                "_display_channel": "DAPI",
            },
            "PlateA/A02_DAPI.tif": {
                "filename": "PlateA/A02_DAPI.tif",
                "well": "A02",
                "_display_channel": "DAPI",
            },
            "PlateA_results/A01_DAPI.csv": {
                "filename": "PlateA_results/A01_DAPI.csv",
                "well": "A01",
                "_display_channel": "DAPI",
            },
            "PlateB/A01_DAPI.tif": {
                "filename": "PlateB/A01_DAPI.tif",
                "well": "A01",
                "_display_channel": "DAPI",
            },
            "PlateA/A01_FITC.tif": {
                "filename": "PlateA/A01_FITC.tif",
                "well": "A01",
                "_display_channel": "FITC",
            },
        }
        self.updated = {}

    def _extract_well_id(self, metadata: dict) -> str:
        return metadata["well"]

    def _update_table_with_filtered_items(self, files_dict):
        self.updated = files_dict


def test_image_browser_filter_controller_applies_folder_well_and_column_filters():
    browser = _Browser()
    controller = ImageBrowserFilterController(browser)

    controller.apply_combined_filters()

    assert set(browser.updated) == {
        "PlateA/A01_DAPI.tif",
        "PlateA_results/A01_DAPI.csv",
    }

