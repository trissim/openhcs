import json
from pathlib import Path

import numpy as np
from polystore.base import ensure_storage_registry, storage_registry
from polystore.filemanager import FileManager


def bioformats_filemanager() -> FileManager:
    ensure_storage_registry()
    return FileManager(dict(storage_registry))


def write_bioformats_manifest_fixture(root: Path) -> np.ndarray:
    root.mkdir(parents=True, exist_ok=True)
    stack = np.arange(1 * 1 * 2 * 3 * 4, dtype=np.uint16).reshape(1, 1, 2, 3, 4)
    np.save(root / "stack.npy", stack)
    payload = {
        "plates": [
            {
                "name": "fixture",
                "wells": [
                    {
                        "row": 0,
                        "column": 0,
                        "samples": [
                            {
                                "image_id": "image:0",
                                "index": 0,
                            }
                        ],
                    }
                ],
            }
        ],
        "images": [
            {
                "image_id": "image:0",
                "source_path": "stack.npy",
                "series_index": 0,
                "reader": "npy",
                "channel_names": ["DAPI", "GFP"],
                "pixel_size": 0.5,
                "pixels": {
                    "size_c": 2,
                    "size_z": 1,
                    "size_t": 1,
                    "planes": [
                        {"c": 1, "z": 1, "t": 1, "index": 0},
                        {"c": 2, "z": 1, "t": 1, "index": 1},
                    ],
                },
            }
        ],
    }
    (root / "bioformats_spw.json").write_text(json.dumps(payload), encoding="utf-8")
    return stack
