# Detailed Code Definition Matrix
Generated on: 2025-05-23 02:25:03

### Detailed Matrix for `openhcs/processing/backends/pos_gen/ashlar_processor_cupy.py`

| Definition Type | Name | Parent Class | Parameters | Return Type | Lines |
| --- | --- | --- | --- | --- | --- |
| function | _validate_cupy_array |  | array: Any, name: str | <complex_annotation> | 38-59 |
| function | phase_correlation |  | image1: <complex_annotation>, image2: <complex_annotation> | Tuple[float, float] | 62-142 |
| function | gpu_ashlar_align_cupy |  | tiles: <complex_annotation>, num_rows: int, num_cols: int | Tuple[<complex_annotation>, <complex_annotation>] | 147-273 |

### Detailed Matrix for `openhcs/processing/backends/pos_gen/mist_processor_cupy.py`

| Definition Type | Name | Parent Class | Parameters | Return Type | Lines |
| --- | --- | --- | --- | --- | --- |
| function | _validate_cupy_array |  | array: Any, name: str | <complex_annotation> | 40-61 |
| function | phase_correlation |  | image1: <complex_annotation>, image2: <complex_annotation> | Union[Tuple[float, float], Tuple[<complex_annotation>, Tuple[int, int]]] | 64-152 |
| function | extract_patch |  | image: <complex_annotation>, center_y: int, center_x: int, patch_size: int | <complex_annotation> | 155-198 |
| function | mist_compute_tile_positions |  | image_stack: <complex_annotation>, num_rows: int, num_cols: int | Tuple[<complex_annotation>, <complex_annotation>] | 203-497 |

