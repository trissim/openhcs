# Pyclesperanto Function Inventory & Dimensional Analysis

## Overview
Complete inventory of all pyclesperanto filter functions categorized by dimensional capabilities for creating 2D/3D wrapper system.

**Total Functions Identified**: 300+ functions across 7 tiers

## Dimensional Categories

### 🔵 **DIMENSION-AGNOSTIC FUNCTIONS** (Work with both 2D and 3D)
*These functions work on any dimensional input and should have both 2D and 3D wrappers*

#### **Tier 1 - Basic Operations (78 functions)**

**Arithmetic & Logic:**
- `absolute()` - Absolute value
- `add_image_and_scalar()` - Add scalar to image
- `add_images_weighted()` - Weighted addition
- `divide_images()` - Element-wise division
- `divide_scalar_by_image()` - Scalar divided by image
- `multiply_image_and_scalar()` - Multiply by scalar
- `multiply_images()` - Element-wise multiplication
- `power()` - Power operation
- `power_images()` - Image power
- `reciprocal()` - Reciprocal values
- `square_root()` - Square root
- `cubic_root()` - Cubic root
- `exponential()` - Exponential function
- `logarithm()` - Logarithm
- `sign()` - Sign function
- `nan_to_num()` - Replace NaN with numbers
- `undefined_to_zero()` - Replace undefined with zero

**Binary Operations:**
- `binary_and()` - Logical AND
- `binary_or()` - Logical OR
- `binary_not()` - Logical NOT
- `binary_xor()` - Logical XOR
- `binary_subtract()` - Binary subtraction
- `binary_dilate()` - Binary dilation
- `binary_erode()` - Binary erosion
- `binary_edge_detection()` - Edge detection
- `binary_infsup()` - Infimum supremum
- `binary_supinf()` - Supremum infimum

**Comparison Operations:**
- `equal()` - Element equality
- `equal_constant()` - Equal to constant
- `greater()` - Greater than
- `greater_constant()` - Greater than constant
- `greater_or_equal()` - Greater or equal
- `greater_or_equal_constant()` - Greater or equal constant
- `smaller()` - Smaller than
- `smaller_constant()` - Smaller than constant
- `smaller_or_equal()` - Smaller or equal
- `smaller_or_equal_constant()` - Smaller or equal constant
- `not_equal()` - Not equal
- `not_equal_constant()` - Not equal constant

**Filtering & Morphology:**
- `gaussian_blur()` - Gaussian blur filter
- `convolve()` - Convolution operation
- `median()` - Median filter
- `mean_filter()` - Mean filter
- `maximum_filter()` - Maximum filter
- `minimum_filter()` - Minimum filter
- `variance_filter()` - Variance filter
- `dilation()` - Morphological dilation
- `erosion()` - Morphological erosion
- `laplace()` - Laplacian filter
- `sobel()` - Sobel edge detection

**Box/Sphere Filters:**
- `dilate_box()` - Box dilation
- `dilate_sphere()` - Sphere dilation
- `erode_box()` - Box erosion
- `erode_sphere()` - Sphere erosion
- `maximum_box()` - Box maximum
- `maximum_sphere()` - Sphere maximum
- `minimum_box()` - Box minimum
- `minimum_sphere()` - Sphere minimum
- `mean_box()` - Box mean
- `mean_sphere()` - Sphere mean
- `median_box()` - Box median
- `median_sphere()` - Sphere median
- `variance_box()` - Box variance
- `variance_sphere()` - Sphere variance
- `mode_box()` - Box mode
- `mode_sphere()` - Sphere mode
- `laplace_box()` - Box Laplacian
- `laplace_diamond()` - Diamond Laplacian

**Grayscale Morphology:**
- `grayscale_dilate()` - Grayscale dilation
- `grayscale_erode()` - Grayscale erosion

**Nonzero Operations:**
- `nonzero_maximum()` - Nonzero maximum
- `nonzero_maximum_box()` - Nonzero box maximum
- `nonzero_maximum_diamond()` - Nonzero diamond maximum
- `nonzero_minimum()` - Nonzero minimum
- `nonzero_minimum_box()` - Nonzero box minimum
- `nonzero_minimum_diamond()` - Nonzero diamond minimum
- `onlyzero_overwrite_maximum()` - Overwrite maximum at zeros
- `onlyzero_overwrite_maximum_box()` - Box overwrite maximum
- `onlyzero_overwrite_maximum_diamond()` - Diamond overwrite maximum

**Image Manipulation:**
- `copy()` - Copy image
- `crop()` - Crop image
- `flip()` - Flip image
- `pad()` - Pad image
- `unpad()` - Remove padding
- `paste()` - Paste image
- `mask()` - Apply mask
- `mask_label()` - Mask by label
- `circular_shift()` - Circular shift
- `range()` - Value range
- `clip()` - Clip values

**Value Replacement:**
- `replace_intensity()` - Replace intensity
- `replace_intensities()` - Replace multiple intensities
- `replace_value()` - Replace value
- `replace_values()` - Replace multiple values
- `set()` - Set values
- `set_image_borders()` - Set border values

#### **Tier 2 - Composite Operations (35 functions)**

**Morphological Operations:**
- `binary_closing()` - Binary closing
- `binary_opening()` - Binary opening
- `closing()` - Morphological closing
- `closing_box()` - Box closing
- `closing_sphere()` - Sphere closing
- `opening()` - Morphological opening
- `opening_box()` - Box opening
- `opening_sphere()` - Sphere opening
- `grayscale_closing()` - Grayscale closing
- `grayscale_opening()` - Grayscale opening

**Top/Bottom Hat:**
- `top_hat()` - Top hat transform
- `top_hat_box()` - Box top hat
- `top_hat_sphere()` - Sphere top hat
- `bottom_hat()` - Bottom hat transform
- `bottom_hat_box()` - Box bottom hat
- `bottom_hat_sphere()` - Sphere bottom hat

**Image Processing:**
- `absolute_difference()` - Absolute difference
- `add_images()` - Add images
- `subtract_images()` - Subtract images
- `squared_difference()` - Squared difference
- `difference_of_gaussian()` - DoG filter
- `divide_by_gaussian_background()` - Background division
- `subtract_gaussian_background()` - Background subtraction
- `invert()` - Invert image
- `square()` - Square values
- `clip()` - Clip values

**Statistical Operations:**
- `standard_deviation()` - Standard deviation
- `standard_deviation_box()` - Box standard deviation
- `standard_deviation_sphere()` - Sphere standard deviation

**Detection:**
- `detect_maxima()` - Detect local maxima
- `detect_maxima_box()` - Box maxima detection
- `detect_minima()` - Detect local minima
- `detect_minima_box()` - Box minima detection

**Utility:**
- `degrees_to_radians()` - Convert degrees to radians
- `radians_to_degrees()` - Convert radians to degrees
- `crop_border()` - Crop border
- `sub_stack()` - Extract sub-stack
- `reduce_stack()` - Reduce stack

### 🔴 **3D-SPECIFIC FUNCTIONS** (Require 3D input)
*These functions only work with 3D data and need 3D wrappers only*

#### **Projection Operations (12 functions)**
- `maximum_x_projection()` - Maximum projection along X
- `maximum_y_projection()` - Maximum projection along Y
- `maximum_z_projection()` - Maximum projection along Z
- `minimum_x_projection()` - Minimum projection along X
- `minimum_y_projection()` - Minimum projection along Y
- `minimum_z_projection()` - Minimum projection along Z
- `mean_x_projection()` - Mean projection along X
- `mean_y_projection()` - Mean projection along Y
- `mean_z_projection()` - Mean projection along Z
- `sum_x_projection()` - Sum projection along X
- `sum_y_projection()` - Sum projection along Y
- `sum_z_projection()` - Sum projection along Z
- `std_z_projection()` - Standard deviation Z projection
- `extended_depth_of_focus_variance_projection()` - Extended DOF projection

#### **3D Gradients (3 functions)**
- `gradient_x()` - Gradient along X axis
- `gradient_y()` - Gradient along Y axis
- `gradient_z()` - Gradient along Z axis

#### **3D Slice Operations (6 functions)**
- `copy_slice()` - Copy 2D slice from 3D
- `copy_horizontal_slice()` - Copy horizontal slice
- `copy_vertical_slice()` - Copy vertical slice
- `set_plane()` - Set plane values

#### **3D Transpose Operations (3 functions)**
- `transpose_xy()` - Transpose XY dimensions
- `transpose_xz()` - Transpose XZ dimensions
- `transpose_yz()` - Transpose YZ dimensions

#### **3D Position Operations (6 functions)**
- `x_position_of_maximum_x_projection()` - X position of max in X projection
- `x_position_of_minimum_x_projection()` - X position of min in X projection
- `y_position_of_maximum_y_projection()` - Y position of max in Y projection
- `y_position_of_minimum_y_projection()` - Y position of min in Y projection
- `z_position_of_maximum_z_projection()` - Z position of max in Z projection
- `z_position_of_minimum_z_projection()` - Z position of min in Z projection
- `z_position_projection()` - Z position projection

#### **3D Concatenation (3 functions)**
- `concatenate_along_x()` - Concatenate along X axis
- `concatenate_along_y()` - Concatenate along Y axis
- `concatenate_along_z()` - Concatenate along Z axis

#### **3D Ramp Generation (3 functions)**
- `set_ramp_x()` - Set X ramp
- `set_ramp_y()` - Set Y ramp
- `set_ramp_z()` - Set Z ramp

### 🟡 **2D-SPECIFIC FUNCTIONS** (Work best with 2D input)
*These functions are optimized for 2D data but may work with 3D slice-by-slice*

#### **Matrix Operations (4 functions)**
- `multiply_matrix()` - Matrix multiplication
- `generate_distance_matrix()` - Distance matrix generation
- `set_column()` - Set matrix column
- `set_row()` - Set matrix row

#### **2D Coordinate Operations (3 functions)**
- `set_where_x_equals_y()` - Set where X equals Y
- `set_where_x_greater_than_y()` - Set where X > Y
- `set_where_x_smaller_than_y()` - Set where X < Y

#### **Position-based Operations (4 functions)**
- `multiply_image_and_position()` - Multiply by position
- `read_values_from_positions()` - Read at positions
- `write_values_to_positions()` - Write at positions
- `set_nonzero_pixels_to_pixelindex()` - Set nonzero to pixel index

### 🟢 **LABEL-SPECIFIC FUNCTIONS** (Work with label images)
*These functions work with labeled images and are dimension-agnostic*

#### **Tier 3-7 Label Operations (50+ functions)**

**Label Analysis:**
- `statistics_of_labelled_pixels()` - Label statistics
- `statistics_of_background_and_labelled_pixels()` - Background + label stats
- `centroids_of_labels()` - Label centroids
- `center_of_mass()` - Center of mass
- `bounding_box()` - Label bounding boxes
- `label_bounding_box()` - Individual label bounding box

**Label Filtering:**
- `exclude_labels()` - Exclude specific labels
- `exclude_labels_on_edges()` - Exclude edge labels
- `remove_labels()` - Remove specific labels
- `remove_labels_on_edges()` - Remove edge labels
- `exclude_large_labels()` - Exclude large labels
- `exclude_small_labels()` - Exclude small labels
- `remove_large_labels()` - Remove large labels
- `remove_small_labels()` - Remove small labels
- `exclude_labels_outside_size_range()` - Size range filtering
- `filter_label_by_size()` - Filter by size
- `exclude_labels_with_map_values_out_of_range()` - Map value filtering
- `exclude_labels_with_map_values_within_range()` - Map value filtering
- `remove_labels_with_map_values_out_of_range()` - Remove by map values
- `remove_labels_with_map_values_within_range()` - Remove by map values

**Label Morphology:**
- `dilate_labels()` - Dilate labels
- `erode_labels()` - Erode labels
- `erode_connected_labels()` - Erode connected labels
- `closing_labels()` - Close labels
- `opening_labels()` - Open labels

**Label Generation:**
- `connected_component_labeling()` - Connected components
- `connected_components_labeling()` - Connected components (alias)
- `label_spots()` - Label detected spots
- `voronoi_labeling()` - Voronoi labeling
- `masked_voronoi_labeling()` - Masked Voronoi labeling
- `gauss_otsu_labeling()` - Gaussian + Otsu labeling
- `eroded_otsu_labeling()` - Eroded Otsu labeling
- `voronoi_otsu_labeling()` - Voronoi + Otsu labeling

**Label Utilities:**
- `combine_labels()` - Combine label images
- `relabel_sequential()` - Sequential relabeling
- `reduce_labels_to_centroids()` - Reduce to centroids
- `reduce_labels_to_label_edges()` - Reduce to edges
- `detect_label_edges()` - Detect label edges
- `extend_labeling_via_voronoi()` - Extend via Voronoi
- `flag_existing_labels()` - Flag existing labels

**Label Maps:**
- `label_pixel_count_map()` - Pixel count map
- `pixel_count_map()` - Pixel count map (alias)
- `maximum_intensity_map()` - Maximum intensity map
- `minimum_intensity_map()` - Minimum intensity map
- `mean_intensity_map()` - Mean intensity map
- `standard_deviation_intensity_map()` - Std dev intensity map
- `extension_ratio_map()` - Extension ratio map
- `maximum_extension_map()` - Maximum extension map
- `mean_extension_map()` - Mean extension map

### 🟣 **TRANSFORMATION FUNCTIONS** (Geometric transformations)
*These functions perform geometric transformations and are dimension-agnostic*

#### **Tier 7 Transformations (8 functions)**
- `affine_transform()` - Affine transformation
- `rigid_transform()` - Rigid transformation
- `rotate()` - Rotation
- `scale()` - Scaling
- `translate()` - Translation

### 🟠 **STATISTICAL/ANALYSIS FUNCTIONS** (Global analysis)
*These functions perform global analysis and return scalar values*

#### **Global Statistics (15 functions)**
- `maximum_of_all_pixels()` - Global maximum
- `minimum_of_all_pixels()` - Global minimum
- `minimum_of_masked_pixels()` - Masked minimum
- `mean_of_all_pixels()` - Global mean
- `sum_of_all_pixels()` - Global sum
- `sum_reduction_x()` - Sum reduction along X
- `maximum_position()` - Position of maximum
- `minimum_position()` - Position of minimum
- `histogram()` - Image histogram
- `jaccard_index()` - Jaccard index
- `array_equal()` - Array equality test
- `mean_squared_error()` - MSE calculation
- `threshold_otsu()` - Otsu thresholding

#### **Advanced Analysis (8 functions)**
- `hessian_eigenvalues()` - Hessian eigenvalues
- `large_hessian_eigenvalue()` - Large Hessian eigenvalue
- `small_hessian_eigenvalue()` - Small Hessian eigenvalue
- `local_cross_correlation()` - Local cross-correlation
- `generate_binary_overlap_matrix()` - Binary overlap matrix
- `generate_touch_matrix()` - Touch matrix
- `count_touching_neighbors()` - Count touching neighbors
- `morphological_chan_vese()` - Chan-Vese segmentation

#### **Utility Functions (8 functions)**
- `clahe()` - CLAHE enhancement
- `gamma_correction()` - Gamma correction
- `block_enumerate()` - Block enumeration
- `labelled_spots_to_pointlist()` - Spots to points
- `spots_to_pointlist()` - Spots to points (alias)

## WRAPPER SYSTEM DESIGN

### **Naming Convention**
```python
# For dimension-agnostic functions:
def function_name_2d(image_2d, **kwargs) -> Array:
    """2D wrapper for pyclesperanto.function_name"""

def function_name_3d(image_3d, **kwargs) -> Array:
    """3D wrapper for pyclesperanto.function_name"""

# For dimension-specific functions:
def function_name_3d(image_3d, **kwargs) -> Array:
    """3D-only wrapper for pyclesperanto.function_name"""
```

### **Function Count Summary**
- **Dimension-Agnostic**: ~150 functions (need both 2D and 3D wrappers)
- **3D-Specific**: ~40 functions (need 3D wrapper only)
- **2D-Specific**: ~15 functions (need 2D wrapper only)
- **Label-Specific**: ~60 functions (need both 2D and 3D wrappers)
- **Transformation**: ~8 functions (need both 2D and 3D wrappers)
- **Statistical**: ~30 functions (need both 2D and 3D wrappers)

**Total Wrapper Functions to Create**: ~600 wrapper functions
- ~300 functions × 2 (2D + 3D) = 600 wrappers
- ~40 3D-only functions = 40 wrappers
- ~15 2D-only functions = 15 wrappers
- **Grand Total**: ~655 wrapper functions

### **Implementation Priority**
1. **High Priority**: Dimension-agnostic basic operations (Tier 1)
2. **Medium Priority**: Morphological and composite operations (Tier 2)
3. **Medium Priority**: 3D-specific projections and gradients
4. **Low Priority**: Label-specific operations (Tiers 3-7)
5. **Low Priority**: Statistical and analysis functions

### **Decorator Requirements**
- GPU memory management
- Input validation (2D vs 3D)
- Error handling and cleanup
- Performance monitoring
- Type conversion (numpy ↔ cle.Array)
