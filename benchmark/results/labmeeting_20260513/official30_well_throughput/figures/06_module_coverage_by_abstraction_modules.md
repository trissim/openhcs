# Module Coverage by Shared Abstractions

## Explicitly covered (53)

- `Watershed` via `AdvancedMasked3D`
- `Closing` via `AdvancedUnmasked3D`
- `ErodeImage` via `AdvancedUnmasked3D`
- `ErodeObjects` via `AdvancedUnmasked3D`
- `Medianfilter` via `AdvancedUnmasked3D`
- `RemoveHoles` via `AdvancedUnmasked3D`
- `CalculateMath` via `DataToolsUnmasked2D`
- `DisplayDataOnImage` via `DataToolsUnmasked2D`
- `ExportToDatabase` via `FileProcessingMasked3D`
- `SaveImages` via `FileProcessingMasked3D`
- `Align` via `ImageProcessingMasked2D`
- `CorrectIlluminationCalculate` via `ImageProcessingMasked2D`
- `Crop` via `ImageProcessingMasked2D`
- `EnhanceEdges` via `ImageProcessingMasked2D`
- `OverlayObjects` via `ImageProcessingMasked2D`
- `Smooth` via `ImageProcessingMasked2D`
- `EnhanceOrSuppressFeatures` via `ImageProcessingMasked3D`
- `ImageMath` via `ImageProcessingMasked3D`
- `MaskImage` via `ImageProcessingMasked3D`
- `RescaleIntensity` via `ImageProcessingMasked3D`
- `Resize` via `ImageProcessingMasked3D`
- `Threshold` via `ImageProcessingMasked3D`
- `ColorToGray` via `ImageProcessingUnmasked2D`
- `GrayToColor` via `ImageProcessingUnmasked2D`
- `Tile` via `ImageProcessingUnmasked2D`
- `CorrectIlluminationApply` via `ImageProcessingUnmasked2DApply`
- `OverlayOutlines` via `ImageProcessingUnmasked3D`
- `MeasureObjectIntensityDistribution` via `MeasurementMasked2D`
- `MeasureColocalization` via `MeasurementMasked3D`
- `MeasureGranularity` via `MeasurementMasked3D`
- `MeasureImageAreaOccupiedBinary` via `MeasurementMasked3D`
- `MeasureImageIntensity` via `MeasurementMasked3D`
- `MeasureImageQuality` via `MeasurementMasked3D`
- `MeasureObjectIntensity` via `MeasurementMasked3D`
- `MeasureTexture` via `MeasurementMasked3D`
- `MeasureObjectNeighbors` via `MeasurementUnmasked3D`
- `MeasureObjectSizeShape` via `MeasurementUnmasked3D`
- `ExpandOrShrinkObjects` via `ObjectProcessingMasked2D`
- `IdentifyObjectsInGrid` via `ObjectProcessingMasked2D`
- `IdentifyPrimaryObjects` via `ObjectProcessingMasked2D`
- `IdentifySecondaryObjects` via `ObjectProcessingMasked2D`
- `IdentifyTertiaryObjects` via `ObjectProcessingMasked2D`
- `MaskObjects` via `ObjectProcessingMasked2D`
- `TrackObjects` via `ObjectProcessingMasked2D`
- `ConvertObjectsToImage` via `ObjectProcessingMasked3D`
- `FilterObjects` via `ObjectProcessingMasked3D`
- `RelateObjects` via `ObjectProcessingMasked3D`
- `ClassifyObjectsSingleMeasurement` via `ObjectProcessingUnmasked2D`
- `Combineobjects` via `ObjectProcessingUnmasked3D`
- `ResizeObjects` via `ObjectProcessingUnmasked3D`
- `DefineGridManual` via `OtherUnmasked2D`
- `StraightenWorms` via `WormToolboxMasked2D`
- `UntangleWorms` via `WormToolboxMasked2D`

## Covered by shared abstraction (28)

- `DilateImage` via `AdvancedUnmasked3D` from Closing;ErodeImage;ErodeObjects;RemoveHoles
- `DilateObjects` via `AdvancedUnmasked3D` from Closing;ErodeImage;ErodeObjects;RemoveHoles
- `FillObjects` via `AdvancedUnmasked3D` from Closing;ErodeImage;ErodeObjects;RemoveHoles
- `GaussianFilter` via `AdvancedUnmasked3D` from Closing;ErodeImage;ErodeObjects;RemoveHoles
- `Medialaxis` via `AdvancedUnmasked3D` from Closing;ErodeImage;ErodeObjects;RemoveHoles
- `Morphologicalskeleton` via `AdvancedUnmasked3D` from Closing;ErodeImage;ErodeObjects;RemoveHoles
- `Opening` via `AdvancedUnmasked3D` from Closing;ErodeImage;ErodeObjects;RemoveHoles
- `Reducenoise` via `AdvancedUnmasked3D` from Closing;ErodeImage;ErodeObjects;RemoveHoles
- `ShrinkToObjectCenters` via `AdvancedUnmasked3D` from Closing;ErodeImage;ErodeObjects;RemoveHoles
- `CalculateStatistics` via `DataToolsUnmasked2D` from CalculateMath;DisplayDataOnImage
- `DisplayDensityPlot` via `DataToolsUnmasked2D` from CalculateMath;DisplayDataOnImage
- `DisplayHistogram` via `DataToolsUnmasked2D` from CalculateMath;DisplayDataOnImage
- `DisplayPlatemap` via `DataToolsUnmasked2D` from CalculateMath;DisplayDataOnImage
- `DisplayScatterPlot` via `DataToolsUnmasked2D` from CalculateMath;DisplayDataOnImage
- `FindMaxima` via `DataToolsUnmasked2D` from CalculateMath;DisplayDataOnImage
- `FlagImage` via `DataToolsUnmasked2D` from CalculateMath;DisplayDataOnImage
- `SaveCroppedObjects` via `FileProcessingMasked3D` from ExportToDatabase;SaveImages
- `MakeProjection` via `ImageProcessingMasked2D` from Align;CorrectIlluminationCalculate;Crop;EnhanceEdges;OverlayObjects;Smooth
- `Morph` via `ImageProcessingMasked2D` from Align;CorrectIlluminationCalculate;Crop;EnhanceEdges;OverlayObjects;Smooth
- `FlipAndRotate` via `ImageProcessingUnmasked2D` from ColorToGray;GrayToColor;Tile
- `InvertForPrinting` via `ImageProcessingUnmasked2D` from ColorToGray;GrayToColor;Tile
- `UnmixColors` via `ImageProcessingUnmasked2D` from ColorToGray;GrayToColor;Tile
- `MeasureImageSkeleton` via `MeasurementMasked3D` from MeasureColocalization;MeasureGranularity;MeasureImageIntensity;MeasureImageQuality;MeasureObjectIntensity;MeasureTexture
- `Measureimageoverlap` via `MeasurementMasked3D` from MeasureColocalization;MeasureGranularity;MeasureImageIntensity;MeasureImageQuality;MeasureObjectIntensity;MeasureTexture
- `EditObjectsManually` via `ObjectProcessingMasked2D` from ExpandOrShrinkObjects;IdentifyObjectsInGrid;IdentifyPrimaryObjects;IdentifySecondaryObjects;IdentifyTertiaryObjects;MaskObjects;TrackObjects
- `SplitOrMergeObjects` via `ObjectProcessingMasked2D` from ExpandOrShrinkObjects;IdentifyObjectsInGrid;IdentifyPrimaryObjects;IdentifySecondaryObjects;IdentifyTertiaryObjects;MaskObjects;TrackObjects
- `ConvertImageToObjects` via `ObjectProcessingUnmasked3D` from ResizeObjects
- `IdentifyDeadWorms` via `WormToolboxMasked2D` from StraightenWorms;UntangleWorms

## Not covered (8)

- `ComputeAggregateMeasurements`
- `MatchTemplate` via `AdvancedUnmasked2D`
- `RunImagejMacro` via `AdvancedUnmasked2D`
- `LabelImages` via `FileProcessingUnmasked2D`
- `CreateBatchFiles` via `FileProcessingUnmasked3D`
- `MeasureObjectOverlap` via `MeasurementUnmasked2D`
- `MeasureObjectSkeleton` via `MeasurementUnmasked2D`
- `IdentifyObjectsManually` via `ObjectProcessingUnmasked2D`
