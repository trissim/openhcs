Testing
=======

This guide explains how to test EZStitcher.

Test Organization
----------------

The tests are organized into the following directories:

- **`unit/`**: Unit tests for individual components
- **`integration/`**: Integration tests for the full workflow
- **`generators/`**: Synthetic data generators for testing

Running Tests
------------

To run all tests:

.. code-block:: bash

    pytest

To run a specific test file:

.. code-block:: bash

    pytest tests/unit/test_image_processor.py

To run a specific test class:

.. code-block:: bash

    pytest tests/unit/test_image_processor.py::TestImageProcessor

To run a specific test method:

.. code-block:: bash

    pytest tests/unit/test_image_processor.py::TestImageProcessor::test_blur

Test Coverage
------------

To generate a test coverage report:

.. code-block:: bash

    pytest --cov=ezstitcher tests/

For a detailed HTML report:

.. code-block:: bash

    pytest --cov=ezstitcher --cov-report=html tests/

Writing Tests
------------

When writing tests for EZStitcher, follow these guidelines:

1. **Use pytest fixtures**: Use fixtures to set up test data and dependencies
2. **Test one thing at a time**: Each test should test one specific functionality
3. **Use descriptive names**: Test names should describe what is being tested
4. **Use assertions**: Use assertions to verify expected behavior
5. **Clean up after tests**: Clean up any temporary files or directories created during tests

Here's an example of a unit test:

.. code-block:: python

    import pytest
    import numpy as np
    from ezstitcher.core.image_processor import ImageProcessor

    class TestImageProcessor:
        """Tests for the ImageProcessor class."""

        def test_blur(self):
            """Test the blur method."""
            # Create a test image
            image = np.ones((100, 100), dtype=np.uint16) * 1000
            image[40:60, 40:60] = 5000  # Add a bright square

            # Apply blur
            blurred = ImageProcessor.blur(image, sigma=2.0)

            # Verify that the image was blurred
            assert blurred.shape == image.shape
            assert blurred.dtype == image.dtype
            assert np.mean(blurred[40:60, 40:60]) < 5000  # Blurring should reduce the intensity
            assert np.mean(blurred[40:60, 40:60]) > 1000  # But it should still be brighter than the background

        def test_normalize(self):
            """Test the normalize method."""
            # Create a test image
            image = np.ones((100, 100), dtype=np.uint16) * 1000
            image[40:60, 40:60] = 5000  # Add a bright square

            # Apply normalization
            normalized = ImageProcessor.normalize(image, target_min=0, target_max=65535)

            # Verify that the image was normalized
            assert normalized.shape == image.shape
            assert normalized.dtype == image.dtype
            assert np.min(normalized) == 0
            assert np.max(normalized) == 65535

Here's an example of an integration test:

.. code-block:: python

    import pytest
    import numpy as np
    from pathlib import Path
    from ezstitcher.core.config import PipelineConfig
    from ezstitcher.core.pipeline_orchestrator import PipelineOrchestrator
    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step, PositionGenerationStep, ImageStitchingStep
    from ezstitcher.core.image_processor import ImageProcessor as IP
    from ezstitcher.tests.generators.generate_synthetic_data import SyntheticMicroscopyGenerator

    @pytest.fixture
    def flat_plate_dir(tmp_path):
        """Create synthetic flat plate data for testing."""
        plate_dir = tmp_path / "flat_plate"

        # Generate synthetic data
        generator = SyntheticMicroscopyGenerator(
            output_dir=str(plate_dir),
            grid_size=(3, 3),
            tile_size=(128, 128),
            overlap_percent=10,
            wavelengths=2,
            z_stack_levels=1,  # Flat plate has only 1 Z-level
            cell_size_range=(5, 10),
            wells=["A01", "B01"],
            format="ImageXpress"
        )
        generator.generate_dataset()

        return plate_dir

    def test_pipeline_architecture(flat_plate_dir):
        """Test the pipeline architecture with the orchestrator."""
        # Create configuration
        config = PipelineConfig(
            num_workers=2  # Use 2 worker threads
        )

        # Create orchestrator
        orchestrator = PipelineOrchestrator(
            config=config,
            plate_path=flat_plate_dir
        )

        # Create position generation pipeline
        position_pipeline = Pipeline(
            steps=[
                # Step 1: Flatten Z-stacks (using function tuple for parameters)
                Step(name="Z-Stack Flattening",
                     func=(IP.create_projection, {'method': 'max_projection'}),
                     variable_components=['z_index'],
                     input_dir=orchestrator.workspace_path),  # First step uses workspace_path

                # Step 2: Process channels with a sequence of functions
                Step(name="Image Enhancement",
                     func=[
                         (IP.sharpen, {'amount': 1.5}),
                         IP.stack_percentile_normalize
                     ]),

                # Step 3: Create composite with weights (70% channel 1, 30% channel 2)
                Step(func=(IP.create_composite, {'weights': [0.7, 0.3]}),  # Pass weights as a list
                     variable_components=['channel']),

                # Step 4: Generate positions
                PositionGenerationStep()
            ],
            name="Position Generation Pipeline"
        )

        # Create image assembly pipeline
        assembly_pipeline = Pipeline(
            steps=[
                # Step 1: Process images
                Step(name="Image Processing",
                     func=IP.stack_percentile_normalize,
                     input_dir=orchestrator.workspace_path),

                # Step 2: Stitch images
                ImageStitchingStep()
            ],
            name="Image Assembly Pipeline"
        )

        # Run the orchestrator with the pipelines
        success = orchestrator.run(pipelines=[position_pipeline, assembly_pipeline])

        # Verify that the pipeline ran successfully
        assert success, "Pipeline execution failed"

Generating Test Data
------------------

EZStitcher includes a synthetic data generator for testing. The preferred way to generate test data is using the `SyntheticMicroscopyGenerator` class:

.. code-block:: python

    from ezstitcher.tests.generators.generate_synthetic_data import SyntheticMicroscopyGenerator
    from pathlib import Path

    # Create a generator for synthetic data
    generator = SyntheticMicroscopyGenerator(
        output_dir=str(Path("tests/data/synthetic_plate")),
        grid_size=(3, 3),           # 3x3 grid of tiles
        tile_size=(128, 128),       # Each tile is 128x128 pixels
        overlap_percent=10,         # 10% overlap between tiles
        wavelengths=2,              # 2 channels
        z_stack_levels=3,           # 3 Z-stack levels
        cell_size_range=(5, 10),    # Cell size range for synthetic cells
        wells=["A01", "A02"],       # Generate data for these wells
        format="ImageXpress"        # Use ImageXpress format
    )

    # Generate the dataset
    generator.generate_dataset()

Mocking
-------

When testing components that depend on external resources, use mocking to isolate the component being tested:

.. code-block:: python

    import pytest
    from unittest.mock import Mock, patch
    from pathlib import Path
    from ezstitcher.core.stitcher import Stitcher
    from ezstitcher.core.config import StitcherConfig
    from ezstitcher.core.microscope_interfaces import MicroscopeHandler

    def test_generate_positions_with_mock():
        """Test generate_positions with mocked dependencies."""
        # Create a mock microscope handler
        mock_handler = Mock(spec=MicroscopeHandler)
        mock_handler.parser.parse_filename.return_value = {
            'well': 'A01',
            'site': '1',
            'channel': '1',
            'extension': '.tif'
        }
        mock_handler.parser.construct_filename.return_value = "A01_s{iii}_w1.tif"

        # Create a list of mock image paths
        mock_image_paths = [
            Path("path/to/images/A01_s001_w1.tif"),
            Path("path/to/images/A01_s002_w1.tif"),
            Path("path/to/images/A01_s003_w1.tif"),
            Path("path/to/images/A01_s004_w1.tif")
        ]

        # Create a stitcher with the mock handler
        stitcher = Stitcher(StitcherConfig(), filename_parser=mock_handler)

        # Mock the find_images method to return our mock image paths
        with patch('ezstitcher.core.image_locator.ImageLocator.find_images',
                  return_value=mock_image_paths):

            # Mock the _generate_positions_ashlar method
            with patch.object(stitcher, '_generate_positions_ashlar', return_value=True) as mock_method:
                # Call the method being tested
                result = stitcher.generate_positions(
                    well="A01",
                    image_dir=Path("path/to/images"),
                    positions_path=Path("path/to/positions.csv")
                )

                # Verify that the method was called
                assert mock_method.called

                # Verify the result
                assert result is True

Debugging Tests
--------------

To debug tests, you can use the `--pdb` option to drop into the debugger when a test fails:

.. code-block:: bash

    pytest --pdb

You can also use the `breakpoint()` function to set a breakpoint in your test:

.. code-block:: python

    def test_something():
        # Some test code
        breakpoint()  # Debugger will stop here
        # More test code
