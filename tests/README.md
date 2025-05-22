# EZStitcher Tests

This directory contains tests for the EZStitcher package. The tests cover core functionality including:

1. **Image Processing** - Tests for image processing functions like blur, edge detection, and normalization
2. **Focus Detection** - Tests for focus quality detection algorithms
3. **Stitching** - Tests for image stitching and position calculation
4. **Z-Stack Handling** - Tests for Z-stack organization and projection creation
5. **Integration** - Tests for the full workflow

## Test Organization

The tests are organized into the following directories:

- **`unit/`**: Unit tests for individual components
- **`integration/`**: Integration tests for the full workflow
- **`generators/`**: Synthetic data generators for testing

### Unit Tests

These tests focus on testing individual components in isolation:

- **`test_image_locator_integration.py`**: Tests for the ImageLocator class, which handles image location operations.
- **`test_microscope_auto_detection.py`**: Tests for microscope type auto-detection.

### Integration Tests

These tests focus on testing how components work together:

- **`test_synthetic_imagexpress_auto.py`**: Tests for the full workflow using synthetic ImageXpress data.
- **`test_synthetic_opera_phenix_auto.py`**: Tests for the full workflow using synthetic Opera Phenix data.
- **`test_auto_config.py`**: Tests for automatic configuration detection and handling.

## Running the Tests

### Setup

Before running the tests, make sure you have installed the package in development mode:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install the package in development mode
pip install -e .

# Install test dependencies
pip install pytest pytest-cov
```

### Running All Tests

You can run all tests using the provided script:

```bash
python run_tests.py
```

Or using pytest directly:

```bash
pytest tests/
```

### Running Specific Tests

To run a specific test file:

```bash
pytest tests/test_image_processing.py
```

To run a specific test class:

```bash
pytest tests/test_image_processing.py::TestImageProcessing
```

To run a specific test method:

```bash
pytest tests/test_image_processing.py::TestImageProcessing::test_blur
```

### Test Coverage

To generate a test coverage report:

```bash
pytest --cov=ezstitcher tests/
```

For a detailed HTML report:

```bash
pytest --cov=ezstitcher --cov-report=html tests/
```

## Troubleshooting

If you encounter issues with NumPy or other dependencies:

1. Make sure you're using the correct Python version (3.6+)
2. Try reinstalling NumPy: `pip uninstall numpy && pip install numpy`
3. Check that all dependencies are installed: `pip install -r requirements.txt`

## Test Data Management

Test data is managed as follows:

1. All test data is stored in the `/tests/tests_data/` directory
2. Each test file has its own subdirectory
3. Each test method has its own subdirectory
4. A copy of the original data is kept with an `_original` suffix
5. Test data is cleaned up before each test run

## Adding New Tests

When adding new tests:

1. Follow the existing pattern of test files
2. Use descriptive test method names that explain what is being tested
3. Include assertions that verify both the functionality and edge cases
4. Add the new test file to this README if it tests a new component

## Test Organization Guidelines

1. **Isolation**: Each test should be isolated from other tests. Tests should not depend on the state created by other tests.
2. **Cleanup**: Tests should clean up after themselves, removing any temporary files or directories they create.
3. **Documentation**: Tests should include docstrings explaining what they're testing and how they're testing it.
4. **Synthetic Data**: Tests should use synthetic data rather than real data to ensure reproducibility and to avoid dependencies on external data.
5. **Test Directory Structure**: Each test should create its own test data directory with a unique name to avoid interference between tests.
