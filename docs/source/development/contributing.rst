Contributing
============

This guide explains how to contribute to EZStitcher.

Setting Up Development Environment
-------------------------------

1. **Fork the repository** on GitHub
2. **Clone your fork**:

   .. code-block:: bash

       git clone https://github.com/your-username/ezstitcher.git
       cd ezstitcher

3. **Create a virtual environment**:

   .. code-block:: bash

       python -m venv .venv
       source .venv/bin/activate  # Linux/macOS
       # or
       .venv\Scripts\activate     # Windows

4. **Install development dependencies**:

   .. code-block:: bash

       pip install -e ".[dev]"

5. **Install pre-commit hooks**:

   .. code-block:: bash

       pre-commit install

Making Changes
------------

1. **Create a new branch** for your changes:

   .. code-block:: bash

       git checkout -b feature/your-feature-name

2. **Make your changes** to the codebase
3. **Run tests** to ensure your changes don't break existing functionality:

   .. code-block:: bash

       pytest

4. **Run linters** to ensure your code follows the project's style guidelines:

   .. code-block:: bash

       flake8
       black .
       isort .

5. **Commit your changes** with a descriptive commit message:

   .. code-block:: bash

       git add .
       git commit -m "Add feature: your feature description"

6. **Push your changes** to your fork:

   .. code-block:: bash

       git push origin feature/your-feature-name

7. **Create a pull request** from your fork to the main repository

Code Style
--------

EZStitcher follows these code style guidelines:

- **PEP 8**: Follow the `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ style guide
- **Black**: Use `Black <https://black.readthedocs.io/>`_ for code formatting
- **isort**: Use `isort <https://pycqa.github.io/isort/>`_ for import sorting
- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Use type hints for function and method signatures

Example of a well-formatted function:

.. code-block:: python

    def process_image(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Process an image with a Gaussian filter.

        Args:
            image: Input image
            sigma: Standard deviation for Gaussian kernel

        Returns:
            Processed image
        """
        # Convert to float for processing
        image_float = image.astype(np.float32)

        # Apply Gaussian filter
        processed = ndimage.gaussian_filter(image_float, sigma=sigma)

        # Convert back to original dtype
        return processed.astype(image.dtype)

Documentation
-----------

All code contributions should include documentation:

1. **Add docstrings** to all modules, classes, and functions:

   .. code-block:: python

       def some_function(param1, param2):
           """
           Brief description of the function.

           Args:
               param1 (type): Description of param1
               param2 (type): Description of param2

           Returns:
               type: Description of return value

           Raises:
               ExceptionType: When and why this exception is raised
           """
           # Function implementation

2. **Update the documentation** if you change existing functionality:

   .. code-block:: bash

       cd docs
       make html

3. **Add examples** for new features:

   .. code-block:: python

       # Example usage of new feature
       from ezstitcher.core import new_feature

       result = new_feature(input_data)
       print(result)

Testing
------

All code contributions should include tests:

1. **Add unit tests** for new functionality:

   .. code-block:: python

       def test_new_feature():
           """Test the new feature."""
           # Test implementation
           result = new_feature(input_data)
           assert result == expected_result

2. **Add integration tests** for new components:

   .. code-block:: python

       def test_new_component_integration():
           """Test the new component in the full pipeline."""
           # Test implementation
           pipeline = Pipeline(new_component)
           result = pipeline.run(input_data)
           assert result == expected_result

3. **Run all tests** before submitting a pull request:

   .. code-block:: bash

       pytest

Pull Request Process
-----------------

1. **Ensure all tests pass** on your local machine
2. **Update the documentation** to reflect your changes
3. **Document your changes** thoroughly in code and documentation
4. **Submit your pull request** with a clear description of the changes
5. **Address any feedback** from the code review
6. **Wait for approval** from a maintainer

Release Process
------------

1. **Update version number** in `ezstitcher/__init__.py`
2. **Prepare a detailed changelog** for the release
3. **Create and push a new tag**:

   .. code-block:: bash

       git tag -a v{version} -m "Release version {version}"
       git push origin v{version}

4. **Build and upload** the package to PyPI manually:

   .. code-block:: bash

       # Build the package
       python -m build

       # Upload to PyPI (you'll need your PyPI token)
       python -m twine upload dist/*

   Note: Make sure you have build and twine installed: `pip install build twine`

Code of Conduct
------------

Please follow these guidelines when contributing to EZStitcher:

1. **Be respectful** of other contributors
2. **Be constructive** in your feedback
3. **Be patient** with new contributors
4. **Be inclusive** and welcoming to all
5. **Be collaborative** and work together to solve problems
