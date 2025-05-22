# OpenHCS: Open High-Content Screening

OpenHCS is an open-source platform for high-content screening image analysis, designed to provide a flexible, extensible framework for processing and analyzing microscopy images.

## Overview

OpenHCS provides a comprehensive suite of tools for:

- Image processing and analysis
- Data management and organization
- Pipeline creation and execution
- Result visualization and reporting

The platform is designed to be modular, allowing users to create custom processing pipelines tailored to their specific needs.

## Key Features

- **Modular Architecture**: Easily extend and customize the platform with plugins
- **Pipeline-Based Processing**: Create, save, and reuse processing pipelines
- **Multi-Format Support**: Handle various microscopy image formats
- **Batch Processing**: Process large datasets efficiently
- **GPU Acceleration**: Leverage GPU capabilities for faster processing
- **Visualization Tools**: Analyze and visualize results
- **Extensible API**: Integrate with other tools and platforms

## Installation

```bash
# Clone the repository
git clone https://github.com/trissim/openhcs.git
cd openhcs

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Usage

### Basic Usage

```python
from openhcs import Pipeline
from openhcs.steps import LoadImages, DetectCells, MeasureIntensity, ExportResults

# Create a pipeline
pipeline = Pipeline()

# Add processing steps
pipeline.add_step(LoadImages(input_dir="path/to/images"))
pipeline.add_step(DetectCells(method="threshold"))
pipeline.add_step(MeasureIntensity(channels=["DAPI", "GFP"]))
pipeline.add_step(ExportResults(output_dir="path/to/results"))

# Run the pipeline
pipeline.run()
```

### Command Line Interface

```bash
# Run a pipeline
openhcs run --pipeline path/to/pipeline.json --input path/to/images --output path/to/results

# Create a new pipeline
openhcs create-pipeline --output path/to/pipeline.json

# List available processing steps
openhcs list-steps
```

## Components

### Core Components

- **Pipeline**: The main processing unit that orchestrates the execution of steps
- **Step**: Individual processing operations that can be chained together
- **FileManager**: Handles file operations across different storage backends
- **DataModel**: Represents and manages the data being processed
- **Visualization**: Tools for visualizing results and data

### Plugins

OpenHCS supports plugins for extending its functionality:

- **Image Processing**: Custom image processing algorithms
- **Cell Detection**: Methods for detecting and segmenting cells
- **Feature Extraction**: Extract features from detected objects
- **Data Analysis**: Analyze and interpret results
- **Visualization**: Custom visualization methods
- **Export/Import**: Support for additional file formats

## Development

### Project Structure

```
openhcs/
├── core/                  # Core functionality
├── io/                    # Input/output operations
├── steps/                 # Processing steps
├── plugins/               # Plugin system
├── visualization/         # Visualization tools
├── utils/                 # Utility functions
├── cli.py                 # Command-line interface
└── __main__.py            # Entry point
```

### Creating Custom Steps

You can create custom processing steps by subclassing the `Step` class:

```python
from openhcs.core import Step

class MyCustomStep(Step):
    def __init__(self, param1=None, param2=None):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def process(self, data):
        # Implement your processing logic here
        # Modify the data object as needed
        return data
```

### Creating Plugins

You can create plugins to extend OpenHCS functionality:

```python
from openhcs.plugins import Plugin

class MyPlugin(Plugin):
    @property
    def name(self):
        return "my_plugin"

    @property
    def version(self):
        return "0.1.0"

    @property
    def description(self):
        return "My custom plugin"

    def initialize(self):
        # Initialize your plugin
        pass

    def get_steps(self):
        # Return a list of custom steps provided by this plugin
        return [MyCustomStep]
```

## Semantic Matrix Analyzer

OpenHCS includes the Semantic Matrix Analyzer (SMA), a tool for analyzing codebases using AST to create semantically dense matrices that correlate intent with AST correctness. SMA is designed to be universally applicable to any codebase while reducing cognitive load for humans and increasing correctness through AI agents bounded by semantic and AST correctness.

For more information about SMA, see the [SMA README](semantic_matrix_analyzer/README.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
