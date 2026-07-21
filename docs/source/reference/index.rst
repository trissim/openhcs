Reference Documentation
=======================

Complete reference documentation for OpenHCS configuration formats, syntax specifications, and technical details.

.. toctree::
   :maxdepth: 2

   experimental_config_syntax

Configuration References
------------------------

- :doc:`experimental_config_syntax` - Complete syntax reference for Excel-based experimental design configuration

File Format Specifications
--------------------------

**Experimental Configuration**:
- Excel-based configuration format for complex experimental designs
- Support for biological and technical replicates
- Multi-plate experiment configuration
- Control well definition and normalization

**Supported Microscope Formats**:
- ThermoFisher CX5 format (EDDU_CX5)
- MetaXpress format (EDDU_metaxpress)

Example files
-------------

Download complete example configuration files:

- :download:`experimental_config_example.xlsx <../examples/experimental_config_example.xlsx>` - Working experimental configuration example

Quick Reference
---------------

**Essential Configuration Elements**:

.. code-block:: text

   N                    3                    # Number of biological replicates
   Scope               EDDU_metaxpress       # Microscope format
   
   Condition           Drug_Treatment
   Dose                0    10   50   100
   Wells1              A01  A02  A03  A04    # Biological replicate 1
   Plate Group         1    1    1    1

**Common Patterns**:
- Technical replicates: Multiple Wells1 rows for same biological replicate
- Multi-plate: Different Plate Group numbers for wells
- Controls: Separate Controls block with Group N assignments

See the complete syntax reference for detailed specifications and advanced features.
