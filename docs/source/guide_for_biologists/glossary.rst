Glossary
========

The shared :doc:`../appendices/glossary` defines microscopy, pipeline,
compiler, source, artifact, and runtime terms used throughout the user and
architecture guides.

Three distinctions matter especially when editing a step:

- ``variable_components`` says what varies along the image stack axis;
- ``group_by`` partitions arrays after the stack exists;
- ``ProcessingContract`` says whether a function is plane-local or depends on
  the whole stack.
