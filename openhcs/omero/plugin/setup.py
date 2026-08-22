#!/usr/bin/env python
"""
OMERO.web plugin for OpenHCS integration.

This plugin adds OpenHCS processing capabilities to OMERO.web,
allowing users to submit GPU processing pipelines directly from
the browser interface.
"""

from setuptools import setup, find_packages

setup(
    name="omero-openhcs",
    version="0.1.0",
    description="OMERO.web plugin for OpenHCS GPU processing",
    long_description=__doc__,
    author="OpenHCS Development Team",
    author_email="",
    url="https://github.com/OpenHCSDev/openhcs",
    license="MIT",
    packages=find_packages(),
    package_data={
        "omero_openhcs": [
            "templates/omero_openhcs/*.html",
            "templates/omero_openhcs/webclient_plugins/*.html",
        ]
    },
    include_package_data=False,
    install_requires=[
        "omero-web>=5.6.0",
        "pyzmq>=25.0.0",  # Required for communication with OpenHCS execution server
        "zmqruntime>=0.2.6",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Web Environment",
        "Framework :: Django",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
