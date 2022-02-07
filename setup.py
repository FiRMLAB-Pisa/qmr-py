# -*- coding: utf-8 -*-
""" Setup script for stage-py."""
import os

import setuptools

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md")) as f:
    long_description = f.read()

install_requires = ["numpy", "pydicom"]

setuptools.setup(
    name="stagepy",
    version=0.1,

    description="Python-based processing for STAGE MRI mapping (and other quantitative techniques).",
    long_description=long_description,
    long_description_content_type="text/markdown",

    url="https://github.com/FiRMLAB-Pisa/stage-py",

    author="Matteo Cencini",
    author_email="matteo.cencini@gmail.com",

    license="MIT",

    classifiers=[
        "Development Status ::2 - Pre-Alpha",

        "Intended Audience :: Education",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",

        "License :: OSI Approved :: MIT License",

        "Programming Language :: Python :: 3",

        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Physics",
    ],

    keywords=["MRI", "quantitative-mri"],

    packages=["stagepy"],
    package_dir={"stagepy": "stagepy"},
    python_requires=">=3.7",

    install_requires=install_requires,
)
