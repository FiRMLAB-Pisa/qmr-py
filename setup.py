# -*- coding: utf-8 -*-
""" Setup script for qmr-py."""
import os

import setuptools

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md")) as f:
    long_description = f.read()

install_requires = ["click==8.0.3", 
                    "matplotlib==3.5.1", 
                    "nibabel==3.2.2", 
                    "numba==0.53.1", 
                    "numpy==1.21.1", 
                    "pydicom==2.2.2", 
                    "pytest==7.0.0", 
                    "scipy==1.7.3", 
                    "scikit-image==0.19.3"
                    "tqdm==4.62.3"]

setuptools.setup(
    name="qmrpy",
    version=0.1,

    description="Python-based quantitative MR inference package.",
    long_description=long_description,
    long_description_content_type="text/markdown",

    url="https://github.com/FiRMLAB-Pisa/qmr-py",

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

    packages=["qmrpy"],
    package_dir={"qmrpy": "qmrpy"},
    python_requires=">=3.7",

    install_requires=install_requires,
    
    entry_points={'console_scripts': ['qmri = qmrpy.app:cli']},


)
