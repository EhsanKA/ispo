#!/usr/bin/env python3
"""Setup script for ISPO package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ispo",
    version="0.1.0",
    author="ISPO Team",
    description="In-Silico Perturbation Optimization Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EhsanKA/ispo",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ispo=main:main",
            "ispo-baseline=ispo.scripts.run_baseline:main",
            "ispo-optimized=ispo.scripts.run_optimized:main",
            "ispo-bayesian=ispo.scripts.run_bayesian:main",
            "ispo-evaluate-sciplex2=ispo.scripts.evaluate_sciplex2:main",
        ],
    },
)



