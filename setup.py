# This Python file uses the following encoding: utf-8
import glob

from setuptools import find_packages, setup

setup(
    name="snakemaketools",
    packages=find_packages(),
    version="0.0.1",
    description="Tools for Snakemake",
    long_description="Extend Snakemake to work nicely with longer pipelines.",
    author="Mateusz Krzysztof Łącki & Michał Startek",
    author_email="matteo.lacki@gmail.com",
    url="https://github.com/midiaIDorg/snakemaketools.git",
    keywords=[
        "bioinformatics",
        "shit",
        "boring",
        "wasting my life",
        "lollipop",
        "finding an ignorant in Bruker on high position: not impossible",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    install_requires=[],
    scripts=glob.glob("tools/*.py"),
)