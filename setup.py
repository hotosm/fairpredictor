import io

from setuptools import find_packages, setup

with io.open("README.md", encoding="utf-8") as f:
    readme = f.read()

setup(
    name="fairpredictor",
    version="0.0.9",
    url="https://github.com/kshitijrajsharma/fairpredictor",
    author="Kshitij Raj Sharma",
    author_email="skshitizraj@gmail.com",
    description="A package for running predictions using fAIr",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=[
        "numpy",
        "tensorflow",
        "gdal",
        "rtree",
        "tqdm<=4.62.3",
        "pandas==1.5.3",
        "Pillow<=9.0.1",
        "geopandas<=0.10.2",
        "shapely",
        "rasterio",
    ],
)
