import io

from setuptools import find_packages, setup

with io.open("README.md", encoding="utf-8") as f:
    readme = f.read()

setup(
    name="fairpredictor",
    version="0.0.37",
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
        "requests<=2.31.0",
        "Pillow<=9.1.0",
        "rtree>=1.0.0,<=1.1.0",
        "tqdm>=4.0.0,<=4.67.0",
        "geopandas>=0.12.0,<=0.14.4",
        "shapely>=1.0.0,<=2.0.2",
        "rasterio>=1.0.0,<=1.3.8",
        "orthogonalizer<=0.0.4",
        # "hot-fair-utilities==2.0.2",
    ],
)
