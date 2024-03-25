# Standard library imports
import os
import uuid
from glob import glob
from pathlib import Path

# Third party imports
import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.merge import merge
from shapely.geometry import Polygon, shape
from tqdm import tqdm

TOLERANCE = 0.5
AREA_THRESHOLD = 5


def vectorize(
    input_path: str,
    output_path: str = None,
    tolerance: float = 0.5,
    area_threshold: float = 5,
) -> None:
    """Polygonize raster tiles from the input path.

    Note that as input, we are expecting GeoTIF images with EPSG:3857 as
    CRS here. CRS of the resulting GeoJSON file will be EPSG:4326.

    Args:
        input_path: Path of the directory where the TIF files are stored.
        output_path: Path of the output file.
        tolerance (float, optional): Tolerance parameter for simplifying polygons. Defaults to 0.5 m. Percentage Tolerance = (Tolerance in Meters / Arc Length in Meters ​)×100
        area_threshold (float, optional): Threshold for filtering polygon areas. Defaults to 5 sqm.

    Example::

        vectorize("data/masks_v2/4", "labels.geojson", tolerance=0.5, area_threshold=5)
    """
    if output_path is None:
        # Generate a temporary download path using a UUID
        temp_dir = os.path.join("/tmp", "vectorize", str(uuid.uuid4()))
        os.makedirs(temp_dir, exist_ok=True)
        output_path = os.path.join(temp_dir, "prediction.geojson")
    base_path = Path(output_path).parents[0]
    base_path.mkdir(exist_ok=True, parents=True)

    raster_paths = glob(f"{input_path}/*.tif")
    with rio.open(raster_paths[0]) as src:
        kwargs = src.meta.copy()

    rasters = [rio.open(path) for path in raster_paths]
    mosaic, output = merge(
        rasters,
        resampling=Resampling.nearest,
    )

    # Close raster files after merging
    for raster in rasters:
        raster.close()

    polygons = [shape(s) for s, _ in shapes(mosaic, transform=output)]
    gs = gpd.GeoSeries(polygons, crs=kwargs["crs"])

    # Explode MultiPolygons
    gs = gs.explode()

    # Filter by area threshold
    gs = gs[gs.area >= area_threshold]

    gs = gs.simplify(tolerance)
    if gs.empty:
        raise ValueError("No Features Found")
    gs.to_crs("EPSG:4326").to_file(output_path)
    return output_path
