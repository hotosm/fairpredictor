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
    merge_adjancent_polygons=True,
) -> None:
    """Polygonize raster tiles from the input path.

    Note that as input, we are expecting GeoTIF images with EPSG:3857 as
    CRS here. CRS of the resulting GeoJSON file will be EPSG:4326.

    Args:
        input_path: Path of the directory where the TIF files are stored.
        output_path: Path of the output file.
        tolerance (float, optional): Tolerance parameter for simplifying polygons. Defaults to 0.5 m. Percentage Tolerance = (Tolerance in Meters / Arc Length in Meters ​)×100
        area_threshold (float, optional): Threshold for filtering polygon areas. Defaults to 5 sqm.
        merge_adjancent_polygons(bool,optional) : Merges adjacent self intersecting or containing each other polygons

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
    merged_polygons = polygons
    if merge_adjancent_polygons:
        # Merge adjacent polygons
        merged_polygons = []

        for polygon in polygons:
            if not merged_polygons:
                merged_polygons.append(polygon)
            else:
                merged = False
                for i, merged_polygon in enumerate(merged_polygons):
                    if (
                        polygon.intersects(merged_polygon)
                        or polygon.contains(merged_polygon)
                        or merged_polygon.contains(polygon)
                    ):
                        merged_polygons[i] = merged_polygon.union(polygon)
                        merged = True
                        break
                if not merged:
                    merged_polygons.append(polygon)

    areas = [poly.area for poly in merged_polygons]
    max_area, median_area = np.max(areas), np.median(areas)
    polygons_filtered = []
    for multi_polygon in merged_polygons:
        if multi_polygon.is_empty:
            continue

        # If it's a MultiPolygon, iterate through individual polygons
        if multi_polygon.geom_type == "MultiPolygon":
            for polygon in multi_polygon.geoms:
                if (
                    polygon.area != max_area
                    and polygon.area / median_area > area_threshold
                ):
                    polygons_filtered.append(Polygon(polygon.exterior))
        # If it's a single Polygon, directly append it
        elif (
            multi_polygon.area != max_area
            and multi_polygon.area / median_area > area_threshold
        ):
            polygons_filtered.append(Polygon(multi_polygon.exterior))

    gs = gpd.GeoSeries(polygons_filtered, crs=kwargs["crs"]).simplify(tolerance)
    if gs.empty:
        raise ValueError("No Features Found")
    gs.to_crs("EPSG:4326").to_file(output_path)
    return output_path
