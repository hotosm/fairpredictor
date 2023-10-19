# Standard library imports
import os
from glob import glob
from pathlib import Path

import rasterio
from rasterio.transform import from_bounds

# Third party imports
from tqdm import tqdm

from .utils import get_bounding_box


def georeference(
    input_path: str, output_path: str, is_mask=False, tile_overlap_distance=0.15
) -> None:
    """Perform georeferencing and remove the fourth band from images (if any).

    CRS of the georeferenced images will be EPSG:3857 ('WGS 84 / Pseudo-Mercator').

    Args:
        input_path: Path of the directory where the input data are stored.
        output_path: Path of the directory where the output data will go.
        is_mask: Whether the image is binary or not.
        tile_overlap_distance : Default overlap distance between two tiles to omit the strip between tiles

    Example::

        georeference(
            "data/prediction-dataset/5x5/1-19",
            "data/georeferenced_input/1-19"
        )
    """
    os.makedirs(output_path, exist_ok=True)

    for path in tqdm(
        glob(f"{input_path}/*.png"), desc=f"Georeferencing for {Path(input_path).stem}"
    ):
        filename = Path(path).stem
        in_file = f"{input_path}/{filename}.png"
        out_file = f"{output_path}/{filename}.tif"
        # Get bounding box in EPSG:3857
        x_min, y_min, x_max, y_max = get_bounding_box(filename)
        x_min -= tile_overlap_distance
        y_min -= tile_overlap_distance
        x_max += tile_overlap_distance
        y_max += tile_overlap_distance

        # Use one band for masks and the first three bands for images
        bands = [1] if is_mask else [1, 2, 3]
        crs = {"init": "epsg:3857"}

        with rasterio.open(in_file) as src:
            # Read image data
            data = src.read(bands)
            transform = from_bounds(
                x_min, y_min, x_max, y_max, data.shape[2], data.shape[1]
            )
            _, height, width = data.shape
            metadata = {
                "driver": "GTiff",
                "width": width,
                "height": height,
                "transform": transform,
                "count": len(bands),
                "dtype": data.dtype,
                "crs": crs,
            }

            # Write georeferenced image to output file
            with rasterio.open(out_file, "w", **metadata) as dst:
                dst.write(data, indexes=bands)
