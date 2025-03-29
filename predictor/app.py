import json
import os
import shutil
import time
import uuid

from geomltoolkits.downloader import tms as TMSDownloader
from geomltoolkits.regularizer import VectorizeMasks
from geomltoolkits.utils import merge_rasters

from .prediction import run_prediction
from .utils import download_or_validate_model


async def predict(
    bbox,
    model_path,
    zoom_level,
    tms_url='"https://apps.kontur.io/raster-tiler/oam/mosaic/{z}/{x}/{y}.png"',
    base_path=None,
    confidence=0.5,
    area_threshold=3,
    tolerance=0.5,
    remove_metadata=True,
    orthogonalize=True,
    vectorization_algorithm="rasterio",
):
    """
    Parameters:
        bbox : Bounding box of the area you want to run prediction on
        model_path : Path of your downloaded model checkpoint
        zoom_level : Zoom level of the tiles to be used for prediction
        tms_url : Your Image URL on which you want to detect feature
        tile_size : Optional >> Tile size to be used in pixel default : 256*256
        base_path : Optional >> Basepath for your working dir of prediction
        confidence: Optional >> Threshold probability for filtering out low-confidence predictions, Defaults to 0.5
        area_threshold (float, optional): Threshold for filtering polygon areas. Defaults to 3 sqm.
        tolerance (float, optional): Tolerance parameter for simplifying polygons. Defaults to 0.5 m. Percentage Tolerance = (Tolerance in Meters / Arc Length in Meters ​)×100
    """
    if confidence < 0 or confidence > 1:
        raise ValueError("Confidence must be between 0 and 1")
    if vectorization_algorithm not in ["potrace", "rasterio"]:
        raise ValueError(
            f"Vectorization algorithm {vectorization_algorithm} is not supported"
        )
    if base_path:
        base_path = os.path.join(base_path, "prediction", str(uuid.uuid4()))
    else:
        base_path = os.path.join(os.getcwd(), "prediction", str(uuid.uuid4()))

    model_path = download_or_validate_model(model_path)
    os.makedirs(base_path, exist_ok=True)
    download_path = os.path.join(base_path, "image")
    os.makedirs(download_path, exist_ok=True)

    image_download_path = await TMSDownloader.download_tiles(
        bbox=bbox,
        zoom=zoom_level,
        tms=tms_url,
        out=download_path,
        georeference=True,
        crs="3857",
        # dump=True,
    )

    # merge_rasters(image_download_path, os.path.join(base_path, "merged_image_chips.tif"))

    prediction_path = os.path.join(base_path, "prediction")
    os.makedirs(prediction_path, exist_ok=True)

    prediction_path = run_prediction(
        model_path,
        image_download_path,
        prediction_path=prediction_path,
        confidence=confidence,
        crs="3857",
    )
    start = time.time()

    geojson_path = os.path.join(base_path, "geojson")
    os.makedirs(geojson_path, exist_ok=True)
    prediction_geojson_path = os.path.join(geojson_path, "prediction.geojson")

    prediction_merged_mask_path = os.path.join(base_path, "merged_prediction_mask.tif")
    os.makedirs(os.path.dirname(prediction_merged_mask_path), exist_ok=True)

    # Merge rasters
    merge_rasters(prediction_path, prediction_merged_mask_path)

    converter = VectorizeMasks(
        simplify_tolerance=tolerance,
        min_area=area_threshold,
        orthogonalize=orthogonalize,
        algorithm=vectorization_algorithm,
        tmp_dir=os.path.join(base_path, "tmp"),
    )
    gdf = converter.convert(prediction_merged_mask_path, prediction_geojson_path)

    print(f"It took {round(time.time()-start)} sec to extract polygons")

    if gdf.crs and gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    elif not gdf.crs:
        # if not defined assume its 3857 because above 3857 is hardcoded
        gdf.set_crs("EPSG:3857", inplace=True)
        gdf = gdf.to_crs("EPSG:4326")

    gdf["building"] = "yes"
    gdf["source"] = "fAIr"

    prediction_geojson_data = json.loads(gdf.to_json())

    if remove_metadata:
        shutil.rmtree(base_path)

    return prediction_geojson_data
